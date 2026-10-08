"""数据集路径与配置环境变量展开（跨机器/跨系统可移植性）。

这两件事决定了"同一份代码换到 Windows 上要不要改文件"：
1. 数据目录默认值 + ``BCI_DATA_ROOT`` 覆盖（:mod:`core.data_paths`）；
2. 配置里 ``${VAR}`` / ``${VAR:-默认值}`` 展开（:func:`core.pipeline.load_config`）。
"""
from pathlib import Path

import pytest

from core import data_paths as DP
from core.pipeline import expand_env, load_config


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(DP.ENV_VAR, raising=False)
    yield


# ------------------------------------------------------------ 数据目录解析
def test_default_when_env_absent():
    assert DP.data_root() == DP.DEFAULT_DATA_ROOT
    assert DP.data_root("BNCI2014001") == DP.DEFAULT_DATA_ROOT / "BNCI2014001"
    assert DP.bnci_root() == DP.DEFAULT_DATA_ROOT / "BNCI2014001"
    assert DP.physionet_root() == DP.DEFAULT_DATA_ROOT / "PhysioNet_MI"
    # 不能出现 …/BNCI2014001/BNCI2014001 这种重复拼接
    assert DP.bnci_root().name == "BNCI2014001"


def test_env_override_base_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(DP.ENV_VAR, str(tmp_path))
    assert DP.data_root() == tmp_path
    assert DP.bnci_root() == tmp_path / "BNCI2014001"
    assert DP.physionet_root() == tmp_path / "PhysioNet_MI"
    assert DP.env_root() == tmp_path


def test_env_pointing_directly_at_dataset(monkeypatch, tmp_path):
    target = tmp_path / "BNCI2014001"
    monkeypatch.setenv(DP.ENV_VAR, str(target))
    assert DP.bnci_root() == target                     # 不再重复拼一层
    assert DP.physionet_root() == target / "PhysioNet_MI"


def test_env_windows_style_value_and_quotes(monkeypatch):
    monkeypatch.setenv(DP.ENV_VAR, '"D:\\BCI\\data"')   # 用户可能带引号
    root = DP.env_root()
    assert root is not None and '"' not in str(root)     # 引号会被去掉
    assert str(DP.data_root("BNCI2014001")).endswith("BNCI2014001")


def test_env_empty_string_falls_back(monkeypatch):
    monkeypatch.setenv(DP.ENV_VAR, "   ")
    assert DP.data_root() == DP.DEFAULT_DATA_ROOT


def test_describe_mentions_env_var_and_paths():
    text = DP.describe()
    assert DP.ENV_VAR in text and "BNCI2014001" in text and "PhysioNet" in text


# ------------------------------------------------------- 配置里的变量展开
def test_expand_env_plain_and_default():
    env = {"ROOT": "/data"}
    assert expand_env("${ROOT}/BNCI2014001", env) == "/data/BNCI2014001"
    assert expand_env("${MISSING:-/fallback}", env) == "/fallback"
    assert expand_env("${MISSING}", env) == ""
    assert expand_env("${ROOT:-/fallback}", env) == "/data"          # 有值就用值
    assert expand_env("${ROOT}/a/${MISSING:-b}", env) == "/data/a/b"


def test_expand_env_windows_default_with_colon():
    # Windows 默认值本身带盘符冒号，不能被占位符解析截断
    assert expand_env("${BCI_DATA_ROOT:-D:\\BCI\\data}\\BNCI2014001", {}) \
        == "D:\\BCI\\data\\BNCI2014001"


def test_expand_env_nested_and_non_strings():
    cfg = {"a": ["${X:-1}", 2, {"b": "${X:-1}"}], "c": 3.5, "d": None, "e": True}
    assert expand_env(cfg, {}) == {"a": ["1", 2, {"b": "1"}], "c": 3.5, "d": None, "e": True}


def test_load_config_expands_env(monkeypatch, tmp_path):
    cfg_file = tmp_path / "t.yaml"
    cfg_file.write_text(
        'dataset:\n'
        '  root: ${BCI_DATA_ROOT:-/opt/bci/data}/BNCI2014001\n'
        '  subjects: "1"\n', encoding="utf-8")
    monkeypatch.delenv("BCI_DATA_ROOT", raising=False)
    assert load_config(cfg_file)["dataset"]["root"].endswith("data/BNCI2014001")
    monkeypatch.setenv("BCI_DATA_ROOT", "D:/BCI/data")
    assert load_config(cfg_file)["dataset"]["root"] == "D:/BCI/data/BNCI2014001"


@pytest.mark.parametrize("cfg,expect_tail", [
    ("configs/preprocess_bnci2014001.yaml", "BNCI2014001"),
    ("configs/preprocess_bnci2014001_200hz.yaml", "BNCI2014001"),
    ("configs/preprocess_physionet_mi.yaml", "PhysioNet_MI"),
])
def test_preprocess_configs_are_portable(cfg, expect_tail, monkeypatch, tmp_path):
    """仓库里的预处理配置必须"换机器只设环境变量"，而不是硬编码某台机器的路径。"""
    text = Path(cfg).read_text(encoding="utf-8")
    assert DP.ENV_VAR in text                                # 带环境变量占位符
    monkeypatch.setenv(DP.ENV_VAR, str(tmp_path))
    root = load_config(cfg)["dataset"]["root"]
    assert (Path(root) if root.startswith("/") else Path(root)).name == expect_tail
    assert str(tmp_path) in str(root) or root == str(tmp_path / expect_tail)


def test_no_hardcoded_macos_user_paths():
    """仓库里不允许出现某台机器的用户名路径（隐私 + 可移植）：一律用 ~ / Path.home()。"""
    for f in ("ui/data_page.py", "tests/test_bnci.py", "tests/test_physionet.py",
              "docs/demo_features.py", "core/data_paths.py",
              "configs/preprocess_bnci2014001.yaml", "README.md"):
        text = Path(f).read_text(encoding="utf-8")
        assert "/Users/" not in text, f"{f} 里还有硬编码的本机路径"
    # 默认值必须由家目录推出，而不是写死某人的用户名
    assert "Path.home()" in Path("core/data_paths.py").read_text(encoding="utf-8")
    assert DP.DEFAULT_DATA_ROOT.is_absolute()
    assert "Desktop" in DP.DEFAULT_DATA_ROOT.as_posix()


def test_tilde_expansion_in_config(monkeypatch, tmp_path):
    """``${VAR:-~/...}`` 里的 ~ 要展开成家目录（配置才能既通用又能在本机直接用）。"""
    cfg = tmp_path / "t.yaml"
    cfg.write_text('dataset:\n  root: ${BCI_DATA_ROOT:-~/Desktop/DL/BCI/data}/BNCI2014001\n',
                   encoding="utf-8")
    monkeypatch.delenv("BCI_DATA_ROOT", raising=False)
    got = load_config(cfg)["dataset"]["root"]
    assert got.startswith(str(Path.home())) and got.endswith("BNCI2014001")
