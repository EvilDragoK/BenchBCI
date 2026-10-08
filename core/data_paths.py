"""数据集路径解析：本机默认路径 + 环境变量覆盖（换机器/换系统不用改代码）。

默认布局（本机 macOS）::

    ~/Desktop/DL/BCI/data/BNCI2014001/A01T.mat …
    ~/Desktop/DL/BCI/data/PhysioNet_MI/S001/S001R01.edf …

换到 Windows / Linux / 别人的机器时，只要设一次环境变量指向**数据集根目录**：

    Windows PowerShell（当前会话）:  $env:BCI_DATA_ROOT = "D:\\BCI\\data"
    Windows cmd（永久）:             setx BCI_DATA_ROOT "D:\\BCI\\data"
    macOS / Linux:                   export BCI_DATA_ROOT=~/data

也允许把变量直接指到某个数据集目录（如 ``.../BNCI2014001``），此时按名字匹配直接使用。
配置文件里可以用 ``${BCI_DATA_ROOT:-~/Desktop/DL/BCI/data}`` 这种带默认值的写法
（见 :func:`core.pipeline.load_config`），这样"本机默认 + 其他机器一个变量搞定"两不误。
"""
from __future__ import annotations

import os
from pathlib import Path

#: 环境变量名（指向"数据集根目录"，其下应有 BNCI2014001 / PhysioNet_MI 等子目录）
ENV_VAR = "BCI_DATA_ROOT"

#: 默认数据集根目录 = 家目录下的 Desktop/DL/BCI/data（仅作兜底；
#: 其他机器/其他布局请用环境变量 BCI_DATA_ROOT 覆盖，不许硬编码别人的用户名）
DEFAULT_DATA_ROOT = Path.home() / "Desktop" / "DL" / "BCI" / "data"

#: 已知数据集目录名
BNCI_DIR = "BNCI2014001"
PHYSIONET_DIR = "PhysioNet_MI"


def env_root() -> Path | None:
    """读环境变量（未设或为空返回 None）。"""
    val = (os.environ.get(ENV_VAR) or "").strip().strip('"').strip("'")
    return Path(val).expanduser() if val else None


def data_root(name: str | Path | None = None, default: str | Path | None = None) -> Path:
    """数据集目录。

    ``name`` 给了就拼在根目录后面（如 ``BNCI2014001``）；若环境变量本身已经指向
    该名字的目录，则直接返回它（避免出现 ``…/BNCI2014001/BNCI2014001``）。
    ``default`` 可覆盖"根目录"的兜底值（默认 :data:`DEFAULT_DATA_ROOT`）。
    """
    base = env_root()
    if base is not None:
        if name and base.name.lower() == str(name).lower():
            return base
        return base / str(name) if name else base
    root = Path(default) if default else DEFAULT_DATA_ROOT
    return root / str(name) if name else root


def bnci_root() -> Path:
    """BNCI2014001（BCI Competition IV 2a）目录。"""
    return data_root(BNCI_DIR)


def physionet_root() -> Path:
    """PhysioNet MI（EEG Motor Movement/Imagery）目录。"""
    return data_root(PHYSIONET_DIR)


def default_path(name: str | Path) -> Path:
    """本机默认路径（忽略环境变量），用于"默认值 + 覆盖"的展示。"""
    return DEFAULT_DATA_ROOT / str(name)


def abbrev(path: str | Path) -> str:
    """把家目录前缀缩写成 ``~``（日志/演示输出更友好，也不会把用户名写进文件）。"""
    p = str(path)
    home = str(Path.home())
    return ("~" + p[len(home):]) if p.startswith(home) else p


def describe() -> str:
    """一行说明当前解析结果（界面提示/日志用；路径以 ~ 缩写）。"""
    src = f"环境变量 {ENV_VAR}" if env_root() else "本机默认路径"
    return (f"数据集根目录 = {abbrev(data_root())}（来源：{src}）\n"
            f"· BNCI2014001: {abbrev(bnci_root())}"
            + ("（存在）" if bnci_root().exists() else "（不存在）")
            + f"\n· PhysioNet MI: {abbrev(physionet_root())}"
            + ("（存在）" if physionet_root().exists() else "（不存在）")
            + ("" if env_root() else f"\n· 换机器请设环境变量 {ENV_VAR} 指向数据集根目录"))
