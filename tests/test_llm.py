"""论文超参抽取单测：规则抽取（中英文/否定）、LLM 解析与降级、配置映射。

所有网络调用都被替换掉（``_post_json``），因此**不需要任何 API 密钥**即可跑。
"""
import json
from pathlib import Path

import pytest

from core import llm as L

EN = """2.3 Implementation details
We trained EEGNet for 200 epochs with the AdamW optimizer (learning rate = 1e-3,
weight decay 0.01, cosine annealing schedule). The mini-batch size was set to 64.
All experiments used a fixed random seed of 42. Input signals were band-pass filtered
between 4 and 40 Hz and standardized with a z-score computed per channel. We applied
Euclidean alignment to each subject before training. Data augmentation was used,
including time shift, additive Gaussian noise and channel dropout (p = 0.1).
Evaluation followed a leave-one-subject-out protocol."""

ZH = """2.3 实验设置
本文在 BNCI2014-001 上使用 EEGNet 训练 100 个 epoch，批大小为 32，采用 Adam 优化器，
初始学习率设为 0.0005，权重衰减为 1e-4，并使用余弦退火调度。随机种子固定为 2024。
预处理阶段对每个通道做了 z-score 标准化，未使用欧几里得对齐。本文未使用数据增强，
仅做了 4-38 Hz 带通滤波。划分方式为留一被试交叉验证。"""


def _vals(res: dict) -> dict:
    return {k: (v or {}).get("value") for k, v in res["fields"].items()}


def _rules(text: str) -> dict:
    """短句片段直接测规则层（extract_train_params 对粘贴文本有最小长度保护）。"""
    return L.extract_by_rules(text)


# ------------------------------------------------------------------ 规则抽取
def test_rule_extract_english_full():
    res = L.extract_train_params(EN, use_llm=False, log=lambda *_: None)
    v = _vals(res)
    assert v["epochs"] == 200 and v["batch_size"] == 64
    assert v["lr"] == pytest.approx(1e-3)
    assert v["optimizer"] == "adamw"
    assert v["weight_decay"] == pytest.approx(0.01)
    assert v["scheduler"] == "cosine"
    assert v["seed"] == 42
    assert v["normalization"] == "zscore_channel"
    assert v["euclidean_alignment"] is True
    assert v["augmentation"] is True
    assert set(v["augmentation_types"]) == {"time_shift", "gaussian_noise", "channel_dropout"}
    assert v["split_scheme"] == "loso"
    # 每个识别到的字段都要能给出原文依据
    for k, item in res["fields"].items():
        if item["value"] is not None:
            assert item["evidence"], f"{k} 缺少依据片段"
    assert res["meta"]["used_llm"] is False and res["meta"]["n_rules"] >= 10


def test_rule_extract_chinese_and_negation():
    res = L.extract_train_params(ZH, use_llm=False, log=lambda *_: None)
    v = _vals(res)
    assert v["epochs"] == 100 and v["batch_size"] == 32
    assert v["lr"] == pytest.approx(5e-4)
    assert v["optimizer"] == "adam"
    assert v["normalization"] == "zscore_channel"
    assert v["euclidean_alignment"] is False       # 未使用欧几里得对齐
    assert v["augmentation"] is False              # 未使用数据增强
    assert v["split_scheme"] == "loso"


@pytest.mark.parametrize("text,key,expect", [
    ("We did not apply data augmentation in this study.", "augmentation", False),
    ("No data augmentation was used.", "augmentation", False),
    ("without any data augmentation", "augmentation", False),
    ("Data augmentation was applied as follows.", "augmentation", True),
    ("we used Euclidean alignment for each subject", "euclidean_alignment", True),
    ("no Euclidean alignment was performed", "euclidean_alignment", False),
    ("未做欧氏对齐", "euclidean_alignment", False),
    ("本文使用了欧几里得对齐", "euclidean_alignment", True),
])
def test_negation_and_positive_detection(text, key, expect):
    got = _rules(text).get(key, {}).get("value")
    assert got is expect


@pytest.mark.parametrize("text,expect", [
    ("input was standardized per channel", "zscore_channel"),
    ("we applied z-score normalization", "zscore_channel"),
    ("instance normalization was used for each trial", "zscore_epoch"),
    ("applied instance norm (per-epoch)", "zscore_epoch"),
    ("signals were scaled to the range [0, 1]", "minmax_channel"),
    ("no normalization was applied to raw signals", "none"),
    ("使用了分位数归一化", "quantile"),
    ("做了滑动标准化（EMA）", "ema"),
])
def test_normalization_mapping(text, expect):
    assert _rules(text).get("normalization", {}).get("value") == expect


@pytest.mark.parametrize("text,expect", [
    ("learning rate = 1e-3", 1e-3),
    ("learning rate = 0.001", 0.001),
    ("initial learning rate of 5e-4", 5e-4),
    ("学习率设为 1×10-3", 1e-3),
    ("lr: 3e-5", 3e-5),
])
def test_learning_rate_notations(text, expect):
    got = _rules(text + " and we trained for 100 epochs").get("lr", {}).get("value")
    assert got == pytest.approx(expect)


def test_epochs_and_batch_both_word_orders():
    assert _rules("The model was trained for 300 epochs.")["epochs"]["value"] == 300
    assert _rules("epochs = 300")["epochs"]["value"] == 300
    assert _rules("The mini-batch size was set to 128 while training."
                  )["batch_size"]["value"] == 128
    d = _rules("训练 250 轮，批大小设置为 16")
    assert d["epochs"]["value"] == 250 and d["batch_size"]["value"] == 16
    # lr 的宽松匹配不能把别的参数的数字抓走
    assert "lr" not in _rules("we report the learning rate and the batch size 64")


def test_illegal_text_raises():
    with pytest.raises(ValueError):
        L.extract_train_params("too short", log=lambda *_: None)


# ------------------------------------------------------------ 配置映射/解析
def test_to_train_config_paths_and_merge():
    res = L.extract_train_params(EN, use_llm=False, log=lambda *_: None)
    cfg = L.to_train_config(res["fields"])
    assert cfg["trainer"]["epochs"] == 200 and cfg["trainer"]["batch_size"] == 64
    assert cfg["trainer"]["optimizer"] == "adamw" and cfg["trainer"]["seed"] == 42
    assert cfg["normalization"]["mode"] == "zscore_channel"
    assert cfg["normalization"]["euclidean_alignment"] is True
    assert cfg["split"]["scheme"] == "loso"
    # 列了具体增强手段 → 给出非零默认强度，直接可用
    assert cfg["augmentation"]["time_shift"] > 0
    assert cfg["augmentation"]["channel_dropout"] == pytest.approx(0.1)
    assert "enabled" not in cfg["augmentation"]
    # base 合并：不覆盖未识别字段
    base = {"model": {"name": "eegnet"}, "trainer": {"device": "auto", "epochs": 60}}
    merged = L.to_train_config(res["fields"], base)
    assert merged["model"]["name"] == "eegnet" and merged["trainer"]["device"] == "auto"
    assert merged["trainer"]["epochs"] == 200
    assert base["trainer"]["epochs"] == 60          # 不修改传入的 base


def test_to_train_config_disables_augmentation_when_paper_says_no():
    res = L.extract_train_params(ZH, use_llm=False, log=lambda *_: None)
    cfg = L.to_train_config(res["fields"], {"augmentation": {"time_shift": 50,
                                                            "gaussian_noise": 0.3,
                                                            "channel_dropout": 0.2}})
    assert cfg["augmentation"] == {"time_shift": 0, "gaussian_noise": 0, "channel_dropout": 0}
    assert cfg["normalization"]["euclidean_alignment"] is False


def test_unsupported_augmentation_types_are_reported_not_silently_written():
    fields = {"augmentation": {"value": True}, "augmentation_types":
              {"value": ["mixup", "time_shift", "crop"]}}
    cfg = L.to_train_config(fields)
    assert cfg["augmentation"]["time_shift"] > 0
    assert set(cfg["augmentation"]["unsupported_types"]) == {"mixup", "crop"}


def test_augmentation_turns_on_over_zero_base_but_keeps_user_values():
    """base 里的 0 表示"未启用"，不该阻止论文里的增强被写进来；非零用户值要保留。"""
    fields = {"augmentation": {"value": True},
              "augmentation_types": {"value": ["time_shift", "gaussian_noise"]}}
    zeros = {"augmentation": {"time_shift": 0, "gaussian_noise": 0.0, "channel_dropout": 0.0}}
    cfg = L.to_train_config(fields, zeros)
    assert cfg["augmentation"]["time_shift"] > 0 and cfg["augmentation"]["gaussian_noise"] > 0
    assert zeros["augmentation"]["time_shift"] == 0          # 不修改传入对象
    tuned = L.to_train_config(fields, {"augmentation": {"time_shift": 99,
                                                       "gaussian_noise": 0.5}})
    assert tuned["augmentation"]["time_shift"] == 99         # 用户调过的值保持不动
    assert tuned["augmentation"]["gaussian_noise"] == 0.5


@pytest.mark.parametrize("key,text,expect", [
    ("epochs", "150", 150), ("epochs", "", None), ("epochs", "未识别", None),
    ("lr", "0.002", 0.002), ("optimizer", "AdamW", "adamw"),
    ("optimizer", "随便写", None), ("euclidean_alignment", "是", True),
    ("euclidean_alignment", "否", False), ("euclidean_alignment", "也许", None),
    ("augmentation_types", "time_shift, gaussian_noise", ["time_shift", "gaussian_noise"]),
])
def test_parse_field_text(key, text, expect):
    assert L.parse_field_text(key, text) == expect


# ------------------------------------------------------------------ LLM 路径
def _fake_post_factory(body: dict):
    def _fake(url, headers, payload, timeout):
        return {"choices": [{"message": {"content": "```json\n" + json.dumps(body) + "\n```"}}]}
    return _fake


def test_llm_merge_agreement_and_conflict(monkeypatch):
    body = {
        "epochs": {"value": 200, "evidence": "for 200 epochs", "confidence": 0.9},
        "batch_size": {"value": 64, "evidence": "mini-batch size was set to 64"},
        "lr": {"value": 3e-4, "evidence": "learning rate = 3e-4"},   # 与规则冲突
        "optimizer": {"value": "AdamW, with betas"},
        "seed": {"value": None},                                    # 空值要忽略
        "not_a_field": {"value": 1},
    }
    monkeypatch.setattr(L, "_post_json", _fake_post_factory(body))
    cfg = L.LLMConfig(provider="deepseek", api_key="sk-test")
    res = L.extract_train_params(EN, cfg=cfg, log=lambda *_: None)
    f = res["fields"]
    assert res["meta"]["used_llm"] is True
    assert f["epochs"]["source"] == "LLM + 规则一致"
    assert f["batch_size"]["source"] == "LLM + 规则一致"
    assert f["lr"]["value"] == pytest.approx(3e-4)
    assert "不一致" in f["lr"]["source"] and "0.001" in f["lr"]["conflict"]
    assert f["optimizer"]["value"] == "adamw"                       # 子串兜底
    assert f["seed"]["value"] == 42 and f["seed"]["source"].startswith("规则")
    assert "not_a_field" not in f


def test_llm_bad_json_falls_back_to_rules(monkeypatch):
    monkeypatch.setattr(L, "_post_json", lambda *a, **k: {
        "choices": [{"message": {"content": "抱歉，我无法完成这个请求。"}}]})
    cfg = L.LLMConfig(provider="deepseek", api_key="sk-test")
    res = L.extract_train_params(EN, cfg=cfg, log=lambda *_: None)
    assert res["meta"]["used_llm"] is False
    assert res["fields"]["epochs"]["value"] == 200
    assert "降级" in res["meta"]["warnings"][0]


def test_llm_http_error_falls_back(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("HTTP 401：invalid api key")
    monkeypatch.setattr(L, "_post_json", boom)
    cfg = L.LLMConfig(provider="deepseek", api_key="bad")
    res = L.extract_train_params(EN, cfg=cfg, log=lambda *_: None)
    assert res["meta"]["used_llm"] is False
    assert res["fields"]["batch_size"]["value"] == 64
    assert "HTTP 401" in res["meta"]["warnings"][0]


def test_not_configured_means_rules_only_with_hint():
    res = L.extract_train_params(EN, cfg=L.LLMConfig(provider="deepseek", api_key=""),
                                 log=lambda *_: None)
    assert res["meta"]["used_llm"] is False
    assert res["meta"]["warnings"] and "未配置 LLM" in res["meta"]["warnings"][0]
    assert res["fields"]["lr"]["value"] == pytest.approx(1e-3)
    # 显式指定"只用规则"时不打扰用户
    res2 = L.extract_train_params(EN, use_llm=False, log=lambda *_: None)
    assert res2["meta"]["warnings"] == []


def test_config_roundtrip_and_gitignore(monkeypatch, tmp_path):
    monkeypatch.setattr(L, "PROJECT_ROOT", tmp_path)
    (tmp_path / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    cfg = L.LLMConfig(provider="openai", model="gpt-4o-mini",
                      base_url="https://api.openai.com/v1", api_key="sk-secret")
    p = L.save_config_file(cfg, path=tmp_path / "llm.yaml")
    assert p.exists()
    back = L.load_config_file(p)
    assert back["provider"] == "openai" and back["api_key"] == "sk-secret"
    assert "configs/llm.yaml" in (tmp_path / ".gitignore").read_text(encoding="utf-8")


def test_api_key_from_env(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("BCI_LLM_API_KEY", raising=False)
    assert L.api_key_from_env("deepseek") == ""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")
    assert L.api_key_from_env("deepseek") == "sk-env"
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    monkeypatch.setenv("BCI_LLM_API_KEY", "sk-generic")
    assert L.api_key_from_env("deepseek") == "sk-generic"


def test_config_configured_flag_and_describe():
    assert L.LLMConfig(provider="ollama").configured          # 本地无需密钥
    assert not L.LLMConfig(provider="deepseek", api_key="").configured
    c = L.LLMConfig(provider="deepseek", api_key="k").resolved()
    assert c.base_url.endswith("/v1") and c.model == "deepseek-chat"
    assert "DeepSeek" in c.describe()


def test_chat_reports_missing_key_clearly():
    cfg = L.LLMConfig(provider="deepseek", api_key="")
    with pytest.raises(ValueError) as ei:
        L.chat(cfg, "sys", "user", log=lambda *_: None)
    assert "API 密钥" in str(ei.value) and "DEEPSEEK_API_KEY" in str(ei.value)


def test_build_prompt_contains_schema_and_text():
    prompt = L.build_prompt(EN[:200])
    for key in ("epochs", "batch_size", "euclidean_alignment", "augmentation",
                "normalization"):
        assert key in prompt
    assert EN[:50] in prompt
    assert "只输出 JSON" in L.SYSTEM_PROMPT


def test_parse_json_tolerates_wrappers():
    assert L._parse_json('{"a": 1}') == {"a": 1}
    assert L._parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert L._parse_json('好的，结果如下：{"a": 1} 希望有帮助') == {"a": 1}
    assert L._parse_json("{'a': 1}") == {"a": 1}
    assert L._parse_json('{"a": 1,}') == {"a": 1}
    assert L._parse_json("完全不是 JSON") is None


def test_read_text_file(tmp_path):
    p = tmp_path / "paper.txt"
    p.write_text(EN, encoding="utf-8")
    assert "EEGNet" in L.read_text_file(p)
    with pytest.raises(FileNotFoundError):
        L.read_text_file(tmp_path / "nope.txt")
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")                   # 故意造一个坏 PDF
    try:
        import pypdf  # noqa: F401

        with pytest.raises(Exception):             # 装了 pypdf → 坏文件必须报错，不能静默返回空
            L.read_text_file(pdf)
    except ImportError as e:                       # 没装 pypdf → 提示要装
        assert "pypdf" in str(e)


def test_format_report_mentions_evidence_and_warnings():
    res = L.extract_train_params(EN, use_llm=False, log=lambda *_: None)
    txt = L.format_report(res)
    assert "epochs" in txt and "依据" in txt and "adamw" in txt.lower()
    res["meta"]["warnings"] = ["测试告警"]
    assert "测试告警" in L.format_report(res)
