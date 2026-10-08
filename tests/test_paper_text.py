"""论文文件处理单测：PDF 读取、正文/补充材料判定、参数区定位、多来源组装、溯源落盘。

PDF 用**代码手写**生成（不依赖 reportlab 等额外包），因此这些测试完全离线可跑。
"""
import json
from pathlib import Path

import pytest

from core import paper_text as PT
from core.experiment import ExperimentManager
from core.llm import extract_train_params
from core.train.evaluation import export_markdown_report


# ------------------------------------------------------------------ PDF 构造
#: 复用核心里的最小 PDF 生成器（测试与演示共用一份实现）
make_pdf = PT.write_sample_pdf


MAIN_TEXT = """Neural decoding of motor imagery with EEGNet
3.2 Methods
We recorded 9 subjects with 22 EEG channels at 250 Hz. Signals were band-pass
filtered between 4 and 40 Hz. Model training details and all hyperparameters are
provided in the Supplementary Material (see Table S1).
3.3 Results
Our method achieved 78.5% accuracy.
"""

SUPP_TEXT = """Supplementary Material
S1 Data preprocessing
Signals were re-referenced to the common average.
S2 Implementation details
All models were trained for 200 epochs with the AdamW optimizer (learning rate = 1e-3,
weight decay = 0.01) and a cosine annealing schedule. The mini-batch size was set to 64.
The random seed was fixed to 42. Input signals were standardized with a z-score per
channel. Euclidean alignment was applied to each subject. Data augmentation included
time shift, Gaussian noise and channel dropout (p = 0.1).
Table S1. Hyperparameter search space.
S3 Additional results
Per-subject accuracies are listed below.
"""


# ------------------------------------------------------------------ 基础能力
def test_make_pdf_and_read_two_pages(tmp_path):
    pdf = make_pdf(tmp_path / "Smith2023_supplementary.pdf", [MAIN_TEXT, SUPP_TEXT])
    src, text = PT.read_source(pdf, log=lambda *_: None)
    assert src.n_pages == 2 and src.n_chars > 200
    assert "Supplementary Material" in text and "Implementation details" in text
    assert src.kind == "supplementary" and src.sha256
    assert src.name == "Smith2023_supplementary.pdf"
    assert "补充材料" in src.label() and "2 页" in src.label()


@pytest.mark.parametrize("name,kind", [
    ("Smith2023_supplementary.pdf", "supplementary"),
    ("paper_SI.pdf", "supplementary"),      # 补充材料常见的 *_SI 命名
    ("Smith_Supp_Info.pdf", "supplementary"),
    ("附录.pdf", "supplementary"),
    ("补充材料-实验细节.pdf", "supplementary"),
    ("Smith2023_main.pdf", "main"),
    ("manuscript.pdf", "main"),
    ("eegnet.pdf", "unknown"),
])
def test_guess_kind_from_filename(name, kind):
    assert PT.guess_kind(name, "") == kind


def test_guess_kind_from_content_when_name_is_neutral():
    assert PT.guess_kind("download.pdf", SUPP_TEXT) == "supplementary"
    assert PT.guess_kind("download.pdf", "We propose a new method. Try it.") == "unknown"


def test_is_supported():
    for ok in ("a.pdf", "a.txt", "a.md", "a.tex", "A.PDF"):
        assert PT.is_supported(ok)
    for bad in ("a.docx", "a.xlsx", "a.png", "noext"):
        assert not PT.is_supported(bad)


def test_read_source_empty_and_scanned_pdf(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("   ", encoding="utf-8")
    src, _ = PT.read_source(empty, log=lambda *_: None)
    assert "没有抽到任何文本" in src.note

    scanned = make_pdf(tmp_path / "scan.pdf", ["x"], per_page_chars=20)
    src2, _ = PT.read_source(scanned, log=lambda *_: None)
    assert "扫描版" in src2.note


def test_read_source_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        PT.read_source(tmp_path / "nope.pdf", log=lambda *_: None)


# ------------------------------------------------------------ 参数区定位/组装
def test_find_param_sections_hits_implementation_details():
    secs = PT.find_param_sections(SUPP_TEXT, source="supp.pdf", kind="supplementary")
    titles = [s.title for s in secs]
    assert any("Implementation details" in t for t in titles)
    assert secs[0].score > 0 and secs[0].kind == "supplementary"
    assert "S2 Implementation details" in secs[0].label()


def test_find_param_sections_ignores_unrelated_text():
    assert PT.find_param_sections("We love kittens. Cats are nice. " * 5) == []


def test_assemble_prefers_supplementary_and_records_sources(tmp_path):
    (tmp_path / "Smith2023_main.txt").write_text(MAIN_TEXT, encoding="utf-8")
    (tmp_path / "Smith2023_supplementary.txt").write_text(SUPP_TEXT, encoding="utf-8")
    srcs, texts = PT.read_sources([tmp_path / "Smith2023_main.txt",
                                   tmp_path / "Smith2023_supplementary.txt"],
                                  log=lambda *_: None)
    merged, info = PT.assemble_param_text([(s, texts[s.path]) for s in srcs],
                                          log=lambda *_: None)
    assert merged.index("Implementation details") < merged.find("3.2 Methods")
    assert "【来源 1：" in merged and "补充材料" in merged
    assert info["source_summary"] and len(info["sources"]) == 2
    assert info["used_sections"] and "Implementation details" in info["used_sections"][0]
    assert info["n_chars"] == len(merged)
    assert any("补充材料" in n for n in info["notes"])      # 正文里的交叉引用提示
    assert any("没找到" in n for n in info["notes"])        # 正文里没有参数小节 → 兜底取开头


def test_assemble_truncates_with_note(tmp_path):
    # 一个超长的"实现细节"小节（真实补充材料的常见形态：一节写满所有超参）
    big = ("S2 Implementation details\n"
           + "All models were trained for 200 epochs with the AdamW optimizer. " * 200)
    (tmp_path / "s_supplementary.txt").write_text(big, encoding="utf-8")
    srcs, texts = PT.read_sources([tmp_path / "s_supplementary.txt"], log=lambda *_: None)
    merged, info = PT.assemble_param_text([(s, texts[s.path]) for s in srcs],
                                          max_chars=1500, log=lambda *_: None)
    assert info["truncated"] is True and len(merged) <= 1500
    assert info["n_chars"] == len(merged)
    assert any("截断" in n for n in info["notes"])


def test_assemble_requires_input():
    with pytest.raises(ValueError):
        PT.assemble_param_text([], log=lambda *_: None)


# --------------------------------------------------- 端到端：PDF → 参数 → 溯源
def test_pdf_to_params_end_to_end(tmp_path):
    pdf = make_pdf(tmp_path / "Smith2023_supplementary.pdf", [MAIN_TEXT, SUPP_TEXT])
    srcs, texts = PT.read_sources([pdf], log=lambda *_: None)
    merged, info = PT.assemble_param_text([(s, texts[s.path]) for s in srcs],
                                          log=lambda *_: None)
    res = extract_train_params(merged, use_llm=False, log=lambda *_: None)
    v = {k: res["fields"][k]["value"] for k in
         ("epochs", "batch_size", "lr", "optimizer", "seed", "normalization",
          "euclidean_alignment", "augmentation")}
    assert v["epochs"] == 200 and v["batch_size"] == 64
    assert v["lr"] == pytest.approx(1e-3) and v["optimizer"] == "adamw"
    assert v["seed"] == 42 and v["normalization"] == "zscore_channel"
    assert v["euclidean_alignment"] is True and v["augmentation"] is True


def test_provenance_block_and_markdown(tmp_path):
    pdf = make_pdf(tmp_path / "Smith2023_supplementary.pdf", [MAIN_TEXT, SUPP_TEXT])
    srcs, texts = PT.read_sources([pdf], log=lambda *_: None)
    merged, info = PT.assemble_param_text([(s, texts[s.path]) for s in srcs],
                                          log=lambda *_: None)
    res = extract_train_params(merged, use_llm=False, log=lambda *_: None)
    prov = PT.provenance(res, merged, info)
    assert prov["method"] == "rules" and prov["applied_to_config"] is True
    assert prov["fields"]["epochs"]["value"] == 200
    assert prov["fields"]["epochs"]["evidence"]
    inp = prov["input"]
    assert inp["text_sha256"] == PT.sha256_text(merged)
    assert inp["sources"][0]["name"].endswith(".pdf") and inp["sources"][0]["sha256"]
    assert inp["used_sections"] and inp["excerpt"]
    one = PT.summarize_provenance(prov)
    assert "Smith2023_supplementary.pdf" in one and "补充材料" in one and "规则抽取" in one
    md = PT.markdown_provenance(prov)
    assert "## 训练参数来源（论文抽取）" in md and "原文依据" in md
    assert "Implementation details" in md and "sha256" in md


def test_provenance_marks_unsupported_augmentation():
    res = {"meta": {"used_llm": False, "warnings": ["测试告警"]},
           "fields": {"augmentation": {"value": True},
                      "augmentation_types": {"value": ["mixup", "time_shift"]}}}
    prov = PT.provenance(res, "text", None)
    assert prov["unsupported_augmentation"] == ["mixup"]
    md = PT.markdown_provenance(prov)
    assert "mixup" in md and "测试告警" in md


# --------------------------------------------- 绑定实验记录（config.yaml + 报告）
SUMMARY_STUB = {
    "experiment": "paper_run", "timestamp": "2026-10-08T10:00:00", "model": "eegnet",
    "scheme": "session", "data_dir": "processed/x", "n_trials": 100, "n_classes": 4,
    "aggregate": {"n_folds": 2, "acc_mean": 0.7, "acc_std": 0.02, "kappa_mean": 0.6},
    "folds": [], "per_subject": [],
}


def test_binding_to_experiment_record(tmp_path):
    prov = PT.provenance(
        {"meta": {"used_llm": True, "model": "deepseek-chat", "provider": "deepseek"},
         "fields": {"epochs": {"value": 200, "source": "LLM + 规则一致",
                               "evidence": "for 200 epochs"}}},
        "Some paper text", {"source_summary": "supp.pdf（补充材料）",
                            "used_sections": ["supp.pdf（补充材料）｜S2 Implementation details"],
                            "sources": [{"name": "supp.pdf", "kind": "supplementary",
                                         "n_pages": 3, "n_chars": 500, "sha256": "ab" * 32}],
                            "n_chars": 15})
    config = {"experiment": {"name": "paper_run"}, "trainer": {"epochs": 200},
              "paper_params": prov}
    mgr = ExperimentManager(root=tmp_path / "experiments")
    exp = mgr.create("paper_run", config)
    saved = json.loads(json.dumps(config))                # 只做结构校验
    assert saved["paper_params"]["fields"]["epochs"]["value"] == 200
    text = (exp.dir / "config.yaml").read_text(encoding="utf-8")
    assert "paper_params" in text and "supp.pdf" in text and "Implementation details" in text

    report = export_markdown_report(exp.dir, SUMMARY_STUB, config)
    md = Path(report).read_text(encoding="utf-8")
    assert "## 训练参数来源（论文抽取）" in md
    assert "deepseek-chat" in md and "for 200 epochs" in md


def test_report_without_provenance_has_no_section(tmp_path):
    report = export_markdown_report(tmp_path, SUMMARY_STUB, {"trainer": {"epochs": 60}})
    assert "训练参数来源" not in Path(report).read_text(encoding="utf-8")
