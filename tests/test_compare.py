"""双实验对比：一致性检查（数据集/类别必须先一致）+ 指标对照 + 结果页对比视图。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from core.experiment import ExperimentManager
from core.train.compare import (check_compatible, class_key, compare_metrics,
                                compare_per_subject, dataset_key, winner_summary)


def _summary(dataset="bnci2014001", classes=("feet", "left_hand"),
             acc=0.5, scheme="session", model="eegnet", n_trials=100,
             per_subject=None, brier=0.8):
    return {
        "data_dir": f"/tmp/processed/{dataset}",
        "model": model, "scheme": scheme, "n_trials": n_trials,
        "n_classes": len(classes), "class_names": list(classes),
        "folds": [{"fold": 0, "tag": "t", "acc": acc, "kappa": 0.2}],
        "aggregate": {"n_folds": 1, "acc_mean": acc, "acc_std": 0.0,
                      "kappa_mean": 0.2, "kappa_std": 0.0, "mcc_mean": 0.1,
                      "f1_macro_mean": acc - 0.05, "brier_mean": brier,
                      "ece_mean": 0.1, "confusion_total": [[3, 1], [2, 4]]},
        "per_subject": per_subject or [{"subject": 1, "acc_mean": acc, "acc_std": 0.0}],
    }


# ------------------------------------------------------------------ 一致性
def test_dataset_key_from_data_dir():
    assert dataset_key(_summary("bnci2014001")) == "bnci2014001"
    assert dataset_key({"config": {"data": {"processed_dir": "/x/processed/physionet"}}}) \
        == "physionet"
    assert dataset_key({}) == "（未知）"


def test_check_compatible_same_dataset_ok():
    chk = check_compatible(_summary(acc=0.5), _summary(acc=0.6, model="csp_lda"))
    assert chk["ok"] and not chk["blocking"]
    names = {c["name"]: c for c in chk["checks"]}
    assert names["数据集"]["ok"] and names["类别集合"]["ok"] and names["模型"]["ok"]


def test_check_compatible_different_dataset_blocks():
    chk = check_compatible(_summary("bnci2014001"), _summary("physionet"))
    assert not chk["ok"]
    assert any(c["name"] == "数据集" and c["blocking"] for c in chk["blocking"])
    assert any("没有可比性" in c["detail"] for c in chk["blocking"])


def test_check_compatible_different_classes_blocks():
    chk = check_compatible(_summary(classes=("feet", "left_hand")),
                           _summary(classes=("left_hand", "tongue")))
    assert not chk["ok"]
    assert any(c["name"] == "类别集合" for c in chk["blocking"])
    assert class_key(_summary(classes=("a", "b"))) == ("a", "b")


def test_check_compatible_different_scheme_only_warns():
    chk = check_compatible(_summary(scheme="session"), _summary(scheme="loso"))
    assert chk["ok"], "划分方案不同不应阻止对比"
    assert any("口径不同" in n for n in chk["notes"])


# ------------------------------------------------------------------ 指标对照
def test_compare_metrics_diff_and_winner():
    rows = compare_metrics(_summary(acc=0.50, brier=0.80),
                           _summary(acc=0.60, brier=0.70))
    by = {r["key"]: r for r in rows}
    assert abs(by["acc"]["diff"] - 0.10) < 1e-9
    assert by["acc"]["better"] == "B"                 # 越大越好
    assert by["brier"]["better"] == "B"               # 越小越好（0.70 < 0.80）
    assert by["accuracy" if False else "acc"]["pct"] is True
    w = winner_summary(rows)
    assert w["b"] >= 2 and w["a"] == 0

    # B 更差时赢家反过来
    rows2 = compare_metrics(_summary(acc=0.60, brier=0.60),
                            _summary(acc=0.50, brier=0.90))
    by2 = {r["key"]: r for r in rows2}
    assert by2["acc"]["better"] == "A" and by2["brier"]["better"] == "A"


def test_compare_metrics_tie():
    rows = compare_metrics(_summary(acc=0.5), _summary(acc=0.5))
    assert {r["better"] for r in rows} == {"="}


def test_compare_metrics_missing_values():
    a = _summary(acc=0.5)
    b = _summary(acc=0.5)
    b["aggregate"].pop("ece_mean")
    rows = compare_metrics(a, b)
    ece = next(r for r in rows if r["key"] == "ece")
    assert ece["b"] is None and ece["diff"] is None and ece["better"] is None


def test_compare_per_subject():
    a = _summary(acc=0.5, per_subject=[{"subject": 1, "acc_mean": 0.5},
                                       {"subject": 2, "acc_mean": 0.4}])
    b = _summary(acc=0.6, per_subject=[{"subject": 1, "acc_mean": 0.7},
                                       {"subject": 3, "acc_mean": 0.2}])
    rows = compare_per_subject(a, b)
    by = {r["subject"]: r for r in rows}
    assert set(by) == {1, 2, 3}
    assert abs(by[1]["diff"] - 0.2) < 1e-9
    assert by[2]["b"] is None and by[3]["a"] is None


# ------------------------------------------------------------------ 结果页视图
@pytest.fixture()
def page(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    _ = app

    from ui.main_window import AppState
    from ui.result_page import ResultPage

    root = tmp_path / "experiments"
    mgr = ExperimentManager(root=root)

    def add(name, summary):
        exp = mgr.create(name, {"experiment": {"name": name}})
        (exp.dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (exp.dir / "metrics.csv").write_text(
            "fold,epoch,train_loss,val_loss,train_acc,val_acc\n"
            "0,1,1.30,1.25,0.30,0.35\n0,2,0.90,0.95,0.45,0.50\n", encoding="utf-8")
        mgr.register(exp, name=name, model=summary["model"], scheme=summary["scheme"],
                     acc_mean=summary["aggregate"]["acc_mean"])
        return exp

    monkeypatch.setattr("ui.result_page.ExperimentManager",
                        lambda *a, **k: ExperimentManager(root=root))
    p = ResultPage(AppState())
    p.add = add                                        # type: ignore[attr-defined]
    return p


def test_result_page_tabs_and_single_view(page):
    """图表分成三个子页，不再挤在一张图里。"""
    a = page.add("unittest_cmp_a", _summary(acc=0.55))
    page.refresh()
    page.load_experiment_dir(str(a.dir))
    assert page.tabs.count() == 3
    assert [page.tabs.tabText(i) for i in range(3)] == \
        ["训练曲线", "混淆矩阵", "按被试准确率"]
    # 每页独立 Figure
    assert len({id(page.fig_curve), id(page.fig_conf), id(page.fig_subj)}) == 3
    assert page.fig is page.fig_curve
    page.tabs.setCurrentIndex(1)
    assert page.fig is page.fig_conf
    assert len(page.fig_conf.axes) == 2            # 热力图 + 色条
    assert page.tbl.rowCount() >= 1 and page._compare is False


def test_result_page_compare_same_dataset(page):
    a = page.add("unittest_cmp_a", _summary(acc=0.55, model="eegnet"))
    b = page.add("unittest_cmp_b", _summary(acc=0.62, model="csp_lda"))
    page.load_comparison(str(a.dir), str(b.dir))
    assert page._compare is True and page._b_dir == str(b.dir)
    assert len(page.fig_conf.axes) == 4            # 两张热力图 + 两个色条
    headers = [page.tbl.horizontalHeaderItem(j).text()
               for j in range(page.tbl.columnCount())]
    assert headers[0] == "指标" and "差值 (B−A)" in headers
    text = page.txt_metrics.toPlainText()
    assert "一致性检查" in text and "可对比" in text
    acc_row = [i for i in range(page.tbl.rowCount())
               if page.tbl.item(i, 0).text().startswith("Accuracy")][0]
    assert page.tbl.item(acc_row, 3).text().startswith("+")
    assert page.tbl.item(acc_row, 4).text() == "B 更好"


def test_result_page_compare_refuses_different_dataset(page, monkeypatch):
    a = page.add("unittest_cmp_a", _summary("bnci2014001"))
    b = page.add("unittest_cmp_b", _summary("physionet"))
    warned: dict = {}
    monkeypatch.setattr("ui.result_page.QMessageBox.warning",
                        lambda *args, **kw: warned.update(msg=args[2]))
    page.load_comparison(str(a.dir), str(b.dir))
    assert page._compare is False                  # 未进入对比
    assert "无法对比" in warned.get("msg", "") or warned
    assert page._b_summary is None
