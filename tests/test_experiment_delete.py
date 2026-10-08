"""实验删除功能：索引 + 磁盘文件，含"拒绝删除目录外内容"的安全约束。

``test_result_page_delete_flow`` 走的是结果分析页的真实按钮流程（确认框被打桩）。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtCore import Qt  # noqa: E402

from core.experiment import ExperimentManager, dir_size  # noqa: E402


def _make_exp(mgr: ExperimentManager, name: str, acc: float = 0.5):
    import json

    exp = mgr.create(name, {"experiment": {"name": name}})
    summary = {
        "model": "eegnet", "scheme": "session", "n_trials": 10, "n_classes": 2,
        "class_names": ["a", "b"],
        "folds": [{"fold": 0, "tag": "holdout-1:1", "acc": acc, "kappa": 0.2}],
        "aggregate": {"n_folds": 1, "acc_mean": acc, "acc_std": 0.0, "kappa_mean": 0.2,
                      "kappa_std": 0.0, "confusion_total": [[3, 1], [2, 4]]},
        "per_subject": [],
    }
    (exp.dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (exp.ckpt_dir / "best.pt").write_bytes(b"0" * 2048)
    mgr.register(exp, name=name, model="eegnet", scheme="session", acc_mean=acc)
    return exp


# ------------------------------------------------------------------ 管理器层
def test_delete_experiment_removes_dir_and_row(tmp_path):
    mgr = ExperimentManager(root=tmp_path)
    exp = _make_exp(mgr, "unittest_del_a")
    assert len(mgr.list_experiments()) == 1
    assert dir_size(exp.dir) > 2000

    info = mgr.delete_experiment(exp.dir)
    assert info["removed_dir"] is True
    assert info["removed_rows"] == 1
    assert info["freed_bytes"] >= 2048
    assert not exp.dir.exists()
    assert mgr.list_experiments() == []
    assert mgr.find(str(exp.dir)) is None


def test_delete_experiment_keep_files(tmp_path):
    """仅从列表移除：索引没了，磁盘文件还在。"""
    mgr = ExperimentManager(root=tmp_path)
    exp = _make_exp(mgr, "unittest_del_keep")
    info = mgr.delete_experiment(exp.dir, remove_files=False)
    assert info["removed_dir"] is False and info["removed_rows"] == 1
    assert exp.dir.is_dir() and (exp.dir / "summary.json").exists()
    assert mgr.list_experiments() == []


def test_delete_experiment_refuses_outside_root(tmp_path):
    root = tmp_path / "experiments"
    mgr = ExperimentManager(root=root)
    outside = tmp_path / "别处的目录"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")

    with pytest.raises(ValueError, match="不在实验目录"):
        mgr.delete_experiment(outside)
    assert (outside / "keep.txt").exists()          # 一根毛都没动

    with pytest.raises(ValueError, match="根目录本身"):
        mgr.delete_experiment(root)
    assert root.is_dir()


def test_delete_missing_experiment_only_cleans_index(tmp_path):
    """目录已被手工删除：删除操作仍应清掉残留索引且不报错。"""
    mgr = ExperimentManager(root=tmp_path)
    exp = _make_exp(mgr, "unittest_del_gone")
    import shutil

    shutil.rmtree(exp.dir)
    info = mgr.delete_experiment(exp.dir)
    assert info["removed_dir"] is False and info["removed_rows"] == 1


# ------------------------------------------------------------------ 结果页流程
def _list_dirs(page) -> list[Path]:
    return [Path(page.list_exp.item(i).data(Qt.ItemDataRole.UserRole))
            for i in range(page.list_exp.count())]


def _select(page, dirs: set[Path]) -> int:
    """在列表里选中指定目录的项，返回选中的个数。"""
    n = 0
    for i in range(page.list_exp.count()):
        it = page.list_exp.item(i)
        if Path(it.data(Qt.ItemDataRole.UserRole)) in dirs:
            it.setSelected(True)
            n += 1
    return n


def _make_result_page(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])   # noqa: F841

    from ui.main_window import AppState
    from ui.result_page import ResultPage

    root = tmp_path / "experiments"
    mgr = ExperimentManager(root=root)
    # 结果页默认用 PROJECT_ROOT/experiments → 打桩成临时目录
    monkeypatch.setattr("ui.result_page.ExperimentManager",
                        lambda *args, **kw: ExperimentManager(root=root))
    return mgr, root, ResultPage(AppState())


def test_result_page_delete_flow(tmp_path, monkeypatch):
    mgr, root, page = _make_result_page(tmp_path, monkeypatch)
    a = _make_exp(mgr, "unittest_ui_a", acc=0.6)
    b = _make_exp(mgr, "unittest_ui_b", acc=0.4)
    page.refresh()

    assert page.list_exp.count() == 2
    assert page.btn_delete.text().startswith("删除选中")

    # 选中 b 并加载，再删除它 → 视图应被清空、列表只剩 a
    assert _select(page, {b.dir}) == 1
    page.load_selected()
    assert page._exp_dir == str(b.dir)
    assert page._summary is not None

    monkeypatch.setattr(page, "_ask_delete", lambda dirs, n, size: "files")
    page.delete_selected()

    assert not b.dir.exists()
    assert a.dir.is_dir()
    assert _list_dirs(page) == [a.dir]
    assert page._exp_dir is None and page._summary is None
    assert page.tbl.rowCount() == 0

    # 取消时什么都不发生
    _select(page, {a.dir})
    monkeypatch.setattr(page, "_ask_delete", lambda dirs, n, size: None)
    page.delete_selected()
    assert a.dir.is_dir() and page.list_exp.count() == 1


def test_result_page_delete_multiple(tmp_path, monkeypatch):
    """多选一次删掉多个实验。"""
    mgr, root, page = _make_result_page(tmp_path, monkeypatch)
    exps = [_make_exp(mgr, f"unittest_multi_{i}") for i in range(3)]
    keep = _make_exp(mgr, "unittest_multi_keep")
    page.refresh()
    assert page.list_exp.count() == 4

    assert _select(page, {e.dir for e in exps}) == 3
    assert len(page.selected_dirs()) == 3

    captured: dict = {}

    def fake_ask(dirs, n_files, total_size):
        captured.update(n=len(dirs), files=n_files, size=total_size)
        return "files"

    monkeypatch.setattr(page, "_ask_delete", fake_ask)
    page.delete_selected()

    # 每个实验 3 个文件（config.yaml / summary.json / ckpt/best.pt）
    assert captured["n"] == 3 and captured["files"] == 9
    assert captured["size"] > 0
    for e in exps:
        assert not e.dir.exists()
    assert keep.dir.is_dir()
    assert _list_dirs(page) == [keep.dir]
