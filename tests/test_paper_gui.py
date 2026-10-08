"""「从论文提取参数」对话框的界面链路测试（无头运行）。

覆盖：拖拽 PDF 进入对话框 → 正文/补充材料识别与组装 → 离线规则抽取 →
溯源生成 → 应用到训练控制台 → 训练配置里带上 paper_params。
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QMimeData, QPointF, Qt, QUrl          # noqa: E402
from PyQt6.QtGui import QDragEnterEvent, QDropEvent            # noqa: E402
from PyQt6.QtWidgets import QApplication                       # noqa: E402

from tests.test_paper_text import MAIN_TEXT, SUPP_TEXT, make_pdf  # noqa: E402

pytestmark = pytest.mark.usefixtures("qapp")


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _drop(dlg, paths):
    """模拟把文件拖进对话框（走真正的 dragEnter/drop 事件处理）。"""
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    enter = QDragEnterEvent(QPointF(20, 20).toPoint(), Qt.DropAction.CopyAction,
                            mime, Qt.MouseButton.LeftButton,
                            Qt.KeyboardModifier.NoModifier)
    dlg.dragEnterEvent(enter)
    ev = QDropEvent(QPointF(20, 20), Qt.DropAction.CopyAction, mime,
                    Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    dlg.dropEvent(ev)
    return enter.isAccepted(), ev.isAccepted()


def _wait(dlg, timeout_s=20.0):
    from PyQt6.QtTest import QTest

    loops = int(timeout_s * 20)
    for _ in range(loops):
        QTest.qWait(50)
        QApplication.processEvents()
        if dlg._thread is None or not dlg._thread.isRunning():
            return True
    return False


def test_drop_pdf_and_apply_to_train_config(tmp_path, qapp):
    from ui.train_page import TrainPage
    from ui.widgets.paper_extract_dialog import PaperExtractDialog

    main_pdf = make_pdf(tmp_path / "Smith2023_main.pdf", [MAIN_TEXT])
    supp_pdf = make_pdf(tmp_path / "Smith2023_supplementary.pdf", [SUPP_TEXT])

    class State:            # 训练页只需要一个可容纳 raw 的状态对象
        raw = None
        dataset = None

    page = TrainPage(State())
    dlg = PaperExtractDialog(parent=page, on_apply=page._apply_from_paper,
                             base_config=page.build_config)
    dlg.chk_use_llm.setChecked(False)            # 离线规则，不联网

    enter_ok, drop_ok = _drop(dlg, [main_pdf, supp_pdf])
    assert enter_ok and drop_ok, "支持的后缀应该被 dragEnter/drop 接受"
    assert _wait(dlg), "读取文件的后台线程没有结束"

    # 文件列表：两个文件 + 类型判定
    assert dlg.list_files.count() == 2
    items = [dlg.list_files.item(i).text() for i in range(dlg.list_files.count())]
    assert any("论文正文" in t for t in items) and any("补充材料" in t for t in items)
    # 组装：补充材料排在正文前
    assert dlg._asm_text.index("Implementation details") < dlg._asm_text.find("3.2 Methods")
    assert "优先小节" in dlg.lbl_asm.text()
    assert dlg.txt_paper.toPlainText() == dlg._asm_text
    assert dlg._is_assembled_text()

    # 不支持的类型要拒绝
    bad = tmp_path / "figure.png"
    bad.write_bytes(b"\x89PNG\r\n")
    enter_bad, _ = _drop(dlg, [bad])
    assert not enter_bad

    # 识别（离线）→ 溯源
    dlg.run_extract()
    assert _wait(dlg), "识别线程没有结束"
    assert dlg._result is not None
    vals = {k: dlg._result["fields"][k]["value"] for k in
            ("epochs", "batch_size", "lr", "optimizer", "seed",
             "euclidean_alignment", "augmentation")}
    assert vals == {"epochs": 200, "batch_size": 64, "lr": pytest.approx(1e-3),
                    "optimizer": "adamw", "seed": 42, "euclidean_alignment": True,
                    "augmentation": True}
    assert dlg._prov is not None and dlg._prov["method"] == "rules"
    assert dlg._prov["input"]["sources"] and dlg._prov["input"]["used_sections"]
    assert dlg.chk_bind.isChecked() and dlg.btn_prov.isEnabled()

    # 应用 → 训练页数值 + 来源绑定
    dlg._apply()
    assert (page.spin_epochs.value(), page.spin_batch.value(),
            page.combo_opt.currentText(), page.spin_seed.value()) == (200, 64, "adamw", 42)
    assert page.spin_lr.value() == pytest.approx(1e-3)
    assert page.chk_ea.isChecked() is True
    assert page.spin_shift.value() > 0 and page.spin_chdrop.value() > 0
    assert page._paper_params and "补充材料" in page.lbl_paper.text()
    cfg = page.build_config()
    assert cfg["paper_params"]["fields"]["epochs"]["value"] == 200
    assert "Smith2023_supplementary.pdf" in str(cfg["paper_params"]["input"]["sources"])

    # 清除按钮：数值保留，来源不再写入
    page._set_paper_params(None)
    assert "paper_params" not in page.build_config()
    assert page.spin_epochs.value() == 200
    dlg.close()


def test_paste_text_path_has_no_file_provenance(tmp_path, qapp):
    from ui.widgets.paper_extract_dialog import PaperExtractDialog, SAMPLE

    dlg = PaperExtractDialog()
    dlg.chk_use_llm.setChecked(False)
    dlg.txt_paper.setPlainText(SAMPLE)
    dlg.run_extract()
    assert _wait(dlg)
    assert dlg._prov is not None
    assert dlg._prov["input"]["sources"] == []
    assert any("手工粘贴" in n for n in dlg._prov["input"]["notes"])
    assert dlg.list_files.count() == 0
    dlg.close()


def test_unbind_checkbox_prevents_provenance(tmp_path, qapp):
    from ui.widgets.paper_extract_dialog import PaperExtractDialog, SAMPLE

    got = {}

    def on_apply(cfg, prov=None):
        got["cfg"], got["prov"] = cfg, prov

    dlg = PaperExtractDialog(on_apply=on_apply)
    dlg.chk_use_llm.setChecked(False)
    dlg.chk_bind.setChecked(False)
    dlg.txt_paper.setPlainText(SAMPLE)
    dlg.run_extract()
    assert _wait(dlg)
    dlg._apply()
    assert got["prov"] is None and "trainer" in got["cfg"]
    dlg.close()


def test_one_arg_apply_callback_still_works(qapp):
    from ui.widgets.paper_extract_dialog import PaperExtractDialog, SAMPLE

    got = {}
    dlg = PaperExtractDialog(on_apply=lambda cfg: got.update(cfg=cfg))
    dlg.chk_use_llm.setChecked(False)
    dlg.txt_paper.setPlainText(SAMPLE)
    dlg.run_extract()
    assert _wait(dlg)
    dlg._apply()                     # 旧签名（1 参数）不应报错
    assert got["cfg"]["trainer"]["epochs"] == 200
    dlg.close()
