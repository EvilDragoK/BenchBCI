"""新手引导演示：一键跑通「原始数据 → 清洗 → 分段 → 训练 → 结果」并弹出链路拼图。

设计要点：
- 后台线程只负责计算（读数据 / 跑流水线 / 训练），Qt 组件与绘图一律在主线程，
  避免跨线程操作 GUI。
- 生成的图与「完整链路演示」一致：上排 = Raw→清洗→Epochs，下排 = 训练结果三图。
"""
from __future__ import annotations

import shutil
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (QCheckBox, QDialog, QFileDialog, QFormLayout,
                             QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit,
                             QPushButton, QScrollArea, QSpinBox, QVBoxLayout)

from core import PROJECT_ROOT
from ..workers import FunctionThread


def dataset_root() -> str:
    """从标准预处理配置里取数据集根目录。"""
    try:
        from core.pipeline import load_config

        cfg = load_config(PROJECT_ROOT / "configs" / "preprocess_bnci2014001.yaml")
        return str((cfg.get("dataset") or {}).get("root", ""))
    except Exception:  # noqa: BLE001
        return ""


class DemoResultDialog(QDialog):
    """展示链路拼图，并支持另存。"""

    def __init__(self, image_path: str, out: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("新手引导演示 —— 整条链路结果")
        self.resize(1140, 920)
        self._path = Path(image_path)

        acc = out.get("acc")
        msg = ("上排：Raw → 清洗 → 分段（Epochs）；下排：训练结果"
               "（训练曲线 / 混淆矩阵 / 按被试准确率）" if acc
               else "上排：Raw → 清洗 → 分段（Epochs）——本次未包含训练步骤")
        if acc:
            msg += (f"\n本次训练 {out.get('epochs', '')} epochs，"
                    f"acc = {acc['acc_mean'] * 100:.2f}% ± {acc['acc_std'] * 100:.2f}%"
                    f"（kappa {acc['kappa_mean']:.3f}）")
            if out.get("exp_dir"):
                msg += f"\n实验目录：{out['exp_dir']}"
        lb = QLabel(msg)
        lb.setWordWrap(True)
        lb.setStyleSheet("background:#f2f5f8; border:1px solid #dde3ea;"
                         "border-radius:4px; padding:8px;")

        view = QLabel()
        view.setPixmap(QPixmap(str(self._path)).scaled(
            1080, 780, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        area = QScrollArea()
        area.setWidget(view)
        area.setWidgetResizable(True)

        btn_save = QPushButton("另存为…")
        btn_close = QPushButton("关闭")
        btn_save.clicked.connect(self._save)
        btn_close.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addWidget(btn_save)
        row.addStretch(1)
        row.addWidget(btn_close)

        lay = QVBoxLayout(self)
        lay.addWidget(lb)
        lay.addWidget(area, 1)
        lay.addLayout(row)

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存链路图", "bci_full_chain.png",
                                              "PNG (*.png)")
        if path:
            shutil.copyfile(self._path, path)
            QMessageBox.information(self, "完成", f"已保存 {path}")


class DemoWizardDialog(QDialog):
    """新手引导演示向导：设置 → 后台跑链路 → 弹出拼图。"""

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.setWindowTitle("新手引导演示 —— 一键跑通整条链路")
        self.resize(780, 580)
        self._thread = None

        intro = QLabel(
            "点「▶ 开始演示」将自动依次执行：\n"
            "  ① 读取 BNCI2014001 一名被试的原始数据（Raw）\n"
            "  ② 清洗：50 Hz 陷波 → 4–40 Hz 带通 → 平均重参考\n"
            "  ③ 分段：cue 后 0.5–2.5 s + 振幅剔除 → Epochs\n"
            "  ④ 训练：EEGNet（按被试 T session 训练、E session 测试）\n"
            "  ⑤ 汇总：把每步结果拼成一张链路图弹出显示\n\n"
            "说明：第 ④ 步较慢（几十秒），可取消勾选只看预处理部分。")
        intro.setWordWrap(True)
        intro.setStyleSheet("background:#f2f5f8; border:1px solid #dde3ea;"
                           "border-radius:4px; padding:8px;")

        self.spin_subj = QSpinBox()
        self.spin_subj.setRange(1, 9)
        self.spin_subj.setValue(1)
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 200)
        self.spin_epochs.setValue(10)
        self.chk_train = QCheckBox("包含训练步骤（④⑤）")
        self.chk_train.setChecked(True)
        self.chk_train.toggled.connect(self.spin_epochs.setEnabled)

        f = QFormLayout()
        f.addRow("受试者编号", self.spin_subj)
        f.addRow("训练轮数 (epochs)", self.spin_epochs)
        f.addRow("", self.chk_train)

        self.btn_start = QPushButton("▶ 开始演示")
        self.btn_start.setStyleSheet("font-weight: bold;")
        self.btn_close = QPushButton("关闭")
        row = QHBoxLayout()
        row.addWidget(self.btn_start)
        row.addStretch(1)
        row.addWidget(self.btn_close)

        self.txt = QPlainTextEdit()
        self.txt.setReadOnly(True)

        lay = QVBoxLayout(self)
        lay.addWidget(intro)
        lay.addLayout(f)
        lay.addLayout(row)
        lay.addWidget(QLabel("进度日志"))
        lay.addWidget(self.txt, 1)

        self.btn_start.clicked.connect(self.start)
        self.btn_close.clicked.connect(self.reject)

    # ------------------------------------------------------------------ 执行
    def start(self):
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "提示", "演示正在运行中，请稍候…")
            return
        root = dataset_root()
        if not root or not Path(root).exists():
            QMessageBox.warning(
                self, "找不到数据",
                f"未找到数据集目录：{root or '（未配置）'}\n"
                f"请在「预处理工作台」确认数据集根目录，或修改 "
                f"configs/preprocess_bnci2014001.yaml。")
            return
        subj = int(self.spin_subj.value())
        epochs = int(self.spin_epochs.value())
        with_train = self.chk_train.isChecked()
        self.txt.clear()
        self.btn_start.setEnabled(False)
        self._log(f"数据集：{root}")

        def job(log):
            return self._run_pipeline(root, subj, epochs, with_train, log)

        th = FunctionThread(job, parent=self)
        self._thread = th
        th.logged.connect(self._log)
        th.ok.connect(self._done)
        th.fail.connect(self._fail)
        th.start()

    @staticmethod
    def _run_pipeline(root: str, subj: int, epochs: int, with_train: bool, log):
        """后台线程执行：读数据 → 清洗 → 分段 →（可选）训练。"""
        from core.io.bnci import load_subject_raw
        from core.pipeline import Pipeline, load_config
        from core.preprocess.operators import OPERATORS
        from ..preprocess_page import PreprocessPage

        def op(name, **kw):
            return (OPERATORS[name](kw), True)

        log("① 读取原始数据 …")
        raw, meta = load_subject_raw(root, subj, "T", eeg_only=True)
        log(f"   Raw：{len(raw.ch_names)} 通道 × {raw.n_times} 采样点 | "
            f"{raw.info['sfreq']:.0f} Hz | {raw.n_times / raw.info['sfreq'] / 60:.1f} 分钟"
            f" | cue {meta['n_trials']} 个")

        cut, used = PreprocessPage._preview_crop(raw)
        log(f"   （阶段 A 预览用前 {used:.0f} s 数据加速；阶段 B 用全量）")

        raw_steps = [
            op("set_montage", montage="standard_1005"),
            op("pick_eeg", keep_eog=False),
            op("notch_filter", freqs=[50.0]),
            op("bandpass_filter", l_freq=4.0, h_freq=40.0, method="fir", iir_order=4),
            op("re_reference", mode="average", channels=[]),
        ]

        log("② 阶段 A：清洗（陷波 → 带通 → 重参考）…")
        a_a, c_a = Pipeline(raw_steps).run(cut.copy())
        log(f"   输出仍是 {type(a_a).__name__}，形状 {a_a.get_data().shape}")

        log("③ 阶段 B：分段（0.5–2.5 s）+ 振幅剔除 …")
        steps_b = raw_steps + [
            op("epoching", tmin=0.5, tmax=2.5, baseline=True, classes=[]),
            op("amplitude_reject", threshold_uv=150.0),
        ]
        a_b, c_b = Pipeline(steps_b).run(raw.copy())
        log(f"   输出 Epochs：{len(a_b)} 个 × {len(a_b.ch_names)} 通道 × "
            f"{len(a_b.times)} 采样点（剔除 {c_b.get('rejected_trials', 0)} 个坏试次）")

        out = {"cut": cut, "aA": a_a, "cA": c_a, "raw": raw, "aB": a_b, "cB": c_b,
               "subj": subj, "epochs": epochs, "exp_dir": None, "acc": None}
        if with_train:
            from core.train import run_training

            log(f"④ 训练 EEGNet（{epochs} epochs，T→E 逐被试）…")
            cfg = load_config(PROJECT_ROOT / "configs" / "train_eegnet.yaml")
            cfg.setdefault("trainer", {})["epochs"] = int(epochs)
            cfg.setdefault("experiment", {})["name"] = "demo_eegnet"
            cfg["experiment"]["note"] = "新手引导演示"
            summary = run_training(cfg, log=log)
            out["exp_dir"] = summary.get("exp_dir")
            out["acc"] = summary["aggregate"]
        return out

    # ------------------------------------------------------------------ 收尾
    def _done(self, out):
        self.btn_start.setEnabled(True)
        self._log("⑤ 生成链路图 …")
        try:
            path = self._make_figures(out)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "绘图失败", f"{type(e).__name__}: {e}")
            return
        self._log(f"完成 → {path}")
        DemoResultDialog(str(path), out, self).exec()

    def _fail(self, msg):
        self.btn_start.setEnabled(True)
        self._log(f"失败：{msg}")
        QMessageBox.critical(self, "演示失败", msg)

    def _make_figures(self, out: dict) -> Path:
        """主线程绘图：三张分图 + 一张拼图（与「完整链路演示」一致）。"""
        from PIL import Image

        from ..result_page import ResultPage
        from .pipeline_preview_dialog import PipelinePreviewDialog

        outdir = PROJECT_ROOT / "experiments"
        outdir.mkdir(exist_ok=True)
        p1 = outdir / "demo_1_raw_preview.png"
        p2 = outdir / "demo_2_epochs_preview.png"

        d1 = PipelinePreviewDialog(out["cut"], out["aA"], out["cA"],
                                   "阶段 A：清洗（Raw → Raw）")
        d1.fig.savefig(p1, dpi=150, bbox_inches="tight")
        d2 = PipelinePreviewDialog(out["raw"], out["aB"], out["cB"],
                                   "阶段 B：分段（Raw → Epochs）")
        d2.fig.savefig(p2, dpi=150, bbox_inches="tight")

        p3 = None
        if out.get("exp_dir"):
            rp = ResultPage(self.state)
            rp.load_experiment_dir(out["exp_dir"])
            p3 = outdir / "demo_3_result_page.png"
            # 结果页现在是三个子页 → 分别出图后拼成一排（曲线 | 混淆矩阵 | 按被试）
            figs = [rp.fig_curve, rp.fig_conf, rp.fig_subj]
            parts = []
            for i, fig in enumerate(figs):
                tmp = outdir / f"_tmp_result_{i}.png"
                fig.savefig(tmp, dpi=140, bbox_inches="tight")
                parts.append(Image.open(tmp).convert("RGB"))
                tmp.unlink(missing_ok=True)
            width0 = 1500
            half = width0 // 2
            a1, a2, a3 = parts
            a1 = a1.resize((half, max(1, int(a1.height * half / a1.width))), Image.LANCZOS)
            a2 = a2.resize((half, max(1, int(a2.height * half / a2.width))), Image.LANCZOS)
            a3 = a3.resize((width0, max(1, int(a3.height * width0 / a3.width))),
                           Image.LANCZOS)
            hh = max(a1.height, a2.height)
            sheet = Image.new("RGB", (width0, hh + a3.height), "white")
            sheet.paste(a1, (0, 0))
            sheet.paste(a2, (half, 0))
            sheet.paste(a3, (0, hh))
            sheet.save(p3)

        width = 1500

        def scaled(path, w):
            im = Image.open(path).convert("RGB")
            return im.resize((w, max(1, int(im.height * w / im.width))), Image.LANCZOS)

        t1, t2 = scaled(p1, width // 2), scaled(p2, width // 2)
        h = max(t1.height, t2.height)
        t1, t2 = t1.resize((width // 2, h)), t2.resize((width // 2, h))
        full = outdir / "demo_full_chain.png"
        if p3 is not None:
            b = scaled(p3, width)
            canvas = Image.new("RGB", (width, h + b.height), "white")
            canvas.paste(t1, (0, 0))
            canvas.paste(t2, (width // 2, 0))
            canvas.paste(b, (0, h))
        else:
            canvas = Image.new("RGB", (width, h), "white")
            canvas.paste(t1, (0, 0))
            canvas.paste(t2, (width // 2, 0))
        canvas.save(full)
        return full

    def _log(self, text):
        self.txt.appendPlainText(str(text))
