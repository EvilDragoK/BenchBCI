"""结果分析页：**实验列表（可多选） + 图表分页 + 双实验对比 + 批量删除**。

- 左侧列表可多选：选 1 个 → 单实验视图；选 2 个 → 对比视图（先做数据集/类别
  一致性检查，通过才展示）；删除支持多选一次清掉。
- 右侧图表**分页显示**（训练曲线 / 混淆矩阵 / 按被试准确率），不再挤在一张图里。
- 图表配色跟随页面主题；导出图片导出的是**当前这一页**。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QListWidget,
                             QListWidgetItem, QMessageBox, QPlainTextEdit,
                             QPushButton, QSplitter, QTabWidget, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from core.experiment import ExperimentManager, dir_size
from core.train.compare import (check_compatible, compare_metrics,
                                compare_per_subject, dataset_key, winner_summary)
from core.train.evaluation import export_markdown_report
from .widgets.plotting import new_canvas


class ResultPage(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._summary: dict | None = None      # 主实验（A）
        self._config: dict | None = None
        self._exp_dir: str | None = None
        self._b_summary: dict | None = None    # 对比实验（B）
        self._b_dir: str | None = None
        self._compare = False

        # ================= 左：实验列表（多选） =================
        self.list_exp = QListWidget()
        self.list_exp.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection)
        self.list_exp.setToolTip("可多选：Ctrl/⌘ 点选、Shift 连选\n"
                                 "选 1 个 → 单实验视图；选 2 个 → 对比；删除可多选")
        self.list_exp.itemSelectionChanged.connect(self._on_selection_changed)

        self.btn_load = QPushButton("加载 / 对比")
        self.btn_load.setProperty("accent", True)
        self.btn_load.setToolTip("选中 1 个实验=查看；选中 2 个=对比（会先检查一致性）")
        self.btn_refresh = QPushButton("刷新列表")
        self.btn_delete = QPushButton("删除选中…")
        self.btn_delete.setProperty("danger", True)
        self.btn_delete.setToolTip("删除所有选中的实验（可多选），删除前会弹确认框")

        left_top = QLabel("实验列表（可多选）")
        left_tip = QLabel("Ctrl/⌘ 多选 · Shift 连选")
        left_tip.setStyleSheet("color:#888; font-size:11px;")

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_load, 1)
        btn_row.addWidget(self.btn_refresh)
        del_row = QHBoxLayout()
        del_row.addWidget(self.btn_delete, 1)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 6, 0)
        ll.addWidget(left_top)
        ll.addWidget(self.list_exp, 1)
        ll.addWidget(left_tip)
        ll.addLayout(btn_row)
        ll.addLayout(del_row)

        # ================= 右：图表分页 + 表格 =================
        self.lbl_loaded = QLabel("尚未加载实验")
        self.lbl_loaded.setWordWrap(True)
        self.btn_csv = QPushButton("导出 CSV")
        self.btn_png = QPushButton("导出当前页图片")
        self.btn_report = QPushButton("生成报告")

        tool = QHBoxLayout()
        tool.addWidget(self.lbl_loaded, 1)
        tool.addWidget(self.btn_csv)
        tool.addWidget(self.btn_png)
        tool.addWidget(self.btn_report)

        self.tabs = QTabWidget()
        self.fig_curve, self.canvas_curve = new_canvas(8.4, 5.2)
        self.fig_conf, self.canvas_conf = new_canvas(8.4, 5.4)
        self.fig_subj, self.canvas_subj = new_canvas(8.4, 4.6)
        self.tabs.addTab(self.canvas_curve, "训练曲线")
        self.tabs.addTab(self.canvas_conf, "混淆矩阵")
        self.tabs.addTab(self.canvas_subj, "按被试准确率")
        # 兼容旧脚本：fig 指向当前页
        self.ax_curve = self.fig_curve.add_subplot(111)
        self.ax_conf = self.fig_conf.add_subplot(111)
        self.ax_subj = self.fig_subj.add_subplot(111)

        self.tbl = QTableWidget(0, 0)
        self.txt_metrics = QPlainTextEdit()
        self.txt_metrics.setReadOnly(True)
        self.txt_metrics.setMaximumHeight(190)
        self.txt_metrics.setStyleSheet("font-size: 11px;")

        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(self.tbl, 1)
        self.lbl_bottom = QLabel("指标汇总（全部指标；↓ 表示越小越好）")
        bl.addWidget(self.lbl_bottom)
        bl.addWidget(self.txt_metrics)

        right_split = QSplitter(Qt.Orientation.Vertical)
        tabs_w = QWidget()
        tl = QVBoxLayout(tabs_w)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self.tabs)
        right_split.addWidget(tabs_w)
        right_split.addWidget(bottom)
        right_split.setStretchFactor(0, 3)
        right_split.setStretchFactor(1, 2)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addLayout(tool)
        rl.addWidget(right_split, 1)

        root_split = QSplitter(Qt.Orientation.Horizontal)
        root_split.addWidget(left)
        root_split.addWidget(right)
        root_split.setStretchFactor(0, 0)
        root_split.setStretchFactor(1, 1)
        root_split.setSizes([300, 1100])

        root = QVBoxLayout(self)
        root.addWidget(root_split, 1)

        # ================= 事件 =================
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_load.clicked.connect(self.load_selected)
        self.btn_csv.clicked.connect(self.export_csv)
        self.btn_png.clicked.connect(self.export_png)
        self.btn_report.clicked.connect(self.export_report)
        self.btn_delete.clicked.connect(self.delete_selected)
        self.refresh()

    # ================================================================ 工具
    @property
    def fig(self):
        """当前页的 matplotlib Figure（导出图片用）。"""
        return {0: self.fig_curve, 1: self.fig_conf, 2: self.fig_subj}.get(
            self.tabs.currentIndex(), self.fig_curve)

    @property
    def canvas(self):
        return {0: self.canvas_curve, 1: self.canvas_conf, 2: self.canvas_subj}.get(
            self.tabs.currentIndex(), self.canvas_curve)

    @staticmethod
    def _name(exp_dir: str) -> str:
        return Path(str(exp_dir)).name

    @staticmethod
    def _human_size(n: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024 or unit == "GB":
                return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
            n /= 1024.0
        return f"{n:.1f} GB"

    def selected_dirs(self) -> list[str]:
        """按列表顺序返回选中的实验目录。"""
        return [it.data(Qt.ItemDataRole.UserRole) for it in self.list_exp.selectedItems()]

    # ================================================================ 列表
    def refresh(self):
        keep = set(self.selected_dirs())
        self.list_exp.blockSignals(True)
        self.list_exp.clear()
        try:
            rows = ExperimentManager().list_experiments()
        except Exception:  # noqa: BLE001
            rows = []
        for r in rows:
            acc = r.get("acc_mean")
            it = QListWidgetItem(
                f"{self._name(r['dir'])}\n"
                f"    {r.get('model', '')} · {r.get('scheme', '')} · "
                f"acc={((acc or 0) * 100):.2f}%")
            it.setData(Qt.ItemDataRole.UserRole, r["dir"])
            it.setToolTip(f"{r['dir']}\n模型 {r.get('model', '')}　划分 "
                          f"{r.get('scheme', '')}　试次 {r.get('n_trials', '')}\n"
                          f"备注：{r.get('note', '') or '（无）'}")
            self.list_exp.addItem(it)
            if r["dir"] in keep:
                it.setSelected(True)
        self.list_exp.blockSignals(False)
        self._on_selection_changed()

    def _on_selection_changed(self):
        n = len(self.list_exp.selectedItems())
        self.btn_load.setText({0: "加载 / 对比", 1: "加载", 2: "对比这两个"}.get(n, "加载 / 对比"))
        self.btn_delete.setText(f"删除选中（{n}）…" if n else "删除选中…")

    def load_selected(self):
        dirs = self.selected_dirs()
        if not dirs:
            QMessageBox.information(self, "提示", "请先在左侧选择实验（Ctrl/⌘ 可多选）")
            return
        if len(dirs) == 1:
            self.load_experiment_dir(dirs[0])
        elif len(dirs) == 2:
            self.load_comparison(dirs[0], dirs[1])
        else:
            QMessageBox.information(
                self, "提示", f"已选 {len(dirs)} 个实验。\n"
                "对比最多支持 2 个实验；删除可以多选（点「删除选中」）。")

    # ================================================================ 加载
    def _read_summary(self, exp_dir: str) -> dict | None:
        p = Path(exp_dir) / "summary.json"
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "加载失败", f"读取 {p} 失败：{e}")
            return None

    def load_experiment_dir(self, exp_dir: str):
        """单实验模式（训练页双击实验也会走这里）。"""
        s = self._read_summary(exp_dir)
        if s is None:
            return
        self._summary, self._exp_dir = s, str(exp_dir)
        self._b_summary = self._b_dir = None
        self._compare = False
        try:
            import yaml

            self._config = yaml.safe_load(
                (Path(exp_dir) / "config.yaml").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            self._config = None
        self._select_in_list(exp_dir)
        self._render()

    def load_comparison(self, dir_a: str, dir_b: str):
        """双实验对比：**先检查数据集/类别一致性**，通过才展示。"""
        sa, sb = self._read_summary(dir_a), self._read_summary(dir_b)
        if sa is None or sb is None:
            return
        chk = check_compatible(sa, sb)
        if not chk["ok"]:
            detail = "\n".join(f"· {c['name']}：{c['a']}  vs  {c['b']}\n    {c['detail']}"
                               for c in chk["blocking"])
            QMessageBox.warning(
                self, "无法对比",
                "两个实验的数据条件不一致，不能直接对比：\n\n" + detail
                + "\n\n提示：请选择「同一个数据集、同一套类别」的两次实验"
                  "（例如同一数据集上换模型的两次训练）。")
            return
        self._summary, self._exp_dir = sa, str(dir_a)
        self._b_summary, self._b_dir = sb, str(dir_b)
        self._compare = True
        self._config = None
        self._render()

    def _select_in_list(self, exp_dir: str):
        for i in range(self.list_exp.count()):
            if self.list_exp.item(i).data(Qt.ItemDataRole.UserRole) == str(exp_dir):
                self.list_exp.blockSignals(True)
                self.list_exp.clearSelection()
                self.list_exp.item(i).setSelected(True)
                self.list_exp.scrollToItem(self.list_exp.item(i))
                self.list_exp.blockSignals(False)
                self._on_selection_changed()
                return

    def on_theme_changed(self):
        """切换页面主题后自动重绘（图表配色随主题走）。"""
        if self._summary is not None:
            self._render()

    # ================================================================ 渲染
    def _render(self):
        if self._summary is None:
            self._clear_view()
            return
        if self._compare and self._b_summary is not None:
            self.lbl_loaded.setText(
                f"<b>对比模式</b>　A：{self._name(self._exp_dir)}　vs　"
                f"B：{self._name(self._b_dir)}")
            self._plot_curves_pair()
            self._plot_confusion_pair()
            self._plot_subject_pair()
            self._fill_compare_table()
            self.lbl_bottom.setText("指标对比（A=先选中的实验，B=第二个；差 = B − A）")
        else:
            s = self._summary
            self.lbl_loaded.setText(
                f"<b>{self._name(self._exp_dir)}</b>　模型 {s.get('model', '')}　"
                f"划分 {s.get('scheme', '')}　折数 "
                f"{(s.get('aggregate') or {}).get('n_folds', '—')}")
            self._plot_curves()
            self._plot_confusion()
            self._plot_per_subject()
            self._fill_table(s)
            self.lbl_bottom.setText("各折指标（最后一行是合计；↓ 表示越小越好）")
            self._fill_metrics(s)
        for c in (self.canvas_curve, self.canvas_conf, self.canvas_subj):
            c.draw()

    def _clear_view(self):
        self._summary = self._config = self._exp_dir = None
        self._b_summary = self._b_dir = None
        self._compare = False
        for fig, ax, msg in ((self.fig_curve, self.ax_curve, "（未加载实验）"),
                             (self.fig_conf, self.ax_conf, ""),
                             (self.fig_subj, self.ax_subj, "")):
            fig.clear()
            ax = fig.add_subplot(111)
            ax.axis("off")
            if msg:
                ax.text(0.5, 0.5, msg, ha="center", va="center", color="#888",
                        fontsize=11)
        self.ax_curve = self.fig_curve.axes[0]
        self.ax_conf = self.fig_conf.axes[0]
        self.ax_subj = self.fig_subj.axes[0]
        for c in (self.canvas_curve, self.canvas_conf, self.canvas_subj):
            c.draw()
        self.tbl.setRowCount(0)
        self.tbl.setColumnCount(0)
        self.txt_metrics.setPlainText("")
        self.lbl_loaded.setText("尚未加载实验")

    @staticmethod
    def _read_metrics_csv(exp_dir: str, fold: str = "0") -> list[dict]:
        import csv

        path = Path(exp_dir) / "metrics.csv"
        if not path.exists():
            return []
        with open(path, newline="", encoding="utf-8") as f:
            return [r for r in csv.DictReader(f) if r.get("fold") == fold]

    def _draw_curve(self, ax, exp_dir: str, s: dict, tag: str, colors: list[str],
                    base: bool):
        """把一次实验的训练曲线画到 ax 上（base=实线，对比项=虚线）。"""
        from .theme import plot_palette

        colors = list(colors) or plot_palette()
        rows = self._read_metrics_csv(exp_dir)
        if not rows:
            if base:
                ax.text(0.5, 0.5, "（该实验没有 metrics.csv，无法画训练曲线）",
                        ha="center", va="center", color="#888", fontsize=10)
            return
        ep = [int(float(r["epoch"])) for r in rows]
        styles = [(colors[0], "-"), (colors[1], "--")]
        ls = styles[1][1] if not base else styles[0][1]
        c = (colors[1], colors[3]) if not base else (colors[0], colors[3])
        if rows[0].get("train_loss"):
            ax.plot(ep, [float(r["train_loss"]) for r in rows],
                    color=c[0], ls=ls, lw=1.7,
                    label=f"{tag} train_loss" if tag else "train_loss")
        if rows[0].get("val_loss"):
            ax.plot(ep, [float(r["val_loss"]) for r in rows],
                    color=c[1], ls="--" if base else ":", lw=1.4,
                    label=f"{tag} val_loss" if tag else "val_loss")
        acc_c = colors[2] if not base else colors[3]
        if rows[0].get("train_acc"):
            ax.plot(ep, [float(r["train_acc"]) for r in rows],
                    color=c[1], ls=ls, lw=1.7, alpha=0.9 if not base else 1.0,
                    label=f"{tag} train_acc" if tag else "train_acc")
        if rows[0].get("val_acc"):
            ax.plot(ep, [float(r["val_acc"]) for r in rows],
                    color=acc_c, ls="--" if base else ":", lw=1.4,
                    label=f"{tag} val_acc" if tag else "val_acc")

    def _plot_curves(self):
        from .theme import plot_palette

        self.fig_curve.clear()
        ax = self.fig_curve.add_subplot(111)
        self.ax_curve = ax
        ax.set_title("训练曲线（第 1 折）")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss / acc")
        self._draw_curve(ax, self._exp_dir, self._summary, "", plot_palette(), True)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    def _plot_curves_pair(self):
        from .theme import plot_palette

        colors = plot_palette()
        self.fig_curve.clear()
        ax = self.fig_curve.add_subplot(111)
        self.ax_curve = ax
        ax.set_title("训练曲线对比（第 1 折）")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss / acc")
        self._draw_curve(ax, self._exp_dir, self._summary, "A·", colors, True)
        self._draw_curve(ax, self._b_dir, self._b_summary, "B·", colors, False)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=7.5, ncol=2)
        ax.grid(alpha=0.3)

    def _draw_confusion(self, ax, s: dict, title: str, fig):
        from .theme import theme_cmap

        conf = np.asarray((s.get("aggregate") or {}).get("confusion_total"))
        if conf.size == 0:
            ax.axis("off")
            ax.text(0.5, 0.5, "（无混淆矩阵）", ha="center", va="center", color="#888")
            return
        names = s.get("class_names") or [str(i) for i in range(conf.shape[0])]
        im = ax.imshow(conf, cmap=theme_cmap())
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.set_xlabel("预测")
        ax.set_ylabel("真实")
        ax.set_title(title)
        thresh = conf.max() / 2 if conf.max() else 0.5
        for i in range(conf.shape[0]):
            for j in range(conf.shape[1]):
                ax.text(j, i, str(int(conf[i, j])), ha="center", va="center",
                        color="white" if conf[i, j] > thresh else "black", fontsize=8)

    def _plot_confusion(self):
        self.fig_conf.clear()
        ax = self.fig_conf.add_subplot(111)
        self.ax_conf = ax
        self._draw_confusion(ax, self._summary, "混淆矩阵（各折合计）", self.fig_conf)

    def _plot_confusion_pair(self):
        self.fig_conf.clear()
        axa = self.fig_conf.add_subplot(121)
        axb = self.fig_conf.add_subplot(122)
        self.ax_conf = axa
        self._draw_confusion(axa, self._summary,
                             f"A：{self._name(self._exp_dir)}", self.fig_conf)
        self._draw_confusion(axb, self._b_summary,
                             f"B：{self._name(self._b_dir)}", self.fig_conf)

    def _plot_per_subject(self):
        from .theme import current_theme

        t = current_theme()
        self.fig_subj.clear()
        ax = self.fig_subj.add_subplot(111)
        self.ax_subj = ax
        ps = self._summary.get("per_subject") or []
        if len(ps) < 1:
            ax.axis("off")
            ax.text(0.5, 0.5, "（该实验没有按被试聚合结果）\n"
                              "逐被试指标来自 LOSO / few-shot / 逐被试划分的实验",
                    ha="center", va="center", color="#888", fontsize=10)
            return
        names = [f"S{p['subject']:03d}" for p in ps]
        accs = [float(p.get("acc_mean", 0.0)) for p in ps]
        errs = [float(p.get("acc_std", 0.0) or 0.0) for p in ps]
        x = np.arange(len(ps))
        ax.bar(x, accs, color=t.accent, width=0.7)
        if any(errs):
            ax.errorbar(x, accs, yerr=errs, fmt="none", ecolor=t.mid, capsize=2.5, lw=1)
        m = float(np.mean(accs))
        ax.axhline(m, color=t.deep, ls="--", lw=1.2, label=f"均值 {m * 100:.1f}%")
        for xi, a in zip(x, accs):
            ax.text(xi, min(a + 0.03, 0.98), f"{a * 100:.0f}", ha="center",
                    fontsize=6.5, color=t.text)
        ax.set_xticks(x, names, rotation=60, fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_ylabel("准确率")
        ax.set_title(f"按被试准确率（{len(ps)} 名）")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

    def _plot_subject_pair(self):
        from .theme import current_theme

        t = current_theme()
        rows = compare_per_subject(self._summary, self._b_summary)
        self.fig_subj.clear()
        ax = self.fig_subj.add_subplot(111)
        self.ax_subj = ax
        if not rows:
            ax.axis("off")
            ax.text(0.5, 0.5, "（两个实验都没有按被试聚合结果）", ha="center",
                    va="center", color="#888", fontsize=10)
            return
        x = np.arange(len(rows))
        w = 0.38
        av = [np.nan if r["a"] is None else r["a"] for r in rows]
        bv = [np.nan if r["b"] is None else r["b"] for r in rows]
        ax.bar(x - w / 2, av, width=w, color=t.accent, label=f"A：{self._name(self._exp_dir)}")
        ax.bar(x + w / 2, bv, width=w, color=t.deep, alpha=0.85,
               label=f"B：{self._name(self._b_dir)}")
        ax.set_ylim(0, 1)
        ax.set_xticks(x, [f"S{r['subject']:03d}" for r in rows], rotation=60, fontsize=7)
        ax.set_ylabel("准确率")
        ax.set_title("按被试准确率对比")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

    # ================================================================ 表格
    def _fill_table(self, s):
        """各折指标表（多指标列）。"""
        folds = s.get("folds", [])
        headers = ["折", "tag", "acc", "balanced", "kappa", "MCC", "F1(macro)",
                   "特异度", "AUC", "best_val"]
        self.tbl.setColumnCount(len(headers))
        self.tbl.setRowCount(len(folds) + 1)
        self.tbl.setHorizontalHeaderLabels(headers)

        def g(d, key, nd=4):
            v = d.get(key)
            return "—" if v is None else f"{float(v):.{nd}f}"

        for i, f in enumerate(folds):
            vals = [f.get("fold"), f.get("tag"), g(f, "acc"), g(f, "balanced_acc"),
                    g(f, "kappa", 3), g(f, "mcc", 3), g(f, "f1_macro"),
                    g(f, "specificity_macro"), g(f, "auc_macro", 3),
                    f.get("best_val_acc") or "—"]
            for j, v in enumerate(vals):
                self.tbl.setItem(i, j, QTableWidgetItem(str(v)))

        agg = s.get("aggregate", {})

        def a(key, nd=4):
            m, sd = agg.get(f"{key}_mean"), agg.get(f"{key}_std")
            if m is None:
                return "—"
            return f"{m:.{nd}f}±{(sd or 0):.{nd}f}"

        total = [f"合计({agg.get('n_folds', len(folds))}折)", "",
                 a("acc"), a("balanced_acc"), a("kappa", nd=3), a("mcc", nd=3),
                 a("f1_macro"), a("specificity_macro"), a("auc_macro", nd=3), ""]
        for j, v in enumerate(total):
            it = QTableWidgetItem(str(v))
            it.setBackground(Qt.GlobalColor.lightGray)
            self.tbl.setItem(len(folds), j, it)
        self.tbl.resizeColumnsToContents()

    def _fill_compare_table(self):
        """对比模式：指标 / A / B / 差值 / 谁更好。"""
        rows = compare_metrics(self._summary, self._b_summary)
        headers = ["指标", f"A：{self._name(self._exp_dir)}",
                   f"B：{self._name(self._b_dir)}", "差值 (B−A)", "更好"]
        self.tbl.setColumnCount(len(headers))
        self.tbl.setHorizontalHeaderLabels(headers)
        self.tbl.setRowCount(len(rows))

        def fmt(v, std, pct):
            if v is None:
                return "—"
            if pct:
                return f"{v * 100:.2f} %" + (f" ± {std * 100:.2f}" if std else "")
            if abs(v) >= 100 or (std and abs(std) >= 100):
                return f"{v:,.1f}" + (f" ± {std:,.1f}" if std else "")
            return f"{v:.4f}" + (f" ± {std:.4f}" if std else "")

        for i, r in enumerate(rows):
            d = r["diff"]
            if d is None:
                dtxt = "—"
            elif r["pct"]:
                dtxt = f"{d * 100:+.2f} %"
            elif abs(d) >= 100:
                dtxt = f"{d:+,.1f}"
            else:
                dtxt = f"{d:+.4f}"
            mark = {"A": "A 更好", "B": "B 更好", "=": "持平", None: "—"}[r["better"]]
            for j, v in enumerate([r["label"], fmt(r["a"], r["a_std"], r["pct"]),
                                   fmt(r["b"], r["b_std"], r["pct"]), dtxt, mark]):
                it = QTableWidgetItem(str(v))
                if j == 3 and d is not None and r["lower_better"] is not None and abs(d) > 1e-12:
                    it.setForeground(Qt.GlobalColor.darkGreen if r["better"] == "B"
                                     else Qt.GlobalColor.darkRed)
                self.tbl.setItem(i, j, it)
        self.tbl.resizeColumnsToContents()

        chk = check_compatible(self._summary, self._b_summary)
        win = winner_summary(rows)
        lines = ["【一致性检查】"]
        for c in chk["checks"]:
            flag = "✓" if c["ok"] else ("✗" if c["blocking"] else "!")
            lines.append(f"  {flag} {c['name']}：{c['a']}　vs　{c['b']}"
                         + (f"　→ {c['detail']}" if c["detail"] else ""))
        lines += ["", "【结论】",
                  f"  数据集 {dataset_key(self._summary)}（一致，可对比）",
                  f"  有明确优劣的指标共 {win['n']} 项：A 更好 {win['a']} 项、"
                  f"B 更好 {win['b']} 项、持平 {win['tie']} 项"]
        acc_a = (self._summary.get("aggregate") or {}).get("acc_mean")
        acc_b = (self._b_summary.get("aggregate") or {}).get("acc_mean")
        if acc_a is not None and acc_b is not None:
            lines.append(f"  准确率：A {acc_a * 100:.2f}% → B {acc_b * 100:.2f}%"
                         f"（{(acc_b - acc_a) * 100:+.2f} 个百分点）")
        if chk["notes"]:
            lines += ["", "【注意】"] + [f"  · {x}" for x in chk["notes"]]
        self.txt_metrics.setPlainText("\n".join(lines))

    def _fill_metrics(self, s):
        """完整指标汇总（含 ITR / Brier / ECE / 加权平均 / 按类明细）。"""
        agg = s.get("aggregate", {})

        def row(label, key, pct=False, nd=3):
            m, sd = agg.get(f"{key}_mean"), agg.get(f"{key}_std")
            if m is None:
                return f"{label:<28}  —"
            if pct:
                return f"{label:<28}  {m * 100:7.2f} % ± {(sd or 0) * 100:5.2f} %"
            return f"{label:<28}  {m:8.{nd}f} ± {(sd or 0):6.{nd}f}"

        lines = [
            f"折数 {agg.get('n_folds', '—')}    模型 {s.get('model', '')}    "
            f"划分 {s.get('scheme', '')}    总试次 {s.get('n_trials', '—')}    "
            f"耗时 {s.get('total_seconds', '—')} s",
            "",
            row("Accuracy", "acc", pct=True),
            row("Balanced accuracy", "balanced_acc", pct=True),
            row("Cohen's kappa", "kappa"),
            row("MCC（Matthews 相关系数）", "mcc"),
            row("Precision (macro)", "precision_macro"),
            row("Recall / Sensitivity (macro)", "recall_macro"),
            row("Specificity (macro)", "specificity_macro"),
            row("F1 (macro)", "f1_macro"),
            row("Precision (weighted)", "precision_weighted"),
            row("Recall (weighted)", "recall_weighted"),
            row("F1 (weighted)", "f1_weighted"),
            row("AUC-ROC (macro, OvR)", "auc_macro"),
            row("AUC-PR (macro)", "auc_pr_macro"),
            row("Brier score ↓", "brier"),
            row("ECE ↓（期望校准误差）", "ece"),
        ]
        itr = agg.get("itr_mean")
        lines.append(
            f"{'ITR（信息传输率）':<28}  "
            + (f"{itr:8.2f} ± {(agg.get('itr_std') or 0):6.2f} bits/min"
               if itr is not None else "—（需配置 evaluation.trial_seconds）"))

        pcs = [f["per_class"] for f in (s.get("folds") or []) if f.get("per_class")]
        if pcs:
            names = pcs[0].get("names") or []
            lines += ["", "按类指标（跨折平均）："]
            for key, label in (("precision", "精确率"), ("recall", "召回率"),
                               ("specificity", "特异度"), ("f1", "F1    "),
                               ("auc", "AUC   ")):
                mat = np.array([[np.nan if v is None else float(v)
                                 for v in (pc.get(key) or [])] for pc in pcs], dtype=float)
                if mat.size == 0:
                    continue
                mean = np.nanmean(mat, axis=0)
                lines.append(f"  {label}: " + "   ".join(
                    f"{n}: {m:.3f}" for n, m in zip(names, mean)))
        self.txt_metrics.setPlainText("\n".join(lines))

    # ================================================================ 删除
    def _delete_box(self, dirs: list[Path], n_files: int, total_size: int):
        """构造批量删除确认框（按钮挂在 box._b_del / _b_keep / _b_cancel 上）。"""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("删除实验")
        head = dirs[0].name if len(dirs) == 1 else f"{len(dirs)} 个实验"
        box.setText(f"确定要删除{head}吗？")
        shown = "\n".join(f"· {d.name}" for d in dirs[:8])
        if len(dirs) > 8:
            shown += f"\n… 等共 {len(dirs)} 个"
        box.setInformativeText(
            f"{shown}\n\n合计：{n_files} 个文件，占用 {self._human_size(total_size)}\n"
            "「删除实验与文件」会移除权重、曲线与报告，无法恢复；\n"
            "「仅从列表移除」只清掉索引记录，磁盘文件保留。")
        box._b_del = box.addButton("删除实验与文件", QMessageBox.ButtonRole.DestructiveRole)
        box._b_keep = box.addButton("仅从列表移除", QMessageBox.ButtonRole.ActionRole)
        box._b_cancel = box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(box._b_cancel)      # 默认停在"取消"，防误触
        return box

    def _ask_delete(self, dirs: list[Path], n_files: int, total_size: int) -> str | None:
        """确认框：返回 'files'（删目录）/ 'keep'（仅移除索引）/ None（取消）。"""
        box = self._delete_box(dirs, n_files, total_size)
        box.exec()
        clicked = box.clickedButton()
        if clicked is box._b_del:
            return "files"
        if clicked is box._b_keep:
            return "keep"
        return None

    def delete_selected(self):
        """批量删除选中的实验（多选，先确认，再删索引与/或磁盘文件）。"""
        dirs = [Path(d) for d in self.selected_dirs()]
        if not dirs:
            QMessageBox.information(self, "提示", "请先在左侧选择要删除的实验（可多选）")
            return
        n_files = sum(sum(1 for f in d.rglob("*") if f.is_file())
                      for d in dirs if d.is_dir())
        total_size = sum(dir_size(d) for d in dirs)
        choice = self._ask_delete(dirs, n_files, total_size)
        if choice is None:
            return

        mgr = ExperimentManager()
        ok, freed, removed_dirs, failures = 0, 0, 0, []
        for d in dirs:
            try:
                info = mgr.delete_experiment(d, remove_files=(choice == "files"))
                ok += 1
                freed += info["freed_bytes"]
                removed_dirs += int(info["removed_dir"])
            except Exception as e:  # noqa: BLE001
                failures.append(f"{d.name}：{type(e).__name__}: {e}")

        loaded_gone = any(self._is_loaded(d) for d in dirs)
        if loaded_gone:
            self._clear_view()
        self.refresh()

        if choice == "files":
            msg = (f"已删除 {ok} 个实验"
                   + (f"（{removed_dirs} 个目录，释放 {self._human_size(freed)}）"
                      if removed_dirs else ""))
        else:
            msg = f"已从列表移除 {ok} 个实验（磁盘文件仍保留在 experiments/ 下）"
        if failures:
            msg += f"；{len(failures)} 个失败"
            QMessageBox.warning(self, "部分失败",
                                msg + "\n\n" + "\n".join(failures[:8]))
        sb = self.window().statusBar() if hasattr(self.window(), "statusBar") else None
        if sb is not None:
            sb.showMessage(msg, 6000)
        if choice == "keep" and not failures:
            QMessageBox.information(self, "完成", msg + "\n\n（想彻底删除可手动删这些目录）")

    def _is_loaded(self, d: Path) -> bool:
        for cur in (self._exp_dir, self._b_dir):
            if cur and Path(cur) == d:
                return True
        return False

    # ================================================================ 导出
    def export_csv(self):
        if self._summary is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出 CSV", "metrics.csv", "CSV (*.csv)")
        if not path:
            return
        import csv

        keys = ["exp", "fold", "tag", "acc", "kappa", "best_val_acc"]
        exps = [(self._name(self._exp_dir), self._summary)]
        if self._compare and self._b_summary is not None:
            exps.append((self._name(self._b_dir), self._b_summary))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for name, s in exps:
                for fd in s.get("folds", []):
                    w.writerow({"exp": name, **{k: v for k, v in fd.items()
                                                if k in keys}})
        QMessageBox.information(self, "完成", f"已导出 {path}")

    def export_png(self):
        if self._summary is None:
            return
        tab = self.tabs.tabText(self.tabs.currentIndex())
        path, _ = QFileDialog.getSaveFileName(
            self, "导出当前页图片", f"{tab}.png", "PNG (*.png);;SVG (*.svg)")
        if not path:
            return
        self.fig.savefig(path, dpi=150, bbox_inches="tight")
        QMessageBox.information(self, "完成", f"已导出「{tab}」→ {path}")

    def export_report(self):
        if self._summary is None or self._config is None:
            QMessageBox.information(
                self, "提示",
                "请先在左侧选中**单个**实验再生成报告（对比模式不支持生成报告）")
            return
        out = export_markdown_report(self._exp_dir, self._summary, self._config)
        figdir = Path(self._exp_dir) / "figures"
        figdir.mkdir(exist_ok=True)
        self.fig_curve.savefig(figdir / "curves.png", dpi=150, bbox_inches="tight")
        self.fig_conf.savefig(figdir / "confusion.png", dpi=150, bbox_inches="tight")
        QMessageBox.information(self, "完成", f"报告已生成：\n{out}")
