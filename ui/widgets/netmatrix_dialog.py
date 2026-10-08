"""协变 / 相似性 / 外部矩阵工作台。

覆盖**非 EEG 时间序列**的脑网络来源（都是"表格 / 矩阵"形态）：

1. **T1 结构协变**：行=被试、列=ROI 的灰质体积/皮层厚度表 → 组水平协变矩阵
2. **ASL-CBF / PET**：行=ROI、列=指标（或多被试多指标）→ 个体水平相似性网络
   （Pearson / Spearman / KL / JSD / **MIND 多指标联合**）
3. **外部矩阵文件**：其他软件导出的连接矩阵（如 PANDA 的网络矩阵、SPM-DCM 的 A 矩阵）
   （``.mat`` / ``.csv`` / ``.npy``）→ 稀疏化 + 图论指标 + 可视化

不依赖任何影像解码库：数据怎么来（FreeSurfer / CAT12 / PANDA…）由用户决定，
这里负责"进来之后的分析"。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                             QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
                             QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                             QPlainTextEdit, QPushButton, QRadioButton, QSpinBox,
                             QSplitter, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

from core.linked import export_tables, linked_analysis, plot_linked_heatmap
from core.netcov import (LAG_INTERP_LABELS, SIMILARITY_LABELS, SIMILARITY_METHODS,
                         group_covariance, individual_similarity, load_matrix_file,
                         load_table_file, save_matrix_file)
from core.networks import (CONNECT_LABELS, CONNECT_METHODS, SPARSIFY_LABELS,
                           SPARSIFY_STRATEGIES, binarize, connectivity_matrix,
                           graph_metrics, sparsify)
from core.neuroimg import (ATLAS_PRESETS, HAVE_NIFTI, INSTALL_HINT, auto_extract,
                           describe_image, save_roi_table)

from .plotting import new_canvas


class NetMatrixDialog(QDialog):
    """表格/矩阵 → 脑网络（协变、相似性、外部矩阵）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("协变 / 相似性 / 变量关联 / 外部矩阵工作台")
        self.resize(1080, 700)
        self._matrix: np.ndarray | None = None
        self._labels: list[str] = []
        self._table: np.ndarray | None = None
        self._col_names: list[str] = []          # 变量表的列名（数值变量）
        self._group_cols: dict = {}              # 非数值列（可作为 LMM 分组变量，如被试编号）
        self._linked: dict | None = None         # 最近一次变量关联分析结果

        # ---------------- 数据来源 ----------------
        self.rb_table = QRadioButton("ROI 数值表（CSV / NPY / TXT）")
        self.rb_file = QRadioButton("已有连接矩阵（.npy / .csv / .mat）")
        self.rb_image = QRadioButton("影像 + 图集（NIfTI）")
        self.rb_table.setChecked(True)
        self.btn_open = QPushButton("打开文件…")
        self.btn_open.clicked.connect(self.open_file)
        self.lbl_file = QLabel("（尚未选择文件）")
        self.lbl_file.setStyleSheet("color:#888; font-size:11px;")
        src_row = QHBoxLayout()
        src_row.addWidget(self.rb_table)
        src_row.addWidget(self.rb_file)
        src_row.addWidget(self.rb_image)
        src_row.addWidget(self.btn_open)
        src_row.addWidget(self.lbl_file, 1)

        # ---------------- 影像 + 图集（可选依赖） ----------------
        self.btn_img = QPushButton("影像文件…")
        self.btn_img.setToolTip("4D BOLD（.nii.gz 多时间点）或 3D 影像（ASL-CBF / PET-SUV / T1）")
        self.btn_img.clicked.connect(self.choose_image)
        self.lbl_img = QLabel("（未选择影像）")
        self.lbl_img.setStyleSheet("color:#888; font-size:11px;")
        self.combo_atlas = QComboBox()
        self.combo_atlas.addItem("本地标签图（选择文件）…", userData="local")
        for key, (_fn, display, note) in ATLAS_PRESETS.items():
            self.combo_atlas.addItem(f"内置图集：{display}", userData=key)
        self.combo_atlas.addItem("内置图集：Harvard-Oxford 皮层左（cortl）",
                                 userData="harvard_oxford_cortl")
        self.combo_atlas.addItem("球坐标 ROI 表（CSV）…", userData="spheres")
        self.lbl_atlas = QLabel("")
        self.lbl_atlas.setStyleSheet("color:#888; font-size:11px;")
        self.spin_tr = QDoubleSpinBox()
        self.spin_tr.setRange(0.05, 20.0)
        self.spin_tr.setDecimals(3)
        self.spin_tr.setValue(2.0)
        self.spin_tr.setToolTip("重复时间 TR（秒）；4D 影像不填则读影像头")
        self.spin_fwhm = QDoubleSpinBox()
        self.spin_fwhm.setRange(0.0, 20.0)
        self.spin_fwhm.setDecimals(1)
        self.spin_fwhm.setValue(0.0)
        self.spin_fwhm.setToolTip("空间平滑 FWHM（毫米），0 = 不平滑")
        self.chk_standardize = QCheckBox("标准化（z-score）")
        self.chk_detrend = QCheckBox("去线性漂移")
        self.chk_detrend.setChecked(True)
        self.btn_extract = QPushButton("提取 ROI")
        self.btn_extract.clicked.connect(self.extract_image)

        img_form = QFormLayout()
        img_row = QHBoxLayout()
        img_row.addWidget(self.btn_img)
        img_row.addWidget(self.lbl_img, 1)
        img_form.addRow("影像", img_row)
        atlas_row = QHBoxLayout()
        atlas_row.addWidget(self.combo_atlas)
        atlas_row.addWidget(self.lbl_atlas, 1)
        img_form.addRow("图集", atlas_row)
        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("TR"))
        param_row.addWidget(self.spin_tr)
        param_row.addWidget(QLabel("平滑"))
        param_row.addWidget(self.spin_fwhm)
        param_row.addWidget(self.chk_standardize)
        param_row.addWidget(self.chk_detrend)
        param_row.addWidget(self.btn_extract)
        img_form.addRow(param_row)
        self.box_img = QGroupBox("影像提取（可选依赖 nibabel + nilearn）")
        self.box_img.setLayout(img_form)
        if not HAVE_NIFTI:
            self.box_img.setEnabled(False)
            self.rb_image.setEnabled(False)
            self.lbl_img.setText("未安装可选依赖：pip install nibabel nilearn")
        else:
            self.lbl_img.setText("（未选择影像）")
        self._img_path = ""
        self._atlas_path = ""
        self._spheres_path = ""
        self._ts = None
        self.combo_atlas.currentIndexChanged.connect(self._on_atlas_changed)

        # ---------------- 参数 ----------------
        self.combo_sim = QComboBox()
        for k in SIMILARITY_METHODS:
            self.combo_sim.addItem(SIMILARITY_LABELS[k], userData=k)
        self.combo_rows = QComboBox()
        self.combo_rows.addItem("行 = 被试（→ 组水平结构协变）", userData="subjects")
        self.combo_rows.addItem("行 = ROI（→ 个体水平相似性）", userData="roi")
        self.spin_metrics = QSpinBox()
        self.spin_metrics.setRange(1, 20)
        self.spin_metrics.setValue(1)
        self.spin_metrics.setToolTip("仅 MIND 用：把列分成几组指标（如 体积/厚度/曲率 三指标）")
        self.edit_weights = QLineEdit("")
        self.edit_weights.setPlaceholderText("仅 MIND 用，如 2,1,1（留空=等权）")

        self.combo_sparsify = QComboBox()
        for k in SPARSIFY_STRATEGIES:
            self.combo_sparsify.addItem(SPARSIFY_LABELS[k], userData=k)
        self.combo_sparsify.setCurrentIndex(1)          # 默认比例阈值
        self.spin_density = QDoubleSpinBox()
        self.spin_density.setRange(0.05, 1.0)
        self.spin_density.setSingleStep(0.05)
        self.spin_density.setValue(0.2)
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(-1e6, 1e6)
        self.spin_threshold.setDecimals(4)
        self.spin_threshold.setValue(0.0)
        self.spin_threshold.setToolTip("仅「绝对阈值」用；按相似性量纲填（矩阵文件模式）")

        self.combo_kind = QComboBox()
        self.combo_kind.addItem("相似性 / 协变（表格：ROI × 指标）", userData="sim")
        self.combo_kind.addItem("时间序列连接（ROI × 时间，12 种方法）", userData="conn")
        self.combo_kind.addItem("变量关联（RF 重要性 + Pearson + LMM）", userData="linked")
        self.combo_kind.currentIndexChanged.connect(self._on_kind_changed)

        form = QFormLayout()
        form.addRow("分析类别", self.combo_kind)
        form.addRow("方法", self.combo_sim)
        form.addRow("表方向", self.combo_rows)
        form.addRow("MIND 指标数", self.spin_metrics)
        form.addRow("MIND 权重", self.edit_weights)
        form.addRow("稀疏化", self.combo_sparsify)
        form.addRow("密度 / 分位", self.spin_density)
        form.addRow("绝对阈值", self.spin_threshold)
        box = QGroupBox("分析参数")
        box.setLayout(form)

        # ---------------- 变量关联分析（RF + Pearson + LMM） ----------------
        self.combo_response = QComboBox()
        self.combo_response.setToolTip("目标变量（因变量）：图中央那个「hub」")
        self.combo_group = QComboBox()
        self.combo_group.setToolTip(
            "LMM 的分组变量（随机截距）：同一被试/地点/年份的重复观测必须填它，\n"
            "否则组间差异会被错算成变量效应。留空 = 用 OLS")
        self.spin_trees = QSpinBox()
        self.spin_trees.setRange(100, 4000)
        self.spin_trees.setSingleStep(100)
        self.spin_trees.setValue(500)
        self.spin_trees.setToolTip("随机森林树数（越多越稳、越慢）")
        self.combo_effect = QComboBox()
        self.combo_effect.addItem("线性混合效应模型（LMM，控制分组）", userData="lmm")
        self.combo_effect.addItem("普通最小二乘（OLS，忽略分组）", userData="ols")
        self.chk_linked_std = QCheckBox("标准化")
        self.chk_linked_std.setChecked(True)
        self.chk_linked_std.setToolTip("先把目标变量与各自变量 z-score：标准化 β 之间才可比（弧线宽度才有意义）")
        self.chk_linked_log = QCheckBox("对正偏变量取 log10")
        self.chk_linked_log.setToolTip(
            "频带功率、物种丰度这类正偏重尾指标建议勾选：\n"
            "不取对数时少数极端值会主导方差，随机森林 R² 甚至可能为负\n"
            "（要求目标变量与自变量全为正数，否则自动跳过）")
        self.chk_linked_annot = QCheckBox("标注显著弧线的 p 值")
        self.chk_linked_topcorr = QCheckBox("标注最强相关对")
        self.chk_linked_imp = QCheckBox("标注重要性数值")

        linked_form = QFormLayout()
        linked_form.addRow("目标变量", self.combo_response)
        linked_form.addRow("分组变量", self.combo_group)
        eff_row = QHBoxLayout()
        eff_row.addWidget(self.combo_effect, 1)
        eff_row.addWidget(QLabel("树数"))
        eff_row.addWidget(self.spin_trees)
        linked_form.addRow("效应模型", eff_row)
        annot_row = QHBoxLayout()
        annot_row.addWidget(self.chk_linked_std)
        annot_row.addWidget(self.chk_linked_log)
        annot_row.addWidget(self.chk_linked_annot)
        annot_row.addWidget(self.chk_linked_topcorr)
        annot_row.addWidget(self.chk_linked_imp)
        linked_form.addRow("变换 / 标注", annot_row)
        self.lbl_linked = QLabel("载入一个「变量表」（第一行是变量名）后即可选择目标变量")
        self.lbl_linked.setStyleSheet("color:#888; font-size:11px;")
        self.lbl_linked.setWordWrap(True)
        self.lbl_linked.setMinimumWidth(120)
        linked_form.addRow(self.lbl_linked)
        self.box_linked = QGroupBox("变量关联分析（Linked Heatmap）")
        self.box_linked.setLayout(linked_form)
        self.box_linked.setToolTip(
            "一张图回答三件事：\n· 顶部柱条 = 随机森林（留出集置换）重要性\n"
            "· 中间三角 = 变量两两 Pearson 相关（颜色=方向，方块=强度）\n"
            "· 底部弧线 = LMM/OLS 效应（颜色=方向，线宽∝|标准化β|，实/虚=显著性）\n"
            "注：相关不是因果；RF 重要性只表示重要程度，不含方向")

        self.btn_calc = QPushButton("计算网络")
        self.btn_calc.setProperty("accent", True)
        self.btn_calc.clicked.connect(self.compute)
        self.btn_export = QPushButton("导出矩阵")
        self.btn_export.clicked.connect(self.export)
        self.btn_png = QPushButton("导出图片")
        self.btn_png.clicked.connect(self.export_fig)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_calc)
        btn_row.addWidget(self.btn_export)
        btn_row.addWidget(self.btn_png)
        btn_row.addStretch(1)

        self.lbl_sum = QLabel("提示：T1 组协变用「行=被试」；ASL/PET 的 ROI 数值用「行=ROI」。")
        self.lbl_sum.setWordWrap(True)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addLayout(src_row)
        ll.addWidget(self.box_img)
        ll.addWidget(box)
        ll.addWidget(self.box_linked)
        ll.addLayout(btn_row)
        ll.addWidget(self.lbl_sum)
        ll.addStretch(1)

        # ---------------- 右侧：热力图 + 图属性 ----------------
        self.fig, self.canvas = new_canvas(7.2, 5.4)
        self.tbl = QTableWidget(0, 5)
        self.tbl.setHorizontalHeaderLabels(["节点", "度", "聚类系数", "局部效率", "强度"])
        self.tbl.verticalHeader().setVisible(False)
        self.txt = QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setMaximumHeight(150)
        self.txt.setStyleSheet("font-size:11px;")

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(self.canvas, 1)
        rl.addWidget(self.tbl, 1)
        rl.addWidget(self.txt)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 720])
        self._splitter = splitter

        root = QVBoxLayout(self)
        root.addWidget(splitter, 1)
        self._on_kind_changed()          # 初始化方法下拉框

    # ================================================================ 载入
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择数据文件", str(Path.home()),
            "数据文件 (*.csv *.txt *.tsv *.npy *.mat);;所有文件 (*)")
        if not path:
            return
        self.lbl_file.setText(str(path))
        try:
            M, labels = load_matrix_file(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "读取失败", f"{type(e).__name__}: {e}")
            return
        self._labels = labels
        if M.shape[0] == M.shape[1]:
            self.rb_file.setChecked(True)
            self._matrix = M
            self.rb_table.setEnabled(False)
            self._draw(M, f"已载入矩阵 {Path(path).name}（{M.shape[0]}×{M.shape[1]}）")
        else:
            self.rb_table.setChecked(True)
            self.rb_table.setEnabled(True)
            self._table = M
            # 变量表还要**列名**（变量名）：矩阵读取器只给行标签，这里用专用读取器补齐
            self._col_names = [f"col{i + 1}" for i in range(M.shape[1])]
            self._group_cols = {}
            try:
                table, col_names, _rows, non_num = load_table_file(path)
                table = np.asarray(table, dtype=float)
                if table.shape[1] == M.shape[1] or len(non_num):
                    self._table, self._col_names, self._group_cols = \
                        table, list(col_names), dict(non_num)
            except Exception:  # noqa: BLE001 - 读不到列名不影响后续网络分析
                pass
            self._fill_linked_combos()
            self._draw(self._table, f"已载入数值表 {Path(path).name}"
                                    f"（{self._table.shape[0]} 行 × {self._table.shape[1]} 列）"
                                    f"→ 选择「分析类别」后点「计算」")

    # ================================================================ 影像
    def _on_kind_changed(self, *_):
        """切换分析类别 → 重新填充方法下拉框 / 启停相关参数。"""
        kind = self.combo_kind.currentData() or "sim"
        self.combo_sim.blockSignals(True)
        self.combo_sim.clear()
        if kind == "conn":
            for k in CONNECT_METHODS:
                self.combo_sim.addItem(f"{k} · {CONNECT_LABELS.get(k, k)}", userData=k)
        else:
            for k in SIMILARITY_METHODS:
                self.combo_sim.addItem(SIMILARITY_LABELS[k], userData=k)
        self.combo_sim.blockSignals(False)
        for w in (self.combo_rows, self.spin_metrics, self.edit_weights):
            w.setEnabled(kind == "sim")
        self.spin_tr.setEnabled(kind == "conn")
        is_linked = kind == "linked"
        self.box_linked.setVisible(is_linked)
        # 变量关联模式下，连接方法/稀疏化/密度没有意义 → 一并禁用，避免误操作
        for w in (self.combo_sim, self.combo_sparsify, self.spin_density, self.spin_threshold):
            w.setEnabled(not is_linked)
        if is_linked:
            self._fill_linked_combos()

    def _fill_linked_combos(self):
        """把变量表的列名填进「目标变量 / 分组变量」下拉框。"""
        names = list(self._col_names)
        n_rows = 0 if self._table is None else int(self._table.shape[0])
        self.combo_response.blockSignals(True)
        self.combo_group.blockSignals(True)
        self.combo_response.clear()
        self.combo_group.clear()
        self.combo_response.addItems(names)
        self.combo_group.addItem("（不使用分组 → OLS）", userData="")
        # 非数值列（被试编号 / 分组标签）优先作为 LMM 分组变量的候选
        for g in self._group_cols:
            self.combo_group.addItem(f"{g}（文本列，适合分组）", userData=f"!{g}")
        self.combo_group.addItems(names)
        if names:                                  # 默认拿最后一列当目标变量（常见排版）
            self.combo_response.setCurrentIndex(len(names) - 1)
        self.combo_response.blockSignals(False)
        self.combo_group.blockSignals(False)
        extra = (f"；另有 {len(self._group_cols)} 个文本列（{', '.join(self._group_cols)}）"
                 f"可作分组变量" if self._group_cols else "")
        self.lbl_linked.setText(
            f"表格：{n_rows} 行 × {len(names)} 个数值变量；目标变量默认最后一列"
            f"（{names[-1] if names else '—'}）{extra}。"
            f"若同一被试/地点有重复观测，请把「分组变量」也选上（LMM 随机截距）。")

    def _on_atlas_changed(self, *_):
        key = self.combo_atlas.currentData() or "local"
        self._atlas_path, self._spheres_path = "", ""
        if key == "local":
            path, _ = QFileDialog.getOpenFileName(
                self, "选择图集标签图（NIfTI，体素值为整数标签）", str(Path.home()),
                "NIfTI (*.nii *.nii.gz);;所有文件 (*)")
            if not path:
                return
            self._atlas_path = path
            self.lbl_atlas.setText(Path(path).name)
        elif key == "spheres":
            path, _ = QFileDialog.getOpenFileName(
                self, "选择球坐标 ROI 表（x, y, z, radius, name）", str(Path.home()),
                "CSV (*.csv *.txt);;所有文件 (*)")
            if not path:
                return
            self._spheres_path = path
            self.lbl_atlas.setText(Path(path).name)
        else:
            self.lbl_atlas.setText(f"内置图集 {key}（首次使用会联网下载）")

    def choose_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 NIfTI 影像", str(Path.home()),
            "NIfTI (*.nii *.nii.gz);;所有文件 (*)")
        if not path:
            return
        self._img_path = path
        self.rb_image.setChecked(True)
        try:
            info = describe_image(path)
            text = (f"{Path(path).name}　{info['shape']}　"
                    + ("4D BOLD" if info["is_4d"] else "3D 影像")
                    + (f"　TR={info['tr']:g}s" if info.get("tr") else ""))
            self.lbl_img.setText(text)
            if info.get("tr"):
                self.spin_tr.setValue(float(info["tr"]))
        except Exception as e:  # noqa: BLE001
            self.lbl_img.setText(f"{Path(path).name}（读取失败：{e}）")

    def extract_image(self):
        """影像 + 图集 → ROI 时序/数值，并自动选好后续分析类别。"""
        if not self._img_path:
            QMessageBox.information(self, "提示", "请先选择 NIfTI 影像文件")
            return
        if not self._atlas_path and not self._spheres_path:
            QMessageBox.information(self, "提示", "请选择图集（本地标签图 / 内置图集 / 球坐标表）")
            return
        kw = dict(atlas=self._atlas_path or None,
                  spheres_csv=self._spheres_path or None,
                  t_r=float(self.spin_tr.value()),
                  smoothing_fwhm=float(self.spin_fwhm.value()) or None,
                  standardize=self.chk_standardize.isChecked(),
                  detrend=self.chk_detrend.isChecked())
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            data, labels, meta = auto_extract(self._img_path, **kw)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "提取失败", f"{type(e).__name__}: {e}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._labels = list(labels)
        self._ts = np.atleast_2d(np.asarray(data, dtype=float))
        if self._ts.shape[0] != len(self._labels) and self._ts.shape[1] == len(self._labels):
            self._ts = self._ts.T
        tr = float(meta.get("t_r") or self.spin_tr.value() or 2.0)
        self.spin_tr.setValue(tr)
        is_ts = int(meta.get("n_time", 1)) > 1
        # 时序 → 直接做连接；单值（ASL/PET/T1 均值）→ 数值相似性
        self.combo_kind.setCurrentIndex(1 if is_ts else 0)
        if not is_ts:
            idx = [self.combo_sim.itemData(i) for i in range(self.combo_sim.count())].index("mind")
            self.combo_sim.setCurrentIndex(idx)
        self.lbl_sum.setText(
            f"已提取：{meta['atlas']}　{meta['n_roi']} 个 ROI"
            + (f" × {meta['n_time']} 时间点（TR={tr:g}s）" if is_ts else "（单值 / 每 ROI 一个数）")
            + f"\n影像：{Path(self._img_path).name}"
            + ("　→ 已切到「时间序列连接」，点「计算网络」" if is_ts
               else "　→ 已切到「相似性/协变 · MIND」，点「计算网络」"))
        self._draw_preview()

    def _draw_preview(self):
        """提取结果预览：时序画曲线，单值画柱状图。"""
        from ..theme import plot_palette

        if self._ts is None:
            return
        colors = plot_palette()
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        if self._ts.shape[1] > 1:
            show = min(self._ts.shape[0], 20)
            for i in range(show):
                ax.plot(self._ts[i], lw=0.9, color=colors[i % len(colors)], alpha=0.85,
                        label=str(self._labels[i]) if show <= 8 else None)
            ax.set_xlabel("时间点")
            ax.set_ylabel("ROI 信号（已按 masker 参数处理）")
            ax.set_title(f"ROI 时间序列（{self._ts.shape[0]} 个 ROI，显示前 {show} 条）")
            if show <= 8:
                ax.legend(fontsize=8, ncol=2)
        else:
            v = self._ts[:, 0]
            ax.bar(np.arange(len(v)), v, color=colors[0])
            ax.set_xticks(np.arange(len(v)), [str(x) for x in self._labels],
                          rotation=60, fontsize=7)
            ax.set_ylabel("ROI 均值")
            ax.set_title("ROI 数值（3D 影像：CBF / SUV / 体积…）")
        ax.grid(alpha=0.3)
        self.canvas.draw()

    # ================================================================ 计算
    def compute(self):
        method = self.combo_sim.currentData() or "pearson"
        kind = self.combo_kind.currentData() or "sim"
        if kind == "linked":
            self._compute_linked()
            return
        try:
            if kind == "conn":
                ts = self._ts
                if ts is None:
                    QMessageBox.information(self, "提示", "请先提取 ROI 时序（影像），"
                                                          "或打开一个 ROI×时间 的数值表")
                    return
                tr = float(self.spin_tr.value()) or 2.0
                A = np.asarray(connectivity_matrix(ts[None, :, :], 1.0 / tr, method=method))
                A = A.mean(axis=0) if A.ndim == 3 else A
                self._matrix = A
                self._draw(A, f"{CONNECT_LABELS.get(method, method)}　"
                              f"（{ts.shape[0]} ROI × {ts.shape[1]} 时间点，TR={tr:g}s）")
                return
            if self.rb_file.isChecked():
                if self._matrix is None:
                    QMessageBox.information(self, "提示", "请先打开一个矩阵文件")
                    return
                A = np.asarray(self._matrix, dtype=float)
                note = (f"外部矩阵 {A.shape[0]}×{A.shape[1]}"
                        + (f"（标签：{', '.join(self._labels[:5])}…）" if self._labels else ""))
            else:
                table = self._table
                row_kind = self.combo_rows.currentData() or "subjects"
                if self.rb_image.isChecked() and self._ts is not None:
                    if self._ts.shape[1] == 1:            # 3D 影像：每 ROI 一个数值
                        table, row_kind = self._ts, "roi"
                    else:
                        QMessageBox.information(
                            self, "提示", "这是 ROI×时间 的时序数据，"
                                          "请把「分析类别」切到「时间序列连接」")
                        return
                if table is None:
                    QMessageBox.information(self, "提示", "请先打开一个 ROI 数值表"
                                                          "或提取影像 ROI")
                    return
                if row_kind == "subjects":
                    A = group_covariance(table, method=method)
                    note = (f"组水平结构协变（{table.shape[0]} 名被试 × "
                            f"{table.shape[1]} 个 ROI，{SIMILARITY_LABELS[method]}）")
                else:
                    weights = None
                    if self.edit_weights.text().strip():
                        try:
                            weights = [float(x) for x in
                                       self.edit_weights.text().replace("，", ",").split(",")
                                       if x.strip()]
                        except ValueError:
                            weights = None
                    A = individual_similarity(table, method=method,
                                              n_metrics=int(self.spin_metrics.value()),
                                              weights=weights)
                    note = (f"个体水平相似性（{table.shape[0]} 个 ROI，"
                            f"{SIMILARITY_LABELS[method]}"
                            + (f"，{self.spin_metrics.value()} 指标联合" if method == "mind"
                               else "") + "）")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "计算失败", f"{type(e).__name__}: {e}")
            return
        self._matrix = A
        self._draw(A, note)

    # ======================================================== 变量关联分析
    def _compute_linked(self):
        """RF 重要性 + Pearson 相关 + LMM/OLS → Nature 风格三联图。"""
        table = self._table
        names = list(self._col_names)
        if table is None or len(names) != np.asarray(table).shape[1]:
            if self._ts is not None and np.asarray(self._ts).shape[1] == 1:
                table, names = self._ts, [str(x) for x in self._labels]
            else:
                QMessageBox.information(
                    self, "提示",
                    "请先打开一个变量表（CSV / Excel；第一行是变量名），\n"
                    "或先用影像 + 图集提取出「每 ROI 一个数值」的表")
                return
        table = np.asarray(table, dtype=float)
        response = self.combo_response.currentText().strip()
        if not response:
            QMessageBox.information(self, "提示", "请选择目标变量")
            return
        group_raw = str(self.combo_group.currentData() or "")
        groups, group_col = None, ""
        if group_raw.startswith("!"):              # 文本列（被试编号 / 分组标签）
            group_col = group_raw[1:]
            groups = self._group_cols.get(group_col)
        elif group_raw:                            # 数值列也能当分组（如编号 1..9）
            group_col = group_raw
            groups = table[:, names.index(group_col)]
        predictors = [c for c in names if c not in (response, group_col)]

        self.txt.clear()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            res = linked_analysis(
                table, response, names, predictors=predictors, groups=groups,
                n_estimators=int(self.spin_trees.value()),
                method=self.combo_effect.currentData() or "lmm",
                standardize=self.chk_linked_std.isChecked(),
                log_transform=self.chk_linked_log.isChecked(),
                log=lambda s: (self.txt.appendPlainText(str(s)),
                               QApplication.processEvents()))
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "分析失败", f"{type(e).__name__}: {e}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._linked = res
        # 这张图天生很宽（宽度 ≈ 变量数 × 0.62 英寸 + 5.5），把显示区放大才看得清
        try:
            self.resize(max(self.width(), 1560), max(self.height(), 980))
            self._splitter.setSizes([400, 1160])
        except Exception:  # noqa: BLE001
            pass
        model = str(res["meta"]["effects"]["model"]).upper()
        plot_linked_heatmap(res["corr"], res["predictors"], res["importance"], res["effects"],
                            response, fig=self.fig,
                            title=f"{response} · RF 重要性 + Pearson + {model}",
                            annotate_links=self.chk_linked_annot.isChecked(),
                            annotate_top_corr=self.chk_linked_topcorr.isChecked(),
                            show_importance_values=self.chk_linked_imp.isChecked())
        self.canvas.draw()

        # 右侧表：效应 + 重要性一览
        imp = {r["variable"]: r for r in res["importance"]}
        idx = {c: i for i, c in enumerate(res["predictors"])}
        rows = []
        for r in res["effects"]:
            v = r["variable"]
            rmax = max((abs(float(res["corr"][idx[v], idx[o]]))
                        for o in res["predictors"] if o != v), default=0.0)
            rows.append([v, f"{r['beta']:+.3f}", f"{r['p_value']:.3g}",
                         "是" if r["p_value"] < 0.05 else "",
                         f"{imp.get(v, {}).get('importance', 0.0):.4f}", f"{rmax:.2f}"])
        self.tbl.clear()
        self.tbl.setColumnCount(6)
        self.tbl.setHorizontalHeaderLabels(["变量", "β(标准化)", "p", "显著",
                                            "RF 重要性", "max|r|"])
        self.tbl.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                self.tbl.setItem(i, j, QTableWidgetItem(str(val)))
        self.tbl.resizeColumnsToContents()

        meta = res["meta"]
        sig_txt = ", ".join(f"{r['variable']}(β={r['beta']:+.2f})"
                            for r in res["effects"] if r["p_value"] < 0.05) or "无"
        summary = (f"目标变量：{response}　效应模型：{model}"
                   + (f"（{meta['effects']['n_groups']} 组随机截距）"
                      if meta["effects"]["model"] == "lmm" else "（未考虑分组）")
                   + f"\n样本 {meta['prepare']['n_obs']} 行 × {len(res['predictors'])} 个自变量；"
                     f"RF R²(训练)={meta['rf']['r2_train']:.3f}、R²(留出)={meta['rf']['r2_test']:.3f}"
                   + f"\n显著（p<0.05）：{sig_txt}"
                   + "\n读图：顶部看重要性、中间看相关、底部看模型效应；"
                     "相关不等于因果，RF 重要性本身不含方向")
        self.txt.setPlainText((self.txt.toPlainText() + "\n" + summary).strip())
        self.lbl_sum.setText(summary)

    # ================================================================ 绘图
    def _draw(self, M: np.ndarray, note: str):
        from ..theme import theme_cmap

        M = np.asarray(M, dtype=float)
        C = M.shape[0]
        strategy = self.combo_sparsify.currentData() or "proportional"
        W = sparsify(M, strategy=strategy, density=self.spin_density.value(),
                     threshold=(self.spin_threshold.value()
                                if strategy == "absolute"
                                and abs(self.spin_threshold.value()) > 1e-12 else None),
                     percentile=float(self.spin_density.value()) * 100)
        B = (np.abs(W) > 0).astype(int)

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        im = ax.imshow(W, cmap=theme_cmap(), aspect="auto")
        self.fig.colorbar(im, ax=ax, shrink=0.8)
        show_lab = C <= 40
        names = (self._labels[:C] if len(self._labels) >= C
                 else [f"R{i + 1}" for i in range(C)])
        if show_lab:
            ax.set_xticks(range(C), names, rotation=90, fontsize=7)
            ax.set_yticks(range(C), names, fontsize=7)
        ax.set_title(f"稀疏化后网络（{strategy}，保留 {int(B.sum() // 2)} 条边）")
        self.canvas.draw()

        gm = graph_metrics(B) if C > 1 else None
        tot_e = C * (C - 1) / 2
        keep = float(B.sum() // 2) / max(tot_e, 1)
        lines = [note, f"节点数 {C}　保留边 {int(B.sum() // 2)}/{int(tot_e)}"
                       f"（密度 {keep * 100:.1f}%）"]
        if gm is not None:
            lines += [f"全局效率 {gm['global_efficiency']:.3f}　"
                      f"平均最短路径 {gm['avg_path_length']:.3f}　"
                      f"平均聚类系数 {gm['clustering'].mean():.3f}"]
            strength = np.abs(W).sum(axis=1)
            self.tbl.setRowCount(C)
            for i in range(C):
                for j, v in enumerate([names[i], f"{gm['degree'][i]:.0f}",
                                       f"{gm['clustering'][i]:.3f}",
                                       f"{gm['local_efficiency'][i]:.3f}",
                                       f"{strength[i]:.4f}"]):
                    self.tbl.setItem(i, j, QTableWidgetItem(str(v)))
            self.tbl.resizeColumnsToContents()
        self.txt.setPlainText("\n".join(lines))
        self.lbl_sum.setText("\n".join(lines))

    # ================================================================ 导出
    def export(self):
        if self.combo_kind.currentData() == "linked":
            self._export_linked()
            return
        if self._matrix is None:
            QMessageBox.information(self, "提示", "还没有可导出的矩阵")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出矩阵", "network.csv",
                                              "CSV (*.csv);;NumPy (*.npy)")
        if not path:
            return
        C = self._matrix.shape[0]
        labels = self._labels[:C] if len(self._labels) >= C else [f"R{i + 1}" for i in range(C)]
        save_matrix_file(path, self._matrix, labels=labels)
        QMessageBox.information(self, "完成", f"已导出 {path}")

    def _export_linked(self):
        """导出三张结果表（重要性 / 相关 / 效应），可直接进论文附录。"""
        import csv

        if self._linked is None:
            QMessageBox.information(self, "提示", "请先点「计算」得到变量关联结果")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出结果表（会自动生成 3 个 CSV）",
                                              "linked_results.csv", "CSV (*.csv)")
        if not path:
            return
        stem = str(Path(path).with_suffix(""))
        written = []
        for name, (head, rows) in export_tables(self._linked).items():
            p = f"{stem}_{name}"
            with open(p, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(head)
                w.writerows(rows)
            written.append(Path(p).name)
        QMessageBox.information(self, "完成", "已导出：\n" + "\n".join(written))

    def export_fig(self):
        if self._matrix is None and self._linked is None:
            QMessageBox.information(self, "提示", "还没有可导出的图")
            return
        default = "linked_heatmap.png" if self.combo_kind.currentData() == "linked" else "network.png"
        path, _ = QFileDialog.getSaveFileName(self, "导出图片", default,
                                              "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            self.fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            QMessageBox.information(self, "完成", f"已导出 {path}")
