"""划分方案介绍：把每种协议在真实数据上的 train / val / test 切分画出来。

关键点：颜色**直接来自 `core.train.splits.make_splits` 的真实输出**，所以图与
训练时实际使用的划分完全一致（不会出现"画的和做的不一样"）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.patches import Patch, Rectangle
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QDialog, QFileDialog, QHBoxLayout, QLabel,
                             QListWidget, QListWidgetItem, QMessageBox,
                             QPushButton, QSpinBox, QTextBrowser, QVBoxLayout,
                             QWidget)

from .plotting import new_canvas

_COLORS = {"train": "#4c78a8", "val": "#9ecae9", "test": "#d62728", "unused": "#eceff1"}
_LEGEND = [("训练 train", _COLORS["train"]), ("验证 val", _COLORS["val"]),
           ("测试 test", _COLORS["test"]), ("未使用", _COLORS["unused"])]

# (协议名, 中文名, 一句话说明, 额外参数, 是否可绘制)
_SCHEMES: list[tuple[str, str, str, dict, bool]] = [
    ("session", "跨会话 T→E",
     "同一被试的 T session 训练、E session 测试。折数 = 被试数（每被试一折）。\n"
     "回答：换个时间还认不认得（标准 BCI 基准）。", {}, True),
    ("within", "被试内随机划分",
     "每名被试内部按比例随机切分（默认测试占 1/k）；设置 train_percentage 则按"
     "「每被试每类」比例抽训练样本（对齐 EEG-FM-Benchmark）。折数 = 被试数 × n_repeats。\n"
     "回答：被试内随机划分的真实水平。", {}, True),
    ("kfold", "逐被试 k 折",
     "每名被试内部切 k 折轮流当测试，跨被试聚合（不同被试绝不混切）。折数 = 被试数 × k。\n"
     "回答：专门为这个人训练，性能上限是多少。", {}, True),
    ("few_shot", "少样本校准",
     "每被试每类只取 n_shot 个 trial 训练，其余全部测试。折数 = 被试数 × n_repeats。\n"
     "回答：现场校准需要多少数据。", {"val_frac": 0.0}, True),
    ("block", "按记录块划分",
     "每名被试内部按记录块（run）分组，每次留出一整组做测试 —— 同一时间段内的相邻"
     "trial 绝不跨训练/测试。需要预处理输出含 block.npy。", {}, True),
    ("loso", "留一被试",
     "每次留出 1 名被试（其全部数据）做测试，其余被试的全部数据训练。折数 = 被试数。\n"
     "回答：换个人还能不能用（跨被试泛化，最难）。", {}, True),
    ("subject_kfold", "跨被试分组 k 折",
     "把被试分成 k 组，每次留出一整组被试做测试。折数 = k。\n"
     "回答：跨被试泛化的加速版（LOSO 的 k 折形式）。", {}, True),
    ("session_kfold", "同人跨会话双向",
     "同一被试的 T→E 与 E→T 两个方向各算一折。折数 = 被试数 × 2。\n"
     "回答：跨会话稳定性，排除方向偏好。", {}, True),
    ("custom", "自定义折",
     "每一折手工指定 训练 / 验证 / 测试 的受试者；验证留空时自动从该折训练集划出。\n"
     "回答：复现论文的特定分组或做对照实验。",
     {"custom_folds": [{"name": "fold1", "train": [2, 3, 4, 5, 6, 7, 8, 9], "val": [], "test": [1]},
                       {"name": "fold2", "train": [1, 3, 4, 5, 6, 7, 8, 9], "val": [], "test": [2]}]}, True),
    ("holdout", "全体随机划分",
     "所有被试的数据混在一起随机切一刀（测试占 1/k）。只有 1 折，结果波动大。\n"
     "回答：快速调试流程是否跑通。", {}, True),
    ("nested_cv", "嵌套交叉验证",
     "外层折与 kfold 相同；差别在于**每个训练折内部**再切 inner_k 折挑选超参，"
     "超参不接触测试折 → 指标无偏。", {}, True),
    ("cross_dataset", "跨数据集迁移",
     "源域（一个预处理目录）训练 → 目标域（另一个目录）测试，标签/被试号自动对齐。\n"
     "需要两份数据，此处不做图示（见 configs/train_cross_dataset.yaml）。", {}, False),
]

_SCHEME_PARAMS = {
    "few_shot": {"n_shot": 10, "n_repeats": 1},
    "within": {"n_repeats": 1},
    "kfold": {"k": 5},
    "block": {"k": 5},
    "subject_kfold": {"k": 5},
    "holdout": {"k": 5},
    "nested_cv": {"k": 5},
}


def load_split_data(processed_dir: str):
    """读取 sid / sess / y；失败返回 None。"""
    d = Path(processed_dir)
    try:
        return (np.load(d / "sid.npy"), np.load(d / "sess.npy"), np.load(d / "y.npy"),
                np.load(d / "block.npy") if (d / "block.npy").exists() else None)
    except Exception:  # noqa: BLE001
        return None


def synthetic_data(n_sub: int = 9, per: int = 288):
    """无数据时的示意图：n_sub 名被试 × 2 个 session，4 类均衡。"""
    sid = np.repeat(np.arange(1, n_sub + 1), per * 2)
    sess = np.tile(np.array(["T"] * per + ["E"] * per), n_sub)
    y = np.tile(np.repeat([1, 2, 3, 4], per // 4), n_sub * 2)
    return sid, sess, y, None


def draw_split(ax, scheme: str, sid, sess, y, block=None, fold: int = 0) -> str:
    """把某方案第 fold 折的切分画到 ax 上，返回标题文字。"""
    from core.train.splits import make_splits

    extra = next((kw for name, _cn, _d, kw, _dr in _SCHEMES if name == scheme), {})
    params = {**_SCHEME_PARAMS.get(scheme, {}), **extra}
    splits = make_splits(y, sid, sess, scheme=scheme, seed=42, block=block, **params)
    fold = max(0, min(int(fold), len(splits) - 1))
    sp = splits[fold]

    role = np.full(len(y), 3, dtype=int)      # 3 = 未参与
    role[sp.train] = 0
    role[sp.val] = 1
    role[sp.test] = 2

    subjects = sorted({int(v) for v in sid})
    sessions = sorted({str(v) for v in sess})
    ax.clear()
    for i, s in enumerate(subjects):
        for j, se in enumerate(sessions):
            m = (np.asarray(sid) == s) & (np.asarray(sess) == se)
            n = int(m.sum())
            if n == 0:
                continue
            counts = [int(((role == k) & m).sum()) for k in range(4)]
            x0 = float(j)
            for k, c in enumerate(counts):
                if c == 0:
                    continue
                w = c / n
                key = ("train", "val", "test", "unused")[k]
                ax.add_patch(Rectangle((x0, i + 0.1), w, 0.8,
                                       facecolor=_COLORS[key], edgecolor="white", lw=0.6))
                x0 += w
            ax.text(j + 0.5, i + 0.5, str(n), ha="center", va="center",
                    fontsize=7.5, color="#263238")
    ax.set_xlim(0, len(sessions))
    ax.set_ylim(len(subjects), 0)
    ax.set_xticks(np.arange(len(sessions)) + 0.5, sessions, fontsize=9)
    ax.set_yticks(np.arange(len(subjects)) + 0.5, [f"S{s:02d}" for s in subjects], fontsize=8)
    ax.set_xlabel("session")
    ax.set_ylabel("被试")
    ax.set_title(f"{scheme} — 第 {fold + 1}/{len(splits)} 折　"
                 f"（train {len(sp.train)} / val {len(sp.val)} / test {len(sp.test)}）",
                 fontsize=10)
    ax.legend(handles=[Patch(facecolor=c, label=l) for l, c in _LEGEND],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8,
              frameon=False)
    return f"{scheme}：共 {len(splits)} 折"


class SplitIntroDialog(QDialog):
    """划分方案介绍：左选方案，右看该方案在数据上的真实切分。"""

    def __init__(self, processed_dir: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("划分方案介绍 —— 看每种协议如何切分数据")
        self.resize(1080, 660)

        data = load_split_data(processed_dir) if processed_dir else None
        if data is None:
            data = synthetic_data()
            self._data_src = "（未找到预处理数据，使用 9 被试 × T/E 的示意图）"
        else:
            self._data_src = f"数据来源：{processed_dir}"
        self.sid, self.sess, self.y, self.block = data

        # 左：方案列表
        self.list_schemes = QListWidget()
        self.list_schemes.setFixedWidth(230)
        for name, cn, _desc, _kw, drawable in _SCHEMES:
            it = QListWidgetItem(f"{name}\n{cn}")
            if not drawable:
                it.setForeground(Qt.GlobalColor.gray)
            self.list_schemes.addItem(it)

        # 右上：图；右下：说明 + 折号
        self.fig, self.canvas = new_canvas(8.2, 5.4)
        self.ax = self.fig.add_subplot(111)

        self.lbl_fold = QLabel("折序号")
        self.spin_fold = QSpinBox()
        self.spin_fold.setRange(1, 1)
        self.spin_fold.valueChanged.connect(lambda _: self._render())
        row = QHBoxLayout()
        row.addWidget(self.lbl_fold)
        row.addWidget(self.spin_fold)
        row.addStretch(1)

        self.txt = QTextBrowser()
        self.txt.setMaximumHeight(150)
        self.lbl_src = QLabel(self._data_src)
        self.lbl_src.setStyleSheet("color:#888; font-size:11px;")

        btn_png = QPushButton("导出当前示意图")
        btn_close = QPushButton("关闭")
        btn_png.clicked.connect(self._export)
        btn_close.clicked.connect(self.accept)
        btnrow = QHBoxLayout()
        btnrow.addWidget(btn_png)
        btnrow.addStretch(1)
        btnrow.addWidget(btn_close)

        right = QVBoxLayout()
        right.addWidget(self.canvas, 1)
        right.addLayout(row)
        right.addWidget(self.txt)
        right.addWidget(self.lbl_src)
        right.addLayout(btnrow)

        lay = QHBoxLayout(self)
        lay.addWidget(self.list_schemes)
        lay.addLayout(right, 1)

        self.list_schemes.currentRowChanged.connect(self._on_scheme_changed)
        self.list_schemes.setCurrentRow(0)

    # ------------------------------------------------------------------
    def _on_scheme_changed(self, row: int):
        if row < 0:
            return
        self._render()

    def _render(self):
        row = max(self.list_schemes.currentRow(), 0)
        name, cn, desc, _kw, drawable = _SCHEMES[row]
        self.txt.setPlainText(f"【{name}】{cn}\n\n{desc}")
        if not drawable:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "该方案需要两份数据集，暂无示意图\n"
                                   "参见 configs/train_cross_dataset.yaml",
                         ha="center", va="center", fontsize=11, color="#888")
            self.ax.set_axis_off()
            self.canvas.draw()
            return
        try:
            self.ax.set_axis_on()
            info = draw_split(self.ax, name, self.sid, self.sess, self.y,
                              block=self.block, fold=self.spin_fold.value() - 1)
        except Exception as e:  # noqa: BLE001
            self.ax.clear()
            self.ax.set_axis_off()
            self.ax.text(0.5, 0.5, f"{type(e).__name__}: {e}",
                         ha="center", va="center", fontsize=10, color="#c8102e",
                         wrap=True)
            self.canvas.draw()
            return
        # 更新折号范围（不触发递归）
        n_folds = int(info.split("共 ")[1].split(" 折")[0])
        self.spin_fold.blockSignals(True)
        self.spin_fold.setRange(1, max(n_folds, 1))
        self.spin_fold.setSuffix(f" / {n_folds}")
        self.spin_fold.blockSignals(False)
        # 注意：画布由 new_canvas(constrained_layout=True) 创建，这里**不能**再调
        # tight_layout()（两种 layout 引擎冲突会刷 "figure layout has changed to tight" 警告）
        self.canvas.draw()

    def _export(self):
        row = max(self.list_schemes.currentRow(), 0)
        name = _SCHEMES[row][0]
        path, _ = QFileDialog.getSaveFileName(self, "导出示意图",
                                              f"split_{name}.png", "PNG (*.png)")
        if path:
            self.fig.savefig(path, dpi=170, bbox_inches="tight")
            QMessageBox.information(self, "完成", f"已导出 {path}")
