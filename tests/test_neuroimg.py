"""可选 NIfTI + 图集提取：影像 → ROI 时序 / ROI 数值 → 网络。

影像全部用程序合成（不需要下载任何数据）：
- 标签图：4 个立方体 ROI（1..4）+ 侧车 ``*_labels.txt`` 名称
- 4D BOLD：ROI1 与 ROI2 共享时间序列，ROI3/4 独立 → 提取后相关矩阵必须体现出来
- 3D CBF：4 个 ROI 的均值是 40 / 44 / 25 / 12 → 提取值必须对得上

同时覆盖「没装 nibabel+nilearn 时优雅降级」的路径（打桩 HAVE_NIFTI=False）。
"""
from __future__ import annotations

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("nilearn")

from core.neuroimg import (ATLAS_PRESETS, HAVE_NIFTI, INSTALL_HINT, auto_extract,  # noqa: E402
                           describe_image, extract_roi_timeseries, fetch_atlas,
                           load_spheres_csv, require_nifti, save_roi_table)
from core.networks import connectivity_matrix  # noqa: E402

LABELS = ["额叶", "顶叶", "枕叶", "颞叶"]
BOXES = {1: (slice(2, 8), slice(2, 8)), 2: (slice(12, 18), slice(2, 8)),
         3: (slice(2, 8), slice(12, 18)), 4: (slice(12, 18), slice(12, 18))}
CBF_VALUES = {1: 40.0, 2: 44.0, 3: 25.0, 4: 12.0}


@pytest.fixture(scope="module")
def imgs(tmp_path_factory):
    """合成：图集 + 4D BOLD + 3D CBF（外加侧车标签）。"""
    d = tmp_path_factory.mktemp("neuroimg")
    atlas = np.zeros((20, 20, 20), dtype=np.int16)
    for lab, (ys, xs) in BOXES.items():
        atlas[ys, xs, 2:8] = lab
    atlas_path = d / "atlas.nii.gz"
    nib.save(nib.Nifti1Image(atlas, np.eye(4)), atlas_path)
    (d / "atlas_labels.txt").write_text("\n".join(LABELS), encoding="utf-8")

    rng = np.random.default_rng(0)
    T, tr = 60, 2.0
    common = rng.standard_normal(T)
    bold = (np.zeros((20, 20, 20, T), np.float32)
            + rng.standard_normal((20, 20, 20, T)).astype(np.float32) * 0.3)
    series = {1: common, 2: common * 0.95 + 0.3 * rng.standard_normal(T),
              3: rng.standard_normal(T), 4: rng.standard_normal(T)}
    for lab, ts in series.items():
        m = atlas == lab
        bold[m] = ts[None, :] + 0.2 * rng.standard_normal((int(m.sum()), T))
    bold_path = d / "bold.nii.gz"
    img = nib.Nifti1Image(bold, np.eye(4))
    img.header.set_zooms((3.0, 3.0, 3.0, tr))
    nib.save(img, bold_path)

    cbf = np.zeros((20, 20, 20), np.float32)
    for lab, v in CBF_VALUES.items():
        cbf[atlas == lab] = v
    cbf += rng.standard_normal((20, 20, 20)).astype(np.float32) * 0.5
    cbf_path = d / "cbf.nii.gz"
    nib.save(nib.Nifti1Image(cbf, np.eye(4)), cbf_path)

    spheres = d / "spheres.csv"
    spheres.write_text("0,0,0,10,种子A\n20,0,0,8,种子B\n", encoding="utf-8")
    return {"dir": d, "atlas": atlas_path, "bold": bold_path, "cbf": cbf_path,
            "spheres": spheres, "tr": tr, "labels_file": d / "atlas_labels.txt"}


# ------------------------------------------------------------------ 基本信息
def test_optional_deps_available():
    assert HAVE_NIFTI is True
    require_nifti()                                    # 不应抛错
    assert "pip install nibabel nilearn" in INSTALL_HINT


def test_describe_image(imgs):
    info = describe_image(imgs["bold"])
    assert info["is_4d"] and info["ndim"] == 4
    assert info["n_volumes"] == 60 and info["tr"] == pytest.approx(2.0)
    assert info["voxel_mm"] == [3.0, 3.0, 3.0]
    info3 = describe_image(imgs["cbf"])
    assert info3["ndim"] == 3 and info3["is_4d"] is False


def test_describe_missing_file():
    with pytest.raises(FileNotFoundError):
        describe_image("/tmp/definitely_missing_20261007.nii.gz")


def test_atlas_presets_and_unknown():
    assert {"aal", "harvard_oxford", "destrieux"} <= set(ATLAS_PRESETS)
    with pytest.raises(KeyError, match="未知内置图集"):
        fetch_atlas("no_such_atlas")


# ------------------------------------------------------------------ 4D → 时序
def test_extract_timeseries_labels_and_shapes(imgs):
    ts, labels, meta = extract_roi_timeseries(imgs["bold"], atlas=imgs["atlas"],
                                              t_r=imgs["tr"])
    assert ts.shape == (4, 60)
    assert labels == LABELS, f"标签应与侧车文件一一对应，实际 {labels}"
    assert meta["n_roi"] == 4 and meta["n_time"] == 60
    assert meta["t_r"] == pytest.approx(2.0)
    assert "本地标签图" in meta["atlas"]


def test_extracted_coupling_matches_design(imgs):
    """设计的耦合关系必须能被还原，并且能直接接 12 种连接方法。"""
    ts, labels, meta = auto_extract(imgs["bold"], atlas=imgs["atlas"], t_r=imgs["tr"])
    assert meta["mode"] == "timeseries"
    C = np.corrcoef(ts)
    assert C[0, 1] > 0.8, f"ROI1-ROI2 应强相关，实际 {C[0,1]:.3f}"
    assert abs(C[0, 2]) < 0.4, f"ROI1-ROI3 应弱相关，实际 {C[0,2]:.3f}"

    A = np.asarray(connectivity_matrix(ts[None, :, :], 1.0 / meta["t_r"],
                                       method="corr")).mean(axis=0)
    assert A.shape == (4, 4) and A[0, 1] > 0.8


def test_tr_defaults_to_header(imgs):
    _ts, _labels, meta = extract_roi_timeseries(imgs["bold"], atlas=imgs["atlas"])
    assert meta["t_r"] == pytest.approx(2.0)           # 来自影像头 zooms[3]


# ------------------------------------------------------------------ 3D → 数值
def test_extract_values_from_3d(imgs):
    values, labels, meta = auto_extract(imgs["cbf"], atlas=imgs["atlas"])
    assert meta["mode"] == "values" and meta["n_time"] == 1
    assert values.shape == (4,) and labels == LABELS
    for lab in BOXES:
        assert values[lab - 1] == pytest.approx(CBF_VALUES[lab], abs=2.0)


def test_spheres_extraction(imgs):
    ts, labels, meta = extract_roi_timeseries(imgs["bold"], spheres_csv=imgs["spheres"],
                                             t_r=imgs["tr"])
    assert labels == ["种子A", "种子B"]
    assert ts.shape == (2, 60)
    assert "球坐标" in meta["atlas"]


def test_load_spheres_csv_errors(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("name,x,y\nfoo,1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="没有解析到坐标行"):
        load_spheres_csv(bad)


def test_save_roi_table_roundtrip(imgs, tmp_path):
    import csv

    ts, labels, _meta = auto_extract(imgs["bold"], atlas=imgs["atlas"])
    p = save_roi_table(tmp_path / "roi.csv", ts, labels)
    rows = list(csv.reader(p.open(encoding="utf-8")))
    assert rows[0][0] == "ROI" and len(rows) == 5
    assert [r[0] for r in rows[1:]] == LABELS
    assert float(rows[1][1]) == pytest.approx(ts[0, 0], abs=1e-4)


# ------------------------------------------------------------------ 降级路径
def test_missing_deps_graceful(monkeypatch):
    import core.neuroimg as ni

    monkeypatch.setattr(ni, "HAVE_NIFTI", False)
    monkeypatch.setattr(ni, "_IMPORT_ERROR", "ModuleNotFoundError: no module named 'nibabel'")
    with pytest.raises(ImportError, match="pip install nibabel nilearn"):
        ni.require_nifti()
    with pytest.raises(ImportError):
        ni.describe_image("whatever.nii")


def test_dialog_disables_image_when_missing_deps(monkeypatch):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])   # noqa: F841
    import ui.widgets.netmatrix_dialog as nd

    monkeypatch.setattr(nd, "HAVE_NIFTI", False)
    dlg = nd.NetMatrixDialog()
    assert dlg.rb_image.isEnabled() is False
    assert dlg.box_img.isEnabled() is False
    assert "nibabel" in dlg.lbl_img.text()


# ------------------------------------------------------------------ 界面链路
def test_dialog_image_to_network(imgs):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])   # noqa: F841
    from ui.widgets.netmatrix_dialog import NetMatrixDialog

    # 4D BOLD → 时序 → 连接矩阵
    dlg = NetMatrixDialog()
    dlg._img_path = str(imgs["bold"])
    dlg._atlas_path = str(imgs["atlas"])
    dlg.rb_image.setChecked(True)
    dlg.extract_image()
    assert dlg._ts.shape == (4, 60)
    assert dlg.combo_kind.currentData() == "conn", "4D 提取后应自动切到时间序列连接"
    dlg.compute()
    assert dlg._matrix.shape == (4, 4)
    assert dlg._matrix[0, 1] > 0.8
    assert "时间点" in dlg.txt.toPlainText()
    assert dlg.tbl.rowCount() == 4

    # 3D CBF → 数值 → MIND 相似性
    dlg2 = NetMatrixDialog()
    dlg2._img_path = str(imgs["cbf"])
    dlg2._atlas_path = str(imgs["atlas"])
    dlg2.rb_image.setChecked(True)
    dlg2.extract_image()
    assert dlg2.combo_kind.currentData() == "sim"
    assert dlg2.combo_sim.currentData() == "mind"
    dlg2.compute()
    assert dlg2._matrix.shape == (4, 4)
    assert np.allclose(np.diag(dlg2._matrix), 1.0)
    # 数值接近的 ROI（40 与 44）应比差距大的（40 与 12）更相似
    assert dlg2._matrix[0, 1] > dlg2._matrix[0, 3]
