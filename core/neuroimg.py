"""**可选**的 NIfTI + 图集提取：影像 → ROI 时序 / ROI 数值。

装了 ``nibabel`` + ``nilearn`` 时自动启用；没装则相关入口隐藏，并给出安装提示
（本模块不做硬依赖，import 失败不会影响其它功能）。

覆盖两条链路：

1. **4D BOLD**（``.nii.gz``，多时间点）→ 每个 ROI 的平均时间序列
   → 直接喂给 :mod:`core.networks` 的 12 种连接方法（BOLD 功能连接网络的完整链路）
2. **3D 影像**（ASL-CBF / PET-SUV / T1 灰质体积）→ 每个 ROI 的均值
   → 喂给 :mod:`core.netcov` 的 Pearson / KL / JSD / MIND 相似性网络

图集（atlas）三种来源：

1. **本地标签图**：``.nii/.nii.gz``，体素值是整数标签；可放一个
   ``<同名>_labels.txt`` 提供 ROI 名称
2. **nilearn 内置图集**：``aal`` / ``harvard_oxford`` / ``destrieux``…
   首次使用需要联网下载（会缓存到 ``~/nilearn_data``）
3. **球坐标 ROI 列表**：CSV（``x, y, z, radius, name``）→ 不需要图集文件
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

#: 可选依赖是否可用
try:                                   # pragma: no cover - 取决于环境
    import nibabel  # noqa: F401
    from nilearn.maskers import NiftiLabelsMasker, NiftiSpheresMasker  # noqa: F401

    HAVE_NIFTI = True
    _IMPORT_ERROR = ""
except Exception as e:                 # noqa: BLE001
    HAVE_NIFTI = False
    _IMPORT_ERROR = f"{type(e).__name__}: {e}"

#: 安装提示（GUI 与 CLI 共用）
INSTALL_HINT = ("读取 NIfTI 影像需要可选依赖：\n"
                "    pip install nibabel nilearn\n"
                "装好后重启程序，即可直接从影像 + 图集提取 ROI 时序/数值。")

#: nilearn 内置图集：键 → (取用函数名, 显示名, 说明)
ATLAS_PRESETS: dict[str, tuple[str, str, str]] = {
    "aal": ("fetch_atlas_aal", "AAL（自动解剖标记）",
            "116 个脑区（含小脑），最常用的解剖图集"),
    "harvard_oxford": ("fetch_atlas_harvard_oxford", "Harvard-Oxford（皮层+皮层下）",
                       "48 个皮层区 / 21 个皮层下区，可按需选 cortl/cortr/sub"),
    "destrieux": ("fetch_atlas_destrieux_2009", "Destrieux 2009（精细解剖）",
                  "148 个区，比 AAL 细，适合结构协变研究"),
    "msdl": ("fetch_atlas_msdl", "MSDL（功能网络）",
             "39 个功能网络节点（DMN、感觉运动等），适合静息态功能连接"),
}

#: Harvard-Oxford 的常用子图集
HO_ATLASES = {
    "harvard_oxford_cortl": "cortl",
    "harvard_oxford_cortr": "cortr",
    "harvard_oxford_sub": "sub",
}


def require_nifti() -> None:
    """没装可选依赖时抛出带安装说明的错误。"""
    if not HAVE_NIFTI:
        raise ImportError(f"可选依赖不可用（{_IMPORT_ERROR}）\n{INSTALL_HINT}")


def import_error() -> str:
    return _IMPORT_ERROR


# ================================================================ 影像信息
def describe_image(path) -> dict:
    """读取影像头信息（维度 / 体素大小 / 是否 4D / TR）。"""
    require_nifti()
    import nibabel as nib

    p = Path(str(path))
    if not p.exists():
        raise FileNotFoundError(f"影像文件不存在：{p}")
    img = nib.load(str(p))
    zooms = [float(z) for z in img.header.get_zooms()[:3]]
    shape = tuple(int(s) for s in img.shape)
    tr = None
    if img.ndim == 4:
        try:
            z4 = float(img.header.get_zooms()[3])
            tr = z4 if np.isfinite(z4) and z4 > 0 else None
        except Exception:  # noqa: BLE001
            tr = None
    return {
        "file": str(p),
        "shape": shape,
        "ndim": int(img.ndim),
        "voxel_mm": zooms,
        "is_4d": img.ndim == 4,
        "n_volumes": shape[3] if img.ndim == 4 else 1,
        "tr": tr,
        "dtype": str(img.get_data_dtype()),
    }


# ================================================================ 图集
def fetch_atlas(name: str):
    """取 nilearn 内置图集 → ``(标签图路径, 名称列表, 说明)``。"""
    require_nifti()
    import nilearn.datasets as ds

    key = str(name).lower()
    if key.startswith("harvard_oxford"):
        sub = HO_ATLASES.get(key, "cortl")
        at = ds.fetch_atlas_harvard_oxford(sub)
        return at.maps, list(at.labels), f"Harvard-Oxford（{sub}）"
    if key not in ATLAS_PRESETS:
        raise KeyError(f"未知内置图集：{name}（可选 {sorted(ATLAS_PRESETS)} 或 "
                       f"{sorted(HO_ATLASES)}）")
    fn, display, note = ATLAS_PRESETS[key]
    at = getattr(ds, fn)()
    labels = getattr(at, "labels", None)
    if labels is None and isinstance(at, dict):
        labels = at.get("labels")
    labels = [str(x) for x in (labels or [])]
    maps = getattr(at, "maps", None) or (at.get("maps") if isinstance(at, dict) else None)
    return maps, labels, f"{display}　— {note}"


def _sidecar_labels(atlas_path: Path) -> list[str]:
    for cand in (atlas_path.with_name(atlas_path.name.split(".")[0] + "_labels.txt"),
                 atlas_path.with_name(atlas_path.stem + "_labels.txt")):
        if cand.exists():
            return [x.strip() for x in cand.read_text(encoding="utf-8").splitlines()
                    if x.strip()]
    return []


def _label_names(atlas_path: Path, label_values: list[int],
                 preset_labels: list[str] | None = None) -> list[str]:
    """给每个标签值配名字：优先 sidecar，其次图集自带，最后 ROI<值>。"""
    names = _sidecar_labels(atlas_path)
    if not names and preset_labels:
        names = list(preset_labels)
    if names:
        n, mx = len(names), max(label_values)
        if n == mx:                       # 名称数与最大标签一致 → 1 基映射
            return [names[v - 1] if 0 < v <= n else f"ROI{v}" for v in label_values]
        if n == mx + 1:                   # 含背景 → 0 基映射
            return [names[v] if 0 <= v < n else f"ROI{v}" for v in label_values]
    return [f"ROI{v}" for v in label_values]


def load_spheres_csv(path) -> tuple[list[list[float]], list[float], list[str]]:
    """球坐标 ROI 表：``x, y, z[, radius][, name]``（毫米，MNI 空间）。"""
    import csv

    coords, radii, names = [], [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            row = [c.strip() for c in row if c.strip() != ""]
            if not row or not _is_num(row[0]):
                continue
            xyz = [float(row[i]) for i in range(min(3, len(row)))]
            if len(xyz) < 3:
                continue
            coords.append(xyz)
            r = float(row[3]) if len(row) > 3 and _is_num(row[3]) else 8.0
            radii.append(r)
            names.append(row[4] if len(row) > 4 else f"ROI{len(coords)}")
    if not coords:
        raise ValueError(f"{path} 里没有解析到坐标行（格式：x, y, z, radius, name）")
    return coords, radii, names


def _spheres_to_labels(img_path, coords, radii):
    """把球坐标 ROI 画成标签图（**每个 ROI 可以有自己的半径**）。

    nilearn 的 ``NiftiSpheresMasker`` 只接受单一半径，所以这里按世界坐标自己画：
    对每个球只在其包围盒内计算距离，避免大影像上做全量体素遍历。
    """
    import nibabel as nib

    img = nib.load(str(img_path))
    shape = img.shape[:3]
    aff = img.affine
    inv = np.linalg.inv(aff)
    vox_size = np.abs(aff[:3, :3]).max(axis=0)             # 每个轴的体素毫米数
    out = np.zeros(shape, dtype=np.int16)
    for k, (c, r) in enumerate(zip(coords, radii), start=1):
        cijk = (inv @ np.append(np.asarray(c, dtype=float), 1.0))[:3]
        half = r / np.maximum(vox_size, 1e-6) + 2
        lo = np.maximum(np.floor(cijk - half).astype(int), 0)
        hi = np.minimum(np.ceil(cijk + half).astype(int) + 1, np.asarray(shape))
        if np.any(hi <= lo):
            continue
        axes = [np.arange(lo[d], hi[d]) for d in range(3)]
        ii, jj, kk = np.meshgrid(*axes, indexing="ij")
        ijk = np.stack([ii, jj, kk], axis=-1).reshape(-1, 3)
        xyz = ijk @ aff[:3, :3].T + aff[:3, 3]
        inside = ((xyz - np.asarray(c, dtype=float)) ** 2).sum(axis=1) <= r * r
        sel = ijk[inside]
        if sel.size:
            flat = tuple(sel.T)
            empty = out[flat] == 0
            out[flat] = np.where(empty, k, out[flat])
    return nib.Nifti1Image(out, aff)


def _is_num(s) -> bool:
    try:
        float(str(s))
        return True
    except (TypeError, ValueError):
        return False


# ================================================================ 提取
def extract_roi_timeseries(img_path, atlas=None, spheres_csv=None, t_r=None,
                           mask_path=None, smoothing_fwhm=None,
                           standardize: bool = False, detrend: bool = True,
                           max_roi: int | None = None):
    """影像 + 图集 → **ROI × 时间** 矩阵。

    Parameters
    ----------
    img_path : 4D（BOLD）或 3D 影像路径
    atlas : 本地标签图路径，或内置图集名（``aal`` / ``harvard_oxford_cortl``…）
    spheres_csv : 球坐标 ROI 表（与 atlas 二选一）
    t_r : 重复时间（秒）；不给则用影像头里的第 4 维体素大小
    smoothing_fwhm / standardize / detrend : 传给 nilearn masker 的预处理开关

    Returns
    -------
    (ts, labels, meta) : ``ts`` 形状 (n_roi, n_time)，``meta`` 含影像信息与图集来源。
    """
    require_nifti()
    p = Path(str(img_path))
    info = describe_image(p)

    # nilearn 的开关：3D 影像去漂移无意义；standardize 用新写法避免 0.15 弃用告警
    std = "zscore_sample" if standardize else None
    detrend_ok = bool(detrend and info["is_4d"])

    if spheres_csv:
        coords, radii, names = load_spheres_csv(spheres_csv)
        labels_img = _spheres_to_labels(p, coords, radii)   # 支持每球不同半径
        masker = NiftiLabelsMasker(labels_img=labels_img, t_r=t_r,
                                   mask_img=str(mask_path) if mask_path else None,
                                   smoothing_fwhm=smoothing_fwhm,
                                   standardize=std, detrend=detrend_ok)
        masker.fit()
        raw = masker.transform(str(p))
        labels = names
        atlas_desc = (f"球坐标 ROI ×{len(coords)}"
                      f"（半径 {min(radii):g}~{max(radii):g} mm）")
        empty = np.asarray(raw, dtype=float)
        if empty.ndim == 2 and empty.shape[1] < len(names):
            labels = labels[:empty.shape[1]]
    else:
        if not atlas:
            raise ValueError("请指定图集（本地标签图 / 内置图集名 / 球坐标 CSV）")
        preset_labels = None
        atlas_path, atlas_desc = str(atlas), f"本地标签图 {Path(str(atlas)).name}"
        if not Path(str(atlas)).exists():                 # 当作内置图集名
            atlas_path, preset_labels, atlas_desc = fetch_atlas(str(atlas))
        masker = NiftiLabelsMasker(labels_img=str(atlas_path), t_r=t_r,
                                   mask_img=str(mask_path) if mask_path else None,
                                   smoothing_fwhm=smoothing_fwhm,
                                   standardize=std, detrend=detrend_ok)
        masker.fit()
        raw = masker.transform(str(p))
        raw_arr = np.asarray(raw, dtype=float)
        n_out = raw_arr.shape[1] if raw_arr.ndim == 2 else raw_arr.shape[0]
        vals = [int(v) for v in np.ravel(masker.labels_)]
        # nilearn 的 labels_ 会带上背景 0，而输出里没有背景 → 对齐一下
        if len(vals) == n_out + 1 and vals[0] == 0:
            vals = vals[1:]
        labels = _label_names(Path(str(atlas_path)), vals, preset_labels)

    # 统一成 **(脑区, 特征)**：4D → (ROI, 时间点)；3D → (ROI, 1)
    ts = np.asarray(raw, dtype=float)
    if ts.ndim == 1:                       # 3D 影像：nilearn 直接给每个 ROI 一个值
        ts = ts[:, None]
    elif ts.ndim == 2:
        ts = ts.T                          # (时间点, ROI) → (ROI, 时间点)
    else:
        ts = ts.reshape(ts.shape[1], -1) if ts.shape[1] < ts.shape[2] else \
            ts.reshape(ts.shape[0], -1)
    if max_roi and ts.shape[0] > max_roi:                  # ROI 太多时截断（加速）
        ts, labels = ts[:max_roi], labels[:max_roi]

    meta = {**info, "atlas": atlas_desc, "n_roi": int(ts.shape[0]),
            "n_time": int(ts.shape[1]),
            "t_r": float(t_r or info.get("tr") or 1.0)}
    return ts, labels, meta


def extract_roi_values(img_path, **kw):
    """3D 影像（ASL-CBF / PET-SUV / T1 体积）→ 每个 ROI 的均值向量。"""
    ts, labels, meta = extract_roi_timeseries(img_path, **kw)
    values = np.asarray(ts, dtype=float).mean(axis=1)
    meta["mode"] = "values"
    return values, labels, meta


def auto_extract(img_path, **kw):
    """自动判断：4D → ROI 时序；3D → ROI 数值。返回 ``(数据, 标签, meta)``。"""
    info = describe_image(img_path)
    if info["is_4d"] and info["n_volumes"] > 1:
        ts, labels, meta = extract_roi_timeseries(img_path, **kw)
        meta["mode"] = "timeseries"
        return ts, labels, meta
    values, labels, meta = extract_roi_values(img_path, **kw)
    return values, labels, meta


def save_roi_table(path, data, labels: list[str]) -> Path:
    """把提取结果存成 CSV（行=ROI，列=时间点/单列数值），便于复用。"""
    import csv

    p = Path(str(path))
    M = np.atleast_2d(np.asarray(data, dtype=float))
    if M.shape[0] != len(labels) and M.shape[1] == len(labels):
        M = M.T
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ROI"] + [f"t{i + 1}" for i in range(M.shape[1])])
        for i, row in enumerate(M):
            w.writerow([labels[i] if i < len(labels) else f"ROI{i + 1}"]
                       + [f"{v:.6g}" for v in row])
    return p
