"""非时间序列类脑网络：**结构协变（T1）、分布相似性（ASL/PET）、滞后矩阵、外部矩阵导入**。

我们程序本体是 EEG 平台，但下面这些都只吃「表格 / 矩阵」，所以能顺带覆盖另外几种模态：

- **T1 结构协变**：FreeSurfer / CAT12 导出的「被试 × ROI 灰质体积/皮层厚度」CSV
  → 组水平协变矩阵（被试维度上的相关）
- **ASL-CBF / PET**：ROI 的 CBF / SUV 数值表
  → 个体水平相似性网络（Pearson / Spearman / KL / JSD / MIND 多指标联合）
- **有效连接滞后矩阵**：多通道时序 → 亚采样精度的峰值滞后（抛物线 / 三次样条插值）
- **外部矩阵导入**：其他软件导出的连接矩阵（如 PANDA 的网络矩阵、SPM-DCM 的 A 矩阵）
  （``.mat`` / ``.csv`` / ``.npy``）→ 统一成矩阵 + 标签，接着用图论指标继续分析

**为什么不在平台里重跑这些流程**：计算本身依赖 MATLAB / 专用工具链，跑在我们这里既不
现实也没必要——把它们的**产物**接进来做统一分析与可视化，才是这个平台能提供的增量。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

#: 相似性/协变方法
SIMILARITY_METHODS = ["pearson", "spearman", "kl", "jsd", "mind"]
SIMILARITY_LABELS = {
    "pearson": "Pearson 相关",
    "spearman": "Spearman 秩相关",
    "kl": "KL 散度（对称化 → 相似度）",
    "jsd": "Jensen-Shannon 距离（→ 相似度）",
    "mind": "MIND 多指标联合",
}

#: 滞后矩阵的亚采样插值方式
LAG_INTERP = ["parabolic", "cubic"]
LAG_INTERP_LABELS = {
    "parabolic": "抛物线插值（3 点，快）",
    "cubic": "三次样条插值（更平滑）",
}


# ================================================================ 基础工具
def _check_table(table) -> np.ndarray:
    T = np.asarray(table, dtype=float)
    if T.ndim != 2:
        raise ValueError(f"需要二维表格（行=R OI/被试，列=指标），当前形状 {T.shape}")
    return np.nan_to_num(T, nan=0.0)


def _corr(M: np.ndarray, method: str = "pearson") -> np.ndarray:
    """行向量两两相关（M: (R,F)）→ (R,R)，对角=1。"""
    if method == "spearman":
        from scipy.stats import rankdata

        M = np.apply_along_axis(rankdata, 1, M)
    R = M.shape[0]
    out = np.ones((R, R))
    for a in range(R):
        for b in range(a + 1, R):
            va, vb = M[a], M[b]
            if np.std(va) < 1e-12 or np.std(vb) < 1e-12:
                v = 0.0
            else:
                v = float(np.corrcoef(va, vb)[0, 1])
            out[a, b] = out[b, a] = 0.0 if not np.isfinite(v) else v
    return out


def _as_distribution(M: np.ndarray, rows: bool = True) -> np.ndarray:
    """把每行（或每列）变成非负、和为 1 的分布（KL/JSD 用）。"""
    V = np.asarray(M, dtype=float)
    if not rows:
        V = V.T
    V = V - V.min(axis=1, keepdims=True)              # 平移成非负
    V = np.clip(V, 0.0, None) + 1e-12
    return V / V.sum(axis=1, keepdims=True)


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p‖q)。"""
    return float(np.sum(np.where(p > 0, p * np.log((p + 1e-12) / (q + 1e-12)), 0.0)))


def _js_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon 距离（以 2 为底的开方 JSD，∈[0,1]）。"""
    m = 0.5 * (p + q)
    return float(np.sqrt(max(0.5 * _kl(p, m) + 0.5 * _kl(q, m), 0.0) / np.log(2.0)))


def _distribution_similarity(M: np.ndarray, method: str = "jsd") -> np.ndarray:
    V = _as_distribution(M)
    R = V.shape[0]
    out = np.ones((R, R))
    for a in range(R):
        for b in range(a + 1, R):
            if method == "jsd":
                v = 1.0 - _js_distance(V[a], V[b])        # 距离 → 相似度
            else:                                          # kl：对称化后转相似度
                d = 0.5 * (_kl(V[a], V[b]) + _kl(V[b], V[a]))
                v = 1.0 / (1.0 + d)
            out[a, b] = out[b, a] = float(np.clip(v, 0.0, 1.0))
    return out


# ================================================================ 相似性网络
def individual_similarity(table, method: str = "pearson", n_metrics: int | None = None,
                          weights=None, metric_names: list[str] | None = None):
    """个体水平相似性网络：**每个 ROI 一行**，行内是它的指标取值。

    Parameters
    ----------
    table : (n_roi, n_features)
        - ``pearson`` / ``spearman``：行内取值直接算相关
        - ``kl`` / ``jsd``：行内取值当**分布**（自动平移成非负并归一化）
        - ``mind``：按 ``n_metrics`` 把列分成若干指标，逐指标算相似性再加权融合
    method : pearson / spearman / kl / jsd / mind
    n_metrics : 仅 mind 用——列要分成几组指标（需能整除列数）
    weights : 仅 mind 用——各指标权重（默认等权）

    Returns
    -------
    ndarray (n_roi, n_roi)：相似性矩阵，对角=1。
    """
    M = _check_table(table)
    method = (method or "pearson").lower()
    if method == "mind":
        return mind_matrix(M, n_metrics=n_metrics or 1, weights=weights)
    if method in ("pearson", "spearman"):
        return _corr(M, method)
    if method in ("kl", "jsd"):
        return _distribution_similarity(M, method)
    raise ValueError(f"未知相似性方法：{method}（可选 {SIMILARITY_METHODS}）")


def _value_similarity(values) -> np.ndarray:
    """单指标数值相似性：``1/(1+|vi−vj|/尺度)``（尺度取差值的中位数，自适应量纲）。"""
    v = np.asarray(values, dtype=float).ravel()
    d = np.abs(v[:, None] - v[None, :])
    scale = float(np.median(d[d > 0])) if (d > 0).any() else 1.0
    sim = 1.0 / (1.0 + d / max(scale, 1e-12))
    np.fill_diagonal(sim, 1.0)
    return sim


def mind_matrix(table, n_metrics: int = 1, weights=None, method: str = "jsd"):
    """MIND（多指标联合）：逐指标算相似性矩阵，再按权重线性融合。

    ``n_metrics`` = 指标个数（列数能被它整除），例如 T1 的「灰质体积 / 皮层厚度 /
    表面曲率」三指标。``method`` 是**每个指标内部**用什么相似性（默认 JSD）。

    特例：只有**一列**时（如单个被试的 ASL-CBF ROI 均值）无法算多指标分布相似性，
    改用「数值越接近越相似」的 :func:`_value_similarity`。
    """
    M = _check_table(table)
    C, F = M.shape
    n_metrics = int(n_metrics or 1)
    if n_metrics <= 1 or F % n_metrics:
        if F == 1:
            return _value_similarity(M[:, 0])
        return individual_similarity(M, method=method)
    cols = np.array_split(np.arange(F), n_metrics)
    w = np.asarray(weights if weights is not None else np.ones(n_metrics), dtype=float)
    w = w / max(w.sum(), 1e-12)
    out = np.zeros((C, C))
    for k, idx in enumerate(cols):
        sub = M[:, idx]
        sim = (_value_similarity(sub[:, 0]) if sub.shape[1] == 1
               else individual_similarity(sub, method=method))
        out += w[k] * sim
    np.fill_diagonal(out, 1.0)
    return (out + out.T) / 2.0


def group_covariance(table, method: str = "pearson"):
    """**组水平结构协变矩阵**：行=被试、列=ROI（灰质体积/皮层厚度…）。

    协变 = 跨被试的「这个 ROI 大的人，那个 ROI 是否也大」。
    返回 (n_roi, n_roi)，对角=1。
    """
    M = _check_table(table)
    if M.shape[0] < 3:
        raise ValueError(f"组水平协变至少需要 3 名被试（当前 {M.shape[0]} 行）")
    return _corr(M.T, method if method in ("pearson", "spearman") else "pearson")


# ================================================================ 滞后矩阵
def lag_matrix(X, sfreq: float, max_lag_ms: float = 120.0, interp: str = "parabolic",
               max_samples: int = 20000):
    """滞后矩阵：通道对之间互相关的**峰值滞后**（含亚采样插值）。

    ``L[i,j]`` = j 相对 i 的时间延迟（ms，正 = j 落后于 i）；反对称，对角=0。
    亚采样插值把"整数采样点"的峰值细化到采样间隔以下：

    - ``parabolic``：峰值三点抛物线拟合（快，工程常用）
    - ``cubic``    ：峰值邻域三次样条（更平滑，略慢）

    X (N,C,T) 或 (C,T) → (C,C)。
    """
    M = _as_channels(X)
    C, T = M.shape
    if T < 16:
        raise ValueError("数据太短，无法估计滞后（至少 16 个采样点）")
    max_lag = max(int(round(max_lag_ms / 1000.0 * sfreq)), 1)
    max_lag = min(max_lag, max(T // 4, 1))
    M = M - M.mean(axis=1, keepdims=True)
    sf = max_samples / T
    L = np.zeros((C, C))
    for i in range(C):
        xi = M[i]
        for j in range(i + 1, C):
            xj = M[j]
            # 互相关（FFT）→ 取 ±max_lag 区间
            n = int(2 ** np.ceil(np.log2(2 * T)))
            cc = np.fft.irfft(np.fft.rfft(xi, n) * np.conj(np.fft.rfft(xj, n)), n)
            cc = np.concatenate([cc[-max_lag:], cc[:max_lag + 1]])   # 索引 → 滞后
            k = int(np.argmax(np.abs(cc)))
            lag = k - max_lag
            refined = float(lag)
            if 0 < k < len(cc) - 1:
                y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
                if interp == "cubic":
                    from scipy.interpolate import CubicSpline

                    xs = np.arange(k - 3, k + 4)
                    xs = xs[(xs >= 0) & (xs < len(cc))]
                    try:
                        spl = CubicSpline(xs, cc[xs])
                        grid = np.arange(max(k - 1.5, 0.0), min(k + 1.5, len(cc) - 1), 0.01)
                        idx = int(np.argmax(np.abs(spl(grid))))
                        refined = float(grid[idx] - max_lag)
                    except Exception:  # noqa: BLE001 - 样条失败退回抛物线
                        refined = float(lag + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
                                        if abs(y0 - 2 * y1 + y2) > 1e-12 else lag)
                else:
                    denom = (y0 - 2 * y1 + y2)
                    if abs(denom) > 1e-12:
                        refined = float(lag + 0.5 * (y0 - y2) / denom)
                # 亚采样细化最多把峰值挪动 ±1 个采样点：样条在噪声尖峰附近可能过冲，
                # 这一步保证结果落在物理合理的范围内。
                refined = float(np.clip(refined, lag - 1.0, lag + 1.0))
            L[i, j] = refined / sfreq * 1000.0
            L[j, i] = -L[i, j]
    return L


def _as_channels(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 3:                       # (N,C,T) → 跨 trial 拼接
        N, C, T = X.shape
        return X.transpose(1, 0, 2).reshape(C, N * T)
    if X.ndim == 2:
        return X
    raise ValueError(f"需要 (N,C,T) 或 (C,T)，当前形状 {X.shape}")


# ================================================================ 矩阵读写
def _read_labels(path: Path) -> list[str]:
    for cand in (path.with_name(path.stem + "_labels.txt"),
                 path.with_name(path.stem + "_labels.csv"),
                 path.with_suffix(".labels")):
        if cand.exists():
            txt = cand.read_text(encoding="utf-8").strip().splitlines()
            out = []
            for line in txt:
                out += [x.strip() for x in line.replace("\t", ",").split(",") if x.strip()]
            if out:
                return out
    return []


def load_matrix_file(path) -> tuple[np.ndarray, list[str]]:
    """读取连接矩阵：``.npy`` / ``.csv`` / ``.txt`` / ``.mat``。

    自动处理：带表头的 CSV（第一行/第一列是标签）、``.mat`` 里挑最大的方阵变量。
    返回 ``(矩阵, 标签列表)``（标签可能为空）。
    """
    p = Path(str(path))
    if not p.exists():
        raise FileNotFoundError(f"矩阵文件不存在：{p}")
    suffix = p.suffix.lower()

    if suffix == ".npy":
        M = np.load(p).astype(float)
        return M, _read_labels(p)

    if suffix == ".mat":
        from scipy.io import loadmat

        data = loadmat(p)
        best, name = None, ""
        for k, v in data.items():
            if k.startswith("__") or not isinstance(v, np.ndarray) or v.ndim != 2:
                continue
            if v.shape[0] != v.shape[1] or not np.issubdtype(v.dtype, np.number):
                continue
            if best is None or v.shape[0] > best.shape[0]:
                best, name = v.astype(float), k
        if best is None:
            raise ValueError(f"{p.name} 里没有找到方阵数值变量（已有："
                             f"{[k for k in data if not k.startswith('__')]}）")
        labels = _read_labels(p)
        if not labels:
            try:                                  # 有的 .mat 里带 labels 字段
                for k in ("labels", "channel_names", "names"):
                    if k in data:
                        labels = [str(x[0]) if isinstance(x, np.ndarray) else str(x)
                                  for x in np.ravel(data[k])]
                        break
            except Exception:  # noqa: BLE001
                labels = []
        return best, labels

    # CSV / TXT：嗅探分隔符，识别表头与行标签
    import csv

    text = p.read_text(encoding="utf-8-sig", errors="ignore")
    sample = text.splitlines()[0] if text.strip() else ""
    delim = max([",", ";", "\t", " "], key=sample.count) if sample else ","
    rows = [r for r in csv.reader(text.splitlines(), delimiter=delim) if r]
    labels: list[str] = []
    if rows and not all(_is_num(x) for x in rows[0][1:] or rows[0]):
        header = rows.pop(0)
        # 有关键区分：表头首格是"角落单元格"（后面跟行标签）还是**真正的第一个列名**。
        # 只看首格非空是不够的——纯表头 CSV（无行标签列）会把第一个列名当角落丢掉，
        # 于是所有列名整体错位一位。
        has_label_col = bool(rows) and str(rows[0][0]).strip() != "" and not _is_num(rows[0][0])
        labels = ([str(x).strip() for x in header[1:]] if has_label_col
                  else [str(x).strip() for x in header])
    data_rows, row_labels = [], []
    for r in rows:
        vals = r
        if vals and not _is_num(vals[0]):
            row_labels.append(str(vals[0]).strip())
            vals = vals[1:]
        data_rows.append([float(x) for x in vals if str(x).strip() != ""])
    M = np.asarray(data_rows, dtype=float)
    if not labels and row_labels:
        labels = row_labels
    return M, labels


def _is_num(s) -> bool:
    try:
        float(str(s).strip())
        return True
    except (TypeError, ValueError):
        return False


def load_table_file(path, sheet: str | int | None = None):
    """读取**变量表**（行 = 观测/被试，列 = 变量）。

    Returns
    -------
    ``(M, col_names, row_names, non_numeric)``

    - ``M``：数值列组成的 ``(n, k)`` 矩阵
    - ``col_names``：``M`` 各列的变量名
    - ``row_names``：行名（没有就自动生成 row1..rowN）
    - ``non_numeric``：``{列名: 原始值数组}`` —— 非数值列（如被试编号 ``S01``、
      分组标签 ``A/B/C``、性别 ``M/F``）。它们**不参与回归**，但正是
      LMM 需要的**分组变量**来源，所以单列返回而不是丢掉。

    两种表格布局都支持：① 首格为空的"角落单元格 + 行标签"矩阵排版；
    ② 纯表头（第一行就是全部变量名）。支持
    ``.csv/.txt/.tsv``、``.xlsx/.xls/.xlsm``（需 pandas + openpyxl）、
    ``.npy``、``.mat``。
    """
    import csv

    p = Path(str(path))
    if not p.exists():
        raise FileNotFoundError(f"数据文件不存在：{p}")
    suffix = p.suffix.lower()

    if suffix == ".npy":
        M = np.atleast_2d(np.load(p).astype(float))
        return (M, [f"col{i + 1}" for i in range(M.shape[1])],
                [f"row{i + 1}" for i in range(M.shape[0])], {})

    if suffix == ".mat":
        from scipy.io import loadmat

        data = loadmat(p)
        best, name = None, ""
        for k, v in data.items():
            if k.startswith("__") or not isinstance(v, np.ndarray) or v.ndim != 2:
                continue
            if not np.issubdtype(v.dtype, np.number):
                continue
            if best is None or v.size > best.size:
                best, name = v.astype(float), k
        if best is None:
            raise ValueError(f"{p.name} 里没有找到二维数值变量")
        col_names = []
        for key in ("colnames", "varnames", "labels", "names"):
            if key in data:
                col_names = [str(np.ravel(data[key])[i]) for i in range(len(np.ravel(data[key])))]
                break
        if len(col_names) != best.shape[1]:
            col_names = [f"{name}_{i + 1}" for i in range(best.shape[1])]
        return best, col_names, [f"row{i + 1}" for i in range(best.shape[0])], {}

    # ---- CSV / TXT / Excel：用 pandas 读（能正确识别非数值的分组列）----
    df = None
    if suffix in (".xlsx", ".xls", ".xlsm", ".csv", ".txt", ".tsv"):
        try:
            import pandas as pd
        except ImportError:                            # pragma: no cover - 没有 pandas
            pd = None
        if pd is not None:
            try:
                if suffix in (".xlsx", ".xls", ".xlsm"):
                    df = pd.read_excel(p, sheet_name=sheet if sheet is not None else 0)
                else:
                    df = pd.read_csv(p, sep=None, engine="python")   # 自动嗅探分隔符
            except ImportError as e:                   # 缺 openpyxl
                raise ImportError(
                    "读取 Excel 需要 openpyxl：pip install openpyxl（或另存为 CSV 再打开）") from e
            except Exception:                          # noqa: BLE001 - 退回手工解析
                df = None

    if df is not None and df.shape[1] >= 2:
        cols = [str(c).strip() for c in df.columns]
        row_names: list[str] = []
        # "角落单元格"排版（首格为空）→ 第一列是行标签。
        # 注意 pandas 会把空表头名读成 "Unnamed: 0"，必须一起识别，
        # 否则行标签列会被当作一个"文本变量"混进分组候选里。
        first = cols[0].lower() if cols else ""
        if (df.shape[1] > 2 and (first == "" or first.startswith("unnamed"))
                and not pd.api.types.is_numeric_dtype(df.iloc[:, 0])):
            row_names = [str(v) for v in df.iloc[:, 0]]
            df = df.iloc[:, 1:]
            cols = cols[1:]
        num_cols, num_names, non_numeric = [], [], {}
        for c in df.columns:
            series = df[c]
            if pd.api.types.is_numeric_dtype(series):
                num_cols.append(series.to_numpy(dtype=float))
                num_names.append(str(c).strip())
            else:
                vals = series.to_numpy()
                # 能整体转成数值的（如 "1.2"）仍按数值列处理
                conv = pd.to_numeric(series, errors="coerce")
                if conv.notna().all():
                    num_cols.append(conv.to_numpy(dtype=float))
                    num_names.append(str(c).strip())
                else:
                    non_numeric[str(c).strip()] = vals
        if not num_cols:
            raise ValueError(f"{p.name} 里没有数值列，无法做变量关联分析")
        M = np.column_stack(num_cols)
        return (M, num_names,
                row_names or [f"row{i + 1}" for i in range(M.shape[0])], non_numeric)

    # ---- 没有 pandas 时的退化路径：只解析数值列 ----
    text = p.read_text(encoding="utf-8-sig", errors="ignore")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"{p.name} 是空文件")
    delim = max([",", ";", "\t", " "], key=lines[0].count)
    rows = [r for r in csv.reader(lines, delimiter=delim)
            if any(str(x).strip() for x in r)]
    header: list[str] | None = None
    if any(str(x).strip() != "" and not _is_num(x) for x in rows[0]):
        header = [str(x).strip() for x in rows.pop(0)]
    has_label_col = bool(rows) and str(rows[0][0]).strip() != "" and not _is_num(rows[0][0])
    row_names, data_rows = [], []
    for r in rows:
        vals = r
        if has_label_col:
            row_names.append(str(vals[0]).strip())
            vals = vals[1:]
        data_rows.append([float(x) for x in vals if str(x).strip() != "" and _is_num(x)])
    width = min((len(r) for r in data_rows), default=0)
    if width == 0:
        raise ValueError(f"{p.name} 里没有解析出数值")
    M = np.asarray([r[:width] for r in data_rows], dtype=float)
    col_names = ([f"col{i + 1}" for i in range(width)] if header is None
                 else (header[1:] if has_label_col else header))
    if len(col_names) != width:
        col_names = (col_names + [f"col{i + 1}" for i in range(width)])[:width]
    return M, col_names, row_names or [f"row{i + 1}" for i in range(M.shape[0])], {}


def save_matrix_file(path, matrix, labels: list[str] | None = None) -> Path:
    """保存矩阵：``.npy`` / ``.mat`` / ``.csv``（其余后缀按 CSV 写）。

    - ``.npy``：数组 + 同名 ``_labels.txt``
    - ``.mat``：``scipy.io.savemat``（变量名 ``A`` + ``labels``），
      与 :func:`load_matrix_file` 的 ``.mat`` 分支对称 —— 之前写 ``.mat`` 会**落到
      CSV 分支**、生成一个后缀是 .mat 的文本文件，再用 loadmat 读就会报
      "Unknown mat file type"，往返静默损坏。
    - ``.csv``：带行列标签，Excel 直接可看
    """
    p = Path(str(path))
    M = np.asarray(matrix, dtype=float)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        np.save(p, M)
        if labels:
            p.with_name(p.stem + "_labels.txt").write_text(
                "\n".join(map(str, labels)), encoding="utf-8")
        return p
    if suffix == ".mat":
        from scipy.io import savemat

        names = [str(v) for v in (labels or [f"R{i + 1}" for i in range(M.shape[0])])]
        savemat(str(p), {"A": M, "labels": np.array(names, dtype=object)})
        return p
    import csv

    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        names = list(labels or [f"R{i + 1}" for i in range(M.shape[0])])
        w.writerow([""] + names)
        for i, row in enumerate(M):
            w.writerow([names[i] if i < len(names) else f"R{i + 1}"]
                       + [f"{v:.6g}" for v in row])
    return p


def table_from_matrix(matrix) -> tuple[np.ndarray, np.ndarray]:
    """矩阵 → (行=ROI, 列=各 ROI 的邻接权重) 表格，方便复用相似性/协变分析。"""
    M = np.asarray(matrix, dtype=float)
    return M, np.arange(M.shape[1])
