"""训练设备：**探测本机可用设备 → 解析用户选择 → 真跑一遍验证**。

跨平台支持（配置里 ``trainer.device`` 的写法）::

    auto          自动挑最合适的（CUDA > MPS > XPU > CPU）
    cpu           Windows / Linux / macOS 都能用
    mps           Apple Silicon（Metal 加速）
    cuda          NVIDIA 单卡（= cuda:0）
    cuda:1        指定第 1 号卡
    cuda:0,1      多卡（DataParallel，把 batch 切到多张卡上）
    cuda:0 cuda:1 同上（空格/逗号都认）

主要接口::

    list_devices()        # 本机可用设备（GUI 列表用）
    system_info()         # 系统 / PyTorch / CUDA / cuDNN 版本（GUI 展示用）
    resolve_device(spec)  # "cuda:0,1" → ("cuda:0", [0, 1])
    device_label(spec)    # 人类可读："2× NVIDIA RTX 4090（cuda:0,1 多卡）"
    probe_device(spec)    # 真跑几步矩阵乘 + 反传，返回耗时与错误
    wrap_multi_gpu(...)   # 多卡时包 nn.DataParallel（单卡/CPU/MPS 原样返回）
"""
from __future__ import annotations

import platform
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    """一个可训练设备。"""

    id: str                      # cpu / mps / cuda:0 / xpu:0
    kind: str                    # cpu / cuda / mps / xpu
    name: str                    # 人类可读名称
    index: int | None = None     # 卡号（cuda/xpu）
    memory_gb: float | None = None
    note: str = ""


# ---------------------------------------------------------------- 探测
def _mps_available() -> bool:
    import torch

    mps = getattr(torch.backends, "mps", None)
    try:
        return bool(mps is not None and mps.is_available())
    except Exception:  # noqa: BLE001
        return False


def _cuda_count() -> int:
    import torch

    try:
        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:  # noqa: BLE001
        return 0


def list_devices() -> list[DeviceInfo]:
    """本机可用设备（顺序：CUDA 各卡 → MPS → XPU → CPU 兜底）。"""
    import torch

    out: list[DeviceInfo] = []
    for i in range(_cuda_count()):
        name, mem = f"NVIDIA GPU {i}", None
        try:
            props = torch.cuda.get_device_properties(i)
            name = getattr(props, "name", name)
            total = getattr(props, "total_memory", None)
            mem = round(total / 1024 ** 3, 1) if total else None
        except Exception:  # noqa: BLE001
            pass
        out.append(DeviceInfo(f"cuda:{i}", "cuda", name, i, mem,
                              note="单卡可训练；勾选多张 = 多卡并行"))
    if _mps_available():
        chip = platform.processor() or platform.machine()
        out.append(DeviceInfo("mps", "mps", f"Apple Silicon（{chip}）", None, None,
                              note="Metal 加速；个别注意力算子不支持会自动回退 CPU"))
    xpu = getattr(torch, "xpu", None)
    try:
        if xpu is not None and xpu.is_available():
            for i in range(int(xpu.device_count())):
                out.append(DeviceInfo(f"xpu:{i}", "xpu", f"Intel XPU {i}", i, None))
    except Exception:  # noqa: BLE001
        pass
    out.append(DeviceInfo("cpu", "cpu", f"CPU（{platform.machine()}）", None, None,
                          note="始终可用，最慢但最稳"))
    return out


def system_info() -> dict:
    """系统与框架版本（GUI 顶部展示）。"""
    import torch

    cudnn = "—"
    try:
        if torch.backends.cudnn.is_available():
            cudnn = str(torch.backends.cudnn.version())
    except Exception:  # noqa: BLE001
        pass
    return {
        "系统": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "Python": platform.python_version(),
        "PyTorch": torch.__version__,
        "CUDA（编译版本）": torch.version.cuda or "—",
        "cuDNN": cudnn,
        "CUDA 可用": "是" if _cuda_count() else "否",
        "MPS 可用": "是" if _mps_available() else "否",
        "设备数": f"CUDA {_cuda_count()} 张 / CPU 1 个"
                 + ("／MPS 1 个" if _mps_available() else ""),
    }


# ---------------------------------------------------------------- 解析
def _best_available() -> tuple[str, list[int]]:
    """自动选择：CUDA > MPS > XPU > CPU。"""
    for d in list_devices():
        if d.kind == "cuda" and d.index is not None:
            return d.id, [d.index]
    for d in list_devices():
        if d.kind in ("mps", "xpu"):
            return d.id, ([d.index] if d.index is not None else [])
    return "cpu", []


def resolve_device(spec: str | None = "auto") -> tuple[str, list[int]]:
    """把配置写法解析成 ``(设备字符串, 多卡索引列表)``。

    不可用的设备会**优雅回退**到自动选择（配置里写了 ``cuda`` 但在 Mac 上跑，
    也能直接开训），多卡索引会自动过滤掉不存在的卡号。
    """
    raw = "" if spec is None else str(spec).strip().lower()
    if raw in ("", "auto"):
        return _best_available()

    if raw in ("cpu", "mps") or raw.startswith(("xpu:", "xpu")):
        if any(d.id == raw for d in list_devices()):
            idx = int(raw.split(":")[1]) if raw.startswith("xpu:") else None
            return raw, ([idx] if idx is not None else [])
        return _best_available()

    # CUDA：cuda / cuda:1 / cuda:0,1 / 0,1 / "cuda:0 cuda:1"
    # 空格、分号、逗号都当分隔符，token 里的卡号取冒号后面那段
    tokens = [t for t in raw.replace(";", ",").replace(" ", ",").split(",") if t]
    ids: list[int] = []
    for tok in tokens:
        num = tok.split(":", 1)[1] if ":" in tok else tok
        if num.isdigit():
            ids.append(int(num))
    n = _cuda_count()
    if not n:
        return _best_available()
    ids = sorted({i for i in ids if 0 <= i < n}) or [0]
    return f"cuda:{ids[0]}", ids


def device_label(spec: str | None = "auto") -> str:
    """人类可读的设备描述（日志 / 界面用）。"""
    prim, ids = resolve_device(spec)
    table = {d.id: d for d in list_devices()}
    if len(ids) > 1:
        first = table.get(f"cuda:{ids[0]}")
        base = first.name if first else "CUDA GPU"
        return f"{len(ids)}× {base}（多卡并行，卡号 {ids}）"
    d = table.get(prim)
    if d is None:
        return prim
    mem = f"，{d.memory_gb:g} GB" if d.memory_gb else ""
    return f"{d.name}{mem}（{d.id}）"


def describe_spec(spec: str | None) -> str:
    """配置里写的是什么就显示什么（便于界面回显"auto / cuda:0,1"）。"""
    raw = "" if spec is None else str(spec).strip()
    return raw or "auto"


# ---------------------------------------------------------------- 多卡
def wrap_multi_gpu(model, device_ids: list[int] | None, primary: str = "cuda:0"):
    """多卡时包一层 ``nn.DataParallel``；单卡 / CPU / MPS 原样返回。"""
    import torch
    from torch import nn

    ids = list(device_ids or [])
    if len(ids) < 2 or not str(primary).startswith("cuda"):
        return model
    usable = [i for i in ids if i < _cuda_count()]
    if len(usable) < 2:
        return model
    return nn.DataParallel(model, device_ids=usable)


def unwrap_state_dict(state_dict: dict) -> dict:
    """去掉 DataParallel 的 ``module.`` 前缀 → 权重文件与设备无关。"""
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}


def adapt_state_dict(state_dict: dict, model) -> dict:
    """按当前模型（是否 DataParallel）调整权重键前缀。"""
    from torch import nn

    wrapped = isinstance(model, nn.DataParallel)
    has_prefix = any(k.startswith("module.") for k in state_dict)
    if wrapped and not has_prefix:
        return {f"module.{k}": v for k, v in state_dict.items()}
    if not wrapped and has_prefix:
        return unwrap_state_dict(state_dict)
    return state_dict


def _sync(device: str) -> None:
    """等设备把队列里的活干完（测速必须同步，否则只测了"排队时间"）。"""
    import torch

    try:
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        elif str(device) == "mps":
            torch.mps.synchronize()
        elif str(device).startswith("xpu"):
            torch.xpu.synchronize()
    except Exception:  # noqa: BLE001
        pass


def probe_device(spec: str | None = "auto", *, size: int = 512,
                 steps: int = 3) -> dict:
    """在指定设备上**真跑几步** 矩阵乘 + 反传，验证可用性并测速。

    Returns
    -------
    dict : ``{spec, device, device_ids, name, ok, ms_per_step, error}``
    """
    import torch

    device, ids = resolve_device(spec)
    res = {"spec": describe_spec(spec), "device": device, "device_ids": ids,
           "name": device_label(spec), "ok": False, "ms_per_step": None, "error": ""}
    try:
        dev = torch.device(device)
        torch.manual_seed(0)
        a = torch.randn(size, size, device=dev)
        w = torch.randn(size, size, device=dev, requires_grad=True)

        def step():
            w.grad = None
            (a @ w).relu().sum().backward()

        step()                       # 预热（首次会做内核编译/显存分配）
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(max(int(steps), 1)):
            step()
        _sync(device)
        res["ms_per_step"] = (time.perf_counter() - t0) / max(int(steps), 1) * 1000
        res["ok"] = True
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
    return res
