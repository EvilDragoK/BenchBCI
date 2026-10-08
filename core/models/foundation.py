"""EEG 基础模型（Foundation Model）接入层：注册表 + 预训练权重下载 + 构建。

设计要点
--------
1. **注册表**：覆盖 EEG-FM-Benchmark 列出的基础模型，并绑定
   braindecode 在 HuggingFace 上维护的 curated 权重仓库
   （如 ``braindecode/labram-pretrained``）。
2. **权重下载**：``resolve_weights()`` 支持四种写法：
   - ``auto``            → 按注册表里的 HF 仓库自动下载到本地缓存
   - ``hf:<repo>``       → 下载该 HuggingFace 仓库（需 huggingface_hub）
   - ``hf:<repo>/<file>``→ 直链下载单个文件（无需额外依赖）
   - ``url:<http...>``   → 直链下载
   - 其它                → 视为本地路径，存在即用
   缓存目录：``~/.cache/bci_lab/pretrained``（可用环境变量 ``BCI_LAB_CACHE`` 覆盖）。
3. **构建**：从 braindecode 动态取类，用签名内省自动匹配 ``n_chans / n_outputs /
   n_times / sfreq`` 等参数名（各模型命名不同），缺失依赖时给出中文安装提示。
"""
from __future__ import annotations

import inspect
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

CACHE_ROOT = Path(os.environ.get("BCI_LAB_CACHE", Path.home() / ".cache" / "bci_lab"))
PRETRAINED_DIR = CACHE_ROOT / "pretrained"


@dataclass(frozen=True)
class FoundationSpec:
    """一个基础模型的元信息。"""

    name: str                        # 本项目使用的模型名（小写）
    display: str                     # 显示名
    cls_name: str | None = None      # braindecode 中的类名
    interp_cls: str | None = None    # 通道数不匹配时使用的插值变体类名（Interpolated*）
    hf_repo: str | None = None       # 含预训练权重的 HuggingFace 仓库（auto 下载用）
    arch_repo: str | None = None     # 仅架构/配置的 HF 仓库
    requires: tuple[str, ...] = ("braindecode",)
    source: str = ""                 # 官方源码/权重说明
    note: str = ""


# ---------------------------------------------------------------- 注册表
FOUNDATION_MODELS: dict[str, FoundationSpec] = {
    "labram": FoundationSpec(
        "labram", "LaBraM", cls_name="Labram", interp_cls="InterpolatedLaBraM", hf_repo="braindecode/labram-pretrained",
        arch_repo="braindecode/Labram",
        source="https://github.com/935963004/LaBraM",
        note="ICLR'24，2534 小时 EEG 预训练，patch=200(1s) @200Hz"),
    "cbramod": FoundationSpec(
        "cbramod", "CBraMod", cls_name="CBraMod", hf_repo="braindecode/cbramod-pretrained",
        arch_repo="braindecode/CBraMod",
        source="https://github.com/wjq-learning/CBraMod",
        note="Criss-Cross 注意力，TUEG 2.3 万小时预训练（社区最常用之一）"),
    "eegpt": FoundationSpec(
        "eegpt", "EEGPT", cls_name="EEGPT", hf_repo="braindecode/eegpt-pretrained",
        arch_repo="braindecode/EEGPT",
        source="https://github.com/BINE022/EEGPT", note="自回归预训练的通用 EEG 主干"),
    "biot": FoundationSpec(
        "biot", "BIOT", cls_name="BIOT", interp_cls="InterpolatedBIOT",
        hf_repo="braindecode/biot-pretrained-six-datasets-18chs",
        arch_repo="braindecode/BIOT",
        source="https://github.com/ycq091044/BIOT",
        note="另有 prest-16chs / shhs-prest-18chs 两个权重仓库可选"),
    "bendr": FoundationSpec(
        "bendr", "BENDR", cls_name="BENDR", interp_cls="InterpolatedBENDR", hf_repo="braindecode/braindecode-bendr",
        arch_repo="braindecode/BENDR",
        source="https://github.com/SPOClab-ca/BENDR", note="Transformer + 卷积编码器"),
    "signaljepa": FoundationSpec(
        "signaljepa", "Signal-JEPA", cls_name="SignalJEPA", interp_cls="InterpolatedSignalJEPA",
        hf_repo="braindecode/SignalJEPA-pretrained",
        arch_repo="braindecode/SignalJEPA",
        source="https://github.com/facebookresearch/signal-jepa",
        note="另有 Contextual / PreLocal / PostLocal 变体权重"),
    "brant": FoundationSpec(
        "brant", "Brant", cls_name=None, hf_repo="braindecode/brant-pretrained",
        source="https://github.com/zhiyuan-yao/Brant",
        note="权重仓库已存在（braindecode/brant-pretrained），但当前 braindecode 版本"
             "未提供 Brant 类；需按官方仓库自定义适配器"),
    "brainbert": FoundationSpec(
        "brainbert", "BrainBERT", cls_name=None,
        hf_repo="braindecode/brainbert-pretrained",
        source="https://github.com/kathrinse/BrainBERT",
        note="权重可下载（braindecode/brainbert-pretrained），模型类需按官方仓库接入"),
    "eegdino": FoundationSpec(
        "eegdino", "EEG-DINO", cls_name="EEGDINO",
        hf_repo="braindecode/eegdino-small-pretrained",
        source="https://github.com/yi-ding-cs/EEG-DINO", note="另有 medium 规模权重"),
    "codebrain": FoundationSpec(
        "codebrain", "CodeBrain", cls_name="CodeBrain",
        hf_repo="braindecode/codebrain-pretrained",
        arch_repo="braindecode/CodeBrain", source="braindecode"),
    "luna": FoundationSpec(
        "luna", "LUNA", cls_name="LUNA", hf_repo=None, arch_repo="braindecode/LUNA",
        source="https://github.com/ArlindKadra/LUNA",
        note="在 braindecode 中提供架构；权重需按官方仓库另行获取"),
    # —— 以下基础模型未收录在 braindecode 的 curated 列表里，需按官方仓库手动准备 ——
    "neurogpt": FoundationSpec(
        "neurogpt", "NeuroGPT", cls_name=None, hf_repo=None,
        source="https://github.com/wenhui0206/NeuroGPT",
        note="需按官方仓库准备权重并注册适配器（见 spec.note 与文档说明）"),
    "brainomni": FoundationSpec(
        "brainomni", "BrainOmni", cls_name=None, hf_repo=None,
        arch_repo="Dingkun0817/EEG-FM-Benchmark: models/FM/BrainOmni_Base",
        source="https://github.com/OpenTSLab/BrainOmni",
        note="参考仓库只含 model_cfg.json，权重需从官方获取"),
    "eegmamba": FoundationSpec(
        "eegmamba", "EEGMamba", cls_name=None, hf_repo=None,
        arch_repo="Dingkun0817/EEG-FM-Benchmark: models/FM/EEGMamba",
        source="EEG-FM-Benchmark", note="权重需按官方仓库准备"),
    "single": FoundationSpec(
        "single", "SingLEM", cls_name=None, hf_repo=None,
        arch_repo="Dingkun0817/EEG-FM-Benchmark: models/FM/SingLEM",
        source="EEG-FM-Benchmark", note="权重需按官方仓库准备"),
    "tfmtokenizer": FoundationSpec(
        "tfmtokenizer", "TFMTokenizer", cls_name=None, hf_repo=None,
        arch_repo="Dingkun0817/EEG-FM-Benchmark: models/FM/TFMTokenizer",
        source="EEG-FM-Benchmark", note="权重需按官方仓库准备"),
}

#: 可以直接用 ``pretrained: auto`` 自动下载权重的基础模型
AUTO_WEIGHT_MODELS = tuple(k for k, v in FOUNDATION_MODELS.items() if v.hf_repo)


def is_foundation(name: str) -> bool:
    return str(name).lower() in FOUNDATION_MODELS


def get_spec(name: str) -> FoundationSpec:
    key = str(name).lower()
    if key not in FOUNDATION_MODELS:
        raise KeyError(f"未知基础模型：{name}（可用：{sorted(FOUNDATION_MODELS)}）")
    return FOUNDATION_MODELS[key]


def list_foundations() -> list[str]:
    return list(FOUNDATION_MODELS)


# ---------------------------------------------------------------- 权重下载
def _log_default(msg: str):
    print(msg, flush=True)


def _download_url(url: str, dest: Path, log=_log_default, timeout: int = 120) -> Path:
    """直链下载（带粗略进度）。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "bci-lab/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total and done % (1024 * 1024 * 8) < 1024 * 256:
                    log(f"    下载中 {done / 1e6:.1f}/{total / 1e6:.1f} MB")
    tmp.replace(dest)
    log(f"    已保存 → {dest}（{dest.stat().st_size / 1e6:.1f} MB）")
    return dest


def _hf_download(repo: str, filename: str | None, cache_dir: Path, log) -> Path:
    """从 HuggingFace 下载：有 huggingface_hub 就整仓拉取，否则直链单文件。"""
    repo_dir = cache_dir / repo.replace("/", "__")
    if filename:
        dest = repo_dir / Path(filename).name
        if dest.exists() and dest.stat().st_size > 0:
            log(f"    命中缓存 → {dest}")
            return dest
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        return _download_url(url, dest, log)

    try:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(repo_id=repo, cache_dir=str(cache_dir),
                                      allow_patterns=["*.safetensors", "*.bin", "*.pt",
                                                      "*.pth", "*.json", "*.yaml", "*.yml"]))
        log(f"    已下载仓库 {repo} → {path}")
        return path
    except ImportError as e:  # noqa: BLE001
        raise ImportError(
            f"下载 HuggingFace 仓库 `{repo}` 需要 huggingface_hub：\n"
            f"    pip install huggingface_hub\n"
            f"或改用单文件写法：pretrained: hf:{repo}/<文件名>") from e


def resolve_weights(ref, spec: FoundationSpec | None = None, cache_dir: Path | None = None,
                    log=_log_default, allow_download: bool = True) -> Path | None:
    """把 ``model.pretrained`` 配置解析成本地路径（必要时下载）。

    支持：``auto`` / ``hf:<repo>`` / ``hf:<repo>/<file>`` / ``url:<...>`` / 本地路径。
    """
    cache_dir = Path(cache_dir or PRETRAINED_DIR)
    ref = "" if ref is None else str(ref).strip()
    if not ref:
        return None

    if ref.lower() == "auto":
        if spec is None or not spec.hf_repo:
            raise ValueError(
                f"该模型没有登记可自动下载的权重仓库"
                f"（{spec.display if spec else '?'}）。请手动指定 "
                f"pretrained: hf:<repo> 或本地权重路径；"
                f"官方来源：{(spec.source if spec else '') or '见文档'}")
        ref = f"hf:{spec.hf_repo}"

    if ref.startswith("hf:"):
        body = ref[3:].strip("/")
        parts = body.split("/")
        # 两段 = org/name（整仓）；三段及以上 = org/name/文件路径
        if len(parts) >= 3:
            repo, filename = "/".join(parts[:2]), "/".join(parts[2:])
        else:
            repo, filename = body, ""
        log(f"  下载预训练权重：{repo}" + (f" → {filename}" if filename else "（整仓）"))
        if not allow_download:
            raise RuntimeError(f"当前不允许下载（pretrained={ref}）")
        return _hf_download(repo, filename or None, cache_dir, log)

    if ref.startswith("url:"):
        url = ref[4:]
        dest = cache_dir / "downloads" / Path(url.split("?")[0]).name
        if dest.exists() and dest.stat().st_size > 0:
            log(f"    命中缓存 → {dest}")
            return dest
        log(f"  下载预训练权重：{url}")
        if not allow_download:
            raise RuntimeError(f"当前不允许下载（pretrained={ref}）")
        return _download_url(url, dest, log)

    path = Path(os.path.expanduser(ref))
    if not path.is_absolute():
        from .. import PROJECT_ROOT

        candidate = PROJECT_ROOT / path
        path = candidate if candidate.exists() else path
    if path.exists():
        if path.is_dir():
            log(f"  使用本地权重目录：{path}")
        else:
            log(f"  使用本地权重文件：{path}（{path.stat().st_size / 1e6:.1f} MB）")
        return path
    raise FileNotFoundError(f"预训练权重不存在：{ref}")


# ---------------------------------------------------------------- 构建模型
def _compatible_kwargs(cls, params: dict, n_classes: int, n_ch: int, n_times: int,
                       sfreq: float, ch_names: list[str] | None = None) -> dict:
    """按构造签名挑参数名（各基础模型的命名不一致，这里做自动匹配）。"""
    try:
        names = set(inspect.signature(cls.__init__).parameters)
    except (TypeError, ValueError):
        names = set()
    out: dict = {}
    if ch_names and ("chs_info" in names or "ch_names" in names):
        # braindecode 的插值类需要知道电极名/位置才能把数据插值到原生骨架
        try:
            import mne

            info = mne.create_info(list(ch_names), float(sfreq), "eeg")
            info.set_montage("standard_1005", on_missing="ignore")
            if "chs_info" in names:
                out["chs_info"] = info["chs"]
            else:
                out["ch_names"] = list(ch_names)
        except Exception:  # noqa: BLE001  没有 mne 时退化为只给名字
            if "ch_names" in names:
                out["ch_names"] = list(ch_names)
    for key in ("n_outputs", "n_classes", "num_classes", "output_size"):
        if key in names:
            out[key] = int(n_classes)
            break
    for key in ("n_chans", "n_channels", "in_chans", "n_eeg_channels"):
        if key in names:
            out[key] = int(n_ch)
            break
    has_times = False
    for key in ("n_times", "input_window_samples", "n_times_out", "seq_len"):
        if key in names:
            out[key] = int(n_times)
            has_times = True
            break
    if "sfreq" in names:
        out["sfreq"] = float(sfreq)
    if not has_times and "input_window_seconds" in names:
        out["input_window_seconds"] = float(n_times) / float(sfreq)
    for key, value in (params or {}).items():
        if key in names:
            out[key] = value
    return out


def _CropAlign(backbone, n_times: int):
    """把时间维裁剪到 patch 的整数倍后再送入主干（很多 FM 有此硬性要求）。"""
    import torch.nn as nn

    class CropAlign(nn.Module):
        def __init__(self, encoder, keep: int):
            super().__init__()
            self.backbone = encoder
            self.keep = int(keep)

        def forward(self, x):
            return self.backbone(x[..., : self.keep])

    return CropAlign(backbone, n_times)


def _wrap_with_head(backbone, n_classes: int, n_ch: int, n_times: int,
                    probe_out, log=_log_default, display: str = ""):
    """基础模型若只输出特征（无分类头），自动补一个分类头。

    命名保持 ``backbone.*`` / ``classifier.*``，便于微调策略按模块名冻结
    （``freeze: encoder`` 会冻结 backbone、只训练 classifier）。
    """
    import torch
    import torch.nn as nn

    if probe_out.ndim == 2 and probe_out.shape[1] == n_classes:
        return backbone                      # 自带分类头，直接用

    class FoundationWithHead(nn.Module):
        def __init__(self, encoder, feat_dim: int):
            super().__init__()
            self.backbone = encoder
            self.classifier = nn.Linear(feat_dim, n_classes)

        def features(self, x):
            f = self.backbone(x)
            if f.ndim > 2:                   # (B, C, T) → 时间维平均
                f = f.mean(dim=-1)
            if f.ndim > 2:                   # 剩余维度展平
                f = f.flatten(1)
            return f

        def forward(self, x):
            return self.classifier(self.features(x))

    with torch.no_grad():
        feat = backbone(torch.zeros(1, n_ch, n_times))
        if feat.ndim > 2:
            feat = feat.mean(dim=-1)
        if feat.ndim > 2:
            feat = feat.flatten(1)
    wrapped = FoundationWithHead(backbone, int(feat.shape[1]))
    log(f"  {display} 只输出特征（{tuple(probe_out.shape)}），"
        f"已自动补分类头 Linear({feat.shape[1]} → {n_classes})")
    return wrapped


def build_foundation_model(name: str, n_classes: int, n_ch: int, n_times: int,
                           sfreq: float = 250.0, params: dict | None = None,
                           ch_names: list[str] | None = None, log=_log_default):
    """构建基础模型（依赖 braindecode）。缺依赖/类名不匹配时给出明确中文提示。"""
    spec = get_spec(name)
    try:
        import braindecode.models as bmodels
    except ImportError as e:  # noqa: BLE001
        raise ImportError(
            f"加载基础模型「{spec.display}」需要 braindecode：\n"
            f"    pip install braindecode\n"
            f"官方来源：{spec.source or spec.arch_repo or 'braindecode'}") from e

    def _resolve(cls_name: str | None):
        if not cls_name:
            return None
        cls = getattr(bmodels, cls_name, None)
        if cls is None:                 # 兜底：按名字模糊匹配
            target = cls_name.lower().replace("_", "")
            for attr in dir(bmodels):
                if attr.lower().replace("_", "") == target:
                    return getattr(bmodels, attr)
        return cls

    cls = _resolve(spec.cls_name) or _resolve(spec.display)
    if cls is None:
        avail = sorted(a for a in dir(bmodels) if a[0].isupper())[:25]
        raise AttributeError(
            f"当前 braindecode 版本里找不到「{spec.display}」（期望类名 {spec.cls_name}）。\n"
            f"可用模型示例：{avail}\n"
            f"可尝试：pip install -U braindecode")

    def _instantiate(target):
        kwargs = _compatible_kwargs(target, params or {}, n_classes, n_ch, n_times,
                                    sfreq, ch_names)
        log(f"  构建基础模型 {spec.display}（{target.__name__}）："
            + ", ".join(f"{a}={v if not isinstance(v, list) else f'<{len(v)} 项>'}"
                        for a, v in kwargs.items()))
        return target(**kwargs)

    def _try(target):
        """构造 + 前向试跑（通道数/Patch 对齐问题往往到 forward 才暴露）。"""
        import re

        import torch

        model = _instantiate(target)
        try:
            with torch.no_grad():
                out = model(torch.zeros(1, n_ch, n_times))
        except Exception as e:  # noqa: BLE001
            m = re.search(r"chunks of (\d+)", str(e))
            if not m:                     # 不是 patch 对齐问题 → 原样抛出
                raise
            patch = int(m.group(1))
            t2 = (n_times // patch) * patch
            if t2 <= 0:
                raise
            log(f"  输入长度 {n_times} 点不是 patch({patch}) 的整数倍，"
                f"自动裁剪到 {t2} 点（丢弃尾部 {n_times - t2} 点）")
            model = _CropAlign(_instantiate(target), t2)
            with torch.no_grad():
                out = model(torch.zeros(1, n_ch, n_times))
        return _wrap_with_head(model, n_classes, n_ch, n_times, out, log, spec.display)

    try:
        return _try(cls)
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        if spec.interp_cls and ("channel" in msg or "ch_names" in msg or "chs_info" in msg):
            interp = _resolve(spec.interp_cls)
            if interp is not None:
                log(f"  {spec.display} 原生版本不接受 {n_ch} 通道 / 缺少电极信息，"
                    f"自动改用插值变体 {spec.interp_cls}")
                try:
                    return _try(interp)
                except Exception as e2:  # noqa: BLE001
                    raise _friendly_error(spec, n_ch, n_times, sfreq, e2) from e2
        raise _friendly_error(spec, n_ch, n_times, sfreq, e) from e


def _friendly_error(spec: FoundationSpec, n_ch: int, n_times: int, sfreq: float,
                    err: Exception) -> ValueError:
    """把底层报错翻译成可操作的中文提示。"""
    msg = str(err)
    low = msg.lower()
    hints = [f"基础模型 {spec.display} 构建/前向失败。",
             f"  输入条件：{n_ch} 通道 × {n_times} 采样点 @ {sfreq:g} Hz"]
    if any(t in low for t in ("divisible", "patch", "multiple of", "expected", "shape")):
        hints.append("  该模型对输入长度/采样率有硬性要求（常见：200 Hz、patch = 200 点，"
                     "即 n_times 需为 200 的整数倍）→ 请在「预处理工作台」加「重采样」步骤"
                     "并调整分段长度，详见 docs/FOUNDATION_MODELS.md")
    if "channel" in low or "ch_names" in low:
        hints.append(f"  通道数不匹配（当前 {n_ch}）：可换用插值变体，或按模型的通道要求"
                     f"调整数据（如 EEG-DINO 官方权重为 19 通道）")
    hints.append(f"  原始错误：{msg}")
    return ValueError("\n".join(hints))
