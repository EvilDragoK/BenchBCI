"""自定义模型：**以内置模型定义为模板改写 → 自动检查 → 保存为新模型**。

目录结构::

    models_custom/
        <模型名>.py        # torch.nn.Module 源码（可直接再编辑）
        <模型名>.json      # 元信息（基础模型 / 说明 / 创建时间 / 检查时的维度）

检查（``check_code``）不是"语法检查"，而是**真的把网络跑一遍**：

1. 编译并执行代码，找出对外暴露的 ``nn.Module`` 子类；
2. 按当前数据的 (类别数, 通道数, 采样点) 实例化；
3. 前向传播：喂 ``(B, 1, C, T)`` 随机张量，确认输出为 ``(B, n_classes)``；
4. **特征流向**：用 forward hook 逐层记录每个层的输入/输出形状与参数量；
5. 反向传播：``loss.backward()`` 确认可训练，并揪出"未参与前向"的悬空参数；
6. 数值健康：输出无 NaN/INF、中间特征没有 0 维。

约定：文件里可用 ``__model__ = "类名"`` 显式指定对外提供的模型类；不写则取文件里
**最后一个** ``nn.Module`` 子类。训练时框架按
``build_model(name, n_classes, n_ch, n_times, params)`` 实例化，因此构造函数请保留
这三个维度参数（命名可以是 ``n_classes`` / ``n_ch`` / ``n_times`` 或常见别名）。
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import re
from datetime import datetime
from pathlib import Path

from .. import PROJECT_ROOT

#: 自定义模型目录（不放 core 里，避免被当成内置代码）
CUSTOM_DIR = PROJECT_ROOT / "models_custom"

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ================================================================ 代码片段库
SNIPPETS: dict[str, str] = {
    "金字塔（多尺度）池化": '''
class PyramidPool(nn.Module):
    """金字塔/多尺度池化：把特征在多个时间尺度上池化后**展平拼接**。

    输入 (B, C, 1, T) 或 (B, C, T)；输出通道 = C × Σscales。
    例：C=16、scales=(1,2,4,8) → 输出 (B, 16×15) = (B, 240)。
    """

    def __init__(self, scales=(1, 2, 4, 8)):
        super().__init__()
        self.scales = tuple(int(s) for s in scales)

    def forward(self, x):
        if x.ndim == 4:                       # (B, C, 1, T)
            levels = [nn.functional.adaptive_avg_pool2d(x, (1, s)) for s in self.scales]
        else:                                 # (B, C, T)
            levels = [nn.functional.adaptive_avg_pool1d(x, s) for s in self.scales]
        return torch.cat([lv.flatten(1) for lv in levels], dim=1)
''',
    "SE 通道注意力": '''
class SEBlock(nn.Module):
    """通道注意力（Squeeze-Excitation）：全局池化 → 两层 FC → 逐通道加权。"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(int(channels) // max(int(reduction), 1), 4)
        self.fc = nn.Sequential(nn.Linear(channels, hidden), nn.ELU(),
                                nn.Linear(hidden, channels), nn.Sigmoid())

    def forward(self, x):
        c = x.shape[1]
        w = self.fc(x.mean(dim=tuple(range(2, x.ndim))))
        return x * w.view(x.shape[0], c, *([1] * (x.ndim - 2)))
''',
    "残差块（时间卷积）": '''
class ResidualBlock2d(nn.Module):
    """残差块：1×1 投影对齐通道 + 两层 3×3 时间卷积（输出形状与输入一致）。"""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.25):
        super().__init__()
        self.proj = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                     if in_ch != out_ch else nn.Identity())
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(out_ch), nn.ELU(),
            nn.Conv2d(out_ch, out_ch, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(out_ch), nn.Dropout(dropout),
        )
        self.act = nn.ELU()

    def forward(self, x):
        return self.act(self.body(x) + self.proj(x))
''',
    "时间注意力（软加权）": '''
class TemporalAttention(nn.Module):
    """时间维注意力：对每个时间点打分并归一化（输入 (B, C, T)，输出同形状）。"""

    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, 1)

    def forward(self, x):
        w = torch.softmax(self.score(x), dim=-1)
        return x * w
''',
}


# ================================================================ 注册表
def list_custom() -> dict[str, dict]:
    """列出已保存的自定义模型：名称 → 元信息。"""
    out: dict[str, dict] = {}
    if not CUSTOM_DIR.is_dir():
        return out
    for f in sorted(CUSTOM_DIR.glob("*.py")):
        meta: dict = {}
        mf = f.with_suffix(".json")
        if mf.exists():
            try:
                meta = json.loads(mf.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                meta = {}
        meta.setdefault("base", "")
        meta.setdefault("note", "")
        meta.setdefault("created", "")
        meta["file"] = str(f)
        meta["lines"] = len(f.read_text(encoding="utf-8").splitlines())
        out[f.stem] = meta
    return out


def custom_names() -> list[str]:
    return list(list_custom())


def is_custom(name) -> bool:
    """该名字是否是已保存的自定义模型。"""
    return bool(_IDENT.match(str(name))) and (CUSTOM_DIR / f"{name}.py").is_file()


def custom_path(name: str) -> Path:
    return CUSTOM_DIR / f"{name}.py"


# ================================================================ 模板生成
def _module_source(base: str) -> tuple[str, str]:
    """取内置模型的**整份源码**（含模块级辅助函数），返回 (类名, 源码)。

    必须是整份文件：像 EEGNet 用到了模块级的 ``_odd`` / ``_same_padding``，
    只取 class 源码会得到 NameError。
    """
    from . import TORCH_MODELS

    if base not in TORCH_MODELS:
        raise KeyError(f"暂不支持以 {base} 为模板（可选：{sorted(TORCH_MODELS)}）")
    cls = TORCH_MODELS[base]
    return cls.__name__, Path(inspect.getsourcefile(cls)).read_text(encoding="utf-8")


def _split_source(src: str) -> tuple[list[str], str]:
    """拆出 (``__future__`` 导入行, 其余源码)。

    用 AST 定位，避免手工匹配字符串时踩到多行 docstring 的坑：
    - 模块级 docstring 直接丢弃（模板会写自己的说明）；
    - ``from __future__ import ...`` 必须留在文件最前，单独提出来。
    """
    import ast

    lines = src.splitlines(keepends=True)
    drop: set[int] = set()
    futures: list[str] = []
    for node in ast.parse(src).body:
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            drop.update(range(node.lineno - 1, node.end_lineno))       # docstring
        elif isinstance(node, ast.ImportFrom) and node.module == "__future__":
            futures += [lines[i].strip()
                        for i in range(node.lineno - 1, node.end_lineno)]
            drop.update(range(node.lineno - 1, node.end_lineno))
    rest = "".join(l for i, l in enumerate(lines) if i not in drop)
    return futures, rest.lstrip("\n")


def template_source(base: str, new_name: str, note: str = "") -> str:
    """生成自定义模型模板：内置源码 + 文件头 + （可选）类名重写。"""
    base_cls, src = _module_source(base)
    cls_name = new_name if _IDENT.match(str(new_name)) else f"{base_cls}Custom"

    futures, rest = _split_source(src)
    if cls_name != base_cls:
        rest = re.sub(rf"^class\s+{base_cls}\s*\(", f"class {cls_name}(", rest,
                      count=1, flags=re.M)
    head_future = ("\n".join(futures) + "\n\n") if futures else ""
    header = f'''"""自定义模型「{cls_name}」——基于 {base}（{base_cls}）改写。

{note or "在此说明你的改动（例如：在 block1/block2 之后加入金字塔池化特征）。"}

要点（改完点「检查」会按当前数据的维度真跑一遍）：
- 构造函数请保留 n_classes / n_ch / n_times 三个维度参数；
- 层的输出形状必须能一路走到分类头，最后输出 (B, n_classes)；
- 下面用 forward hook 逐层核对输入/输出形状，维度对不上会直接报出来。
"""

# 对外暴露的模型类（改名时请同步修改）
__model__ = "{cls_name}"

'''
    return head_future + header + rest + "\n"


# ---------------------------------------------------------------- 从零设计
#: 「从零设计」可选的骨架风格：(键, 下拉框显示名)
BLANK_SOURCES: list[tuple[str, str]] = [
    ("blank_cnn", "（从零设计）空白骨架：卷积 + 自动分类头"),
    ("blank_linear", "（从零设计）最小骨架：线性基线"),
]


def sanitize_identifier(name: str, fallback: str = "MyEEGNet") -> str:
    """把用户输入的模型名转成合法的 Python 类名（标识符）。"""
    s = re.sub(r"\W+", "_", str(name or "").strip())
    s = s.strip("_")
    if not s:
        return fallback
    if s[0].isdigit():
        s = "M" + s
    return s


def blank_template(new_name: str = "MyEEGNet", n_classes: int = 4, n_ch: int = 22,
                   n_times: int = 501, style: str = "blank_cnn", note: str = "") -> str:
    """**从零设计**的骨架：按你输入的英文模型名生成 ``class <名称>(nn.Module)``。

    生成的代码里已经把三件事写死并注释清楚：

    - **输入约定**：``x`` 为 ``(B, 1, n_ch, n_times)``（也兼容 ``(B, n_ch, n_times)``）；
    - **中间特征**：每一层的形状注释，改动后用「检查」逐层核对；
    - **输出约定**：``(B, n_classes)``，分类头维度**自动探测**（改了卷积层不用手算）。

    style : ``blank_cnn`` 最小卷积网络（推荐起点）；``blank_linear`` 极简线性基线。
    """
    cls = sanitize_identifier(new_name)
    if style == "blank_linear":
        body = f'''class {cls}(nn.Module):
    """最小骨架：先把「输入 → 输出」的通道打通，再逐层往里加东西。"""

    def __init__(self, n_classes: int = {int(n_classes)}, n_ch: int = {int(n_ch)},
                 n_times: int = {int(n_times)}, dropout: float = 0.25):
        super().__init__()
        self.n_ch, self.n_times = int(n_ch), int(n_times)
        # 输入约定：x = (B, 1, n_ch, n_times)
        # 这里把每个通道的时间信号平均成一个标量 → 中间特征维度 = n_ch
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.n_ch, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:                       # (B, 1, n_ch, n_times)
            x = x.squeeze(1)                  # → (B, n_ch, n_times)
        feat = x.mean(dim=-1)                 # (B, n_ch)        ← 中间特征
        return self.classifier(self.dropout(feat))   # (B, n_classes) ← 输出约定
'''
    else:
        body = f'''class {cls}(nn.Module):
    """空白骨架：一个**能直接跑通**的最小卷积网络，在它的基础上加层即可。"""

    def __init__(self, n_classes: int = {int(n_classes)}, n_ch: int = {int(n_ch)},
                 n_times: int = {int(n_times)}, F1: int = 8, F2: int = 16,
                 kernel_length: int = 63, dropout: float = 0.25):
        super().__init__()
        self.n_ch, self.n_times = int(n_ch), int(n_times)
        k = int(kernel_length)
        k = k if k % 2 == 1 else k + 1
        pad = (k - 1) // 2        # 对称填充：时间长度保持不变

        # ---- 输入约定：x = (B, 1, n_ch, n_times) ----
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, k), padding=(0, pad), bias=False),  # → (B, F1, n_ch, T)
            nn.BatchNorm2d(F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),                                    # → (B, F1, n_ch, T/4)
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F2, (n_ch, 1), bias=False),                # 跨通道融合 → (B, F2, 1, T/4)
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),                                    # → (B, F2, 1, T/32)
            nn.Dropout(dropout),
        )
        # ---- 分类头维度自动探测：改了上面的层也不用手算 feat ----
        with torch.no_grad():
            feat = self.block2(self.block1(
                torch.zeros(1, 1, n_ch, n_times))).flatten(1).shape[1]
        self.classifier = nn.Linear(feat, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:                    # (B, n_ch, n_times) → (B, 1, n_ch, n_times)
            x = x.unsqueeze(1)
        h = self.block1(x)                 # (B, F1, n_ch, T/4)
        h = self.block2(h)                 # (B, F2, 1, T/32)
        return self.classifier(h.flatten(1))     # (B, n_classes)   ← 输出约定
'''
    header = f'''"""自定义模型「{cls}」——从零设计。

{note or "（在这里写清楚你的设计思路）"}

输入 / 输出约定（点「检查」会按当前数据的维度真跑一遍并逐层核对）：
    输入  x  : (B, 1, {int(n_ch)}, {int(n_times)})    也兼容 (B, {int(n_ch)}, {int(n_times)})
    中间特征 : 见「检查结果」表的逐层「输入/输出形状」
    输出  y  : (B, {int(n_classes)})                  ← 必须是「batch × 类别数」

改法建议
--------
1. 先保持现在的形状跑通（点「检查」应为通过）；
2. 再逐层替换/加宽：每改一处就点一次「检查」，维度对不上会直接指出是**哪一层**
   （最常见：忘了 flatten、池化核把时间维压成 0、改了卷积忘了同步分类头）；
3. 分类头的 in_features 已用「自动探测」写好，加层后通常不用手算。
"""
from __future__ import annotations

import torch
import torch.nn as nn

# 对外暴露的模型类（改名时请同步修改）
__model__ = "{cls}"


'''
    return header + body


def save_custom(name: str, code: str, base: str = "", note: str = "",
                shape: dict | None = None) -> Path:
    """把代码保存为新模型（``models_custom/<name>.py`` + 同名 json 元信息）。"""
    name = str(name).strip()
    if not _IDENT.match(name):
        raise ValueError(f"模型名必须是合法的 Python 标识符（字母/数字/下划线，不能数字开头）：{name!r}")
    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    path = custom_path(name)
    path.write_text(code, encoding="utf-8")
    meta = {"name": name, "base": base or "", "note": note or "",
            "created": datetime.now().isoformat(timespec="seconds"),
            "shape": dict(shape or {})}
    path.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def delete_custom(name: str) -> bool:
    """删除自定义模型（源码 + 元信息）。"""
    f = custom_path(name)
    if not f.exists():
        return False
    f.unlink()
    f.with_suffix(".json").unlink(missing_ok=True)
    return True


def read_source(name: str) -> str:
    return custom_path(name).read_text(encoding="utf-8")


# ================================================================ 加载
def _exec_code(code: str) -> dict:
    """在受控命名空间里执行用户代码（torch/nn/np/F/math 已就绪）。"""
    import math

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    ns = {"__name__": "bci_custom_model", "__file__": "<自定义模型>",
          "torch": torch, "nn": nn, "np": np, "F": F, "math": math}
    exec(compile(code, "<自定义模型>", "exec"), ns)  # noqa: S102 - 用户自己的模型定义
    return ns


def model_classes(ns: dict) -> list[type]:
    """本文件里定义的全部 nn.Module 子类（按定义顺序）。"""
    import torch.nn as nn

    out = []
    for v in ns.values():
        if (isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module
                and v.__module__ == ns.get("__name__")):
            out.append(v)
    return out


def pick_class(ns: dict) -> type:
    """取对外暴露的模型类（优先 ``__model__``，否则取最后一个）。"""
    classes = model_classes(ns)
    if not classes:
        raise ValueError("代码里没有找到 nn.Module 子类定义（class Xxx(nn.Module)）")
    want = ns.get("__model__")
    if want:
        for c in classes:
            if c.__name__ == want:
                return c
        raise ValueError(f"__model__ 指定的类 {want!r} 不存在；"
                         f"文件里的类：{[c.__name__ for c in classes]}")
    return classes[-1]


def load_class(name: str) -> type:
    """按名称加载自定义模型的类。"""
    path = custom_path(name)
    if not path.exists():
        raise FileNotFoundError(f"自定义模型不存在：{path}")
    ns = _exec_code(path.read_text(encoding="utf-8"))
    return pick_class(ns)


# ================================================================ 构建
def instantiate(cls, n_classes: int, n_ch: int, n_times: int, params: dict | None = None):
    """按 (n_classes, n_ch, n_times) 实例化，返回 (model, 说明)。

    依次尝试：完整关键字 → 按签名过滤的关键字 → 位置参数 → 仅 params → 无参，
    这样既支持标准命名，也容忍 ``n_chans`` / ``n_outputs`` 之类的别名。
    """
    params = dict(params or {})
    dims = {"n_classes": int(n_classes), "n_ch": int(n_ch), "n_times": int(n_times)}
    aliases = {
        "n_classes": ("n_classes", "num_classes", "n_outputs", "n_class"),
        "n_ch": ("n_ch", "n_chans", "n_channels", "in_chans"),
        "n_times": ("n_times", "input_window_samples", "seq_len"),
    }
    sig = None
    try:
        sig = inspect.signature(cls.__init__)
        names = {p for p in sig.parameters if p != "self"}
    except (TypeError, ValueError):  # pragma: no cover
        names = set()

    attempts: list[tuple[str, dict]] = []
    full = {**dims, **params}
    attempts.append(("完整关键字", full))
    filtered = {k: v for k, v in params.items() if k in names}
    for key, cands in aliases.items():
        for alias in cands:
            if alias in names:
                filtered[alias] = dims[key]
                break
    attempts.append(("按签名匹配", filtered))
    attempts.append(("仅 params", params))

    last_err: Exception | None = None
    for tag, kwargs in attempts:
        try:
            return cls(**kwargs), f"{tag}：{kwargs or '无参数'}"
        except TypeError as e:
            last_err = e
        except Exception as e:  # noqa: BLE001 - 构造内部的错误直接暴露
            last_err = e
    try:
        return cls(int(n_classes), int(n_ch), int(n_times)), "位置参数：n_classes, n_ch, n_times"
    except Exception as e:  # noqa: BLE001
        last_err = e
    try:
        return cls(), "无参数构造"
    except Exception as e:  # noqa: BLE001
        raise TypeError(
            f"无法实例化 {cls.__name__}：{type(last_err).__name__}: {last_err}\n"
            f"请检查构造函数签名是否为 "
            f"(__init__(self, n_classes=..., n_ch=..., n_times=..., ...))；"
            f"当前签名：__init__{sig if names else '(无法解析)'}") from e


def _leaf_modules(model):
    """只取叶子层（真正干活的层），避免容器重复计数。"""
    for name, mod in model.named_modules():
        if mod is model:
            continue
        if len(list(mod.children())):
            continue
        yield name, mod


def check_code(code: str, n_classes: int = 4, n_ch: int = 22, n_times: int = 501,
               params: dict | None = None, batch: int = 4) -> dict:
    """检查模型定义：构建 → 前向（逐层维度）→ 反向 → 数值健康。

    返回字典（可直接喂给 GUI 表格）::

        {ok, errors[], warnings[], notes[], steps[{i,name,type,in,out,params}],
         classes[], target_class, input_shape, output_shape, n_params, n_classes, ...}
    """
    import torch
    import torch.nn as nn

    res: dict = {
        "ok": False, "errors": [], "warnings": [], "notes": [], "steps": [],
        "classes": [], "target_class": None, "input_shape": None,
        "output_shape": None, "n_params": 0, "n_layers": 0,
        "n_classes": int(n_classes), "n_ch": int(n_ch), "n_times": int(n_times),
        "batch": int(batch), "model_repr": "",
    }

    # ---- 1) 执行代码 ----
    try:
        ns = _exec_code(code)
    except SyntaxError as e:
        res["errors"].append(f"语法错误（第 {e.lineno} 行）：{e.msg}")
        return res
    except Exception as e:  # noqa: BLE001
        res["errors"].append(f"代码执行失败：{type(e).__name__}: {e}")
        return res

    classes = model_classes(ns)
    res["classes"] = [c.__name__ for c in classes]
    if not classes:
        res["errors"].append(
            "没有找到 nn.Module 子类。请在文件里定义 class 你的模型(nn.Module)…，"
            "并用 __model__ = \"类名\" 指明（不写则取最后一个类）。")
        return res
    try:
        cls = pick_class(ns)
    except ValueError as e:
        res["errors"].append(str(e))
        return res
    res["target_class"] = cls.__name__
    res["notes"].append(f"对外模型类：{cls.__name__}"
                        + (f"（文件里共 {len(classes)} 个类）" if len(classes) > 1 else ""))

    # ---- 2) 实例化 ----
    try:
        model, how = instantiate(cls, n_classes, n_ch, n_times, params)
    except Exception as e:  # noqa: BLE001
        res["errors"].append(f"实例化失败：{e}")
        return res
    res["notes"].append(f"实例化方式：{how}")
    res["model_repr"] = str(model)[:2000]

    # ---- 3) 前向 + 逐层形状（hook 记录特征流向）----
    steps: list[dict] = []

    def make_hook(name: str):
        def _hook(mod, inp, out):
            in_shape = (tuple(inp[0].shape) if inp and isinstance(inp[0], torch.Tensor)
                        else None)
            out_shape = (tuple(out.shape) if isinstance(out, torch.Tensor)
                         else type(out).__name__)
            steps.append({
                "i": len(steps) + 1, "name": name, "type": type(mod).__name__,
                "in": in_shape, "out": out_shape,
                "params": sum(p.numel() for p in mod.parameters()),
            })
        return _hook

    handles = [mod.register_forward_hook(make_hook(name))
               for name, mod in _leaf_modules(model)]

    model.eval()
    y = None
    tried: list[str] = []
    with torch.no_grad():
        for shape in ((batch, 1, n_ch, n_times), (batch, n_ch, n_times)):
            tried.append(str(shape))
            steps.clear()
            try:
                x = torch.randn(*shape)
                y = model(x)
                res["input_shape"] = list(shape)
                break
            except Exception as e:  # noqa: BLE001
                y, last = None, e
        if y is None:
            for h in handles:
                h.remove()
            msg = str(last)
            hint = ""
            if "shapes cannot be multiplied" in msg or "mat1 and mat2" in msg:
                # 用户最常踩的坑：加了新 block，但分类头 Linear 的输入维数没同步改
                hint = ("\n→ 分类头与特征维数不一致：改了 block（如插入金字塔/换池化）后，"
                        "必须同步改 nn.Linear 的 in_features。最省事的写法是让它自动探测：\n"
                        "    feat = self.block2(self.block1(torch.zeros(1, 1, n_ch, n_times)))"
                        ".flatten(1).shape[1]\n"
                        "    self.classifier = nn.Linear(feat, n_classes)")
            elif "Calculated output size" in msg or "output size is too small" in msg:
                hint = "\n→ 中间特征被池化/卷积压成 0 维：请增大 n_times 或减小 kernel/池化核。"
            res["errors"].append(
                "前向传播失败（已尝试输入形状："
                + "、".join(tried) + f"）：{type(last).__name__}: {msg}{hint}")
            return res
    for h in handles:
        h.remove()

    res["steps"] = steps
    res["n_layers"] = len(steps)
    res["n_params"] = sum(p.numel() for p in model.parameters())
    res["output_shape"] = (list(y.shape) if isinstance(y, torch.Tensor) else None)

    # ---- 4) 输出维度 / 数值健康 ----
    if not isinstance(y, torch.Tensor):
        res["errors"].append(f"forward 返回值不是 torch.Tensor，而是 {type(y).__name__}")
        return res
    if tuple(y.shape) != (batch, n_classes):
        hint = (f"期望输出 (batch={batch}, n_classes={n_classes})，实际 {tuple(y.shape)}。"
                "常见原因：分类头 Linear 的输入维度与卷积输出不匹配（改了 block 就要"
                "同步改 Linear 的 in_features），或忘记 flatten(1)。")
        res["errors"].append(f"输出维度不对：{hint}")
    if not torch.isfinite(y).all():
        res["errors"].append("输出含 NaN / INF：检查是否出现除零、log(0) 或尺度爆炸")
    zeros = [s["name"] for s in steps
             if isinstance(s["out"], tuple) and 0 in s["out"]]
    if zeros:
        res["errors"].append("中间特征出现 0 维（该层会把特征「抽干」）："
                             + "、".join(zeros[:6])
                             + "。常见原因：池化核大于当前特征长度，需减小 kernel 或去掉池化。")

    # ---- 5) 反向：可训练性 + 悬空参数 ----
    model.zero_grad(set_to_none=True)
    try:
        out = model(torch.randn(batch, 1, n_ch, n_times)
                    if res["input_shape"][1] == 1 else torch.randn(batch, n_ch, n_times))
        out.sum().backward()
        dead = [n for n, p in model.named_parameters()
                if p.requires_grad and p.grad is None]
        if dead:
            res["warnings"].append(
                "以下参数没有梯度（未参与前向传播，等于白占显存）："
                + "、".join(dead[:8]) + ("…" if len(dead) > 8 else ""))
    except Exception as e:  # noqa: BLE001
        res["errors"].append(f"反向传播失败（网络不可训练）：{type(e).__name__}: {e}")

    # ---- 6) 小结 ----
    res["ok"] = not res["errors"]
    if res["ok"]:
        res["notes"].append(
            f"参数量 {res['n_params']:,}；逐层核对 {len(steps)} 层："
            f"输入 {res['input_shape']} → 输出 {res['output_shape']}，"
            f"目标类别数 {n_classes} 一致，反传正常。")
    return res
