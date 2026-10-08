"""论文超参抽取：可选的 LLM 接口（OpenAI 兼容 / Anthropic / 本地 Ollama）+ 规则兜底。

用法::

    from core.llm import extract_train_params, to_train_config

    res = extract_train_params(paper_text)          # 没配 key 也能跑（走规则）
    cfg = to_train_config(res["fields"])            # → 可直接喂给训练页/训练配置

设计原则
--------
1. **不硬编码密钥**：优先环境变量（``OPENAI_API_KEY`` / ``DEEPSEEK_API_KEY`` …），
   其次本地 ``configs/llm.yaml``（已在 .gitignore 里）。密钥只在内存/本机文件里流转。
2. **离线可用**：不配置 LLM 时用中英文正则规则抽取（同样给出"依据片段"），
   LLM 只是"理解力更强"的增强项；网络失败/超时/返回脏 JSON 都自动降级到规则。
3. **可核对**：每个字段都带 ``evidence``（原文片段）、``source``（LLM / 规则 / 两者）
   与冲突标记 —— 论文里的超参错了代价很大，必须让人一眼核对，而不是只给一个数字。
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field as dc_field
from pathlib import Path

from . import PROJECT_ROOT

#: LLM 供应商注册表（style: openai=OpenAI 兼容 /chat/completions，anthropic=/messages）
PROVIDERS: dict[str, dict] = {
    "deepseek": {
        "label": "DeepSeek（深度求索）", "style": "openai",
        "base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat",
        "env": "DEEPSEEK_API_KEY", "json_mode": True,
    },
    "openai": {
        "label": "OpenAI", "style": "openai",
        "base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini",
        "env": "OPENAI_API_KEY", "json_mode": True,
    },
    "dashscope": {
        "label": "阿里云百炼（Qwen）", "style": "openai",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus", "env": "DASHSCOPE_API_KEY", "json_mode": True,
    },
    "moonshot": {
        "label": "Moonshot Kimi", "style": "openai",
        "base_url": "https://api.moonshot.cn/v1", "model": "moonshot-v1-8k",
        "env": "MOONSHOT_API_KEY", "json_mode": True,
    },
    "zhipu": {
        "label": "智谱 GLM", "style": "openai",
        "base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4-flash",
        "env": "ZHIPU_API_KEY", "json_mode": False,
    },
    "siliconflow": {
        "label": "硅基流动 SiliconFlow", "style": "openai",
        "base_url": "https://api.siliconflow.cn/v1", "model": "Qwen/Qwen2.5-7B-Instruct",
        "env": "SILICONFLOW_API_KEY", "json_mode": True,
    },
    "anthropic": {
        "label": "Anthropic Claude", "style": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-3-5-sonnet-latest", "env": "ANTHROPIC_API_KEY",
        "json_mode": False,
    },
    "ollama": {
        "label": "本地 Ollama（无需密钥）", "style": "openai",
        "base_url": "http://localhost:11434/v1", "model": "qwen2.5:7b",
        "env": "", "json_mode": False,
    },
    "custom": {
        "label": "自定义（OpenAI 兼容端点）", "style": "openai",
        "base_url": "", "model": "", "env": "BCI_LLM_API_KEY", "json_mode": True,
    },
}

#: 本地密钥文件（放 configs/ 下，已在 .gitignore 中；不要提交/分享）
CONFIG_PATH = PROJECT_ROOT / "configs" / "llm.yaml"


# ==================================================================== 配置
@dataclass
class LLMConfig:
    """一次调用的全部参数（provider 决定 base_url/model 默认值）。"""

    provider: str = "deepseek"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 1200
    timeout: float = 60.0
    json_mode: bool | None = None       # None → 用供应商默认
    max_chars: int = 12000              # 论文文本截断长度

    @property
    def spec(self) -> dict:
        return PROVIDERS.get(self.provider, PROVIDERS["custom"])

    def resolved(self) -> "LLMConfig":
        """补全 base_url / model / json_mode（未填则用供应商默认）。"""
        spec = self.spec
        out = LLMConfig(**asdict(self))
        out.base_url = (self.base_url or spec["base_url"]).rstrip("/")
        out.model = self.model or spec["model"]
        out.temperature = float(self.temperature)
        out.timeout = float(self.timeout)
        if out.json_mode is None:
            out.json_mode = bool(spec.get("json_mode", False))
        return out

    @property
    def needs_key(self) -> bool:
        return bool(self.spec.get("env"))

    @property
    def configured(self) -> bool:
        """能否真正发起调用（本地 Ollama / 自定义端点可不填密钥）。"""
        r = self.resolved()
        return bool(r.base_url and r.model and (r.api_key or not r.needs_key))

    def describe(self) -> str:
        r = self.resolved()
        return f"{self.spec['label']} · {r.model or '(未填模型)'} · {r.base_url or '(未填地址)'}"


def default_config() -> LLMConfig:
    """按「环境变量 → 本地 configs/llm.yaml → 默认」的顺序给出可用配置。"""
    saved = load_config_file()
    provider = str(saved.get("provider") or os.environ.get("BCI_LLM_PROVIDER") or "deepseek")
    cfg = LLMConfig(
        provider=provider,
        model=str(saved.get("model") or ""),
        base_url=str(saved.get("base_url") or ""),
        api_key=str(saved.get("api_key") or ""),
        temperature=float(saved.get("temperature", 0.0)),
        max_tokens=int(saved.get("max_tokens", 1200)),
        timeout=float(saved.get("timeout", 60.0)),
        json_mode=saved.get("json_mode"),
        max_chars=int(saved.get("max_chars", 12000)),
    )
    if not cfg.api_key:
        cfg.api_key = api_key_from_env(provider)
    return cfg.resolved()


def api_key_from_env(provider: str) -> str:
    """从环境变量取密钥（供应商专属变量 → 通用 BCI_LLM_API_KEY）。"""
    env = PROVIDERS.get(provider, {}).get("env") or ""
    for name in (env, "BCI_LLM_API_KEY"):
        if name and os.environ.get(name):
            return str(os.environ[name])
    return ""


def load_config_file(path: str | Path | None = None) -> dict:
    p = Path(path) if path else CONFIG_PATH
    if not p.exists():
        return {}
    try:
        import yaml

        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 - 配置坏了不该让程序起不来
        return {}


def save_config_file(cfg: LLMConfig, path: str | Path | None = None) -> Path:
    """把配置（**含密钥**）写到本地文件，并确保它有 .gitignore 保护。"""
    import yaml

    p = Path(path) if path else CONFIG_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in asdict(cfg).items() if v not in ("", None)}
    p.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
                 encoding="utf-8")
    try:                                    # 顺手保证密钥文件不会被误提交
        gi = PROJECT_ROOT / ".gitignore"
        text = gi.read_text(encoding="utf-8") if gi.exists() else ""
        if "configs/llm.yaml" not in text:
            gi.write_text(text.rstrip("\n") + "\nconfigs/llm.yaml\n", encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return p


# ================================================================ HTTP 调用
def _post_json(url: str, headers: dict, payload: dict, timeout: float) -> dict:
    """POST JSON（单独抽出来，便于测试里替换掉真实网络请求）。"""
    import requests

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code >= 400:
        detail = resp.text[:300]
        raise RuntimeError(f"HTTP {resp.status_code}：{detail}")
    return resp.json()


def chat(cfg: LLMConfig, system: str, user: str, log=print) -> str:
    """一次对话调用，返回模型输出的文本（自动按供应商拼 URL/头/负载）。"""
    import requests  # noqa: F401 - 提前报错提示缺依赖

    r = cfg.resolved()
    if not r.base_url or not r.model:
        raise ValueError("LLM 未配置完整：请填写 base_url 与 model")
    if r.needs_key and not r.api_key:
        env = r.spec.get("env") or "BCI_LLM_API_KEY"
        raise ValueError(
            f"缺少 API 密钥：请设置环境变量 {env}（或写入 {CONFIG_PATH}）——"
            f"也可以直接不配置 LLM，用内置规则抽取")

    if r.spec["style"] == "anthropic":
        url = f"{r.base_url}/messages"
        headers = {"x-api-key": r.api_key, "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        payload = {"model": r.model, "max_tokens": int(r.max_tokens),
                   "temperature": float(r.temperature), "system": system,
                   "messages": [{"role": "user", "content": user}]}
        data = _post_json(url, headers, payload, r.timeout)
        parts = data.get("content") or []
        text = "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict))
        return text

    url = f"{r.base_url}/chat/completions"
    headers = {"content-type": "application/json"}
    if r.api_key:
        headers["Authorization"] = f"Bearer {r.api_key}"
    payload = {"model": r.model, "temperature": float(r.temperature),
               "max_tokens": int(r.max_tokens),
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}]}
    if r.json_mode:
        payload["response_format"] = {"type": "json_object"}
    data = _post_json(url, headers, payload, r.timeout)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"返回体里没有 choices：{str(data)[:200]}")
    return str((choices[0].get("message") or {}).get("content") or "")


def test_connection(cfg: LLMConfig, log=print) -> str:
    """轻量连通性测试：让模型回一个固定 JSON。"""
    t0 = time.perf_counter()
    text = chat(cfg, "你是测试助手，只输出 JSON。",
                '只输出这个 JSON，不要多写任何字：{"ok": true, "msg": "pong"}')
    dt = time.perf_counter() - t0
    got = _parse_json(text)
    ok = bool(isinstance(got, dict) and (got.get("ok") is True))
    return f"{'✔ 连接成功' if ok else '△ 有响应但格式不符'}（{dt:.1f}s）：{text.strip()[:120]}"


# ============================================================ 字段定义
@dataclass
class Field:
    key: str
    label: str
    kind: str                       # int | float | str | bool | list
    path: tuple[str, ...]           # 写进训练配置的位置
    options: tuple[str, ...] = ()
    hint: str = ""


#: 抽取字段（顺序 = 界面/表格顺序）。path 与训练配置一一对应，可直接套用。
FIELDS: tuple[Field, ...] = (
    Field("epochs", "训练轮数 epochs", "int", ("trainer", "epochs"), hint="整数"),
    Field("batch_size", "批大小 batch size", "int", ("trainer", "batch_size")),
    Field("lr", "学习率 learning rate", "float", ("trainer", "lr"), hint="如 0.001 / 1e-3"),
    Field("optimizer", "优化器 optimizer", "str", ("trainer", "optimizer"),
          options=("adamw", "adam", "sgd", "rmsprop", "adagrad", "lamb")),
    Field("weight_decay", "权重衰减 weight decay", "float", ("trainer", "weight_decay")),
    Field("scheduler", "学习率调度 scheduler", "str", ("trainer", "scheduler"),
          options=("cosine", "step", "none", "plateau", "onecycle")),
    Field("seed", "随机种子 seed", "int", ("trainer", "seed")),
    Field("normalization", "标准化 normalization", "str", ("normalization", "mode"),
          options=("zscore_channel", "zscore_epoch", "minmax_channel", "minmax_pm",
                   "p95", "x_div", "iqr", "quantile", "ema", "none")),
    Field("euclidean_alignment", "欧几里得对齐 EA", "bool", ("normalization", "euclidean_alignment")),
    Field("augmentation", "是否用数据增强", "bool", ("augmentation", "enabled")),
    Field("augmentation_types", "数据增强方式", "list", ("augmentation", "types")),
    Field("split_scheme", "划分方案", "str", ("split", "scheme"),
          options=("session", "loso", "kfold", "within", "few_shot", "holdout")),
)

FIELD_BY_KEY = {f.key: f for f in FIELDS}

#: 标准化说法 → 本平台归一化模式
NORM_SYNONYMS: tuple[tuple[str, str], ...] = (
    (r"instance\s*norm|per[\s-]?(?:trial|epoch)\s*(?:norm|standardiz)|逐\s*(?:trial|试次|epoch)"
     r"(?:归一化|标准化)|每个?\s*(?:trial|试次|epoch)独立", "zscore_epoch"),
    (r"exponential\s*moving\s*average|\bEMA\b|滑动(?:平均|标准化)|在线标准化", "ema"),
    (r"quantile|robust\s*scal|分位数|稳健标准化", "quantile"),
    (r"z[\s-]*score|zscore|standardiz|零均值|单位方差|标准化|正态化", "zscore_channel"),
    (r"min[\s-]*max|(?:0\s*[,，]\s*1)|(?:-\s*1\s*[,，]\s*1)|归一化到|最大最小", "minmax_channel"),
    (r"p95|95\s*(?:th)?\s*百分位", "p95"),
    (r"除以|devide?\s*by|/\s*1000|尺度因子", "x_div"),
    (r"\bIQR\b|四分位", "iqr"),
    (r"no\s*(?:normaliz|standardiz)|without\s*(?:normaliz|standardiz)|未(?:做)?(?:标准化|归一化)|"
     r"不做(?:标准化|归一化)|raw\s*(?:EEG|signal|data)", "none"),
)

#: 数据增强类型关键词
AUG_TYPES: tuple[tuple[str, str], ...] = (
    (r"time\s*shift|时间平移|时间偏移|shift|循环移位", "time_shift"),
    (r"gauss\s*noise|gaussian\s*noise|additive\s*noise|\bnoise\b|噪声|加噪", "gaussian_noise"),
    (r"channel\s*drop|drop\s*(?:out)?\s*channel|通道(?:丢弃|随机)|电极丢弃", "channel_dropout"),
    (r"mix\s*up|mixup|混叠", "mixup"),
    (r"crop|切(?:片|割)|裁剪|random\s*crop", "crop"),
    (r"scal(?:e|ing)|幅值(?:缩放|扰动)|magnitude", "scaling"),
    (r"frequency\s*mask|band\s*mask|频(?:谱|带)(?:掩码|遮挡)", "freq_mask"),
    (r"sliding\s*window|滑动窗|窗口切分|data\s*segment", "sliding_window"),
)

#: 论文里的增强手段 → 本平台的增强参数（缺省强度，用户可在界面再调）
AUG_TYPE_TO_PARAM: dict[str, tuple[str, float]] = {
    "time_shift": ("time_shift", 25.0),         # 采样点（250 Hz 下 ≈ 0.1 s）
    "gaussian_noise": ("gaussian_noise", 0.1),  # 相对信号标准差
    "channel_dropout": ("channel_dropout", 0.1),
}

#: 本平台训练器暂不支持的手段（会写进配置并提示，不静默丢弃）
AUG_UNSUPPORTED: tuple[str, ...] = ("mixup", "crop", "scaling", "freq_mask", "sliding_window")

#: 划分方案关键词
SPLIT_SYNONYMS: tuple[tuple[str, str], ...] = (
    (r"leave[\s-]*one[\s-]*subject[\s-]*out|\bLOSO\b|留一被试|跨被试(?:验证)?", "loso"),
    (r"session[\s-]*to[\s-]*session|跨\s*session|训练\s*session|session\s*1\s*to\s*2", "session"),
    (r"few[\s-]*shot|少样本|小样本", "few_shot"),
    (r"\b(?:\d+)?[\s-]*fold(?:s)?\b|k\s*折|交叉验证", "kfold"),
    (r"hold[\s-]*out|留出", "holdout"),
    (r"within[\s-]*subject|被试内", "within"),
)

#: 否定词（用于识别"未使用数据增强/未做 EA"）
NEG_CUES = re.compile(
    r"(?:\b(?:no|not|without|never|none|didn'?t|did\s+not|free\s+of)\b|未|没有|不|无|免)", re.I)

_NUM = r"[-+]?\d+(?:\.\d+)?(?:\s*[eE][-+]?\d+)?"
#: 数值的完整写法：1e-3 / 0.001 / 1×10^-3 / 1×10−3 / 10-3
#: 注意把"带 ×10 的写法"放在前面：否则 "1×10-3" 会被 `_NUM` 先匹配成 "1"
_NUM_FULL = rf"(?:\d+(?:\.\d+)?\s*[x×*]\s*10\s*[\^\-−–]?\s*\d+|{_NUM})"
#: 宽松连接词里不该跨越的"别的参数名"（防止 lr 把 batch size 的数字抓走）
_OTHER_PARAMS = re.compile(
    r"batch|mini|epoch|seed|decay|dropout|depth|filter|channel|size|"
    r"轮|批|种子|衰减|通道", re.I)


def _num(text: str) -> float | None:
    """把 1e-3 / 1×10^-3 / 1×10−³ / 0.001 这类写法统一成浮点数。"""
    s = str(text).strip().replace(" ", "")
    s = s.replace("×", "x").replace("*", "x").replace("−", "-").replace("–", "-")
    m = re.match(rf"^({_NUM})$", s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    m = re.match(r"^(\d+(?:\.\d+)?)x10\^?\(?-?(\d+)\)?$", s, re.I)
    if m:
        return float(m.group(1)) * (10.0 ** -int(m.group(2)))
    m = re.match(r"^10\^?-?(\d+)$", s)
    if m:
        return 10.0 ** -int(m.group(1))
    # 上标数字：10⁻³ / 10-3
    sup = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
    s2 = s.translate(sup)
    m = re.match(r"^(\d+(?:\.\d+)?)x10\(?-?(\d+)\)?$", s2, re.I)
    if m:
        return float(m.group(1)) * (10.0 ** -int(m.group(2)))
    return None


def _snippet(text: str, start: int, end: int, width: int = 46) -> str:
    a = max(0, start - width)
    b = min(len(text), end + width)
    return (("…" if a > 0 else "") + re.sub(r"\s+", " ", text[a:b]).strip()
            + ("…" if b < len(text) else ""))


def _negated(text: str, pos: int, window: int = 34) -> bool:
    """关键词前一小段里出现否定词 → 认为是否定表述。"""
    return bool(NEG_CUES.search(text[max(0, pos - window):pos]))


def _find(text: str, pattern: str, group: int = 1):
    m = re.search(pattern, text, re.I)
    return (m.group(group), m) if m else (None, None)


# ======================================================== 规则抽取（离线可用）
def extract_by_rules(text: str) -> dict[str, dict]:
    """中英文正则抽取（不联网）。返回 ``{字段: {value, evidence, source}}``。"""
    text = str(text or "")
    flat = re.sub(r"[ \t]+", " ", text)
    out: dict[str, dict] = {}

    def put(key: str, value, ev: str, why: str = ""):
        if value is None:
            return
        if isinstance(value, (int, float)) and isinstance(value, float) and value != value:
            return
        f = FIELD_BY_KEY[key]
        if f.options and isinstance(value, str) and value not in f.options:
            return
        cur = out.get(key)
        if cur is None or (cur.get("_rank", 0) < 1):
            out[key] = {"value": value, "evidence": ev, "source": "规则",
                        "source_detail": why or "正则匹配", "_rank": 1}

    # ---- epochs：数字既可能在词后（epochs = 200），也可能在词前（for 200 epochs）----
    for pat in (r"(?:epochs?|max[_\s-]*epochs?|num[_\s-]*epochs?)\s*(?:=|:|of|为|是|:)\s*(\d{1,4})",
                r"(?:epochs?|max[_\s-]*epochs?|num[_\s-]*epochs?)\s+(?:is|was|are)?\s*(\d{1,4})\b",
                r"(\d{1,4})\s*(?:个)?\s*(?:training\s+|训练\s*)?epochs?\b",
                r"(?:训练|迭代)\s*(?:轮数|轮次|次数|数)?\s*(?:=|:|为|是)?\s*(\d{1,4})\s*(?:个)?"
                r"\s*(?:epochs?|轮|次)",
                r"trained\s+(?:\w+\s+){0,3}?for\s+(\d{1,4})\s+epochs?"):
        v, m = _find(flat, pat)
        if v:
            put("epochs", int(v), _snippet(flat, m.start(), m.end()), "epochs 关键词")
            break
    # ---- batch size：连接词可能有好几个（"mini-batch size was set to 64"）----
    for pat in (r"(?:batch[\s_-]*size|mini[\s-]*batch\s*size|batch)\s*"
                r"(?:=|:|of|為|为|是|is|was|were|used|set\s+to|设为|设置为|取)?[^\d\n]{0,14}?(\d{1,5})",
                r"(?:批(?:次)?大小|批量大小|批大小|批次规模)\s*"
                r"(?:=|:|为|是|取|设为|设置为)?[^\d\n]{0,8}?(\d{1,5})",
                r"\bBS\s*=\s*(\d{1,5})"):
        v, m = _find(flat, pat)
        if v:
            put("batch_size", int(v), _snippet(flat, m.start(), m.end()), "batch size 关键词")
            break
    # ---- learning rate：先严格（lr = …），再宽松（"设为/ was set to" 等连接词）----
    lr_kw = r"(?:learning[_\s-]*rate|initial[_\s-]*lr|\blr\b|学习率|初始学习率)"
    for pat, why in ((rf"{lr_kw}\s*(?:=|:|is|was|of|为|是)\s*({_NUM_FULL})", "严格匹配"),
                     (rf"{lr_kw}\s*(?:设为|设置为|取|用|set\s+to|used|took)?[^\d\n]{{0,12}}?"
                      rf"({_NUM_FULL})", "连接词匹配")):
        v, m = _find(flat, pat)
        if not v:
            continue
        span = flat[m.start():m.end()]
        if "连接词匹配" == why and _OTHER_PARAMS.search(span.replace(v, "", 1)):
            continue                        # 抓到了别的参数的数字 → 放弃这次匹配
        val = _num(v)
        if val is not None and 0 < val <= 1.0:
            put("lr", val, _snippet(flat, m.start(), m.end()), f"learning rate（{why}）")
            break
    # ---- optimizer ----
    v, m = _find(flat, r"\b(AdamW|Adam|SGD|RMSprop|Adagrad|LAMB|Lookahead|Adadelta)\b")
    if not v:
        v, m = _find(flat, r"(AdamW|Adam|SGD|随机梯度下降|自适应矩估计|动量法)", 1)
    if v:
        name = str(v).lower()
        name = {"随机梯度下降": "sgd", "自适应矩估计": "adam", "动量法": "sgd",
                "adamw": "adamw", "adam": "adam", "sgd": "sgd", "rmsprop": "rmsprop",
                "adagrad": "adagrad", "lamb": "lamb", "lookahead": "adamw",
                "adadelta": "adadelta"}.get(name, name)
        put("optimizer", name, _snippet(flat, m.start(), m.end()), "优化器关键词")
    # ---- weight decay ----
    for pat in (rf"(?:weight[_\s-]*decay|L2\s*(?:正则|regulariz\w*)|权重衰减)\s*"
                rf"(?:=|:|of|为|是)?\s*({_NUM_FULL})",):
        v, m = _find(flat, pat)
        val = _num(v) if v else None
        if val is not None and 0 <= val <= 1:
            put("weight_decay", val, _snippet(flat, m.start(), m.end()), "weight decay 关键词")
            break
    # ---- scheduler ----
    v, m = _find(flat, r"(cosine(?:\s*anneal\w*)?|余弦退火|one[\s-]*cycle|step\s*(?:lr|decay)|"
                      r"阶梯(?:衰减)?|plateau|ReduceLROnPlateau|linear\s*warm[\s-]*up|"
                      r"warm[\s-]*up|常数学习率|constant\s*lr|无调度)")
    if v:
        s = str(v).lower()
        sch = ("cosine" if "cosine" in s or "余弦" in s else
               "onecycle" if "cycle" in s else
               "step" if "step" in s or "阶梯" in s else
               "plateau" if "plateau" in s else
               "none" if "constant" in s or "常数" in s or "无调度" in s else "none")
        put("scheduler", sch, _snippet(flat, m.start(), m.end()), "调度器关键词")
    # ---- seed："random seed was fixed to 42" 这类连接词很长，需宽松匹配 ----
    seed_kw = r"(?:random[_\s-]*seed|rand[_\s-]*seed|\bseed\b|随机种子|种子)"
    seed_num = r"(\d{1,6})(?!\s*(?:[eE][-+]?\d|\.\d|[x×*]\s*10))"   # 别把 1e-3 的 1 当种子
    for pat, why in ((rf"{seed_kw}\s*(?:=|:|is|was|of|为|是)\s*{seed_num}", "严格匹配"),
                     (rf"{seed_kw}\s*(?:set\s+to|fixed\s+to|was|is|values?|设为|设置为|取|用)?"
                      rf"[^\d\n]{{0,16}}?{seed_num}", "连接词匹配")):
        v, m = _find(flat, pat)
        if not v:
            continue
        span = flat[m.start():m.end()].replace(v, "", 1)
        if any(ch.isdigit() for ch in span):     # 连接词里还夹着别的数字 → 多半抓错，放弃
            continue
        put("seed", int(v), _snippet(flat, m.start(), m.end()), f"随机种子（{why}）")
        break
    # ---- normalization ----
    for pat, mode in NORM_SYNONYMS:
        m = re.search(pat, flat, re.I)
        if m and not (mode != "none" and _negated(flat, m.start())):
            if mode == "none" and not _negated(flat, m.start()) and \
                    not re.search(r"no|未|without", m.group(0), re.I):
                continue
            put("normalization", mode, _snippet(flat, m.start(), m.end()), "标准化说法匹配")
            break
    # ---- euclidean alignment ----
    m = re.search(r"euclidean\s*alignment|euclidean\s*align|\bEA\b|欧(?:几里得|氏)对齐", flat, re.I)
    if m:
        neg = _negated(flat, m.start())
        put("euclidean_alignment", not neg, _snippet(flat, m.start(), m.end()),
            "未使用 EA" if neg else "提到欧几里得对齐")
    else:
        m2 = re.search(r"(?:with|using|apply(?:ing)?)\s+\bEA\b", flat, re.I)
        if m2:
            put("euclidean_alignment", True, _snippet(flat, m2.start(), m2.end()), "使用 EA")
    # ---- data augmentation ----
    m = re.search(r"data\s*augment\w*|augment(?:ation|ed|ing)?|数据增强|增广|"
                  r"augmented\s+(?:data|training)", flat, re.I)
    types, type_hits = [], []
    for pat, tag in AUG_TYPES:
        mm = re.search(pat, flat, re.I)
        if mm and not _negated(flat, mm.start()):
            types.append(tag)
            type_hits.append(mm)
    if m:
        neg = _negated(flat, m.start())
        if neg:
            put("augmentation", False, _snippet(flat, m.start(), m.end()), "否定表述")
        else:
            put("augmentation", True, _snippet(flat, m.start(), m.end()), "提到数据增强")
    elif types:
        # 没写"augmentation"这个词，但列了具体增强手段 → 也算用了增强
        idx = min(re.search(p, flat, re.I).start() for p, t in AUG_TYPES if t in types)
        put("augmentation", True, _snippet(flat, idx, idx + 40), "列出了具体增强手段")
    if types:
        first = min(type_hits, key=lambda x: x.start())
        put("augmentation_types", sorted(set(types)),
            _snippet(flat, first.start(), first.end()) + "（含 "
            + "、".join(sorted(set(types))) + "）", "增强手段关键词")
    # ---- split scheme ----
    for pat, scheme in SPLIT_SYNONYMS:
        m = re.search(pat, flat, re.I)
        if m:
            put("split_scheme", scheme, _snippet(flat, m.start(), m.end()), "划分方案关键词")
            break
    for v in out.values():
        v.pop("_rank", None)
    return out


# ==================================================== LLM 抽取（可选增强）
SYSTEM_PROMPT = (
    "你是 EEG/脑机接口论文的方法学信息抽取助手。用户会给一段论文方法（可能是中英文），"
    "你要抽取训练超参，**只输出 JSON**，不要解释、不要 Markdown 代码块。"
    "找不到的字段值填 null，绝不要瞎猜或套用常见默认值。"
)


def build_prompt(text: str) -> str:
    """把字段 schema 拼进提示词（要求模型给出依据片段与置信度）。"""
    lines = [
        "请从下面的论文方法段落中抽取训练/评测设置，按这个 JSON 结构输出：",
        "{",
        '  "epochs": {"value": <int|null>, "evidence": "<原文片段>", "confidence": <0-1>},',
        '  "batch_size": {"value": <int|null>, "evidence": "...", "confidence": <0-1>},',
        '  "lr": {"value": <float|null>, "evidence": "...", "confidence": <0-1>},',
        '  "optimizer": {"value": "adamw|adam|sgd|rmsprop|adagrad|lamb|other|null", ...},',
        '  "weight_decay": {"value": <float|null>, ...},',
        '  "scheduler": {"value": "cosine|step|none|plateau|onecycle|null", ...},',
        '  "seed": {"value": <int|null>, ...},',
        '  "normalization": {"value": "zscore_channel|zscore_epoch|minmax_channel|'
        'minmax_pm|p95|x_div|iqr|quantile|ema|none|null", ...},',
        '  "euclidean_alignment": {"value": <true|false|null>, ...},',
        '  "augmentation": {"value": <true|false|null>, ...},',
        '  "augmentation_types": {"value": ["time_shift", "gaussian_noise", '
        '"channel_dropout", "mixup", "crop", "scaling", "freq_mask", "sliding_window"], ...},',
        '  "split_scheme": {"value": "session|loso|kfold|within|few_shot|holdout|null", ...}',
        "}",
        "",
        "规则：",
        "1. value 必须是原文明确写到的；没写就 null（不要用领域常识补），"
        "例如原文只说“训练 100 个 epoch”，就别填 batch_size。",
        "2. evidence 必须是原文里支持该值的**连续片段**（可截断），不要改写。",
        "3. 注意否定与限定语：如 without data augmentation / 未使用欧几里得对齐 → false。",
        "4. 学习率写成 1e-3 / 1×10⁻³ / 0.001 都要转成小数 0.001。",
        "5. normalization 要映射到给定取值：z-score→zscore_channel、逐 trial/instance norm→"
        "zscore_epoch、min-max→minmax_channel、未做标准化→none。",
        "",
        "论文正文：",
        "-----",
        text,
        "-----",
    ]
    return "\n".join(lines)


def _parse_json(text: str) -> dict | None:
    """从模型输出里稳健地抠出 JSON（容忍 ```json 围栏、前后废话、单引号）。"""
    if not text:
        return None
    s = str(text).strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.M).strip()
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        pass
    start, end = s.find("{"), s.rfind("}")
    if start >= 0 and end > start:
        chunk = s[start:end + 1]
        for cand in (chunk, re.sub(r"'", '"', chunk),
                     re.sub(r",\s*([}\]])", r"\1", chunk)):
            try:
                return json.loads(cand)
            except Exception:  # noqa: BLE001
                continue
    return None


def _coerce(key: str, raw) -> tuple:
    """把模型给的 value 规整到字段类型/可选值；返回 (值, 备注)。"""
    f = FIELD_BY_KEY.get(key)
    if f is None or raw is None:
        return None, ""
    if isinstance(raw, dict):                      # {"value": ..., "evidence": ...}
        raw = raw.get("value")
        if raw is None:
            return None, ""
    try:
        if f.kind == "int":
            v = int(float(str(raw).strip()))
            return (v, "") if v > 0 else (None, "非正数已忽略")
        if f.kind == "float":
            v = _num(raw) if not isinstance(raw, (int, float)) else float(raw)
            if v is None:
                return None, "无法解析为数字"
            return (v, "") if v >= 0 else (None, "负值已忽略")
        if f.kind == "bool":
            if isinstance(raw, bool):
                return raw, ""
            s = str(raw).strip().lower()
            if s in ("true", "yes", "1", "是", "有", "用了"):
                return True, ""
            if s in ("false", "no", "0", "否", "没有", "未使用"):
                return False, ""
            return None, f"无法判断真假：{raw}"
        if f.kind == "list":
            items = raw if isinstance(raw, (list, tuple)) else \
                [x for x in re.split(r"[,，;；、/\s]+", str(raw)) if x]
            allowed = {"time_shift", "gaussian_noise", "channel_dropout", "mixup",
                       "crop", "scaling", "freq_mask", "sliding_window"}
            keep = [str(x).strip() for x in items if str(x).strip() in allowed]
            return (keep, "") if keep else (None, f"取值不在允许集合：{items}")
        s = str(raw).strip().lower()
        if f.options:
            alias = {"adam": "adam", "adamw": "adamw", "sgd": "sgd", "rmsprop": "rmsprop",
                     "adagrad": "adagrad", "lamb": "lamb", "随机梯度下降": "sgd",
                     "自适应矩估计": "adam", "none": "none", "无": "none",
                     "cosine": "cosine", "余弦": "cosine", "step": "step",
                     "plateau": "plateau", "onecycle": "onecycle"}
            s = alias.get(s, s)
            if s in f.options:
                return s, ""
            for opt in f.options:                  # 子串兜底（如 "adamw optimizer"）
                if opt in s:
                    return opt, ""
            return None, f"取值不在允许集合：{raw}"
        return s, ""
    except Exception as e:  # noqa: BLE001
        return None, f"类型转换失败：{type(e).__name__}"


def extract_by_llm(text: str, cfg: LLMConfig | None = None, log=print) -> dict[str, dict]:
    """调 LLM 抽取。失败会抛异常，由 :func:`extract_train_params` 决定是否降级。"""
    cfg = (cfg or default_config()).resolved()
    payload = text[: int(cfg.max_chars)]
    if len(text) > len(payload):
        log(f"[LLM] 论文文本较长，已截断到 {len(payload)} 字（可用 max_chars 调整）")
    t0 = time.perf_counter()
    raw = chat(cfg, SYSTEM_PROMPT, build_prompt(payload), log=log)
    log(f"[LLM] {cfg.describe()} 返回 {len(raw)} 字，耗时 {time.perf_counter() - t0:.1f}s")
    data = _parse_json(raw)
    if data is None:
        raise RuntimeError(f"模型输出不是合法 JSON：{raw[:200]}")
    out: dict[str, dict] = {}
    for key in FIELD_BY_KEY:
        item = data.get(key)
        if item is None:
            continue
        value, note = _coerce(key, item)
        if value is None:
            if note:
                log(f"[LLM] 字段 {key} 忽略：{note}")
            continue
        ev = item.get("evidence") if isinstance(item, dict) else ""
        conf = item.get("confidence") if isinstance(item, dict) else None
        out[key] = {"value": value, "evidence": str(ev or "").strip()[:300],
                    "source": "LLM", "source_detail": "模型抽取",
                    "confidence": float(conf) if isinstance(conf, (int, float)) else None}
    return out


# ============================================================ 合并与落配置
def _merge(rules: dict[str, dict], model: dict[str, dict], log=print) -> dict[str, dict]:
    """规则与 LLM 结果合并：一致 → 高可信；不一致 → 保留 LLM 值但标记冲突。"""
    out: dict[str, dict] = {}
    conflicts: list[str] = []
    for key in FIELD_BY_KEY:
        r, m = rules.get(key), model.get(key)
        if r and m:
            same = (r["value"] == m["value"])
            if same:
                out[key] = {**m, "source": "LLM + 规则一致",
                            "evidence": m.get("evidence") or r["evidence"]}
            else:
                out[key] = {**m, "source": "LLM（与规则不一致）",
                            "evidence": m.get("evidence") or "",
                            "conflict": f"规则识别为 {r['value']}（依据：{r['evidence'][:60]}…）"}
                conflicts.append(key)
        elif m:
            out[key] = dict(m)
        elif r:
            out[key] = {**r, "source": "规则（未用 LLM）"}
    if conflicts:
        log(f"[核对] 这些字段 LLM 与规则不一致，已保留 LLM 结果并标注原文依据：{conflicts}")
    return out


def extract_train_params(text: str, cfg: LLMConfig | None = None, use_llm: bool | None = None,
                         log=print) -> dict:
    """从论文文本抽取训练超参（LLM 可选，失败自动降级到规则）。

    Returns
    -------
    dict:: ``{"fields": {...}, "meta": {provider, model, used_llm, warnings, elapsed}}``
    """
    text = str(text or "")
    if len(text.strip()) < 20:
        raise ValueError("论文文本太短（至少 20 个字符），请粘贴方法段落")
    t0 = time.perf_counter()
    warnings: list[str] = []
    rules = extract_by_rules(text)
    log(f"[规则] 命中 {len(rules)} 个字段：{', '.join(rules) or '无'}")
    llm_out: dict[str, dict] = {}
    cfg = (cfg or default_config()).resolved()
    want_llm = cfg.configured if use_llm is None else bool(use_llm)
    if not cfg.configured:
        note = ("未配置 LLM（缺 API 密钥或 base_url）→ 已用内置规则抽取；"
                "想让它理解更绕的表述，可在「LLM 设置」里填密钥")
        if use_llm is not False:
            warnings.append(note)
            log(f"[提示] {note}")
        want_llm = False
    elif use_llm is False:
        log("[提示] 已按你的选择跳过 LLM，只用规则抽取")
    if want_llm:
        try:
            llm_out = extract_by_llm(text, cfg, log=log)
            log(f"[LLM] 命中 {len(llm_out)} 个字段：{', '.join(llm_out) or '无'}")
        except Exception as e:  # noqa: BLE001 - 网络/密钥/格式问题都不该让功能塌掉
            warnings.append(f"LLM 调用失败（{type(e).__name__}: {str(e)[:120]}）→ 已降级为规则抽取")
            log(f"[警告] {warnings[-1]}")
            llm_out = {}
    fields = _merge(rules, llm_out, log=log)
    for f in FIELDS:
        if f.key not in fields:
            fields[f.key] = {"value": None, "evidence": "", "source": "未识别"}
    return {
        "fields": fields,
        "meta": {
            "used_llm": bool(llm_out),
            "provider": cfg.provider,
            "model": cfg.model if llm_out else "",
            "n_rules": len(rules),
            "n_llm": len(llm_out),
            "warnings": warnings,
            "elapsed": round(time.perf_counter() - t0, 2),
            "n_chars": len(text),
        },
    }


def to_train_config(fields: dict, base: dict | None = None) -> dict:
    """把抽取结果写成训练配置片段（只包含识别到的字段，便于与现有配置合并）。

    ``base`` 给定时会**深合并**到它上面（不覆盖未识别字段），可直接喂给训练页。

    特殊处理：
    - ``augmentation=False`` → 三个增强开关全置 0（论文明确说不用增强）
    - ``augmentation=True`` + 具体手段 → 按 :data:`AUG_TYPE_TO_PARAM` 给出**非零默认强度**
      （让"应用到训练配置"真的开出增强），本平台暂不支持的手段写进
      ``augmentation.unsupported_types`` 并在界面提示，不静默丢弃
    """
    import copy

    cfg = copy.deepcopy(base) if base else {}

    def _set(path: tuple, value):
        cur = cfg
        for k in path[:-1]:
            cur = cur.setdefault(k, {})
            if not isinstance(cur, dict):        # 与 base 结构冲突时直接覆盖
                cur = {}
        cur[path[-1]] = value

    def _get(key: str):
        item = (fields or {}).get(key)
        return item.get("value") if isinstance(item, dict) else item

    for key in FIELD_BY_KEY:
        if key in ("augmentation", "augmentation_types"):   # 单独处理（不是训练器字段）
            continue
        val = _get(key)
        if val is not None:
            _set(FIELD_BY_KEY[key].path, val)

    aug = cfg.setdefault("augmentation", {})
    aug.pop("enabled", None)
    enabled, types = _get("augmentation"), _get("augmentation_types") or []
    if enabled is False:
        for k in ("time_shift", "gaussian_noise", "channel_dropout"):
            aug[k] = 0
        aug.pop("unsupported_types", None)
    else:
        for t in types:
            if t in AUG_TYPE_TO_PARAM:
                k, v = AUG_TYPE_TO_PARAM[t]
                # 当前是 0/缺失（= 未启用）→ 用默认强度打开；用户已设非零值则不覆盖
                if not aug.get(k):
                    aug[k] = v
        unsupported = [t for t in types if t in AUG_UNSUPPORTED]
        if unsupported:
            aug["unsupported_types"] = unsupported
    if not aug:
        cfg.pop("augmentation", None)
    return cfg


# ================================================================ 文件读取
def read_text_file(path: str | Path) -> str:
    """读取论文文本：``.txt/.md/.tex/.json`` 直接读，``.pdf`` 需要 pypdf。"""
    p = Path(str(path))
    if not p.exists():
        raise FileNotFoundError(f"文件不存在：{p}")
    if p.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ImportError(
                "读取 PDF 需要 pypdf：pip install pypdf（或先复制文本再粘贴）") from e
        reader = PdfReader(str(p))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return p.read_text(encoding="utf-8", errors="ignore")


def parse_field_text(key: str, text):
    """把界面里手填的文本转成字段值（空/占位符 → None）。供"我的修正"列使用。"""
    f = FIELD_BY_KEY.get(key)
    if f is None:
        return None
    s = str(text).strip()
    if not s or s in ("—", "-", "未识别", "None", "null"):
        return None
    if f.kind == "bool":
        low = s.lower()
        if low in ("是", "true", "yes", "1", "用了", "有", "使用"):
            return True
        if low in ("否", "false", "no", "0", "没用", "未使用", "不用"):
            return False
        return None
    if f.kind == "list":
        items = [x for x in re.split(r"[,，;；、/\s]+", s) if x]
        allowed = {"time_shift", "gaussian_noise", "channel_dropout", "mixup",
                   "crop", "scaling", "freq_mask", "sliding_window"}
        keep = [x for x in items if x in allowed]
        return keep or None
    value, _note = _coerce(key, s)
    return value


def format_report(res: dict) -> str:
    """把抽取结果渲染成可读文本（GUI 日志/CLI 输出用）。"""
    fields, meta = res["fields"], res["meta"]
    lines = [f"论文超参抽取：{meta['n_chars']} 字，耗时 {meta['elapsed']}s，"
             f"规则命中 {meta['n_rules']} 项"
             + (f"，LLM（{meta['model']}）命中 {meta['n_llm']} 项" if meta["used_llm"] else "")]
    for f in FIELDS:
        item = fields.get(f.key) or {}
        v = item.get("value")
        if v is None:
            continue
        show = "、".join(v) if isinstance(v, list) else \
            (f"{v:.6g}" if isinstance(v, float) else str(v))
        lines.append(f"  · {f.label}: {show}    [{item.get('source')}]")
        if item.get("evidence"):
            lines.append(f"      依据：{item['evidence'][:110]}")
        if item.get("conflict"):
            lines.append(f"      冲突：{item['conflict'][:110]}")
    for w in meta.get("warnings") or []:
        lines.append(f"  ! {w}")
    return "\n".join(lines)
