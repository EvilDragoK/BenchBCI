"""论文文件（PDF/文本）读取 + "参数区"定位 + 多来源组装 + 抽取溯源。

为什么单独一个模块：BCI/EEG 论文的训练细节**往往不在正文**，而是放在正文之后的
补充材料（Supplementary / Supporting Information / Appendix / 附录、Table S1…）——
所以"拖一个 PDF 进来"这件事必须做三件超出"抽文本"的事：

1. **区分正文与补充材料**（文件名 + 内容线索），并在多个来源间**优先取补充材料**里
   写着实现细节/超参的那一小节，而不是把整篇 30 页正文塞给模型（那样既贵又容易被
   正文里的无关数字带偏）。
2. **定位"参数区"**：按小节标题（Implementation details / Experimental setup /
   Hyperparameters / 实验设置 / 超参数）和参数值密度打分，挑出最可能写着超参的连续文本。
3. **留下溯源**：每个来源文件记 sha256、页数、字符数，组装后的文本记 hash，抽取结果
   里每个字段的来源与原文依据一并保存 —— 写进实验记录的 ``paper_params``，
   半年后也能回答"这套超参是从哪篇论文哪一节抄的"。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

#: 可拖入的文件类型
SUPPORTED_SUFFIXES = (".pdf", ".txt", ".md", ".markdown", ".tex", ".rst")

#: 来源类型 → 组装优先级（数字小 = 优先）
KIND_PRIORITY = {"supplementary": 0, "main": 1, "unknown": 2}
KIND_LABELS = {"supplementary": "补充材料", "main": "论文正文", "unknown": "未知来源"}

#: 文件名线索
#: 注意也覆盖 "_SI" / "SI_" / "additional file" 这类常见补充材料命名
SUPP_NAME_HINTS = re.compile(
    r"supp|supporting|appendix|append|esi|extra|additional|"
    r"(?:^|[_\-\s])s\.?i(?:$|[_\-\s.])|"
    r"补充|附件|附录|支持信息|附加", re.I)
MAIN_NAME_HINTS = re.compile(r"main|manuscript|paper|正文|正文版|全文", re.I)

#: 小节标题里的"参数区"关键词（强）
PARAM_HEADINGS = re.compile(
    r"implementation\s*details?|experimental?\s*(?:setup|settings?|details?|protocol)|"
    r"hyper[\s-]*parameters?|training\s*(?:details?|settings?|config\w*|procedure|protocol)|"
    r"model\s*(?:training|configuration)|optimization\s*(?:details?|setup)|"
    r"network\s*(?:training|configuration)|"
    r"实现细节|实验设置|实验细节|超参数|网络(?:训练|参数|结构)?设置|"
    r"训练(?:设置|细节|配置|参数|过程)|参数设置|模型训练|具体设置|优化设置", re.I)

#: 评分用的弱线索（Table S1 / 表 S2 这类超参表）
PARAM_TABLES = re.compile(r"(?:table|tab\.?|表)\s*s?\s*\d+|补充表|附表", re.I)

#: 正文里出现"详见补充材料"的交叉引用（说明参数不在正文）
CROSS_REF = re.compile(
    r"(?:see|refer\s+to|provided\s+in|given\s+in|details?\s+in)\s+(?:the\s+)?"
    r"(?:supplement\w*|supporting\s+information|appendix|appendix\s+[A-Z])|"
    r"(?:详见|参见|见)\s*(?:补充材料|附录|附件|支持信息)", re.I)

#: 正文里出现参数取值的线索（用于给小节打分）
VALUE_HINTS = re.compile(
    r"learning\s*rate|\blr\b|batch[\s_-]*size|epochs?|optimizer|adamw?|\bsgd\b|"
    r"weight\s*decay|random\s*seed|\bseed\b|scheduler|cosine|"
    r"z[\s-]*score|standardiz|normaliz|augment|euclidean\s*align|"
    r"学习率|批大小|批量大小|迭代|优化器|权重衰减|随机种子|标准化|归一化|数据增强|"
    r"欧(?:几里得|氏)对齐|交叉验证|留一", re.I)

#: 小节标题行的一般形态（编号 / 全大写 / 短行冒号结尾 / 中文序号）
HEADING_LIKE = re.compile(
    r"^\s*(?:[SA]?\d+(?:\.\d+)*[\.\)]?\s+\S|"
    r"(?:supplementary|supporting|appendix|补充材料|附录|附件)\b.{0,60}$|"
    r"table\s+s?\d+.{0,80}$|表\s*s?\d+.{0,80}$|"
    r"[A-Z][A-Za-z0-9 ,\-/&()]{3,70}$|"
    r"[一二三四五六七八九十]+[、.．]\s*\S.{0,40}$|"
    r"\S{2,40}[:：]\s*$)", re.I)


# ================================================================ 来源文件
@dataclass
class SourceFile:
    """一个被拖进来的论文来源（正文或补充材料）。"""

    path: str
    name: str = ""
    kind: str = "unknown"
    n_pages: int = 0
    n_chars: int = 0
    sha256: str = ""
    note: str = ""
    chars_per_page: float = 0.0

    def __post_init__(self):
        if not self.name:
            self.name = Path(self.path).name

    @property
    def kind_label(self) -> str:
        return KIND_LABELS.get(self.kind, self.kind)

    def label(self) -> str:
        extra = f"，{self.n_pages} 页" if self.n_pages else ""
        extra += f"，{self.n_chars} 字"
        return f"{self.name}（{self.kind_label}{extra}）"


@dataclass
class Section:
    """文本里的一段"可能是参数区"的小节。"""

    title: str
    source: str = ""
    kind: str = "unknown"
    start: int = 0
    end: int = 0
    score: float = 0.0
    body_chars: int = 0

    def label(self) -> str:
        return f"{self.source}（{KIND_LABELS.get(self.kind, self.kind)}）｜{self.title}"


def _pdf_escape(t: str) -> str:
    return (str(t).replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            .encode("ascii", "ignore").decode())


def write_sample_pdf(path: str | Path, pages, per_page_chars: int | None = None) -> Path:
    """手写一个最小可用 PDF（Helvetica 文本）。

    仅用于**自检与演示**（测试、`docs/demo_features.py`）：不依赖 reportlab 等额外包，
    也能验证"拖 PDF 进来 → 抽参数"这条链路。``per_page_chars`` 可造出"每页字数极少"
    的假扫描件，用来验证扫描版提示。
    """
    n = max(1, len(list(pages)))
    pages = list(pages) or [""]
    page_ids = [4 + 2 * i for i in range(n)]
    content_ids = [5 + 2 * i for i in range(n)]
    bodies: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: (f"<< /Type /Pages /Kids [{' '.join(f'{i} 0 R' for i in page_ids)}] "
            f"/Count {n} >>").encode(),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for i, text in enumerate(pages):
        raw = str(text)
        if per_page_chars is not None:
            raw = (raw + " " * per_page_chars)[:per_page_chars]
        lines = raw.splitlines() or [""]
        stream = ("BT /F1 11 Tf 40 760 Td 13 TL\n"
                  + "".join(f"({_pdf_escape(line)}) Tj T*\n" for line in lines)
                  + "ET").encode()
        bodies[content_ids[i]] = (b"<< /Length " + str(len(stream)).encode()
                                  + b" >>\nstream\n" + stream + b"\nendstream")
        bodies[page_ids[i]] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_ids[i]} 0 R /Resources << /Font << /F1 3 0 R >> >> >>"
        ).encode()

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for oid in sorted(bodies):
        offsets[oid] = len(out)
        out += f"{oid} 0 obj\n".encode() + bodies[oid] + b"\nendobj\n"
    xref_pos = len(out)
    max_id = max(bodies)
    out += f"xref\n0 {max_id + 1}\n".encode() + b"0000000000 65535 f \n"
    for oid in range(1, max_id + 1):
        out += f"{offsets.get(oid, 0):010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\nstartxref\n"
            f"{xref_pos}\n%%EOF\n").encode()
    p = Path(str(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(bytes(out))
    return p


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(str(text).encode("utf-8", errors="ignore"))


def is_supported(path: str | Path) -> bool:
    """这个文件能不能被"论文文件"入口接受。"""
    return Path(str(path)).suffix.lower() in SUPPORTED_SUFFIXES


def guess_kind(name: str, text: str = "") -> str:
    """判断是论文正文还是补充材料（文件名优先，其次看内容线索）。"""
    n = Path(str(name)).stem
    if SUPP_NAME_HINTS.search(n):
        return "supplementary"
    if MAIN_NAME_HINTS.search(n):
        return "main"
    head = str(text)[:4000]
    if PARAM_HEADINGS.search(head) and re.search(
            r"supplement|supporting|appendix|补充材料|附录", head, re.I):
        return "supplementary"
    return "unknown"


def read_source(path: str | Path, log=print) -> tuple[SourceFile, str]:
    """读取一个来源文件 → ``(SourceFile, 文本)``。PDF 需要 pypdf。"""
    p = Path(str(path))
    if not p.exists():
        raise FileNotFoundError(f"文件不存在：{p}")
    raw = p.read_bytes()
    n_pages = 0
    if p.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:                    # pragma: no cover - 已装 pypdf
            raise ImportError("读取 PDF 需要 pypdf：pip install pypdf") from e
        reader = PdfReader(str(p))
        n_pages = len(reader.pages)
        pages = []
        for i, page in enumerate(reader.pages):
            try:
                pages.append(page.extract_text() or "")
            except Exception as e:                  # noqa: BLE001 - 单页坏了不该整篇失败
                pages.append("")
                log(f"[PDF] {p.name} 第 {i + 1} 页解析失败（{type(e).__name__}），已跳过")
        text = "\n".join(pages)
    else:
        text = p.read_text(encoding="utf-8", errors="ignore")

    src = SourceFile(path=str(p), name=p.name, kind=guess_kind(p.name, text),
                     n_pages=n_pages, n_chars=len(text), sha256=sha256_bytes(raw))
    src.chars_per_page = (len(text) / n_pages) if n_pages else float(len(text))
    if n_pages and src.chars_per_page < 120:
        src.note = (f"每页仅 {src.chars_per_page:.0f} 字，可能是扫描版/图片版 PDF："
                    f"文本抽取会失败，请先用 OCR 或复制文字")
        log(f"[警告] {src.note}")
    if not text.strip():
        src.note = "没有抽到任何文本（空文件或加密/扫描件）"
    return src, text


def read_sources(paths, log=print) -> tuple[list[SourceFile], dict[str, str]]:
    """批量读取（正文 + 补充材料），返回 ``(来源列表, {路径: 文本})``。"""
    out, texts = [], {}
    for path in paths:
        src, text = read_source(path, log=log)
        out.append(src)
        texts[src.path] = text
        log(f"[读取] {src.label()}")
    return out, texts


# ================================================================ 参数区定位
def _iter_headings(text: str) -> list[tuple[int, str]]:
    """找出"像标题"的行（行号, 行内容）。"""
    heads = []
    for i, line in enumerate(text.splitlines()):
        s = line.strip()
        if not s or len(s) > 110:
            continue
        if HEADING_LIKE.match(s):
            heads.append((i, s))
    return heads


def find_param_sections(text: str, source: str = "", kind: str = "unknown",
                        max_body: int = 8000, top: int = 4) -> list[Section]:
    """在文本里找"参数区"小节并按可能性打分（补充材料优先）。

    打分 = 来源优先级 + 标题关键词 + 表引用 + 正文里参数取值的密度。
    """
    text = str(text or "")
    if not text.strip():
        return []
    lines = text.splitlines()
    heads = _iter_headings(text)
    out: list[Section] = []
    for k, (ln, title) in enumerate(heads):
        # 小节体 = 本标题到下一个标题（或 max_body 字符）
        start_line = ln
        end_line = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        body = "\n".join(lines[start_line:end_line])
        if len(body) > max_body:
            body = body[:max_body]
        strong = bool(PARAM_HEADINGS.search(title))
        table = bool(PARAM_TABLES.search(title))
        hits = len(VALUE_HINTS.findall(body))
        if not (strong or table or hits >= 4):
            continue
        score = (3.0 if kind == "supplementary" else 0.0)
        score += 3.0 if strong else 0.0
        score += 1.5 if table else 0.0
        score += min(3.0, hits / 2.0)
        out.append(Section(title=title, source=source, kind=kind, start=ln, end=end_line,
                           score=round(score, 2), body_chars=len(body)))
    out.sort(key=lambda s: -s.score)
    return out[:top]


# ================================================================ 多来源组装
def assemble_param_text(items, max_chars: int = 12000,
                        prefer_supplementary: bool = True, log=print) -> tuple[str, dict]:
    """把多个来源组装成"喂给抽取器"的文本，补充材料优先。

    Parameters
    ----------
    items : list of ``(SourceFile, text)``

    Returns
    -------
    ``(组装后的文本, 说明信息)``；说明信息含 sources / sections / notes / n_chars，
    可直接写进实验记录的 ``paper_params.input``。
    """
    items = list(items)
    if not items:
        raise ValueError("没有任何输入文件")
    order = (sorted(items, key=lambda it: KIND_PRIORITY.get(it[0].kind, 2))
             if prefer_supplementary else list(items))
    blocks: list[str] = []
    all_secs: list[Section] = []
    info = {"sources": [asdict(s) for s, _ in items], "sections": [],
            "notes": [], "n_chars": 0, "truncated": False,
            "used_chars_by_source": {}}
    for idx, (src, text) in enumerate(order, start=1):
        secs = find_param_sections(text, source=src.name, kind=src.kind)
        all_secs += secs
        if secs:
            for s in secs[:2]:                      # 每个来源最多取 2 个最像的小节
                body = "\n".join(text.splitlines()[s.start:s.end]).strip()
                blocks.append(
                    f"【来源 {idx}：{src.name}（{src.kind_label}）｜小节：{s.title}】\n{body}")
                info["used_chars_by_source"][src.name] = \
                    info["used_chars_by_source"].get(src.name, 0) + len(body)
        else:
            take = text.strip()[:3000]
            if take:
                blocks.append(
                    f"【来源 {idx}：{src.name}（{src.kind_label}）｜未找到明确的参数小节，"
                    f"取开头 {len(take)} 字】\n{take}")
                info["used_chars_by_source"][src.name] = len(take)
                info["notes"].append(
                    f"{src.name}：没找到 Implementation details / 超参数 之类的小节，"
                    f"已取开头文本兜底")
        if CROSS_REF.search(text):
            info["notes"].append(
                f"{src.name}：正文提到「详见补充材料/附录」——参数很可能在补充材料里，"
                f"建议把补充材料一起拖进来（本次已优先解析补充材料）")
    merged = "\n\n".join(blocks)
    head = "\n".join(
        f"【来源 {i}：{s.name}（{s.kind_label}，{s.n_pages or '-'} 页，{s.n_chars} 字）"
        + (f" ⚠{s.note}" if s.note else "") + "】"
        for i, (s, _) in enumerate(order, start=1))
    limit = max(500, int(max_chars) - len(head) - 2)      # 上限要把来源头算进去
    if len(merged) > limit:
        info["truncated"] = True
        cut = merged[:limit]
        # 尽量在块边界截断，避免把小节切一半
        p = cut.rfind("【来源 ")
        merged = cut[:p].rstrip() if p > limit * 0.5 else cut.rstrip()
        info["notes"].append(f"文本超长，已截断到约 {len(merged)} 字（可调大「最多字符」）")
    final = f"{head}\n\n{merged}"
    info["n_chars"] = len(final)
    info["source_summary"] = "；".join(
        f"{s.name}（{s.kind_label}）" for s, _ in order)
    info["sections"] = [asdict(s) for s in all_secs]
    info["used_sections"] = [s.label() for s in sorted(all_secs, key=lambda x: -x.score)[:4]]
    log(f"[组装] {len(order)} 个来源 → {info['n_chars']} 字"
        + ("（已截断）" if info["truncated"] else ""))
    for s in info["used_sections"]:
        log(f"    优先小节：{s}")
    for n in info["notes"]:
        log(f"    [提示] {n}")
    return final, info


# ================================================================ 溯源信息
def provenance(res: dict, text: str = "", info: dict | None = None,
               files: list[SourceFile] | None = None, applied: bool = True) -> dict:
    """把一次抽取整理成可落盘的溯源块（写进 ``config.yaml`` 的 ``paper_params``）。"""
    info = dict(info or {})
    meta = dict((res or {}).get("meta") or {})
    fields = {}
    for key, item in ((res or {}).get("fields") or {}).items():
        if not item or item.get("value") is None:
            continue
        f = {"value": item["value"], "source": item.get("source"),
             "evidence": item.get("evidence") or ""}
        if item.get("conflict"):
            f["conflict"] = item["conflict"]
        fields[key] = f
    out = {
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
        "applied_to_config": bool(applied),
        "method": "llm" if meta.get("used_llm") else "rules",
        "provider": meta.get("provider", ""),
        "model": meta.get("model", "") or "",
        "input": {
            "source_summary": info.get("source_summary", ""),
            "sources": info.get("sources") or [asdict(s) for s in (files or [])],
            "used_sections": info.get("used_sections") or [],
            "sections": info.get("sections") or [],
            "n_chars": info.get("n_chars") or len(text or ""),
            "truncated": bool(info.get("truncated")),
            "notes": info.get("notes") or [],
            "text_sha256": sha256_text(text or ""),
            "excerpt": (text or "")[:1500],
        },
        "fields": fields,
        "unsupported_augmentation": [],
        "warnings": meta.get("warnings") or [],
    }
    from .llm import to_train_config

    try:                                            # 记录"平台不支持"的增强手段，便于追溯
        aug = to_train_config(fields).get("augmentation") or {}
        out["unsupported_augmentation"] = aug.get("unsupported_types") or []
    except Exception:                               # noqa: BLE001
        pass
    return out


def summarize_provenance(prov: dict) -> str:
    """一行话概括来源（写进实验列表的备注 / 界面标签）。"""
    if not prov:
        return ""
    inp = prov.get("input") or {}
    files = inp.get("sources") or []
    names = "、".join(f"{f.get('name')}（{KIND_LABELS.get(f.get('kind'), '未知')}）"
                     for f in files[:3]) or inp.get("source_summary", "")
    secs = inp.get("used_sections") or []
    method = "LLM" if prov.get("method") == "llm" else "规则"
    txt = f"{names} · {method}抽取" if names else f"{method}抽取"
    if secs:
        txt += f" · 取自 {secs[0]}"
    return txt


def markdown_provenance(prov: dict) -> str:
    """把溯源块渲染成 Markdown（插进实验报告）。"""
    if not prov:
        return ""
    from .llm import FIELD_BY_KEY

    inp = prov.get("input") or {}
    lines = ["## 训练参数来源（论文抽取）", "",
             f"- 抽取时间：{prov.get('extracted_at', '')}",
             f"- 方式：{'大模型（' + str(prov.get('model') or prov.get('provider')) + '）' if prov.get('method') == 'llm' else '内置规则（离线）'}",
             f"- 输入：{inp.get('source_summary') or '（未记录）'}"
             f"，{inp.get('n_chars', '?')} 字"
             + ("（已截断）" if inp.get("truncated") else "")]
    for f in inp.get("sources") or []:
        lines.append(f"  - `{f.get('name')}`（{KIND_LABELS.get(f.get('kind'), '未知')}）"
                     f" · {f.get('n_pages') or '-'} 页 · {f.get('n_chars')} 字"
                     f" · sha256 `{str(f.get('sha256'))[:12]}…`")
    for s in inp.get("used_sections") or []:
        lines.append(f"  - 采用小节：{s}")
    if inp.get("text_sha256"):
        lines.append(f"- 抽取所用文本 sha256：`{str(inp['text_sha256'])[:16]}…`")
    lines += ["", "| 字段 | 取值 | 来源 | 原文依据 |", "|---|---|---|---|"]
    for key, item in (prov.get("fields") or {}).items():
        label = FIELD_BY_KEY[key].label if key in FIELD_BY_KEY else key
        v = item.get("value")
        show = "、".join(v) if isinstance(v, list) else str(v)
        ev = str(item.get("evidence") or "").replace("|", "\\|")[:120]
        src = str(item.get("source") or "")
        if item.get("conflict"):
            src += "（⚠ 与规则不一致）"
        lines.append(f"| {label} | {show} | {src} | {ev} |")
    if prov.get("unsupported_augmentation"):
        lines.append("")
        lines.append(f"> 论文提到的这些增强手段本平台暂不支持："
                     f"{'、'.join(prov['unsupported_augmentation'])}")
    for w in prov.get("warnings") or []:
        lines.append(f"> ⚠ {w}")
    lines.append("")
    return "\n".join(lines)
