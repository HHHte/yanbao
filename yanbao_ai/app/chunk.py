"""Chunk：把 full.md 切成检索用的块，结构优先 + 长度兜底。

策略（应对 MinerU 标题错位）：
- 先按 Markdown 标题（#/##/###）划分逻辑节，维护 heading_path 面包屑。
- 每节按目标 token 数（~1000）打包；超长段落按 token 二次切，留 overlap(~120)。
- 标题识别不可靠时（正文被并进标题行），仍能靠长度兜底不产生超大/超碎块。
- 块内关联图片：记录 ![](images/xxx.jpg) 与其上文最近的 Exhibit/图注标题 → image_refs。
- token 用 tiktoken 估算（cl100k_base，中英通用近似），失败退字符数/1.6 估。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

TARGET_TOKENS = 1000
OVERLAP_TOKENS = 120
MAX_TOKENS = 1400  # 硬上限，超过必须二次切

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_IMAGE = re.compile(r"!\[[^\]]*\]\((images/[^)]+)\)")
_EXHIBIT = re.compile(r"(?:Exhibit|Figure|Table|图表?|表)\s*[\d\-.]+", re.I)


def _get_encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


_ENC = _get_encoder()


def count_tokens(text: str) -> int:
    if _ENC is not None:
        return len(_ENC.encode(text, disallowed_special=()))
    # 兜底：中英混排粗估，1 token ≈ 1.6 字符
    return max(1, int(len(text) / 1.6))


@dataclass
class Chunk:
    seq: int
    heading_path: str
    text: str
    char_start: int
    char_end: int
    image_refs: list[dict] = field(default_factory=list)
    token_est: int = 0


@dataclass
class _Section:
    heading_path: str
    lines: list[str]
    char_start: int


def _split_sections(text: str) -> list[_Section]:
    """按标题切逻辑节，维护 heading_path（章>节>小节）。"""
    sections: list[_Section] = []
    # heading stack: list of (level, title)
    stack: list[tuple[int, str]] = []
    cur_lines: list[str] = []
    cur_start = 0
    pos = 0

    def flush(start: int):
        if cur_lines and any(l.strip() for l in cur_lines):
            path = " > ".join(t for _, t in stack)
            sections.append(_Section(path, list(cur_lines), start))

    for line in text.splitlines(keepends=True):
        m = _HEADING.match(line.rstrip("\r\n"))
        if m:
            # 遇到新标题，先结算上一节
            flush(cur_start)
            cur_lines = []
            cur_start = pos + len(line)
            level = len(m.group(1))
            title = m.group(2).strip()
            # 维护标题栈：弹出 >= 当前层级的
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            if not cur_lines:
                cur_start = pos
            cur_lines.append(line)
        pos += len(line)
    flush(cur_start)
    return sections


def _collect_images(text: str) -> list[dict]:
    """收集块内图片及其最近上文的 Exhibit/图注标题。"""
    refs: list[dict] = []
    last_caption = None
    for line in text.splitlines():
        cap = _EXHIBIT.search(line)
        if cap:
            last_caption = line.strip()
        for im in _IMAGE.finditer(line):
            refs.append({"path": im.group(1), "caption": last_caption})
    return refs


def _pack_section(sec: _Section, seq_start: int) -> list[Chunk]:
    """把一节按 token 目标打包成若干块；超长段落二次切。"""
    text = "".join(sec.lines)
    if not text.strip():
        return []

    # 段落级切分（空行分段），逐段累加到接近 TARGET
    paras = re.split(r"(\n\s*\n)", text)  # 保留分隔符以还原 char 偏移
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_tokens = 0
    seq = seq_start
    offset = sec.char_start
    buf_char_start = offset

    def emit(force_overlap_from: list[str] | None = None):
        nonlocal buf, buf_tokens, seq, buf_char_start
        body = "".join(buf)
        if not body.strip():
            buf, buf_tokens = [], 0
            return
        c = Chunk(
            seq=seq,
            heading_path=sec.heading_path,
            text=body,
            char_start=buf_char_start,
            char_end=buf_char_start + len(body),
            image_refs=_collect_images(body),
            token_est=count_tokens(body),
        )
        chunks.append(c)
        seq += 1
        buf, buf_tokens = [], 0

    for part in paras:
        ptok = count_tokens(part)
        # 单段就超硬上限：按 token 窗口二次切
        if ptok > MAX_TOKENS:
            if buf:
                emit()
                buf_char_start = offset
            for piece in _split_long(part):
                c = Chunk(
                    seq=seq,
                    heading_path=sec.heading_path,
                    text=piece,
                    char_start=buf_char_start,
                    char_end=buf_char_start + len(piece),
                    image_refs=_collect_images(piece),
                    token_est=count_tokens(piece),
                )
                chunks.append(c)
                seq += 1
                buf_char_start += len(piece)
            offset += len(part)
            buf_char_start = offset
            continue

        if buf_tokens + ptok > TARGET_TOKENS and buf:
            emit()
            buf_char_start = offset
        if not buf:
            buf_char_start = offset
        buf.append(part)
        buf_tokens += ptok
        offset += len(part)

    if buf:
        emit()
    return chunks


def _split_long(text: str) -> list[str]:
    """把超长文本按 token 窗口切，留 overlap。用编码器时按 token，否则按字符近似。"""
    if _ENC is None:
        # 字符近似窗口
        step = int(TARGET_TOKENS * 1.6)
        ov = int(OVERLAP_TOKENS * 1.6)
        out = []
        i = 0
        while i < len(text):
            out.append(text[i : i + step])
            i += step - ov
        return out
    toks = _ENC.encode(text, disallowed_special=())
    out = []
    i = 0
    step = TARGET_TOKENS
    ov = OVERLAP_TOKENS
    while i < len(toks):
        window = toks[i : i + step]
        out.append(_ENC.decode(window))
        i += step - ov
    return out


MIN_TOKENS = 50  # 低于此视为超碎块，尝试并入相邻块


def _merge_tiny(chunks: list[Chunk]) -> list[Chunk]:
    """把超碎块（<MIN_TOKENS）并入相邻块，减少孤立标题/短行拉低检索质量。

    合并优先级：并入同 heading_path 的前一块；否则并入后一块；
    整篇仅一个块时不动。合并后重排 seq，重算 token/char/图片。
    """
    if len(chunks) <= 1:
        return chunks

    # 超碎块多是 MinerU 拆出的孤立标题行，各自 heading_path 不同，
    # 故不强求同路径：无条件并入前一块（保留前块的 heading_path，
    # 碎块文本作为其内容一并可检索）；无前块则并入后一块。
    merged: list[Chunk] = []
    for c in chunks:
        # 仅当并入后不超硬上限时才并前块，避免把接近上限的块撑爆
        if (
            c.token_est < MIN_TOKENS
            and merged
            and merged[-1].token_est + c.token_est <= MAX_TOKENS
        ):
            prev = merged[-1]
            # 用换行拼接，保留碎块（常是标题词）以增强前块可检索性
            prev.text = prev.text.rstrip() + "\n" + c.text
            prev.char_end = c.char_end
            prev.image_refs = prev.image_refs + c.image_refs
            prev.token_est = count_tokens(prev.text)
        else:
            merged.append(c)

    # 首块若仍超碎（整篇第一块就短），并入后一块
    out: list[Chunk] = []
    i = 0
    while i < len(merged):
        c = merged[i]
        if (
            c.token_est < MIN_TOKENS
            and i + 1 < len(merged)
            and merged[i + 1].token_est + c.token_est <= MAX_TOKENS
        ):
            nxt = merged[i + 1]
            nxt.text = c.text.rstrip() + "\n" + nxt.text
            nxt.char_start = c.char_start
            nxt.image_refs = c.image_refs + nxt.image_refs
            nxt.token_est = count_tokens(nxt.text)
            i += 1
            continue
        out.append(c)
        i += 1

    for new_seq, c in enumerate(out):
        c.seq = new_seq
    return out


def chunk_text(text: str) -> list[Chunk]:
    """把整篇 full.md 切块。返回按 seq 递增的 Chunk 列表。"""
    sections = _split_sections(text)
    all_chunks: list[Chunk] = []
    seq = 0
    for sec in sections:
        packed = _pack_section(sec, seq)
        all_chunks.extend(packed)
        seq += len(packed)
    return _merge_tiny(all_chunks)


def chunk_file(md_path: Path) -> list[Chunk]:
    text = md_path.read_text(encoding="utf-8", errors="replace")
    return chunk_text(text)
