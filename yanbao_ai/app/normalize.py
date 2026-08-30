"""Normalize：对 MinerU 产出的 full.md 做保守、可回滚的规范化。

原则（保守优先，宁可漏修不可误改）：
- 复用 mineru_pipeline/clean_markdown.py 的起点财经广告块清除。
- 连字丢失（ff→f 等）只按**白名单精确整词**替换，绝不做规则化猜测——
  语料里 prefecture / defect / benefit 等是正常词，规则化会误伤。
- 行首误识别的项目符号 `n ` → `- `，仅当整行以 "n " 开头且后接文本。
- 绝不动图片引用 ![](images/...) 与 Exhibit/图注标题。
- 输出保留原文一份、规范化一份，便于对比回滚（由调用方决定是否落盘）。
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

# 复用 mineru_pipeline/clean_markdown.py（不在包内，按路径加载）
_CLEAN_MD = (
    Path(__file__).resolve().parent.parent.parent
    / "mineru_pipeline"
    / "clean_markdown.py"
)


def _load_clean():
    """动态加载 clean_markdown.clean；加载失败时返回恒等函数（不清广告，但不报错）。"""
    try:
        spec = importlib.util.spec_from_file_location("clean_markdown", _CLEAN_MD)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod.clean
    except Exception:
        return lambda text: (text, 0)


_clean_ads = _load_clean()

# ---- 连字丢失白名单：错拼 → 正确拼写（仅高频、无歧义的整词）----
# 全部经语料实测（eficiency/diferentiation/oferings/efect），大小写各列一版。
_HYPHEN_FIXES = {
    "eficiency": "efficiency",
    "eficiencies": "efficiencies",
    "eficient": "efficient",
    "eficiently": "efficiently",
    "diferentiation": "differentiation",
    "diferentiated": "differentiated",
    "diferent": "different",
    "diference": "difference",
    "diferences": "differences",
    "oferings": "offerings",
    "ofering": "offering",
    "ofer": "offer",
    "ofers": "offers",
    "ofered": "offered",
    "efect": "effect",
    "efects": "effects",
    "efective": "effective",
    "efectively": "effectively",
    "ofset": "offset",
    "ofsets": "offsets",
    "dificult": "difficult",
    "dificulty": "difficulty",
    "sufered": "suffered",
    "sufering": "suffering",
}

# 预编译为整词边界正则（\b 两侧），大小写不敏感匹配但保留首字母大小写
_HYPHEN_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _HYPHEN_FIXES) + r")\b",
    re.IGNORECASE,
)

_IMAGE_LINE = re.compile(r"^\s*!?\[[^\]]*\]\(images/[^)]+\)\s*$", re.I)
_EXHIBIT = re.compile(r"^\s*(Exhibit|Figure|Table|图|表)\s*\d", re.I)
# 行首误识别项目符号：整行以 "n " 开头且后接非空文本
_NBULLET = re.compile(r"^n\s+(?=\S)")


def _fix_hyphen_word(m: re.Match) -> str:
    bad = m.group(1)
    good = _HYPHEN_FIXES[bad.lower()]
    # 保留首字母大小写（句首/标题里可能是 Efect→Effect）
    if bad[:1].isupper():
        return good[:1].upper() + good[1:]
    return good


def normalize_text(text: str) -> tuple[str, int]:
    """返回 (规范化文本, 改动计数)。先清广告，再逐行保守修复。"""
    cleaned, _ = _clean_ads(text)
    out_lines: list[str] = []
    changes = 0
    for line in cleaned.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        eol = line[len(body):]

        # 图片行与 Exhibit/图注标题：原样保留，绝不动
        if _IMAGE_LINE.match(body) or _EXHIBIT.match(body):
            out_lines.append(line)
            continue

        new = body
        # 1) 行首误识别项目符号 n → -
        if _NBULLET.match(new):
            new = _NBULLET.sub("- ", new, count=1)

        # 2) 连字丢失白名单修复
        new, n = _HYPHEN_PATTERN.subn(_fix_hyphen_word, new)
        changes += n

        if new != body:
            changes += 1
        out_lines.append(new + eol)
    return "".join(out_lines), changes


def normalize_file(md_path: Path) -> tuple[str, int]:
    text = md_path.read_text(encoding="utf-8", errors="replace")
    return normalize_text(text)
