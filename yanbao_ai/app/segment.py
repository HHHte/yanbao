"""Segment：中文分词（jieba），供 FTS 索引与查询两侧共用。

为什么需要（已探针实证）：
- FTS5 内置 trigram 分词对中文有两处硬伤：
  1. 2 字词（白酒/批价/茅台）短于 trigram 下限 3，MATCH 必空；
  2. 自然语言长句被切成连续 3-gram，`商品板块二季度` 恰好连中多个滑窗被 BM25
     误判为高相关，真正相关的块反被压下（假阳性排序）。
- 解法是中文 IR 标准做法：索引与查询两侧都用 jieba 分词，把 `白酒`/`批价` 变成
  真正的词元，FTS 用 unicode61 按空格切词，BM25 按概念级词频排序。

关键约束：**索引侧与查询侧必须用同一套分词**，否则词元对不上。故统一走本模块。
"""
from __future__ import annotations

import re

# U+FFFD 替换字符：MinerU 抽取 PDF 时字符丢失的残留，近半文档含之（已探针）。
# 分词前先剔除，避免它进入词元污染索引；正文展示用的清理在 normalize 里做。
_FFFD = "�"

_jieba = None
_jieba_lock_tried = False


def _get_jieba():
    """惰性加载 jieba（首次 ~0.6s 建前缀词典）。加载失败返回 None → 调用方退回按字符。"""
    global _jieba, _jieba_lock_tried
    if _jieba is None and not _jieba_lock_tried:
        _jieba_lock_tried = True
        try:
            import jieba

            jieba.initialize()
            _jieba = jieba
        except Exception:
            _jieba = None
    return _jieba


def _fallback_tokens(text: str) -> list[str]:
    """无 jieba 时的退路：CJK 逐字 + 连续 ASCII 整词。保证仍可检索，只是不分词。"""
    toks: list[str] = []
    for m in re.finditer(r"[A-Za-z0-9_]+|[一-鿿㐀-䶿]", text):
        toks.append(m.group(0))
    return toks


def tokenize(text: str) -> list[str]:
    """把文本切成词元列表：jieba 中文分词 + 保留 ASCII 词，去空白/标点/U+FFFD。"""
    if not text:
        return []
    text = text.replace(_FFFD, " ")
    jb = _get_jieba()
    if jb is None:
        return _fallback_tokens(text)
    out: list[str] = []
    for w in jb.cut(text, cut_all=False):
        w = w.strip()
        if not w:
            continue
        # 丢纯标点/空白词元；保留中文、字母、数字（含带小数点的数字如 1650）
        if re.fullmatch(r"[\W_]+", w, flags=re.UNICODE) and not re.search(r"\w", w):
            continue
        out.append(w)
    return out


def segment_for_index(text: str) -> str:
    """索引侧：把正文分词后用空格连接，喂给 unicode61 FTS（按空格切词元）。"""
    return " ".join(tokenize(text))


def segment_query(q: str) -> list[str]:
    """查询侧：分词后去重保序，供构造 FTS MATCH 串。"""
    seen: set[str] = set()
    out: list[str] = []
    for t in tokenize(q):
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out
