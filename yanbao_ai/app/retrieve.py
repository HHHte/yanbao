"""Retrieve：混合检索。元数据预过滤 → BM25 ⊕ 稠密向量 → RRF 融合 → 可选重排。

流程（方案 §6.4）：
1. 元数据预过滤：date/institution/category/stock 缩小候选文档集（SQL where）。
2. 词法路：FTS5(trigram) MATCH BM25 topK；查询 <3 字（trigram 下限）走 LIKE 兜底。
3. 稠密路：sqlite-vec KNN topK（需查询向量；无 key/无向量表则跳过）。
4. RRF 融合：rank 融合两路，1/(k+rank)，k=60，取 top~N。
5. 可选 Claude 重排（此处留接口，默认关，由上层 generate 决定是否调）。

无向量时自动退纯词法（只走 BM25/LIKE），保证无 key 环境仍可用。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import Config
from . import store
from .embed import Embedder, EmbedError, _vec_to_bytes

RRF_K = 60          # RRF 平滑常数（经验值 60）
DEFAULT_TOPK = 40   # 每路召回条数
DEFAULT_FUSED = 30  # 融合后返回上限
_TRIGRAM_MIN = 3    # trigram 分词下限，短于此 MATCH 命中不稳，走 LIKE


@dataclass
class Filters:
    """元数据预过滤条件（全部可选，均为 AND）。"""

    date_from: str | None = None   # ISO，含
    date_to: str | None = None     # ISO，含
    institution: str | None = None  # 精确/子串（LIKE）
    category: str | None = None    # 国内券商报告 / 投行报告
    stock_code: str | None = None
    lang: str | None = None        # zh / en

    def where(self) -> tuple[str, list]:
        """生成作用于 documents 的 WHERE 片段与参数。"""
        clauses: list[str] = []
        params: list = []
        if self.date_from:
            clauses.append("d.report_date >= ?")
            params.append(self.date_from)
        if self.date_to:
            clauses.append("d.report_date <= ?")
            params.append(self.date_to)
        if self.institution:
            clauses.append("d.institution LIKE ?")
            params.append(f"%{self.institution}%")
        if self.category:
            clauses.append("d.category = ?")
            params.append(self.category)
        if self.stock_code:
            clauses.append("d.stock_code = ?")
            params.append(self.stock_code)
        if self.lang:
            clauses.append("d.lang = ?")
            params.append(self.lang)
        sql = (" AND " + " AND ".join(clauses)) if clauses else ""
        return sql, params


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    seq: int
    heading_path: str
    text: str
    image_refs: str        # JSON 原文
    title: str | None = None
    institution: str | None = None
    category: str | None = None
    report_date: str | None = None
    score: float = 0.0     # 融合分
    bm25_rank: int | None = None
    dense_rank: int | None = None


_CJK_RE = re.compile(r"[㐀-鿿豈-﫿぀-ヿ]+")


def _fts_terms(q: str) -> list[str]:
    """查询侧词元：jieba 分词（与索引侧 segment_for_index 对齐）。

    索引侧存的是 jieba 分词后按空格连接的文本、unicode61 按空格切词元；查询侧
    必须用同一套分词，词元才能对上。`白酒`/`批价` 成为真正词元，2 字词正常命中，
    BM25 按概念级词频排序，消除 trigram 滑窗的假阳性（见 app/segment.py 探针记录）。
    """
    from .segment import segment_query

    return segment_query(q)


def _fts_query(q: str) -> str:
    """把用户查询转成 FTS5 MATCH 串：滑窗/词元 OR 组合，各元加引号防语法注入。"""
    quoted = [f'"{t.replace(chr(34), chr(34) * 2)}"' for t in _fts_terms(q)]
    return " OR ".join(quoted) if quoted else '""'


def _like_terms(q: str) -> list[str]:
    """LIKE 兜底用的子串词元：jieba 分词后的词，逐词子串匹配。

    仅在 FTS 分词结果为空（极端输入）时才用；正常中文查询 jieba 都能出词元。
    """
    from .segment import segment_query

    return segment_query(q)


def _need_like_fallback(q: str) -> bool:
    """jieba 分词为空（查询里没有任何可成词的片段）→ 才走 LIKE 兜底。

    jieba + unicode61 下 `白酒`/`批价` 等 2 字词已能正常 MATCH，兜底几乎不触发；
    仅当查询全是标点/空白等无词元时才走 LIKE。
    """
    return not _fts_terms(q)


def _lexical_search(conn, query: str, filters: Filters, topk: int) -> list[tuple[str, int]]:
    """词法召回，返回 [(chunk_id, rank)]，rank 从 0 起。

    trigram 可覆盖（所有词 >=3 字）→ FTS5 MATCH + bm25 排序；
    否则 LIKE 兜底（对每个 <3 字词做 text LIKE，OR 合并），无 BM25 打分、按 seq 稳定序。
    """
    where_sql, where_params = filters.where()
    use_like = _need_like_fallback(query)

    if not use_like:
        match = _fts_query(query)
        sql = (
            "SELECT c.chunk_id "
            "FROM chunks_fts f "
            "JOIN chunks c ON c.rowid = f.rowid "
            "JOIN documents d ON d.doc_id = c.doc_id "
            "WHERE chunks_fts MATCH ?" + where_sql + " "
            "ORDER BY bm25(chunks_fts) LIMIT ?"
        )
        params = [match, *where_params, topk]
        rows = conn.execute(sql, params).fetchall()
        return [(r["chunk_id"], i) for i, r in enumerate(rows)]

    # LIKE 兜底：对每个词做子串匹配（含 <3 字 CJK 词），OR 合并
    terms = _like_terms(query)
    if not terms:
        return []
    like_clauses = " OR ".join(["c.text LIKE ?"] * len(terms))
    like_params = [f"%{t}%" for t in terms]
    sql = (
        "SELECT c.chunk_id "
        "FROM chunks c "
        "JOIN documents d ON d.doc_id = c.doc_id "
        "WHERE (" + like_clauses + ")" + where_sql + " "
        "ORDER BY c.doc_id, c.seq LIMIT ?"
    )
    rows = conn.execute(sql, [*like_params, *where_params, topk]).fetchall()
    return [(r["chunk_id"], i) for i, r in enumerate(rows)]


def _dense_search(
    conn, query_vec: bytes, filters: Filters, topk: int
) -> list[tuple[str, int]]:
    """稠密召回：sqlite-vec KNN。返回 [(chunk_id, rank)]，rank 从 0 起。

    先在过滤后的文档集上取候选 rowid，再对候选做 KNN——vec0 的 MATCH KNN 不便直接
    JOIN 过滤，故用 rowid IN 子查询限定候选（候选量受 documents 过滤控制）。
    """
    where_sql, where_params = filters.where()
    # 候选 rowid（过滤后文档的全部块）
    cand_sql = (
        "SELECT c.rowid FROM chunks c "
        "JOIN documents d ON d.doc_id = c.doc_id "
        "WHERE 1=1" + where_sql
    )
    cand = [r[0] for r in conn.execute(cand_sql, where_params).fetchall()]
    if not cand:
        return []

    # vec0 KNN：k 限定 + rowid 约束到候选集
    placeholders = ",".join("?" * len(cand))
    sql = (
        f"SELECT v.rowid FROM chunk_vec v "
        f"WHERE v.embedding MATCH ? AND k = ? "
        f"AND v.rowid IN ({placeholders}) "
        f"ORDER BY distance"
    )
    try:
        rows = conn.execute(sql, [query_vec, topk, *cand]).fetchall()
    except Exception:
        # 某些 vec0 版本不支持 rowid IN 与 k 同用 → 退：全量 KNN 后在 Python 侧过滤
        rows = conn.execute(
            "SELECT v.rowid FROM chunk_vec v "
            "WHERE v.embedding MATCH ? AND k = ? ORDER BY distance",
            [query_vec, max(topk * 4, topk)],
        ).fetchall()
        cand_set = set(cand)
        rows = [r for r in rows if r[0] in cand_set][:topk]

    # rowid → chunk_id
    out: list[tuple[str, int]] = []
    for i, r in enumerate(rows):
        rid = r[0] if not isinstance(r, (list, tuple)) else r[0]
        cid = conn.execute(
            "SELECT chunk_id FROM chunks WHERE rowid=?", (rid,)
        ).fetchone()
        if cid:
            out.append((cid[0], i))
    return out


def _rrf_fuse(
    lex: list[tuple[str, int]],
    dense: list[tuple[str, int]],
    limit: int,
) -> list[tuple[str, float, int | None, int | None]]:
    """RRF 融合两路 rank。返回 [(chunk_id, score, bm25_rank, dense_rank)] 降序。"""
    lex_rank = {cid: r for cid, r in lex}
    dense_rank = {cid: r for cid, r in dense}
    all_ids = set(lex_rank) | set(dense_rank)
    scored = []
    for cid in all_ids:
        score = 0.0
        lr = lex_rank.get(cid)
        dr = dense_rank.get(cid)
        if lr is not None:
            score += 1.0 / (RRF_K + lr)
        if dr is not None:
            score += 1.0 / (RRF_K + dr)
        scored.append((cid, score, lr, dr))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:limit]


def _hydrate(conn, fused) -> list[Hit]:
    """把融合结果补全为 Hit（带文档元数据）。"""
    hits: list[Hit] = []
    for cid, score, lr, dr in fused:
        row = conn.execute(
            "SELECT c.chunk_id, c.doc_id, c.seq, c.heading_path, c.text, "
            "c.image_refs, d.title, d.institution, d.category, d.report_date "
            "FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
            "WHERE c.chunk_id=?",
            (cid,),
        ).fetchone()
        if not row:
            continue
        hits.append(
            Hit(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                seq=row["seq"],
                heading_path=row["heading_path"],
                text=row["text"],
                image_refs=row["image_refs"],
                title=row["title"],
                institution=row["institution"],
                category=row["category"],
                report_date=row["report_date"],
                score=score,
                bm25_rank=lr,
                dense_rank=dr,
            )
        )
    return hits


@dataclass
class RetrieveResult:
    hits: list[Hit] = field(default_factory=list)
    mode: str = "hybrid"       # hybrid / lexical
    used_like: bool = False    # 是否走了 LIKE 兜底
    dense_ok: bool = False     # 稠密路是否实际执行


def retrieve(
    cfg: Config,
    query: str,
    *,
    filters: Filters | None = None,
    topk: int = DEFAULT_TOPK,
    limit: int = DEFAULT_FUSED,
    conn=None,
    mode: str = "auto",        # auto / lexical
) -> RetrieveResult:
    """混合检索主入口。

    mode='lexical' 强制纯词法；'auto' 时若有 key+向量表则混合，否则自动退词法。
    """
    filters = filters or Filters()
    own_conn = conn is None
    if own_conn:
        conn, vec_ok = store.connect_with_vec(cfg.paths.db, cfg.embed.dimensions)
    else:
        vec_ok = store.load_vec(conn)

    result = RetrieveResult()
    try:
        # 词法路（总是执行）
        lex = _lexical_search(conn, query, filters, topk)
        result.used_like = _need_like_fallback(query)

        # 稠密路（auto + 有 key + 向量表可用 + 库里有向量）
        dense: list[tuple[str, int]] = []
        # cfg.embed.usable = enabled 开关 且 有 key。开关关掉时这里直接短路，
        # **一次嵌入请求都不发**（否则每次问答都要白挨一个 401/超时往返）。
        want_dense = (
            mode != "lexical"
            and vec_ok
            and cfg.embed.usable
            and store.count_chunks(conn) > 0
        )
        if want_dense:
            has_vec = conn.execute(
                "SELECT COUNT(*) FROM chunk_vec"
            ).fetchone()[0]
            if has_vec:
                try:
                    embedder = Embedder(cfg.embed, cfg.paths.db)
                    try:
                        qvec = embedder.embed([query])[0]
                    finally:
                        embedder.close()
                    dense = _dense_search(
                        conn, _vec_to_bytes(qvec), filters, topk
                    )
                    result.dense_ok = True
                except EmbedError:
                    dense = []  # 嵌入失败 → 退纯词法

        result.mode = "hybrid" if result.dense_ok else "lexical"
        fused = _rrf_fuse(lex, dense, limit)
        result.hits = _hydrate(conn, fused)
    finally:
        if own_conn:
            conn.close()

    return result
