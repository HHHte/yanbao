"""Index：把 documents 里的 full.md 规范化→切块→嵌入→落库（chunks/FTS/向量）。

编排（幂等，按 doc_id 先清后写）：
    normalize_text  →  chunk_text  →  Embedder.embed  →  store.write_chunk
每篇文档一个事务：先 delete_doc_chunks，再逐块写入，最后 commit。

向量策略：
- 有 OpenAI key 且 sqlite-vec 可加载 → 稠密向量入 chunk_vec，混合检索可用。
- 无 key（--no-embed 或 key 未设）→ 只建 chunks + FTS，检索退纯词法（BM25/LIKE）。
- sqlite-vec 加载失败 → 同上退纯词法，并告警。

只读 documents.md_path（catalog 已存绝对路径），不重复解析文件名。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from . import store, normalize, chunk
from .embed import Embedder, EmbedStats, EmbedError, _vec_to_bytes


@dataclass
class IndexStats:
    docs_total: int = 0        # 待索引文档数
    docs_done: int = 0         # 成功索引
    docs_skipped: int = 0      # md 缺失等跳过
    chunks_written: int = 0    # 写入块数
    norm_changes: int = 0      # 规范化改动累计
    embed: EmbedStats = field(default_factory=EmbedStats)
    vec_enabled: bool = False  # 本次是否写向量
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"文档：{self.docs_done}/{self.docs_total} 成功"
            f"（跳过 {self.docs_skipped}）",
            f"块：写入 {self.chunks_written} 个，规范化改动 {self.norm_changes} 处",
            f"向量：{'启用' if self.vec_enabled else '未启用（纯词法检索）'}",
        ]
        if self.vec_enabled:
            e = self.embed
            lines.append(
                f"嵌入：请求 {e.requested}，命中缓存 {e.from_cache}，"
                f"调 API {e.from_api}（{e.api_calls} 批）"
            )
        if self.errors:
            lines.append(f"错误 {len(self.errors)} 条（前 5）：{self.errors[:5]}")
        return "\n".join(lines)


def _iter_documents(conn, only_doc_id: str | None, only_new: bool = False):
    """遍历 documents，产出 (doc_id, md_path)。

    only_doc_id 指定时只取该篇；only_new=True 时只取尚无 chunks 的文档（增量）。
    """
    if only_doc_id:
        rows = conn.execute(
            "SELECT doc_id, md_path FROM documents WHERE doc_id=?", (only_doc_id,)
        ).fetchall()
    elif only_new:
        # 只取 chunks 里没有任何块的文档（新文档或上次索引失败的）
        rows = conn.execute(
            "SELECT d.doc_id, d.md_path FROM documents d "
            "WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.doc_id = d.doc_id) "
            "ORDER BY d.report_date DESC"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT doc_id, md_path FROM documents ORDER BY report_date DESC"
        ).fetchall()
    for r in rows:
        yield r["doc_id"], r["md_path"]


def build_index(
    cfg: Config,
    conn=None,
    *,
    embed: bool = True,
    only_doc_id: str | None = None,
    only_new: bool = False,
) -> IndexStats:
    """全量（或单篇/仅新）建索引。返回统计。

    embed=False 或无 key → 跳过嵌入，只建 chunks + FTS（纯词法可用）。
    only_new=True → 只索引尚无 chunk 的文档（增量 update 用）。
    """
    st = IndexStats()

    own_conn = conn is None
    if own_conn:
        conn, vec_ok = store.connect_with_vec(cfg.paths.db, cfg.embed.dimensions)
    else:
        vec_ok = store.load_vec(conn)
        if vec_ok:
            store.ensure_vec_table(conn, cfg.embed.dimensions)

    # 嵌入器：需 key；无 key 或 embed=False 则纯词法
    embedder: Embedder | None = None
    do_embed = embed and vec_ok and cfg.embed.usable
    if embed and not vec_ok:
        st.errors.append("sqlite-vec 未加载，本次退纯词法（无向量）")
    if embed and vec_ok and not cfg.embed.enabled:
        st.errors.append("[embed].enabled=false，本次不写向量（检索走纯 BM25）")
    elif embed and vec_ok and not cfg.embed.api_key:
        st.errors.append("未设置 embedding key，本次退纯词法（无向量）")
    if do_embed:
        try:
            embedder = Embedder(cfg.embed, cfg.paths.db)
        except EmbedError as exc:
            st.errors.append(f"嵌入器初始化失败，退纯词法：{exc}")
            do_embed = False
    st.vec_enabled = do_embed

    try:
        docs = list(_iter_documents(conn, only_doc_id, only_new))
        st.docs_total = len(docs)
        for doc_id, md_path_s in docs:
            md_path = Path(md_path_s) if md_path_s else None
            if md_path is None or not md_path.is_file():
                st.docs_skipped += 1
                st.errors.append(f"{doc_id}: md 缺失 {md_path_s}")
                continue

            raw = md_path.read_text(encoding="utf-8", errors="replace")
            text, n_changes = normalize.normalize_text(raw)
            st.norm_changes += n_changes
            chunks = chunk.chunk_text(text)
            if not chunks:
                st.docs_skipped += 1
                continue

            # 嵌入（保序）
            vectors: list[bytes | None] = [None] * len(chunks)
            if do_embed and embedder is not None:
                try:
                    vecs = embedder.embed([c.text for c in chunks], st.embed)
                    vectors = [_vec_to_bytes(v) for v in vecs]
                except EmbedError as exc:
                    st.errors.append(f"{doc_id}: 嵌入失败，本篇跳过向量：{exc}")
                    vectors = [None] * len(chunks)

            # 单篇一事务：先清后写，幂等
            store.delete_doc_chunks(conn, doc_id, vec_ok)
            for c, emb in zip(chunks, vectors):
                store.write_chunk(
                    conn,
                    chunk_id=f"{doc_id}#{c.seq}",
                    doc_id=doc_id,
                    seq=c.seq,
                    heading_path=c.heading_path,
                    text=c.text,
                    char_start=c.char_start,
                    char_end=c.char_end,
                    image_refs_json=json.dumps(c.image_refs, ensure_ascii=False),
                    token_est=c.token_est,
                    embedding=emb,
                    vec_ok=vec_ok,
                )
                st.chunks_written += 1
            conn.commit()
            st.docs_done += 1
    finally:
        if embedder is not None:
            embedder.close()
        if own_conn:
            conn.close()

    return st


@dataclass
class BackfillStats:
    chunks_total: int = 0      # 库内块总数
    already_vec: int = 0       # 已有向量、跳过
    embedded: int = 0          # 本次写入向量
    embed: EmbedStats = field(default_factory=EmbedStats)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"块：{self.chunks_total} 总，已有向量 {self.already_vec}，"
            f"本次写入 {self.embedded}",
            f"嵌入：请求 {self.embed.requested}，命中缓存 {self.embed.from_cache}，"
            f"调 API {self.embed.from_api}（{self.embed.api_calls} 批）",
        ]
        if self.errors:
            lines.append(f"错误 {len(self.errors)} 条（前 5）：{self.errors[:5]}")
        return "\n".join(lines)


def embed_existing(
    cfg: Config,
    conn=None,
    *,
    batch_rows: int = 500,
    max_errors: int = 200,
) -> BackfillStats:
    """给库里已有的 chunks 补向量到 chunk_vec（不重新切块/不动 FTS），rowid 对齐。

    适用场景：chunks/FTS 已建好（--no-embed 索引过），后来才有 embedding key，
    只想把稠密向量补上，不想重跑几分钟的全量切块。幂等且可断点续跑：
    - 只处理 chunk_vec 里尚无该 rowid 的块（已嵌入的跳过）；
    - Embedder 自带 sha256 缓存，重复文本不重复付费；
    - 每 batch_rows 提交一次，中断后重跑从断点继续。
    """
    st = BackfillStats()

    own_conn = conn is None
    if own_conn:
        conn, vec_ok = store.connect_with_vec(cfg.paths.db, cfg.embed.dimensions)
    else:
        vec_ok = store.load_vec(conn)
        if vec_ok:
            store.ensure_vec_table(conn, cfg.embed.dimensions)

    if not vec_ok:
        st.errors.append("sqlite-vec 未加载，无法补向量")
        if own_conn:
            conn.close()
        return st
    if not cfg.embed.enabled:
        st.errors.append(
            "[embed].enabled=false，回填已跳过。要回填请先在 config.toml 置 true"
        )
        if own_conn:
            conn.close()
        return st
    if not cfg.embed.api_key:
        st.errors.append("未设置 embedding key，无法补向量")
        if own_conn:
            conn.close()
        return st

    embedder: Embedder | None = None
    try:
        embedder = Embedder(cfg.embed, cfg.paths.db)
    except EmbedError as exc:
        st.errors.append(f"嵌入器初始化失败：{exc}")
        if own_conn:
            conn.close()
        return st

    try:
        st.chunks_total = store.count_chunks(conn)
        # 只取尚无向量的块（rowid 不在 chunk_vec 里），按 rowid 稳定序，分批处理。
        rows = conn.execute(
            "SELECT c.rowid, c.text FROM chunks c "
            "WHERE NOT EXISTS (SELECT 1 FROM chunk_vec v WHERE v.rowid = c.rowid) "
            "ORDER BY c.rowid"
        ).fetchall()
        st.already_vec = st.chunks_total - len(rows)

        # 单个坏块/瞬时失败不该拖垮整轮 3800+ 批：失败的 500-块降级为逐个 embed，
        # 定位并跳过真正有问题的行，其余照常写入。错误累计到上限才中止。
        for start in range(0, len(rows), batch_rows):
            batch = rows[start : start + batch_rows]
            rids = [r["rowid"] for r in batch]
            texts = [r["text"] for r in batch]
            try:
                vecs = embedder.embed(texts, st.embed)
                pairs = list(zip(rids, vecs))
            except EmbedError:
                # 整块失败 → 逐行重试，隔离出真正坏的行（其余仍入库）
                pairs = []
                for rid, text in zip(rids, texts):
                    try:
                        v = embedder.embed([text], st.embed)[0]
                        pairs.append((rid, v))
                    except EmbedError as exc:
                        st.errors.append(f"rowid {rid}: 嵌入失败：{exc}")
                        if len(st.errors) >= max_errors:
                            break
            for rid, vec in pairs:
                conn.execute(
                    "INSERT INTO chunk_vec(rowid, embedding) VALUES (?,?)",
                    (rid, _vec_to_bytes(vec)),
                )
                st.embedded += 1
            conn.commit()
            if len(st.errors) >= max_errors:
                st.errors.append(f"错误达上限 {max_errors}，中止（已写入的向量已提交，可重跑续填）")
                break
    finally:
        if embedder is not None:
            embedder.close()
        if own_conn:
            conn.close()

    return st
