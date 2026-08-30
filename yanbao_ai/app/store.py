"""SQLite 存储层：建表 + 连接管理。

阶段 0 只需要 documents 表可用；chunks/facts/doc_themes/FTS5 先建空结构占位，
阶段 1+ 直接填充。向量表（sqlite-vec）留到阶段 1（依赖扩展加载）再建。

一切以 doc_id（内容哈希，= canonical 目录名 = sha256[:32]）为锚，幂等可重跑。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- 文档主表（manifest 为主 + 文件系统校验重建）
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,   -- = canonical 目录名 = sha256[:32]
    sha256          TEXT,               -- 完整 64 位哈希
    title           TEXT,
    institution     TEXT,
    category        TEXT,               -- 国内券商报告 / 投行报告
    lang            TEXT,               -- zh / en
    report_date     TEXT,               -- ISO: 2026-06-26
    week            TEXT,               -- 第一周 / 第二周
    month           TEXT,               -- 2026年7月
    stock_code      TEXT,
    page_count      INTEGER,
    md_path         TEXT,
    images_dir      TEXT,
    char_count      INTEGER,
    image_count     INTEGER,
    source_filename TEXT,               -- 磁盘真实原名
    source_relative TEXT,               -- manifest 相对路径
    pipeline_version INTEGER NOT NULL DEFAULT 1,
    needs_review    INTEGER NOT NULL DEFAULT 0,
    review_reason   TEXT,
    indexed_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_documents_report_date ON documents(report_date);
CREATE INDEX IF NOT EXISTS idx_documents_institution ON documents(institution);
CREATE INDEX IF NOT EXISTS idx_documents_category    ON documents(category);
CREATE INDEX IF NOT EXISTS idx_documents_needs_review ON documents(needs_review);

-- 以下为阶段 1+ 占位结构（阶段 0 不写入）
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,      -- doc_id#序号
    doc_id       TEXT NOT NULL,
    seq          INTEGER NOT NULL,
    heading_path TEXT,
    text         TEXT NOT NULL,
    char_start   INTEGER,
    char_end     INTEGER,
    image_refs   TEXT,                  -- JSON
    token_est    INTEGER,
    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS facts (
    fact_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT NOT NULL,
    report_date TEXT,
    entity      TEXT,
    entity_code TEXT,
    metric      TEXT,
    value_num   REAL,
    value_text  TEXT,
    unit        TEXT,
    direction   TEXT,
    as_of_date  TEXT,
    quote       TEXT,
    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
);
CREATE INDEX IF NOT EXISTS idx_facts_entity ON facts(entity);
CREATE INDEX IF NOT EXISTS idx_facts_metric ON facts(metric);

CREATE TABLE IF NOT EXISTS doc_themes (
    doc_id TEXT NOT NULL,
    theme  TEXT NOT NULL,
    PRIMARY KEY(doc_id, theme),
    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
);
CREATE INDEX IF NOT EXISTS idx_facts_report_date ON facts(report_date);
CREATE INDEX IF NOT EXISTS idx_facts_entity_code ON facts(entity_code);
CREATE INDEX IF NOT EXISTS idx_doc_themes_theme ON doc_themes(theme);

-- 抽取进度跟踪：记录每篇已抽取的模型/版本，用于增量续跑与升级重抽（幂等）。
CREATE TABLE IF NOT EXISTS extraction_log (
    doc_id       TEXT PRIMARY KEY,
    model        TEXT,
    schema_ver   INTEGER NOT NULL DEFAULT 1,
    fact_count   INTEGER,
    theme_count  INTEGER,
    extracted_at TEXT,
    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
);

-- 产业链结构（AI + 网络搜索一次性构建，落库后稳定不常变；趋势面板下钻用）。
-- 层级：theme（大主题，如 国产算力）→ segment_group（上/中/下游或需求/供给侧）
--   → segment（具体环节，如 光模块/CPO、半导体设备）。每个环节挂代表标的（tickers）
--   与关键词（keywords，供该环节的中外研报计数 LIKE 匹配）。
-- node_type: 'group'（分组行，tickers/keywords 可空）| 'segment'（叶子环节）。
-- 一条链一次构建、人工/研报佐证后入库；node_id 稳定，便于缓存与下钻定位。
CREATE TABLE IF NOT EXISTS industry_chain (
    node_id       TEXT PRIMARY KEY,       -- 稳定 id：theme|group|segment slug 拼接
    theme         TEXT NOT NULL,          -- 大主题：国产算力 / 半导体材料
    node_type     TEXT NOT NULL,          -- group / segment
    parent_id     TEXT,                   -- 上级 node_id（group 的 parent 为 NULL）
    seq           INTEGER NOT NULL DEFAULT 0,  -- 同级展示顺序
    name          TEXT NOT NULL,          -- 环节/分组名
    stage         TEXT,                   -- 上游/中游/下游 或 需求侧/供给侧
    summary       TEXT,                   -- 一句话定位（壁垒/国产化率/格局）
    tickers       TEXT,                   -- JSON 数组：代表标的 [{name,code,role}]
    keywords      TEXT,                   -- JSON 数组：该环节检索关键词（中英）
    built_by      TEXT,                   -- 构建来源标记：model / web / manual
    built_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_chain_theme  ON industry_chain(theme);
CREATE INDEX IF NOT EXISTS idx_chain_parent ON industry_chain(parent_id);

-- 趋势解读缓存：按（范围键 + 时间桶）缓存 AI 解读结果，同周同环节再点直接读库，
-- 不重复烧钱；新一周才重算。scope_key 统一编码查询范围（主题名 或 chain 节点 id），
-- period 为时间桶（如 '2026-W27' 或 'all'=不分周的总解读）。
CREATE TABLE IF NOT EXISTS trend_cache (
    scope_key    TEXT NOT NULL,           -- 主题名 / chain node_id / 'theme:xxx'
    period       TEXT NOT NULL,           -- 时间桶键：'2026-W27' / '2026-07' / 'all'
    by_unit      TEXT NOT NULL DEFAULT 'month',  -- month / week（生成时的粒度）
    payload      TEXT NOT NULL,           -- JSON：{interpretation, buckets, split, model}
    model        TEXT,
    input_hash   TEXT,                    -- 底层计数的哈希，数据变了可判失效
    created_at   TEXT,
    PRIMARY KEY(scope_key, period, by_unit)
);

-- 产业链漂移候选（chain_candidate）：漂移检测发现「库里热度够、但现有链未覆盖」的
-- 方向时，写到这里**等人审**，绝不自动写进正式链（industry_chain）。理由：自动写入
-- 会让链的权威性被噪声污染，而链是判断核心标的的骨架，脏了下游全脏。
-- 审核状态只三态（故意不设 accepted）：
--   watching  观察中——像个方向但证据还不够厚，下次再看
--   merged    已并入——已通过 build-chain 重建把它纳入正式链，本行留档
--   rejected  已否决——题材包装/与本链无关，之后不再提示
-- 为什么没有 accepted：「接受」的真实动作是重建那条链，不是在候选表打个勾。
-- 若允许标 accepted 而链里其实没有，就造出了「标了却没生效」的不一致状态。
CREATE TABLE IF NOT EXISTS chain_candidate (
    theme        TEXT NOT NULL,           -- 归属的链主题（如 国产算力）
    cand_key     TEXT NOT NULL,           -- 候选方向的规范形（segnorm.canonical）
    name         TEXT NOT NULL,           -- 原始展示名（取代表写法）
    doc_count    INTEGER NOT NULL DEFAULT 0,  -- 支持文档数
    inst_count   INTEGER NOT NULL DEFAULT 0,  -- 提及机构数（跨机构才可信）
    first_seen   TEXT,                    -- 最早出现的 report_date
    last_seen    TEXT,                    -- 最近出现的 report_date
    sample_docs  TEXT,                    -- JSON：代表研报 [{doc_id,title,institution,report_date}]
    status       TEXT NOT NULL DEFAULT 'watching',  -- watching / merged / rejected
    note         TEXT,
    detected_at  TEXT,
    reviewed_at  TEXT,
    PRIMARY KEY(theme, cand_key)
);
CREATE INDEX IF NOT EXISTS idx_cand_status ON chain_candidate(status);

-- 全文检索（BM25），阶段 1 填充。
-- 分词策略经探针实证后定为 **jieba 预分词 + unicode61**（见 app/segment.py）：
--   trigram 对中文有两处硬伤——2 字词（茅台/批价）短于 3 字下限 MATCH 必空；
--   长句被切成连续 3-gram 导致假阳性排序（`商品板块二季度` 误判高相关）。
--   改为索引侧用 jieba 把正文分词后按空格连接再入 FTS，unicode61 按空格切词元，
--   查询侧同样 jieba 分词——词元对齐，BM25 按概念级词频排序，2 字词正常命中。
-- 注意：因存的是「分词后文本」而非原文，chunks_fts 不能再用 content='chunks'
--   外部内容模式（否则 FTS 会按 rowid 回读 chunks.text 原文，与写入的分词文本不符）。
--   改为独立存储（contentless 不行，需 bm25 打分），rowid 仍与 chunks 对齐。
--   remove_diacritics=2 让检索对变音符号不敏感（英文语料友好）。
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    tokenize="unicode61 remove_diacritics 2"
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    """打开连接并确保 schema 存在。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def load_vec(conn: sqlite3.Connection) -> bool:
    """加载 sqlite-vec 扩展。成功返回 True；环境不支持时返回 False（可退纯词法）。"""
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return True
    except Exception:
        return False


def ensure_vec_table(conn: sqlite3.Connection, dim: int) -> None:
    """建向量表（vec0，维度须与 embedding 配置一致）。需先 load_vec 成功。

    chunk_vec.rowid 对齐 chunks.rowid，便于与 chunks/chunks_fts JOIN。
    """
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec USING vec0("
        f"embedding FLOAT[{dim}])"
    )


def connect_with_vec(db_path: Path, dim: int) -> tuple[sqlite3.Connection, bool]:
    """连接 + 加载 sqlite-vec + 建向量表。返回 (conn, vec_ok)。"""
    conn = connect(db_path)
    vec_ok = load_vec(conn)
    if vec_ok:
        ensure_vec_table(conn, dim)
    return conn, vec_ok


def count_documents(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]


def count_chunks(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


def get_chunk(conn: sqlite3.Connection, chunk_id: str) -> dict | None:
    """按 chunk_id 取一块原文 + 所属文档元数据，供引用溯源（点开 [n] 看原文）。

    返回 None 表示无此块。text 是切块后的原文（未分词），带机构/标题/发布日/标题路径，
    让用户能把 AI 论断和真实研报段落逐句对照，判断是否"真如研报所说"。
    """
    row = conn.execute(
        "SELECT c.chunk_id, c.doc_id, c.seq, c.heading_path, c.text, "
        "d.title, d.institution, d.category, d.report_date "
        "FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
        "WHERE c.chunk_id = ?",
        (chunk_id,),
    ).fetchone()
    return dict(row) if row else None


def count_fts(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]


def rebuild_fts(conn: sqlite3.Connection, *, batch: int = 2000) -> int:
    """从既有 chunks 重建 chunks_fts（jieba 分词后入库）。返回重建条数。

    分词策略变更（trigram → jieba+unicode61）后用：清空 FTS，逐块把 chunks.text
    分词后按 rowid 对齐重写。**不重新切块、不重新 embed**——只重刷全文索引，几分钟级。
    幂等：可反复跑，每次都从 chunks 原文重新分词填充。
    """
    from .segment import segment_for_index

    # schema 变更（trigram → unicode61 且不再是 content='chunks'）后，旧表定义仍在库里，
    # CREATE IF NOT EXISTS 不会改它 —— 必须先 DROP 再按新 SCHEMA 重建，否则分词器不生效。
    conn.execute("DROP TABLE IF EXISTS chunks_fts")
    conn.execute(
        "CREATE VIRTUAL TABLE chunks_fts USING fts5("
        "text, tokenize=\"unicode61 remove_diacritics 2\")"
    )
    n = 0
    rows = conn.execute("SELECT rowid, text FROM chunks ORDER BY rowid").fetchall()
    for rid, text in rows:
        conn.execute(
            "INSERT INTO chunks_fts(rowid, text) VALUES (?,?)",
            (rid, segment_for_index(text)),
        )
        n += 1
        if n % batch == 0:
            conn.commit()
    conn.commit()
    return n


def delete_doc_chunks(conn: sqlite3.Connection, doc_id: str, vec_ok: bool) -> None:
    """删除某文档的所有块（chunks/FTS/向量），用于重建时先清后写，保证幂等。"""
    rows = conn.execute(
        "SELECT rowid FROM chunks WHERE doc_id=?", (doc_id,)
    ).fetchall()
    rowids = [r[0] for r in rows]
    for rid in rowids:
        conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (rid,))
        if vec_ok:
            conn.execute("DELETE FROM chunk_vec WHERE rowid=?", (rid,))
    conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))


def write_chunk(
    conn: sqlite3.Connection,
    chunk_id: str,
    doc_id: str,
    seq: int,
    heading_path: str,
    text: str,
    char_start: int,
    char_end: int,
    image_refs_json: str,
    token_est: int,
    embedding: bytes | None,
    vec_ok: bool,
) -> None:
    """写一个块到 chunks + chunks_fts（+ chunk_vec，若有向量）。rowid 三表对齐。"""
    cur = conn.execute(
        "INSERT INTO chunks(chunk_id, doc_id, seq, heading_path, text, "
        "char_start, char_end, image_refs, token_est) VALUES (?,?,?,?,?,?,?,?,?)",
        (chunk_id, doc_id, seq, heading_path, text, char_start, char_end,
         image_refs_json, token_est),
    )
    rowid = cur.lastrowid
    # FTS 存 jieba 分词后的文本（空格分隔），与查询侧分词对齐；见 segment.py。
    from .segment import segment_for_index

    conn.execute(
        "INSERT INTO chunks_fts(rowid, text) VALUES (?,?)",
        (rowid, segment_for_index(text)),
    )
    if vec_ok and embedding is not None:
        conn.execute(
            "INSERT INTO chunk_vec(rowid, embedding) VALUES (?,?)", (rowid, embedding)
        )


# ---- Facts / Themes（阶段 2 时间线骨架）----

def delete_doc_facts(conn: sqlite3.Connection, doc_id: str) -> None:
    """删除某文档的 facts + doc_themes，用于重抽前先清后写，保证幂等。"""
    conn.execute("DELETE FROM facts WHERE doc_id=?", (doc_id,))
    conn.execute("DELETE FROM doc_themes WHERE doc_id=?", (doc_id,))


def write_fact(
    conn: sqlite3.Connection,
    doc_id: str,
    report_date: str | None,
    entity: str | None,
    entity_code: str | None,
    metric: str | None,
    value_num: float | None,
    value_text: str | None,
    unit: str | None,
    direction: str | None,
    as_of_date: str | None,
    quote: str | None,
) -> None:
    """写一条结构化事实。"""
    conn.execute(
        "INSERT INTO facts(doc_id, report_date, entity, entity_code, metric, "
        "value_num, value_text, unit, direction, as_of_date, quote) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (doc_id, report_date, entity, entity_code, metric, value_num,
         value_text, unit, direction, as_of_date, quote),
    )


def write_theme(conn: sqlite3.Connection, doc_id: str, theme: str) -> None:
    """写一个主题标签（同 doc 同 theme 幂等）。"""
    conn.execute(
        "INSERT OR IGNORE INTO doc_themes(doc_id, theme) VALUES (?,?)",
        (doc_id, theme),
    )


def log_extraction(
    conn: sqlite3.Connection,
    doc_id: str,
    model: str,
    schema_ver: int,
    fact_count: int,
    theme_count: int,
) -> None:
    """记录抽取进度（幂等，重抽覆盖）。"""
    conn.execute(
        "INSERT INTO extraction_log(doc_id, model, schema_ver, fact_count, "
        "theme_count, extracted_at) VALUES (?,?,?,?,?,datetime('now')) "
        "ON CONFLICT(doc_id) DO UPDATE SET model=excluded.model, "
        "schema_ver=excluded.schema_ver, fact_count=excluded.fact_count, "
        "theme_count=excluded.theme_count, extracted_at=excluded.extracted_at",
        (doc_id, model, schema_ver, fact_count, theme_count),
    )


def extracted_doc_ids(conn: sqlite3.Connection, schema_ver: int) -> set[str]:
    """返回已按当前 schema 版本抽取过的 doc_id 集合，供增量跳过。"""
    rows = conn.execute(
        "SELECT doc_id FROM extraction_log WHERE schema_ver=?", (schema_ver,)
    ).fetchall()
    return {r[0] for r in rows}


def count_facts(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]


# ---- 产业链结构（industry_chain）----

def delete_chain(conn: sqlite3.Connection, theme: str) -> None:
    """删除某主题的整条链（重建前先清，保证幂等）。"""
    conn.execute("DELETE FROM industry_chain WHERE theme=?", (theme,))


def write_chain_node(
    conn: sqlite3.Connection,
    node_id: str,
    theme: str,
    node_type: str,
    parent_id: str | None,
    seq: int,
    name: str,
    stage: str | None,
    summary: str | None,
    tickers_json: str | None,
    keywords_json: str | None,
    built_by: str,
) -> None:
    """写一个产业链节点（幂等覆盖 node_id）。"""
    conn.execute(
        "INSERT INTO industry_chain(node_id, theme, node_type, parent_id, seq, "
        "name, stage, summary, tickers, keywords, built_by, built_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,datetime('now')) "
        "ON CONFLICT(node_id) DO UPDATE SET theme=excluded.theme, "
        "node_type=excluded.node_type, parent_id=excluded.parent_id, "
        "seq=excluded.seq, name=excluded.name, stage=excluded.stage, "
        "summary=excluded.summary, tickers=excluded.tickers, "
        "keywords=excluded.keywords, built_by=excluded.built_by, "
        "built_at=excluded.built_at",
        (node_id, theme, node_type, parent_id, seq, name, stage, summary,
         tickers_json, keywords_json, built_by),
    )


def get_chain(conn: sqlite3.Connection, theme: str) -> list[dict]:
    """取某主题整条链的所有节点（按 parent 分组、seq 升序），返回 dict 列表。"""
    rows = conn.execute(
        "SELECT node_id, theme, node_type, parent_id, seq, name, stage, "
        "summary, tickers, keywords, built_by, built_at "
        "FROM industry_chain WHERE theme=? "
        "ORDER BY (parent_id IS NOT NULL), parent_id, seq",
        (theme,),
    ).fetchall()
    return [dict(r) for r in rows]


def list_chain_themes(conn: sqlite3.Connection) -> list[str]:
    """已构建产业链的主题名列表（供前端下拉/入口）。"""
    rows = conn.execute(
        "SELECT DISTINCT theme FROM industry_chain ORDER BY theme"
    ).fetchall()
    return [r[0] for r in rows]


# ---- 产业链漂移候选（chain_candidate）----

def upsert_candidate(
    conn: sqlite3.Connection,
    theme: str,
    cand_key: str,
    name: str,
    doc_count: int,
    inst_count: int,
    first_seen: str | None,
    last_seen: str | None,
    sample_docs_json: str | None,
) -> None:
    """写入/刷新一条漂移候选。**已审过的行只更新统计，不覆盖 status/note**——
    否则每次检测都会把用户标的 rejected 冲回 watching，同一个噪声反复来烦人。
    """
    conn.execute(
        "INSERT INTO chain_candidate(theme, cand_key, name, doc_count, inst_count, "
        "first_seen, last_seen, sample_docs, status, detected_at) "
        "VALUES (?,?,?,?,?,?,?,?,'watching',datetime('now')) "
        "ON CONFLICT(theme, cand_key) DO UPDATE SET "
        "name=excluded.name, doc_count=excluded.doc_count, "
        "inst_count=excluded.inst_count, first_seen=excluded.first_seen, "
        "last_seen=excluded.last_seen, sample_docs=excluded.sample_docs, "
        "detected_at=excluded.detected_at",
        (theme, cand_key, name, doc_count, inst_count,
         first_seen, last_seen, sample_docs_json),
    )


def list_candidates(
    conn: sqlite3.Connection,
    theme: str | None = None,
    status: str | None = None,
) -> list[dict]:
    """列出漂移候选。默认全部；可按主题/状态过滤。按支持机构数、文档数降序。"""
    import json as _json

    clauses, params = [], []
    if theme:
        clauses.append("theme=?")
        params.append(theme)
    if status:
        clauses.append("status=?")
        params.append(status)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = conn.execute(
        "SELECT theme, cand_key, name, doc_count, inst_count, first_seen, "
        "last_seen, sample_docs, status, note, detected_at, reviewed_at "
        f"FROM chain_candidate{where} "
        "ORDER BY inst_count DESC, doc_count DESC",
        params,
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["sample_docs"] = _json.loads(d.get("sample_docs") or "[]")
        except (ValueError, TypeError):
            d["sample_docs"] = []
        out.append(d)
    return out


def set_candidate_status(
    conn: sqlite3.Connection, theme: str, cand_key: str,
    status: str, note: str | None = None,
) -> bool:
    """标记候选的审核状态（watching / merged / rejected）。返回是否命中行。"""
    if status not in ("watching", "merged", "rejected"):
        raise ValueError(f"非法状态：{status}（只允许 watching/merged/rejected）")
    cur = conn.execute(
        "UPDATE chain_candidate SET status=?, note=COALESCE(?, note), "
        "reviewed_at=datetime('now') WHERE theme=? AND cand_key=?",
        (status, note, theme, cand_key),
    )
    return cur.rowcount > 0


# ---- 趋势解读缓存（trend_cache）----

def get_trend_cache(
    conn: sqlite3.Connection, scope_key: str, period: str, by_unit: str
) -> dict | None:
    """读缓存的 AI 趋势解读。命中返回 {payload(dict), model, input_hash, created_at}。"""
    import json as _json

    row = conn.execute(
        "SELECT payload, model, input_hash, created_at FROM trend_cache "
        "WHERE scope_key=? AND period=? AND by_unit=?",
        (scope_key, period, by_unit),
    ).fetchone()
    if not row:
        return None
    try:
        payload = _json.loads(row["payload"])
    except (ValueError, TypeError):
        return None
    return {
        "payload": payload,
        "model": row["model"],
        "input_hash": row["input_hash"],
        "created_at": row["created_at"],
    }


def put_trend_cache(
    conn: sqlite3.Connection,
    scope_key: str,
    period: str,
    by_unit: str,
    payload_json: str,
    model: str | None,
    input_hash: str | None,
) -> None:
    """写/覆盖 AI 趋势解读缓存。"""
    conn.execute(
        "INSERT INTO trend_cache(scope_key, period, by_unit, payload, model, "
        "input_hash, created_at) VALUES (?,?,?,?,?,?,datetime('now')) "
        "ON CONFLICT(scope_key, period, by_unit) DO UPDATE SET "
        "payload=excluded.payload, model=excluded.model, "
        "input_hash=excluded.input_hash, created_at=excluded.created_at",
        (scope_key, period, by_unit, payload_json, model, input_hash),
    )
