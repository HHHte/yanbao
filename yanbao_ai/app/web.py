"""Web：本地 FastAPI 界面（方案 §6.7 阶段 3）。对话/引用/时间线，绑定 127.0.0.1。

安全边界（重要）：
- 仅监听 127.0.0.1，**无鉴权**——面向本机单用户。绝不要暴露到公网或 0.0.0.0：
  否则任何人都能查询（付费）研报库、触发 Claude/OpenAI 计费。
- 密钥只在后端使用，绝不下发前端；前端只拿到答案文本、来源元数据、时间线数据。

端点：
- GET  /                → 单文件前端（对话 + 时间线两个面板）。
- POST /api/ask         → {query, filters, lexical, cheap} → 答案 + 来源。
- POST /api/retrieve    → 只检索不生成（零 LLM 成本，看命中）。
- GET  /api/timeline/metric → 指标时间线。
- GET  /api/timeline/theme  → 主题热度（按月/周或总榜）。
- GET  /api/doc/{doc_id}    → 文档元数据 + 分块概览（溯源用）。
- GET  /api/images/{doc_id}/{name} → 内联 Exhibit 图（限定 canonical 下，防穿越）。
- GET  /api/health          → 端点/库状态（key 脱敏）。

注意：本模块**不**使用 `from __future__ import annotations`。FastAPI 需要在运行时
解析路由参数注解，而请求模型（AskReq）是 create_app 内的局部类；若注解被延迟成字符串，
FastAPI 无法解析该前向引用，会把请求体模型误当查询参数。Python 3.13 原生支持
`X | None` 运行时求值，无需 future 导入。
"""
from .config import Config, load_config


def create_app(cfg: Config | None = None):
    """构造 FastAPI 应用。cfg 为空时按默认加载。"""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, FileResponse
    from pydantic import BaseModel

    cfg = cfg or load_config()
    app = FastAPI(title="研报本地 AI", docs_url="/api/docs")

    # ---- 请求模型 ----
    class AskReq(BaseModel):
        query: str
        category: str | None = None
        institution: str | None = None
        stock_code: str | None = None
        lang: str | None = None
        date_from: str | None = None
        date_to: str | None = None
        limit: int = 12
        lexical: bool = False
        cheap: bool = False
        # analyst=True → 产业链分析模式（带行业理解梳理核心标的），否则事实问答。
        analyst: bool = False
        # history：既往对话轮次，用于追问。[{"role":"user"/"assistant","content":str}]
        history: list[dict] | None = None

    def _filters(r: AskReq):
        from .retrieve import Filters

        return Filters(
            date_from=r.date_from, date_to=r.date_to,
            institution=r.institution, category=r.category,
            stock_code=r.stock_code, lang=r.lang,
        )

    # ---- 页面 ----
    @app.get("/", response_class=HTMLResponse)
    def index():
        return _INDEX_HTML

    # ---- 健康检查（脱敏）----
    @app.get("/api/health")
    def health():
        import sqlite3

        from . import store

        out = {
            "canonical_ok": cfg.paths.canonical.is_dir(),
            "manifest_ok": cfg.paths.manifest.is_file(),
            "db": str(cfg.paths.db),
            "llm_base_url": cfg.llm.base_url or None,
            "llm_source": cfg.llm.source,
            "llm_key": cfg.llm.key_redacted,
            "embed_model": f"{cfg.embed.model}@{cfg.embed.dimensions}",
            "embed_key": cfg.embed.key_redacted,
            # 前端据此显示"检索：BM25"还是"BM25+向量"。开关关掉时 key 仍在（备查），
            # 所以不能拿 embed_key 推断检索模式——必须单独报。
            "embed_enabled": cfg.embed.enabled,
            "retrieval_mode": "BM25 + 向量" if cfg.embed.usable else "纯 BM25",
        }
        if cfg.paths.db.is_file():
            try:
                con = sqlite3.connect(cfg.paths.db)
                out["documents"] = store.count_documents(con)
                out["chunks"] = store.count_chunks(con)
                try:
                    out["facts"] = store.count_facts(con)
                except sqlite3.Error:
                    out["facts"] = 0
                con.close()
            except sqlite3.Error as exc:
                out["db_error"] = str(exc)
        return out

    # ---- 机构列表（供前端下拉框；按覆盖文档数降序）----
    @app.get("/api/institutions")
    def api_institutions():
        import sqlite3

        if not cfg.paths.db.is_file():
            return {"institutions": []}
        con = sqlite3.connect(cfg.paths.db)
        try:
            rows = con.execute(
                "SELECT institution, COUNT(*) c FROM documents "
                "WHERE institution IS NOT NULL AND TRIM(institution) <> '' "
                "GROUP BY institution ORDER BY c DESC"
            ).fetchall()
            # 只保留覆盖 >=3 篇的机构，滤掉解析噪声（如把标题误当机构的长尾单篇）。
            # 返回 {name,count} 对象，与前端下拉框（读 it.name/it.count）对齐。
            return {"institutions": [
                {"name": r[0], "count": r[1]} for r in rows if r[1] >= 3
            ]}
        finally:
            con.close()

    # ---- 检索（无 LLM）----
    @app.post("/api/retrieve")
    def api_retrieve(r: AskReq):
        from .retrieve import retrieve

        res = retrieve(
            cfg, r.query, filters=_filters(r), limit=r.limit,
            mode="lexical" if r.lexical else "auto",
        )
        return {
            "mode": res.mode,
            "used_like": res.used_like,
            "dense_ok": res.dense_ok,
            "hits": [
                {
                    "chunk_id": h.chunk_id, "doc_id": h.doc_id, "seq": h.seq,
                    "heading_path": h.heading_path, "text": h.text,
                    "title": h.title, "institution": h.institution,
                    "category": h.category, "report_date": h.report_date,
                    "score": h.score,
                }
                for h in res.hits
            ],
        }

    # ---- 问答（检索 + 生成）----
    @app.post("/api/ask")
    def api_ask(r: AskReq):
        from . import generate as gen_mod

        # 产业链分析模式走 analyze（多路检索）：先把问题拆成子环节、每环各检索一次、
        # 合并去重再综合，规避"单次 top-K 向量漏掉整条支线"的召回缺口（如国产算力漏超节点
        # 系统层）。非分析模式仍走 ask（单路检索 + 事实问答提示）。history 支持追问。
        subqueries: list[str] = []
        try:
            if r.analyst:
                answer, res, subqueries = gen_mod.analyze(
                    cfg, r.query, filters=_filters(r),
                    mode="lexical" if r.lexical else "auto",
                    model_source="cheap" if r.cheap else "gen",
                    history=r.history,
                )
            else:
                answer, res = gen_mod.ask(
                    cfg, r.query, filters=_filters(r), limit=r.limit,
                    mode="lexical" if r.lexical else "auto",
                    model_source="cheap" if r.cheap else "gen",
                    system=gen_mod._SYSTEM, history=r.history,
                )
        except gen_mod.GenerateError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        return {
            "answer": answer.text,
            "model": answer.model,
            "mode": res.mode,
            "used_like": res.used_like,
            "analyst": r.analyst,
            "subqueries": subqueries,   # 分析模式下拆出的子环节检索词，供前端展示"查了哪些线"
            "sources": [
                {
                    "ref": s.ref, "chunk_id": s.chunk_id, "doc_id": s.doc_id,
                    "title": s.title, "institution": s.institution,
                    "report_date": s.report_date, "heading_path": s.heading_path,
                }
                for s in answer.sources
            ],
        }

    # ---- 时间线 ----
    @app.get("/api/timeline/metric")
    def api_tl_metric(entity: str, metric: str | None = None,
                      date_from: str | None = None, date_to: str | None = None):
        from . import timeline as tl

        s = tl.metric_timeline(cfg, entity, metric,
                               date_from=date_from, date_to=date_to)
        return {
            "entity": s.entity, "metric": s.metric,
            "points": [
                {
                    "date": p.date, "value_num": p.value_num,
                    "value_text": p.value_text, "unit": p.unit,
                    "direction": p.direction, "entity": p.entity,
                    "metric": p.metric, "institution": p.institution,
                    "title": p.title, "doc_id": p.doc_id, "quote": p.quote,
                    "norm_num": p.norm_num, "norm_unit": p.norm_unit,
                    "report_date": p.report_date, "as_of_date": p.as_of_date,
                }
                for p in s.points
            ],
        }

    # ---- 时间线 AI 解读（按需，才产生 LLM 费用）----
    # 混合方案：SQL 先零成本索引出候选 facts（api_tl_metric），用户点「AI 解读」时
    # 才把这批碎片交给 Claude 读成连贯叙述——分清真时间序列与单篇多年预测横截面、
    # 去重纠口径。不点则永不调用 LLM，时间线表格/图仍是确定性、免费的。
    @app.get("/api/timeline/metric/interpret")
    def api_tl_interpret(entity: str, metric: str | None = None,
                         date_from: str | None = None, date_to: str | None = None,
                         cheap: bool = False):
        from . import timeline as tl
        from . import generate as gen_mod

        s = tl.metric_timeline(cfg, entity, metric,
                               date_from=date_from, date_to=date_to)
        try:
            ans = gen_mod.interpret_timeline(
                cfg, s.entity, s.metric, s.points,
                model_source="cheap" if cheap else "gen",
            )
        except gen_mod.GenerateError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        return {
            "entity": s.entity, "metric": s.metric,
            "points": len(s.points),
            "model": ans.model,
            "interpretation": ans.text,
        }

    @app.get("/api/timeline/theme")
    def api_tl_theme(theme: str | None = None, week: bool = False, top: int = 30):
        from . import timeline as tl

        if theme:
            buckets = tl.theme_heat(cfg, theme, by="week" if week else "month")
        else:
            buckets = tl.top_themes(cfg, limit=top)
        return {
            "theme": theme, "by": "week" if week else "month",
            "buckets": [{"bucket": b.bucket, "count": b.count} for b in buckets],
            "split": tl.domestic_foreign_split(cfg, theme),
        }

    # ---- 主题趋势 AI 解读（按需，才产生 LLM 费用）----
    # 时间线面板的「AI 查询」：SQL 先零成本聚出主题热度桶（api_tl_theme），点 AI 查询时
    # 才把这批计数交给 Claude 读成「近期研报在关注什么、哪些在升温/降温」的趋势解读。
    @app.get("/api/timeline/theme/interpret")
    def api_tl_theme_interpret(theme: str | None = None, week: bool = False,
                               top: int = 30, cheap: bool = False,
                               refresh: bool = False):
        import hashlib
        import json as _json

        from . import timeline as tl
        from . import generate as gen_mod
        from . import store

        by_unit = "week" if week else "month"
        if theme:
            buckets = tl.theme_heat(cfg, theme, by=by_unit)
        else:
            buckets = tl.top_themes(cfg, limit=top)
        bucket_list = [{"bucket": b.bucket, "count": b.count} for b in buckets]
        split = tl.domestic_foreign_split(cfg, theme)

        # 缓存键：范围（主题 or 总榜）+ 当前时间桶 + 粒度。底层计数变了（input_hash 不同）
        # 或显式 refresh 时才重算，否则读库直返，同周同范围不重复烧钱。
        scope_key = f"theme:{theme}" if theme else "board"
        # period 取最新的时间桶（该范围数据的“新鲜度”锚点）；总榜用 'all'。
        real = [b for b in bucket_list if b["bucket"] != "未知"]
        period = (max(b["bucket"] for b in real) if (theme and real) else "all")
        input_hash = hashlib.sha256(
            _json.dumps([bucket_list, split], ensure_ascii=False, sort_keys=True)
            .encode("utf-8")
        ).hexdigest()[:16]

        con = store.connect(cfg.paths.db)
        try:
            if not refresh:
                cached = store.get_trend_cache(con, scope_key, period, by_unit)
                if cached and cached.get("input_hash") == input_hash:
                    p = cached["payload"]
                    return {
                        "theme": theme, "by": by_unit,
                        "buckets": p.get("buckets", bucket_list),
                        "split": p.get("split", split),
                        "model": cached.get("model"),
                        "interpretation": p.get("interpretation", ""),
                        "cached": True, "period": period,
                        "cached_at": cached.get("created_at"),
                    }
            pairs = [(b["bucket"], b["count"]) for b in bucket_list]
            try:
                ans = gen_mod.interpret_theme_trend(
                    cfg, pairs, theme=theme, by=by_unit,
                    model_source="cheap" if cheap else "gen",
                )
            except gen_mod.GenerateError as exc:
                raise HTTPException(status_code=503, detail=str(exc))
            payload = {
                "interpretation": ans.text, "buckets": bucket_list, "split": split,
            }
            store.put_trend_cache(
                con, scope_key, period, by_unit,
                _json.dumps(payload, ensure_ascii=False), ans.model, input_hash,
            )
            con.commit()
            return {
                "theme": theme, "by": by_unit,
                "buckets": bucket_list, "split": split,
                "model": ans.model, "interpretation": ans.text,
                "cached": False, "period": period,
            }
        finally:
            con.close()

    # ---- 产业链结构（已构建落库的主题）----
    # 趋势面板下钻用：列出已建链的主题；读某主题的 分组→环节 树（每环节带代表标的 +
    # 中外研报占比）。纯读库 + SQL 计数，零 LLM 成本（构建时才烧 token，之后免费）。
    @app.get("/api/chain/themes")
    def api_chain_themes():
        from . import store

        con = store.connect(cfg.paths.db)
        try:
            return {"themes": store.list_chain_themes(con)}
        finally:
            con.close()

    @app.get("/api/chain")
    def api_chain(theme: str):
        from . import chain as chain_mod

        view = chain_mod.get_chain_view(cfg, theme)
        if not view.get("groups"):
            raise HTTPException(
                status_code=404,
                detail=f"主题「{theme}」尚未构建产业链（先跑 build-chain）",
            )
        return view

    # ---- 链路 AI 解读：顺着已落库的产业链结构，多路检索中外资研报 → 综合判断核心/格局 ----
    # 结构读库零成本，只有这步综合分析烧 token，故做成按钮 + 缓存（trend_cache，scope chain:X）。
    #
    # 缓存键是四要素合成的，**不是只有链结构**：
    #   structure_hash  分组/环节名（结构变了自然要重解读）
    #   evidence_hash   各环节当前命中的文档全集指纹（纯 SQL 现算，见 chain.evidence_signature）
    #   prompt_version  提示词/问题模板版本（改了提示词就该重算）
    #   model_id        换模型输出不同，不该复用
    # 早期只锚 structure_hash 是个真 bug：解读内容来自 analyze() 检索到的几十篇具体研报，
    # 新研报进来而结构没变时会永久返回过期解读。证据指纹解决了这点，且无需任何人
    # 手工维护版本计数器——派生数据不该靠人记得同步。
    @app.get("/api/chain/interpret")
    def api_chain_interpret(theme: str, cheap: bool = False, refresh: bool = False):
        import hashlib
        import json as _json

        from . import chain as chain_mod
        from . import store

        view = chain_mod.get_chain_view(cfg, theme)
        if not view.get("groups"):
            raise HTTPException(
                status_code=404,
                detail=f"主题「{theme}」尚未构建产业链（先跑 build-chain）",
            )
        struct_sig = [
            [g["name"], g.get("stage"), [s["name"] for s in g.get("segments", [])]]
            for g in view["groups"]
        ]
        structure_hash = hashlib.sha256(
            _json.dumps(struct_sig, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        # 证据指纹：纯 SQL，不调 LLM，所以「判断缓存是否有效」本身不花钱。
        ev = chain_mod.evidence_signature(cfg, theme)
        model_id = cfg.llm.model_cheap if cheap else cfg.llm.model_gen
        input_hash = hashlib.sha256("|".join([
            structure_hash,
            ev["evidence_hash"],
            chain_mod.INTERPRET_PROMPT_VERSION,
            model_id or "",
        ]).encode("utf-8")).hexdigest()[:16]
        scope_key = f"chain:{theme}"

        con = store.connect(cfg.paths.db)
        try:
            if not refresh:
                cached = store.get_trend_cache(con, scope_key, "all", "chain")
                if cached and cached.get("input_hash") == input_hash:
                    p = cached["payload"]
                    return {
                        "theme": theme,
                        "interpretation": p.get("interpretation", ""),
                        "model": cached.get("model"),
                        "cached": True,
                        "cached_at": cached.get("created_at"),
                        "evidence_docs": ev["docs"],
                    }
            try:
                ans, evidence_ids = chain_mod.interpret_chain(
                    cfg, theme, model_source="cheap" if cheap else "gen",
                )
            except chain_mod.ChainError as exc:
                raise HTTPException(status_code=503, detail=str(exc))
            payload = {
                "interpretation": ans.text,
                # 存下本次真正入 prompt 的 chunk 级证据，便于日后核对"这条解读是基于哪批材料"。
                "evidence_chunks": evidence_ids,
                "structure_hash": structure_hash,
                "evidence_hash": ev["evidence_hash"],
                "prompt_version": chain_mod.INTERPRET_PROMPT_VERSION,
            }
            store.put_trend_cache(
                con, scope_key, "all", "chain",
                _json.dumps(payload, ensure_ascii=False), ans.model, input_hash,
            )
            con.commit()
            return {
                "theme": theme, "interpretation": ans.text,
                "model": ans.model, "cached": False,
                "evidence_docs": ev["docs"],
            }
        finally:
            con.close()

    # ---- 主题下钻：某主题的"子分类" ----
    # 全库总榜的主题是扁平标签（doc_themes），本身无父子结构。要"点开大类看子分类、
    # 理解这个大类代表什么"，最有信息量的信号是**共现主题**：与该主题打在同一批文档上的
    # 其它主题——它们刻画了这个主题实际覆盖的方向。不同大类自然会共享子类（如「AIDC」与
    # 「液冷」互为共现），这种重叠是真实且有用的，不隐藏。
    # 若该主题已构建产业链，附带 has_chain=true，前端可引导去下钻看权威分段。
    @app.get("/api/theme/subcats")
    def api_theme_subcats(theme: str, limit: int = 12):
        from . import store

        con = store.connect(cfg.paths.db)
        try:
            # 该主题命中的文档集里，其它主题的共现文档数（降序）。
            rows = con.execute(
                "SELECT t2.theme AS theme, COUNT(DISTINCT t2.doc_id) AS c "
                "FROM doc_themes t1 JOIN doc_themes t2 ON t1.doc_id = t2.doc_id "
                "WHERE t1.theme = ? AND t2.theme != ? "
                "GROUP BY t2.theme ORDER BY c DESC LIMIT ?",
                (theme, theme, limit),
            ).fetchall()
            base = con.execute(
                "SELECT COUNT(DISTINCT doc_id) AS c FROM doc_themes WHERE theme = ?",
                (theme,),
            ).fetchone()
            total = base["c"] if base else 0
            has_chain = theme in set(store.list_chain_themes(con))
            return {
                "theme": theme,
                "total": total,
                "has_chain": has_chain,
                "subcats": [
                    {
                        "theme": r["theme"],
                        "count": r["c"],
                        "pct": round(r["c"] / total * 100, 1) if total else 0.0,
                    }
                    for r in rows
                ],
            }
        finally:
            con.close()

    # ---- 引用溯源：按 chunk_id 取该块研报原文（验证 AI 论断是否真有出处）----
    # 前端把答案里的 [n] 渲染成可点击，点开调此端点显示对应块原文 + 文档元数据，
    # 让用户一眼比对"AI 说的"与"研报写的"。只读，零 LLM 成本。
    @app.get("/api/chunk/{chunk_id}")
    def api_chunk(chunk_id: str):
        from . import store

        con = store.connect(cfg.paths.db)
        try:
            row = store.get_chunk(con, chunk_id)
            if not row:
                raise HTTPException(status_code=404, detail="未找到该引用块")
            return dict(row)
        finally:
            con.close()

    # ---- 文档溯源 ----
    @app.get("/api/doc/{doc_id}")
    def api_doc(doc_id: str):
        import sqlite3

        from . import store

        con = store.connect(cfg.paths.db)
        try:
            row = con.execute(
                "SELECT * FROM documents WHERE doc_id=?", (doc_id,)
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="未找到该文档")
            doc = dict(row)
            chunks = con.execute(
                "SELECT seq, heading_path, token_est FROM chunks "
                "WHERE doc_id=? ORDER BY seq", (doc_id,)
            ).fetchall()
            doc["chunk_count"] = len(chunks)
            doc["chunks"] = [dict(c) for c in chunks]
            return doc
        finally:
            con.close()

    # ---- 内联图片（防路径穿越：限定 canonical/<doc_id>/images 下）----
    @app.get("/api/images/{doc_id}/{name}")
    def api_image(doc_id: str, name: str):
        base = (cfg.paths.canonical / doc_id / "images").resolve()
        target = (base / name).resolve()
        # 目标必须严格位于 images 目录内，且真实存在
        if base not in target.parents or not target.is_file():
            raise HTTPException(status_code=404, detail="图片不存在")
        return FileResponse(target)

    return app


def serve(cfg: Config | None = None, host: str = "127.0.0.1", port: int = 8000):
    """启动本地服务。默认仅 127.0.0.1（无鉴权，勿暴露公网）。"""
    import uvicorn

    if host not in ("127.0.0.1", "localhost"):
        print(
            f"[警告] 你把服务绑到 {host}——本服务无鉴权，会暴露付费研报库与 "
            f"LLM 计费入口。仅在可信网络内这样做。"
        )
    app = create_app(cfg)
    uvicorn.run(app, host=host, port=port)


# ------------------------- 单文件前端 -------------------------
# 用 raw string：内嵌 JS 正则含 \[ \d \] 等反斜杠，普通字符串会触发 SyntaxWarning
# 并可能误解析转义。整段 HTML 无依赖 Python 转义的序列，全部按字面交给浏览器。
_INDEX_HTML = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>研报本地 AI</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --line:#262b36; --fg:#e6e8ec;
          --muted:#8b93a1; --accent:#4f8cff; --chip:#222835; }
  * { box-sizing:border-box; }
  /* 滚动条：与暗色主题一致的低调样式（原来是系统默认的亮色，突兀）。
     Firefox 用 scrollbar-width/color；Chromium/WebKit 用 ::-webkit-scrollbar。 */
  * { scrollbar-width:thin; scrollbar-color:#3a4152 transparent; }
  ::-webkit-scrollbar { width:10px; height:10px; }
  ::-webkit-scrollbar-track { background:transparent; }
  ::-webkit-scrollbar-thumb { background:#2c3340; border-radius:6px;
    border:2px solid var(--bg); }
  ::-webkit-scrollbar-thumb:hover { background:#3f4859; }
  ::-webkit-scrollbar-corner { background:transparent; }
  html,body { height:100%; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:15px/1.65 -apple-system,Segoe UI,Roboto,"Microsoft YaHei",sans-serif;
         display:flex; flex-direction:column; }
  header { padding:14px 20px; border-bottom:1px solid var(--line);
           display:flex; align-items:center; gap:14px; flex:0 0 auto; }
  header h1 { font-size:16px; margin:0; font-weight:600; }
  header .meta { color:var(--muted); font-size:12px; }
  .tabs { display:flex; gap:8px; margin-left:auto; }
  .tab { padding:6px 14px; border:1px solid var(--line); border-radius:8px;
         cursor:pointer; color:var(--muted); }
  .tab.on { color:var(--fg); border-color:var(--accent); }
  /* main 占满宽度、不再自己居中——这样滚动条落在窗口最右侧，而不是 1000px 内容框的
     中缝（之前滚轴卡在中间就是因为滚动发生在居中的 main 上）。内容由各 .wrap 居中。 */
  main { width:100%; margin:0; padding:0;
         flex:1 1 auto; min-height:0; display:flex; flex-direction:column; }
  .panel { display:none; flex:1 1 auto; min-height:0; }
  /* 趋势页整体滚动：滚动容器铺满宽度（滚轴贴右边缘），内部 .wrap 居中到 1000px */
  .panel.on { display:block; overflow-y:auto; }
  .wrap { max-width:1000px; margin:0 auto; padding:20px; }
  /* 问答页：对话区自适应滚动，输入 dock 固定在底部（像聊天应用，追问不用拉到底） */
  #p-ask.on { display:flex; flex-direction:column; overflow:hidden; }
  /* 对话滚动容器铺满宽（滚轴贴右），内部 #convo 居中到 1000px */
  #convo-scroll { flex:1 1 auto; min-height:0; overflow-y:auto; }
  .row { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px; }
  input,select,textarea { background:var(--panel); color:var(--fg);
    border:1px solid var(--line); border-radius:8px; padding:8px 10px; font:inherit; }
  input[type=text],textarea { flex:1; min-width:200px; }
  textarea { width:100%; resize:vertical; min-height:52px; }
  button { background:var(--accent); color:#fff; border:0; border-radius:8px;
    padding:8px 18px; cursor:pointer; font:inherit; }
  button.ghost { background:var(--chip); color:var(--fg); }
  button:disabled { opacity:.5; cursor:default; }
  .chips { display:flex; gap:6px; flex-wrap:wrap; }
  .chip { background:var(--chip); border:1px solid var(--line); border-radius:20px;
    padding:3px 12px; font-size:12px; color:var(--muted); cursor:pointer; }
  .answer { background:var(--panel); border:1px solid var(--line); border-radius:12px;
    padding:16px 18px; margin-top:14px; white-space:pre-wrap; }
  .src { border-top:1px solid var(--line); margin-top:14px; padding-top:10px; }
  .src h4 { margin:0 0 8px; font-size:13px; color:var(--muted); }
  .src-item { font-size:12.5px; padding:4px 0; color:var(--muted); }
  .src-item b { color:var(--fg); }
  /* 折叠的引用/检索详情：默认收起，点摘要行展开。正文 [n] 引用不受影响。 */
  .src-fold { margin-top:14px; border-top:1px solid var(--line); padding-top:8px; }
  .src-fold > summary { cursor:pointer; font-size:12.5px; color:var(--muted);
    list-style:none; user-select:none; padding:2px 0; }
  .src-fold > summary::-webkit-details-marker { display:none; }
  .src-fold > summary::before { content:"▸ "; color:var(--accent); }
  .src-fold[open] > summary::before { content:"▾ "; }
  .src-fold > summary:hover { color:var(--fg); }
  .src-sec { margin-top:8px; }
  .src-lbl { font-size:12px; color:var(--muted); margin-bottom:4px; }
  .hit { background:var(--panel); border:1px solid var(--line); border-radius:10px;
    padding:12px 14px; margin-top:10px; }
  .hit .h-meta { font-size:12px; color:var(--muted); margin-bottom:6px; }
  .hit .h-text { font-size:13px; max-height:120px; overflow:auto; }
  .status { color:var(--muted); font-size:12.5px; margin-top:8px; min-height:18px; }
  .err { color:#ff6b6b; }
  table { border-collapse:collapse; width:100%; margin-top:12px; }
  th,td { border:1px solid var(--line); padding:6px 10px; text-align:left; font-size:13px; }
  th { color:var(--muted); font-weight:500; }
  .bar { display:inline-block; height:10px; background:var(--accent); border-radius:3px; }
  /* 总榜主题行可点击下钻看共现子分类 */
  .th-row { cursor:pointer; }
  .th-row:hover { background:var(--chip); }
  .drill { color:var(--accent); font-size:11.5px; margin-left:6px; white-space:nowrap; }
  .sub-row td { background:#12151b; }
  .sub-box { padding:4px 2px; }
  .sub-hd { color:var(--muted); font-size:12px; margin-bottom:8px; }
  .sub-cats { display:flex; flex-wrap:wrap; gap:6px; }
  .subcat { background:var(--chip); border:1px solid var(--line); border-radius:6px;
    padding:3px 9px; font-size:12.5px; }
  .subcat i { color:var(--accent); font-style:normal; margin-left:4px; font-size:11.5px; }
  .sub-tip { color:var(--muted); font-size:12px; margin-top:8px; }
  .chart { background:var(--panel); border:1px solid var(--line); border-radius:12px;
    padding:12px 14px; margin-top:12px; }
  .chart svg { display:block; }
  .chart-cap { color:var(--muted); font-size:11.5px; margin-top:6px; text-align:right; }
  .warn { color:#e0a53f; font-size:11.5px; }
  .ai { background:var(--panel); border:1px solid var(--accent); border-radius:12px;
    padding:14px 16px; margin-top:12px; white-space:pre-wrap; }
  .ai h4 { margin:0 0 8px; font-size:13px; color:var(--accent); }
  /* 对话历史气泡：用户右侧、AI 左侧，追问时逐轮累积。
     #convo 不再自己滚动（滚动交给外层 #convo-scroll，滚轴贴窗口右边缘），
     这里只负责把气泡居中到 1000px 内容宽。 */
  #convo { max-width:1000px; margin:0 auto; padding:20px 20px 12px; }
  #convo:empty::before { content:"问点什么开始——AI 会梳理产业链、判断核心/边缘标的，"
    "研报依据标 [n]、行业判断标【判断】。"; color:var(--muted); font-size:13px;
    display:block; padding:24px 8px; }
  .bubble { display:block; max-width:92%; padding:12px 16px; border-radius:12px;
    white-space:pre-wrap; margin-bottom:12px; font-size:15px; line-height:1.7; }
  .bubble.user { background:var(--accent); color:#fff; margin-left:auto; }
  .bubble.ai { background:var(--panel); border:1px solid var(--line); }
  .bubble.ai .src { border-top:1px solid var(--line); margin-top:12px; padding-top:10px; }
  /* 底部输入 dock：吸在问答页底部，追问不用滚到底。dock 铺满宽（顶边线通栏），
     内部 .dock-wrap 居中到 1000px，与对话区对齐。 */
  .dock { flex:0 0 auto; border-top:1px solid var(--line); background:var(--bg); }
  .dock-wrap { max-width:1000px; margin:0 auto; padding:10px 20px 14px; }
  .dock-filters { display:none; gap:8px; flex-wrap:wrap; margin-bottom:8px; }
  .dock-filters.on { display:flex; }
  .dock-input { display:flex; gap:8px; align-items:flex-end; }
  .dock-input textarea { flex:1; min-height:44px; max-height:180px; resize:none;
    line-height:1.5; overflow-y:auto; }
  /* 按钮列：小号「筛选」在上，主操作「深度分析」在下（贴输入框底、最顺手的位置） */
  .dock-btns { display:flex; flex-direction:column; gap:6px; flex:0 0 auto; }
  .dock-btns button { white-space:nowrap; }
  .dock-btns .chip { padding:5px 12px; font-size:12px; }
  .dock-btns #btn-ask { padding:10px 18px; font-weight:600; }
  .date-sep { color:var(--muted); align-self:center; font-size:12px; }
  /* 可点击引用编号 [n]：点开看该块研报原文 */
  .ref { color:var(--accent); cursor:pointer; font-weight:600; }
  .ref:hover { text-decoration:underline; }
  /* markdown 渲染容器：块级布局，故覆盖父级 pre-wrap（否则块间多余空白） */
  .md { white-space:normal; }
  .md > :first-child { margin-top:0; }
  .md > :last-child { margin-bottom:0; }
  .md h1,.md h2,.md h3,.md h4,.md h5,.md h6 {
    margin:14px 0 6px; line-height:1.35; font-weight:600; }
  .md h1 { font-size:18px; } .md h2 { font-size:16px; }
  .md h3 { font-size:14.5px; } .md h4,.md h5,.md h6 { font-size:13.5px; color:var(--muted); }
  .md p { margin:8px 0; }
  .md ul,.md ol { margin:8px 0; padding-left:22px; }
  .md li { margin:3px 0; }
  .md strong { color:#7fb0ff; font-weight:600; }
  .md em { color:var(--fg); font-style:italic; }
  .md code { background:var(--chip); border:1px solid var(--line); border-radius:4px;
    padding:1px 5px; font-size:12.5px; font-family:Consolas,Menlo,monospace; }
  .md blockquote { margin:8px 0; padding:4px 12px; border-left:3px solid var(--accent);
    color:var(--muted); }
  .md hr { border:0; border-top:1px solid var(--line); margin:12px 0; }
  .md table { margin:10px 0; }
  .md thead th { color:var(--muted); font-weight:600; }
  /* 股票代码徽标：让"具体标的"在长文里一眼可见 */
  .tk { display:inline-block; background:rgba(79,140,255,.16); color:#8fb8ff;
    border:1px solid rgba(79,140,255,.35); border-radius:5px; padding:0 5px;
    font-size:12.5px; font-family:Consolas,Menlo,monospace; font-weight:600; }
  /* 国内外研报占比：双色比例条 */
  .split { background:var(--panel); border:1px solid var(--line); border-radius:10px;
    padding:12px 14px; margin-top:12px; }
  .split-t { font-size:12.5px; color:var(--muted); margin-bottom:8px; }
  .split-bar { display:flex; height:16px; border-radius:6px; overflow:hidden;
    background:var(--chip); }
  .split-bar .seg { display:block; height:100%; }
  .split-bar .seg.dom { background:var(--accent); }
  .split-bar .seg.for { background:#e0a53f; }
  .split-lg { display:flex; gap:18px; margin-top:8px; font-size:12.5px; color:var(--muted); }
  .split-lg .dot { display:inline-block; width:9px; height:9px; border-radius:2px;
    margin-right:5px; vertical-align:middle; }
  .split-lg .dot.dom { background:var(--accent); }
  .split-lg .dot.for { background:#e0a53f; }
  /* 产业链下钻：大类 → 展开子环节（点开显示代表标的 + 该环节中外占比） */
  .chain-hd { display:flex; align-items:center; gap:10px; margin:14px 0 6px; }
  .chain-hd .c-theme { font-size:14px; font-weight:600; }
  .chain-hd .c-sub { color:var(--muted); font-size:12px; }
  .cgroup { border:1px solid var(--line); border-radius:10px; margin-top:10px;
    overflow:hidden; background:var(--panel); }
  .cgroup-hd { padding:11px 14px; cursor:pointer; display:flex; align-items:center;
    gap:10px; }
  .cgroup-hd:hover { background:var(--chip); }
  .cgroup-hd .caret { color:var(--muted); font-size:11px; transition:transform .15s;
    display:inline-block; width:12px; }
  .cgroup.open .cgroup-hd .caret { transform:rotate(90deg); }
  /* 分组标题行：允许换行，别再用 nowrap+ellipsis 截断（"上中下游部分字显示不全"）。
     名称/阶段徽标在首行，定位摘要整段换到下一行完整显示。 */
  .cgroup-hd { flex-wrap:wrap; }
  .cgroup-hd .g-name { font-weight:600; font-size:13.5px; }
  .cgroup-hd .g-stage { font-size:11px; color:var(--accent); border:1px solid var(--line);
    border-radius:10px; padding:1px 8px; white-space:nowrap; }
  .cgroup-hd .g-sum { color:var(--muted); font-size:12px; line-height:1.5;
    flex-basis:100%; margin-left:22px; }
  .cgroup-bd { display:none; padding:4px 14px 12px; }
  .cgroup.open .cgroup-bd { display:block; }
  .cseg { border-top:1px solid var(--line); padding:10px 0; }
  .cseg:first-child { border-top:0; }
  .cseg .s-name { font-weight:600; font-size:13px; }
  .cseg .s-sum { color:var(--muted); font-size:12px; margin:3px 0 6px; }
  .cseg .s-tks { display:flex; flex-wrap:wrap; gap:6px; margin:5px 0; }
  .cseg .s-tk { background:rgba(79,140,255,.16); color:#8fb8ff;
    border:1px solid rgba(79,140,255,.35); border-radius:5px; padding:1px 7px;
    font-size:12px; }
  .cseg .s-tk .tk-code { color:var(--muted); font-family:Consolas,monospace;
    font-size:11px; margin-left:4px; }
  .cseg .s-split { margin-top:6px; }
  .cseg .s-splitbar { display:flex; height:12px; border-radius:5px; overflow:hidden;
    background:var(--chip); max-width:360px; }
  .cseg .s-splitbar .seg { height:100%; }
  .cseg .s-splitbar .seg.dom { background:var(--accent); }
  .cseg .s-splitbar .seg.for { background:#e0a53f; }
  .cseg .s-splitlg { font-size:11.5px; color:var(--muted); margin-top:4px; }
  /* 可搜索主题组合框：文本框 + 浮层候选列表（模糊过滤）。原生 select 不能模糊搜。 */
  .combo { position:relative; flex:1; min-width:240px; }
  .combo > input { width:100%; }
  .combo-list { display:none; position:absolute; z-index:20; left:0; right:0; top:100%;
    margin-top:4px; max-height:280px; overflow-y:auto; background:var(--panel);
    border:1px solid var(--line); border-radius:8px; box-shadow:0 8px 24px rgba(0,0,0,.4); }
  .combo-list.on { display:block; }
  .combo-item { padding:8px 12px; cursor:pointer; font-size:13px; }
  .combo-item:hover { background:var(--chip); }
  .combo-empty { padding:10px 12px; color:var(--muted); font-size:12.5px; }
  /* 引用原文弹层 */
  .modal-mask { position:fixed; inset:0; background:rgba(0,0,0,.6); display:none;
    align-items:center; justify-content:center; z-index:50; }
  .modal-mask.on { display:flex; }
  .modal-box { position:relative; background:var(--panel); border:1px solid var(--line);
    border-radius:12px; max-width:760px; width:90%; max-height:80vh; overflow:auto;
    padding:18px 20px; }
  .modal-x { position:absolute; top:10px; right:16px; cursor:pointer;
    color:var(--muted); font-size:22px; line-height:1; }
  .modal-meta { color:var(--muted); font-size:12.5px; margin-bottom:6px; }
  .modal-head { color:var(--fg); font-size:13px; margin-bottom:10px; }
  .modal-text { font-size:13.5px; line-height:1.7; white-space:pre-wrap; }
  .modal-cap { color:var(--muted); font-size:11px; margin-top:12px; }
  /* 历史会话侧栏：localStorage 持久化，可翻看/续问/删除 */
  .drawer-mask { position:fixed; inset:0; background:rgba(0,0,0,.5); display:none; z-index:40; }
  .drawer-mask.on { display:block; }
  .drawer { position:fixed; top:0; left:0; width:320px; max-width:86%; height:100%;
    background:var(--panel); border-right:1px solid var(--line); z-index:41;
    transform:translateX(-100%); transition:transform .18s ease; display:flex; flex-direction:column; }
  .drawer.on { transform:translateX(0); }
  .drawer-hd { padding:14px 16px; border-bottom:1px solid var(--line);
    display:flex; align-items:center; gap:10px; }
  .drawer-hd h3 { margin:0; font-size:14px; font-weight:600; flex:1; }
  .drawer-list { flex:1; overflow:auto; padding:8px; }
  .sess { border:1px solid var(--line); border-radius:8px; padding:10px 12px; margin-bottom:8px;
    cursor:pointer; }
  .sess:hover { border-color:var(--accent); }
  .sess.on { border-color:var(--accent); background:var(--chip); }
  .sess .s-title { font-size:13px; color:var(--fg); margin-bottom:4px;
    overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .sess .s-meta { font-size:11px; color:var(--muted); display:flex; gap:8px; }
  .sess .s-del { color:var(--muted); cursor:pointer; margin-left:auto; }
  .sess .s-del:hover { color:#ff6b6b; }
  .drawer-empty { color:var(--muted); font-size:12.5px; padding:16px; text-align:center; }
  /* 侧栏底部占用条：显式给出条数/上限与字节数，避免配额静默失败无从察觉 */
  .drawer-quota { flex:0 0 auto; border-top:1px solid var(--line); padding:8px 16px;
    color:var(--muted); font-size:11.5px; }
</style>
</head>
<body>
<header>
  <button class="ghost" id="btn-hist" title="查看历史会话">☰ 历史</button>
  <button class="ghost" id="btn-reset" title="清空当前对话，开始新一轮">＋ 新对话</button>
  <h1>研报本地 AI</h1>
  <span class="meta" id="meta">加载中…</span>
  <div class="tabs">
    <div class="tab on" data-p="ask">问答</div>
    <div class="tab" data-p="tl">研报趋势</div>
  </div>
</header>

<!-- 历史会话侧栏：localStorage 持久化，可翻看历次问答、点开续问、单条/全部删除 -->
<div class="drawer-mask" id="drawer-mask"></div>
<aside class="drawer" id="drawer">
  <div class="drawer-hd">
    <h3>历史会话</h3>
    <button class="ghost" id="btn-hist-new" title="开始新会话">＋ 新建</button>
    <button class="ghost" id="btn-hist-clear" title="清空全部历史">清空</button>
    <div class="modal-x" id="drawer-x" style="position:static;font-size:20px">×</div>
  </div>
  <div class="drawer-list" id="drawer-list"></div>
  <div class="drawer-quota" id="drawer-quota"></div>
</aside>
<main>
  <!-- 问答面板：默认深度分析（产业链梳理 + 中外资多路检索）。
       布局改为聊天式：对话历史 #convo 占据上方可滚动区，输入 dock 吸底固定，
       追问时无需拉到页面底部。筛选条件收进可折叠的 #ask-filters，保持 dock 紧凑。 -->
  <section class="panel on" id="p-ask">
    <!-- 滚动容器铺满宽度（滚轴贴窗口右缘），内部 #convo 居中到 1000px -->
    <div id="convo-scroll"><div id="convo"></div></div>
    <div class="dock" id="ask-dock">
      <div class="dock-wrap">
        <div class="status" id="ask-status"></div>
        <div class="dock-filters" id="ask-filters">
          <select id="category">
            <option value="">全部类别</option>
            <option>国内券商报告</option><option>投行报告</option>
          </select>
          <select id="institution"><option value="">全部机构</option></select>
          <input type="text" id="stock" placeholder="股票代码（选填）"/>
          <input type="date" id="date_from" title="起始日期"/>
          <span class="date-sep">至</span>
          <input type="date" id="date_to" title="截止日期"/>
          <button class="chip" id="btn-date-clear" type="button" title="清除日期限制，检索全部时段">清除日期</button>
        </div>
        <div class="dock-input">
          <textarea id="q" rows="1" placeholder="问点什么，例如：国产算力产业链有哪些核心标的？逻辑是什么？（Enter 发送，Shift+Enter 换行）"></textarea>
          <div class="dock-btns">
            <button class="chip" id="btn-filters" type="button" title="展开/收起筛选条件">筛选 ▾</button>
            <button id="btn-ask" title="深度分析：多路检索中外资研报 + 产业链梳理">深度分析</button>
          </div>
        </div>
      </div>
    </div>
    <div id="ask-out" style="display:none"></div>
  </section>

  <!-- 研报趋势面板：主题热度 / 总榜，可留空直接看全库近期关注；AI 查询读成趋势解读。
       另有「产业链下钻」：选已构建的主题，展开上/中/下游分组 → 具体环节，
       每环节显示代表标的 + 该环节中外研报占比（点开大分类看子分类）。 -->
  <section class="panel" id="p-tl">
   <div class="wrap">
    <div class="row">
      <input type="text" id="tl-theme" placeholder="主题（留空看全库总榜，如 AIDC / 固态电池）"/>
      <label class="chip"><input type="checkbox" id="tl-week"/> 按周</label>
      <button id="btn-tl">查看趋势</button>
      <button id="btn-tl-ai" class="ghost" title="调用 Claude 把热度数据读成近期研报关注趋势的解读（产生费用）">AI 查询</button>
      <button id="btn-tl-clear" class="ghost" type="button" title="清空趋势结果，腾出下方产业链下钻的空间">清空结果</button>
      <span class="hint">看研报关注度：留空=全库主题总榜；填主题=该主题按月/周的热度曲线。</span>
    </div>
    <div class="status" id="tl-status"></div>
    <div id="tl-split"></div>
    <div id="tl-ai"></div>
    <div id="tl-out"></div>

    <!-- 产业链下钻：读已落库的产业链结构（零 LLM 成本），点开分组看子环节 + 中外占比。
         主题下拉用可搜索组合框（输入模糊过滤，回车/点击选中）——已建 29 条链，纯 select 难找。 -->
    <div class="chain-wrap">
      <div class="row">
        <div class="combo" id="chain-combo">
          <input type="text" id="chain-theme" autocomplete="off"
                 placeholder="选产业链主题下钻（可输入模糊搜索，如 半导 / 锂电 / 机器人）"/>
          <div class="combo-list" id="chain-list"></div>
        </div>
        <button id="btn-chain-ai" class="ghost" type="button"
                title="调用 Claude 结合这条产业链结构 + 检索到的研报，深度解读上/中/下游格局与核心标的（产生费用）">链路 AI 解读</button>
        <span class="hint">产业链结构一次性构建后落库、稳定不变；纯读库展开上/中/下游各环节的代表标的与中外研报占比。</span>
      </div>
      <div class="status" id="chain-status"></div>
      <div id="chain-ai"></div>
      <div id="chain-out"></div>
    </div>
   </div>
  </section>
</main>
<!-- 引用溯源弹层：点 [n] 打开，显示该块研报原文，供对照 AI 论断是否属实 -->
<div id="modal" class="modal-mask">
  <div class="modal-box">
    <div class="modal-x" id="modal-x">×</div>
    <div id="modal-body"></div>
  </div>
</div>
<script>
const $ = s => document.querySelector(s);
const esc = s => (s||"").replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

// 标签切换
// 切到指定面板（ask / tl）。抽成函数，供 tab 点击与「历史跳转」复用。
function switchPanel(p){
  document.querySelectorAll('.tab').forEach(x => x.classList.toggle('on', x.dataset.p===p));
  document.querySelectorAll('.panel').forEach(x => x.classList.remove('on'));
  $('#p-'+p).classList.add('on');
}
document.querySelectorAll('.tab').forEach(t => t.onclick = () => switchPanel(t.dataset.p));

// 健康信息
fetch('/api/health').then(r=>r.json()).then(h=>{
  $('#meta').textContent =
    `文档 ${h.documents??'?'} · 块 ${h.chunks??'?'} · 事实 ${h.facts??'?'} · `
    + `Claude ${h.llm_base_url?'✓':'✗'} · 检索 ${h.retrieval_mode??'?'}`;
}).catch(()=>{ $('#meta').textContent='健康检查失败'; });

function askBody(){
  return {
    query: $('#q').value.trim(),
    category: $('#category').value || null,
    institution: $('#institution').value || null,
    stock_code: $('#stock').value.trim() || null,
    date_from: $('#date_from').value.trim() || null,
    date_to: $('#date_to').value.trim() || null,
    analyst: true,          // 默认深度分析（多路检索 + 产业链梳理）
    limit: 12
  };
}

// 机构下拉：从后端拉全部机构（按研报数降序）填充，供筛选。子串匹配，选全名也命中。
fetch('/api/institutions').then(r=>r.json()).then(d=>{
  const sel = $('#institution');
  (d.institutions||[]).forEach(it=>{
    const o = document.createElement('option');
    o.value = it.name; o.textContent = `${it.name}（${it.count}）`;
    sel.appendChild(o);
  });
}).catch(()=>{});

// 日期默认近 3 个月：截止=今天，起始=今天前 90 天。用户可改或「清除日期」看全时段。
(function initDates(){
  const fmt = d => d.toISOString().slice(0,10);
  const to = new Date(), from = new Date();
  from.setDate(from.getDate()-90);
  $('#date_to').value = fmt(to);
  $('#date_from').value = fmt(from);
})();
$('#btn-date-clear').onclick = () => { $('#date_from').value=''; $('#date_to').value=''; };

// ---- 对话状态：支持追问 + 持久化历史 ----
// history 是发给后端的轮次数组 [{role,content}]；chunkMap 累积所有轮的 [n]→来源，
// 供点击溯源。每轮回答里的 [n] 渲染成可点标记，点开拉该 chunk 原文对照。
// 持久化：sessions[] 存 localStorage，每条会话含 turns[]（每轮的问题+完整应答），
// 刷新/关页后仍可从侧栏翻看历次问答、点开续问。纯前端存储，不涉后端/数据库/密钥。
let history = [];
let chunkMap = {};   // ref编号(字符串) → {chunk_id,title,institution,report_date,heading_path}

// v2 起改为「指针式」存储，解决 localStorage 配额被撑爆后静默丢记录的问题：
//  · AI 解读（trend/chain）只存指针（kind + 主题 + 粒度 + 标题 + 时间），**不存正文**。
//    正文已在后端 trend_cache 表里，点历史时重新请接口、命中缓存直返（0.2s，不重复计费）。
//  · 问答轮次瘦身：只存 answer 文本 + 来源的 chunk_id/最少元数据，不再存整个响应体。
//  · 硬上限 MAX_SESSIONS 条，超出丢最旧 —— 无论怎么用，占用恒定有界、不会再静默丢新数据。
const LS_KEY = 'yanbao_sessions_v2';
const MAX_SESSIONS = 100;   // 历史记录条数上限（超出丢最旧）
let sessions = [];   // [{id,kind,title,ts, turns?[], ptr?{}}]
let curId = null;    // 当前会话 id（null=尚未落盘的空白会话）

function loadSessions(){
  try { sessions = JSON.parse(localStorage.getItem(LS_KEY)||'[]'); }
  catch(e){ sessions = []; }
  if(!Array.isArray(sessions)) sessions = [];
}
// 落盘：先按上限裁剪，写失败（配额满）则逐步丢最旧重试，最后仍失败才放弃。
// 这样即使某条特别大，也会腾地方而不是静默丢弃新记录。
function saveSessions(){
  if(sessions.length > MAX_SESSIONS) sessions = sessions.slice(0, MAX_SESSIONS);
  for(let i=0;i<6;i++){
    try { localStorage.setItem(LS_KEY, JSON.stringify(sessions)); return true; }
    catch(e){
      if(sessions.length<=1) break;
      sessions = sessions.slice(0, Math.max(1, Math.floor(sessions.length*0.7)));
    }
  }
  return false;
}
function curSession(){ return sessions.find(s=>s.id===curId) || null; }

// 把响应瘦身成落盘所需的最小集合：正文 + 复现引用所需的最少来源字段。
// 原来存整个 /api/ask 响应（含最多 72 条来源的全部元数据），单轮 15-25KB；
// 瘦身后只留 answer + 每条来源的 ref/chunk_id/标题机构日期，体积降一大半。
// 摘要/子查询等展示性字段丢弃（回放时不影响读答案与点 [n] 溯源）。
function slimAnswer(d){
  return {
    answer: d.answer, mode: d.mode, model: d.model,
    sources: (d.sources||[]).map(s=>({
      ref:s.ref, chunk_id:s.chunk_id, title:s.title,
      institution:s.institution, report_date:s.report_date,
    })),
  };
}

// 把当前这一轮（问题 q + 瘦身后的响应）写入当前会话并落盘；无当前会话则新建一条。
function persistTurn(q, d){
  let s = curSession();
  if(!s){
    s = { id: Date.now()+''+Math.floor(Math.random()*1000), title: q.slice(0,40),
          ts: Date.now(), turns: [] };
    sessions.unshift(s); curId = s.id;
  }
  s.turns.push({ q, d: slimAnswer(d) });
  s.ts = Date.now();
  if(s.turns.length===1) s.title = q.slice(0,40);  // 用首问作标题
  saveSessions(); renderSessions();
}

// 开新会话：清空当前对话视图与状态（历史仍保留在 localStorage）。
// goAsk=true 时**先切回问答页**：「新对话」按钮在 header 里、两个页面共用，在研报趋势页
// 点它若不切页，只会静默清空看不见的问答状态、页面停在趋势页，像是按钮坏了。
// 但删除单条历史时连带调用本函数只为清状态，不该把人从趋势页弹走 → 那里传 false。
function resetConvo(goAsk){
  if(goAsk){ switchPanel('ask'); }
  history = []; chunkMap = {}; curId = null;
  $('#convo').innerHTML=''; $('#ask-out').innerHTML=''; $('#ask-status').textContent='';
  renderSessions();
  if(goAsk){ $('#q').focus(); }   // 切过来就能直接打字
}
$('#btn-reset').onclick = () => resetConvo(true);

// ---- 历史侧栏 ----
function openDrawer(){ renderSessions(); $('#drawer').classList.add('on'); $('#drawer-mask').classList.add('on'); }
function closeDrawer(){ $('#drawer').classList.remove('on'); $('#drawer-mask').classList.remove('on'); }
$('#btn-hist').onclick = openDrawer;
$('#drawer-x').onclick = closeDrawer;
$('#drawer-mask').onclick = closeDrawer;
$('#btn-hist-new').onclick = () => { resetConvo(true); closeDrawer(); };
$('#btn-hist-clear').onclick = () => {
  if(!sessions.length) return;
  if(!confirm('清空全部历史会话？此操作不可撤销。')) return;
  sessions = []; saveSessions(); resetConvo(true);
};

// 历史条目分两类：问答会话（kind 缺省/ask，含 turns）与 AI 解读（kind trend/chain，
// 含 payload）。解读条目点开 → 跳到研报趋势页对应位置就地重渲染（不重复计费）。
const KIND_TAG = { trend:'📈 趋势解读', chain:'🔗 链路解读' };
// 已删除的旧链 → 新体系去向（2026-07-28 产业链重建：旧 29 条 → 新 60 条）。
// 历史里存的链路解读指针若指向已删旧链，点开会 404；这里给出合并去向的友好提示，
// 并自动清掉这条失效指针（stale 数据在客户端 localStorage，服务端删不到）。
const DEAD_CHAIN_MAP = {
  '人工智能':'AI算力数据中心 / 大模型AI应用', '算力':'AI算力数据中心',
  'AIDC':'AI算力数据中心', '国产算力':'AI算力数据中心', '大模型':'大模型AI应用',
  '芯片':'半导体', '有色':'铜 / 铝 / 稀有金属战略金属',
  '化工':'炼化石化 / 煤化工 / 基础化工 / 农化 / 精细化工电子化学品',
  '石油石化':'石油天然气 / 天然气LNG / 炼化石化',
  '新能源':'光伏 / 风电 / 核电 等', '新能源车':'新能源汽车', '储能':'新型储能',
  '军工':'航空航天军工', '机器人':'机器人具身智能', '人形机器人':'机器人具身智能',
  '创新药':'化学制药 / 生物制药', '医药':'化学制药 / 生物制药 / 医疗器械IVD / 医药研发生产外包',
  '农业':'农业种业 / 养殖饲料动保', '固态电池':'锂电池（技术路线标签）', '数据要素':'（已并入宏观，无独立链）',
};
// 侧栏底部显示占用情况：条数/上限 + 实际字节数。指针化后正常不会接近配额，
// 但显式给个量，免得又出现"以为存上了其实丢了"的静默失败。
function renderQuota(){
  const box = $('#drawer-quota'); if(!box) return;
  let kb = 0;
  try { kb = Math.round(JSON.stringify(sessions).length/1024); } catch(e){}
  box.textContent = `${sessions.length} / ${MAX_SESSIONS} 条 · 约 ${kb} KB`;
}

function renderSessions(){
  renderQuota();
  const box = $('#drawer-list');
  if(!sessions.length){ box.innerHTML = '<div class="drawer-empty">还没有历史记录</div>'; return; }
  box.innerHTML = sessions.map(s=>{
    const d = new Date(s.ts);
    const when = `${d.getMonth()+1}/${d.getDate()} ${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`;
    const tag = KIND_TAG[s.kind] || `💬 ${(s.turns||[]).length} 轮`;
    return `<div class="sess${s.id===curId?' on':''}" data-sid="${s.id}">`
      + `<div class="s-title">${esc(s.title||'(空)')}</div>`
      + `<div class="s-meta"><span>${esc(when)}</span><span>${esc(tag)}</span>`
      + `<span class="s-del" data-del="${s.id}" title="删除这条记录">🗑</span></div></div>`;
  }).join('');
}

// 点侧栏：删除单条 or 载入该会话续问。
$('#drawer-list').onclick = (ev) => {
  const del = ev.target.closest('.s-del');
  if(del){
    const id = del.dataset.del;
    sessions = sessions.filter(s=>s.id!==id);
    if(curId===id) resetConvo();
    saveSessions(); renderSessions();
    return;
  }
  const card = ev.target.closest('.sess');
  if(card){ loadSession(card.dataset.sid); closeDrawer(); }
};

// 载入一条历史记录：问答会话 → 重建对话续问；AI 解读 → 切到趋势页就地重渲染。
function loadSession(id){
  const s = sessions.find(x=>x.id===id); if(!s) return;
  if(s.kind==='trend' || s.kind==='chain'){ loadInterp(s); return; }
  switchPanel('ask');
  curId = id; history = []; chunkMap = {};
  $('#convo').innerHTML=''; $('#ask-out').innerHTML=''; $('#ask-status').textContent='';
  (s.turns||[]).forEach(t=>{
    appendTurn(t.q, t.d);
    history.push({role:'user', content:t.q});
    history.push({role:'assistant', content:t.d.answer});
  });
  renderSessions();
}

// 把一次 AI 解读记进历史——**只存指针，不存正文**。解读原文已在后端 trend_cache 表里
// （同主题同粒度直返、0.2s、不再计费），前端再存一份 1 万多字的 md 是重复存储，
// 攒几十条就顶到 localStorage 配额、然后静默丢数据。故这里只记 kind + 重放所需的
// 上下文（主题/粒度），约百字节；点历史时按这些参数回调接口，命中后端缓存即可。
function saveInterp(kind, title, meta){
  const s = { id: Date.now()+''+Math.floor(Math.random()*1000),
              kind, title, meta: meta||{}, ts: Date.now() };
  sessions.unshift(s); saveSessions(); renderSessions();
}

// 点历史里的 AI 解读：切到趋势页，按存下的参数重新请求接口。
// 后端 trend_cache 命中 → 秒回、不计费；只有底层数据变了（input_hash 不同）才会重算，
// 那种情况下重算本身是应该的（旧解读已经对不上新数据了）。
async function loadInterp(s){
  switchPanel('tl');
  const meta = s.meta || {};
  const isChain = s.kind==='chain';
  const box = isChain ? $('#chain-ai') : $('#tl-ai');
  const st  = isChain ? $('#chain-status') : $('#tl-status');
  box.innerHTML = `<div class="ai"><h4>${esc(s.title)}</h4>读取解读（后端缓存）…</div>`;
  box.scrollIntoView({behavior:'smooth', block:'start'});
  try {
    let url;
    if(isChain){
      url = '/api/chain/interpret?theme='+encodeURIComponent(meta.theme||'');
    } else {
      const q = new URLSearchParams();
      if(meta.theme) q.set('theme', meta.theme);
      if(meta.week) q.set('week','true');
      url = '/api/timeline/theme/interpret?'+q.toString();
    }
    const r = await fetch(url);
    if(!r.ok){
      const e = await r.json().catch(()=>({}));
      // 链路 404 = 该链已在重建中被删除/合并。给出新去向，并把这条 stale 历史指针自动清掉
      //（它永远不会再成功，留着只会反复报错）。
      if(isChain && r.status===404 && DEAD_CHAIN_MAP[meta.theme]!==undefined){
        const to = DEAD_CHAIN_MAP[meta.theme];
        sessions = sessions.filter(x=>x.id!==s.id); saveSessions(); renderSessions();
        box.innerHTML = `<div class="ai"><h4>${esc(s.title)}</h4>`
          + `<span class="err">「${esc(meta.theme)}」链已在产业链重建中删除，`
          + `${to?('内容已并入 <b>'+esc(to)+'</b>，请在上方主题里改选。'):'现无对应链。'}`
          + `<br>这条失效的历史记录已自动移除。</span></div>`;
        return;
      }
      throw new Error(e.detail||r.status);
    }
    const d = await r.json();
    // 链存在（fetch 已成功）才展开它的结构，回到当时的上下文——放到这里避免死链时
    // 下方结构区也弹"尚未构建"的冗余提示。
    if(isChain && meta.theme){ $('#chain-theme').value = meta.theme; chainSel = meta.theme; loadChain(meta.theme); }
    box.innerHTML = `<div class="ai"><h4>${esc(s.title)}`
      + `${d.model?' · '+esc(d.model):''}${d.cached?' · 后端缓存':' · 已重算（底层数据有更新）'}</h4>`
      + `${renderMarkdown(d.interpretation)}</div>`;
    st.textContent='';
  } catch(e){
    box.innerHTML = `<div class="ai"><h4>${esc(s.title)}</h4>`
      + `<span class="err">读取解读失败：${esc(e.message)}</span></div>`;
  }
}

// 行内 markdown：先转义防注入（esc 只处理 &<>，markdown 记号 *`# 等原样保留），
// 再依次处理 行内代码 → 粗体 → 斜体 → 引用编号 [n] → 股票代码高亮。
// 股票突出：粗体（研报里公司名多用 **名** 标注）染成强调色；( 300xxx / 688xxx / .HK
// 等代码 ) 包成 tk 徽标，让"具体标的"在长文里一眼可见。
function inlineMd(t){
  t = t.replace(/`([^`]+)`/g, '<code>$1</code>');
  t = t.replace(/\*\*([^*]+?)\*\*/g, '<strong>$1</strong>');
  t = t.replace(/(^|[^*])\*([^*\n]+?)\*(?!\*)/g, '$1<em>$2</em>');
  t = t.replace(/\[(\d+)\]/g, (m, n) =>
    chunkMap[n] ? `<span class="ref" data-ref="${n}">[${n}]</span>` : m);
  // 括号内 5-6 位数字（可带 .SH/.SZ/.HK 后缀）视作股票代码，高亮。
  t = t.replace(/([（(])\s*(\d{5,6}(?:\.[A-Za-z]{2,4})?)\s*([）)])/g,
    '$1<span class="tk">$2</span>$3');
  return t;
}

// 轻量 markdown → HTML（无外部依赖）。逐行分块：标题/表格/有序无序列表/引用/
// 分隔线/段落，块内走 inlineMd。整体裹 .md 供样式收敛。用于问答与 AI 解读输出。
function renderMarkdown(src){
  const lines = esc(String(src||'')).replace(/\r\n?/g,'\n').split('\n');
  const out = [];
  let i = 0;
  const isSep = s => /^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$/.test(s) && s.includes('-');
  const cells = s => s.replace(/^\s*\|/,'').replace(/\|\s*$/,'').split('|').map(c=>c.trim());
  while(i < lines.length){
    const line = lines[i];
    if(/^\s*$/.test(line)){ i++; continue; }
    // 分隔线
    if(/^\s*([-*_])\1{2,}\s*$/.test(line)){ out.push('<hr>'); i++; continue; }
    // 标题
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if(h){ const lv=h[1].length; out.push(`<h${lv}>${inlineMd(h[2])}</h${lv}>`); i++; continue; }
    // 表格：本行含 | 且下一行是分隔行
    if(line.includes('|') && i+1<lines.length && isSep(lines[i+1])){
      const head = cells(line);
      i += 2;
      const body = [];
      while(i<lines.length && lines[i].includes('|') && !/^\s*$/.test(lines[i])){
        body.push(cells(lines[i])); i++;
      }
      let t = '<table><thead><tr>'+head.map(c=>`<th>${inlineMd(c)}</th>`).join('')+'</tr></thead><tbody>';
      body.forEach(r=>{ t += '<tr>'+r.map(c=>`<td>${inlineMd(c)}</td>`).join('')+'</tr>'; });
      t += '</tbody></table>';
      out.push(t); continue;
    }
    // 引用块
    if(/^\s*>\s?/.test(line)){
      const buf=[];
      while(i<lines.length && /^\s*>\s?/.test(lines[i])){ buf.push(lines[i].replace(/^\s*>\s?/,'')); i++; }
      out.push(`<blockquote>${inlineMd(buf.join(' '))}</blockquote>`); continue;
    }
    // 有序列表
    if(/^\s*\d+[.)]\s+/.test(line)){
      const buf=[];
      while(i<lines.length && /^\s*\d+[.)]\s+/.test(lines[i])){
        buf.push(`<li>${inlineMd(lines[i].replace(/^\s*\d+[.)]\s+/,''))}</li>`); i++;
      }
      out.push(`<ol>${buf.join('')}</ol>`); continue;
    }
    // 无序列表
    if(/^\s*[-*+]\s+/.test(line)){
      const buf=[];
      while(i<lines.length && /^\s*[-*+]\s+/.test(lines[i])){
        buf.push(`<li>${inlineMd(lines[i].replace(/^\s*[-*+]\s+/,''))}</li>`); i++;
      }
      out.push(`<ul>${buf.join('')}</ul>`); continue;
    }
    // 段落：吃到空行/块级起始为止，行间 <br>
    const buf=[];
    while(i<lines.length && !/^\s*$/.test(lines[i])
          && !/^(#{1,6})\s/.test(lines[i]) && !/^\s*([-*_])\1{2,}\s*$/.test(lines[i])
          && !/^\s*>\s?/.test(lines[i]) && !/^\s*\d+[.)]\s+/.test(lines[i])
          && !/^\s*[-*+]\s+/.test(lines[i])
          && !(lines[i].includes('|') && i+1<lines.length && isSep(lines[i+1]))){
      buf.push(lines[i]); i++;
    }
    out.push(`<p>${inlineMd(buf.join('\n')).replace(/\n/g,'<br>')}</p>`);
  }
  return `<div class="md">${out.join('')}</div>`;
}

// 追加一轮对话到 #convo（user 气泡 + assistant 气泡含来源脚注）。
function appendTurn(q, d){
  const conv = $('#convo');
  const uq = document.createElement('div');
  uq.className = 'bubble user'; uq.textContent = q;
  conv.appendChild(uq);

  // 记录本轮来源到全局 chunkMap（编号跨轮可能重复，后到覆盖——够用）。
  (d.sources||[]).forEach(s => { chunkMap[s.ref] = s; });

  let foot = '';
  // 引用信息默认折叠（<details>），点摘要行展开。正文里的 [n] 仍可点，行为不变。
  // 深度分析：展示这次按哪些「子环节」分别检索（多路检索覆盖面），让用户看到检索广度、
  // 也便于发现"某条线没被拆出来"——即召回缺口所在。
  const inner = [];
  if(d.subqueries?.length){
    inner.push(`<div class="src-sec"><div class="src-lbl">按 ${d.subqueries.length} 条子环节线检索</div>`
          + `<div class="src-item">${d.subqueries.map(esc).join('　·　')}</div></div>`);
  }
  if(d.sources?.length){
    let s0 = `<div class="src-sec"><div class="src-lbl">来源（模式 ${d.mode}${d.used_like?' · LIKE兜底':''}${d.model?' · '+esc(d.model):''}）· 点 [n] 看研报原文</div>`;
    d.sources.forEach(s=>{
      const meta=[s.institution,s.title,s.report_date].filter(Boolean).map(esc).join(' · ');
      s0 += `<div class="src-item"><span class="ref" data-ref="${s.ref}">[${s.ref}]</span> `
            + `<b>${meta||esc(s.doc_id)}</b> ${esc(s.heading_path||'')}</div>`;
    });
    inner.push(s0 + `</div>`);
  }
  if(inner.length){
    const n = d.sources?.length || 0;
    foot = `<details class="src-fold"><summary>📎 引用与检索详情`
         + (n?`（${n} 篇研报）`:'') + `</summary>${inner.join('')}</details>`;
  }
  const ab = document.createElement('div');
  ab.className = 'bubble ai';
  ab.innerHTML = renderMarkdown(d.answer) + foot;
  conv.appendChild(ab);
  conv.scrollTop = conv.scrollHeight;
}

$('#btn-ask').onclick = async () => {
  const b = askBody(); if(!b.query){ $('#ask-status').textContent='请输入问题'; return; }
  b.history = history;   // 带上既往轮次 → 支持追问（analyst 已在 askBody 中恒为 true）
  const q = b.query;
  $('#btn-ask').disabled = true;
  $('#ask-status').textContent = '深度分析中（拆解产业链 · 多路检索中外资研报 · 生成）…';
  $('#ask-out').innerHTML='';
  try {
    const r = await fetch('/api/ask', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(b)});
    if(!r.ok){ const e=await r.json(); throw new Error(e.detail||r.status); }
    const d = await r.json();
    appendTurn(q, d);
    // 累积对话历史（存纯文本答案，不含 HTML），供下一轮追问。
    history.push({role:'user', content:q});
    history.push({role:'assistant', content:d.answer});
    persistTurn(q, d);   // 落盘到 localStorage，刷新后仍可从侧栏翻看/续问
    $('#q').value=''; autosize(); $('#ask-status').textContent='';
  } catch(e){ $('#ask-status').innerHTML = `<span class="err">出错：${esc(e.message)}</span>`; }
  finally { $('#btn-ask').disabled = false; }
};

// 筛选条件默认收起，点「筛选 ▾」展开/收起，保持 dock 紧凑。
$('#btn-filters').onclick = () => {
  const on = $('#ask-filters').classList.toggle('on');
  $('#btn-filters').textContent = on ? '筛选 ▴' : '筛选 ▾';
};

// 输入框：Enter 发送、Shift+Enter 换行；随内容高度自适应（上限见 CSS max-height）。
const qEl = $('#q');
function autosize(){ qEl.style.height='auto'; qEl.style.height=Math.min(qEl.scrollHeight,180)+'px'; }
qEl.addEventListener('input', autosize);
qEl.addEventListener('keydown', (ev) => {
  if(ev.key==='Enter' && !ev.shiftKey && !ev.isComposing){
    ev.preventDefault();
    if(!$('#btn-ask').disabled) $('#btn-ask').click();
  }
});

// 点击 [n] 标记 → 拉该 chunk 原文，弹层显示，供逐句对照 AI 论断与研报原文。
document.addEventListener('click', async (ev) => {
  const el = ev.target.closest('.ref'); if(!el) return;
  const s = chunkMap[el.dataset.ref]; if(!s || !s.chunk_id) return;
  openChunk(s.chunk_id);
});

async function openChunk(chunk_id){
  const mask=$('#modal'), body=$('#modal-body');
  body.innerHTML='加载原文中…'; mask.style.display='flex';
  try {
    const r=await fetch('/api/chunk/'+encodeURIComponent(chunk_id));
    if(!r.ok){ const e=await r.json(); throw new Error(e.detail||r.status); }
    const c=await r.json();
    const meta=[c.institution,c.title,c.report_date].filter(Boolean).map(esc).join(' · ');
    body.innerHTML = `<div class="modal-meta">${meta}</div>`
      + `<div class="modal-head">${esc(c.heading_path||'')}</div>`
      + `<div class="modal-text">${esc(c.text||'')}</div>`
      + `<div class="modal-cap">${esc(c.chunk_id)}</div>`;
  } catch(e){ body.innerHTML = `<span class="err">取原文失败：${esc(e.message)}</span>`; }
}

// 关闭弹层：点 ×、点遮罩空白、按 Esc。
function closeModal(){ $('#modal').style.display='none'; }
$('#modal-x').onclick = closeModal;
$('#modal').onclick = (ev) => { if(ev.target === $('#modal')) closeModal(); };
document.addEventListener('keydown', (ev) => { if(ev.key==='Escape') closeModal(); });

// 研报趋势查看：只做主题热度/总榜（指标时间线已移除）。主题留空 → 全库总榜；
// 填主题 → 该主题按月/周的关注度曲线。纯 SQL 计数，零 LLM 成本。
$('#btn-tl').onclick = async () => {
  const th=$('#tl-theme').value.trim();
  $('#tl-status').textContent='查询中…'; $('#tl-out').innerHTML=''; $('#tl-ai').innerHTML=''; $('#tl-split').innerHTML='';
  try {
    if(th){
      const q=new URLSearchParams({theme:th}); if($('#tl-week').checked)q.set('week','true');
      const d=await (await fetch('/api/timeline/theme?'+q)).json();
      renderBars(d.buckets, `${d.theme} 按${d.by==='week'?'周':'月'}关注度`, d.by==='week'?'周次':'月份');
      renderSplit(d.split, d.theme);
    } else {
      const d=await (await fetch('/api/timeline/theme?top=30')).json();
      renderBars(d.buckets, '全库主题关注度总榜', '主题');
      renderSplit(d.split, null);
    }
  } catch(e){ $('#tl-status').innerHTML = `<span class="err">出错：${esc(e.message)}</span>`; }
};

// 清空趋势结果：总榜/曲线查出来后占满版面、挡住下方产业链下钻，给个一键清空的出口。
$('#btn-tl-clear').onclick = () => {
  $('#tl-theme').value=''; $('#tl-status').textContent='';
  $('#tl-out').innerHTML=''; $('#tl-ai').innerHTML=''; $('#tl-split').innerHTML='';
};

// AI 查询：把当前主题热度数据（总榜 or 某主题按月/周）交给 Claude 读成"近期研报在
// 关注什么、哪些在升温/降温"的趋势解读。点了才调 LLM（产生费用）。与查看趋势同参数。
$('#btn-tl-ai').onclick = async () => {
  const th=$('#tl-theme').value.trim();
  const q=new URLSearchParams(); if(th)q.set('theme',th);
  if($('#tl-week').checked)q.set('week','true');
  $('#btn-tl-ai').disabled=true; $('#tl-status').textContent='AI 查询中（调用 Claude 解读趋势）…';
  try {
    const r=await fetch('/api/timeline/theme/interpret?'+q.toString());
    if(!r.ok){ const e=await r.json(); throw new Error(e.detail||r.status); }
    const d=await r.json();
    const head = d.theme ? `主题「${esc(d.theme)}」趋势解读` : '全库主题关注度趋势解读';
    $('#tl-ai').innerHTML = `<div class="ai"><h4>${head}${d.model?' · '+esc(d.model):''}</h4>`
      + `${renderMarkdown(d.interpretation)}</div>`;
    // 顺带把底层热度数据也画出来，让解读有据可查。
    if(d.buckets?.length) renderBars(d.buckets, d.theme ? `${esc(d.theme)} 关注度` : '主题关注度总榜', d.theme ? (d.by==='week'?'周次':'月份') : '主题');
    else $('#tl-status').textContent='';
    renderSplit(d.split, d.theme);
    saveInterp('trend', head, {theme:th, week:$('#tl-week').checked});
  } catch(e){ $('#tl-status').innerHTML = `<span class="err">AI 查询出错：${esc(e.message)}</span>`; }
  finally { $('#btn-tl-ai').disabled=false; }
};

// col：首列列名，随场景变化——总榜是「主题」，按月/周曲线是「月份」/「周次」。
// 原来固定写「分桶」，语义含糊（既指主题也指时间桶）；按场景命名更好懂。
function renderBars(buckets, title, col){
  $('#tl-status').textContent = `${title} · ${buckets.length} 项`;
  if(!buckets.length){ $('#tl-out').innerHTML='（无数据——先跑 facts 抽取）'; return; }
  const max=Math.max(...buckets.map(b=>b.count),1);
  const c0 = col || '项目';
  // 总榜（col==='主题'）的每一行可点击 → 下钻看该主题的产业链子分类（若已构建）。
  const clickable = c0==='主题';
  $('#tl-out').innerHTML = `<table><tr><th>${esc(c0)}</th><th>研报数</th><th></th></tr>`
    + buckets.map(b=>`<tr${clickable?` class="th-row" data-theme="${esc(b.bucket)}"`:''}>`
        + `<td>${esc(b.bucket)}${clickable?' <span class="drill">▸ 子分类</span>':''}</td><td>${b.count}</td>`
        + `<td><span class="bar" style="width:${Math.round(b.count/max*240)}px"></span></td></tr>`).join('')
    + `</table>`;
}

// 点总榜某主题行 → 在其下方就地展开该主题的"子分类"（共现主题）。
// 共现刻画了这个大类实际覆盖哪些方向；不同大类会共享子类（真实重叠，不隐藏）。
// 已建产业链的主题额外给一条引导，去下面的下钻看权威上/中/下游分段。
$('#tl-out') && ($('#tl-out').onclick = async (ev) => {
  const drill=ev.target.closest('.drill'); const row=ev.target.closest('.th-row');
  if(!row || !drill) return;
  const theme=row.dataset.theme;
  const next=row.nextElementSibling;
  if(next && next.classList.contains('sub-row')){  // 已展开 → 收起
    next.remove(); drill.textContent='▸ 子分类'; return;
  }
  drill.textContent='加载中…';
  try {
    const d=await (await fetch('/api/theme/subcats?theme='+encodeURIComponent(theme))).json();
    const tr=document.createElement('tr'); tr.className='sub-row';
    const cells=(d.subcats||[]).map(s=>
      `<span class="subcat" title="与「${esc(theme)}」共现于 ${s.count} 篇研报">`
      + `${esc(s.theme)} <i>${s.pct}%</i></span>`).join('');
    const chainTip = d.has_chain
      ? `<div class="sub-tip">该主题已建产业链 → 下方「产业链下钻」可看上/中/下游权威分段。</div>` : '';
    tr.innerHTML = `<td colspan="3"><div class="sub-box">`
      + `<div class="sub-hd">「${esc(theme)}」共现子分类 · 按同现研报数（共 ${d.total} 篇打此标签）</div>`
      + (cells?`<div class="sub-cats">${cells}</div>`:'<div class="sub-tip">无共现主题</div>')
      + chainTip + `</div></td>`;
    row.after(tr); drill.textContent='▾ 子分类';
  } catch(e){ drill.textContent='▸ 子分类'; alert('取子分类失败：'+e.message); }
});

// 国内外研报占比：一根双色条 + 文字。domestic=国内券商(zh)，foreign=外资投行(en)。
function renderSplit(sp, theme){
  const box=$('#tl-split'); if(!sp || !sp.total){ box.innerHTML=''; return; }
  const scope = theme ? `主题「${esc(theme)}」` : '全库';
  box.innerHTML = `<div class="split"><div class="split-t">${scope}国内外研报占比`
    + `（共 ${sp.total} 篇）</div>`
    + `<div class="split-bar">`
    + `<span class="seg dom" style="width:${sp.domestic_pct}%" title="国内券商 ${sp.domestic} 篇"></span>`
    + `<span class="seg for" style="width:${sp.foreign_pct}%" title="外资投行 ${sp.foreign} 篇"></span>`
    + `</div>`
    + `<div class="split-lg"><span><i class="dot dom"></i>国内券商 ${sp.domestic}（${sp.domestic_pct}%）</span>`
    + `<span><i class="dot for"></i>外资投行 ${sp.foreign}（${sp.foreign_pct}%）</span></div></div>`;
}

// ---- 产业链下钻：读已落库的产业链结构（零 LLM 成本）----
// 页面加载时拉已构建主题填下拉；选主题 → 拉整条链，渲染成 分组卡片（可展开/收起），
// 每张卡片下列各环节：定位 + 代表标的徽标 + 该环节中外研报占比迷你条。
// “点开大分类看子分类”即分组卡片默认收起、点标题展开。
// 可搜索主题组合框：拉全部已建链主题存在内存，输入即模糊过滤下拉，点/回车选中。
let chainThemes = [];      // 全部已建链主题
let chainSel = '';         // 当前选中的主题（供 AI 解读复用）
function initChainThemes(){
  fetch('/api/chain/themes').then(r=>r.json()).then(d=>{
    chainThemes = d.themes||[];
  }).catch(()=>{});
}

// 依据输入过滤并渲染下拉候选。空输入 → 列全部。子串命中（大小写不敏感，主要是中文）。
function renderChainList(){
  const box=$('#chain-list'), kw=$('#chain-theme').value.trim().toLowerCase();
  const hits = kw ? chainThemes.filter(t=>t.toLowerCase().includes(kw)) : chainThemes;
  if(!hits.length){ box.innerHTML='<div class="combo-empty">无匹配主题</div>'; box.classList.add('on'); return; }
  box.innerHTML = hits.map(t=>`<div class="combo-item" data-th="${esc(t)}">${esc(t)}</div>`).join('');
  box.classList.add('on');
}
function hideChainList(){ $('#chain-list').classList.remove('on'); }

async function loadChain(th){
  chainSel = th;
  $('#chain-out').innerHTML=''; $('#chain-ai').innerHTML=''; $('#chain-status').textContent='';
  if(!th) return;
  $('#chain-status').textContent='读取产业链结构…';
  try {
    const d=await (await fetch('/api/chain?theme='+encodeURIComponent(th))).json();
    renderChain(d);
  } catch(e){ $('#chain-status').innerHTML=`<span class="err">出错：${esc(e.message)}</span>`; }
}

if($('#chain-theme')){
  const inp=$('#chain-theme');
  inp.addEventListener('focus', renderChainList);
  inp.addEventListener('input', renderChainList);
  // 回车：若正好等于某主题或只剩一个候选，直接选中加载。
  inp.addEventListener('keydown', (ev)=>{
    if(ev.key!=='Enter') return;
    const kw=inp.value.trim();
    const exact=chainThemes.find(t=>t===kw);
    const hits=chainThemes.filter(t=>t.toLowerCase().includes(kw.toLowerCase()));
    const pick=exact || (hits.length===1?hits[0]:null);
    if(pick){ inp.value=pick; hideChainList(); loadChain(pick); }
  });
  // 点候选项选中。
  $('#chain-list').onclick=(ev)=>{
    const it=ev.target.closest('.combo-item'); if(!it) return;
    inp.value=it.dataset.th; hideChainList(); loadChain(it.dataset.th);
  };
  // 点组合框外部收起下拉。
  document.addEventListener('click', (ev)=>{ if(!ev.target.closest('#chain-combo')) hideChainList(); });
}

// 链路 AI 解读：把这条链结构 + 各环节检索到的研报交给 Claude，深读上/中/下游格局与核心标的。
$('#btn-chain-ai') && ($('#btn-chain-ai').onclick = async () => {
  if(!chainSel){ $('#chain-status').textContent='请先选一个产业链主题'; return; }
  $('#btn-chain-ai').disabled=true;
  $('#chain-status').textContent='链路 AI 解读中（读产业链结构 + 检索研报 · 生成）…';
  try {
    const r=await fetch('/api/chain/interpret?theme='+encodeURIComponent(chainSel));
    if(!r.ok){ const e=await r.json(); throw new Error(e.detail||r.status); }
    const d=await r.json();
    $('#chain-ai').innerHTML=`<div class="ai"><h4>「${esc(chainSel)}」产业链 AI 解读`
      + `${d.model?' · '+esc(d.model):''}${d.cached?' · 缓存':''}</h4>${renderMarkdown(d.interpretation)}</div>`;
    $('#chain-status').textContent='';
    // 记入历史：点历史项可跳回趋势页、重载这条链的结构 + 解读。
    saveInterp('chain', `链路解读·${chainSel}`, {theme:chainSel});
  } catch(e){ $('#chain-status').innerHTML=`<span class="err">AI 解读出错：${esc(e.message)}</span>`; }
  finally { $('#btn-chain-ai').disabled=false; }
});

// 迷你中外占比条（环节级）。用 .cseg .s-split* 系列 CSS。
function miniSplit(sp){
  if(!sp || !sp.total) return '<div class="s-splitlg">该环节暂无匹配研报</div>';
  return `<div class="s-splitbar" title="国内券商 ${sp.domestic} / 外资投行 ${sp.foreign}">`
    + `<span class="seg dom" style="width:${sp.domestic_pct}%"></span>`
    + `<span class="seg for" style="width:${sp.foreign_pct}%"></span></div>`
    + `<div class="s-splitlg">国内 ${sp.domestic_pct}% · 外资 ${sp.foreign_pct}%（共 ${sp.total} 篇）</div>`;
}

function renderChain(d){
  const groups=d.groups||[];
  if(!groups.length){
    $('#chain-status').textContent='';
    $('#chain-out').innerHTML='<div class="drawer-empty">该主题尚未构建产业链结构。可在后台运行 build-chain 构建。</div>';
    return;
  }
  $('#chain-status').innerHTML=`<div class="chain-hd"><span class="c-theme">${esc(d.theme)}</span>`
    + `<span class="c-sub">${groups.length} 个分组 · 点标题展开看子环节</span></div>`;
  $('#chain-out').innerHTML = groups.map((g,gi)=>{
    const segs=(g.segments||[]).map(s=>{
      const tk=(s.tickers||[]).map(t=>{
        const code=t.code?`<span class="tk-code">${esc(t.code)}</span>`:'';
        const role=t.role?`（${esc(t.role)}）`:'';
        return `<span class="s-tk">${esc(t.name||'')}${code}${role}</span>`;
      }).join('');
      return `<div class="cseg">`
        + `<div class="s-name">${esc(s.name||'')}</div>`
        + (s.summary?`<div class="s-sum">${esc(s.summary)}</div>`:'')
        + (tk?`<div class="s-tks">${tk}</div>`:'')
        + `<div class="s-split">${miniSplit(s.split)}</div>`
        + `</div>`;
    }).join('');
    // 全部默认收起——用户点标题才展开（不再默认摊开上游）。
    return `<div class="cgroup">`
      + `<div class="cgroup-hd"><span class="caret">▸</span>`
      + `<span class="g-name">${esc(g.name||'')}</span>`
      + (g.stage?`<span class="g-stage">${esc(g.stage)}</span>`:'')
      + (g.summary?`<span class="g-sum">${esc(g.summary)}</span>`:'')
      + `</div>`
      + `<div class="cgroup-bd">${segs}</div></div>`;
  }).join('');
}

// 分组卡片展开/收起（事件委托，点标题切换）。
$('#chain-out') && ($('#chain-out').onclick = (ev) => {
  const hd=ev.target.closest('.cgroup-hd'); if(!hd) return;
  hd.parentElement.classList.toggle('open');
});

// 数字紧凑展示：整数直出，小数保留 2 位，去尾零。
function fmtNum(n){
  if(n==null) return '';
  if(Number.isInteger(n)) return ''+n;
  return (+n.toFixed(2)).toString();
}

// metric 时间线折线图（内联 SVG，无外部依赖）。
// 关键：绝不混画不同单位/货币——468美元与2888新台币画同一根轴会造出假尖峰。
// 策略：先把点按「口径」分组（优先归一单位 norm_unit，否则原始 unit），只画点数最多的
// 那一组（主口径），其余组的点数在图注里说明被略去。x 轴按日期升序、按真实时间比例定位。
function renderLine(points){
  // 每个点归一化取值：优先 norm_num（跨量级可比），否则 value_num；单位同理。
  const cand = points.map(p=>({
    date: p.date,
    y: (p.norm_num!=null? p.norm_num : p.value_num),
    unit: (p.norm_num!=null? (p.norm_unit||'') : (p.unit||'')),
  })).filter(p=>p.date && p.y!=null && isFinite(p.y));
  if(cand.length<2) return '';

  // 按单位口径分组，选点数最多的组为主口径（同口径才可比）。
  const groups = {};
  cand.forEach(p=>{ (groups[p.unit]=groups[p.unit]||[]).push(p); });
  const units = Object.keys(groups).sort((a,b)=>groups[b].length-groups[a].length);
  const mainUnit = units[0];
  const pts = groups[mainUnit].slice().sort((a,b)=> (a.date<b.date?-1:a.date>b.date?1:0));
  const dropped = cand.length - pts.length;  // 被略去的异口径点数
  if(pts.length<2) return '';

  // x 轴按真实日期比例（而非等距序号）定位，避免时间稀密失真。
  const W=920, H=220, pad=44;
  const toTs = s => { const d=new Date(s.length===7? s+'-01' : s); return isNaN(d)? null : d.getTime(); };
  const ts = pts.map(p=>toTs(p.date));
  const tmin=Math.min(...ts), tmax=Math.max(...ts), tr=(tmax-tmin)||1;
  const ys=pts.map(p=>p.y);
  const ymin=Math.min(...ys), ymax=Math.max(...ys), yr=(ymax-ymin)||1;
  const X=i=> pad + (ts[i]-tmin)/tr*(W-2*pad);
  const Y=v=> H-pad - (v-ymin)/yr*(H-2*pad);
  const line=pts.map((p,i)=>`${i?'L':'M'}${X(i).toFixed(1)},${Y(p.y).toFixed(1)}`).join(' ');
  const dots=pts.map((p,i)=>
    `<circle cx="${X(i).toFixed(1)}" cy="${Y(p.y).toFixed(1)}" r="3" fill="#4f8cff">`
    + `<title>${esc(p.date)}：${fmtNum(p.y)}${esc(p.unit)}</title></circle>`).join('');
  const yTop=`<text x="4" y="${(pad+4)}" fill="#8b93a1" font-size="11">${fmtNum(ymax)}${esc(mainUnit)}</text>`;
  const yBot=`<text x="4" y="${(H-pad+4)}" fill="#8b93a1" font-size="11">${fmtNum(ymin)}${esc(mainUnit)}</text>`;
  const xL=`<text x="${pad}" y="${H-8}" fill="#8b93a1" font-size="11">${esc(pts[0].date)}</text>`;
  const xR=`<text x="${W-pad}" y="${H-8}" fill="#8b93a1" font-size="11" text-anchor="end">${esc(pts[pts.length-1].date)}</text>`;
  const cap = `主口径「${esc(mainUnit||'无单位')}」· ${pts.length} 点`
            + (dropped>0? `（另有 ${dropped} 点为其他单位/货币，未混入图中）` : '');
  return `<div class="chart"><svg viewBox="0 0 ${W} ${H}" width="100%" preserveAspectRatio="xMidYMid meet">`
    + `<line x1="${pad}" y1="${H-pad}" x2="${W-pad}" y2="${H-pad}" stroke="#262b36"/>`
    + `<line x1="${pad}" y1="${pad}" x2="${pad}" y2="${H-pad}" stroke="#262b36"/>`
    + `<path d="${line}" fill="none" stroke="#4f8cff" stroke-width="1.5"/>`
    + dots + yTop + yBot + xL + xR
    + `</svg><div class="chart-cap">${cap}</div></div>`;
}

// 启动：所有声明就绪后再载入历史侧栏（let/const 有暂时性死区，初始化不能提前到声明之上）。
loadSessions(); renderSessions();
initChainThemes();
</script>
</body>
</html>"""
