"""命令行入口：build-catalog / index / ask / facts / timeline / doctor。

用法：
    python -m app.cli build-catalog
    python -m app.cli index [--no-embed] [--doc DOC_ID]
    python -m app.cli ask "问题" [--category ...] [--institution ...]
                          [--date-from ISO] [--date-to ISO] [--lexical] [--cheap]
    python -m app.cli facts [--strong] [--reextract] [--doc DOC_ID] [--limit N]
    python -m app.cli timeline metric --entity 茅台 [--metric 批价]
    python -m app.cli timeline theme [--weekly] [--top N]
    python -m app.cli doctor
"""
from __future__ import annotations

import argparse
import sqlite3
import sys

from .config import load_config, Config, InputsMissingError
from . import catalog, store

_FACTS_CONCURRENCY_FALLBACK = 5


def _facts_default_concurrency() -> int:
    """取 facts.DEFAULT_CONCURRENCY 作为 --concurrency 默认值（单一口径）。

    **惰性导入**：本模块其余命令都是在函数内才 import 各子模块，因为 `facts` 会连带
    拉起 anthropic SDK。parser 是模块级构建的，若在顶层 import facts，则连 `--help`
    都要先加载 SDK（慢，且 SDK 缺失时整个 CLI 直接崩）。故这里进函数才导，并对
    ImportError 兜底 —— 拿不到常量时退回字面量，只影响默认并发数，不该让 CLI 不可用。
    """
    try:
        from . import facts as facts_mod

        return facts_mod.DEFAULT_CONCURRENCY
    except ImportError:
        return _FACTS_CONCURRENCY_FALLBACK


def _cmd_index(cfg: Config, args) -> int:
    from . import index as index_mod

    st = index_mod.build_index(
        cfg,
        embed=not args.no_embed,
        only_doc_id=args.doc,
        only_new=args.only_new,
    )
    print("=== index 完成 ===")
    print(st.summary())
    return 0


def _cmd_embed_existing(cfg: Config, args) -> int:
    from . import index as index_mod

    st = index_mod.embed_existing(cfg)
    print("=== embed-existing 完成 ===")
    print(st.summary())
    return 0


def _cmd_update(cfg: Config, args) -> int:
    from . import update as update_mod

    rep = update_mod.run_update(
        cfg,
        embed=not args.no_embed,
        do_facts=not args.no_facts,
        facts_strong=args.strong,
    )
    print("=== update 完成 ===")
    print(rep.summary())
    return 0


def _cmd_ask(cfg: Config, args) -> int:
    from .retrieve import Filters
    from . import generate as gen_mod

    filters = Filters(
        date_from=args.date_from,
        date_to=args.date_to,
        institution=args.institution,
        category=args.category,
        stock_code=args.stock,
        lang=args.lang,
    )
    try:
        answer, result = gen_mod.ask(
            cfg,
            args.query,
            filters=filters,
            limit=args.limit,
            mode="lexical" if args.lexical else "auto",
            model_source="cheap" if args.cheap else "gen",
        )
    except gen_mod.GenerateError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    print(f"=== 回答（模式：{result.mode}"
          f"{' · LIKE 兜底' if result.used_like else ''}"
          f" · 命中 {len(result.hits)}）===\n")
    print(answer.text)
    if answer.sources:
        print("\n--- 来源 ---")
        for s in answer.sources:
            meta = " · ".join(
                x for x in [s.institution, s.title, s.report_date] if x
            )
            print(f"  [{s.ref}] {meta or s.doc_id}  ({s.chunk_id})")
    if answer.model:
        print(f"\n模型：{answer.model}")
    return 0


def _cmd_facts(cfg: Config, args) -> int:
    from . import facts as facts_mod

    try:
        st = facts_mod.build_facts(
            cfg,
            strong=args.strong,
            only_doc_id=args.doc,
            reextract=args.reextract,
            limit=args.limit,
            concurrency=args.concurrency,
        )
    except facts_mod.FactsError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    print("=== facts 抽取完成 ===")
    print(st.summary())
    return 0


def _cmd_timeline(cfg: Config, args) -> int:
    from . import timeline as tl

    conn = store.connect(cfg.paths.db)
    try:
        if args.kind == "metric":
            series = tl.metric_timeline(
                cfg,
                args.entity,
                args.metric,
                conn=conn,
                date_from=args.date_from,
                date_to=args.date_to,
            )
            if not series.points:
                print("（无匹配事实）")
                return 0
            print(f"=== {series.summary()} ===")
            for p in series.points:
                val = p.value_text or (
                    f"{p.value_num}" if p.value_num is not None else "")
                arrow = {"up": "↑", "down": "↓", "flat": "→"}.get(
                    p.direction or "", "")
                # 金额归一后附注统一口径（如 百万元 → 亿元），便于跨报告比较
                norm = ""
                if p.norm_num is not None and p.norm_unit:
                    norm = f"  [≈{p.norm_num:g}{p.norm_unit}]"
                print(f"  {p.date or '?'}  "
                      f"{p.entity or ''} · {p.metric or ''}: {val}{p.unit or ''} {arrow}{norm}")
                if p.quote:
                    print(f"      「{p.quote}」— {p.institution or ''} {p.title or ''}")
            # --ai：按需调 Claude 把上面这批碎片读成连贯解读（分清真时序与单篇多年预测）。
            if getattr(args, "ai", False):
                from . import generate as gen_mod
                print("\n=== AI 解读（调用 Claude）===")
                try:
                    ans = gen_mod.interpret_timeline(
                        cfg, series.entity, series.metric, series.points,
                        model_source="cheap" if args.cheap else "gen",
                    )
                    print(ans.text)
                    if ans.model:
                        print(f"\n（模型 {ans.model}·基于 {ans.used_hits} 条事实）")
                except gen_mod.GenerateError as exc:
                    print(f"[解读失败] {exc}", file=sys.stderr)
                    return 3
        else:  # theme：给 --theme 出该主题热度曲线，否则出全库总榜
            if args.theme:
                buckets = tl.theme_heat(cfg, args.theme, conn=conn,
                                        by="week" if args.week else "month")
                if not buckets:
                    print("（无匹配主题）")
                    return 0
                unit = "周" if args.week else "月"
                print(f"=== 主题热度：{args.theme}（按{unit}）===")
                for b in buckets:
                    print(f"  {b.bucket}: {b.count}")
            else:
                buckets = tl.top_themes(cfg, conn=conn, limit=args.top)
                print(f"=== 主题热度总榜（top {args.top}）===")
                for b in buckets:
                    print(f"  {b.bucket}: {b.count}")
        return 0
    finally:
        conn.close()


def _cmd_build_chain(cfg: Config, args) -> int:
    """构建某主题的产业链结构并落库（industry_chain）。一次性构建、稳定不常变，
    之后趋势面板下钻纯读库、零 LLM 成本。--web 尝试联网校验（relay 不支持则退化纯知识）。"""
    from . import chain as chain_mod

    theme = args.theme.strip()
    if not theme:
        print("[错误] 需要 --theme", file=sys.stderr)
        return 2
    print(f"构建产业链：{theme}"
          f"{'（含联网校验）' if args.web else '（纯模型知识）'} …")
    try:
        res = chain_mod.build_chain(
            cfg, theme, use_web=args.web,
            model_source="cheap" if args.cheap else "gen",
        )
    except chain_mod.ChainError as exc:
        print(f"[构建失败] {exc}", file=sys.stderr)
        return 3
    src = "web+model" if res.get("web") else "model"
    print(f"完成：{res['groups']} 个分组 / {res['segments']} 个环节（来源 {src}）。"
          f"到「研报趋势」页选该主题下钻查看。")
    # 重建 diff：链是 LLM 生成的，两次结果不会完全一致。打出增删让你一眼看出
    # 这次重建是不是把原本有用的环节弄丢了（丢了就再跑一次，或人工核对）。
    diff = res.get("diff") or {}
    if diff.get("had_old"):
        added, removed = diff.get("added") or [], diff.get("removed") or []
        print(f"\n--- 与旧链对比（保留 {diff.get('kept', 0)} 个环节）---")
        if added:
            print(f"  + 新增 {len(added)}：{'、'.join(added)}")
        if removed:
            print(f"  - 消失 {len(removed)}：{'、'.join(removed)}")
            print("    注意：消失的环节若本该保留，重跑一次或人工核对（生成有随机性）。")
        if not added and not removed:
            print("  环节集合无变化。")
    if getattr(args, "show", False):
        view = chain_mod.get_chain_view(cfg, theme)
        print(f"\n=== {theme} 产业链结构 ===")
        for g in view["groups"]:
            stage = f"[{g['stage']}]" if g.get("stage") else ""
            print(f"\n▸ {g['name']} {stage}")
            for s in g["segments"]:
                sp = s.get("split") or {}
                tks = "、".join(
                    t.get("name", "") for t in (s.get("tickers") or [])
                )
                print(f"    · {s['name']}"
                      f"（国内 {sp.get('domestic', 0)} / 外资 {sp.get('foreign', 0)}）")
                if tks:
                    print(f"        标的：{tks}")
    return 0


def _cmd_chain_drift(cfg: Config, args) -> int:
    """产业链漂移检测 / 候选审核（零 LLM 成本，纯 SQL + segnorm）。

    新研报进来后链结构不会自己长出新环节，这里找出「库里热度够、跨机构提、且现有
    环节未覆盖」的方向，写进 chain_candidate 等人审——**绝不自动并入正式链**。
    审核用 --mark <cand_key> <watching|merged|rejected>；「接受」的真实动作是
    重跑 build-chain 把它纳入结构，然后标 merged 留档。
    """
    from . import chain as chain_mod

    con = store.connect(cfg.paths.db)
    try:
        # --mark：标记某候选的审核状态。
        if getattr(args, "mark", None):
            key, status = args.mark
            theme = (args.theme or "").strip()
            if not theme:
                print("[错误] --mark 需要同时给出主题（位置参数）", file=sys.stderr)
                return 2
            try:
                ok = store.set_candidate_status(con, theme, key, status, args.note)
            except ValueError as exc:
                print(f"[错误] {exc}", file=sys.stderr)
                return 2
            con.commit()
            print(f"{'已标记' if ok else '未找到该候选'}：{theme} / {key} → {status}")
            return 0 if ok else 4

        # --list：只看已存的候选，不重新检测。
        if getattr(args, "list_only", False):
            rows = store.list_candidates(con, theme=args.theme, status=args.status)
            if not rows:
                print("（无候选记录）")
                return 0
            print(f"=== 漂移候选 {len(rows)} 条 ===")
            for r in rows:
                print(f"  [{r['status']:<9}] {r['theme']} → {r['name']}"
                      f"（key={r['cand_key']}）"
                      f" 文档 {r['doc_count']} / 机构 {r['inst_count']}"
                      f" / {r['first_seen']}~{r['last_seen']}")
                if r.get("note"):
                    print(f"      备注：{r['note']}")
            return 0

        # 默认：跑检测。给主题则只查该链，否则遍历全部已建链。
        themes = [args.theme.strip()] if args.theme else store.list_chain_themes(con)
        if not themes:
            print("（库里还没有已构建的产业链，先跑 build-chain）")
            return 0
        kw = {}
        if args.min_docs is not None:
            kw["min_docs"] = args.min_docs
        if args.min_insts is not None:
            kw["min_insts"] = args.min_insts
        if args.min_lift is not None:
            kw["min_lift"] = args.min_lift

        total = 0
        for th in themes:
            try:
                cands = chain_mod.detect_drift(
                    cfg, th, conn=con, persist=not args.no_persist, **kw
                )
            except chain_mod.ChainError as exc:
                print(f"[跳过] {th}：{exc}", file=sys.stderr)
                continue
            if not cands:
                continue
            total += len(cands)
            print(f"\n=== {th}：{len(cands)} 个未覆盖方向 ===")
            for c in cands:
                print(f"  · {c['name']}（key={c['cand_key']}）"
                      f" 文档 {c['doc_count']} / 机构 {c['inst_count']}"
                      f" / 集中度 {c['lift']}x / {c['first_seen']}~{c['last_seen']}")
                for s in c["samples"][:3]:
                    print(f"      - {s.get('institution') or '?'}"
                          f" {s.get('report_date') or ''}"
                          f" {(s.get('title') or '')[:48]}")
        if not total:
            print("未发现未覆盖的新方向（现有链已覆盖库内热点）。")
        else:
            print(f"\n共 {total} 个候选"
                  f"{'（已写入 chain_candidate 等审核）' if not args.no_persist else ''}。"
                  f"\n审核：chain-drift <主题> --mark <key> <merged|rejected|watching>")
        return 0
    finally:
        con.close()


def _cmd_verify(cfg: Config, args) -> int:
    """体检 facts：扫全库，报 value_num 与 value_text 量级不符、及 as_of_date 明显背离
    report_date 的可疑行（只读，不改库）。

    两类校验：
    - value_num 自洽（factnorm.verify_value_num）：默认只列「量级不符」，--include-fillable
      时也列「value_num 空但 value_text 是纯数字」的可补项。
    - as_of_date 合理性（factnorm.verify_as_of_date）：as_of 与报告日相差过大（模型常把
      历史行情日/未来目标年月误填进 as_of_date），标记为可疑——timeline 已改用 report_date
      为主排序规避，此处只做体检报告。
    """
    from . import factnorm

    if not cfg.paths.db.is_file():
        print("[错误] 库不存在，先跑 build-catalog + index + facts", file=sys.stderr)
        return 2
    conn = store.connect(cfg.paths.db)
    try:
        rows = conn.execute(
            "SELECT fact_id, doc_id, entity, metric, value_num, value_text, unit, "
            "report_date, as_of_date FROM facts"
        ).fetchall()
        total = len(rows)
        suspicious: list[tuple] = []
        fillable = 0
        date_susp: list[tuple] = []
        for r in rows:
            msg = factnorm.verify_value_num(
                r["value_text"], r["value_num"], r["unit"]
            )
            if msg:
                if "可补" in msg:
                    fillable += 1
                    if args.include_fillable:
                        suspicious.append((r["fact_id"], r["entity"], r["metric"],
                                           r["value_num"], r["value_text"],
                                           r["unit"], msg))
                else:
                    suspicious.append((r["fact_id"], r["entity"], r["metric"],
                                       r["value_num"], r["value_text"],
                                       r["unit"], msg))
            dmsg = factnorm.verify_as_of_date(
                r["as_of_date"], r["report_date"], r["metric"]
            )
            if dmsg:
                date_susp.append((r["fact_id"], r["entity"], r["metric"],
                                  r["as_of_date"], r["report_date"], dmsg))

        value_susp = len(suspicious) - (fillable if args.include_fillable else 0)
        print("=== facts 校验（value_num + as_of_date）===")
        print(f"总事实：{total}")
        print(f"  value_num 量级可疑：{value_susp}；value_num 可补：{fillable}")
        print(f"  as_of_date 背离报告日：{len(date_susp)}")

        print("\n[value_num 量级可疑]")
        shown = suspicious[: args.limit]
        for fid, ent, met, vn, vt, unit, msg in shown:
            print(f"  #{fid} {ent or ''} · {met or ''}: "
                  f"value_num={vn} value_text={vt!r} unit={unit or ''}")
            print(f"      → {msg}")
        if len(suspicious) > len(shown):
            print(f"  …还有 {len(suspicious) - len(shown)} 条（--limit 调大查看）")

        print("\n[as_of_date 背离报告日]")
        dshown = date_susp[: args.limit]
        for fid, ent, met, aod, rd, msg in dshown:
            print(f"  #{fid} {ent or ''} · {met or ''}: "
                  f"as_of={aod} report_date={rd}")
            print(f"      → {msg}")
        if len(date_susp) > len(dshown):
            print(f"  …还有 {len(date_susp) - len(dshown)} 条（--limit 调大查看）")
        return 0
    finally:
        conn.close()


def _cmd_build_catalog(cfg: Config) -> int:
    try:
        report = catalog.build_catalog(cfg)
    except InputsMissingError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    print("=== build-catalog 完成 ===")
    print(report.summary())
    if report.skipped_docs:
        print(f"\n跳过的 doc_id（前 10）：{report.skipped_docs[:10]}")
    return 0


def _cmd_reindex_fts(cfg: Config) -> int:
    """从现有 chunks 重建 FTS（jieba 分词），不重新切块/嵌入。schema 变更后用。"""
    import sqlite3

    from . import store

    if not cfg.paths.db.is_file():
        print("[错误] 库不存在，先跑 build-catalog + index", file=sys.stderr)
        return 2
    conn = sqlite3.connect(cfg.paths.db)
    conn.row_factory = sqlite3.Row
    try:
        n = store.rebuild_fts(conn)
        conn.commit()
    finally:
        conn.close()
    print(f"=== reindex-fts 完成：重建 {n} 块的 FTS（jieba 分词）===")
    return 0


def _cmd_serve(cfg: Config, args) -> int:
    try:
        from . import web
    except ImportError as exc:
        print(f"[错误] Web 依赖缺失（fastapi/uvicorn）：{exc}", file=sys.stderr)
        print("  → pip install fastapi uvicorn", file=sys.stderr)
        return 2
    print(f"=== 启动本地 Web（http://{args.host}:{args.port}）===")
    if args.host in ("127.0.0.1", "localhost"):
        print("  仅本机可访问、无鉴权——付费研报库与 LLM 计费入口不暴露公网。")
    web.serve(cfg, host=args.host, port=args.port)
    return 0


def _cmd_doctor(cfg: Config) -> int:
    print("=== doctor 体检 ===")
    print(f"Python: {sys.version.split()[0]}")

    # 输入产物
    print("\n[输入产物]")
    canonical_ok = cfg.paths.canonical.is_dir()
    manifest_ok = cfg.paths.manifest.is_file()
    print(f"  canonical: {cfg.paths.canonical}  {'OK' if canonical_ok else '缺失'}")
    print(f"  manifest : {cfg.paths.manifest}  {'OK' if manifest_ok else '缺失'}")
    if canonical_ok:
        n_dirs = sum(1 for p in cfg.paths.canonical.iterdir() if (p / "full.md").is_file())
        print(f"  canonical 下含 full.md 的目录数: {n_dirs}")
    if manifest_ok:
        n_success = sum(1 for _ in catalog._iter_manifest_success(cfg.paths.manifest))
        print(f"  manifest status=success 行数: {n_success}")

    # Claude 端点（只报 base_url + 来源，绝不打印 key 值）
    print("\n[Claude 端点]")
    print(f"  base_url: {cfg.llm.base_url or '<未解析到>'}")
    print(f"  来源    : {cfg.llm.source}")
    print(f"  key     : {cfg.llm.key_redacted}")

    # Embedding 端点
    print("\n[Embedding 端点]")
    print(f"  base_url  : {cfg.embed.base_url}")
    print(f"  model     : {cfg.embed.model} @ {cfg.embed.dimensions} 维")
    print(f"  key       : {cfg.embed.key_redacted}")
    # 开关与 key 分开报：key 留着但开关关掉是**正常配置**，不是异常。
    # 不显式打出来的话，"检索为什么没走向量"要翻源码才知道。
    if cfg.embed.enabled:
        print("  向量检索  : 启用（BM25 + 稠密 RRF 融合）")
    else:
        print("  向量检索  : 已关闭（[embed].enabled=false）→ 检索走纯 BM25，"
              "不调嵌入端点")

    # 数据库
    print("\n[数据库]")
    print(f"  db: {cfg.paths.db}")
    if cfg.paths.db.is_file():
        try:
            conn = sqlite3.connect(cfg.paths.db)
            n = store.count_documents(conn)
            n_review = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE needs_review=1"
            ).fetchone()[0]
            conn.close()
            print(f"  documents 行数: {n}（needs_review: {n_review}）")
        except sqlite3.Error as exc:
            print(f"  读库失败: {exc}")
    else:
        print("  （尚未建库，先跑 build-catalog）")

    # 依赖
    print("\n[阶段 1 依赖预检]")
    for mod in ("openai", "tiktoken", "sqlite_vec"):
        try:
            __import__(mod)
            print(f"  {mod}: 已装")
        except ImportError:
            print(f"  {mod}: 未装（阶段 1 需要）")

    print("\n[阶段 3 Web 依赖预检]")
    for mod in ("fastapi", "uvicorn"):
        try:
            __import__(mod)
            print(f"  {mod}: 已装")
        except ImportError:
            print(f"  {mod}: 未装（serve 需要）")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="yanbao", description="研报本地长效 AI 系统 CLI")
    parser.add_argument("--config", help="config.toml 路径（默认 yanbao_ai/config.toml）")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-catalog", help="manifest 为主重建 documents 表")

    p_index = sub.add_parser("index", help="规范化→切块→嵌入→落库（chunks/FTS/向量）")
    p_index.add_argument("--no-embed", action="store_true",
                         help="跳过嵌入，只建 chunks + FTS（纯词法检索）")
    p_index.add_argument("--doc", help="只索引指定 doc_id（默认全量）")
    p_index.add_argument("--only-new", dest="only_new", action="store_true",
                         help="只索引尚无 chunk 的文档（增量）")

    sub.add_parser("embed-existing",
                   help="为已有 chunks 回填向量（不重切块，断点续跑，qwen 迁移用）")

    p_update = sub.add_parser("update", help="增量更新一键跑：catalog→index(仅新)→facts(增量)")
    p_update.add_argument("--no-embed", action="store_true",
                          help="index 步跳过嵌入（纯词法）")
    p_update.add_argument("--no-facts", action="store_true",
                          help="跳过 facts 抽取步（无 Claude 端点/额度时）")
    p_update.add_argument("--strong", action="store_true",
                          help="facts 步用 model_gen（强）")

    p_ask = sub.add_parser("ask", help="检索 + Claude 生成带引用的答案")
    p_ask.add_argument("query", help="问题")
    p_ask.add_argument("--category", help="国内券商报告 / 投行报告")
    p_ask.add_argument("--institution", help="机构名（子串匹配）")
    p_ask.add_argument("--stock", help="股票代码")
    p_ask.add_argument("--lang", help="zh / en")
    p_ask.add_argument("--date-from", dest="date_from", help="起始日期 ISO（含）")
    p_ask.add_argument("--date-to", dest="date_to", help="截止日期 ISO（含）")
    p_ask.add_argument("--limit", type=int, default=12, help="送入生成的材料块数上限")
    p_ask.add_argument("--lexical", action="store_true", help="强制纯词法检索（不调嵌入）")
    p_ask.add_argument("--cheap", action="store_true", help="用 model_cheap（省）生成")

    p_facts = sub.add_parser("facts", help="Claude 结构化抽取事实 + 主题（时间线骨架）")
    p_facts.add_argument("--strong", action="store_true",
                         help="用 model_gen（强）抽取，默认 model_cheap（省）")
    p_facts.add_argument("--doc", help="只抽取指定 doc_id（默认全量增量）")
    p_facts.add_argument("--reextract", action="store_true",
                         help="强制重抽（忽略已抽记录，先清后写）")
    p_facts.add_argument("--limit", type=int, help="只处理前 N 篇（试跑用）")
    # 默认值引用 facts.DEFAULT_CONCURRENCY，**不要在这里再写一个字面量**：
    # 同一口径存两处，改了模块常量而漏改 CLI 就会出现"命令行跑和代码里跑并发不同"的
    # 静默不一致（本仓库已在 _client 上栽过一次同类跟头）。
    p_facts.add_argument("--concurrency", type=int, default=_facts_default_concurrency(),
                         help=f"并发抽取线程数（默认 {_facts_default_concurrency()}；"
                              f"上游拥堵时调高无用，见 facts.py 注释）")

    p_tl = sub.add_parser("timeline", help="指标/主题时间线查询（纯 SQL，零成本）")
    tl_sub = p_tl.add_subparsers(dest="kind", required=True)
    p_tl_m = tl_sub.add_parser("metric", help="某实体某指标的取值时间线")
    p_tl_m.add_argument("entity", help="实体名（子串匹配，如 茅台）")
    p_tl_m.add_argument("--metric", help="指标名（精确，如 批价）")
    p_tl_m.add_argument("--date-from", dest="date_from", help="起始日期 ISO（含）")
    p_tl_m.add_argument("--date-to", dest="date_to", help="截止日期 ISO（含）")
    p_tl_m.add_argument("--ai", action="store_true",
                        help="把 SQL 捞到的碎片交给 Claude 读成连贯解读（产生费用）")
    p_tl_m.add_argument("--cheap", action="store_true",
                        help="--ai 时用省钱模型")
    p_tl_t = tl_sub.add_parser("theme", help="主题热度：按月/周聚合或全库总榜")
    p_tl_t.add_argument("--theme", help="主题名（给定则出该主题热度曲线）")
    p_tl_t.add_argument("--week", action="store_true", help="按周聚合（默认按月）")
    p_tl_t.add_argument("--top", type=int, default=30, help="无 --theme 时列总榜前 N")

    p_serve = sub.add_parser("serve", help="启动本地 Web 界面（FastAPI，绑 127.0.0.1）")
    p_serve.add_argument("--host", default="127.0.0.1",
                         help="监听地址（默认 127.0.0.1；非本机会警告，无鉴权勿暴露公网）")
    p_serve.add_argument("--port", type=int, default=8000, help="端口（默认 8000）")

    sub.add_parser("reindex-fts", help="仅重建 FTS 全文索引（jieba 分词，不重切块/不重嵌入）")

    p_verify = sub.add_parser("verify", help="校验 facts 的 value_num 与 value_text 是否自洽（只读，不改库）")
    p_verify.add_argument("--limit", type=int, default=50,
                          help="最多列出前 N 条可疑记录（默认 50）")
    p_verify.add_argument("--include-fillable", dest="include_fillable",
                          action="store_true",
                          help="也列出 value_num 空但 value_text 是纯数字的可补项")

    p_chain = sub.add_parser("build-chain",
                             help="用 Claude 构建某主题的产业链结构并落库（供趋势面板下钻）")
    p_chain.add_argument("theme", help="主题名（如 国产算力 / 半导体材料）")
    p_chain.add_argument("--web", action="store_true",
                         help="尝试用中转站 web_search 联网校验（不支持则退化为纯知识）")
    p_chain.add_argument("--cheap", action="store_true",
                         help="用 model_cheap（默认强模型 model_gen，构建质量优先）")
    p_chain.add_argument("--show", action="store_true",
                         help="构建后打印读回的链结构（含各环节中外占比）")

    p_drift = sub.add_parser(
        "chain-drift",
        help="产业链漂移检测：找出研报里热度够、但现有链未覆盖的方向（写候选表等人审，零 LLM 成本）")
    p_drift.add_argument("theme", nargs="?",
                         help="主题名；留空则检测全部已建链的主题")
    p_drift.add_argument("--min-docs", type=int, default=None,
                         help="支持文档数下限（默认 8）")
    p_drift.add_argument("--min-insts", type=int, default=None,
                         help="提及机构数下限（默认 3，跨机构才可信）")
    p_drift.add_argument("--min-lift", type=float, default=None,
                         help="集中度下限（默认 2.5，挡掉到处都有的泛标签）")
    p_drift.add_argument("--no-persist", action="store_true",
                         help="只看不写候选表")
    p_drift.add_argument("--list", dest="list_only", action="store_true",
                         help="不检测，只列出候选表现有条目")
    p_drift.add_argument("--status", help="配合 --list 按状态过滤（watching/merged/rejected）")
    p_drift.add_argument("--mark", nargs=2, metavar=("CAND_KEY", "STATUS"),
                         help="标记某候选的审核状态：watching / merged / rejected")
    p_drift.add_argument("--note", help="配合 --mark 写一句备注（为什么这么判）")

    sub.add_parser("doctor", help="体检：产物/端点/库/依赖")

    args = parser.parse_args(argv)
    from pathlib import Path

    cfg = load_config(Path(args.config) if args.config else None)

    if args.command == "build-catalog":
        return _cmd_build_catalog(cfg)
    if args.command == "index":
        return _cmd_index(cfg, args)
    if args.command == "embed-existing":
        return _cmd_embed_existing(cfg, args)
    if args.command == "update":
        return _cmd_update(cfg, args)
    if args.command == "ask":
        return _cmd_ask(cfg, args)
    if args.command == "facts":
        return _cmd_facts(cfg, args)
    if args.command == "timeline":
        return _cmd_timeline(cfg, args)
    if args.command == "doctor":
        return _cmd_doctor(cfg)
    if args.command == "reindex-fts":
        return _cmd_reindex_fts(cfg)
    if args.command == "verify":
        return _cmd_verify(cfg, args)
    if args.command == "serve":
        return _cmd_serve(cfg, args)
    if args.command == "build-chain":
        return _cmd_build_chain(cfg, args)
    if args.command == "chain-drift":
        return _cmd_chain_drift(cfg, args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
