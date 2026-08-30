"""Chain：产业链结构的 AI 构建 + 读取（含每环节中外研报占比）。

设计（用户确认的方向）：
- 产业链结构相对稳定、有行业共识，故**一次性构建后落库**（industry_chain 表），
  之后趋势面板下钻纯读库、零 LLM 成本。产业链不常变，构建时力求准确。
- 构建来源 = Claude 的行业知识（对成熟产业链共识度高）+ 可选少量网络搜索佐证
  （--web，经中转站的 server 端 web_search 工具；relay 不支持则优雅退化为纯知识）。
  另外用本库已抽取的 themes/标题做“落地校验”——关键词能否在语料里匹配到研报。
- 用 tool-use（emit_chain）拿结构化输出，SDK 保证 schema 合规 dict，
  规避手工 JSON 解析在含引号/换行时的整类失败（沿用 facts.py 的经验）。

读取侧：get_chain_view 把落库的链读成 分组→环节 的树，并为每个叶子环节现算
“该环节研报的中外数量占比”（keywords LIKE 匹配 documents.title + doc_themes.theme，
按 lang 分中资/外资）——这正是用户要的“下面分类的研报国内外占比”。
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time

from .config import Config
from . import store
# 所有 Claude 请求统一走 create_message（内部流式），避免长输出被中转站掐断连接。
from .generate import create_message, is_transient_error

DEFAULT_MODEL_SOURCE = "gen"   # 构建质量优先，默认用强模型
MAX_KEYWORDS_MATCH = 12         # 单环节参与占比统计的关键词上限（防 SQL 过长）


class ChainError(RuntimeError):
    """本模块对外的统一异常。"""


class ChainToolCallError(ChainError):
    """emit_chain 的结果不可用（缺 tool_use 块，或 groups 为空）—— **瞬时**，值得重试。

    继承 ChainError，故调用方原有的 `except ChainError` 不受影响。

    实测由来（2026-08-28，重建半导体链）：`build_chain` 报「本次生成 0 个环节，已回滚」。
    起初怀疑是参考料里负面约束（"本链不要单列 XX"列了 12 条）把模型逼成什么都不输出，
    但拿同一份 941 字长参考料复跑两次，**均正常产出 5 分组 / 30 与 26 个环节**，
    三个目标环节（先进封装/先进制程/封装基板）全部到位 —— 假设被推翻，那次就是一次
    瞬时故障：流被掐断后 tool_use 块残缺，JSON 仍合法但 `groups` 是空数组。

    schema 里 `groups` 不是必填，`{"groups": []}` 完全合法，于是空结果一路静默流到
    `_persist_chain` 才被"0 环节则回滚"拦下 —— 白烧一次 opus-5 长输出且不会重试。
    故：在生成层就把"空 groups"判为瞬时错误并重试，别等到写库才失败。
    """


# 工具化结构输出：让模型把某主题产业链输出成 分组(group)→环节(segment) 两层树。
# 每个环节带 代表标的(tickers) 与 检索关键词(keywords，中英并给，供占比统计 LIKE 匹配)。
_CHAIN_TOOL = {
    "name": "emit_chain",
    "description": "输出某主题的产业链结构：上/中/下游（或需求/供给侧）分组，每组下若干具体环节，每个环节带代表标的与检索关键词。",
    "input_schema": {
        "type": "object",
        "properties": {
            "groups": {
                "type": "array",
                "description": "产业链的一级分组（上游/中游/下游，或需求侧/供给侧等）",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "分组名（如 上游·材料设备 / 需求侧·算力硬件）"},
                        "stage": {"type": "string", "description": "上游 / 中游 / 下游 或 需求侧 / 供给侧"},
                        "summary": {"type": "string", "description": "该分组一句话定位"},
                        "segments": {
                            "type": "array",
                            "description": "该分组下的具体环节",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "环节名（如 光模块/CPO、半导体设备）"},
                                    "summary": {"type": "string", "description": "一句话定位：壁垒/国产化率/竞争格局"},
                                    "tickers": {
                                        "type": "array",
                                        "description": "代表标的（按代表性排序，3-8 个）",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string", "description": "公司名（如 中际旭创）"},
                                                "code": {"type": "string", "description": "股票代码或留空（如 300308）"},
                                                "role": {"type": "string", "description": "在该环节的定位（如 光模块龙头）"},
                                            },
                                            "required": ["name"],
                                        },
                                    },
                                    "keywords": {
                                        "type": "array",
                                        "description": "该环节的检索关键词，中英并给（如 [\"光模块\",\"CPO\",\"optical module\",\"co-packaged optics\"]），用于在研报库匹配该环节相关研报",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": ["name", "keywords"],
                            },
                        },
                    },
                    "required": ["name", "segments"],
                },
            },
        },
        "required": ["groups"],
    },
}

_CHAIN_SYSTEM = """你是产业链结构专家。用户会给一个主题（如 半导体材料 / 网络安全）。
请你凭行业共识，把该主题的结构梳理成**分组（group）→ 具体环节（segment）**两层，
通过 emit_chain 工具输出。要求：

1. **先判断该主题有没有真实的线性上下游关系**，据此选分组方式，严禁硬套造假：
   - 有清晰的资源→加工→制造→应用链条（如煤炭/光伏/半导体/锂电池）：按**上游/中游/下游**分组。
   - 有明确的供需两端但非线性（少数）：按**需求侧/供给侧**分组。
   - **没有线性上下游**（如网络安全/工业自动化/基础软件/军工/医疗器械/食品饮料等）：
     **按细分赛道 / 技术方向 / 应用场景平行分组**，stage 填「细分领域」之类，
     **绝不要硬凑出「上游/中游/下游」这种其实不存在的三段式**——硬凑的分组是假结构，
     会误导下游判断。分组方式要如实反映这个行业真正的组织逻辑。
2. 每个环节给出：一句话定位（壁垒/国产化率/竞争格局），3-8 个代表标的（A股优先给代码），
   以及一组检索关键词——**必须中英并给**（外资投行研报用英文表述同一环节），
   关键词要具体到能在研报里精确匹配（如 光模块 用 [光模块, CPO, optical module, co-packaged optics]）。
3. 覆盖要全：宁可多列一个真实存在的环节，也不要漏掉整条支线。但只列该主题**真实存在**的
   环节，不要硬凑。
4. 代表标的要给该环节里真正有代表性的公司，别把蹭概念的塞进来。
5. **过细的单一产品（如 MDI/制冷剂/钛白粉/草甘膦/银浆/金刚线/光伏胶膜/减速器/丝杠）
   与单一技术路线（如 TOPCon/HJT/BC/钙钛矿/固态电池/钠离子/液流电池/Micro LED/CPO/硅光/
   人形机器人）不要单列成顶层环节**——它们是某个环节内的产品节点或技术标签，
   应并入对应环节的关键词或定位里，别让它们撑起一个独立分组。
6. 若提供了【参考结构】，它给出了流向和主要子链的初步设想，但**这份参考的分类不一定正确，
   甚至可能有错**——你必须结合行业共识核实、纠正、增补，**绝不要照搬**。参考只是线索。"""


def _client(cfg: Config):
    """构造 Anthropic 客户端 —— **直接复用 generate._client，不再自己拼**。

    这里曾有一份独立拷贝（`Anthropic(base_url=..., auth_token=...)`，Bearer 且
    不覆盖 User-Agent）。2026-08-28 改 UA 绕中转站黑名单时只改了 generate 那份，
    facts / chain 各自的拷贝没跟上 → 直测 200、一跑批量却全 403，排查绕了一圈。
    端点/鉴权/UA 的口径只在 generate._client 里维护一次。
    """
    from .generate import _client as _gen_client, GenerateError

    try:
        return _gen_client(cfg)
    except GenerateError as exc:
        # 对外仍抛 ChainError，保持本模块调用方的异常契约不变。
        raise ChainError(str(exc)) from exc


def _slug(s: str) -> str:
    """把名字压成稳定 slug（保留中英数字，其余转 -），用于拼稳定 node_id。"""
    s = re.sub(r"[^\w一-鿿]+", "-", s.strip())
    return s.strip("-")[:40] or "x"


def _web_research(client, model: str, theme: str) -> str | None:
    """可选：用中转站的 server 端 web_search 工具查该主题产业链的权威梳理。

    relay 不支持该工具（多数中转站不透传）时会抛错——捕获后返回 None，
    构建退化为纯模型知识（用户已认可产业链共识度高）。绝不因联网失败拖垮构建。
    """
    try:
        resp = create_message(
            client,
            model=model,
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": (
                    f"检索“{theme}”的产业链结构（上中下游各环节、代表公司）。"
                    f"只用于给产业链梳理做事实校验，简要列出各环节与代表公司即可。"
                ),
            }],
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 4}],
        )
        text = "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        )
        return text.strip() or None
    except Exception:  # noqa: BLE001 - 联网不可用则退化，非致命
        return None


def _is_transient(exc: Exception) -> bool:
    """本模块专属异常在前，其余交给通用判定（见 generate.is_transient_error）。

    这里原先是 `"429" in msg or "timeout" in msg.lower() or ...` 的字符串匹配 ——
    与 facts 里 2026-08-12 静默丢 1605 篇的那份是同一个写法。空串异常（SDK 流层的
    裸 AssertionError）匹配不到任何关键词，会被判成永久错误、一次都不重试。
    """
    # 顺序要紧：子类先判，否则被父类那条 `→ False` 抢先拦成"不重试"。
    if isinstance(exc, ChainToolCallError):
        return True    # 空结构/缺 tool_use = 流被掐断，实测重发多半就成
    if isinstance(exc, ChainError):
        return False   # 其余自己抛的（端点缺失/配置错等）重试无用
    return is_transient_error(exc)


def _emit_chain(client, model: str, theme: str, web_ctx: str | None,
                reference: str | None = None, *, max_retries: int = 5) -> dict:
    """调 Claude 输出结构化产业链（emit_chain 工具）。带瞬时错误退避重试。

    reference：可选的人工参考结构（流向 + 主要子链设想）。注入 prompt 供模型核实纠正，
    prompt 已明确要求「不照搬、须核实」，故传错分类不会被无脑采纳。
    """
    user = f"主题：{theme}"
    if reference:
        user += f"\n\n【参考结构（初步设想，分类不一定对，须核实纠正，勿照搬）】\n{reference.strip()}"
    if web_ctx:
        user += f"\n\n【参考资料（网络搜索，仅供校验）】\n{web_ctx[:4000]}"
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = create_message(
                client,
                model=model,
                max_tokens=4000,
                system=_CHAIN_SYSTEM,
                tools=[_CHAIN_TOOL],
                tool_choice={"type": "tool", "name": "emit_chain"},
                messages=[{"role": "user", "content": user}],
            )
            for b in resp.content:
                if getattr(b, "type", None) == "tool_use" and b.name == "emit_chain":
                    data = b.input
                    # **空结果要在这里就拦住并重试**：schema 里 groups 非必填，
                    # `{"groups": []}` 合法，放过去就会一路静默走到 _persist_chain
                    # 才以"0 环节"失败——白烧一次长输出且不重试（见 ChainToolCallError）。
                    groups = data.get("groups") or []
                    n_seg = sum(
                        len(g.get("segments") or [])
                        for g in groups if isinstance(g, dict)
                    )
                    if not groups or n_seg == 0:
                        raise ChainToolCallError(
                            f"emit_chain 返回空结构（groups={len(groups)}, "
                            f"segments={n_seg}, stop_reason="
                            f"{getattr(resp, 'stop_reason', None)!r}）"
                        )
                    return data
            kinds = [getattr(b, "type", "?") for b in resp.content]
            raise ChainToolCallError(
                f"响应无 emit_chain tool_use 块（blocks={kinds}, "
                f"stop_reason={getattr(resp, 'stop_reason', None)!r}）"
            )
        except Exception as exc:  # noqa: BLE001 - 按错误类别决定重试/放弃
            if not _is_transient(exc) or attempt == max_retries - 1:
                raise
            last_err = exc
            time.sleep(min(2 ** (attempt + 1), 16) + random.uniform(0, 1))
    raise last_err or ChainError("产业链构建失败")


def _generate_chain_data(
    cfg: Config,
    theme: str,
    *,
    reference: str | None = None,
    use_web: bool = False,
    model_source: str = DEFAULT_MODEL_SOURCE,
) -> dict:
    """只跑 LLM，产出该主题产业链的结构 dict——**不碰 DB**。

    抽出来是为了批量重建时能**并发跑 LLM（慢活）、串行写库（快、避免 SQLite 写锁）**。
    返回 {"theme", "groups"(原始 emit_chain 输出), "built_by"}。
    """
    model = cfg.llm.model_cheap if model_source == "cheap" else cfg.llm.model_gen
    client = _client(cfg)
    web_ctx = _web_research(client, model, theme) if use_web else None
    data = _emit_chain(client, model, theme, web_ctx, reference)
    return {
        "theme": theme,
        "groups": data.get("groups", []) or [],
        "built_by": "web+model" if web_ctx else "model",
    }


def _persist_chain(conn, theme: str, groups: list, built_by: str) -> dict:
    """把生成好的产业链结构写库（幂等：先删该主题旧链再写），返回 {groups, segments, diff}。

    **写入安全**（单人本地库，不做版本表/回滚，但下面三件必须有）：
    1. 先取旧链快照，重建后给出 diff（新增/消失了哪些环节）——重建是 LLM 生成，
       每次结果不完全一致，没有 diff 就看不出这次是否把好环节弄丢了。
    2. delete + 全部 write 包在**一个显式事务**里：中途失败整体回滚，不会留下
       "旧链已删、新链只写了一半"的残缺状态。
    3. 0 环节时回滚——宁可保留旧链也不要写出空链。

    注意：不自开/关闭 conn（由调用方管理），故批量重建可复用同一连接串行写。
    """
    from . import segnorm

    old_nodes = store.get_chain(conn, theme)
    old_segs = {
        segnorm.canonical(n["name"]): n["name"]
        for n in old_nodes if n["node_type"] == "segment"
    }

    n_group = n_seg = 0
    new_segs: dict[str, str] = {}
    # 显式事务：delete 与所有 write 要么全成、要么全不成。
    conn.execute("BEGIN IMMEDIATE")
    try:
        store.delete_chain(conn, theme)
        for gi, g in enumerate(groups):
            if not isinstance(g, dict):
                continue
            gname = (g.get("name") or "").strip()
            if not gname:
                continue
            gid = f"{_slug(theme)}|{gi}|{_slug(gname)}"
            store.write_chain_node(
                conn, node_id=gid, theme=theme, node_type="group",
                parent_id=None, seq=gi, name=gname,
                stage=(g.get("stage") or None), summary=(g.get("summary") or None),
                tickers_json=None, keywords_json=None, built_by=built_by,
            )
            n_group += 1
            for si, s in enumerate(g.get("segments", []) or []):
                if not isinstance(s, dict):
                    continue
                sname = (s.get("name") or "").strip()
                if not sname:
                    continue
                sid = f"{gid}|{si}|{_slug(sname)}"
                tickers = s.get("tickers") or []
                keywords = [k for k in (s.get("keywords") or [])
                            if isinstance(k, str) and k.strip()]
                # 环节名本身也并入关键词（保底能匹配），去重保序。
                kw_all: list[str] = []
                seen: set[str] = set()
                for k in [sname, *keywords]:
                    k = k.strip()
                    if k and k.lower() not in seen:
                        seen.add(k.lower())
                        kw_all.append(k)
                store.write_chain_node(
                    conn, node_id=sid, theme=theme, node_type="segment",
                    parent_id=gid, seq=si, name=sname,
                    stage=(g.get("stage") or None),
                    summary=(s.get("summary") or None),
                    tickers_json=json.dumps(tickers, ensure_ascii=False),
                    keywords_json=json.dumps(kw_all, ensure_ascii=False),
                    built_by=built_by,
                )
                new_segs[segnorm.canonical(sname)] = sname
                n_seg += 1
        if n_seg == 0:
            # 一个环节都没写出来说明这次生成是废的——宁可保留旧链也不要空链。
            raise ChainError(
                f"「{theme}」本次生成 0 个环节，已回滚（保留原链，未覆盖）。"
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    diff = {
        "added": [new_segs[k] for k in new_segs.keys() - old_segs.keys()],
        "removed": [old_segs[k] for k in old_segs.keys() - new_segs.keys()],
        "kept": len(new_segs.keys() & old_segs.keys()),
        "had_old": bool(old_segs),
    }
    return {"groups": n_group, "segments": n_seg, "diff": diff}


def build_chain(
    cfg: Config,
    theme: str,
    *,
    conn=None,
    reference: str | None = None,
    use_web: bool = False,
    model_source: str = DEFAULT_MODEL_SOURCE,
) -> dict:
    """构建某主题的产业链并落库（幂等：先删该主题旧链再写）。

    返回 {theme, groups, segments, web, diff}。use_web=True 时先尝试联网校验
    （relay 不支持则自动退化为纯知识）。node_id 由主题+分组+环节 slug 拼成，稳定可复现。
    reference：可选人工参考结构，注入 prompt 供模型核实纠正（prompt 已要求不照搬）。

    单条构建入口：内部 = _generate_chain_data（LLM）+ _persist_chain（事务写库）。
    批量重建请直接用这两个子函数以实现「并发生成 + 串行写」。
    """
    own_conn = conn is None
    if own_conn:
        conn = store.connect(cfg.paths.db)
    try:
        gen = _generate_chain_data(
            cfg, theme, reference=reference, use_web=use_web,
            model_source=model_source,
        )
        res = _persist_chain(conn, theme, gen["groups"], gen["built_by"])
        return {"theme": theme, "groups": res["groups"], "segments": res["segments"],
                "web": gen["built_by"].startswith("web"), "diff": res["diff"]}
    finally:
        if own_conn:
            conn.close()


def _fts_match_for_keywords(keywords: list[str]) -> str | None:
    """把环节关键词列表拼成 FTS5 MATCH 串：每个关键词 jieba 分词后按短语（相邻）匹配，
    多关键词用 OR 连接。返回 None 表示无可用词元。

    与索引侧 segment_for_index 分词对齐（正文按 jieba 分词空格连接入库），
    故关键词也须分词——“光刻胶”“optical module”都能命中正文。各词元加引号防语法注入。
    """
    from .segment import segment_query

    phrases: list[str] = []
    seen: set[str] = set()
    for kw in keywords:
        toks = segment_query(kw)
        if not toks:
            continue
        # 短语匹配：词元空格相连并整体加引号（FTS5 视为相邻短语），双引号转义。
        phrase = " ".join(t.replace('"', '""') for t in toks)
        if phrase and phrase.lower() not in seen:
            seen.add(phrase.lower())
            phrases.append(f'"{phrase}"')
    if not phrases:
        return None
    return " OR ".join(phrases)


def _seg_doc_ids(conn, keywords: list[str]) -> set[str]:
    """某环节关键词命中的文档 id 集合（全文 FTS + 标题/主题兜底）。

    抽成独立函数是为了让"占比统计"与"缓存证据签名"共用同一套命中口径——
    否则两处各写一遍匹配逻辑，早晚漂移成两种含义（缓存以为证据没变、实际变了）。
    """
    kws = [k for k in keywords if k and k.strip()][:MAX_KEYWORDS_MATCH]
    if not kws:
        return set()

    doc_ids: set[str] = set()

    # 1) 全文命中（FTS5）：匹配任一关键词短语的 chunk 所属文档。FTS 不可用则跳过、退兜底。
    match = _fts_match_for_keywords(kws)
    if match:
        try:
            rows = conn.execute(
                "SELECT DISTINCT c.doc_id FROM chunks_fts f "
                "JOIN chunks c ON c.rowid = f.rowid "
                "WHERE chunks_fts MATCH ?",
                (match,),
            ).fetchall()
            doc_ids.update(r[0] for r in rows)
        except Exception:  # noqa: BLE001 - FTS 语法/环境问题不该拖垮整棵树
            pass

    # 2) 标题/主题兜底：细粒度词若正文分词后未成词元，标题/主题仍可能命中。
    title_or = " OR ".join("title LIKE ?" for _ in kws)
    theme_or = " OR ".join("theme LIKE ?" for _ in kws)
    like = [f"%{k}%" for k in kws]
    try:
        rows = conn.execute(
            f"SELECT doc_id FROM documents WHERE {title_or}", like
        ).fetchall()
        doc_ids.update(r[0] for r in rows)
        rows = conn.execute(
            f"SELECT DISTINCT doc_id FROM doc_themes WHERE {theme_or}", like
        ).fetchall()
        doc_ids.update(r[0] for r in rows)
    except Exception:  # noqa: BLE001
        pass
    return doc_ids


def _seg_split(conn, keywords: list[str]) -> dict:
    """某环节的中外研报占比：命中文档去重后按 lang 分中资(zh)/外资(en)。

    早期只用 title/theme LIKE，但 CMP抛光材料/前驱体/引线框架 等细粒度环节词很少
    出现在标题或主题标签里，导致大量环节占比 0/0（漏统计）。改为**优先走 FTS 全文**
    （安集科技的 CMP 内容一定在正文里），再并上 title/theme LIKE 兜底，覆盖大幅提升。
    """
    doc_ids = _seg_doc_ids(conn, keywords)
    if not doc_ids:
        return {"domestic": 0, "foreign": 0, "total": 0,
                "domestic_pct": 0.0, "foreign_pct": 0.0}

    # 按 lang 归中外。doc_ids 可能较多，分批 IN 查询避免 SQL 变量上限。
    dom = for_ = 0
    ids = list(doc_ids)
    for i in range(0, len(ids), 400):
        batch = ids[i:i + 400]
        ph = ",".join("?" for _ in batch)
        rows = conn.execute(
            f"SELECT lang, COUNT(*) c FROM documents "
            f"WHERE doc_id IN ({ph}) GROUP BY lang",
            batch,
        ).fetchall()
        for r in rows:
            if (r["lang"] or "").lower().startswith("en"):
                for_ += r["c"]
            else:
                dom += r["c"]
    total = dom + for_
    return {
        "domestic": dom, "foreign": for_, "total": total,
        "domestic_pct": round(dom / total * 100, 1) if total else 0.0,
        "foreign_pct": round(for_ / total * 100, 1) if total else 0.0,
    }


def evidence_signature(cfg: Config, theme: str, *, conn=None) -> dict:
    """算某条链当前的「证据指纹」，供解读缓存判定是否失效。**纯 SQL，零 LLM 成本。**

    为什么不用递增的 data_revision 计数器：那需要在新增/删除/重新归类/重抽取
    每一个写入点都记得去 bump，漏一处就静默失效——跟它要修的 bug 是同一类。
    这里改为直接对**证据本身**取指纹：把各环节匹配到的 doc_id 全集排序后哈希。
    研报增删、重新归类（主题变→匹配集变）、关键词改动，都会让指纹变；
    没有任何地方需要手工维护。派生数据不该靠人记得同步。

    返回 {docs: 参与证据的文档数, evidence_hash: 16位十六进制}。
    """
    own_conn = conn is None
    if own_conn:
        conn = store.connect(cfg.paths.db)
    try:
        nodes = store.get_chain(conn, theme)
        all_docs: set[str] = set()
        for n in nodes:
            if n["node_type"] != "segment":
                continue
            try:
                kws = json.loads(n["keywords"] or "[]")
            except (ValueError, TypeError):
                kws = []
            all_docs |= _seg_doc_ids(conn, kws)
        digest = hashlib.sha256(
            "|".join(sorted(all_docs)).encode("utf-8")
        ).hexdigest()[:16]
        return {"docs": len(all_docs), "evidence_hash": digest}
    finally:
        if own_conn:
            conn.close()


def get_chain_view(cfg: Config, theme: str, *, conn=None) -> dict:
    """读某主题产业链为 分组→环节 树，每个叶子环节现算中外研报占比。

    返回 {theme, groups:[{...group, segments:[{...seg, tickers, split}]}]}。
    无该主题链时 groups 为空。纯读库 + SQL 计数，零 LLM 成本。
    """
    own_conn = conn is None
    if own_conn:
        conn = store.connect(cfg.paths.db)
    try:
        nodes = store.get_chain(conn, theme)
        groups = [n for n in nodes if n["node_type"] == "group"]
        segs_by_parent: dict[str, list] = {}
        for n in nodes:
            if n["node_type"] == "segment":
                segs_by_parent.setdefault(n["parent_id"], []).append(n)

        out_groups = []
        for g in sorted(groups, key=lambda x: x["seq"]):
            segs = []
            for s in sorted(segs_by_parent.get(g["node_id"], []), key=lambda x: x["seq"]):
                try:
                    kws = json.loads(s["keywords"] or "[]")
                except (ValueError, TypeError):
                    kws = []
                try:
                    tickers = json.loads(s["tickers"] or "[]")
                except (ValueError, TypeError):
                    tickers = []
                segs.append({
                    "node_id": s["node_id"],
                    "name": s["name"],
                    "stage": s["stage"],
                    "summary": s["summary"],
                    "tickers": tickers,
                    "keywords": kws,
                    "split": _seg_split(conn, kws),
                })
            out_groups.append({
                "node_id": g["node_id"],
                "name": g["name"],
                "stage": g["stage"],
                "summary": g["summary"],
                "segments": segs,
            })
        return {"theme": theme, "groups": out_groups}
    finally:
        if own_conn:
            conn.close()


# 解读提示版本号：_ANALYST_SYSTEM / 本函数拼 query 的逻辑一改就 +1，
# 让缓存键随之变化（否则改了提示词却仍命中旧解读，看不到改进效果）。
INTERPRET_PROMPT_VERSION = "chain-interp-v1"


def interpret_chain(cfg: Config, theme: str, *, model_source: str = DEFAULT_MODEL_SOURCE):
    """链路 AI 解读：读已落库的产业链结构 → 组合成一个覆盖全链的深度分析问题 →
    走 generate.analyze（多路检索中外资研报 + _ANALYST_SYSTEM 综合）。

    结构本身零成本读库，只有这一步的综合分析烧 token（故前端做成按钮、结果缓存）。
    把落库的分组/环节/代表标的塞进问题，引导模型顺着这条权威链去判断核心与格局，
    而非从零瞎拆——链是稳定共识，分析才是增量。

    返回 (Answer, evidence_ids)。**evidence_ids 是本次真正参与解读的 chunk_id 排序列表**，
    调用方哈希它作为缓存键的一部分：解读内容源于这批具体研报，只要证据变了（新研报进来、
    重新归类、抽取变化），缓存就必须失效。这比维护一个 data_revision 计数器可靠——
    计数器要在每个写入点记得 bump，漏一处就是静默返回过期解读（正是本次要修的 bug）。
    """
    from . import generate as gen_mod

    view = get_chain_view(cfg, theme)
    groups = view.get("groups") or []
    if not groups:
        raise ChainError(f"主题「{theme}」尚未构建产业链，无法解读。")

    # 把链结构压成一段引导语：列出各分组→环节，供模型顺链分析（不必逐字复述）。
    lines = []
    for g in groups:
        seg_names = "、".join(s["name"] for s in g.get("segments", []))
        lines.append(f"- {g['name']}（{g.get('stage') or ''}）：{seg_names}")
    struct = "\n".join(lines)
    query = (
        f"请深度解读「{theme}」这条产业链。已梳理出的上/中/下游结构如下：\n{struct}\n\n"
        f"请顺着这条链，判断每一环的价值量与壁垒、点出真正的核心标的与理由、"
        f"指出中外资研报关注的差异，并标出研报未充分覆盖、需另行查证的盲区环节。"
    )
    ans, res, _subs = gen_mod.analyze(
        cfg, query, model_source=model_source,
    )
    # 证据指纹用 chunk_id（比 doc_id 更细：同文档换了命中段落也算证据变化）。
    evidence_ids = sorted(h.chunk_id for h in res.hits if h.chunk_id)
    return ans, evidence_ids


# ---- 产业链漂移检测（零 LLM 成本，纯 SQL + segnorm）----------------------------
# 目的：新研报进来后，链结构不会自己长出新环节。这里找出「库里热度够、且现有链
# 未覆盖」的方向，**写进候选表等人审**，绝不自动并入正式链——链是判断核心标的的
# 骨架，让噪声自动写入会把骨架搞脏，下游全脏。
#
# 信号选择与阈值理由：
#  · 用 doc_themes（已抽取的主题标签）作候选来源，而不是从正文挖新词——主题标签
#    已经过一轮模型抽取，噪声远低于裸词频。
#  · **跨机构数是四个信号里最有价值的**：单家券商反复提，多半是它自己的题材包装；
#    多家机构同时提，才更可能是真方向。故 min_insts 默认 3。
#  · 近期性：只看最近 N 天有出现的，避免把早已冷掉的旧题材翻出来。
DRIFT_MIN_DOCS = 8      # 支持文档数下限
DRIFT_MIN_INSTS = 3     # 提及机构数下限（跨机构才可信）
DRIFT_RECENT_DAYS = 60  # 近期窗口：最近出现必须落在这个天数内
DRIFT_MIN_LIFT = 2.5    # 集中度下限（见下）

# **集中度（lift）是把噪声挡住的关键阈值**，第一版没有它，结果不可用：
# 锂电池链报出「出海/光伏/半导体」、创新药链报出「半导体/芯片/算力」——这些不是
# 缺失环节，而是同一篇研报同时覆盖多个行业造成的共现。原因是「原始共现数」会奖励
# 那些跟什么都共现的宽泛标签（出海全库 934 篇，跟谁都搭）。
#   lift = P(候选|本链研报) / P(候选|全库)
# 只有在本链研报里**显著超配**的方向才是这条链自己的东西。出海在锂电池里 65% vs
# 全库 53%，lift 仅 1.2 → 挡掉；真正的细分环节会明显集中。
#
# 另外排除「本身就是一条链的主题」与宏观/风格标签：它们是平级行业或选股风格，
# 不可能是某条链的下级环节。
_DRIFT_STOP = {
    "出海", "国产替代", "高股息", "并购重组", "货币政策", "宏观经济",
    "通胀", "信用债", "利率债", "数据要素", "消费", "银行", "券商",
    # 市场结构/交易风格标签：根本不是产业环节，是"在哪个市场、用什么风格买"。
    # 实测它们会挂在 半导体/煤炭 等链下（ETF/A股策略/大类资产配置/北交所/港股/
    # 行业轮动），lift 能过线纯粹因为策略研报常与某几个行业同批出现。
    "ETF", "A股策略", "大类资产配置", "行业轮动", "北交所", "港股", "A股",
}

# **v1→v2 改名留下的坑（2026-08-12 修）**：下面这些是 rebuild_chains_v2 里
# ORPHANS_TO_DELETE 的旧链名 —— 它们曾是正式链，改名后从 industry_chain 里删了，
# 但 doc_themes 的主题标签仍在大量使用旧词（算力/人工智能/AIDC/有色/化工…）。
# _stop_canon 原本只靠"库内已建链主题名"来排除平级行业，改名后这些词一个都不在
# 链名集合里了，于是平级行业重新涌进候选表：实测 有色×7 条链、算力/机器人/军工/
# 储能/农业 各×6 条链。这份表把旧词补回停用集，与新链名一起生效。
# **新增链或再次改名时，若旧名仍在 doc_themes 里流通，必须往这里补一行。**
_LEGACY_CHAIN_THEMES = {
    "AIDC", "人工智能", "人形机器人", "储能", "军工", "农业", "创新药",
    "化工", "医药", "固态电池", "国产算力", "大模型", "数据要素",
    "新能源", "新能源车", "有色", "机器人", "石油石化", "算力", "芯片",
}
# 比较时用规范形（segnorm.canonical），否则「国产替代」与「国产 替代」等异写漏网。
_DRIFT_STOP_CANON = None   # 延迟构建：见 _stop_canon()


def _stop_canon(conn) -> set[str]:
    """停用词的规范形集合 = 固定宏观/风格标签 + 旧链名 + **库内已建链的主题名**。

    已建链的主题（半导体/光伏/新型储能…）是彼此平级的行业，不可能是另一条链的下级
    环节；不排除的话，锂电池链会把「光伏」「储能」当成自己缺失的环节报出来。
    每次调用现算（链集合会随 build-chain 变化），成本是一次 DISTINCT 查询。

    `_LEGACY_CHAIN_THEMES` 补的是改名后仍在 doc_themes 流通的旧链名 —— 只靠现有链名
    的话，一次重命名就会让这层过滤整体失效（见该常量上方注释）。
    """
    from . import segnorm

    names = set(_DRIFT_STOP) | set(_LEGACY_CHAIN_THEMES)
    try:
        names |= set(store.list_chain_themes(conn))
    except Exception:  # noqa: BLE001 - 取不到就只用固定表，不影响主流程
        pass
    return {segnorm.canonical(n) for n in names if n}


def _is_peer_of_chain(cand_canon: str, chain_canons: set[str]) -> bool:
    """候选名与某条已建链名互为子串 → 判为**平级行业级标签**，不是下级环节。

    机械规则，为的是让改名不再重新打开这个口子（`_LEGACY_CHAIN_THEMES` 是人工补的
    历史包袱，这条是自动生效的）。命中例：`算力`⊂`AI算力数据中心`、`面板`⊂`显示面板`、
    `锂`⊂`锂电池`、`电力`⊂`电力生产运营` —— 这些都是别条链的粒度，落到本链只会
    误导。反例（不会被误杀）：`光模块`/`先进封装`/`CPO`/`硅光` 不是任何链名的子串，
    仍会正常报出，这才是真正想要的漏环节信号。

    ≥2 字才参与子串判定：单字（铜/铝）本身就是链名，靠精确集合命中即可，放开子串
    会把「铜箔」这类真环节按 `铜` 误杀。
    """
    if len(cand_canon) < 2:
        return False
    for c in chain_canons:
        if len(c) < 2:
            continue
        if cand_canon in c or c in cand_canon:
            return True
    return False


def detect_drift(
    cfg: Config,
    theme: str,
    *,
    conn=None,
    min_docs: int = DRIFT_MIN_DOCS,
    min_insts: int = DRIFT_MIN_INSTS,
    min_lift: float = DRIFT_MIN_LIFT,
    recent_days: int = DRIFT_RECENT_DAYS,
    persist: bool = True,
) -> list[dict]:
    """检测某条链的漂移：找出未被现有环节覆盖、但已有足够研报支撑的新方向。

    覆盖判定走 segnorm.covers（环节别名 + 标准化），而不是裸字符串比——否则
    「光模块 / 光通信模块 / optical module」会被报成三个新方向，清单全是噪声。

    persist=True 时把结果 upsert 进 chain_candidate（已被人标 merged/rejected 的
    保留原状态，不会因为再次检出就复活）。返回候选列表（含支撑证据）。
    """
    from . import segnorm

    own_conn = conn is None
    if own_conn:
        conn = store.connect(cfg.paths.db)
    try:
        nodes = store.get_chain(conn, theme)
        if not nodes:
            raise ChainError(f"主题「{theme}」尚未构建产业链，无从比较漂移。")
        existing = [n["name"] for n in nodes if n["node_type"] == "segment"]
        stop_canon = _stop_canon(conn)
        # 平级链名的规范形（供 _is_peer_of_chain 做子串判定），与 stop_canon 分开：
        # stop_canon 是精确命中，这个是"含链名即视为平级粒度"。
        from . import segnorm as _sn

        chain_canons = {
            _sn.canonical(t) for t in store.list_chain_themes(conn) if t
        }

        # 全库文档总数与本链主题的文档数 —— 算 lift（集中度）要用。
        n_all = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] or 1
        n_theme = conn.execute(
            "SELECT COUNT(DISTINCT doc_id) FROM doc_themes WHERE theme LIKE ?",
            (f"%{theme}%",),
        ).fetchone()[0] or 1

        # 候选池：与本链主题共现的其它主题标签（"这条链的研报还在谈什么"）。
        # 除计数外必须算 **lift = 该标签在本链内的出现率 / 在全库的出现率**：
        # 只看共现数会把「出海/国产替代/人工智能」这类几乎和所有主题共现的泛标签
        # 顶到前面（实测锂电池链下 出海102篇/40机构、创新药链下 半导体45篇/32机构，
        # 全是噪声）。lift 高才说明"这个方向特别集中在本链里"，才是真的漏掉的环节。
        rows = conn.execute(
            "SELECT t2.theme AS name, "
            "       COUNT(DISTINCT t2.doc_id) AS docs, "
            "       COUNT(DISTINCT d.institution) AS insts, "
            "       MIN(d.report_date) AS first_seen, "
            "       MAX(d.report_date) AS last_seen, "
            "       (SELECT COUNT(DISTINCT doc_id) FROM doc_themes x "
            "        WHERE x.theme = t2.theme) AS global_docs "
            "FROM doc_themes t1 "
            "JOIN doc_themes t2 ON t1.doc_id = t2.doc_id "
            "JOIN documents d ON d.doc_id = t2.doc_id "
            "WHERE t1.theme LIKE ? AND t2.theme != t1.theme "
            "GROUP BY t2.theme "
            "HAVING docs >= ? AND insts >= ? "
            "ORDER BY insts DESC, docs DESC",
            (f"%{theme}%", min_docs, min_insts),
        ).fetchall()

        # 近期窗口下界。基准取 **min(库内最新 report_date, 真实今天)**：
        #  · 用库内最新而非机器时钟，是为了避免"语料半年没更新、机器时钟往前走"
        #    导致窗口整体扫空（原始意图，保留）。
        #  · 但**必须夹到今天**：report_date 来自人工命名的 PDF 文件名，年份手误必
        #    然会有。实测一篇 `...-270730.pdf`（应为 260730）被解析成 2027-07-30，
        #    于是 cutoff = 2027-05-31，而全库真实最新只到 2026-07-31 → **每个候选都
        #    被判成"已冷掉的旧题材"跳过，60 条链齐刷刷 0 候选**。这个失败还静默：
        #    日志打的是「未检出新漂移候选」，与"链很健康"一模一样。一篇文件名写错
        #    年份就能让整个阶段3 报假阴性，故这里夹住上界。
        newest = conn.execute(
            "SELECT MAX(report_date) FROM documents WHERE report_date IS NOT NULL"
        ).fetchone()[0]
        cutoff = None
        if newest:
            import datetime as _dt

            try:
                base = _dt.date.fromisoformat(newest[:10])
            except ValueError:
                base = None
            if base is not None:
                today = _dt.date.today()
                if base > today:
                    base = today          # 未来日期一律按今天算，不让脏数据放大窗口
                cutoff = (base - _dt.timedelta(days=recent_days)).isoformat()

        out: list[dict] = []
        for r in rows:
            name = r["name"]
            cand_canon = segnorm.canonical(name)
            if cutoff and (r["last_seen"] or "") < cutoff:
                continue                      # 已冷掉的旧题材，不提示
            if cand_canon in stop_canon:
                continue                      # 平级行业/风格因子，不是本链的下级环节
            if _is_peer_of_chain(cand_canon, chain_canons):
                continue                      # 与某条链名互为子串 → 平级粒度，见该函数
            hit = segnorm.covers(existing, name)
            if hit:
                continue                      # 现有环节已覆盖（含同义/复合写法）
            # 集中度（lift）过滤：见上面 DRIFT_MIN_LIFT 的说明。
            # lift = 本链内出现率 / 全库出现率；≈1 表示"到处都有"，纯属陪跑。
            g = r["global_docs"] or 0
            lift = ((r["docs"] / n_theme) / (g / n_all)) if g else 0.0
            if lift < min_lift:
                continue
            # 代表研报：机构分散取样，便于人工快速判断这是真方向还是题材包装。
            samples = conn.execute(
                "SELECT d.doc_id, d.title, d.institution, d.report_date "
                "FROM doc_themes t JOIN documents d ON d.doc_id = t.doc_id "
                "WHERE t.theme = ? ORDER BY d.report_date DESC LIMIT 5",
                (name,),
            ).fetchall()
            cand = {
                "theme": theme,
                "cand_key": segnorm.canonical(name),
                "name": name,
                "doc_count": r["docs"],
                "inst_count": r["insts"],
                "lift": round(lift, 2),
                "first_seen": r["first_seen"],
                "last_seen": r["last_seen"],
                "samples": [dict(s) for s in samples],
            }
            out.append(cand)
            if persist:
                store.upsert_candidate(
                    conn, theme=theme, cand_key=cand["cand_key"], name=name,
                    doc_count=cand["doc_count"], inst_count=cand["inst_count"],
                    first_seen=cand["first_seen"], last_seen=cand["last_seen"],
                    sample_docs_json=json.dumps(cand["samples"], ensure_ascii=False),
                )
        if persist:
            conn.commit()
        return out
    finally:
        if own_conn:
            conn.close()
