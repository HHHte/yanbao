"""Facts：对每篇文档用 Claude 结构化抽取事实行 + 主题标签，写入 facts/doc_themes。

流程（方案 §6.6，时间线骨架）：
1. 读文档 full.md（规范化后），连同元数据（机构/日期/类别）交给 Claude。
2. Claude 按固定 JSON schema 输出：facts[]（entity/metric/value_num/value_text/
   unit/direction/as_of_date/quote）+ themes[]（受控词表，可增长）。
3. 幂等落库：先 delete_doc_facts 再写；extraction_log 记录 model/schema 版本，
   支持增量续跑（跳过已抽取）与升级重抽（schema_ver 递增 → 定向重跑）。

设计要点：
- 走 Claude 中转站（端点/密钥跟随 config→settings.json，同 generate）。
- 用 model_cheap（抽取批量友好、省），可 --strong 切 model_gen。
- 长文按字符预算截断（抽取关注要点，非全文复述）。
- 受控主题词表 + 受控指标词表：作提示锚点，Claude 可在此之外新增但优先复用。
- JSON 稳健解析：容忍 ```json 代码块包裹、前后噪声；解析失败记错误不中断整批。
- 绝不打印 token 值。
"""
from __future__ import annotations

import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from . import store, normalize
# 所有 Claude 请求统一走 create_message（内部流式），避免长输出被中转站掐断连接。
from .generate import create_message, is_transient_error

SCHEMA_VERSION = 1
MAX_DOC_CHARS = 30000   # 送入抽取的正文字符上限
DEFAULT_CONCURRENCY = 5  # 并发抽取线程数（API 调用可并发；DB 写仍在主线程串行）
# 并发取值历史：16 会稳定触发 429 `Concurrency limit exceeded`（2026-07-26 挂掉 971/1429 篇）；
# 8 在端点健康时干净、0 失败；2026-08-28 中转站严重拥堵（16-token 短请求实测 137.9s，
# 正常 2-3s）时 8 路 worker 全卡在 SDK 的 read=600s 超时里等响应，43 分钟零落库 —— 并发高
# 只是让更多线程一起排队，救不了上游慢。故默认降到 5：拥堵时排队更浅、单篇卡死的连带面更小。

# 进度输出用的锁：worker 线程与主线程都会打印，不加锁会交错成乱行。
_PRINT_LOCK = threading.Lock()


def _log(msg: str) -> None:
    """带时间戳的进度行，**必须 flush**。

    2026-08-28 的教训：build_facts 原先从头到尾一行不打，run_update 也是三步跑完才
    输出汇总。于是「上游慢」与「彻底卡死」在外面看起来完全一样 —— 用户跑 update_all
    卡在阶段2 四十多分钟，只能去 poll SQLite 计数和 TCP 连接才判断得出发生了什么。
    后台重定向到文件时 stdout 默认全缓冲，不 flush 就更是什么都看不到。
    """
    with _PRINT_LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# 受控主题词表（首批锚点，可增长）——提示 Claude 优先复用这些标准标签。
CONTROLLED_THEMES = [
    "AIDC", "算力", "人工智能", "大模型", "机器人", "人形机器人",
    "固态电池", "锂电池", "储能", "光伏", "新能源车", "半导体", "芯片",
    "白酒", "食品饮料", "医药", "创新药", "消费", "地产", "银行", "券商",
    "军工", "有色", "煤炭", "石油石化", "化工", "钢铁", "农业",
    "出海", "国产替代", "并购重组", "高股息", "低空经济", "数据要素",
]

# 受控指标词表（锚点）——常见可量化/可追踪的 metric。
CONTROLLED_METRICS = [
    "目标价", "评级", "批价", "出货量", "销量", "营收", "净利润", "毛利率",
    "市占率", "产能", "产能利用率", "库存", "MAU", "DAU", "ARPU",
    "EPS", "PE", "PB", "ROE", "股息率", "同比增速", "环比增速",
]

_SYSTEM = """你是研报结构化抽取引擎。从给定研报正文中抽取可追溯的结构化信息，
通过 emit_facts 工具输出。

规则：
1. 只抽取正文中**明确出现**的事实，绝不臆造或推断未写明的数字。
2. quote 必须是正文原文片段（≤120字），用于溯源核对。
3. value_num 仅在能明确解析为单一数字时填，否则留空（区间/定性只填 value_text）。
4. metric/theme 优先复用给定受控词表中的标准词；正文有而词表无的可新增。
5. 无可抽取事实时 facts 为空数组；themes 至少给出文档主题（可从标题/行业判断）。"""

# 工具化结构输出：SDK 保证 tool_use.input 是 schema 合规的 dict，
# 彻底消除手工解析 JSON 的整类失败（模型常在 quote 里写裸引号/裸控制符，
# 文本解析必崩；tool-use 则由 API 侧保证良构）。已实测 k40 中转站支持。
_TOOL = {
    "name": "emit_facts",
    "description": "输出从研报正文抽取的结构化事实与主题标签。",
    "input_schema": {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "description": "抽取到的结构化事实行（无则空数组）",
                "items": {
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string", "description": "实体名（公司/行业/主题，如 贵州茅台 / AIDC）"},
                        "entity_code": {"type": "string", "description": "股票代码或留空（如 600519）"},
                        "metric": {"type": "string", "description": "指标名（优先复用受控词表）"},
                        "value_num": {"type": ["number", "null"], "description": "可解析为单一数字时填，否则 null"},
                        "value_text": {"type": "string", "description": "原始值文本（如 1580元 / 买入 / +15%）"},
                        "unit": {"type": "string", "description": "单位或留空（元/亿元/%/万吨…）"},
                        "direction": {"type": "string", "description": "up / down / flat 或留空"},
                        "as_of_date": {"type": "string", "description": "该数据本身的日期 ISO，或留空（可能≠报告日期）"},
                        "quote": {"type": "string", "description": "支撑该事实的原文片段（≤120字）"},
                    },
                    "required": ["entity", "metric", "value_text", "quote"],
                },
            },
            "themes": {
                "type": "array",
                "description": "主题标签（优先复用受控词表，可新增）",
                "items": {"type": "string"},
            },
        },
        "required": ["facts", "themes"],
    },
}


class FactsError(RuntimeError):
    """本模块对外的统一异常（配置缺失、端点不可用、抽取失败等）。"""


class FactsToolCallError(FactsError):
    """响应里没有 emit_facts 的 tool_use 块 —— **瞬时**故障，值得重试。

    继承 FactsError，故调用方原有的 `except FactsError` 不受影响。

    为什么单列一类（2026-08-28 实测纠错）：`tool_choice` 是**强制**调用 emit_facts，
    模型没有"不调"的选择权，所以拿不到 tool_use 块不是模型的语义选择，而是响应在
    传输途中残了。我一度把它判为永久错误（注释还写着"重试大概率还是这样"），据此
    全量跑中这类失败一次都没重试 —— 实测推翻：拿失败的 3 篇原样重发，5/6 次直接成功，
    stop_reason 全是 'tool_use'、输出 token 3.4k-5.4k 远未打满 4000 上限，
    唯一的失败是 RemoteProtocolError（连接被掐），**不是截断、也不是模型不肯调工具**。
    响应结构是 ['thinking', 'tool_use']，流若在 thinking 阶段被中转站掐断，就只剩
    thinking 没有 tool_use，恰好长成"模型未按预期调用工具"的样子。故：判瞬时、要重试。
    """


@dataclass
class FactsStats:
    docs_total: int = 0
    docs_done: int = 0
    docs_skipped: int = 0      # 已抽取（增量跳过）
    docs_failed: int = 0
    facts_written: int = 0
    themes_written: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"文档：{self.docs_done}/{self.docs_total} 抽取"
            f"（跳过已抽 {self.docs_skipped}，失败 {self.docs_failed}）",
            f"事实：写入 {self.facts_written} 条；主题：{self.themes_written} 个",
        ]
        if self.errors:
            lines.append(f"错误 {len(self.errors)} 条（前 5）：{self.errors[:5]}")
        return "\n".join(lines)


def _client(cfg: Config):
    """构造 Anthropic 客户端 —— **直接复用 generate._client，不再自己拼**。

    这里曾有一份独立拷贝（`Anthropic(base_url=..., auth_token=...)`，Bearer 且
    不覆盖 User-Agent）。2026-08-28 修 UA 判别器时只改了 generate 那份，facts /
    chain 各自的拷贝没跟上 → 短请求和 tool-use 直测都 200，一跑 build_facts 却
    全篇 403，排查绕了一圈。三份拷贝＝三套鉴权姿势，端点侧一变就得改三处、且
    漏改是静默的（表现为"某个功能全挂"而非报错）。

    故统一为单一实现：端点/鉴权/UA 的口径只在 generate._client 里维护一次。
    """
    from .generate import _client as _gen_client, GenerateError

    try:
        return _gen_client(cfg)
    except GenerateError as exc:
        # 对外仍抛 FactsError，保持本模块调用方的异常契约不变。
        raise FactsError(str(exc)) from exc


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_json(text: str) -> dict:
    """从模型输出稳健解析 JSON：容忍代码块包裹、前后噪声、字符串内裸控制符。

    strict=False：Claude 的 quote 字段常含多行原文（裸换行/制表符），默认 strict
    的 json.loads 会以 "Invalid control character" 拒绝——放宽后这些控制符被接受。
    """
    t = text.strip()
    # 1) 直接解析
    try:
        return json.loads(t, strict=False)
    except json.JSONDecodeError:
        pass
    # 2) ```json ... ``` 代码块
    m = _JSON_BLOCK.search(t)
    if m:
        try:
            return json.loads(m.group(1), strict=False)
        except json.JSONDecodeError:
            pass
    # 3) 截取首个 { 到末个 }
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j != -1 and j > i:
        return json.loads(t[i : j + 1], strict=False)
    raise json.JSONDecodeError("无法解析 JSON", t, 0)


def _to_num(v) -> float | None:
    """把值稳健转 float；无法转返回 None。"""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.replace(",", "").strip()
        m = re.search(r"-?\d+\.?\d*", s)
        if m:
            try:
                return float(m.group())
            except ValueError:
                return None
    return None


def _build_prompt(title, institution, report_date, category, body: str) -> str:
    meta = " / ".join(
        x for x in [institution, title, report_date, category] if x
    )
    return (
        f"【受控主题词表】{', '.join(CONTROLLED_THEMES)}\n"
        f"【受控指标词表】{', '.join(CONTROLLED_METRICS)}\n\n"
        f"【报告元信息】{meta}\n\n"
        f"【报告正文】\n{body}"
    )


def _is_transient(exc: Exception) -> bool:
    """判断异常是否值得重试 —— 本模块专属异常在前，其余交给通用判定。

    通用部分（HTTP 状态码 / 网络层 / SDK 流层 / 编程错误）见
    `generate.is_transient_error`，那里记着"按类型判而非按字符串判"的由来。
    """
    # **顺序要紧**：FactsToolCallError 是 FactsError 的子类，必须先判子类再判父类，
    # 否则被下面那条 `isinstance(exc, FactsError) → False` 抢先拦成"不重试"。
    if isinstance(exc, FactsToolCallError):
        return True    # 响应缺 tool_use 块 = 流被掐断，实测重发多半就成（见该类注释）
    if isinstance(exc, FactsError):
        return False   # 其余自己抛的（端点缺失/配置错等）重试无用
    return is_transient_error(exc)


def _extract_one(
    client, model: str, meta: dict, body: str, max_tokens: int,
    *, max_retries: int = 5,
) -> dict:
    """调 Claude 抽取一篇，经 emit_facts 工具返回结构化 dict（facts/themes）。

    用 tool-use 而非文本 JSON：研报 quote 常含裸引号/裸控制符（如 给予"增持"评级），
    手工 json.loads 对这类必崩（实测 5/5 全挂在 quote 内引号上）。tool_choice 强制
    调用 emit_facts，SDK 侧保证 tool_use.input 是 schema 合规 dict，彻底消除解析失败。

    中转站有并发上限：超了返回 429 `Concurrency limit exceeded`（瞬时，非永久错误）。
    对 429/5xx/网络错误做指数退避重试（带抖动），避免近上限时误判整篇失败。
    """
    prompt = _build_prompt(
        meta.get("title"), meta.get("institution"),
        meta.get("report_date"), meta.get("category"), body,
    )
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = create_message(
                client,
                model=model,
                max_tokens=max_tokens,
                system=_SYSTEM,
                tools=[_TOOL],
                tool_choice={"type": "tool", "name": "emit_facts"},
                messages=[{"role": "user", "content": prompt}],
            )
            # tool_choice 强制调用 emit_facts；SDK 保证 tool_use.input 是 schema 合规 dict。
            for b in resp.content:
                if getattr(b, "type", None) == "tool_use" and b.name == "emit_facts":
                    return b.input
            # 只剩 thinking 块而没有 tool_use，多为流被掐断（见 FactsToolCallError 注释）。
            # 带上实际拿到的块类型与 stop_reason，便于日后区分"传输残了"与真的模型行为异常。
            kinds = [getattr(b, "type", "?") for b in resp.content]
            raise FactsToolCallError(
                f"响应无 emit_facts tool_use 块（blocks={kinds}, "
                f"stop_reason={getattr(resp, 'stop_reason', None)!r}）"
            )
        except Exception as exc:  # noqa: BLE001 - 需按错误类别决定重试/放弃
            if not _is_transient(exc) or attempt == max_retries - 1:
                raise
            last_err = exc
            # 指数退避 + 抖动：2/4/8/16s，摊开并发重试峰值
            time.sleep(min(2 ** (attempt + 1), 16) + random.uniform(0, 1))
    raise last_err or FactsError("抽取失败")


def _extract_worker(client, model: str, row: dict, max_tokens: int) -> dict:
    """并发 worker（只做线程安全的慢活：读文件+normalize+调 Claude+解析）。

    返回 {doc_id, meta, facts, themes}；失败抛异常由主线程记录。**不碰 SQLite**——
    sqlite3 连接非线程安全，落库统一回主线程串行执行。anthropic 客户端底层 httpx
    连接池并发安全，多线程共享同一 client 即可。
    """
    raw = Path(row["md_path"]).read_text(encoding="utf-8", errors="replace")
    body, _ = normalize.normalize_text(raw)
    if len(body) > MAX_DOC_CHARS:
        body = body[:MAX_DOC_CHARS]
    meta = {
        "title": row["title"],
        "institution": row["institution"],
        "report_date": row["report_date"],
        "category": row["category"],
    }
    parsed = _extract_one(client, model, meta, body, max_tokens)
    return {
        "doc_id": row["doc_id"],
        "report_date": row["report_date"],
        "facts": parsed.get("facts", []) or [],
        "themes": parsed.get("themes", []) or [],
    }


def build_facts(
    cfg: Config,
    conn=None,
    *,
    strong: bool = False,
    only_doc_id: str | None = None,
    reextract: bool = False,
    limit: int | None = None,
    max_tokens: int = 4000,
    concurrency: int = DEFAULT_CONCURRENCY,
    progress: bool = True,
) -> FactsStats:
    """全量（或单篇/限量）抽取事实与主题。

    reextract=False（默认）→ 增量：跳过 extraction_log 里当前 schema 版本已抽的文档。
    reextract=True → 强制重抽（先清后写）。
    limit → 只处理前 N 篇（试跑用）。
    """
    st = FactsStats()
    own_conn = conn is None
    if own_conn:
        conn = store.connect(cfg.paths.db)

    model = cfg.llm.model_gen if strong else cfg.llm.model_cheap
    client = _client(cfg)  # 无端点/SDK 直接抛，早失败

    # 待抽取文档
    if only_doc_id:
        rows = conn.execute(
            "SELECT doc_id, title, institution, report_date, category, md_path "
            "FROM documents WHERE doc_id=?", (only_doc_id,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT doc_id, title, institution, report_date, category, md_path "
            "FROM documents ORDER BY report_date DESC"
        ).fetchall()

    done_ids = set() if reextract else store.extracted_doc_ids(conn, SCHEMA_VERSION)

    todo = [r for r in rows if r["doc_id"] not in done_ids]
    st.docs_skipped = len(rows) - len(todo)
    if limit is not None:
        todo = todo[:limit]
    st.docs_total = len(todo)

    if progress:
        _log(f"facts 抽取开始：待抽 {st.docs_total} 篇（已跳过 {st.docs_skipped} 篇已抽）"
             f"，并发 {concurrency}，模型 {model}")
    if st.docs_total == 0:
        if progress:
            _log("没有待抽文档，直接结束。")
        if own_conn:
            conn.close()
        return st

    t_start = time.time()
    # 心跳线程：**卡住时恰恰没有任何文档完成**，只靠"每篇打一行"依然是一片死寂，
    # 与跑得慢无法区分。故独立线程按固定间隔汇报「已完成数 + 距上次落库多久」，
    # 让"上游卡住"在日志里长成可见的样子（连续几条 heartbeat 而 done 不涨）。
    _hb_stop = threading.Event()
    _last_done_at = [time.time()]

    def _heartbeat() -> None:
        while not _hb_stop.wait(60):
            idle = time.time() - _last_done_at[0]
            msg = (f"心跳 · 已完成 {st.docs_done}/{st.docs_total}"
                   f"，距上次落库 {idle:.0f}s")
            if idle > 300:
                msg += "  ← 已超 5 分钟无进展，上游可能拥堵/卡住（可 Ctrl+C 中断，已落库的不会丢）"
            _log(msg)

    hb_thread: threading.Thread | None = None
    if progress:
        hb_thread = threading.Thread(target=_heartbeat, daemon=True)
        hb_thread.start()

    # 并发抽取：worker 只做慢的、线程安全的部分（调 Claude+解析），落库回主线程串行。
    # 线程池并发发请求，把几十小时的串行压到几小时。SQLite 连接非线程安全，写入必须
    # 在主线程逐条完成。
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(_extract_worker, client, model, dict(r), max_tokens): r
                for r in todo
            }
            for fut in as_completed(futures):
                r = futures[fut]
                doc_id = r["doc_id"]
                try:
                    res = fut.result()
                except json.JSONDecodeError as exc:
                    st.docs_failed += 1
                    st.errors.append(f"{doc_id}: JSON 解析失败 {exc}")
                    _last_done_at[0] = time.time()
                    if progress:
                        _log(f"  ✗ [{st.docs_done + st.docs_failed}/{st.docs_total}] "
                             f"{doc_id[:8]} JSON 解析失败")
                    continue
                except Exception as exc:  # noqa: BLE001 - 单篇失败不中断整批
                    st.docs_failed += 1
                    # 带上异常类型名：有些异常 str() 是空的（SDK 流层的裸 AssertionError
                    # 就是），只打 {exc} 会得到 "facts: <doc_id>: " 这种无信息错误行。
                    st.errors.append(f"{doc_id}: {type(exc).__name__}: {exc}")
                    _last_done_at[0] = time.time()
                    if progress:
                        # 失败也要即时可见：全量跑时"错误攒到最后一次性打印"等于跑完才知道，
                        # 而端点异常往往是连片的，早看见才能早中断、别白烧钱。
                        _log(f"  ✗ [{st.docs_done + st.docs_failed}/{st.docs_total}] "
                             f"{doc_id[:8]} {type(exc).__name__}: {str(exc)[:80]}")
                    continue

                # 落库（主线程串行）：幂等先清后写
                store.delete_doc_facts(conn, doc_id)
                n_fact = 0
                for f in res["facts"]:
                    if not isinstance(f, dict):
                        continue
                    store.write_fact(
                        conn,
                        doc_id=doc_id,
                        report_date=res["report_date"],
                        entity=(f.get("entity") or None),
                        entity_code=(f.get("entity_code") or None),
                        metric=(f.get("metric") or None),
                        value_num=_to_num(f.get("value_num")),
                        value_text=(f.get("value_text") or None),
                        unit=(f.get("unit") or None),
                        direction=(f.get("direction") or None),
                        as_of_date=(f.get("as_of_date") or None),
                        quote=(f.get("quote") or None),
                    )
                    n_fact += 1

                n_theme = 0
                for th in res["themes"]:
                    if isinstance(th, str) and th.strip():
                        store.write_theme(conn, doc_id, th.strip())
                        n_theme += 1

                store.log_extraction(
                    conn, doc_id, model, SCHEMA_VERSION, n_fact, n_theme
                )
                conn.commit()
                st.facts_written += n_fact
                st.themes_written += n_theme
                st.docs_done += 1
                _last_done_at[0] = time.time()
                if progress:
                    n_seen = st.docs_done + st.docs_failed
                    elapsed = time.time() - t_start
                    # 速率按"已出结果篇数/已耗时"算（含失败篇），再据此推剩余时间。
                    # 上游快慢会飘，ETA 只是量级参考，用于判断"要不要等"而非精确排期。
                    rate = n_seen / elapsed if elapsed > 0 else 0
                    left = st.docs_total - n_seen
                    eta = f"{left / rate / 60:.0f}min" if rate > 0 else "?"
                    _log(f"  ✓ [{n_seen}/{st.docs_total}] {doc_id[:8]} "
                         f"事实 {n_fact} · 主题 {n_theme} · "
                         f"累计 {st.facts_written} 事实 · "
                         f"{rate * 60:.1f} 篇/min · 剩 {left} 篇约 {eta}")
    finally:
        # 先停心跳再关连接：否则心跳线程可能在收尾后继续打日志，看着像还在跑。
        _hb_stop.set()
        if hb_thread is not None:
            hb_thread.join(timeout=2)
        if progress:
            _log(f"facts 抽取结束：成功 {st.docs_done} / 失败 {st.docs_failed} "
                 f"/ 共 {st.docs_total} 篇，耗时 {time.time() - t_start:.0f}s，"
                 f"写入 {st.facts_written} 事实、{st.themes_written} 主题")
        if own_conn:
            conn.close()

    return st
