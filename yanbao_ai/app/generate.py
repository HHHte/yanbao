"""Generate：把检索命中的块交给 Claude，产出带引用标注的答案。

流程（方案 §6.5）：
1. 取 retrieve 的 Hit 列表，按融合分序拼成带编号的上下文（[1][2]…）。
2. 系统提示强约束：只依据给定材料作答，每个论断标注来源编号，材料不足要明说。
3. 走 Claude（中转站端点，端点/密钥来自 config→settings.json 跟随）。
4. 返回答案文本 + 实际引用到的来源清单（doc 元数据 + chunk_id），便于溯源。

端点鉴权：cc-switch 写的是 ANTHROPIC_AUTH_TOKEN（Bearer），故用 SDK 的 auth_token；
绝不打印 token 值。无 key/SDK 不可用时抛 GenerateError，由 CLI 友好提示。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .config import Config
from .retrieve import Hit, Filters, RetrieveResult, retrieve

MAX_CONTEXT_CHARS = 24000   # 拼进 prompt 的上下文字符上限（防超长）
DEFAULT_MODEL_SOURCE = "gen"  # gen=model_gen（强），cheap=model_cheap（省）

_SYSTEM = """你是研报分析助手。严格依据【材料】回答用户问题，遵守：
1. 只用材料中的事实作答，不要引入材料外的知识或臆测。
2. 每个关键论断后用方括号标注来源编号，如 [1]、[2][3]。
3. 若材料不足以回答，明确说明"材料未提供相关信息"，不要编造。
4. 涉及数字、日期、机构观点时，务必对齐材料原文，注明出处编号。
5. 用中文回答，简洁、结构化。"""

# 时间线解读专用系统提示：把 SQL 捞到的、按字段拆散的 facts 交给模型重新读懂，
# 产出一段连贯叙述。关键约束是分清「真时间序列」与「单篇多年预测横截面」——
# 前者才是随时间演变（如目标价逐周调整），后者只是一篇报告在同一发布日给出的多年预测
# （FY25A/FY26E/2027E），不应被当成时间演变。这正是字段直读会「越搞越乱」的根源。
_TL_SYSTEM = """你是研报时间线分析助手。下面给你的是从研报库里、针对某实体某指标
机械抽取出来的一批事实条目（含发布日、数值、单位、原文引用）。这些条目是按字段拆散的，
可能把「同一篇报告里的多年预测」和「跨报告的真实时间演变」混在一起，还可能有单位重复、
指标口径不一、财年标签（FY25A/FY26E/3Q26）当日期等噪声。

请你读懂这批材料后，产出一段**连贯、去重、可信**的解读，遵守：
1. 先区分两类信息，分别成段：
   （A）真时间序列：同一口径指标在不同发布日之间的演变（如目标价从某周到某周的调整）。
       按发布日先后说明趋势（升/降/持平）与关键拐点。
   （B）单篇预测横截面：某篇报告在同一发布日给出的多年预测（FY25A/26E/27E 等），
       它不是"随时间变化"，要standalone列出，并注明是哪家、哪个发布日给的预测。
2. 只用材料中的数字，严禁自己换算或编造；财年标签保持原样，不要臆测成公历日期。
3. 数值后标注来源编号 [n]。单位以原文为准，不同货币/口径绝不相加或直接比较。
4. 若材料本身就是一堆异质、无法构成序列的碎片，就如实说明"这些条目口径不一，
   无法构成单一时间序列"，并把能归类的部分归好类。
5. 用中文，简洁。不要复述全部条目，抓演变与要点。"""

# 产业链分析师提示：区别于 _SYSTEM（严格只用材料、防幻觉，适合事实问答），本提示
# 用于"梳理产业链/判断核心标的"这类需要模型带行业理解去读多篇研报的场景。核心设计是
# **两类信息强制分离标注**——凡有研报出处的用 [n]；凡是模型自己的行业判断/推理用【判断】
# 前缀，让用户一眼分清"研报说的"和"模型推的"，既不自缚（能排序、能判核心/边缘），也不
# 掩盖推理来源（可验证）。这直接回应了旧 ask 模式"只会堆票、不敢排序"的问题。
#
# 【关键修正·防误杀】旧版把公司分成"核心/蹭概念"两桶，导致模型把"我没检索到证据"
# 悄悄等同于"这是蹭概念"，把有强逻辑但本轮材料没覆盖的标的（如超节点系统层的华丰科技、
# 浪潮信息）错杀成边缘。新版强制**三桶分类**并禁止这个塌缩：核心 / 真边缘（必须能说出
# 主业无关才能归此）/ 研报未覆盖·存疑（有证据缺失就归此，绝不打成蹭概念），且要求模型
# 在回答里**主动摊开自己预期该有、但材料未覆盖的环节**，把盲区暴露给用户而非默默吞掉。
_ANALYST_SYSTEM = """你是资深产业研究分析师。用户会给你一个行业/主题，并附上从多篇研报里
检索出的材料（每条带编号 [n] 和出处）。你的任务不是复述研报，而是像分析师一样**理解
整条产业链**，梳理出真正的核心与脉络。遵守：

1. 先搭框架：把该主题拆成上中下游（或按环节/子板块），说清每一环在做什么、彼此如何衔接、
   价值量与壁垒在哪。这部分可以、且应当运用你自己的产业知识，不必句句有研报出处。
2. 判断时用**三桶分类，不许只分核心/蹭概念两桶**——这是硬性要求，专为防止"误杀"：
   - 【核心】产业链卡脖子/高壁垒环节与公司（不可替代、格局好、价值量高），给出理由
     （技术壁垒、国产化率、竞争格局、客户绑定等），不要只列名字。
   - 【真边缘】只有当你能明确说出"其主业与本产业链无关（如环保、集成灶、柴油机等只是
     沾数据中心/机房边）"时，才可把某标的判为蹭概念。必须给出"主业无关"的具体理由。
   - 【未覆盖·存疑】凡是你凭行业知识认为该属于这条链、但**本次材料里没有证据支撑**的
     公司或环节，一律归入此桶，明确写"研报未覆盖，无法定级"。
     **严禁把"没检索到"当成"蹭概念"**——证据缺失只说明材料没覆盖，不代表它不重要。
3. **主动摊开盲区（硬性要求）**：在回答中专门用一段列出"我凭产业知识预期这条链应当包含、
   但本次材料未覆盖的环节/方向"，提示用户这些是需要补充检索或另行查证的地方。宁可多提醒，
   不可默默漏掉——用户不该靠自己预先知道名字来补你的检索缺口。
4. **两类信息必须分开标注**：研报材料支持的事实/数据/公司点名，句末标 [n]；你基于行业
   理解的推理/排序/判断，用【判断】开头。
5. 不要瞎编：不臆造研报里没有的具体数字、目标价、市占率。行业常识判断可讲，但具体量化
   必须有 [n] 或明说"这是估计"。
6. 落到可用：结论要能指导"该重点看哪几家、为什么"，给出主次分档，而非一张不分主次的名单。
7. **兼顾中外视角**：材料里可能既有中资券商研报、也有高盛/大摩/摩根/UBS 等外资研报，
   它们常对同一家 A 股或全球公司给出不同判断（外资更看重全球供应链地位、估值、周期，
   中资更看重国产替代弹性与政策）。若两方对同一标的看法有分歧或互补，请点出来并注明出处 [n]，
   这种交叉印证比单一口径更可信；不要只挑一种口径而忽略另一种。
8. 用中文作答，但可保留材料中的英文公司名/术语原文；结构清晰（上中下游小标题 + 每环标的与理由 + 盲区提示段）。
9. **排版要好读（用 Markdown）**：多用二级/三级标题分段，长清单用有序/无序列表而非一大段文字；
   关键结论、公司名、数字用 **加粗**。适度在小标题前加一个契合语义的 emoji 作视觉锚点，降低阅读疲劳，
   例如 🏭 上游 / 🔧 中游 / 📦 下游 / 🎯 核心标的 / ⚠️ 盲区存疑 / 🌐 中外视角 / 💡 判断。
   emoji 只用于小标题点缀，宁少勿滥，正文里不要堆砌；股票代码照常写在括号里（如 中际旭创(300308)）。"""

# 子问题拆解提示（多路检索用）：把一个宽泛的产业链问题拆成若干"子环节检索词"，让每个
# 子环节各检索一次，规避"单次 top-K 向量只覆盖到部分环节、整条支线被漏掉"的召回缺口
# （如问"国产算力"，单次检索漏掉了超节点/连接器/系统集成整条线）。只输出检索词，不作答。
_DECOMPOSE_SYSTEM = """你是检索策略助手。用户会给一个宽泛的行业/产业链问题。请你凭产业
知识，把它拆成一批**互补的子环节检索词**，覆盖这条产业链上中下游的各个关键环节
（含容易被忽略的支线，如系统层/互联/封装/材料/设备/零部件等），确保后续按每个子环节
分别检索时不会整条支线漏掉。要求：
- 每行一个检索词，尽量用"行业+环节"的具体短语（如"国产算力 超节点 互联 连接器"、
  "半导体设备 量测检测 国产化"），便于命中研报。
- **中英双语**：语料里约 1/3 是高盛/大摩/摩根/UBS/Bernstein/Nomura 等外资的英文研报，
  它们也覆盖 A 股与全球公司。中文检索词命中不了英文正文，故请在中文子环节之外，**额外
  给 3-5 个英文检索词**，用英文行业术语表达同样的关键环节（如 "domestic substitution
  semiconductor equipment"、"China AI compute self-sufficiency GPU"、"advanced packaging
  CoWoS localization"），让英文研报也能被检索到。
- 只输出检索词列表，每行一个，不要编号、不要解释、不要其他任何文字。
- 共 9-15 行（其中含 3-5 行英文）。"""


class GenerateError(RuntimeError):
    pass


@dataclass
class Source:
    ref: int              # 引用编号（从 1 起）
    chunk_id: str
    doc_id: str
    title: str | None
    institution: str | None
    report_date: str | None
    heading_path: str | None


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    model: str = ""
    used_hits: int = 0


def _build_context(hits: list[Hit], max_chars: int = MAX_CONTEXT_CHARS) -> tuple[str, list[Source]]:
    """把 Hit 拼成带编号材料块，返回 (context_text, sources)。超长按字符预算截断。

    max_chars：字符预算。事实问答用默认（24000）；产业链分析多路检索命中多，需更大预算
    才不把多样支线截掉（analyze 传更大值），配合轮转交错合并，让各支线都能进材料。
    """
    blocks: list[str] = []
    sources: list[Source] = []
    used = 0
    ref = 0
    for h in hits:
        ref += 1
        head = " > ".join(
            x for x in [h.institution, h.title, h.report_date, h.heading_path] if x
        )
        body = h.text.strip()
        block = f"[{ref}] 出处：{head or h.doc_id}\n{body}\n"
        if used + len(block) > max_chars and blocks:
            ref -= 1
            break
        blocks.append(block)
        used += len(block)
        sources.append(
            Source(
                ref=ref,
                chunk_id=h.chunk_id,
                doc_id=h.doc_id,
                title=h.title,
                institution=h.institution,
                report_date=h.report_date,
                heading_path=h.heading_path,
            )
        )
    return "\n".join(blocks), sources


def _client(cfg: Config):
    if not cfg.llm.base_url or not cfg.llm.api_key:
        raise GenerateError(
            "未解析到 Claude 端点/密钥。请在 config.toml 的 [llm] 配置，"
            "或确保 ~/.claude/settings.json 的 env.ANTHROPIC_BASE_URL / "
            "ANTHROPIC_AUTH_TOKEN 存在（cc-switch 管理）。"
        )
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise GenerateError(f"未安装 anthropic SDK：{exc}") from exc
    # 鉴权用 x-api-key（api_key=），**不要用 auth_token=（Bearer）**：实测
    # `Bearer 裸` 会 ReadTimeout 挂住（比 403 更难查），x-api-key 稳定 200。
    #
    # **user-agent 必须覆盖**（2026-08-28 逐头二分实测，端点 sub.100xlabs.space）：
    # 在已知 200 的请求上逐个加 SDK 特征头，只有 UA 一项会翻成 403：
    #   基线（x-api-key + x-app + stream）        → 200
    #   + 全部 12 个 x-stainless-*                → 200
    #   + accept-encoding: gzip,deflate,br,zstd   → 200
    #   + user-agent: Anthropic/Python 0.120.0    → 403 upstream_error  ★判别器
    # 即上游按 UA 拉黑 anthropic Python SDK，与鉴权方式/x-app/x-stainless 都无关。
    # SDK 默认 UA 会被拒 → 显式覆盖成 CLI UA，否则本系统一个请求都发不出去。
    # （用户 2026-08-28 明确要求就用这个端点跑通；端点与 key 均为用户自有。）
    #
    # 排查提示：若某天又出现全量 403，先用这套头做二分——先确认端点本身可用
    # （手写 httpx 带 stream=true 打一次），再逐个加 SDK 特征头找判别器。
    # 不要因为 Claude Code 自己能对话就断定本系统能调通：两者 UA 不同，走的门不同。
    return Anthropic(
        base_url=cfg.llm.base_url,
        api_key=cfg.llm.api_key,
        # **键名大小写必须与 SDK 内部一致（"User-Agent"，首字母大写）**：SDK 的
        # _build_headers 是普通 dict 合并（区分大小写），内部先放 "User-Agent"，
        # 若这里写小写 "user-agent" 就成了两个不同的键，httpx.Headers 会把重复头
        # 用 ", " 拼接 → 实际发出 `Anthropic/Python 0.120.0, claude-cli/...`，
        # 仍含被拉黑的子串，照样 403。用 "User-Agent" 才是真正覆盖。
        default_headers={
            "User-Agent": "claude-cli/2.1.219 (external, cli)",
            "x-app": "cli",
        },
    )


# 明确不值得重试的异常类型：编程错误。重发同一请求只会把同一个 bug 再跑一遍。
# **AssertionError 故意不在此列** —— 它来自 SDK 流层对空响应的 assert，是瞬时故障。
PERMANENT_EXC_TYPES = (TypeError, AttributeError, NameError, KeyError, ImportError)

# 值得重发的 HTTP 状态：限流/超时/冲突/5xx。4xx 其余（400 参数、401 鉴权、403 被拒、
# 404 模型名错）是请求本身的问题，重试无用。
TRANSIENT_STATUS = (408, 409, 429, 500, 502, 503, 504, 529)


def is_transient_error(exc: Exception) -> bool:
    """通用重试判定：**按异常类型判，绝不按错误字符串判**。

    调用方（facts / chain）在此之上再叠各自的模块异常：先问本模块的专属类型，
    剩下的交给这里。抽出来共用是因为两边都踩过同一个坑，逻辑不该各写一份。

    坑的由来（2026-08-12，静默丢了 1605 篇 facts）：原判定是
    `"429" in str(exc) or "timeout" in str(exc).lower() or ...` 的关键词匹配。
    中转站返回"HTTP 200 但空 SSE 流"时，SDK 在 `get_final_message()` 里
    `assert self.__final_message_snapshot is not None` 抛**裸 AssertionError，
    str(exc) 恰好是空字符串** —— 空串匹配不到任何关键词 → 判成永久错误 →
    一次都不重试直接放弃，而且失败是静默的（错误行长得像 "facts: <doc_id>: "）。

    宁可多重发几次，也不要再把瞬时抖动误判成永久失败。
    """
    try:
        from anthropic import APIStatusError
    except ImportError:
        APIStatusError = None
    if APIStatusError is not None and isinstance(exc, APIStatusError):
        return getattr(exc, "status_code", None) in TRANSIENT_STATUS
    if isinstance(exc, PERMANENT_EXC_TYPES):
        return False
    return True        # 其余（网络层/SDK 流层/未知）一律当瞬时，给它重试的机会


def create_message(client, **kwargs):
    """发一次请求，返回 Message 对象——内部走流式，但对调用方等价于 messages.create。

    **为什么必须流式**：中转站会掐掉"长时间零字节"的连接。非流式请求在模型生成
    期间，连接上一个字节都不流动，实测 182s 后被 relay 直接断开：
        httpx.RemoteProtocolError: Server disconnected without sending a response
    （SDK 包成 APIConnectionError，本层再包成 GenerateError，于是前端只看到
    "Claude 调用失败：Connection error"）。同一 prompt 走流式 93s 正常返回，
    因为 SSE 增量事件让连接上持续有字节，不会被判定为空闲。
    输出越长越容易触发，所以恰恰是产业链解读这类长输出必挂、短问答又看着正常。

    这里收完流后用 get_final_message() 还原成完整 Message，故调用方原有的
    resp.content 解析（含 tool_use 分支）完全不用改。
    """
    with client.messages.stream(**kwargs) as s:
        return s.get_final_message()


def generate(
    cfg: Config,
    query: str,
    result: RetrieveResult,
    *,
    model_source: str = DEFAULT_MODEL_SOURCE,
    max_tokens: int = 2000,
    system: str = _SYSTEM,
    history: list[dict] | None = None,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> Answer:
    """基于检索结果生成带引用的答案。materials 为空时直接返回"无材料"提示。

    system：系统提示，默认事实问答（_SYSTEM）；产业链分析走 _ANALYST_SYSTEM。
    history：既往对话轮次 [{"role":"user"/"assistant","content":str}, ...]，用于追问。
      新一轮的【材料】+【问题】作为最后一条 user 消息追加在 history 之后。
    max_chars：材料字符预算，透传给 _build_context；产业链分析多路命中多，需更大值。
    """
    if not result.hits:
        return Answer(text="材料未提供相关信息（检索无命中）。", used_hits=0)

    context, sources = _build_context(result.hits, max_chars=max_chars)
    model = cfg.llm.model_cheap if model_source == "cheap" else cfg.llm.model_gen

    client = _client(cfg)
    user_msg = f"【材料】\n{context}\n\n【问题】\n{query}"
    # 追问：把既往轮次拼在前面，本轮材料+问题作为最后一条 user 消息。
    messages = _clean_history(history) + [{"role": "user", "content": user_msg}]
    try:
        resp = create_message(
            client,
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
    except Exception as exc:  # noqa: BLE001 - 网络/端点各类异常统一转友好错误
        raise GenerateError(f"Claude 调用失败：{exc}") from exc

    text = "".join(
        b.text for b in resp.content if getattr(b, "type", None) == "text"
    )
    return Answer(text=text, sources=sources, model=model, used_hits=len(sources))


def _clean_history(history: list[dict] | None) -> list[dict]:
    """净化前端传来的对话历史：只保留 role∈{user,assistant} 且 content 非空的轮次，
    交替顺序由前端保证，这里只做类型/字段裁剪，防注入非法结构给 API。"""
    if not history:
        return []
    out: list[dict] = []
    for m in history:
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            out.append({"role": role, "content": content})
    return out


def ask(
    cfg: Config,
    query: str,
    *,
    filters: Filters | None = None,
    limit: int = 12,
    mode: str = "auto",
    model_source: str = DEFAULT_MODEL_SOURCE,
    conn=None,
    system: str = _SYSTEM,
    history: list[dict] | None = None,
) -> tuple[Answer, RetrieveResult]:
    """检索 + 生成一体入口。返回 (Answer, RetrieveResult)。

    system 传 _ANALYST_SYSTEM 即切到产业链分析模式；history 传既往轮次即支持追问。
    """
    result = retrieve(cfg, query, filters=filters, limit=limit, mode=mode, conn=conn)
    answer = generate(cfg, query, result, model_source=model_source,
                      system=system, history=history)
    return answer, result


def _decompose(cfg: Config, query: str, *, model_source: str = DEFAULT_MODEL_SOURCE) -> list[str]:
    """把宽泛产业链问题拆成子环节检索词列表（多路检索用）。

    失败/空时回退为 [query]。始终把原问题也纳入（保底覆盖），去重保序，上限 16 条
    （原问题 + 最多 15 子环节，其中含 3-5 条英文，让外资英文研报也能被检索到）。
    这是 B 修复的第一步：让检索按环节铺开，不漏支线，也不漏英文语料。
    """
    model = cfg.llm.model_cheap if model_source == "cheap" else cfg.llm.model_gen
    try:
        client = _client(cfg)
        resp = create_message(
            client,
            model=model,
            max_tokens=500,
            system=_DECOMPOSE_SYSTEM,
            messages=[{"role": "user", "content": query}],
        )
        text = "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        )
    except Exception:  # noqa: BLE001 - 拆解失败不该拖垮整个分析，退回单路检索
        return [query]

    subs = [ln.strip(" -·•\t　") for ln in text.splitlines()]
    subs = [s for s in subs if s]
    seen: set[str] = set()
    out: list[str] = []
    for s in [query, *subs]:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:16]


def analyze(
    cfg: Config,
    query: str,
    *,
    filters: Filters | None = None,
    per_sub_limit: int = 10,
    total_cap: int = 72,
    mode: str = "auto",
    model_source: str = DEFAULT_MODEL_SOURCE,
    history: list[dict] | None = None,
    max_tokens: int = 3000,
) -> tuple[Answer, RetrieveResult, list[str]]:
    """产业链深度分析（B 修复）：多路检索 → 合并去重 → _ANALYST_SYSTEM 综合。

    流程：先 _decompose 把问题拆成子环节检索词，每个子环节各检索一次（并发），按 chunk_id
    去重后**轮转交错**合并（保证每条支线都能进材料，不被单一子查询独占预算），再交给
    分析师提示综合。这样规避了"单次 top-K 向量只覆盖部分环节、整条支线被漏掉"的召回缺口
    （如问"国产算力"漏掉超节点/连接器/系统集成整条线）。返回 (Answer, RetrieveResult, 子查询列表)。
    """
    from concurrent.futures import ThreadPoolExecutor

    subqueries = _decompose(cfg, query, model_source=model_source)

    def _one(sub: str) -> RetrieveResult:
        try:
            return retrieve(cfg, sub, filters=filters, limit=per_sub_limit, mode=mode)
        except Exception:  # noqa: BLE001 - 单路失败不影响其余路，返回空结果
            return RetrieveResult()

    with ThreadPoolExecutor(max_workers=min(8, len(subqueries))) as ex:
        results = list(ex.map(_one, subqueries))

    # 轮转交错合并：第 0 名各取一个、第 1 名各取一个…按 chunk_id 去重，直到 total_cap。
    # 交错而非拼接，是为了让每条支线的高分命中都尽早进入材料，不被某个子查询刷屏。
    lists = [r.hits for r in results]
    any_dense = any(r.dense_ok for r in results)
    merged: list[Hit] = []
    seen_chunk: set[str] = set()
    i = 0
    while len(merged) < total_cap:
        progressed = False
        for hits in lists:
            if i < len(hits):
                progressed = True
                h = hits[i]
                if h.chunk_id not in seen_chunk:
                    seen_chunk.add(h.chunk_id)
                    merged.append(h)
                    if len(merged) >= total_cap:
                        break
        if not progressed:
            break
        i += 1

    result = RetrieveResult(
        hits=merged,
        mode="hybrid" if any_dense else "lexical",
        dense_ok=any_dense,
    )
    # 分析模式材料多（可达 total_cap=72 块），字符预算放大到约 8.2 万，否则轮转交错
    # 铺开的多样支线（含中英双语命中）又会被 24000 的默认预算截掉，多路检索的覆盖优势白费。
    answer = generate(cfg, query, result, model_source=model_source,
                      system=_ANALYST_SYSTEM, history=history, max_tokens=max_tokens,
                      max_chars=82000)
    return answer, result, subqueries


def _tl_material(points) -> str:
    """把 MetricPoint 列表拼成带编号材料，供时间线解读。含发布日/抽取日/值/单位/引用。

    与 _build_context 不同：这里每个"来源"是一条 fact 而非一个 chunk，携带结构化字段，
    让模型能自己判断哪些是同一发布日的多年预测、哪些是跨日演变。
    """
    lines: list[str] = []
    used = 0
    for i, p in enumerate(points, 1):
        parts = [f"[{i}]"]
        src = " · ".join(x for x in [p.institution, p.title] if x)
        if src:
            parts.append(f"出处：{src}")
        if p.report_date:
            parts.append(f"发布日：{p.report_date}")
        # as_of_date 与发布日不同则标出——让模型知道这是模型抽的、可能是财年标签或历史日。
        if p.as_of_date and p.as_of_date != p.report_date:
            parts.append(f"标注期：{p.as_of_date}")
        if p.metric:
            parts.append(f"指标：{p.metric}")
        val = ""
        if p.value_text:
            val = p.value_text
        elif p.value_num is not None:
            val = str(p.value_num)
        if val:
            parts.append(f"值：{val}{p.unit or ''}")
        if p.direction:
            parts.append(f"方向：{p.direction}")
        if p.quote:
            parts.append(f"原文：{p.quote}")
        block = "　".join(parts)
        if used + len(block) > MAX_CONTEXT_CHARS and lines:
            break
        lines.append(block)
        used += len(block)
    # 返回 (材料文本, 实际计入条数)——超长截断时条数会小于 len(points)，
    # 让调用方据实报告"基于 N 条事实"，不夸大。
    return "\n".join(lines), len(lines)


def interpret_timeline(
    cfg: Config,
    entity: str | None,
    metric: str | None,
    points,
    *,
    model_source: str = DEFAULT_MODEL_SOURCE,
    max_tokens: int = 1500,
) -> Answer:
    """让 Claude 读懂一批（已由 SQL 捞出的）时间线 facts，产出连贯解读。

    这是"混合"方案的 AI 层：SQL 负责零成本、确定性地按 entity+metric 索引出候选事实，
    本函数按需（用户点"AI 解读"按钮时）调一次 LLM 把碎片重新读成连贯叙述，
    分清真时间序列与单篇多年预测横截面，去重、纠口径。仅在被显式调用时才产生费用。
    """
    if not points:
        return Answer(text="没有可解读的数据点。", used_hits=0)

    material, used = _tl_material(points)
    model = cfg.llm.model_cheap if model_source == "cheap" else cfg.llm.model_gen
    head = f"实体：{entity or '*'}"
    if metric:
        head += f"　指标：{metric}"
    user_msg = f"【查询】{head}\n\n【材料】\n{material}"

    client = _client(cfg)
    try:
        resp = create_message(
            client,
            model=model,
            max_tokens=max_tokens,
            system=_TL_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as exc:  # noqa: BLE001 - 网络/端点各类异常统一转友好错误
        raise GenerateError(f"Claude 调用失败：{exc}") from exc

    text = "".join(
        b.text for b in resp.content if getattr(b, "type", None) == "text"
    )
    return Answer(text=text, model=model, used_hits=used)


# 主题趋势解读系统提示：把「主题→按月/周文档计数」的热度桶交给模型，读成一段
# 关于"近期研报在关注什么、热度怎么变"的趋势解读。数据只是文档计数（谁被研报提及多），
# 不含涨跌基本面，故严禁把"研报提及多"说成"基本面好/该买"——只谈关注度与结构变化。
_THEME_TREND_SYSTEM = """你是研报趋势观察助手。下面给你的是研报库按主题聚合的热度数据
（每行：主题 → 被研报打标的文档数，或某主题按时间桶的文档数）。这是"研报关注度"信号，
不是基本面或行情信号。请据此产出一段简洁的趋势解读，遵守：
1. 指出当前研报最集中关注的主题有哪些、大致梯队（第一梯队/其次/长尾）。
2. 若给的是某主题的时间序列，说明其关注度随时间的升降趋势与拐点。
3. 可结合你的行业常识解释"为什么近期研报扎堆关注某主题"，但要用【判断】前缀标出这是你的
   推断，与数据本身（文档计数）区分开。
4. **严禁**把"研报提及多"等同于"基本面好""该买入"——热度只反映卖方关注度，可能是风口，
   也可能是拥挤。若某主题热度很高，提示用户注意这既是关注也可能是拥挤。
5. 不编造数据里没有的具体数字。用中文，简洁，给结构化要点。"""


def interpret_theme_trend(
    cfg: Config,
    buckets,
    *,
    theme: str | None = None,
    by: str = "month",
    model_source: str = DEFAULT_MODEL_SOURCE,
    max_tokens: int = 1200,
) -> Answer:
    """让 Claude 读主题热度桶（文档计数），产出"近期研报关注趋势"解读。

    buckets 为 [(bucket_label, count)] 或带 .bucket/.count 的对象列表。theme 为空时
    表示全库主题总榜（buckets=各主题→计数）；非空时表示某主题按时间桶的热度曲线。
    只读关注度信号，不含基本面——提示词已强约束不得把"提及多"当"该买"。
    """
    rows = []
    for b in buckets:
        if hasattr(b, "bucket"):
            rows.append((b.bucket, b.count))
        else:
            rows.append((b[0], b[1]))
    if not rows:
        return Answer(text="没有可解读的主题热度数据。", used_hits=0)

    if theme:
        head = f"主题「{theme}」按{'周' if by == 'week' else '月'}的研报关注度："
        body = "\n".join(f"{lbl}：{cnt} 篇" for lbl, cnt in rows)
    else:
        head = "全库主题关注度总榜（主题 → 被研报打标文档数）："
        body = "\n".join(f"{lbl}：{cnt} 篇" for lbl, cnt in rows)
    user_msg = f"【数据】{head}\n{body}"

    model = cfg.llm.model_cheap if model_source == "cheap" else cfg.llm.model_gen
    client = _client(cfg)
    try:
        resp = create_message(
            client,
            model=model,
            max_tokens=max_tokens,
            system=_THEME_TREND_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as exc:  # noqa: BLE001 - 统一转友好错误
        raise GenerateError(f"Claude 调用失败：{exc}") from exc

    text = "".join(
        b.text for b in resp.content if getattr(b, "type", None) == "text"
    )
    return Answer(text=text, model=model, used_hits=len(rows))
