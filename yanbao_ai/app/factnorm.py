"""FactNorm：事实归一化与校验（短期优化）。

三块能力，全部**只读纯函数**，不改库、不调 API，供 timeline 查询/展示与体检复用：

1. 金额归一（to_canonical_amount）：把同类金额换算到统一基准，便于跨报告比较/画图。
   - 人民币规模系 → 亿元；美元规模系 → 亿美元（不同币种绝不混算）。
   - 价格类（元/吨、美元/桶）、百分比、倍数一律不归一（跨标的无可比性）。

2. value_num 校验（verify_value_num）：检查 value_num 是否与 value_text 里的主数字自洽，
   用于体检发现脏数据（模型偶发把换算写错/量级填错）。只报告不自动改。

3. 实体别名归一（canonical_entity / entity_aliases）：把同一实体的中英文/简称合并，
   如 台积电 ↔ TSMC。查询层用它做 OR 扩展匹配，零破坏、可回滚。
"""
from __future__ import annotations

import re

# ---- 1. 金额单位归一 ----------------------------------------------------------
# 每个单位相对「1 元 / 1 美元」的倍数。归一时先换算到基础币种单位，再除以基准
# （亿=1e8）得到「亿元 / 亿美元」计的数值。价格/比率类不在此表 → 不归一。
# 只收「带量级前缀」的金额单位——金额规模总是写成 万元/亿元/亿美元 等；
# 裸「元/美元/港元」几乎全是每股价格（目标价/股价/批价），绝不能当规模归一
# （否则 468美元目标价 会被换算成 4.68e-06 亿美元这种废值）。故裸单位一律不收。
_CNY_UNITS = {
    "万元": 1e4, "十万元": 1e5, "百万元": 1e6,
    "千万元": 1e7, "亿元": 1e8, "十亿元": 1e9, "百亿元": 1e10,
    "万亿元": 1e12,
}
_USD_UNITS = {
    "万美元": 1e4, "百万美元": 1e6, "千万美元": 1e7,
    "亿美元": 1e8, "十亿美元": 1e9, "百亿美元": 1e10, "万亿美元": 1e12,
}
_HKD_UNITS = {"万港元": 1e4, "百万港元": 1e6, "亿港元": 1e8}

# 归一基准：各币种统一到「亿」级并给出规范单位名。
_BASE = 1e8
_FAMILIES = [
    (_CNY_UNITS, "亿元"),
    (_USD_UNITS, "亿美元"),
    (_HKD_UNITS, "亿港元"),
]
# 单位 → (相对基础单位倍数, 规范单位名)，供快速查表。
_UNIT_MAP: dict[str, tuple[float, str]] = {}
for _table, _canon in _FAMILIES:
    for _u, _mult in _table.items():
        _UNIT_MAP[_u] = (_mult, _canon)


def to_canonical_amount(
    value_num: float | None, unit: str | None
) -> tuple[float, str] | None:
    """把金额规模换算到统一基准（亿元 / 亿美元 / 亿港元）。

    仅当 unit 是可识别的「纯金额规模」单位时才归一；价格类（元/吨）、
    百分比、倍数、非金额单位一律返回 None（表示「不可比，别归一」）。
    """
    if value_num is None or not unit:
        return None
    hit = _UNIT_MAP.get(unit.strip())
    if not hit:
        return None
    mult, canon = hit
    return value_num * mult / _BASE, canon


def canonical_amount_family(unit: str | None) -> str | None:
    """返回该单位归一后的规范单位名（亿元/亿美元/亿港元），不可归一返回 None。"""
    if not unit:
        return None
    hit = _UNIT_MAP.get(unit.strip())
    return hit[1] if hit else None


# ---- 2. value_num 与 value_text 自洽校验 --------------------------------------
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
# value_text 里常见的数量级词 → 相对倍数（用于把文本数字对齐到 value_num 的口径）。
_SCALE_WORDS = {
    "万亿": 1e12, "千亿": 1e11, "百亿": 1e10, "十亿": 1e9, "亿": 1e8,
    "千万": 1e7, "百万": 1e6, "十万": 1e5, "万": 1e4,
    "bn": 1e9, "mn": 1e6, "k": 1e3,
}


def _all_numbers(text: str) -> list[float]:
    """抽出文本里所有数字。研报 value_text 常带年份前缀（2030年1.67万亿元）或
    区间（由500增至800），真实值可能不是第一个数——全取，交给调用方逐一比对。"""
    out: list[float] = []
    for m in _NUM_RE.finditer(text.replace(",", "")):
        try:
            out.append(float(m.group()))
        except ValueError:
            pass
    return out


def verify_value_num(
    value_text: str | None, value_num: float | None, unit: str | None = None
) -> str | None:
    """校验 value_num 是否与 value_text 自洽。返回 None=正常，否则返回问题描述。

    宽松策略（避免误报）：只在能明确判定不一致时报问题。
    - value_num 为空但 value_text 是纯数字 → "可补" 提示。
    - value_num 与 value_text 里**任一**数字都对不上（含数量级换算）→ "量级可疑"。
      扫所有数字而非仅第一个：value_text 常带年份前缀/区间（2030年1.67万亿元、
      由500增至800），只要有一个数字能与 value_num 自洽（直接或经换算词）即视为正常。
    """
    if value_num is None:
        if value_text and _NUM_RE.fullmatch(value_text.strip().replace(",", "")):
            return "value_num 空但 value_text 是纯数字（可补）"
        return None
    if not value_text:
        return None
    # value_num==0 无法做量级比率校验（0/任何数都是0，永远对不上文本里的年份/对比数）。
    # 且这类 0 绝大多数是真实抽取值（"Q1发行归零"、"从20%下调至0"、"同比持平"→0），
    # 强行比率会把它们全部误报——直接视为正常。
    if value_num == 0:
        return None

    nums = [n for n in _all_numbers(value_text) if n != 0]
    if not nums:
        return None

    # value_num 可能等于 value_text 里某个数字，或是其经数量级换算后的结果
    # （如 CNY255,926mn → 2559.26 亿元）。只要任一数字能对齐（容忍 5% 误差）即自洽。
    for txt_num in nums:
        ratio = value_num / txt_num
        if 0.95 <= abs(ratio) <= 1.05:
            return None
        for _, mult in _SCALE_WORDS.items():
            for base in (mult, 1 / mult, mult / _BASE, _BASE / mult):
                if base and 0.95 <= abs(ratio / base) <= 1.05:
                    return None
        # 差异在 100 倍以内且无换算词解释——多为四舍五入/区间，视为可接受。
        if 0.01 < abs(ratio) < 100:
            return None
    return (f"value_num({value_num}) 与 value_text 所有数字{nums[:5]}"
            f"均量级不符")


# ---- 3. 实体别名归一 ----------------------------------------------------------
# 规范名 → 别名列表。查询时把用户输入映射到规范名，再 OR 匹配全部写法。
# 覆盖全库 TOP 实体里明显的中英文/简称重复（台积电↔TSMC 等）。
_ALIAS_GROUPS: dict[str, list[str]] = {
    "台积电": ["台积电", "TSMC", "tsmc", "台積電"],
    "三星电子": ["三星电子", "三星", "Samsung Electronics", "Samsung", "삼성전자"],
    "地平线": ["地平线", "Horizon Robotics", "地平线机器人"],
    "英伟达": ["英伟达", "NVIDIA", "Nvidia", "nvidia", "英偉達"],
    "阿里巴巴": ["阿里巴巴", "Alibaba", "BABA", "阿里"],
    "腾讯控股": ["腾讯控股", "腾讯", "Tencent"],
    "美光科技": ["美光科技", "美光", "Micron", "Micron Technology"],
    "台达电子": ["台达电子", "台达", "Delta", "Delta Electronics"],
    "联发科": ["联发科", "MediaTek", "聯發科"],
    "布伦特原油": ["布伦特原油", "布伦特", "Brent", "Brent原油", "布兰特"],
    "WTI原油": ["WTI原油", "WTI", "西德州原油", "美国原油"],
    "美联储": ["美联储", "Fed", "FED", "联邦储备"],
    "谷歌": ["谷歌", "Google", "Alphabet"],
    "微软": ["微软", "Microsoft", "MSFT"],
    "苹果": ["苹果", "Apple", "AAPL"],
    "亚马逊": ["亚马逊", "Amazon", "AMZN"],
    "特斯拉": ["特斯拉", "Tesla", "TSLA"],
    "宁德时代": ["宁德时代", "CATL", "宁德"],
    "比亚迪": ["比亚迪", "BYD"],
    "贵州茅台": ["贵州茅台", "茅台", "600519"],
}
# 反向索引：任一别名（小写）→ 规范名。
_ALIAS_INDEX: dict[str, str] = {}
for _canon, _aliases in _ALIAS_GROUPS.items():
    for _a in _aliases:
        _ALIAS_INDEX[_a.lower()] = _canon


def canonical_entity(name: str | None) -> str | None:
    """把实体名映射到规范名；未知返回原值。"""
    if not name:
        return name
    return _ALIAS_INDEX.get(name.strip().lower(), name)


def entity_aliases(name: str | None) -> list[str]:
    """返回该实体的全部已知写法（含自身）；无别名时返回 [原值]。

    供查询层做 OR 扩展：用户搜「TSMC」也能命中库里以「台积电」入库的行。
    """
    if not name:
        return []
    canon = canonical_entity(name)
    return _ALIAS_GROUPS.get(canon, [name])


# ---- 4. as_of_date 合理性校验 -------------------------------------------------
def _parse_iso(s: str | None) -> tuple[int, int] | None:
    """把 ISO 日期/年月（2026-07-02 或 2026-07）解析成 (年, 月)；失败返回 None。"""
    if not s:
        return None
    m = re.match(r"(\d{4})-(\d{1,2})", s.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


# 点位估值型指标：其取值是「报告发布时刻的报价/看法」，as_of 理应≈report_date。
# 这类指标配一个「过去」的 as_of，基本是模型把正文历史行情表的日期误当成了报价日
# （如目标价 as_of=半年前）。其余指标（营收/EPS 等预测指向未来财年、大宗商品价格
# 指向历史序列点）as_of 天然偏离 report_date，不算错——故只对这一窄类做校验。
_SPOT_METRICS = {
    "目标价", "现价", "股价", "收盘价", "目标价/现价", "目标价格",
}


def verify_as_of_date(
    as_of_date: str | None, report_date: str | None,
    metric: str | None = None, *, max_months: int = 3
) -> str | None:
    """校验点位估值型指标的 as_of_date 是否合理。返回 None=正常，否则问题描述。

    **只校验点位估值指标（目标价/现价/股价/收盘价）**：这类值是报告发布时刻的报价，
    as_of 理应≈report_date；若配了个「过去」的 as_of，基本是模型把正文历史行情表的
    日期误当成了报价日（你最初发现的 #19837 就是此类）。

    其余指标一律返回 None（不校验）：营收/EPS 等财务预测的 as_of 指向未来财年末、
    大宗商品价格的 as_of 指向历史序列点，都是正常语义，机械比对会造出海量误报。
    未来方向的偏离也不报（目标价本就指向未来，个别把目标年月填进 as_of 属可接受）。

    只报告不改库——timeline 已改用 report_date 为主排序规避，本校验仅供体检发现脏 as_of。
    """
    if not metric or metric.strip() not in _SPOT_METRICS:
        return None
    a = _parse_iso(as_of_date)
    r = _parse_iso(report_date)
    if a is None or r is None:
        return None
    diff = (a[0] - r[0]) * 12 + (a[1] - r[1])   # as_of 相对 report 的月数差
    # 只报「过去」方向：点位报价配历史日 = 误填；配未来日属个别可接受，不报。
    if diff < -max_months:
        return (f"{metric} 的 as_of_date({as_of_date}) 早于报告日({report_date}) "
                f"{-diff} 个月（点位报价疑误系历史行情日）")
    return None
