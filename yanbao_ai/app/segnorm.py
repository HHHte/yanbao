"""SegNorm：产业链「环节名」的标准化与别名展开（只读纯函数，不改库、不调 API）。

为什么需要它：漂移检测要回答"这个新出现的方向，现有链里有没有？"。若直接拿原始
字符串比，同义/异写会造出大量误报，检出的东西没法看。实测库内 439 个去重环节名里，
295 个带复合或异写形式：

  · 分隔符复合   AI服务器/整机、EDA/IP、BMS / EMS、结构件/箔材
  · 空格不一致   「AI 服务器/整机」vs「AI服务器/整机」
  · 括号补充说明 CXO（CRO/CDMO）vs CXO（CRO/CDMO/CMO）、ADC（抗体偶联药物）
  · 中英混写     AI芯片/GPU、光模块 vs optical module

所以本模块做两件事：
1. normalize(name) —— 折叠掉不影响语义的差异（全角→半角、去空格、统一分隔符、
   大小写），得到可比较的标准形。
2. expand(name)   —— 把复合名炸开成各个原子概念（含括号内的补充写法），再对每个
   原子查同义词表。这样「AI服务器/整机」与「AI 服务器」能对上，
   「光模块」与「optical module」也能对上。

覆盖判定 covers(existing_names, candidate) 即：候选的任一原子，命中已有环节的任一
原子，就算已覆盖。宁可判"已覆盖"（漏报），也不要判"新方向"（误报）——漂移检测的
产出是给人审的清单，清单里全是噪声就等于没有。
"""
from __future__ import annotations

import re

# 复合环节名的分隔符：斜杠(半/全角)、顿号、加号、「与」「及」「和」。
# 注意不拆连字符和点号——「CO-packaged」「1.6T」拆了会碎成无意义片段。
_SPLIT_RE = re.compile(r"[/／、+＋]|与|及(?!物)|和(?![成弦])")

# 括号（中英全半角）内的补充说明：既是整体的一部分，也常是别名本身
# （CXO（CRO/CDMO）里括号内就是它的具体形态），故括号内外都取出来当原子。
_PAREN_RE = re.compile(r"[（(]([^）)]*)[）)]")

# 归一时丢弃的装饰性词尾/词头：它们不改变环节所指，却让字符串比对失配。
_NOISE = (
    "产业链", "环节", "板块", "领域", "方向", "相关", "厂商", "企业",
    "制造", "生产", "行业",
)

# 环节级同义词组：canonical → 全部写法。**只收真同义**（同一环节的不同叫法），
# 不收上下位关系（「半导体」与「芯片」不是同义，合并会把两个环节糊成一个）。
# 中英并收，因为库里 35% 是英文研报，外资用 optical module / power semiconductor。
_ALIASES: dict[str, list[str]] = {
    "光模块": ["光模块", "光通信模块", "optical module", "optical transceiver",
              "光收发模块", "opticalmodules"],
    "cpo": ["cpo", "共封装光学", "co-packaged optics", "光电共封装"],
    "ai服务器": ["ai服务器", "人工智能服务器", "ai server", "整机", "服务器整机"],
    # 注意不要收裸「热管理」：它跨域撞车——数据中心的液冷 与 电池包的 BMS/PACK热管理
    # 是两个完全不同的环节，实测「BMS/PACK及热管理」被误并进液冷。故只收明确指向
    # 液冷/温控的写法，"热管理"留给各自领域按上下文自行区分。
    "液冷": ["液冷", "液体冷却", "液冷散热", "liquid cooling", "散热", "温控",
            "液冷温控"],
    "算力芯片": ["算力芯片", "ai芯片", "ai算力芯片", "gpu", "ai accelerator",
               "加速卡", "ai chip"],
    "存储芯片": ["存储芯片", "存储器", "hbm", "dram", "nand", "memory",
               "高带宽内存"],
    "半导体设备": ["半导体设备", "晶圆制造设备", "前道设备", "semiconductor equipment",
                 "wafer fab equipment"],
    "半导体材料": ["半导体材料", "电子材料", "semiconductor material"],
    "cmp抛光材料": ["cmp抛光材料", "cmp材料", "抛光液", "抛光垫", "cmp"],
    "光刻胶": ["光刻胶", "photoresist", "光阻"],
    "电子特气": ["电子特气", "电子特种气体", "特种气体", "electronic specialty gas"],
    "湿电子化学品": ["湿电子化学品", "湿化学品", "wet chemicals"],
    "eda": ["eda", "eda工具", "ip", "eda/ip", "设计工具", "electronic design automation"],
    "先进封装": ["先进封装", "advanced packaging", "chiplet", "2.5d", "3d封装"],
    "晶圆代工": ["晶圆代工", "代工", "foundry", "晶圆制造"],
    "封测": ["封测", "封装测试", "osat", "assembly and test"],
    "pcb": ["pcb", "印制电路板", "覆铜板", "ccl", "printed circuit board"],
    "连接器": ["连接器", "高速连接器", "connector", "高速铜连接"],
    "交换机": ["交换机", "网络交换机", "switch", "以太网交换机", "数据中心交换机"],
    "正极材料": ["正极材料", "正极", "cathode", "cathode material"],
    "负极材料": ["负极材料", "负极", "anode", "anode material"],
    "隔膜": ["隔膜", "separator", "锂电隔膜"],
    "电解液": ["电解液", "electrolyte", "电解质"],
    "固态电池": ["固态电池", "全固态电池", "半固态电池", "solid state battery"],
    "储能": ["储能", "储能系统", "energy storage", "ess", "bess"],
    "逆变器": ["逆变器", "inverter", "光伏逆变器"],
    "硅片": ["硅片", "wafer", "单晶硅片", "silicon wafer"],
    "谐波减速器": ["谐波减速器", "谐波减速机", "harmonic reducer", "减速器"],
    "丝杠": ["丝杠", "滚珠丝杠", "行星滚柱丝杠", "screw", "ball screw"],
    "灵巧手": ["灵巧手", "机械手", "dexterous hand", "末端执行器"],
    "cxo": ["cxo", "cro", "cdmo", "cmo", "医药外包"],
    "创新药": ["创新药", "innovative drug", "新药"],
    "adc": ["adc", "抗体偶联药物", "抗体药物偶联物", "antibody drug conjugate"],
    "大模型": ["大模型", "基础模型", "llm", "large language model", "foundation model"],
    "云计算": ["云计算", "算力租赁", "cloud", "idc", "云服务"],
    "供电": ["供电", "电源", "电源管理", "power supply", "供配电", "hvdc"],
}

def normalize(name: str | None) -> str:
    """把环节名折叠成可比较的标准形：全角→半角、去空格/标点、小写、去装饰词。

    只折叠不影响语义的差异。返回空串表示归一后无有效内容（调用方应跳过）。
    """
    if not name:
        return ""
    s = str(name)
    # 全角字母数字 → 半角（Ａ→A）；全角空格 → 半角。
    s = "".join(
        chr(ord(ch) - 0xFEE0) if "！" <= ch <= "～" else (" " if ch == "　" else ch)
        for ch in s
    )
    s = s.lower()
    # 去掉所有空白与常见标点（保留中日韩、拉丁字母、数字）。
    s = re.sub(r"[\s\-_·.,，。:：'\"“”‘’]", "", s)
    for w in _NOISE:
        s = s.replace(w, "")
    return s.strip()


# 反向索引：任一写法的**归一形** → canonical。必须在 normalize 定义之后构建，
# 且键要过一遍 normalize——否则表里写的 "optical module"（含空格）与查询时的
# 归一形 "opticalmodule" 对不上，英文别名会全部失配（实测踩到过）。
_INDEX: dict[str, str] = {}
for _canon, _forms in _ALIASES.items():
    for _f in _forms:
        _k = normalize(_f)
        if _k:
            _INDEX[_k] = _canon


def expand(name: str | None) -> set[str]:
    """把（可能复合的）环节名炸成原子概念集合，每个原子再映射到 canonical。

    「AI服务器/整机」→ {ai服务器}（两个原子都指向同一 canonical，故合并）
    「CXO（CRO/CDMO）」→ {cxo}（括号内外都是 cxo 系）
    「结构件/箔材」→ {结构件, 箔材}（无同义词表条目，保留各自归一形）
    """
    if not name:
        return set()
    raw = str(name)
    parts: list[str] = []
    # 括号内的补充说明单独取出（它常是别名/具体形态），同时把括号从主串移除。
    for inner in _PAREN_RE.findall(raw):
        parts.extend(_SPLIT_RE.split(inner))
    main = _PAREN_RE.sub(" ", raw)
    parts.extend(_SPLIT_RE.split(main))

    out: set[str] = set()
    for p in parts:
        n = normalize(p)
        if not n or len(n) < 2:      # 单字原子（如拆出来的「与」残留）无区分度，丢弃
            continue
        out.add(_INDEX.get(n, n))
    return out


def canonical(name: str | None) -> str:
    """环节名的规范形：能查到同义词组则返回 canonical，否则返回归一形。"""
    n = normalize(name)
    return _INDEX.get(n, n)


def covers(existing: list[str], candidate: str) -> str | None:
    """已有环节名列表是否已覆盖候选方向。覆盖则返回命中的那个已有名，否则 None。

    判定：候选的任一原子命中已有环节的任一原子即算覆盖。**故意偏宽松**——
    漂移检测的产出是给人过目的清单，误报（把已有的报成新方向）比漏报更伤，
    因为清单一旦充满噪声就没人看了。
    """
    cand = expand(candidate)
    if not cand:
        return None
    for name in existing:
        if expand(name) & cand:
            return name
    return None
