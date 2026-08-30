"""Timeline：基于 facts / doc_themes 的时间线查询与聚合（方案 §6.6）。

两类查询：
1. 指标时间线：某实体某指标随时间的取值序列（如 贵州茅台 · 批价）。
   SELECT as_of_date/report_date, value_num/value_text FROM facts
   WHERE entity LIKE ? [AND metric=?] ORDER BY 日期。
2. 主题热度：某主题按周/月聚合的文档计数（如 AIDC 周度热度曲线）。
   JOIN documents 取 week/month，按时间桶 COUNT。

只读查询，不调任何 API。日期优先用 as_of_date（数据本身日期），缺失退 report_date。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from . import store, factnorm


@dataclass
class MetricPoint:
    date: str | None          # 有效日期：优先 report_date（发布日，确定可信），退 as_of_date
    value_num: float | None
    value_text: str | None
    unit: str | None
    direction: str | None
    entity: str | None
    metric: str | None
    institution: str | None
    title: str | None
    doc_id: str
    quote: str | None
    # report_date / as_of_date 分开保留：前端可对比标注、verify 可校验二者是否背离。
    # as_of_date 是模型抽的（噪声大，常把历史行情日/未来目标年月误填），report_date 是确定的。
    report_date: str | None = None
    as_of_date: str | None = None
    # 金额归一（仅纯金额规模单位有值）：把 百万元/万亿元 等统一到 亿元/亿美元/亿港元，
    # 便于跨报告同尺度比较与画图；价格/比率/倍数类不归一 → 保持 None（见 factnorm）。
    norm_num: float | None = None
    norm_unit: str | None = None


@dataclass
class MetricSeries:
    entity: str | None = None
    metric: str | None = None
    points: list[MetricPoint] = field(default_factory=list)

    def summary(self) -> str:
        head = f"指标时间线：{self.entity or '*'}"
        if self.metric:
            head += f" · {self.metric}"
        head += f"（{len(self.points)} 个数据点）"
        return head


@dataclass
class ThemeBucket:
    bucket: str        # 月份/周次标签
    count: int


def metric_timeline(
    cfg: Config,
    entity: str | None,
    metric: str | None = None,
    *,
    conn=None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 200,
) -> MetricSeries:
    """查某实体某指标的时间线，按有效日期升序。

    entity 用 LIKE 子串匹配（'茅台' 命中 '贵州茅台'）；metric 精确匹配（给定时）。
    有效日期 = COALESCE(report_date, as_of_date)，用于排序与过滤。

    **优先 report_date（报告发布日）而非 as_of_date**：实测 as_of_date 被模型滥用——
    它常把正文里出现的任意日期填进去（历史行情表的旧日期、目标价的未来目标年月），
    如某 260702 发布的报告 as_of 填成 4-20（正文引用的历史行情日）、NT$3100 目标价
    as_of 填成 2027-06（12个月目标年月）。这类噪声会把点甩到时间轴的错误位置。
    report_date 来自文件名/manifest，确定可信，故作为时间轴主键。
    """
    own_conn = conn is None
    if own_conn:
        conn = store.connect(cfg.paths.db)

    try:
        clauses = []
        params: list = []
        if entity:
            # 别名 OR 扩展：搜「TSMC」也命中以「台积电」入库的行（见 factnorm）。
            # 每个写法用 LIKE 子串匹配，多写法用 OR 连接为一个子句组。
            aliases = factnorm.entity_aliases(entity)
            like_terms = " OR ".join("f.entity LIKE ?" for _ in aliases)
            clauses.append(f"({like_terms})")
            params.extend(f"%{a}%" for a in aliases)
        if metric:
            clauses.append("f.metric = ?")
            params.append(metric)
        # 有效日期表达式（复用于过滤与排序）：优先 report_date（可信），退 as_of_date。
        eff = "COALESCE(f.report_date, f.as_of_date)"
        if date_from:
            clauses.append(f"{eff} >= ?")
            params.append(date_from)
        if date_to:
            clauses.append(f"{eff} <= ?")
            params.append(date_to)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        sql = (
            f"SELECT {eff} AS eff_date, f.report_date, f.as_of_date, "
            f"f.value_num, f.value_text, f.unit, "
            f"f.direction, f.entity, f.metric, f.quote, f.doc_id, "
            f"d.institution, d.title "
            f"FROM facts f JOIN documents d ON d.doc_id = f.doc_id"
            f"{where} "
            f"ORDER BY eff_date IS NULL, eff_date ASC LIMIT ?"
        )
        rows = conn.execute(sql, [*params, limit]).fetchall()
        pts = []
        for r in rows:
            norm = factnorm.to_canonical_amount(r["value_num"], r["unit"])
            pts.append(MetricPoint(
                date=r["eff_date"],
                value_num=r["value_num"],
                value_text=r["value_text"],
                unit=r["unit"],
                direction=r["direction"],
                entity=r["entity"],
                metric=r["metric"],
                institution=r["institution"],
                title=r["title"],
                doc_id=r["doc_id"],
                quote=r["quote"],
                report_date=r["report_date"],
                as_of_date=r["as_of_date"],
                norm_num=norm[0] if norm else None,
                norm_unit=norm[1] if norm else None,
            ))
        return MetricSeries(entity=entity, metric=metric, points=pts)
    finally:
        if own_conn:
            conn.close()


def theme_heat(
    cfg: Config,
    theme: str,
    *,
    conn=None,
    by: str = "month",   # month / week
) -> list[ThemeBucket]:
    """某主题按月/周聚合的文档计数（热度曲线骨架）。

    theme 用 LIKE 子串匹配。**时间桶由 report_date 现算**，而非 documents.month/week
    列——后者靠源文件夹路径解析（第一周 / 2026年7月），大量为空，导致整条曲线塌进
    '未知'桶、趋势无从谈起。report_date 是 ISO 日期（2026-06-26），确定可信：
      · 按月 → 取前 7 位 '2026-06'
      · 按周 → strftime('%Y-W%W')（ISO 年+周序），同一发布日归同周
    report_date 缺失的极少数才落 '未知'。
    """
    own_conn = conn is None
    if own_conn:
        conn = store.connect(cfg.paths.db)
    try:
        if by == "week":
            bucket_expr = (
                "CASE WHEN d.report_date IS NULL OR d.report_date='' THEN '未知' "
                "ELSE strftime('%Y-W%W', d.report_date) END"
            )
        else:
            bucket_expr = (
                "CASE WHEN d.report_date IS NULL OR d.report_date='' THEN '未知' "
                "ELSE substr(d.report_date,1,7) END"
            )
        sql = (
            f"SELECT {bucket_expr} AS bucket, COUNT(DISTINCT d.doc_id) AS c "
            f"FROM doc_themes t JOIN documents d ON d.doc_id = t.doc_id "
            f"WHERE t.theme LIKE ? "
            f"GROUP BY bucket ORDER BY bucket"
        )
        rows = conn.execute(sql, (f"%{theme}%",)).fetchall()
        return [ThemeBucket(bucket=r["bucket"], count=r["c"]) for r in rows]
    finally:
        if own_conn:
            conn.close()


def top_themes(cfg: Config, *, conn=None, limit: int = 30) -> list[ThemeBucket]:
    """全库主题热度榜（按打标文档数），用于概览与词表维护。"""
    own_conn = conn is None
    if own_conn:
        conn = store.connect(cfg.paths.db)
    try:
        rows = conn.execute(
            "SELECT theme AS bucket, COUNT(DISTINCT doc_id) AS c "
            "FROM doc_themes GROUP BY theme ORDER BY c DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [ThemeBucket(bucket=r["bucket"], count=r["c"]) for r in rows]
    finally:
        if own_conn:
            conn.close()


def domestic_foreign_split(
    cfg: Config,
    theme: str | None = None,
    *,
    conn=None,
) -> dict:
    """国内 / 国外研报数量占比。theme 为空 → 全库；给定 → 该主题（LIKE 子串）。

    口径以 lang 为准（zh=国内券商 / en=外资投行，实测与 category 一一对应）。
    返回 {domestic, foreign, total, domestic_pct, foreign_pct}，pct 为百分比（0~100）。
    """
    own_conn = conn is None
    if own_conn:
        conn = store.connect(cfg.paths.db)
    try:
        if theme:
            sql = (
                "SELECT d.lang AS lang, COUNT(DISTINCT d.doc_id) AS c "
                "FROM doc_themes t JOIN documents d ON d.doc_id = t.doc_id "
                "WHERE t.theme LIKE ? GROUP BY d.lang"
            )
            rows = conn.execute(sql, (f"%{theme}%",)).fetchall()
        else:
            rows = conn.execute(
                "SELECT lang, COUNT(*) AS c FROM documents GROUP BY lang"
            ).fetchall()
        dom = for_ = 0
        for r in rows:
            if (r["lang"] or "").lower().startswith("en"):
                for_ += r["c"]
            else:
                dom += r["c"]
        total = dom + for_
        return {
            "domestic": dom,
            "foreign": for_,
            "total": total,
            "domestic_pct": round(dom / total * 100, 1) if total else 0.0,
            "foreign_pct": round(for_ / total * 100, 1) if total else 0.0,
        }
    finally:
        if own_conn:
            conn.close()
