"""Catalog：manifest 为主 + 文件系统校验，重建 documents 表。

核心原则（已探查验证）：
- manifest.jsonl 是干净 UTF-8，doc_id/source_relative/sha256/status 均可信；
  早前"乱码"是 GBK 误读假象。因此机构/标题/日期直接从 manifest 采信。
- 只有 manifest 的 result 路径过期（指向旧 04_canonical 布局）——丢弃它，
  md_path/images_dir 用扫描磁盘得到的真实路径。
- 类别用关键词匹配（"国内券商报告"/"投行报告"），不依赖目录序号（02_/03_ 本地可能被改）。
- doc_id = canonical 目录名 = sha256[:32]，以此为锚回连磁盘，幂等可重跑。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from urllib.parse import unquote

from .config import Config
from . import store

# 原始文件名里的可选字段（交叉校验 + 补充）
_STOCK_CODE = re.compile(r"[（(](\d{6})[)）]")
_PAGE_COUNT = re.compile(r"-(\d+)页")

# 类别关键词 → 归一类别名（不依赖 02_/03_ 序号）
_CATEGORY_KEYWORDS = [
    ("国内券商报告", "国内券商报告"),
    ("投行报告", "投行报告"),
]


@dataclass
class RunReport:
    inserted: int = 0
    updated: int = 0
    skipped_no_dir: int = 0
    needs_review: int = 0
    total_success: int = 0
    skipped_docs: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"manifest success: {self.total_success}\n"
            f"入库(新增+更新): {self.inserted + self.updated} "
            f"(新增 {self.inserted} / 更新 {self.updated})\n"
            f"跳过(磁盘无目录): {self.skipped_no_dir}\n"
            f"标记 needs_review: {self.needs_review}"
        )


# 已人工核实的 report_date 订正表：doc_id → 正确 ISO 日期。
#
# **为什么必须放在代码里而不是直接 UPDATE 数据库**：build_catalog 每次 update 都会
# 从文件名重新解析 report_date 并 upsert 回去，手工改的库值会在下一次更新时被覆盖。
# 要么改源 PDF 文件名（治本，但那是用户的原始资料，程序不擅自动），要么在这里留一条
# 可复现、可复核的订正记录 —— 选后者。
#
# 每条都要写清依据，便于日后复核。
_DATE_OVERRIDES: dict[str, str] = {
    # 文件名写作 `-270730.pdf`，但归档路径是 2026年7月\第五周，正文为 Meta Q2'26
    # 季报点评（2026-07-30 发布），故年份为手误。原值 2027-07-30 会成为全库最大
    # report_date，把 detect_drift 的近期窗口整体推到语料之外（见 chain.detect_drift）。
    "c676069b5f3928eb763a50a3efabeca6": "2026-07-30",
}
# 文件名日期的未来容忍上限。研报日期由人工命名的 PDF 文件名给出，年份手误必然发生
# （实测 `...-270730.pdf` 应为 260730，被解析成 2027-07-30）。这类脏日期不只是一行
# 数据难看：任何"以库内最新 report_date 当今天"的逻辑都会被它整体带偏——
# chain.detect_drift 就因此把 60 条链的漂移候选全判成"已冷掉"而静默归零。
# 留 7 天余量：研报偶有小幅提前标注日期（周末发布标下周一等），不该误杀。
_FUTURE_TOLERANCE_DAYS = 7


def _yymmdd_to_iso(yymmdd: str) -> str | None:
    """260626 → 2026-06-26。非法日期返回 None。

    只做格式与下界校验；未来日期的判定见 `_date_reason`（要区分"没解析出来"和
    "解析出来但不可信"，两者的 review_reason 不同）。
    """
    try:
        d = datetime.strptime(yymmdd, "%y%m%d").date()
    except ValueError:
        return None
    # 研报日期不该是未来太久，也不该早于 2000；此处只做基本合法性
    if d < date(2000, 1, 1):
        return None
    return d.isoformat()


def _date_reason(report_date: str | None) -> str | None:
    """校验 report_date 的可信度，返回 review_reason（None = 无问题）。

    未来日期**保留原值不改写**、只标 needs_review：改写会掩盖源文件名的手误，
    而这个仓库的真相在文件名里，得让人去改文件名或手工订正，而不是让程序猜。
    """
    if not report_date:
        return "report_date 未能解析"
    try:
        d = date.fromisoformat(report_date)
    except ValueError:
        return "report_date 格式非法"
    limit = date.today() + timedelta(days=_FUTURE_TOLERANCE_DAYS)
    if d > limit:
        return f"report_date 为未来日期（{report_date}），疑似文件名年份手误"
    return None


def _detect_lang(text: str, sample: int = 4000) -> str:
    """按中文字符占比判定 zh/en。只取样开头一段，避免整篇扫描。"""
    head = text[:sample]
    if not head:
        return "und"
    cjk = sum(1 for ch in head if "一" <= ch <= "鿿")
    letters = sum(1 for ch in head if ch.isascii() and ch.isalpha())
    denom = cjk + letters
    if denom == 0:
        return "und"
    return "zh" if cjk / denom >= 0.2 else "en"


def _category_from_path(source_relative: str) -> str | None:
    for keyword, label in _CATEGORY_KEYWORDS:
        if keyword in source_relative:
            return label
    return None


def _week_month_from_path(source_relative: str) -> tuple[str | None, str | None]:
    """从相对路径解析 周次 / 月份。
    相对路径形如：第一周\\02_国内券商报告-594份\\<file>.pdf（月份可能在更上层）。
    """
    parts = re.split(r"[\\/]", source_relative)
    week = next((p for p in parts if re.fullmatch(r"第[一二三四五六七八九十]+周", p)), None)
    month = next((p for p in parts if re.fullmatch(r"\d{4}年\d{1,2}月", p)), None)
    return week, month


@dataclass
class ParsedName:
    institution: str
    title: str
    yymmdd: str | None
    stock_code: str | None
    page_count: int | None
    reason: str | None = None


# 机构名一般 ≤12 字且不含空格；用于判断"机构在前"还是标题里恰好含 '-'
_MAX_INST_LEN = 12


def _parse_filename(source_filename: str) -> ParsedName:
    """以原始文件名为主解析元数据（每条 manifest 都有、且正确）。

    经全量核对，1762 份里：
    - 1761 份以 ``-YYMMDD.pdf`` 结尾；机构在前（首个 '-' 前 ≤12 字）占 ~1458。
    - manifest 的 doc_id 字段 1739/1762 是裸哈希，不可作元数据源——故不依赖它。

    覆盖的模式：
      A（主）：``<机构>-<标题>-<YYMMDD>.pdf``
      B（变体）：``<标题>(<代码>)…-<YYMMDD>-<机构>-<NN页>.pdf``
      英文：``<Bank>-<title>-<YYMMDD>.pdf``
    """
    stem = Path(source_filename).stem
    # 文件名里可能残留 URL 编码（如 %2b→+）；解码使标题干净。
    # unquote 只转合法 %XX 序列，正常百分号（如“增长18%”）原样保留。
    if "%" in stem:
        stem = unquote(stem)
    reason: str | None = None

    # 1) 抽可选字段：股票代码、页数（不改变主结构）
    sc = _STOCK_CODE.search(stem)
    stock_code = sc.group(1) if sc else None
    pc = _PAGE_COUNT.search(stem)
    page_count = int(pc.group(1)) if pc else None

    # 2) 剥掉尾部 -NN页（若有），便于定位日期
    work = re.sub(r"-\d+页$", "", stem)

    # 3) 找日期锚点 -YYMMDD。取最后一个 6 位数字段作为报告日期锚。
    date_matches = list(re.finditer(r"-(\d{6})(?=-|$)", work))
    if date_matches:
        dm = date_matches[-1]
        yymmdd = dm.group(1)
        before = work[: dm.start()]
        after = work[dm.end():].strip("-")
        # 变体 B：日期后还跟着机构（<标题>-YYMMDD-<机构>）
        if after and "页" not in after and len(after) <= _MAX_INST_LEN:
            institution = after
            title = before
        else:
            # 模式 A：机构在前，标题在中
            institution, _, title = before.partition("-")
            if not title:  # 没有第二个 '-'，说明前面整段既是机构也可能无标题
                title = institution
                institution = ""
    else:
        # 无日期锚（极少）：整体当标题，机构留待下方兜底
        yymmdd = None
        institution, _, title = stem.partition("-")
        if not title:
            title, institution = institution, ""

    # 4) 机构合理性校验。中英文机构名长度差异大：
    #    中文券商名一般 ≤8 汉字；英文投行名（Goldman Sachs / BofA Securities）
    #    含空格且字符数长，属正常。真正"把标题吃进机构"的信号是含全角标题标点
    #    （：（）等）或 CJK 字符过多。据此分别判定，避免误伤英文机构。
    institution = institution.strip()
    cjk_len = sum(1 for ch in institution if "一" <= ch <= "鿿")
    has_title_punct = any(p in institution for p in "：（）:")
    if institution and (cjk_len > _MAX_INST_LEN or has_title_punct):
        reason = "机构解析存疑（疑似含标题）"
    if not institution:
        institution = "未知"
        reason = "机构未能从文件名解析"

    title = title.strip(" -_") or stem
    return ParsedName(institution, title, yymmdd, stock_code, page_count, reason)


def _iter_manifest_success(manifest_path: Path):
    """UTF-8 显式读 manifest.jsonl，逐行 yield status=success 的记录。"""
    with manifest_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") == "success":
                yield rec


def build_catalog(cfg: Config, conn=None) -> RunReport:
    """全量重建 catalog（幂等）。返回 run 报告。"""
    from .config import require_inputs

    require_inputs(cfg)
    own_conn = conn is None
    if own_conn:
        conn = store.connect(cfg.paths.db)

    report = RunReport()
    now = datetime.now().astimezone().isoformat(timespec="seconds")

    try:
        for rec in _iter_manifest_success(cfg.paths.manifest):
            report.total_success += 1
            sha256 = rec.get("sha256", "")
            doc_id = sha256[:32] if sha256 else rec.get("doc_id", "")
            if not doc_id:
                continue

            doc_dir = cfg.paths.canonical / doc_id
            md_path = doc_dir / "full.md"
            if not md_path.is_file():
                report.skipped_no_dir += 1
                report.skipped_docs.append(doc_id)
                continue

            source_relative = rec.get("source_relative", "") or ""
            source_filename = Path(
                (rec.get("source") or source_relative).replace("\\", "/")
            ).name

            # 以文件名为主解析（manifest 的 doc_id 字段多为裸哈希，不可信）
            parsed = _parse_filename(source_filename)
            report_date = _yymmdd_to_iso(parsed.yymmdd) if parsed.yymmdd else None
            # 人工订正优先于文件名解析（文件名本身写错了，见 _DATE_OVERRIDES）
            date_override = _DATE_OVERRIDES.get(doc_id)
            if date_override:
                report_date = date_override

            text = md_path.read_text(encoding="utf-8", errors="replace")
            lang = _detect_lang(text)
            institution = parsed.institution
            title = parsed.title
            inst_reason = parsed.reason

            category = _category_from_path(source_relative)
            week, month = _week_month_from_path(source_relative)

            stock_code = parsed.stock_code
            page_count = parsed.page_count

            images_dir = doc_dir / "images"
            image_count = (
                sum(1 for _ in images_dir.glob("*.jpg")) if images_dir.is_dir() else 0
            )

            # review 判定：机构或日期缺失/不可信即需人工核
            reasons = []
            if inst_reason:
                reasons.append(inst_reason)
            date_reason = _date_reason(report_date)
            if date_reason:
                reasons.append(date_reason)
            if not category:
                reasons.append("category 未能从路径匹配")
            needs_review = 1 if reasons else 0
            if needs_review:
                report.needs_review += 1

            row = {
                "doc_id": doc_id,
                "sha256": sha256,
                "title": title,
                "institution": institution,
                "category": category,
                "lang": lang,
                "report_date": report_date,
                "week": week,
                "month": month,
                "stock_code": stock_code,
                "page_count": page_count,
                "md_path": str(md_path),
                "images_dir": str(images_dir) if images_dir.is_dir() else None,
                "char_count": len(text),
                "image_count": image_count,
                "source_filename": source_filename,
                "source_relative": source_relative,
                "needs_review": needs_review,
                "review_reason": "; ".join(reasons) if reasons else None,
                "indexed_at": now,
            }

            existed = conn.execute(
                "SELECT 1 FROM documents WHERE doc_id=?", (doc_id,)
            ).fetchone()
            _upsert(conn, row)
            if existed:
                report.updated += 1
            else:
                report.inserted += 1

        conn.commit()
    finally:
        if own_conn:
            conn.close()

    return report


def _upsert(conn, row: dict) -> None:
    cols = list(row.keys())
    placeholders = ",".join("?" for _ in cols)
    updates = ",".join(f"{c}=excluded.{c}" for c in cols if c != "doc_id")
    sql = (
        f"INSERT INTO documents ({','.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(doc_id) DO UPDATE SET {updates}"
    )
    conn.execute(sql, [row[c] for c in cols])
