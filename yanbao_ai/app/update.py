"""Update：长效增量更新的一键编排（方案 §7，系统"长效"核心）。

每周流程（幂等、可续跑）：
1. 重建 catalog（manifest 为主 + 文件系统校验）——新 PDF 经 mineru 产出后，
   documents 表补齐新 doc_id；已存在的按 ON CONFLICT 更新元数据。
2. 只索引"尚无 chunk"的文档（新文档或上次失败的）→ 规范化→切块→嵌入→落库。
3. 增量抽取事实/主题：跳过 extraction_log 里当前 schema 版本已抽的文档。

一切以 doc_id 内容哈希为锚，重跑不会重复、不会丢。任一步失败只影响该步的
计数，不中断整体；facts 步无 Claude 端点/额度时可用 --no-facts 跳过。

与逐命令手动跑的区别：update 用同一个连接把三步串起来，且默认只碰增量，
适合每周新增几十份时一条命令收工。首次全量建库仍建议分步跑（index 全量 +
facts 全量），便于观察与断点。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from . import catalog, store, index as index_mod


@dataclass
class UpdateStats:
    catalog_inserted: int = 0
    catalog_updated: int = 0
    index_docs: int = 0        # 本次新索引文档数
    index_chunks: int = 0      # 本次新写块数
    facts_docs: int = 0        # 本次新抽取文档数
    facts_written: int = 0
    themes_written: int = 0
    vec_enabled: bool = False
    steps_skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"catalog：新增 {self.catalog_inserted} / 更新 {self.catalog_updated}",
            f"index（仅新）：{self.index_docs} 篇、{self.index_chunks} 块"
            f"（向量：{'启用' if self.vec_enabled else '未启用'}）",
            f"facts（增量）：{self.facts_docs} 篇、"
            f"{self.facts_written} 事实、{self.themes_written} 主题",
        ]
        if self.steps_skipped:
            lines.append(f"跳过步骤：{', '.join(self.steps_skipped)}")
        if self.errors:
            lines.append(f"错误 {len(self.errors)} 条（前 5）：{self.errors[:5]}")
        return "\n".join(lines)


def run_update(
    cfg: Config,
    *,
    embed: bool = True,
    do_facts: bool = True,
    facts_strong: bool = False,
    facts_limit: int | None = None,
) -> UpdateStats:
    """跑一次增量更新：catalog → index(仅新) → facts(增量)。返回汇总。"""
    st = UpdateStats()

    # 单连接贯穿三步（catalog 用它建 documents，index/facts 复用）
    conn, vec_ok = store.connect_with_vec(cfg.paths.db, cfg.embed.dimensions)
    try:
        # 1) catalog（manifest 为主，幂等 upsert）
        try:
            report = catalog.build_catalog(cfg, conn=conn)
            st.catalog_inserted = report.inserted
            st.catalog_updated = report.updated
        except Exception as exc:  # noqa: BLE001 - catalog 失败不应吞掉后续可诊断信息
            st.errors.append(f"catalog 步失败：{exc}")
            return st  # catalog 是地基，失败则无意义继续

        # 2) index 仅新文档
        idx = index_mod.build_index(cfg, conn=conn, embed=embed, only_new=True)
        st.index_docs = idx.docs_done
        st.index_chunks = idx.chunks_written
        st.vec_enabled = idx.vec_enabled
        st.errors.extend(f"index: {e}" for e in idx.errors)

        # 3) facts 增量抽取（可跳过）
        if not do_facts:
            st.steps_skipped.append("facts（--no-facts）")
            return st
        try:
            from . import facts as facts_mod

            fst = facts_mod.build_facts(
                cfg,
                conn=conn,
                strong=facts_strong,
                reextract=False,       # 增量：跳过已抽
                limit=facts_limit,
            )
            st.facts_docs = fst.docs_done
            st.facts_written = fst.facts_written
            st.themes_written = fst.themes_written
            st.errors.extend(f"facts: {e}" for e in fst.errors)
        except Exception as exc:  # noqa: BLE001 - facts 无端点/额度时不应连累前两步成果
            st.steps_skipped.append("facts（异常）")
            st.errors.append(f"facts 步失败（前两步成果已落库）：{exc}")
    finally:
        conn.close()

    return st
