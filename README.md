# 研报本地长效 AI 系统 · 现状总览

把 MinerU 处理过的研报（Markdown + 图片）沉淀为一个**长效、可增量、可追溯**的本地知识系统。核心能力：**带引用的问答（RAG）**、**指标/主题时间线**、**产业链结构下钻**。全部本地运行，只在生成/嵌入时调用外部 API。

> 这份文档是**系统当前状态**的总览。开工前的原始设计方案已归档在 git 历史里；本文只描述现在实际跑着的东西。
> 子系统各有自己的 README：抓取转换见 [`mineru_pipeline/README.md`](mineru_pipeline/README.md)，索引问答见 [`yanbao_ai/README.md`](yanbao_ai/README.md)。

---

## 目录布局

```
E:\yanbao\
  2026年7月\第一周\...            # 原始 PDF（按 月份\周次\类别 组织；年份可为 2027…）
  2026年6月前\                    # 历史散装 PDF（未全部入流水线）
  mineru_pipeline\               # 前半段：PDF → Markdown（MinerU 云 API 批处理）
    canonical\<doc_id>\          #   产物：full.md + images/（doc_id = sha256(pdf)[:32]）
    manifest\manifest.jsonl      #   一等元数据源（UTF-8）：机构/标题/日期/周次/类别/状态
    raw_downloads\ failed\
  yanbao_ai\                     # 后半段：索引 + 问答 + 时间线 + 产业链（本系统主体）
    app\                         #   Python 包（python -m app.cli <命令>）
    scripts\                     #   批处理/运维脚本（重建产业链、一键更新）
    data\yanbao.db               #   SQLite（gitignore）：文档/块/向量/事实/主题/产业链
```

## 两段式流水线

```
新 PDF（每 1~2 周新增一到两周的量）
  └─[前半段] mineru_pipeline  →  canonical/<doc_id>/{full.md, images/} + manifest.jsonl
                                        │
  ┌──────────────────────────────────────┘  （yanbao_ai 从这里开始消费）
  ▼
 catalog   manifest 为主 + 文件系统校验 → documents 表（机构/日期/周次/类别/语言）
  ▼
 index     规范化 → 结构切块 → qwen 嵌入 → SQLite(chunks + FTS5 + sqlite-vec)
  ▼
 ask       混合检索（BM25 ⊕ 稠密，RRF 融合）+ Claude 带引用生成 / 产业链深度分析
  ▼
 facts     Claude 结构化抽取指标事实 + 主题打标 → 时间线骨架
  ▼
 chain     产业链结构（60 条链，AI 一次性构建落库）+ 漂移检测候选
  ▼
 serve     本地 Web 界面（FastAPI，绑 127.0.0.1，无鉴权）
```

一切以 `doc_id`（内容 sha256）为锚：**幂等、可续跑、重跑不重复不丢**。

## 当前规模（实测）

- **1762 份研报**（中文券商 1145 + 英文投行 617），catalog 解析成功率 99.94%。
- **76750 个文本块**，全部嵌入向量（qwen3.7-text-embedding @ 1024 维）。
- **facts 约 5.3 万条 / doc_themes 约 1.27 万条**，1762/1762 全抽取。
- **60 条产业链**（AI 构建落库，趋势页可下钻），漂移候选表随更新刷新。

## 日常使用

**每隔一到两周有新数据时**，把新 PDF 放进 `2026年8月\第N周\...`（或明年 `2027年…`），然后一条命令跑完全流程：

```powershell
# 在 E:\yanbao\yanbao_ai 目录下
python scripts\update_all.py

python yanbao_ai\scripts\update_all.py
```

它串起：mineru（新 PDF → canonical）→ catalog/index/facts 增量 → 产业链漂移检测（写候选表等人审，**不自动重建链**）。`input_dir` 已设为仓库根，任何新 `YYYY年N月` 目录都会被自动扫到。开关见 `yanbao_ai/README.md`。

**平时查询**用本地 Web：

```powershell
python -m app.cli serve      # 浏览器开 http://127.0.0.1:8000
python -m yanbao_ai.app.cli serve
```

各命令的细节（build-catalog / index / ask / facts / timeline / build-chain / chain-drift / serve / doctor）见 [`yanbao_ai/README.md`](yanbao_ai/README.md)。

## 端点与凭证

两个独立 provider，配置在 `yanbao_ai/config.toml`（gitignore）：

- **`[llm]`（Claude，生成/抽取/重排/建链）**：走中转站的 Anthropic 兼容端点；留空时回落读 `~/.claude/settings.json` 的 `ANTHROPIC_BASE_URL`/`ANTHROPIC_AUTH_TOKEN`（cc-switch 管理）。当前只服务 `claude-opus-4-8` 一个模型。
- **`[embed]`（qwen，向量化）**：阿里云 DashScope 的 OpenAI 兼容端点，`qwen3.7-text-embedding @ 1024 维`（batch 硬上限 20）。

密钥只经 env 或未跟踪的 `config.toml`，**绝不写入库、绝不日志打印明文**。以下明文落盘凭证的轮换属你的手动高风险动作，系统不代改：`mineru_pipeline/mineru_pipeline.json` 的 `MINERU_API_KEY`、`~/.claude/settings.json` 的 `ANTHROPIC_AUTH_TOKEN`。

## 安全边界

- `serve` 仅监听 `127.0.0.1`、**无鉴权**，面向本机单用户。**绝不要绑 `0.0.0.0` 或暴露公网**——否则任何人都能查询付费研报库、触发 Claude/qwen 计费。
- 付费研报的 chunk 文本会外发两处：嵌入 → qwen 端点，生成/抽取 → Claude 中转站。这是走 API 的必然代价，已在约束内接受。

## 成本提示

- `catalog` / `timeline` / `chain-drift` / 时间线与产业链的**读库**：零 API 成本。
- `index`（嵌入）、`ask`、`facts`、`build-chain`、产业链「AI 解读」：调用外部 API 计费。产业链的「链路 AI 解读」首次点某主题必然慢（实测数分钟）且计费，之后同主题命中后端缓存 0.2s 免费。
