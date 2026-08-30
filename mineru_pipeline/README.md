# MinerU PDF 批处理

1. `mineru_pipeline.json` 的输入目录 `input_dir` 已设为**仓库根 `E:\yanbao`** 并递归扫描——任何 `2026年N月\第N周\...`（年份可为 2027）的新目录都会被自动扫到，**新增月份/周次不必再改配置**；输出统一放在 `E:\yanbao\mineru_pipeline`。（全仓库 PDF 只落在月份目录，产物目录零 PDF，故扫仓库根不会误扫。）需要改位置时，改 `input_dir` / `root` 两项。
2. 设置 Key（PowerShell）：`$env:MINERU_API_KEY='你的Key'`。建议只使用环境变量，不要把 Key 明文保存在配置文件或提交到 Git。
3. 待处理 PDF 直接放在 `E:\yanbao` 下任意 `YYYY年N月\第N周\...` 子文件夹中；脚本从原位置直接上传，不复制、移动、改名或删除原文件。**去重靠内容 sha256**——已处理过的周次即使随新月份一起被扫到，也会自动跳过、不重复计费。
4. 先执行单批验证：`python E:\yanbao\mineru_pipeline\mineru_pipeline.py --config E:\yanbao\mineru_pipeline\mineru_pipeline.json --once`。确认正常后去掉 `--once` 连续处理。

成功结果位于 `canonical\<document_id>`，现在只解压 Markdown 文件和 `images\` 文件夹；JSON、PDF、layout 等不会写入新的结果目录。原始包位于 `raw_downloads`，清单位于 `manifest\manifest.jsonl`。脚本不会删除文件。MinerU API 字段可能随版本变化；若控制台使用的 API 不是 `/api/v4/file-urls/batch`，只需调整配置或接口解析逻辑。

程序会显示扫描、上传、轮询、下载和解压进度。遇到 HTTP 429 或临时网络错误时会自动退避重试；持续限流则停止提交新批次，已提交批次保留在清单中，下次启动自动恢复。`upload_workers` 和 `download_workers` 控制传输并发，过高可能触发限流；当前默认值分别为 6 和 4。
