"""Remove the recurring 起点财经 promotional blocks from MinerU Markdown files."""
from __future__ import annotations
import argparse, re
from pathlib import Path

LINE_PATTERNS = [
    re.compile(r"^#\s+(?:\*\*)?每日报告(?:\*\*)?\s*$"),
    re.compile(
        r"^##\s+(?:\*\*)?START\s*YOUR\s*FINANCE(?:\*\*)?\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"^每日微信群内分享7\+最新重磅报告[；;]\s*$"),
    re.compile(r"^行研报告均为公开版，权利归原作者所有[，,]?起点财经仅分发做内部学习[。.]\s*$"),
    re.compile(r"^扫一扫二维码关注公号回复:“研究报告”加入“起点财经”微信群\s*$"),
    re.compile(r"^起点财经，网罗天下报告\s*$"),
]
IMAGE_LINE = re.compile(r"^\s*!?\[[^]]*\]\(images/[^)]+\)\s*$", re.I)

def clean(text: str) -> tuple[str, int]:
    kept, count = [], 0
    remove_next_image = False
    in_start_block = False
    discard_after_start = False
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        original = content
        stripped = content.strip()
        if discard_after_start:
            if content:
                count += 1
            continue
        if re.fullmatch(r"#\s+(?:\*\*)?每日报告(?:\*\*)?", stripped):
            content = ""
            remove_next_image = False
        elif re.fullmatch(
            r"##\s+(?:\*\*)?START\s*YOUR\s*FINANCE(?:\*\*)?",
            stripped,
            re.IGNORECASE,
        ):
            content = ""
            in_start_block = True
            discard_after_start = True
            remove_next_image = True
        elif in_start_block and IMAGE_LINE.fullmatch(stripped):
            content = ""
            in_start_block = False
            remove_next_image = False
        elif remove_next_image and IMAGE_LINE.fullmatch(stripped):
            content = ""
            remove_next_image = False
        # 删除固定广告片段，不删除同一行中广告片段之后的正文。
        for phrase in (
            "每日微信群内分享7+最新重磅报告；",
            "每日微信群内分享7+最新重磅报告;",
            "行研报告均为公开版，权利归原作者所有起点财经仅分发做内部学习。",
            "行研报告均为公开版，权利归原作者所有，起点财经仅分发做内部学习。",
            "扫一扫二维码关注公号回复:“研究报告”加入“起点财经”微信群",
            "起点财经，网罗天下报告",
        ):
            if phrase in content:
                content = content.replace(phrase, "")
                # 第一张二维码紧跟版权宣传语；不要因正文中的“扫一扫”变体误删正文图片。
                if "行研报告均为公开版" in phrase:
                    remove_next_image = True
        if stripped == "起点财经，网罗天下报告":
            content = ""
            in_start_block = False
        if any(pattern.fullmatch(content.strip()) for pattern in LINE_PATTERNS):
            content = ""
        if content != original:
            count += 1
        if content.strip():
            kept.append(content + ("\n" if line.endswith("\n") else ""))
    return "".join(kept), count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"E:\yanbao\mineru_pipeline\canonical")
    ap.add_argument("--in-place", action="store_true", help="原地修改 Markdown，不创建备份文件")
    ap.add_argument("--output", help="输出到另一个目录；不修改原文件")
    args = ap.parse_args()
    root = Path(args.root)
    out_root = Path(args.output) if args.output else None
    if out_root: out_root.mkdir(parents=True, exist_ok=True)
    total = changed = 0
    for src in root.rglob("*.md"):
        text = src.read_text(encoding="utf-8")
        cleaned, n = clean(text)
        total += 1
        if not n: continue
        changed += 1
        if args.in_place:
            src.write_text(cleaned, encoding="utf-8")
        elif out_root:
            dest = out_root / src.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(cleaned, encoding="utf-8")
        else:
            print(f"匹配 {n} 处：{src}")
    print(f"扫描 {total} 个 Markdown，匹配并处理 {changed} 个。")
    if not args.in_place and not out_root:
        print("预览模式：未修改文件。需要修改请加 --in-place，或使用 --output 指定新目录。")

if __name__ == "__main__": main()
