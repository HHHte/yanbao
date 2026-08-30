"""Extract only Markdown and images from MinerU result ZIP files.

This script is offline: it does not call MinerU and does not delete files.
"""
from __future__ import annotations
import argparse
import shutil
import zipfile
from pathlib import Path
from clean_markdown import clean as clean_markdown

def extract_one(zip_path: Path, output_dir: Path, overwrite: bool = False) -> tuple[int, int, int]:
    if output_dir.exists() and any(output_dir.rglob("*.md")) and not overwrite:
        return 0, 0, 0
    output_dir.mkdir(parents=True, exist_ok=True)
    files = images = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = info.filename.replace("\\", "/")
            parts = [p for p in name.split("/") if p]
            if not parts or name.endswith("/") or any(p in (".", "..") for p in parts):
                continue
            image_index = next((i for i, p in enumerate(parts) if p.lower() == "images"), None)
            is_md = name.lower().endswith(".md")
            if not (is_md or image_index is not None):
                continue
            target = output_dir.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            files += 1
            images += int(image_index is not None)
    cleaned = 0
    for md in output_dir.rglob("*.md"):
        text = md.read_text(encoding="utf-8")
        result, count = clean_markdown(text)
        if count:
            md.write_text(result, encoding="utf-8")
            cleaned += count
    return files, images, cleaned

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"E:\yanbao\mineru_pipeline")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已有 canonical 中同名提取结果")
    args = ap.parse_args()
    root = Path(args.root)
    raw = root / "raw_downloads"
    canonical = root / "canonical"
    zips = sorted(raw.glob("*.zip"))
    total_files = total_images = processed = skipped = 0
    for z in zips:
        out = canonical / z.stem
        n, ni, nc = extract_one(z, out, args.overwrite)
        if n == 0 and ni == 0 and nc == 0 and out.exists() and not args.overwrite:
            skipped += 1
        else:
            processed += 1
            total_files += n
            total_images += ni
            print(f"{z.name}: 提取 {n} 个文件，其中 images 文件 {ni} 个，清理广告 {nc} 处")
    print(f"压缩包 {len(zips)} 个；处理 {processed} 个；跳过 {skipped} 个；提取文件 {total_files} 个；images 文件 {total_images} 个")

if __name__ == "__main__":
    main()
