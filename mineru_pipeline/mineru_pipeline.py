"""Batch PDF processing pipeline for MinerU Cloud API v4.

The API accepts multiple files in one ``/file-urls/batch`` request.  This
implementation submits at most 45 files per batch, uploads the returned
presigned URLs concurrently, polls the batch once, and archives each result.
It never deletes or moves source files.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import getpass
import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import requests


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def api_json(method: str, url: str, token: str, payload=None):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    request = Request(url, data=data, headers=headers, method=method)
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def put_file(url: str, path: Path) -> None:
    with path.open("rb") as stream:
        response = requests.put(url, data=stream, timeout=900)
    if response.status_code >= 400:
        raise RuntimeError(f"upload HTTP {response.status_code}: {response.text[:500]}")


def download(url: str, output: Path) -> None:
    request = Request(url, headers={"User-Agent": "mineru-batch/2.0"})
    with urlopen(request, timeout=900) as response, output.open("wb") as stream:
        shutil.copyfileobj(response, stream)


def extract_md_images(zip_path: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            name = info.filename.replace("\\", "/")
            parts = [part for part in name.split("/") if part]
            if not parts or name.endswith("/") or any(part in {".", ".."} for part in parts):
                continue
            image_dir = any(part.lower() == "images" for part in parts)
            if not (name.lower().endswith(".md") or image_dir):
                continue
            target = output.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest)


def result_items(response):
    data = response.get("data", response)
    if isinstance(data, dict):
        return data.get("extract_result") or data.get("results") or []
    return data if isinstance(data, list) else []


def legacy_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mineru_pipeline.json")
    parser.add_argument("--once", action="store_true", help="只提交并处理当前扫描结果的一批")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(config.get("root", "E:/yanbao/mineru_pipeline"))
    input_dir = Path(config.get("input_dir", str(root / "00_inbox")))
    for name in ("manifest", "raw_downloads", "canonical", "failed"):
        (root / name).mkdir(parents=True, exist_ok=True)

    token_env = config.get("token_env", "MINERU_API_KEY")
    token = os.environ.get(token_env) or config.get(token_env) or config.get("api_key")
    if not token:
        token = getpass.getpass("MinerU Token: ").strip()
    if not token:
        raise SystemExit("未提供 MinerU Token")
    base = config.get("api_base", "https://mineru.net/api/v4").rstrip("/")
    manifest_path = root / "manifest" / "manifest.jsonl"
    done = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                done[record.get("sha256")] = record
            except json.JSONDecodeError:
                pass

    pdfs = sorted(input_dir.rglob("*.pdf") if config.get("recursive", True) else input_dir.glob("*.pdf"))
    pending = [p for p in pdfs if not (done.get(sha256(p), {}).get("status") == "success")]
    batch_size = min(int(config.get("batch_size", 45)), 45)
    workers = max(1, int(config.get("upload_workers", 8)))
    print(f"发现 {len(pdfs)} 个 PDF，待处理 {len(pending)} 个，批大小 {batch_size}", flush=True)

    for offset in range(0, len(pending), batch_size):
        batch_files = pending[offset:offset + batch_size]
        records = []
        by_id = {}
        for path in batch_files:
            digest = sha256(path)
            doc_id = digest[:32]  # ASCII, 远低于 MinerU data_id 的 128 字节限制
            record = {"doc_id": doc_id, "source": str(path), "source_relative": str(path.relative_to(input_dir)),
                      "sha256": digest, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "status": "submitted"}
            records.append(record)
            by_id[doc_id] = (path, record)
        stage = "提交任务"
        try:
            payload = {"files": [{"name": path.name, "is_ocr": config.get("is_ocr", False), "data_id": record["doc_id"]}
                                  for path, record in (by_id[item["doc_id"]] for item in records)],
                       "model_version": config.get("model_version", "pipeline")}
            created = api_json("POST", base + "/file-urls/batch", token, payload)
            data = created.get("data", created)
            batch_id = data.get("batch_id")
            urls = data.get("file_urls") or []
            if not batch_id or len(urls) != len(batch_files):
                raise RuntimeError(f"批量提交响应异常: batch_id={batch_id}, file_urls={len(urls)}")
            stage = "上传文件"
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(put_file, url, path) for url, path in zip(urls, batch_files)]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            for record in records:
                record["batch_id"] = batch_id

            stage = "轮询解析"
            items_by_id = {}
            for attempt in range(int(config.get("max_polls", 120))):
                if attempt:
                    time.sleep(float(config.get("poll_seconds", 10)))
                state = api_json("GET", base + f"/extract-results/batch/{batch_id}", token)
                items = result_items(state)
                items_by_id = {str(item.get("data_id")): item for item in items if isinstance(item, dict)}
                if len(items_by_id) >= len(records) and all(str(item.get("state", "")).lower() in {"done", "success", "failed", "error"} for item in items):
                    break
            else:
                raise TimeoutError(f"批次 {batch_id} 轮询超时")

            for record in records:
                path = Path(record["source"])
                item = items_by_id.get(record["doc_id"], {})
                result_url = item.get("full_zip_url") or item.get("zip_url") or item.get("download_url")
                if not result_url:
                    raise RuntimeError(f"{path.name}: {item.get('err_msg') or item}")
                raw = root / "raw_downloads" / f"{record['doc_id']}.zip"
                download(result_url, raw)
                output = root / "canonical" / record["doc_id"]
                extract_md_images(raw, output)
                record.update(status="success", result=str(output), raw_zip=str(raw), finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        except HTTPError as error:
            if error.code in (401, 403):
                raise SystemExit(f"{stage} HTTP {error.code}，请检查 Token 和 MinerU 权限") from error
            for record in records:
                record.update(status="failed", error=f"HTTP {error.code}: {error.reason}", failed_dir=str(root / "failed"))
        except Exception as error:
            for record in records:
                record.update(status="failed", error=f"{stage}: {error}", failed_dir=str(root / "failed"))
        with manifest_path.open("a", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"批次完成 {offset + len(batch_files)}/{len(pending)}", flush=True)
        if args.once:
            break


def main() -> None:
    from mineru_pipeline_core import main as optimized_main

    optimized_main()


if __name__ == "__main__":
    main()
