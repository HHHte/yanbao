"""Reliable, resumable MinerU Cloud API v4 batch pipeline."""
from __future__ import annotations

import argparse
import concurrent.futures
import email.utils
import getpass
import hashlib
import json
import os
import random
import shutil
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import requests

TERMINAL_STATES = {"done", "success", "failed", "error", "cancelled", "canceled"}
RETRYABLE_HTTP_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
# 官方 HTTP 200 但业务失败的状态码（响应 JSON 的 code 字段）
FATAL_BUSINESS_CODES = {"A0202", "A0211", "A0212"}          # Token 错误/过期，不可恢复
DAILY_LIMIT_CODES = {"-60018"}                             # 每日任务数量达到上限，应次日再跑
RETRYABLE_BUSINESS_CODES = {"-60009", "-60007", "-10001"}  # 队列已满/服务繁忙，退避重试


class RateLimitExhausted(RuntimeError):
    """The API remained rate-limited after all retries."""


class RecoverableBatchError(RuntimeError):
    """A submitted batch should remain resumable instead of being failed."""


class DailyLimitReached(RuntimeError):
    """MinerU 返回每日任务上限，应停止并次日重试。"""


class FatalApiError(RuntimeError):
    """Token 无效/过期等不可恢复的业务错误。"""


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _longp(path) -> str:
    r"""把路径转成可跳过 Win32 路径规范化的形式，**仅供 os/open 调用**。

    MAX_PATH=260 的限制来自 Win32 层（CreateFileW 等）的路径规范化；内核与
    NTFS 本身支持约 32767 字符。加 \\?\ 前缀即绕过规范化直达内核。不加前缀时
    261 字符的路径会得到 winerror=3「系统找不到指定的路径」——文件其实在盘上，
    是 API 在查之前就拒绝了该路径（已实测：同一文件裸路径失败、加前缀读到 7.23MB）。

    **返回值绝不可写进 manifest/hash_cache，也绝不可参与路径比较或 relative_to。**
    因为 os.path.abspath 会保留前缀，一旦混入会造成：
      (a) 缓存键与裸路径不相等 → 每次全量重算哈希（静默的性能退化）；
      (b) Path(带前缀).relative_to(input_dir) 抛 ValueError → source_relative 崩。
    故本函数只在调用系统 API 的那一瞬间使用，身份/存储一律用裸路径。

    注意：本前缀绕不过 NTFS 对**单个路径段**的 255 字符上限（那是文件系统本身的
    限制，非 Win32 规范化所致）。单个文件名超 255 仍会失败。

    非 Windows 平台原样返回（那里没有这个限制）。
    """
    s = os.path.abspath(str(path))
    if os.name != "nt" or s.startswith("\\\\?\\"):
        return s
    return "\\\\?\\" + s


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(_longp(path), "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _retry_after(error: HTTPError) -> float | None:
    value = error.headers.get("Retry-After") if error.headers else None
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, email.utils.parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return None


def _backoff(attempt: int, base: float, maximum: float, server_delay: float | None = None) -> float:
    delay = server_delay if server_delay is not None else base * (2**attempt)
    return min(maximum, max(base, delay)) * random.uniform(1.0, 1.15)


def _check_business_code(response: Any, stage: str) -> Any:
    """校验官方 HTTP 200 响应中的业务状态码 code；成功返回原响应。"""
    if not isinstance(response, dict) or "code" not in response:
        return response
    code = str(response.get("code"))
    if code in {"0", "200"}:
        return response
    message = str(response.get("msg") or response.get("message") or "")
    detail = f"{stage} 业务错误 code={code} {message}".strip()
    if code in FATAL_BUSINESS_CODES:
        raise FatalApiError(detail)
    if code in DAILY_LIMIT_CODES:
        raise DailyLimitReached(detail)
    if code in RETRYABLE_BUSINESS_CODES:
        raise RecoverableBatchError(detail)
    raise RuntimeError(detail)


def api_json(method: str, url: str, token: str, payload: Any = None, *, retries: int = 6,
             retry_base: float = 5, retry_max: float = 60, timeout: float = 120,
             stage: str = "API 请求") -> Any:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    for attempt in range(retries + 1):
        try:
            request = Request(url, data=data, headers=headers, method=method)
            with urlopen(request, timeout=timeout) as response:
                parsed = json.loads(response.read().decode("utf-8"))
            try:
                return _check_business_code(parsed, stage)
            except RecoverableBatchError as error:
                if attempt >= retries:
                    raise
                delay = _backoff(attempt, retry_base, retry_max)
                log(f"{stage} {error}，{delay:.0f} 秒后重试 {attempt + 1}/{retries}")
                time.sleep(delay)
                continue
        except HTTPError as error:
            if error.code in (401, 403):
                raise
            if error.code not in RETRYABLE_HTTP_CODES or attempt >= retries:
                if error.code == 429:
                    raise RateLimitExhausted(f"{stage}持续被限流（HTTP 429）") from error
                raise
            delay = _backoff(attempt, retry_base, retry_max, _retry_after(error))
            log(f"{stage} HTTP {error.code}，{delay:.0f} 秒后重试 {attempt + 1}/{retries}")
            time.sleep(delay)
        except (URLError, TimeoutError, OSError) as error:
            if attempt >= retries:
                raise RuntimeError(f"{stage}网络错误，重试已用尽: {error}") from error
            delay = _backoff(attempt, retry_base, retry_max)
            log(f"{stage}网络异常，{delay:.0f} 秒后重试 {attempt + 1}/{retries}: {error}")
            time.sleep(delay)
    raise AssertionError("unreachable")


def put_file(url: str, path: Path, *, retries: int = 3, retry_base: float = 3,
             retry_max: float = 30) -> None:
    for attempt in range(retries + 1):
        try:
            with open(_longp(path), "rb") as stream:   # 前缀：容纳超长路径
                response = requests.put(url, data=stream, timeout=(30, 900))
            if response.status_code < 400:
                return
            if response.status_code not in RETRYABLE_HTTP_CODES or attempt >= retries:
                raise RuntimeError(f"上传 HTTP {response.status_code}: {response.text[:500]}")
            try:
                server_delay = float(response.headers.get("Retry-After", ""))
            except ValueError:
                server_delay = None
            delay = _backoff(attempt, retry_base, retry_max, server_delay)
            log(f"上传 {path.name} 返回 HTTP {response.status_code}，{delay:.0f} 秒后重试")
            time.sleep(delay)
        except requests.RequestException as error:
            if attempt >= retries:
                raise RuntimeError(f"上传 {path.name} 失败: {error}") from error
            delay = _backoff(attempt, retry_base, retry_max)
            log(f"上传 {path.name} 网络异常，{delay:.0f} 秒后重试")
            time.sleep(delay)


def download(url: str, output: Path, *, retries: int = 3, retry_base: float = 3,
             retry_max: float = 30) -> None:
    temp = output.with_suffix(output.suffix + ".part")
    for attempt in range(retries + 1):
        try:
            with requests.get(url, stream=True, timeout=(30, 900)) as response:
                response.raise_for_status()
                with temp.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            stream.write(chunk)
            os.replace(temp, output)
            return
        except (requests.RequestException, OSError) as error:
            if attempt >= retries:
                raise RuntimeError(f"下载结果失败: {error}") from error
            delay = _backoff(attempt, retry_base, retry_max)
            log(f"下载结果网络异常，{delay:.0f} 秒后重试")
            time.sleep(delay)


def extract_md_images(zip_path: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            name = info.filename.replace("\\", "/")
            parts = [part for part in name.split("/") if part]
            if not parts or name.endswith("/") or any(part in {".", ".."} for part in parts):
                continue
            if not (name.lower().endswith(".md") or any(part.lower() == "images" for part in parts)):
                continue
            target = output.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest, 1024 * 1024)


def result_items(response: Any) -> list[dict[str, Any]]:
    if not isinstance(response, dict):
        return response if isinstance(response, list) else []
    data = response.get("data", response)
    items = (data.get("extract_result") or data.get("results") or []) if isinstance(data, dict) else data
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def load_manifest(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_hash: dict[str, dict[str, Any]] = {}
    by_source: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return by_hash, by_source
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("sha256"):
                by_hash[record["sha256"]] = record
            if record.get("source"):
                by_source[os.path.normcase(os.path.abspath(record["source"]))] = record
    return by_hash, by_source


def append_records(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def scan_pdfs(input_dir: Path, recursive: bool, by_source: dict[str, dict[str, Any]],
              cache_path: Path | None = None) \
        -> tuple[list[tuple[Path, str, os.stat_result]], int]:
    pdfs = sorted(input_dir.rglob("*.pdf") if recursive else input_dir.glob("*.pdf"))
    unique: dict[str, tuple[Path, str, os.stat_result]] = {}
    cache_stream = cache_path.open("a", encoding="utf-8") if cache_path is not None else None
    unreadable: list[tuple[Path, str]] = []
    try:
        for index, path in enumerate(pdfs, 1):
            # 单个文件读不到（超长路径/被占用/权限/坏道）只跳过它，绝不中断整批扫描。
            # 此前无此保护：任一文件抛 OSError 会冒到 main() 之外，导致所有健康文件
            # 一起没跑，且 update_all 的汇总显示「新增 0」，看起来像"没有新研报"。
            try:
                stat = os.stat(_longp(path))          # 前缀：容纳超长路径
                cached = by_source.get(os.path.normcase(os.path.abspath(path)))
                if (cached and cached.get("source_size") == stat.st_size
                        and cached.get("source_mtime_ns") == stat.st_mtime_ns and cached.get("sha256")):
                    digest = str(cached["sha256"])
                else:
                    digest = sha256(path)
                    if cache_stream is not None:
                        # 存裸路径：带前缀会让下次的缓存键失配 → 静默全量重算。
                        cache_stream.write(json.dumps({"source": str(path), "source_size": stat.st_size,
                                                       "source_mtime_ns": stat.st_mtime_ns,
                                                       "sha256": digest}, ensure_ascii=False) + "\n")
            except OSError as error:
                unreadable.append((path, f"{type(error).__name__}: {error}"))
                log(f"跳过读不到的文件 {path.name}（{type(error).__name__}: {error}）")
                continue
            unique.setdefault(digest, (path, digest, stat))
            if index % 100 == 0 or index == len(pdfs):
                log(f"扫描校验 {index}/{len(pdfs)}")
    finally:
        if cache_stream is not None:
            cache_stream.close()
    if unreadable:
        log(f"本次扫描跳过 {len(unreadable)} 个读不到的文件（不影响其余 "
            f"{len(pdfs) - len(unreadable)} 个）：")
        for path, why in unreadable[:10]:
            log(f"    {path}  →  {why}")
        if len(unreadable) > 10:
            log(f"    …其余 {len(unreadable) - 10} 个")
    return list(unique.values()), len(pdfs) - len(unique)


def _match_items(records: list[dict[str, Any]], items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id = {str(item.get("data_id")): item for item in items if item.get("data_id") is not None}
    by_name = {str(item.get("file_name") or item.get("name")): item for item in items}
    matched: dict[str, dict[str, Any]] = {}
    for record in records:
        item = by_id.get(record["doc_id"]) or by_name.get(Path(record["source"]).name)
        if item is not None:
            matched[record["doc_id"]] = item
    return matched


def _api_options(config: dict[str, Any]) -> dict[str, Any]:
    return {"retries": int(config.get("api_retries", 6)),
            "retry_base": float(config.get("retry_base_seconds", 5)),
            "retry_max": float(config.get("retry_max_seconds", 60))}


def poll_batch(base: str, token: str, batch_id: str, records: list[dict[str, Any]],
               config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    max_polls = int(config.get("max_polls", 180))
    for attempt in range(1, max_polls + 1):
        response = api_json("GET", base + f"/extract-results/batch/{batch_id}", token,
                            stage=f"轮询批次 {batch_id}", **_api_options(config))
        matched = _match_items(records, result_items(response))
        states = Counter(str(item.get("state", "unknown")).lower() for item in matched.values())
        summary = " ".join(f"{key}={value}" for key, value in sorted(states.items())) or "尚无结果"
        log(f"批次 {batch_id} 轮询 {attempt}/{max_polls}: 已返回 {len(matched)}/{len(records)}，{summary}")
        if len(matched) == len(records) and all(
                str(item.get("state", "")).lower() in TERMINAL_STATES for item in matched.values()):
            return matched
        if attempt < max_polls:
            time.sleep(float(config.get("poll_seconds", 10)))
    raise TimeoutError(f"批次 {batch_id} 轮询超时")


def materialize_result(record: dict[str, Any], item: dict[str, Any], root: Path,
                       config: dict[str, Any]) -> dict[str, Any]:
    state = str(item.get("state", "")).lower()
    if state in {"failed", "error", "cancelled", "canceled"}:
        raise RuntimeError(str(item.get("err_msg") or item.get("message") or f"MinerU 状态: {state}"))
    result_url = item.get("full_zip_url") or item.get("zip_url") or item.get("download_url")
    if not result_url:
        raise RuntimeError(str(item.get("err_msg") or f"响应中没有结果下载地址: {item}"))
    raw = root / "raw_downloads" / f"{record['doc_id']}.zip"
    output = root / "canonical" / record["doc_id"]
    download(str(result_url), raw, retries=int(config.get("transfer_retries", 3)),
             retry_base=float(config.get("retry_base_seconds", 5)),
             retry_max=float(config.get("retry_max_seconds", 60)))
    extract_md_images(raw, output)
    record.update(status="success", result=str(output), raw_zip=str(raw), finished_at=now())
    return record


def finish_batch(base: str, token: str, batch_id: str, records: list[dict[str, Any]], root: Path,
                 manifest_path: Path, config: dict[str, Any]) -> tuple[int, int]:
    items = poll_batch(base, token, batch_id, records, config)
    successes = failures = completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(config.get("download_workers", 4)))) as pool:
        futures = {pool.submit(materialize_result, record, items[record["doc_id"]], root, config): record
                   for record in records}
        for future in concurrent.futures.as_completed(futures):
            record = futures[future]
            try:
                future.result()
                successes += 1
            except Exception as error:
                record.update(status="failed", error=f"处理结果: {error}", finished_at=now())
                failures += 1
            append_records(manifest_path, [record])
            completed += 1
            log(f"批次 {batch_id} 下载/解压 {completed}/{len(records)}（成功 {successes}，失败 {failures}）")
    return successes, failures


def _new_records(batch: list[tuple[Path, str, os.stat_result]], input_dir: Path) -> list[dict[str, Any]]:
    return [{"doc_id": digest[:32], "source": str(path),
             "source_relative": str(path.relative_to(input_dir)), "source_size": stat.st_size,
             "source_mtime_ns": stat.st_mtime_ns, "sha256": digest, "started_at": now(),
             "status": "submitting"} for path, digest, stat in batch]


def submit_batch(base: str, token: str, batch: list[tuple[Path, str, os.stat_result]], root: Path,
                 input_dir: Path, manifest_path: Path, config: dict[str, Any]) -> tuple[int, int]:
    records = _new_records(batch, input_dir)
    payload = {"files": [{"name": path.name, "is_ocr": bool(config.get("is_ocr", False)),
                           "data_id": record["doc_id"]}
                          for (path, _, _), record in zip(batch, records)],
               "model_version": config.get("model_version", "pipeline")}
    created = api_json("POST", base + "/file-urls/batch", token, payload, stage="提交批次",
                       **_api_options(config))
    data = created.get("data", created)
    batch_id, urls = data.get("batch_id"), data.get("file_urls") or []
    if not batch_id or len(urls) != len(records):
        raise RuntimeError(f"批量提交响应异常: batch_id={batch_id}, file_urls={len(urls)}")
    # 官方在文件上传成功后自动开始解析，因此先把 batch_id 落盘，
    # 即使随后有个别文件上传失败也不会丢失已提交任务，续跑时可从清单恢复。
    for record in records:
        record.update(status="submitted", batch_id=batch_id, uploaded_at=now())
    append_records(manifest_path, records)
    log(f"批次 {batch_id} 已创建并落盘，开始并行上传 {len(records)} 个文件")
    completed = 0
    upload_failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(config.get("upload_workers", 6)))) as pool:
        futures = {pool.submit(put_file, str(url), path,
                               retries=int(config.get("transfer_retries", 3)),
                               retry_base=float(config.get("retry_base_seconds", 5)),
                               retry_max=float(config.get("retry_max_seconds", 60))): record
                   for url, (path, _, _), record in zip(urls, batch, records)}
        failed_records: list[dict[str, Any]] = []
        for future in concurrent.futures.as_completed(futures):
            record = futures[future]
            completed += 1
            try:
                future.result()
            except Exception as error:
                upload_failures += 1
                failed_records.append(record)
                record.update(status="failed", error=f"上传: {error}", finished_at=now())
                log(f"批次 {batch_id} 上传失败 {record['source_relative']}: {error}")
            log(f"批次 {batch_id} 上传 {completed}/{len(records)}（失败 {upload_failures}）")
    # 上传失败的文件 MinerU 从未收到，会永远停留在 waiting-file，
    # 立即标记失败并排除出轮询集合，避免拖垮整批轮询超时。
    uploaded = [record for record in records if record.get("status") == "submitted"]
    if failed_records:
        append_records(manifest_path, failed_records)
        log(f"批次 {batch_id} 有 {upload_failures} 个文件上传失败，已标记失败；"
            f"其余 {len(uploaded)} 个文件已自动进入解析。")
    if not uploaded:
        log(f"批次 {batch_id} 无文件成功上传，跳过轮询。")
        return 0, upload_failures
    log(f"批次 {batch_id} 上传完成，任务已写入清单，可安全续跑")
    try:
        ok, failed = finish_batch(base, token, str(batch_id), uploaded, root, manifest_path, config)
        return ok, failed + upload_failures
    except (RateLimitExhausted, DailyLimitReached, FatalApiError, KeyboardInterrupt):
        raise
    except HTTPError as error:
        if error.code in (401, 403):
            raise
        raise RecoverableBatchError(f"批次 {batch_id} 已提交，稍后可续跑: HTTP {error.code}") from error
    except Exception as error:
        raise RecoverableBatchError(f"批次 {batch_id} 已提交，稍后可续跑: {error}") from error


def _mark_failed(records: list[dict[str, Any]], manifest: Path, stage: str, error: Exception) -> int:
    for record in records:
        record.update(status="failed", error=f"{stage}: {error}", finished_at=now())
    append_records(manifest, records)
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="可靠、可续跑的 MinerU PDF 批处理")
    parser.add_argument("--config", default="mineru_pipeline.json")
    parser.add_argument("--once", action="store_true", help="只处理或恢复一个批次")
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
    manifest = root / "manifest" / "manifest.jsonl"
    latest_by_hash, latest_by_source = load_manifest(manifest)
    hash_cache = root / "manifest" / "hash_cache.jsonl"
    _, cached_sources = load_manifest(hash_cache)
    latest_by_source.update(cached_sources)
    log("开始扫描 PDF；首次运行会计算哈希，后续运行将复用文件大小和修改时间缓存")
    scanned, duplicates = scan_pdfs(input_dir, bool(config.get("recursive", True)), latest_by_source, hash_cache)
    active_hashes = {digest for _, digest, _ in scanned
                     if latest_by_hash.get(digest, {}).get("status") == "submitted"
                     and latest_by_hash.get(digest, {}).get("batch_id")}
    pending = [entry for entry in scanned
               if latest_by_hash.get(entry[1], {}).get("status") != "success" and entry[1] not in active_hashes]
    successes = sum(latest_by_hash.get(digest, {}).get("status") == "success" for _, digest, _ in scanned)
    log(f"发现 {len(scanned) + duplicates} 个 PDF（内容重复 {duplicates} 个），已成功 {successes} 个，"
        f"待处理 {len(pending)} 个，待恢复 {len(active_hashes)} 个")

    total_ok = total_failed = batches_done = 0
    active_batches: dict[str, list[dict[str, Any]]] = {}
    for digest in active_hashes:
        record = latest_by_hash[digest]
        active_batches.setdefault(str(record["batch_id"]), []).append(record.copy())
    try:
        for batch_id, records in active_batches.items():
            log(f"恢复未完成批次 {batch_id}，共 {len(records)} 个文件")
            try:
                ok, failed = finish_batch(base, token, batch_id, records, root, manifest, config)
            except (RateLimitExhausted, DailyLimitReached, FatalApiError, KeyboardInterrupt):
                raise
            except HTTPError as error:
                if error.code in (401, 403):
                    raise SystemExit(f"HTTP {error.code}，请检查 Token 和 MinerU 权限") from error
                raise RecoverableBatchError(f"恢复批次 {batch_id} 暂时失败，状态仍已保留: HTTP {error.code}") from error
            except Exception as error:
                raise RecoverableBatchError(f"恢复批次 {batch_id} 暂时失败，状态仍已保留: {error}") from error
            total_ok, total_failed, batches_done = total_ok + ok, total_failed + failed, batches_done + 1
            if args.once:
                pending = []
                break
        batch_size = max(1, min(int(config.get("batch_size", 50)), 50))
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            log(f"开始新批次 {offset // batch_size + 1}/{(len(pending) + batch_size - 1) // batch_size}，"
                f"文件 {offset + 1}-{offset + len(batch)}/{len(pending)}")
            try:
                ok, failed = submit_batch(base, token, batch, root, input_dir, manifest, config)
            except RateLimitExhausted:
                log("MinerU 持续返回 HTTP 429。已停止提交后续批次；请稍后重新运行，程序会自动续跑。")
                break
            except DailyLimitReached as error:
                log(f"{error}。已达每日任务上限，停止提交后续批次；请次日再运行，程序会自动续跑。")
                break
            except FatalApiError as error:
                raise SystemExit(f"{error}。Token 无效或已过期，请更新配置后重试。") from error
            except RecoverableBatchError as error:
                log(str(error))
                break
            except HTTPError as error:
                if error.code in (401, 403):
                    raise SystemExit(f"HTTP {error.code}，请检查 Token 和 MinerU 权限") from error
                ok, failed = 0, _mark_failed(_new_records(batch, input_dir), manifest, "批次处理", error)
                log(f"本批次失败: HTTP {error.code} {error.reason}")
            except KeyboardInterrupt:
                raise
            except Exception as error:
                ok, failed = 0, _mark_failed(_new_records(batch, input_dir), manifest, "批次处理", error)
                log(f"本批次失败: {error}")
            total_ok, total_failed, batches_done = total_ok + ok, total_failed + failed, batches_done + 1
            log(f"批次完成；本次累计成功 {total_ok}，失败 {total_failed}，"
                f"进度 {min(offset + len(batch), len(pending))}/{len(pending)}")
            if args.once:
                break
            cooldown = float(config.get("batch_delay_seconds", 5))
            if cooldown > 0 and offset + len(batch) < len(pending):
                log(f"批次冷却 {cooldown:g} 秒，避免触发 API 限流")
                time.sleep(cooldown)
    except RateLimitExhausted:
        log("恢复批次时持续被限流；状态已保留，请稍后重新运行。")
    except DailyLimitReached as error:
        log(f"{error}。已达每日任务上限；状态已保留，请次日再运行，程序会自动续跑。")
    except FatalApiError as error:
        raise SystemExit(f"{error}。Token 无效或已过期，请更新配置后重试。") from error
    except RecoverableBatchError as error:
        log(str(error))
    except KeyboardInterrupt:
        log("收到中断信号，已安全停止；下次运行会从清单恢复。")
    log(f"本次结束：完成批次 {batches_done}，成功 {total_ok}，失败 {total_failed}")


if __name__ == "__main__":
    main()
