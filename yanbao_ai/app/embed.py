"""Embed：OpenAI 兼容 embedding 客户端，批量 + 本地缓存 + 退避重试。

设计要点：
- 模型 text-embedding-3-large @ dimensions=1024（可配），OpenAI 官方或兼容端点。
- 本地缓存：按 (model, dimensions, sha256(text)) 存向量，重跑/断点续跑不重复付费。
- 批量请求：每批默认 64 条，控制 token 上限。
- 指数退避重试：429/5xx/网络错误自动退避重试，带并发上限（此处串行批次，稳）。
- 不打印 key 值。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .config import EmbedConfig

DEFAULT_BATCH_SIZE = 20  # qwen 兼容端点硬上限 20；OpenAI 可更大，取安全默认
MAX_RETRIES = 6
BASE_DELAY = 2.0  # 秒，指数退避基数


def _text_key(text: str, model: str, dim: int) -> str:
    h = hashlib.sha256(f"{model}\x00{dim}\x00{text}".encode("utf-8")).hexdigest()
    return h


_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS embed_cache (
    key       TEXT PRIMARY KEY,   -- sha256(model|dim|text)
    model     TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    vector    BLOB NOT NULL,      -- float32 小端序原始字节
    created_at TEXT
);
"""


class EmbedError(RuntimeError):
    pass


@dataclass
class EmbedStats:
    requested: int = 0      # 请求嵌入的文本条数
    from_cache: int = 0     # 命中缓存
    from_api: int = 0       # 实际调 API
    api_calls: int = 0      # API 请求次数（批次数）


class Embedder:
    """封装 embedding：优先查缓存，未命中批量调 API 并回写缓存。"""

    def __init__(self, cfg: EmbedConfig, cache_db: Path):
        if not cfg.api_key:
            raise EmbedError(
                "embedding 需要 API key。请在 config.toml 的 [embed].api_key_env "
                "指定的环境变量里设置 OpenAI key。"
            )
        self.cfg = cfg
        self.batch_size = getattr(cfg, "batch_size", None) or DEFAULT_BATCH_SIZE
        self.cache_db = cache_db
        self._client = None
        cache_db.parent.mkdir(parents=True, exist_ok=True)
        self._cache = sqlite3.connect(cache_db)
        self._cache.executescript(_CACHE_SCHEMA)

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key)
        return self._client

    # ---- 缓存读写 ----
    def _cache_get(self, key: str) -> list[float] | None:
        row = self._cache.execute(
            "SELECT vector FROM embed_cache WHERE key=?", (key,)
        ).fetchone()
        if row is None:
            return None
        return _bytes_to_vec(row[0])

    def _cache_put(self, key: str, vec: list[float]) -> None:
        self._cache.execute(
            "INSERT OR REPLACE INTO embed_cache(key, model, dim, vector, created_at) "
            "VALUES (?,?,?,?,datetime('now'))",
            (key, self.cfg.model, self.cfg.dimensions, _vec_to_bytes(vec)),
        )

    # ---- API 调用（带退避重试）----
    def _embed_api(self, texts: list[str]) -> list[list[float]]:
        last_err: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.client.embeddings.create(
                    model=self.cfg.model,
                    input=texts,
                    dimensions=self.cfg.dimensions,
                )
                return [d.embedding for d in resp.data]
            except Exception as exc:  # noqa: BLE001 - 需分辨瞬时/永久错误
                last_err = exc
                # 4xx（除 429）为永久错误：参数非法、鉴权失败、账户欠费（Arrearage）等，
                # 重试只会白白退避 ~62s 还掩盖真因 —— 立刻抛出，让调用方看到原始错误。
                status = getattr(exc, "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    raise EmbedError(f"embedding API 永久错误（HTTP {status}，不重试）：{exc}") from exc
                if attempt == MAX_RETRIES - 1:
                    break
                delay = BASE_DELAY * (2 ** attempt)
                time.sleep(delay)
        raise EmbedError(f"embedding API 连续 {MAX_RETRIES} 次失败：{last_err}") from last_err

    def embed(self, texts: list[str], stats: EmbedStats | None = None) -> list[list[float]]:
        """嵌入一批文本，保序返回向量。命中缓存的不重复调 API。"""
        st = stats or EmbedStats()
        st.requested += len(texts)
        keys = [_text_key(t, self.cfg.model, self.cfg.dimensions) for t in texts]
        results: list[list[float] | None] = [None] * len(texts)

        # 1) 查缓存
        misses: list[int] = []
        for i, k in enumerate(keys):
            cached = self._cache_get(k)
            if cached is not None:
                results[i] = cached
                st.from_cache += 1
            else:
                misses.append(i)

        # 2) 未命中的按批调 API
        for start in range(0, len(misses), self.batch_size):
            batch_idx = misses[start : start + self.batch_size]
            batch_texts = [texts[i] for i in batch_idx]
            vecs = self._embed_api(batch_texts)
            st.api_calls += 1
            st.from_api += len(batch_texts)
            for i, vec in zip(batch_idx, vecs):
                results[i] = vec
                self._cache_put(keys[i], vec)
            self._cache.commit()

        return [r for r in results if r is not None]  # 保序（无 None，misses 已全填）

    def close(self):
        self._cache.close()


def _vec_to_bytes(vec: list[float]) -> bytes:
    import struct

    return struct.pack(f"<{len(vec)}f", *vec)


def _bytes_to_vec(b: bytes) -> list[float]:
    import struct

    n = len(b) // 4
    return list(struct.unpack(f"<{n}f", b))
