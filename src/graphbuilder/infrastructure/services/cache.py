"""LRU caches for LLM dedup decisions and sentence embeddings.

Both caches are bounded, process-local, and asyncio-safe. Hits are
reported to ``PipelineMetrics`` so the dashboard can show how often
the pipeline avoided an LLM round-trip or a sentence-transformer call.

Why bother:
- Within one document, the same (entity-name, candidate-name) pair
  often surfaces across multiple chunks. The LLM dedup answer doesn't
  change between chunks, so caching it eliminates duplicate calls.
- Embeddings on the same string (entity name) recur constantly during
  vector pre-filtering. Caching avoids re-encoding.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import OrderedDict
from typing import Any, Optional


class _LRU:
    """Tiny async-safe LRU."""

    def __init__(self, max_size: int) -> None:
        self._max = max_size
        self._d: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key not in self._d:
                return None
            self._d.move_to_end(key)
            return self._d[key]

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                self._d[key] = value
                return
            self._d[key] = value
            if len(self._d) > self._max:
                self._d.popitem(last=False)

    def size(self) -> int:
        return len(self._d)

    def clear(self) -> None:
        self._d.clear()


def _hash(*parts: Any) -> str:
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, (dict, list)):
            h.update(json.dumps(p, sort_keys=True, default=str).encode("utf-8"))
        else:
            h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class LLMDedupCache:
    """Cache the boolean dedup decision for ``(new, candidates)`` pairs."""

    def __init__(self, max_size: int = 2048) -> None:
        self._lru = _LRU(max_size)

    @staticmethod
    def key_for_entities(new_entities: list[dict], existing: list[dict]) -> str:
        """Stable key — depends only on names + types, not descriptions."""
        new_norm = sorted(
            (e.get("name", "").lower(), e.get("type", "")) for e in new_entities
        )
        ex_norm = sorted(
            (e.get("name", "").lower(), e.get("type", "")) for e in existing
        )
        return _hash("ent_dedup", new_norm, ex_norm)

    @staticmethod
    def key_for_relationship(new_rel: dict, existing: list[dict]) -> str:
        new_norm = (
            (new_rel.get("source", "") or "").lower(),
            (new_rel.get("target", "") or "").lower(),
            new_rel.get("type", ""),
        )
        ex_norm = sorted(
            (
                (r.get("source", "") or "").lower(),
                (r.get("target", "") or "").lower(),
                r.get("type", ""),
            )
            for r in existing
        )
        return _hash("rel_dedup", new_norm, ex_norm)

    async def get(self, key: str) -> Optional[Any]:
        return await self._lru.get(key)

    async def set(self, key: str, value: Any) -> None:
        await self._lru.set(key, value)

    def size(self) -> int:
        return self._lru.size()


class EmbeddingCache:
    """Cache 384-d float vectors keyed by the input text."""

    def __init__(self, max_size: int = 4096) -> None:
        self._lru = _LRU(max_size)

    @staticmethod
    def key(text: str) -> str:
        return _hash("emb", text.strip().lower())

    async def get(self, text: str) -> Optional[list]:
        return await self._lru.get(self.key(text))

    async def set(self, text: str, vector: list) -> None:
        await self._lru.set(self.key(text), vector)

    def size(self) -> int:
        return self._lru.size()


_DEDUP: LLMDedupCache | None = None
_EMB: EmbeddingCache | None = None


def get_dedup_cache() -> LLMDedupCache:
    global _DEDUP
    if _DEDUP is None:
        _DEDUP = LLMDedupCache()
    return _DEDUP


def get_embedding_cache() -> EmbeddingCache:
    global _EMB
    if _EMB is None:
        _EMB = EmbeddingCache()
    return _EMB
