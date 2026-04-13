"""URL-based cache for web crawler results.

Stores crawled page content on disk (``data/cache/``) keyed by URL hash.
When a URL has been crawled before, the cached content is returned
immediately, skipping the HTTP request.

Cache entries include metadata (URL, timestamp, content length) so stale
entries can be identified and purged if needed.
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Default cache directory relative to project root
_DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "data", "cache"
)


@dataclass
class CacheEntry:
    """A single cached page."""

    url: str
    content: str
    fetched_at: float  # Unix timestamp
    content_length: int = 0
    content_type: str = "text/html"

    def __post_init__(self):
        if not self.content_length:
            self.content_length = len(self.content)


class CrawlerCache:
    """Disk-backed URL cache for the web crawler.

    Parameters
    ----------
    cache_dir:
        Directory to store cache files.  Created if it doesn't exist.
    max_age_seconds:
        Maximum age (seconds) before a cached entry is considered stale.
        ``0`` means entries never expire.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_age_seconds: int = 0,
    ) -> None:
        self.cache_dir = os.path.abspath(cache_dir or _DEFAULT_CACHE_DIR)
        self.max_age_seconds = max_age_seconds
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info("CrawlerCache initialised at %s", self.cache_dir)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def url_hash(url: str) -> str:
        """Deterministic filename-safe hash of a URL."""
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def _path_for(self, url: str) -> str:
        return os.path.join(self.cache_dir, self.url_hash(url) + ".json")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get(self, url: str) -> Optional[CacheEntry]:
        """Return cached content for *url*, or ``None`` on cache miss."""
        path = self._path_for(url)
        if not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupt cache entry for %s: %s", url, exc)
            return None

        entry = CacheEntry(**data)

        # Staleness check
        if self.max_age_seconds > 0:
            age = time.time() - entry.fetched_at
            if age > self.max_age_seconds:
                logger.debug("Cache entry stale (%.0fs old): %s", age, url)
                return None

        return entry

    def put(self, url: str, content: str, content_type: str = "text/html") -> CacheEntry:
        """Store *content* for *url*.  Returns the created ``CacheEntry``."""
        entry = CacheEntry(
            url=url,
            content=content,
            fetched_at=time.time(),
            content_type=content_type,
        )
        path = self._path_for(url)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(entry), fh, ensure_ascii=False)
        logger.debug("Cached %d chars for %s", entry.content_length, url)
        return entry

    def has(self, url: str) -> bool:
        """Return ``True`` if url is in cache and not stale."""
        return self.get(url) is not None

    def stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        files = list(Path(self.cache_dir).glob("*.json"))
        total_bytes = sum(f.stat().st_size for f in files)
        return {"entries": len(files), "total_bytes": total_bytes}

    def clear(self) -> int:
        """Remove all cache files.  Returns number deleted."""
        files = list(Path(self.cache_dir).glob("*.json"))
        for f in files:
            f.unlink(missing_ok=True)
        return len(files)
