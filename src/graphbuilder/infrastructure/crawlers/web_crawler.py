"""
Web Crawler - Domain-agnostic async crawler.

Allowed domains, URL limits, request delay, and user-agent are all driven by
``CrawlerConfiguration`` so the crawler has no hard-coded domain knowledge.

Typical usage::

    config = get_config()
    crawler = WebCrawler(config.crawler)
    pages = await crawler.crawl(start_urls=["https://example.com/"])
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

from ..config.settings import CrawlerConfiguration


class WebCrawler:
    """
    Domain-agnostic async web crawler driven by ``CrawlerConfiguration``.

    Parameters
    ----------
    config:
        ``CrawlerConfiguration`` instance.  All filtering (allowed domains,
        URL limit, delay, user-agent) is read from this object.
    visited_file:
        Optional path to a file that persists visited URLs across runs.
        Pass ``None`` (default) to keep state in-memory only.
    """

    def __init__(
        self,
        config: CrawlerConfiguration,
        visited_file: Optional[str] = None,
    ) -> None:
        self.config = config
        self.visited_file = visited_file
        self.logger = logging.getLogger(self.__class__.__name__)
        self._visited: Set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def crawl(
        self,
        start_urls: List[str],
        extra_allowed_domains: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Crawl from *start_urls* up to ``config.max_urls`` pages.

        Domain filtering uses ``config.allowed_domains``.  When that list is
        empty *and* ``extra_allowed_domains`` is also empty, ALL domains found
        via link extraction are followed (open crawl).  Pass either to
        restrict the crawl.

        Parameters
        ----------
        start_urls:
            Seed URLs.
        extra_allowed_domains:
            Additional domains to allow on top of ``config.allowed_domains``.
            Useful for ad-hoc CLI invocations without changing config files.

        Returns
        -------
        dict
            Mapping ``url → page_text`` for every successfully fetched page.
        """
        allowed = set(
            d.strip().lower()
            for d in (self.config.allowed_domains or [])
            if d.strip()
        )
        if extra_allowed_domains:
            allowed.update(d.strip().lower() for d in extra_allowed_domains if d.strip())

        blocked = set(
            d.strip().lower()
            for d in (self.config.blocked_domains or [])
            if d.strip()
        )

        if self.visited_file:
            self._visited = _load_visited(self.visited_file)
        else:
            self._visited = set()

        pages: Dict[str, str] = {}
        queue: List[str] = list(start_urls)

        headers = {"User-Agent": self.config.user_agent}
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)

        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            while queue and len(self._visited) < self.config.max_urls:
                url = queue.pop(0)

                if url in self._visited:
                    continue

                if not self._is_allowed(url, allowed, blocked):
                    self.logger.debug("Skipping (domain filter): %s", url)
                    continue

                text = await self._fetch_text(session, url)
                if text is None:
                    continue

                self._visited.add(url)
                pages[url] = text
                self.logger.info(
                    "Crawled %s  (%d/%d)", url, len(self._visited), self.config.max_urls
                )

                new_links = _extract_links(text, url)
                for link in new_links:
                    if link not in self._visited and link not in queue:
                        queue.append(link)

                await asyncio.sleep(self.config.request_delay)

        if self.visited_file:
            _save_visited(self.visited_file, self._visited)

        self.logger.info("Crawl finished. %d pages collected.", len(pages))
        return pages

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_allowed(self, url: str, allowed: Set[str], blocked: Set[str]) -> bool:
        """Return True if *url* passes the domain allow/block filters."""
        host = urlparse(url).hostname or ""
        host = host.lower().lstrip("www.")

        if any(host == b or host.endswith("." + b) for b in blocked):
            return False

        if not allowed:
            return True  # open crawl

        return any(host == a or host.endswith("." + a) for a in allowed)

    async def _fetch_text(
        self, session: aiohttp.ClientSession, url: str
    ) -> Optional[str]:
        """Fetch *url* and return its text, or ``None`` on error."""
        try:
            async with session.get(
                url,
                max_redirects=self.config.max_redirects,
                allow_redirects=True,
            ) as response:
                content_type = response.headers.get("Content-Type", "")
                if not any(
                    ct in content_type
                    for ct in self.config.allowed_content_types
                ):
                    self.logger.debug(
                        "Skipping unsupported content-type '%s': %s", content_type, url
                    )
                    return None

                content_length = int(response.headers.get("Content-Length", 0))
                if content_length > self.config.max_file_size:
                    self.logger.debug(
                        "Skipping oversized response (%d bytes): %s",
                        content_length,
                        url,
                    )
                    return None

                return await response.text(errors="replace")

        except asyncio.TimeoutError:
            self.logger.warning("Timeout fetching: %s", url)
        except aiohttp.ClientError as exc:
            self.logger.error("HTTP error fetching %s: %s", url, exc)
        except Exception as exc:
            self.logger.error("Unexpected error fetching %s: %s", url, exc)

        return None


# ------------------------------------------------------------------
# Module-level helpers (no domain knowledge)
# ------------------------------------------------------------------


def _extract_links(html: str, base_url: str) -> List[str]:
    """Parse *html* and return all absolute href links."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].strip()
            if href and not href.startswith(("#", "javascript:", "mailto:")):
                links.append(urljoin(base_url, href))
        return links
    except Exception:
        return []


def _load_visited(path: str) -> Set[str]:
    """Load visited URLs from *path*; return empty set if file absent."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return set(line.strip() for line in fh if line.strip())
    return set()


def _save_visited(path: str, visited: Set[str]) -> None:
    """Persist *visited* URLs to *path*."""
    with open(path, "w", encoding="utf-8") as fh:
        for url in sorted(visited):
            fh.write(url + "\n")
