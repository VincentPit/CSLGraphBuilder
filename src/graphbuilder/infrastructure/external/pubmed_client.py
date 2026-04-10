"""
PubMed / NCBI E-utilities client.

Provides async access to the NCBI Entrez E-utilities API for searching
PubMed and fetching structured article metadata.

Public API::

    async with PubMedClient(email="you@example.com") as client:
        result = await client.fetch_articles(
            query="BRCA1 breast cancer",
            max_articles=50,
        )

No NCBI API key is required for low-volume use (<3 req/s).  Set
``api_key`` when you need higher throughput (up to 10 req/s).

Reference: https://www.ncbi.nlm.nih.gov/books/NBK25499/
"""

import asyncio
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover
    AIOHTTP_AVAILABLE = False


_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_POLITE_DELAY = 0.34   # default: stay under 3 req/s without an API key
_API_KEY_DELAY = 0.10  # 10 req/s with a valid API key


@dataclass
class PubMedArticle:
    """Structured representation of a single PubMed article."""

    pmid: str
    title: str
    abstract: str = ""
    authors: List[str] = field(default_factory=list)
    journal: str = ""
    publication_date: str = ""
    mesh_terms: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    doi: Optional[str] = None

    @property
    def full_text(self) -> str:
        """Title + abstract concatenated for downstream processing."""
        parts = [self.title]
        if self.abstract:
            parts.append(self.abstract)
        return "\n\n".join(parts)

    @classmethod
    def from_xml(cls, article_node: ET.Element) -> "PubMedArticle":
        """Parse a ``<PubmedArticle>`` XML element."""
        medline = article_node.find("MedlineCitation")
        if medline is None:
            raise ValueError("Missing MedlineCitation element")

        pmid = _text(medline.find("PMID")) or ""
        article = medline.find("Article") or ET.Element("Article")

        title = _text(article.find("ArticleTitle")) or "(no title)"
        abstract = _collect_abstract(article)
        journal = _text(
            article.find("Journal/Title")
            or article.find("Journal/ISOAbbreviation")
        ) or ""

        # Publication date
        pub_date = article.find("Journal/JournalIssue/PubDate")
        publication_date = _format_pubdate(pub_date) if pub_date is not None else ""

        # Authors
        authors: List[str] = []
        for author in article.findall("AuthorList/Author"):
            last = _text(author.find("LastName")) or ""
            fore = _text(author.find("ForeName")) or _text(author.find("Initials")) or ""
            name = f"{last}, {fore}".strip(", ")
            if name:
                authors.append(name)

        # MeSH
        mesh_terms: List[str] = [
            _text(mh.find("DescriptorName")) or ""
            for mh in medline.findall("MeshHeadingList/MeshHeading")
        ]
        mesh_terms = [m for m in mesh_terms if m]

        # Keywords
        keywords: List[str] = [
            _text(kw) or ""
            for kw in medline.findall("KeywordList/Keyword")
        ]
        keywords = [k for k in keywords if k]

        # DOI
        doi: Optional[str] = None
        for eloc in article.findall("ELocationID"):
            if eloc.get("EIdType") == "doi":
                doi = _text(eloc)
                break

        return cls(
            pmid=pmid,
            title=title,
            abstract=abstract,
            authors=authors,
            journal=journal,
            publication_date=publication_date,
            mesh_terms=mesh_terms,
            keywords=keywords,
            doi=doi,
        )


@dataclass
class PubMedFetchResult:
    """Outcome of a PubMed search + fetch operation."""

    query: str
    articles: List[PubMedArticle]
    total_hits: int
    fetched_count: int
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.errors


class PubMedClient:
    """
    Async client for the NCBI E-utilities (PubMed).

    Parameters
    ----------
    email:
        Required by NCBI policy; used to identify your application.
    api_key:
        Optional NCBI API key for higher rate limits (10 req/s vs 3 req/s).
    tool:
        Tool name sent to NCBI for identification.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        email: str,
        api_key: Optional[str] = None,
        tool: str = "GraphBuilder",
        timeout: int = 30,
    ) -> None:
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError(
                "aiohttp is required for PubMedClient. "
                "Install it with: pip install aiohttp"
            )
        if not email:
            raise ValueError("NCBI requires a valid email address.")
        self._email = email
        self._api_key = api_key
        self._tool = tool
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._delay = _API_KEY_DELAY if api_key else _POLITE_DELAY
        self._session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Context-manager helpers
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "PubMedClient":
        self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(self, query: str, max_ids: int = 500) -> List[str]:
        """
        Run an ESearch query and return a list of PubMed IDs (strings).
        """
        if self._session is None:
            raise RuntimeError("PubMedClient must be used as an async context manager.")

        params: Dict[str, Any] = {
            "db": "pubmed",
            "term": query,
            "retmax": max_ids,
            "retmode": "json",
            "tool": self._tool,
            "email": self._email,
        }
        if self._api_key:
            params["api_key"] = self._api_key

        url = f"{_ESEARCH}?{urlencode(params)}"
        async with self._session.get(url) as response:
            response.raise_for_status()
            data = await response.json(content_type=None)

        ids: List[str] = data.get("esearchresult", {}).get("idlist", [])
        total = int(data.get("esearchresult", {}).get("count", 0))
        self.logger.debug("ESearch '%s': %d total hits, returning %d IDs", query, total, len(ids))
        return ids

    async def fetch(self, pmids: List[str]) -> List[PubMedArticle]:
        """
        Fetch full article records for *pmids* via EFetch (XML).

        Batches into groups of 100 to stay within URL/server limits.
        """
        if self._session is None:
            raise RuntimeError("PubMedClient must be used as an async context manager.")
        if not pmids:
            return []

        articles: List[PubMedArticle] = []
        batch_size = 100

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i: i + batch_size]
            params: Dict[str, Any] = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "rettype": "abstract",
                "tool": self._tool,
                "email": self._email,
            }
            if self._api_key:
                params["api_key"] = self._api_key

            url = f"{_EFETCH}?{urlencode(params)}"
            try:
                async with self._session.get(url) as response:
                    response.raise_for_status()
                    xml_text = await response.text()

                root = ET.fromstring(xml_text)
                for article_node in root.findall("PubmedArticle"):
                    try:
                        articles.append(PubMedArticle.from_xml(article_node))
                    except Exception as exc:
                        self.logger.warning("Skipping malformed article: %s", exc)

            except aiohttp.ClientError as exc:
                self.logger.error("EFetch batch %d failed: %s", i // batch_size, exc)
                raise RuntimeError(f"PubMed EFetch failed: {exc}") from exc

            if i + batch_size < len(pmids):
                await asyncio.sleep(self._delay)

        return articles

    async def fetch_articles(
        self, query: str, max_articles: int = 100
    ) -> PubMedFetchResult:
        """
        Convenience method: search → fetch → return structured result.
        """
        errors: List[str] = []
        try:
            pmids = await self.search(query, max_ids=max_articles)
            total_hits = len(pmids)  # ESearch returns the count; use fetched as proxy
        except Exception as exc:
            return PubMedFetchResult(
                query=query,
                articles=[],
                total_hits=0,
                fetched_count=0,
                errors=[str(exc)],
            )

        try:
            articles = await self.fetch(pmids)
        except Exception as exc:
            errors.append(str(exc))
            articles = []

        return PubMedFetchResult(
            query=query,
            articles=articles,
            total_hits=total_hits,
            fetched_count=len(articles),
            errors=errors,
        )


# ------------------------------------------------------------------
# XML helpers
# ------------------------------------------------------------------


def _text(node: Optional[ET.Element]) -> Optional[str]:
    """Return stripped text of *node*, or ``None`` if node is None/empty."""
    if node is None:
        return None
    return (node.text or "").strip() or None


def _collect_abstract(article: ET.Element) -> str:
    """Collect all AbstractText elements (handles structured abstracts)."""
    parts: List[str] = []
    for ab in article.findall("Abstract/AbstractText"):
        label = ab.get("Label")
        text = (ab.text or "").strip()
        if text:
            parts.append(f"{label}: {text}" if label else text)
    return "\n".join(parts)


def _format_pubdate(pub_date: ET.Element) -> str:
    """Format a PubDate element as 'YYYY Mon DD' or 'YYYY'."""
    year  = _text(pub_date.find("Year"))  or ""
    month = _text(pub_date.find("Month")) or ""
    day   = _text(pub_date.find("Day"))   or ""
    parts = [p for p in [year, month, day] if p]
    return " ".join(parts)
