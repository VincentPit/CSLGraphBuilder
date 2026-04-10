"""
Open Targets GraphQL API Client.

Wraps the Open Targets Platform GraphQL API (https://api.platform.opentargets.org/api/v4/graphql)
providing typed, async access to disease–target associations, disease details,
and associated evidence strings.

All network I/O is async; the caller is responsible for providing an
aiohttp.ClientSession or letting this module manage one per call.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover
    AIOHTTP_AVAILABLE = False


_ENDPOINT = "https://api.platform.opentargets.org/api/v4/graphql"

_DISEASE_INFO_QUERY = """
query DiseaseInfo($diseaseId: String!) {
  disease(efoId: $diseaseId) {
    id
    name
    description
    therapeuticAreas {
      id
      name
    }
    synonyms {
      relation
      terms
    }
  }
}
"""

_ASSOCIATIONS_QUERY = """
query DiseaseAssociations($diseaseId: String!, $page: Pagination!) {
  disease(efoId: $diseaseId) {
    id
    name
    associatedTargets(page: $page) {
      count
      rows {
        target {
          id
          approvedSymbol
          approvedName
          biotype
          functionDescriptions
        }
        score
        datatypeScores {
          componentId: id
          score
        }
      }
    }
  }
}
"""

_TARGET_INFO_QUERY = """
query TargetInfo($targetId: String!) {
  target(ensemblId: $targetId) {
    id
    approvedSymbol
    approvedName
    biotype
    functionDescriptions
    pathways {
      pathway
      pathwayId
    }
  }
}
"""


@dataclass
class DiseaseInfo:
    """Structured representation of an Open Targets disease node."""

    id: str
    name: str
    description: str = ""
    therapeutic_areas: List[Dict[str, str]] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "DiseaseInfo":
        synonyms: List[str] = []
        for syn_group in data.get("synonyms") or []:
            synonyms.extend(syn_group.get("terms") or [])
        return cls(
            id=data["id"],
            name=data.get("name") or "",
            description=data.get("description") or "",
            therapeutic_areas=data.get("therapeuticAreas") or [],
            synonyms=synonyms,
        )


@dataclass
class TargetAssociation:
    """A single target associated with a disease."""

    target_id: str
    target_symbol: str
    target_name: str
    biotype: str
    function_descriptions: List[str]
    association_score: float
    datatype_scores: Dict[str, float]

    @classmethod
    def from_api(cls, row: Dict[str, Any]) -> "TargetAssociation":
        target = row.get("target") or {}
        datatype_scores: Dict[str, float] = {
            d["componentId"]: d["score"]
            for d in (row.get("datatypeScores") or [])
            if d.get("componentId") is not None
        }
        return cls(
            target_id=target.get("id") or "",
            target_symbol=target.get("approvedSymbol") or "",
            target_name=target.get("approvedName") or "",
            biotype=target.get("biotype") or "",
            function_descriptions=target.get("functionDescriptions") or [],
            association_score=row.get("score") or 0.0,
            datatype_scores=datatype_scores,
        )


@dataclass
class IngestResult:
    """Outcome of a paginated disease fetch."""

    disease: Optional[DiseaseInfo]
    associations: List[TargetAssociation]
    total_associations: int
    fetched_pages: int
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.disease is not None and not self.errors


class OpenTargetsClient:
    """
    Async GraphQL client for the Open Targets Platform API.

    Usage::

        async with OpenTargetsClient() as client:
            result = await client.fetch_disease(
                disease_id="EFO_0000275",
                max_associations=250,
            )

    Parameters
    ----------
    endpoint:
        GraphQL endpoint URL. Defaults to the public Open Targets API.
    page_size:
        Number of associations to request per page (max 500 per the API).
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        endpoint: str = _ENDPOINT,
        page_size: int = 100,
        timeout: int = 30,
    ) -> None:
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError(
                "aiohttp is required for OpenTargetsClient. "
                "Install it with: pip install aiohttp"
            )
        self._endpoint = endpoint
        self._page_size = page_size
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Context-manager helpers
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "OpenTargetsClient":
        self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_disease_info(self, disease_id: str) -> Optional[DiseaseInfo]:
        """
        Fetch metadata for a single disease EFO identifier.

        Returns ``None`` if the disease is not found.
        """
        data = await self._query(
            _DISEASE_INFO_QUERY, {"diseaseId": disease_id}
        )
        disease_data = (data.get("data") or {}).get("disease")
        if disease_data is None:
            self.logger.warning("Disease not found: %s", disease_id)
            return None
        return DiseaseInfo.from_api(disease_data)

    async def fetch_associations(
        self,
        disease_id: str,
        max_associations: int = 500,
    ) -> tuple[List[TargetAssociation], int]:
        """
        Fetch paginated disease–target associations.

        Returns ``(associations, total_count)``.
        """
        associations: List[TargetAssociation] = []
        page_index = 0
        total = 0

        while len(associations) < max_associations:
            variables = {
                "diseaseId": disease_id,
                "page": {"index": page_index, "size": self._page_size},
            }
            data = await self._query(_ASSOCIATIONS_QUERY, variables)
            disease_node = (data.get("data") or {}).get("disease") or {}
            associated = disease_node.get("associatedTargets") or {}
            total = associated.get("count") or 0
            rows = associated.get("rows") or []

            if not rows:
                break

            for row in rows:
                if len(associations) >= max_associations:
                    break
                associations.append(TargetAssociation.from_api(row))

            page_index += 1

            if len(associations) >= total:
                break

        return associations, total

    async def fetch_disease(
        self,
        disease_id: str,
        max_associations: int = 500,
    ) -> IngestResult:
        """
        Convenience method combining disease info + associations in one call.
        """
        errors: List[str] = []

        disease = await self.fetch_disease_info(disease_id)
        if disease is None:
            return IngestResult(
                disease=None,
                associations=[],
                total_associations=0,
                fetched_pages=0,
                errors=[f"Disease '{disease_id}' not found in Open Targets"],
            )

        try:
            associations, total = await self.fetch_associations(
                disease_id, max_associations
            )
        except Exception as exc:  # pragma: no cover
            self.logger.error("Failed to fetch associations: %s", exc, exc_info=True)
            errors.append(str(exc))
            associations, total = [], 0

        fetched_pages = (len(associations) + self._page_size - 1) // self._page_size

        return IngestResult(
            disease=disease,
            associations=associations,
            total_associations=total,
            fetched_pages=fetched_pages,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _query(
        self, query: str, variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single GraphQL query."""
        if self._session is None:
            raise RuntimeError(
                "OpenTargetsClient must be used as an async context manager."
            )

        payload = {"query": query, "variables": variables}

        try:
            async with self._session.post(
                self._endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()
                result: Dict[str, Any] = await response.json()

                if "errors" in result:
                    messages = [e.get("message", "unknown") for e in result["errors"]]
                    raise RuntimeError(
                        f"GraphQL errors: {'; '.join(messages)}"
                    )

                return result

        except aiohttp.ClientError as exc:
            raise RuntimeError(f"Open Targets API request failed: {exc}") from exc
