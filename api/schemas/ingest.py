"""Pydantic v2 schemas for external data ingestion."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


OpenTargetsKind = Literal["disease", "target", "drug", "variant", "study"]


class OpenTargetsIngestRequest(BaseModel):
    """Request body for /ingest/open-targets.

    Accepts any Open Targets entity ID (disease/target/drug/variant/study).
    The kind is auto-detected from the ID prefix when ``entity_type`` is
    omitted. The legacy ``disease_id`` field is still accepted as an alias
    for ``entity_id`` so existing API clients keep working.
    """

    entity_id: Optional[str] = Field(
        None,
        description=(
            "Open Targets entity identifier. Examples: EFO_0000400 (disease), "
            "ENSG00000048462 (target/gene), CHEMBL941 (drug), rs7412 (variant), "
            "GCST006085 (study)."
        ),
    )
    entity_type: Optional[OpenTargetsKind] = Field(
        None,
        description=(
            "Override entity kind. When omitted, the kind is auto-detected "
            "from the ID prefix."
        ),
    )
    disease_id: Optional[str] = Field(
        None,
        description="Deprecated — use ``entity_id``. Kept for back-compat.",
    )
    max_associations: int = Field(100, ge=1, le=10000)
    max_known_drugs: int = Field(100, ge=0, le=10000)
    min_association_score: float = Field(0.0, ge=0.0, le=1.0)
    tag: Optional[str] = None

    @model_validator(mode="after")
    def _coerce_id(self) -> "OpenTargetsIngestRequest":
        if not self.entity_id and self.disease_id:
            self.entity_id = self.disease_id
            if self.entity_type is None:
                self.entity_type = "disease"
        if not self.entity_id:
            raise ValueError(
                "OpenTargetsIngestRequest requires `entity_id` (or legacy `disease_id`)."
            )
        return self


class PubMedIngestRequest(BaseModel):
    query: str = Field(..., min_length=1, description="PubMed search query")
    max_articles: int = Field(50, ge=1, le=1000)
    email: str = Field(..., description="Email for NCBI API policy compliance")
    api_key: Optional[str] = None
    include_mesh: bool = True
    include_keywords: bool = True
    tag: Optional[str] = None


class IngestResponse(BaseModel):
    job_id: str
    source: str
    status: str
    message: Optional[str] = None


class CrawlIngestRequest(BaseModel):
    urls: List[str] = Field(..., min_length=1, description="Seed URLs to crawl")
    max_pages: int = Field(10, ge=1, le=1000, description="Max pages to crawl per seed URL")
    allowed_domains: List[str] = Field(default_factory=list, description="Restrict crawl to these domains")
    tag: Optional[str] = "web-crawl"
