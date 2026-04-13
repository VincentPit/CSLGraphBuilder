"""Pydantic v2 schemas for external data ingestion."""

from typing import List, Optional
from pydantic import BaseModel, Field


class OpenTargetsIngestRequest(BaseModel):
    disease_id: str = Field(..., description="Open Targets disease EFO identifier, e.g. EFO_0000400")
    max_associations: int = Field(100, ge=1, le=10000)
    min_association_score: float = Field(0.0, ge=0.0, le=1.0)
    tag: Optional[str] = None


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
