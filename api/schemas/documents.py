"""Pydantic v2 schemas for document processing."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class ProcessDocumentRequest(BaseModel):
    url: Optional[str] = None
    text: Optional[str] = None
    title: Optional[str] = None
    source_label: Optional[str] = Field(None, description="Alias for title (frontend compat)")
    tags: List[str] = []
    chunk_size: Optional[int] = Field(None, ge=64, le=4096)
    chunk_overlap: Optional[int] = Field(None, ge=0, le=512)

    @model_validator(mode="after")
    def coalesce_title(self) -> "ProcessDocumentRequest":
        if not self.title and self.source_label:
            self.title = self.source_label
        return self

    @model_validator(mode="after")
    def require_url_or_text(self) -> "ProcessDocumentRequest":
        if not self.url and not self.text:
            raise ValueError("Either 'url' or 'text' must be provided.")
        return self


class DocumentStatusResponse(BaseModel):
    document_id: str
    job_id: str
    status: str
    url: Optional[str] = None
    title: Optional[str] = None
    chunks_created: int = 0
    entities_extracted: int = 0
    relationships_extracted: int = 0
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    items: List[DocumentStatusResponse]
    total: int
    limit: int
    offset: int


class JobResponse(BaseModel):
    job_id: str
    status: str  # pending | running | completed | failed
    message: Optional[str] = None
    progress: Optional[float] = None  # 0.0 – 1.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
