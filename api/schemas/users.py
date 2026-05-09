"""Pydantic schemas for the /users endpoint (lightweight chatbot identity).

See §14.1 of docs/RAG_QA_PLAN.md (revised 2026-05-09).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class CreateUserRequest(BaseModel):
    display_name: str = Field(
        ..., min_length=1, max_length=80,
        description="Free-text name shown next to the user's chat sessions.",
    )


class UpdateUserRequest(BaseModel):
    display_name: Optional[str] = Field(
        None, min_length=1, max_length=80,
        description="If set, replaces the existing display name.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Free-form bag for per-user state (preferences, etc.).",
    )


class UserResponse(BaseModel):
    id: str
    display_name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str
    last_seen_at: str
