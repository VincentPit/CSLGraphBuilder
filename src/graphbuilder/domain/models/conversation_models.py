"""
Conversation domain models — sessions and turns for the RAG Q&A chatbot.

A ``ConversationSession`` groups one user's interaction over time. Each
``ConversationTurn`` is one (user query, assistant answer) pair plus the
provenance of what was retrieved/cited and how the LLM was used. Turns are
embedded so we can do episodic recall (§5.3 of docs/RAG_QA_PLAN.md).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ConversationTurn:
    """One (user query, assistant answer) exchange.

    All identifiers are stored as flat lists so the Neo4j projection
    matches what ``_to_neo4j_props`` (in ``document_repository.py``) accepts
    without nested map serialisation.
    """

    session_id: str
    idx: int
    user_query: str
    llm_answer: str = ""

    id: str = field(default_factory=lambda: f"turn_{uuid.uuid4().hex[:16]}")
    request_id: Optional[str] = None

    cited_entity_ids: List[str] = field(default_factory=list)
    cited_relationship_ids: List[str] = field(default_factory=list)
    cited_chunk_ids: List[str] = field(default_factory=list)

    query_embedding: Optional[List[float]] = None
    answer_embedding: Optional[List[float]] = None

    feedback_rating: Optional[int] = None
    feedback_comment: Optional[str] = None

    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0

    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "idx": self.idx,
            "user_query": self.user_query,
            "llm_answer": self.llm_answer,
            "request_id": self.request_id,
            "cited_entity_ids": list(self.cited_entity_ids),
            "cited_relationship_ids": list(self.cited_relationship_ids),
            "cited_chunk_ids": list(self.cited_chunk_ids),
            "query_embedding": self.query_embedding,
            "answer_embedding": self.answer_embedding,
            "feedback_rating": self.feedback_rating,
            "feedback_comment": self.feedback_comment,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": self.latency_ms,
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationTurn":
        ts = data.get("created_at")
        if isinstance(ts, str):
            try:
                created_at = datetime.fromisoformat(ts)
            except ValueError:
                created_at = _now()
        else:
            created_at = ts or _now()

        return cls(
            id=data.get("id") or f"turn_{uuid.uuid4().hex[:16]}",
            session_id=data["session_id"],
            idx=int(data.get("idx", 0)),
            user_query=data.get("user_query", ""),
            llm_answer=data.get("llm_answer", ""),
            request_id=data.get("request_id"),
            cited_entity_ids=list(data.get("cited_entity_ids") or []),
            cited_relationship_ids=list(data.get("cited_relationship_ids") or []),
            cited_chunk_ids=list(data.get("cited_chunk_ids") or []),
            query_embedding=data.get("query_embedding"),
            answer_embedding=data.get("answer_embedding"),
            feedback_rating=data.get("feedback_rating"),
            feedback_comment=data.get("feedback_comment"),
            prompt_tokens=int(data.get("prompt_tokens", 0) or 0),
            completion_tokens=int(data.get("completion_tokens", 0) or 0),
            latency_ms=int(data.get("latency_ms", 0) or 0),
            metadata=dict(data.get("metadata") or {}),
            created_at=created_at,
        )


@dataclass
class ConversationSession:
    """A conversation thread owned by a user (or anonymous).

    The ``summary`` field is the rolling summary used by the memory layer
    (§5.2 of docs/RAG_QA_PLAN.md). It's regenerated, not appended, when older
    turns slide out of the working-memory window.
    """

    id: str = field(default_factory=lambda: f"session_{uuid.uuid4().hex[:16]}")
    user_id: Optional[str] = None
    title: Optional[str] = None
    summary: str = ""
    turn_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_now)
    last_active_at: datetime = field(default_factory=_now)

    def touch(self) -> None:
        self.last_active_at = _now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "summary": self.summary,
            "turn_count": self.turn_count,
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
            "last_active_at": self.last_active_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSession":
        def _parse(ts: Any) -> datetime:
            if isinstance(ts, str):
                try:
                    return datetime.fromisoformat(ts)
                except ValueError:
                    return _now()
            return ts or _now()

        return cls(
            id=data.get("id") or f"session_{uuid.uuid4().hex[:16]}",
            user_id=data.get("user_id"),
            title=data.get("title"),
            summary=data.get("summary") or "",
            turn_count=int(data.get("turn_count", 0) or 0),
            metadata=dict(data.get("metadata") or {}),
            created_at=_parse(data.get("created_at")),
            last_active_at=_parse(data.get("last_active_at")),
        )


__all__ = [
    "MessageRole",
    "ConversationTurn",
    "ConversationSession",
]
