"""User domain models — lightweight browser identity for the chatbot.

Per §14.1 of docs/RAG_QA_PLAN.md (revised 2026-05-09 from "treat all
unauthenticated traffic as one anonymous user" to "browser-side user_id
+ display name"). No password, no email, no JWT — just an opaque
``user_id`` minted server-side and stored in the browser, plus a
display name the user picks on first visit.

Compatibility: ``user_id`` on :class:`ConversationSession` remains
``Optional[str]`` so anonymous traffic still works for any client that
hasn't adopted the new flow yet.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class User:
    """A chatbot user.

    Identity is the immutable ``id`` (we mint it on first
    registration). ``display_name`` is mutable — the user can rename
    themselves; we record the latest in ``last_seen_at``.

    ``metadata`` is a free-form bag for future per-user state
    (preferences, semantic-memory hooks for §5.4 / P14, etc.) so we
    don't keep migrating the node shape.
    """

    id: str = field(default_factory=lambda: f"user_{uuid.uuid4().hex[:16]}")
    display_name: str = "Anonymous"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_now)
    last_seen_at: datetime = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
            "last_seen_at": self.last_seen_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "User":
        def _parse(ts: Any) -> datetime:
            if isinstance(ts, str):
                try:
                    return datetime.fromisoformat(ts)
                except ValueError:
                    return _now()
            return ts or _now()

        return cls(
            id=data.get("id") or f"user_{uuid.uuid4().hex[:16]}",
            display_name=data.get("display_name") or "Anonymous",
            metadata=dict(data.get("metadata") or {}),
            created_at=_parse(data.get("created_at")),
            last_seen_at=_parse(data.get("last_seen_at")),
        )

    def touch(self) -> None:
        self.last_seen_at = _now()


__all__ = ["User"]
