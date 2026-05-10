"""Process-scoped store for chatbot-proposed graph mutations (P10).

Per the §14.6 resolution in docs/RAG_QA_PLAN.md, mutating tool calls
from the chatbot never auto-apply — they queue here as
``ProposedMutation`` rows and a curator promotes them through
``POST /qa/proposals/{id}/apply``. The store mirrors the verification-
queue pattern in ``review_store.py`` (in-memory dict, single-instance
deploy) so we don't add infrastructure prematurely; it can graduate to
:class:`PendingMutation` Neo4j nodes when horizontal scaling lands
(see §12.2).

Shape vs. ``review_store.PendingReview``:

- A pending **review** carries a ``ConflictEntryResponse`` payload —
  the verification flow's "trust conflict" record. Decisions just
  flip status; no graph write happens here.
- A pending **mutation** carries the chat tool call (``tool``,
  ``args``) plus full provenance (``proposer_user_id`` /
  ``session_id`` / ``turn_id`` / ``request_id``). On apply, the
  router runs the tool's apply-handler against the graph repo and
  pins the resulting target id back onto the row so the audit trail
  is closed.

We keep the two stores separate so neither's schema constrains the
other; "review" already means a specific thing in this codebase.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional


ProposalStatus = Literal["pending", "approved", "rejected"]


@dataclass
class ProposedMutation:
    """One queued chatbot mutation awaiting curator review."""

    proposal_id: str
    tool: str                                # e.g. "propose_entity"
    args: Dict[str, Any]                     # validated tool args
    summary: str                             # short human-readable diff
    proposer_user_id: Optional[str] = None
    proposer_session_id: Optional[str] = None
    proposer_turn_id: Optional[str] = None
    request_id: Optional[str] = None
    submitted_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    status: ProposalStatus = "pending"
    decided_at: Optional[datetime] = None
    decision_notes: Optional[str] = None
    applied_target_id: Optional[str] = None  # filled after curator approves + applies
    applied_at: Optional[datetime] = None
    apply_error: Optional[str] = None        # captured if the apply step itself fails

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "tool": self.tool,
            "args": dict(self.args),
            "summary": self.summary,
            "proposer_user_id": self.proposer_user_id,
            "proposer_session_id": self.proposer_session_id,
            "proposer_turn_id": self.proposer_turn_id,
            "request_id": self.request_id,
            "submitted_at": self.submitted_at.isoformat(),
            "status": self.status,
            "decided_at": (
                self.decided_at.isoformat() if self.decided_at else None
            ),
            "decision_notes": self.decision_notes,
            "applied_target_id": self.applied_target_id,
            "applied_at": (
                self.applied_at.isoformat() if self.applied_at else None
            ),
            "apply_error": self.apply_error,
        }


# Process-scoped store. Mirrors api/review_store.py + api/job_store.py.
_proposals: Dict[str, ProposedMutation] = {}


def add_proposal(
    *,
    tool: str,
    args: Dict[str, Any],
    summary: str,
    proposer_user_id: Optional[str] = None,
    proposer_session_id: Optional[str] = None,
    proposer_turn_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ProposedMutation:
    """Record a fresh proposal. Returns the persisted row."""
    proposal_id = f"prop_{uuid.uuid4().hex[:16]}"
    p = ProposedMutation(
        proposal_id=proposal_id,
        tool=tool,
        args=dict(args),
        summary=summary,
        proposer_user_id=proposer_user_id,
        proposer_session_id=proposer_session_id,
        proposer_turn_id=proposer_turn_id,
        request_id=request_id,
    )
    _proposals[proposal_id] = p
    return p


def get_proposal(proposal_id: str) -> Optional[ProposedMutation]:
    return _proposals.get(proposal_id)


def list_proposals(
    *,
    status: Optional[ProposalStatus] = None,
    limit: int = 100,
) -> List[ProposedMutation]:
    """Return proposals newest-first, optionally filtered by status."""
    items = list(_proposals.values())
    if status:
        items = [p for p in items if p.status == status]
    items.sort(key=lambda p: p.submitted_at, reverse=True)
    return items[:limit]


def mark_decided(
    proposal_id: str,
    decision: ProposalStatus,
    *,
    notes: Optional[str] = None,
) -> Optional[ProposedMutation]:
    """Mark a proposal approved/rejected. Idempotent on re-decide.

    Note: this only updates status — the actual graph write happens
    via :func:`mark_applied` after the apply-handler succeeds. We
    split the two so a failed apply leaves the row in
    ``status="approved"`` with an ``apply_error`` set, which is what
    a re-try operator wants to see.
    """
    if decision not in ("pending", "approved", "rejected"):
        raise ValueError(f"invalid decision: {decision}")
    p = _proposals.get(proposal_id)
    if p is None:
        return None
    p.status = decision
    p.decided_at = datetime.now(timezone.utc)
    p.decision_notes = notes
    return p


def mark_applied(
    proposal_id: str,
    *,
    target_id: Optional[str] = None,
    error: Optional[str] = None,
) -> Optional[ProposedMutation]:
    """Pin the target id (or error) of an apply. Returns the row."""
    p = _proposals.get(proposal_id)
    if p is None:
        return None
    p.applied_target_id = target_id
    p.apply_error = error
    p.applied_at = datetime.now(timezone.utc) if error is None else None
    return p


def reset_for_tests() -> None:
    """Drop all proposals — used by test fixtures."""
    _proposals.clear()


__all__ = [
    "ProposalStatus",
    "ProposedMutation",
    "add_proposal",
    "get_proposal",
    "list_proposals",
    "mark_applied",
    "mark_decided",
    "reset_for_tests",
]
