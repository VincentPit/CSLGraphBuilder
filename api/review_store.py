"""In-memory pending review queue for trust-conflicted entities."""

import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PendingReview:
    """An entity/relationship that conflicts with trusted knowledge and needs user review."""

    review_id: str
    conflict_data: dict  # Serialised ConflictEntryResponse
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "pending"  # pending | approved | rejected
    notes: Optional[str] = None


# Global store (process-scoped, same pattern as job_store)
_reviews: Dict[str, PendingReview] = {}


def add_review(conflict_data: dict) -> PendingReview:
    """Add a conflict to the pending review queue. Returns the created review."""
    review_id = str(uuid.uuid4())
    review = PendingReview(review_id=review_id, conflict_data=conflict_data)
    _reviews[review_id] = review
    return review


def get_pending_reviews(status: Optional[str] = None) -> List[PendingReview]:
    """Return reviews, optionally filtered by status."""
    items = list(_reviews.values())
    if status:
        items = [r for r in items if r.status == status]
    return sorted(items, key=lambda r: r.submitted_at, reverse=True)


def get_review(review_id: str) -> Optional[PendingReview]:
    return _reviews.get(review_id)


def decide_review(review_id: str, decision: str, notes: Optional[str] = None) -> Optional[PendingReview]:
    """Mark a review as approved or rejected. Returns updated review or None."""
    review = _reviews.get(review_id)
    if review is None:
        return None
    review.status = decision  # "approved" or "rejected"
    review.notes = notes
    return review
