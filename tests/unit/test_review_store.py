"""Tests for api.review_store — covers add/get/decide and concurrent inserts.

The store is process-global; tests create reviews and key off the returned
review_id rather than asserting on totals.
"""

from __future__ import annotations

import threading

from api.review_store import (
    add_review,
    decide_review,
    get_pending_reviews,
    get_review,
)


def _conflict_payload(name: str = "ATP1A1") -> dict:
    return {"entity": {"name": name, "type": "GENE"}, "reasons": ["trust mismatch"]}


def test_add_review_returns_pending_review_with_id():
    r = add_review(_conflict_payload())
    assert r.review_id
    assert r.status == "pending"
    assert get_review(r.review_id) is r


def test_decide_review_marks_approved_with_notes():
    r = add_review(_conflict_payload())
    updated = decide_review(r.review_id, "approved", notes="curator-checked")
    assert updated is not None
    assert updated.status == "approved"
    assert updated.notes == "curator-checked"


def test_decide_review_unknown_id_returns_none():
    assert decide_review("nope", "approved") is None


def test_get_pending_reviews_filters_by_status():
    a = add_review(_conflict_payload("A"))
    b = add_review(_conflict_payload("B"))
    decide_review(a.review_id, "rejected")

    pending_ids = {r.review_id for r in get_pending_reviews(status="pending")}
    assert b.review_id in pending_ids
    assert a.review_id not in pending_ids


def test_concurrent_adds_all_land():
    """Without locking, dict insertion races would sometimes drop entries.

    With the RLock added to review_store, every add_review() should produce
    a unique review_id and be retrievable.
    """
    n_threads = 16
    per_thread = 25
    ids: list[str] = []
    ids_lock = threading.Lock()

    def worker(idx: int) -> None:
        for k in range(per_thread):
            r = add_review(_conflict_payload(f"e-{idx}-{k}"))
            with ids_lock:
                ids.append(r.review_id)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(ids) == n_threads * per_thread
    # All IDs unique → no two threads collided on the same uuid slot.
    assert len(set(ids)) == len(ids)
    # Every one should be retrievable.
    for review_id in ids[:50]:
        assert get_review(review_id) is not None
