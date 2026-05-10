"""Unit tests for the cross-type entity-dedup planner."""

from __future__ import annotations

import os
from typing import List

import pytest

os.environ.setdefault("LLM_API_KEY", "not-configured")

from graphbuilder.application.services.entity_dedup import (  # noqa: E402
    DuplicateGroup,
    EntitySummary,
    apply_dedup_plan,
    plan_dedup,
)


def _es(eid: str, name: str, etype: str, deg: int, created: str = "2026-01-01") -> EntitySummary:
    return EntitySummary(id=eid, name=name, entity_type=etype, degree=deg, created_at=created)


# ---------------------------------------------------------------- planner


def test_singleton_groups_are_dropped():
    plan = plan_dedup([_es("a", "BRCA1", "GENE", 5)])
    assert plan == []


def test_groups_by_lowercased_name():
    plan = plan_dedup([
        _es("a", "BRCA1", "GENE", 60),
        _es("b", "brca1", "Concept", 30),
    ])
    assert len(plan) == 1
    assert plan[0].canonical_key == "brca1"
    assert plan[0].primary.id == "a"   # higher degree wins
    assert plan[0].duplicates[0].id == "b"


def test_type_priority_breaks_degree_tie():
    """When two entities tie on degree, GENE beats Concept."""
    plan = plan_dedup([
        _es("a", "BRCA1", "Concept", 10, "2026-01-01"),
        _es("b", "BRCA1", "GENE", 10, "2026-02-01"),
    ])
    assert plan[0].primary.id == "b"
    assert plan[0].primary.entity_type == "GENE"


def test_oldest_breaks_degree_and_type_tie():
    plan = plan_dedup([
        _es("a", "BRCA1", "GENE", 10, "2026-03-01"),
        _es("b", "BRCA1", "GENE", 10, "2026-01-01"),  # oldest
    ])
    assert plan[0].primary.id == "b"


def test_plan_sorts_largest_collapse_first():
    plan = plan_dedup([
        # group1: 2 members
        _es("a1", "alpha", "GENE", 5),
        _es("a2", "alpha", "Concept", 1),
        # group2: 4 members (largest)
        _es("b1", "beta", "GENE", 5),
        _es("b2", "beta", "Concept", 1),
        _es("b3", "beta", "DRUG", 2),
        _es("b4", "beta", "DISEASE", 0),
    ])
    assert plan[0].canonical_key == "beta"
    assert plan[0].merge_count == 3
    assert plan[1].canonical_key == "alpha"


def test_blank_names_are_ignored():
    plan = plan_dedup([_es("a", "", "GENE", 5), _es("b", "  ", "GENE", 1)])
    assert plan == []


# ---------------------------------------------------------------- applier


@pytest.mark.asyncio
async def test_dry_run_does_not_call_merge_fn():
    plan = plan_dedup([
        _es("a", "BRCA1", "GENE", 5),
        _es("b", "brca1", "Concept", 1),
    ])
    calls: list[tuple[str, str]] = []

    async def merge_fn(primary_id: str, duplicate_id: str):
        calls.append((primary_id, duplicate_id))

    report = await apply_dedup_plan(plan, merge_fn=merge_fn, dry_run=True)
    assert report.merges_planned == 1
    assert report.merges_applied == 0
    assert calls == []


@pytest.mark.asyncio
async def test_apply_calls_merge_fn_for_every_duplicate():
    plan = plan_dedup([
        _es("p", "alpha", "GENE", 5),
        _es("d1", "alpha", "Concept", 1),
        _es("d2", "alpha", "DRUG", 2),
    ])
    calls: list[tuple[str, str]] = []

    async def merge_fn(primary_id: str, duplicate_id: str):
        calls.append((primary_id, duplicate_id))

    report = await apply_dedup_plan(plan, merge_fn=merge_fn, dry_run=False)
    assert report.merges_applied == 2
    assert {c[0] for c in calls} == {"p"}
    assert {c[1] for c in calls} == {"d1", "d2"}


@pytest.mark.asyncio
async def test_apply_collects_failures_and_keeps_going():
    """A failed merge must not abort the rest of the plan — the
    operator can re-run with the failing pair fixed."""
    plan = plan_dedup([
        _es("p1", "alpha", "GENE", 5),
        _es("d1", "alpha", "Concept", 1),
        _es("p2", "beta", "GENE", 5),
        _es("d2", "beta", "Concept", 1),
    ])

    async def merge_fn(primary_id: str, duplicate_id: str):
        if duplicate_id == "d1":
            raise RuntimeError("synthetic failure")
        return None

    report = await apply_dedup_plan(plan, merge_fn=merge_fn, dry_run=False)
    assert report.merges_applied == 1
    assert len(report.failures) == 1
    assert report.failures[0][1] == "d1"
    assert "synthetic failure" in report.failures[0][2]
