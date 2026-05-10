"""Cross-type entity deduplication.

The ingestion-time dedup tiers in
:class:`Neo4jGraphRepository.save_entities_batch` only match within a
single ``entity_type`` — so a "BRCA1" Concept and a "Brca1" GENE end up
as separate nodes (this is what the P13 eval surfaced). This module
provides a one-shot post-hoc pass that finds same-name groups across
types and merges them via the existing :meth:`merge_entities` repo
method.

Design choices
--------------
* **Group by lowercased name only** for v1 — aliases are a follow-up.
  Adding alias matching needs care because aliases are often noisy
  (gene synonym lists from MeSH).
* **Primary selection by degree, ties broken by oldest** — the highest-
  connectivity node "wins" and absorbs the others. This minimises
  edge churn (Cypher's ``merge_entities`` re-creates each surviving
  rel under the primary's id) and keeps the most-cited node id stable.
* **Type priority** — when degrees tie, prefer GENE > DISEASE > DRUG >
  PROTEIN > Concept > everything else. Biomedical Q&A traffic mostly
  asks about typed entities; collapsing the typed one onto a Concept
  is rarely what the user wants.
* **Dry-run first** is the default contract from the script — we
  surface every proposed merge before any DETACH DELETE runs. The
  service itself is just a planner; ``apply_dedup_plan`` does the
  destructive bit.

The functions here are all repo-agnostic — drive them with either
``InMemoryGraphRepository`` (tests) or ``Neo4jGraphRepository``
(production / ad-hoc migrations).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


logger = logging.getLogger("graphbuilder.dedup")


# Lower index → higher priority. GENE first because most user queries
# are about typed biomedical entities; Concept last because ingestion
# tends to over-create Concept nodes when MeSH tags are present.
_TYPE_PRIORITY = ["GENE", "DISEASE", "DRUG", "PROTEIN", "Person", "Organization", "Concept"]


@dataclass(frozen=True)
class EntitySummary:
    """Lightweight projection used by the planner."""

    id: str
    name: str
    entity_type: str
    degree: int
    created_at: Optional[str] = None  # ISO string; sort lexicographically


@dataclass
class DuplicateGroup:
    """One canonical-name group with the chosen primary + duplicates."""

    canonical_key: str  # lowercased name
    primary: EntitySummary
    duplicates: List[EntitySummary] = field(default_factory=list)

    @property
    def merge_count(self) -> int:
        return len(self.duplicates)

    def describe(self) -> str:
        dup_part = ", ".join(
            f"{d.entity_type}/{d.id[:8]} (deg={d.degree})" for d in self.duplicates
        )
        return (
            f"{self.canonical_key!r}: keep "
            f"{self.primary.entity_type}/{self.primary.id[:8]} "
            f"(deg={self.primary.degree}); merge {dup_part}"
        )


@dataclass
class DedupReport:
    """What apply_dedup_plan actually did (or would have, in dry-run)."""

    groups_planned: int
    merges_planned: int
    merges_applied: int
    failures: List[Tuple[str, str, str]] = field(default_factory=list)
    # (primary_id, duplicate_id, error)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "groups_planned": self.groups_planned,
            "merges_planned": self.merges_planned,
            "merges_applied": self.merges_applied,
            "failure_count": len(self.failures),
            "failures": [
                {"primary": p, "duplicate": d, "error": e}
                for (p, d, e) in self.failures
            ],
        }


# ----------------------------------------------------------------------
# Public API — planner
# ----------------------------------------------------------------------

def plan_dedup(entities: Sequence[EntitySummary]) -> List[DuplicateGroup]:
    """Group ``entities`` by lowercased name and pick a primary per group.

    Singleton groups (no duplicate) are dropped from the plan — they
    have nothing to merge. The result is sorted by ``merge_count``
    descending so the biggest collapses get reviewed first when the
    operator scans dry-run output.
    """
    by_key: Dict[str, List[EntitySummary]] = {}
    for e in entities:
        key = (e.name or "").strip().lower()
        if not key:
            continue
        by_key.setdefault(key, []).append(e)

    plan: List[DuplicateGroup] = []
    for key, members in by_key.items():
        if len(members) < 2:
            continue
        primary, dupes = _pick_primary(members)
        plan.append(DuplicateGroup(canonical_key=key, primary=primary, duplicates=dupes))

    plan.sort(key=lambda g: (-g.merge_count, g.canonical_key))
    return plan


def _pick_primary(
    members: Sequence[EntitySummary],
) -> Tuple[EntitySummary, List[EntitySummary]]:
    """Return ``(primary, duplicates_to_merge)``.

    Primary order: highest degree → lowest type-priority index → oldest
    ``created_at``. The created_at tie-breaker is lexicographic on the
    ISO string, so the *earliest* node wins.
    """
    def _key(e: EntitySummary) -> Tuple[int, int, str]:
        try:
            type_idx = _TYPE_PRIORITY.index(e.entity_type)
        except ValueError:
            type_idx = len(_TYPE_PRIORITY)
        return (-e.degree, type_idx, e.created_at or "9999")

    sorted_members = sorted(members, key=_key)
    primary = sorted_members[0]
    return primary, list(sorted_members[1:])


# ----------------------------------------------------------------------
# Public API — fetcher (Neo4j-backed; used by the script)
# ----------------------------------------------------------------------

async def fetch_entity_summaries_neo4j(
    driver: Any,
    *,
    min_degree: int = 0,
    name_filter: Optional[str] = None,
) -> List[EntitySummary]:
    """Pull every entity + its degree directly from Neo4j.

    A repo-bypass query so the dedup pass doesn't load the full
    GraphEntity object for every node — the planner only needs id,
    name, type, degree, created_at. ``min_degree`` lets an operator
    skip orphans that wouldn't change retrieval anyway.
    """
    cypher = """
        MATCH (e:Entity)
        OPTIONAL MATCH (e)-[r:RELATES]-()
        WITH e, count(r) AS deg
        WHERE deg >= $min_degree
        RETURN e.id AS id, e.name AS name, e.entity_type AS type,
               deg AS degree, toString(e.created_at) AS created_at
    """
    params: Dict[str, Any] = {"min_degree": int(min_degree)}
    if name_filter:
        cypher = cypher.replace(
            "WHERE deg >= $min_degree",
            "WHERE deg >= $min_degree AND toLower(e.name) CONTAINS toLower($name_filter)",
        )
        params["name_filter"] = name_filter

    out: List[EntitySummary] = []
    async with driver.session() as session:
        result = await session.run(cypher, params)
        async for record in result:
            out.append(
                EntitySummary(
                    id=record["id"],
                    name=record["name"] or "",
                    entity_type=record["type"] or "",
                    degree=int(record["degree"] or 0),
                    created_at=record["created_at"],
                )
            )
    return out


# ----------------------------------------------------------------------
# Public API — applier
# ----------------------------------------------------------------------

MergeFn = Callable[[str, str], Awaitable[Any]]
"""Async ``(primary_id, duplicate_id) -> merged_entity`` — matches
``GraphRepositoryInterface.merge_entities``."""


async def apply_dedup_plan(
    plan: Sequence[DuplicateGroup],
    *,
    merge_fn: MergeFn,
    dry_run: bool = True,
    on_progress: Optional[Callable[[int, int, DuplicateGroup], None]] = None,
) -> DedupReport:
    """Execute every merge in ``plan`` (or simulate when ``dry_run``).

    Failures don't abort the run — each merge is independent, and
    aborting halfway would leave the graph in a worse state than
    starting. Failures are collected in the returned report so the
    operator can re-run targeted at just the failing pairs.
    """
    merges_planned = sum(g.merge_count for g in plan)
    report = DedupReport(
        groups_planned=len(plan),
        merges_planned=merges_planned,
        merges_applied=0,
    )

    for i, group in enumerate(plan, start=1):
        if on_progress is not None:
            on_progress(i, len(plan), group)
        for dup in group.duplicates:
            if dry_run:
                logger.info(
                    "[dry-run] would merge %s/%s -> %s/%s",
                    dup.entity_type, dup.id[:8],
                    group.primary.entity_type, group.primary.id[:8],
                )
                continue
            try:
                await merge_fn(group.primary.id, dup.id)
                report.merges_applied += 1
                logger.info(
                    "merged %s -> %s (key=%r)",
                    dup.id[:8], group.primary.id[:8], group.canonical_key,
                )
            except Exception as exc:  # noqa: BLE001 — collect, don't abort
                logger.warning(
                    "merge failed: %s -> %s: %s",
                    dup.id[:8], group.primary.id[:8], exc,
                )
                report.failures.append((group.primary.id, dup.id, str(exc)))

    return report


__all__ = [
    "EntitySummary",
    "DuplicateGroup",
    "DedupReport",
    "plan_dedup",
    "apply_dedup_plan",
    "fetch_entity_summaries_neo4j",
]
