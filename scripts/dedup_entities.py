#!/usr/bin/env python3
"""One-shot post-hoc entity dedup against a Neo4j graph (P13 follow-up).

Usage:

    # Dry-run: print every proposed merge, no destructive action.
    python scripts/dedup_entities.py --dry-run

    # Execute: collapse same-name entities across types via merge_entities.
    python scripts/dedup_entities.py --apply

    # Limit the run to a single name (handy when iterating on a fix).
    python scripts/dedup_entities.py --apply --name "BRCA1"

The pass groups :Entity nodes by lowercased name across entity_types
(the existing ingestion-time dedup tiers only match within a single
entity_type, which is why "BRCA1 Concept" + "Brca1 GENE" both end up
in the graph). Primary selection: highest connectivity, ties broken
by type priority (GENE > DISEASE > DRUG > … > Concept) then oldest.

The script never re-ingests documents — it only consolidates nodes
that are already in the graph.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Allow ``python scripts/dedup_entities.py`` to import the local package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphbuilder.application.services.entity_dedup import (  # noqa: E402
    apply_dedup_plan,
    fetch_entity_summaries_neo4j,
    plan_dedup,
)


def _load_dotenv_simple(dotenv_path: Path) -> None:
    """Load ``.env`` without pulling python-dotenv (which has issues
    when invoked from a non-frame context — see the eval flow)."""
    if not dotenv_path.exists():
        return
    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        k, v = raw.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cross-type entity dedup migration for the local Neo4j graph.",
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Plan only; do not merge.")
    mode.add_argument("--apply", action="store_true", help="Apply the plan (destructive).")
    p.add_argument(
        "--name", default=None,
        help="Optional case-insensitive name substring filter (e.g. 'BRCA1').",
    )
    p.add_argument(
        "--min-degree", type=int, default=0,
        help="Skip entities with fewer than N relationships (default 0 = include all).",
    )
    p.add_argument(
        "--limit-groups", type=int, default=None,
        help="Cap the number of duplicate groups acted on (handy for iterative cleanup).",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p


async def _amain(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _load_dotenv_simple(ROOT / ".env")

    try:
        from neo4j import AsyncGraphDatabase
    except ImportError:
        print("error: neo4j driver not installed", file=sys.stderr)
        return 2

    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER", "neo4j")
    pwd = os.environ.get("NEO4J_PASSWORD")
    if not uri or not pwd:
        print("error: NEO4J_URI / NEO4J_PASSWORD not set in env or .env", file=sys.stderr)
        return 2

    driver = AsyncGraphDatabase.driver(uri, auth=(user, pwd))
    try:
        summaries = await fetch_entity_summaries_neo4j(
            driver, min_degree=args.min_degree, name_filter=args.name,
        )
        print(f"loaded {len(summaries)} entities (filtered)")

        plan = plan_dedup(summaries)
        if args.limit_groups is not None:
            plan = plan[: args.limit_groups]

        if not plan:
            print("no duplicate groups found — graph is clean")
            return 0

        print(f"\nproposed {len(plan)} duplicate group(s):")
        for g in plan:
            print(f"  - {g.describe()}")

        # Direction-preserving merge. The repo's existing
        # ``merge_entities()`` always creates outgoing edges from
        # primary, which silently flips the direction of any incoming
        # edges the duplicate had (e.g. ``(paper)-[:DISCUSSES]->(BRCA1
        # GENE)`` would become ``(BRCA1 Concept)-[:DISCUSSES]->(paper)``).
        # We split into outgoing / incoming branches so the resulting
        # edges keep the same orientation they had on the duplicate.
        # External_ids and source_chunk_ids/source_document_ids are
        # also unioned in so we don't lose ENSEMBL IDs etc. when a
        # GENE node merges into a Concept primary.
        async def merge_fn(primary_id: str, duplicate_id: str) -> Any:
            cypher = """
            MATCH (primary:Entity {id: $primary_id})
            MATCH (duplicate:Entity {id: $duplicate_id})
            // Outgoing from duplicate.
            CALL {
                WITH primary, duplicate
                MATCH (duplicate)-[old_rel:RELATES]->(other:Entity)
                  WHERE other.id <> $primary_id
                MERGE (primary)-[new_rel:RELATES {
                    id: randomUUID(),
                    relationship_type: old_rel.relationship_type,
                    strength: old_rel.strength,
                    created_at: datetime()
                }]->(other)
                SET new_rel += properties(old_rel)
                RETURN count(*) AS out_moved
            }
            // Incoming to duplicate.
            CALL {
                WITH primary, duplicate
                MATCH (other:Entity)-[old_rel:RELATES]->(duplicate)
                  WHERE other.id <> $primary_id
                MERGE (other)-[new_rel:RELATES {
                    id: randomUUID(),
                    relationship_type: old_rel.relationship_type,
                    strength: old_rel.strength,
                    created_at: datetime()
                }]->(primary)
                SET new_rel += properties(old_rel)
                RETURN count(*) AS in_moved
            }
            // Add duplicate's name as an alias on primary.
            SET primary.aliases = CASE
                WHEN primary.aliases IS NULL THEN [duplicate.name]
                WHEN duplicate.name IN primary.aliases THEN primary.aliases
                ELSE primary.aliases + [duplicate.name]
            END
            // Union source_chunk_ids and source_document_ids so we
            // don't lose provenance from the merged-away node.
            SET primary.source_chunk_ids =
                [x IN coalesce(primary.source_chunk_ids, []) + coalesce(duplicate.source_chunk_ids, [])
                 WHERE x IS NOT NULL]
            SET primary.source_document_ids =
                [x IN coalesce(primary.source_document_ids, []) + coalesce(duplicate.source_document_ids, [])
                 WHERE x IS NOT NULL]
            // Finally drop the duplicate node + its remaining edges.
            DETACH DELETE duplicate
            RETURN primary, out_moved, in_moved
            """
            async with driver.session() as session:
                result = await session.run(
                    cypher,
                    {"primary_id": primary_id, "duplicate_id": duplicate_id},
                )
                rec = await result.single()
                if not rec:
                    raise RuntimeError("merge query returned no rows")
                return rec["primary"]

        report = await apply_dedup_plan(plan, merge_fn=merge_fn, dry_run=args.dry_run)

        print()
        print(f"groups planned:  {report.groups_planned}")
        print(f"merges planned:  {report.merges_planned}")
        if args.apply:
            print(f"merges applied:  {report.merges_applied}")
            print(f"merge failures:  {len(report.failures)}")
            for primary, dup, err in report.failures:
                print(f"  - {dup[:8]} -> {primary[:8]}: {err}")
        else:
            print("(dry-run: nothing was changed)")
        return 0 if not report.failures else 1
    finally:
        await driver.close()


def main() -> int:
    args = _build_argparser().parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
