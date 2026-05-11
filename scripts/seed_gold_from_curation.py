#!/usr/bin/env python3
"""Seed an eval gold-set draft from the curation review queue.

Resolution of §14 Q5 in ``docs/RAG_QA_PLAN.md`` (2026-05-10): rather
than authoring a gold set from scratch, we mine the approved
``CurationReview`` rows already stamped on entities and relationships
(``metadata.annotations.verification_status = "approved"``). Each
approved fact becomes a draft :class:`GoldQuestion` whose gold ids are
the approved row's id + its source chunks + source documents.

The output is **a draft** — an SME still has to:
- Edit each generated question into natural phrasing.
- Choose ``gold_answer_substrings`` that the model actually has to surface.
- Optionally split a row into multiple questions (e.g. an entity with
  five relationships might warrant a few separate gold rows).

Run:

    python scripts/seed_gold_from_curation.py \\
        --limit 50 \\
        --out tests/eval/rag_gold_draft.yaml

Then ``git diff tests/eval/rag_gold_draft.yaml tests/eval/rag_gold.yaml``
to spot the new rows and SME-edit before merging.

Implementation notes:
- Reads from Neo4j via the same ``create_graph_repository`` factory the
  API uses, so it picks up ``GRAPH_PROVIDER`` / ``NEO4J_URI`` env vars.
- Question templates are deliberately *generic* (the SME tightens them):
  - Entity: "What is {name}?"  / "Tell me about {name}."
  - Relationship: "How is {source_name} related to {target_name}?"
- ``intent`` is a heuristic guess (lookup vs relational) — also for the
  SME to adjust.
- Falls back to a no-op when the underlying graph repo doesn't expose
  ``execute_cypher_query`` (in-memory dev mode) so the script is safe to
  smoke-test locally.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# Make the src/ package importable when invoking from repo root.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from graphbuilder.infrastructure.config.settings import GraphBuilderConfig  # noqa: E402


logger = logging.getLogger("seed_gold")


_APPROVED_STATUSES = {"approved", "verified"}


def _row_status(meta: Any) -> str:
    """Extract ``verification_status`` from Neo4j's metadata column.

    Metadata is JSON-stringified on write; we tolerate both string and
    already-decoded dict forms.
    """
    if not meta:
        return ""
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (ValueError, json.JSONDecodeError):
            return ""
    if not isinstance(meta, dict):
        return ""
    return (meta.get("annotations") or {}).get("verification_status") or ""


def _yaml_escape(text: str) -> str:
    """Minimal YAML string escape — only quote/backslash, not full PyYAML.

    Keeps the script dependency-free so it can run in any environment
    that already has the project's Python deps.
    """
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _dump_yaml(rows: List[Dict[str, Any]]) -> str:
    """Render the seed list as YAML matching ``tests/eval/rag_gold.yaml``.

    Hand-rolled rather than using PyYAML so the output stays readable
    (block-style lists, predictable key order) regardless of YAML
    dumper defaults.
    """
    out: List[str] = [
        "# Seeded from scripts/seed_gold_from_curation.py — DRAFT.",
        "# An SME must edit each row before merging into rag_gold.yaml:",
        "# - Tighten the question into natural phrasing.",
        "# - Choose gold_answer_substrings the model actually has to surface.",
        "# - Verify the intent label and gold id sets.",
        "",
    ]
    for r in rows:
        out.append(f"- id: {r['id']}")
        out.append(f'  question: "{_yaml_escape(r["question"])}"')
        if r.get("intent"):
            out.append(f"  intent: {r['intent']}")
        for key in ("gold_entity_ids", "gold_relationship_ids", "gold_chunk_ids"):
            vals = r.get(key) or []
            if not vals:
                continue
            out.append(f"  {key}:")
            for v in vals:
                out.append(f"    - {v}")
        notes = r.get("notes")
        if notes:
            out.append(f'  notes: "{_yaml_escape(notes)}"')
        out.append("")
    return "\n".join(out)


def _question_for_entity(name: str, entity_type: Optional[str]) -> str:
    kind = (entity_type or "").strip()
    if kind:
        return f"What is {name} ({kind.lower()})?"
    return f"What is {name}?"


def _question_for_relationship(
    source_name: str, target_name: str, rel_type: Optional[str],
) -> str:
    if rel_type:
        verb = rel_type.replace("_", " ").lower()
        return f"How does {source_name} {verb} {target_name}?"
    return f"How is {source_name} related to {target_name}?"


async def _fetch_approved_entities(
    graph_repo, limit: int,
) -> List[Dict[str, Any]]:
    """Pull approved entities with their source chunk/document ids.

    Filter on the same metadata predicate the curation router uses, but
    inverted — we want rows that *are* approved, not pending.
    """
    query = (
        "MATCH (e:Entity) "
        "WHERE e.metadata IS NOT NULL "
        "RETURN "
        "  e.id AS id, e.name AS name, e.entity_type AS entity_type, "
        "  coalesce(e.source_chunk_ids, []) AS chunk_ids, "
        "  coalesce(e.source_document_ids, []) AS document_ids, "
        "  e.metadata AS metadata "
        "LIMIT $limit"
    )
    rows = await graph_repo.execute_cypher_query(query, {"limit": limit * 4})
    out: List[Dict[str, Any]] = []
    for row in rows:
        if _row_status(row.get("metadata")) not in _APPROVED_STATUSES:
            continue
        out.append({
            "kind": "entity",
            "id": row.get("id"),
            "name": row.get("name") or row.get("id"),
            "entity_type": row.get("entity_type"),
            "chunk_ids": list(row.get("chunk_ids") or []),
            "document_ids": list(row.get("document_ids") or []),
        })
        if len(out) >= limit:
            break
    return out


async def _fetch_approved_relationships(
    graph_repo, limit: int,
) -> List[Dict[str, Any]]:
    query = (
        "MATCH (s:Entity)-[r:RELATES]->(t:Entity) "
        "WHERE r.metadata IS NOT NULL "
        "RETURN "
        "  r.id AS id, r.relationship_type AS relationship_type, "
        "  coalesce(r.source_chunk_ids, []) AS chunk_ids, "
        "  coalesce(r.source_document_ids, []) AS document_ids, "
        "  r.metadata AS metadata, "
        "  s.id AS source_id, s.name AS source_name, "
        "  t.id AS target_id, t.name AS target_name "
        "LIMIT $limit"
    )
    rows = await graph_repo.execute_cypher_query(query, {"limit": limit * 4})
    out: List[Dict[str, Any]] = []
    for row in rows:
        if _row_status(row.get("metadata")) not in _APPROVED_STATUSES:
            continue
        out.append({
            "kind": "relationship",
            "id": row.get("id"),
            "relationship_type": row.get("relationship_type"),
            "source_id": row.get("source_id"),
            "source_name": row.get("source_name") or row.get("source_id"),
            "target_id": row.get("target_id"),
            "target_name": row.get("target_name") or row.get("target_id"),
            "chunk_ids": list(row.get("chunk_ids") or []),
            "document_ids": list(row.get("document_ids") or []),
        })
        if len(out) >= limit:
            break
    return out


def _to_gold_row(idx: int, src: Dict[str, Any]) -> Dict[str, Any]:
    if src["kind"] == "entity":
        question = _question_for_entity(src["name"], src.get("entity_type"))
        return {
            "id": f"q_seed_{idx:03d}",
            "question": question,
            "intent": "lookup",
            "gold_entity_ids": [src["id"]] if src.get("id") else [],
            "gold_chunk_ids": src["chunk_ids"][:5],  # cap so SME isn't drowned in ids
            "notes": (
                f"seeded from approved Entity row {src['id']} "
                f"(SME: tighten phrasing + add gold_answer_substrings)"
            ),
        }
    # relationship
    question = _question_for_relationship(
        src["source_name"], src["target_name"], src.get("relationship_type"),
    )
    return {
        "id": f"q_seed_{idx:03d}",
        "question": question,
        "intent": "relational",
        "gold_entity_ids": [
            x for x in (src.get("source_id"), src.get("target_id")) if x
        ],
        "gold_relationship_ids": [src["id"]] if src.get("id") else [],
        "gold_chunk_ids": src["chunk_ids"][:5],
        "notes": (
            f"seeded from approved RELATES row {src['id']} "
            f"(SME: tighten phrasing + add gold_answer_substrings)"
        ),
    }


async def _main_async(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = GraphBuilderConfig()

    # Build the graph repo via the same factory the API uses. If we're
    # not on Neo4j there's no curation queue to mine — bail with a clear
    # message rather than silently emitting an empty file.
    try:
        from graphbuilder.infrastructure.repositories.graph_repository import (
            create_graph_repository,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: cannot import graph repository: {exc}", file=sys.stderr)
        return 2

    driver = None
    if config.database.provider == "neo4j":
        try:
            from neo4j import AsyncGraphDatabase
        except Exception as exc:  # noqa: BLE001
            print(f"error: neo4j driver not installed: {exc}", file=sys.stderr)
            return 2
        driver = AsyncGraphDatabase.driver(
            os.getenv("NEO4J_URI", config.database.host),
            auth=(
                os.getenv("NEO4J_USER", config.database.user or "neo4j"),
                os.getenv("NEO4J_PASSWORD", config.database.password or ""),
            ),
        )

    graph_repo = create_graph_repository(config, neo4j_driver=driver)
    if not hasattr(graph_repo, "execute_cypher_query"):
        print(
            "error: graph repo doesn't expose execute_cypher_query — set "
            "GRAPH_PROVIDER=neo4j to run this script.",
            file=sys.stderr,
        )
        return 2

    logger.info("mining curation queue from %s", type(graph_repo).__name__)

    half = max(1, args.limit // 2)
    entity_rows, rel_rows = await asyncio.gather(
        _fetch_approved_entities(graph_repo, half),
        _fetch_approved_relationships(graph_repo, args.limit - half),
    )
    logger.info(
        "fetched %d approved entities + %d approved relationships",
        len(entity_rows), len(rel_rows),
    )

    rows = entity_rows + rel_rows
    if not rows:
        print(
            "no approved curation rows found — either the queue is empty or "
            "the metadata predicate didn't match. Check "
            "metadata.annotations.verification_status in your graph.",
            file=sys.stderr,
        )
        return 1

    gold_rows = [_to_gold_row(i + 1, r) for i, r in enumerate(rows)]
    yaml_text = _dump_yaml(gold_rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml_text, encoding="utf-8")

    print(f"wrote {len(gold_rows)} draft gold rows → {out_path}")
    print("Next: SME-edits this file then folds rows into tests/eval/rag_gold.yaml.")

    if driver is not None:
        await driver.close()
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--limit", type=int, default=50,
        help="Max rows to emit (split entities/relationships ~50/50).",
    )
    p.add_argument(
        "--out", default="tests/eval/rag_gold_draft.yaml",
        help="Output YAML path (created/overwritten).",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(_main_async(_parse_args())))
