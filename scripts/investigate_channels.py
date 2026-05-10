#!/usr/bin/env python3
"""Channel-quality investigation for the RAG retrieval orchestrator.

The first live eval (§9.5 of docs/RAG_QA_PLAN.md) showed that
``vector_only`` narrowly beat ``all_channels`` on F1 — meaning BM25
and Cypher are bringing in noise that the cross-encoder rerank can't
fully clean up. This script answers *where* the noise comes from by
running each gold question through the orchestrator directly (no
HTTP, no LLM) and capturing:

- per-channel raw hits (ids + scores)
- which of those hits are in the gold set
- which hits survive the cross-encoder rerank into the final top-8
- the "channel attribution" of the final top-8 — how many of the
  surviving items came from vector vs BM25 vs Cypher

Outputs:
- ``per_hit.csv`` — one row per (question, channel, hit_id)
- ``per_question.csv`` — one row per question with aggregate counts
- ``channel_investigation.md`` — narrative summary with the key
  findings + a "noise candidates" table (questions where BM25 or
  Cypher pushed a vector-gold-hit out of the top-8)

Usage:

    python scripts/investigate_channels.py \\
        --gold tests/eval/rag_gold_local.yaml \\
        --out  tests/eval/_reports/channels/

Run against the local Neo4j (uses ``.env``); no API server required.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphbuilder.core.eval.gold import GoldQuestion, load_gold  # noqa: E402
from graphbuilder.core.retrieval import (  # noqa: E402
    Channel,
    ChannelResult,
    RawHit,
    RetrievalConfig,
    RetrievalOrchestrator,
    RetrievedItem,
)
from graphbuilder.core.retrieval.term_extraction import extract_terms  # noqa: E402
from graphbuilder.infrastructure.config.settings import GraphBuilderConfig  # noqa: E402


logger = logging.getLogger("graphbuilder.qa.eval.channels")


# ----------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------

def _load_dotenv_simple(path: Path) -> None:
    """Same lightweight loader the dedup script uses — avoids python-
    dotenv's ``find_dotenv`` AssertionError when invoked outside an
    interactive frame."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        k, v = raw.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


async def _build_orchestrator() -> Tuple[RetrievalOrchestrator, Any]:
    """Build the orchestrator against the local Neo4j (read-only).

    Returns ``(orch, driver)`` so the caller can ``driver.close()``.
    """
    from neo4j import AsyncGraphDatabase
    from graphbuilder.infrastructure.repositories.graph_repository import (
        Neo4jGraphRepository,
    )
    from graphbuilder.infrastructure.repositories.document_repository import (
        Neo4jDocumentRepository,
    )

    uri = os.environ["NEO4J_URI"]
    user = os.environ.get("NEO4J_USER", "neo4j")
    pwd = os.environ["NEO4J_PASSWORD"]
    driver = AsyncGraphDatabase.driver(uri, auth=(user, pwd))

    cfg = GraphBuilderConfig.__new__(GraphBuilderConfig)

    class _P:
        chunk_size = 512
        parallel_workers = 4
    cfg.processing = _P()
    cfg.database = type("D", (), {"provider": "neo4j"})()

    graph_repo = Neo4jGraphRepository(cfg, driver)
    doc_repo = Neo4jDocumentRepository(cfg, driver)
    orch = RetrievalOrchestrator(
        graph_repo=graph_repo,
        document_repo=doc_repo,
        config=RetrievalConfig(),
    )
    return orch, driver


# ----------------------------------------------------------------------
# Diagnostic capture
# ----------------------------------------------------------------------

def _ckey(kind: str, item_id: str) -> str:
    return f"{kind}:{item_id}"


async def _diagnose_one(
    orch: RetrievalOrchestrator,
    question: GoldQuestion,
) -> Dict[str, Any]:
    """Run one question, return the per-channel + final picture.

    We call ``_run_channels`` directly to capture the raw hits with
    ids before fusion, then call ``retrieve()`` separately to get the
    final top-K after RRF + rerank. Two extra Neo4j round-trips per
    question on top of the normal pipeline; trivial for ~25 questions.
    """
    gold_ids = question.all_gold_source_ids()
    terms = extract_terms(question.question)

    # Capture raw per-channel hits.
    embedding = await orch._embed_query(question.question)  # noqa: SLF001
    channel_results: List[ChannelResult] = await orch._run_channels(  # noqa: SLF001
        question.question, embedding, terms,
    )

    # Run retrieve normally to get final items + trace.
    items, trace = await orch.retrieve(
        question.question, query_embedding=embedding,
    )

    # Per-channel breakdown.
    per_channel: Dict[str, Dict[str, Any]] = {}
    for cr in channel_results:
        ids = [_ckey(h.kind.value, h.id) for h in cr.hits]
        in_gold = [i for i in ids if i in gold_ids]
        per_channel[cr.channel.value] = {
            "n_hits": len(ids),
            "hit_ids": ids,
            "n_in_gold": len(in_gold),
            "in_gold_ids": in_gold,
            "latency_ms": cr.latency_ms,
        }

    # Attribute the final top-K back to the channels that contributed.
    final_ids = [_ckey(it.kind.value, it.id) for it in items]
    final_in_gold = [i for i in final_ids if i in gold_ids]
    attribution = Counter()
    for it in items:
        # Skip chunk companions (they're promoted from a parent, no
        # original channel attribution).
        if it.kind.value == "chunk":
            continue
        for ch in it.contributing_channels:
            attribution[ch.value] += 1

    # Identify "noise candidates" — items in the final top-K that came
    # ONLY from BM25 or ONLY from Cypher and aren't in gold. These are
    # the items the rerank failed to demote. Their presence pushes
    # would-be top-K vector items down or out.
    noise = []
    for it in items:
        if it.kind.value == "chunk":
            continue
        cid = _ckey(it.kind.value, it.id)
        if cid in gold_ids:
            continue
        ch_set = {c.value for c in it.contributing_channels}
        if ch_set <= {"bm25"} or ch_set <= {"cypher"} or ch_set == {"bm25", "cypher"}:
            noise.append({
                "id": cid,
                "label": it.label[:60],
                "channels": sorted(ch_set),
                "score_rerank": it.score_rerank,
                "score_rrf": round(it.score_rrf, 4),
            })

    return {
        "question_id": question.id,
        "intent": question.intent,
        "extracted_terms": terms,
        "n_gold": len(gold_ids),
        "per_channel": per_channel,
        "final_top_k_ids": final_ids,
        "final_in_gold": final_in_gold,
        "final_attribution": dict(attribution),
        "noise_candidates": noise,
    }


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def _write_per_hit_csv(records: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["question_id", "channel", "hit_index", "hit_id", "in_gold"])
        for r in records:
            for ch_name, ch in r["per_channel"].items():
                gold = set(ch["in_gold_ids"])
                for i, hid in enumerate(ch["hit_ids"]):
                    w.writerow([r["question_id"], ch_name, i, hid, int(hid in gold)])


def _write_per_question_csv(records: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = ["vector_entity", "vector_relationship", "bm25", "cypher"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        header = (
            ["question_id", "intent", "n_gold", "n_terms",
             "final_n_in_gold", "final_top_k"]
            + [f"{c}_hits" for c in channels]
            + [f"{c}_in_gold" for c in channels]
            + [f"{c}_attribution" for c in channels]
            + ["n_noise"]
        )
        w.writerow(header)
        for r in records:
            row = [
                r["question_id"], r["intent"] or "", r["n_gold"],
                len(r["extracted_terms"]),
                len(r["final_in_gold"]), len(r["final_top_k_ids"]),
            ]
            for c in channels:
                row.append(r["per_channel"].get(c, {}).get("n_hits", 0))
            for c in channels:
                row.append(r["per_channel"].get(c, {}).get("n_in_gold", 0))
            for c in channels:
                row.append(r["final_attribution"].get(c, 0))
            row.append(len(r["noise_candidates"]))
            w.writerow(row)


def _write_markdown(records: Sequence[Dict[str, Any]], path: Path) -> None:
    """Headline narrative — answers the questions the eval surfaced."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate per-channel stats: total hits, total-in-gold,
    # final-top-K attribution share.
    channels = ["vector_entity", "vector_relationship", "bm25", "cypher"]
    totals = {c: {"hits": 0, "in_gold": 0, "attribution": 0} for c in channels}
    for r in records:
        for c in channels:
            ch = r["per_channel"].get(c, {})
            totals[c]["hits"] += ch.get("n_hits", 0)
            totals[c]["in_gold"] += ch.get("n_in_gold", 0)
            totals[c]["attribution"] += r["final_attribution"].get(c, 0)

    final_attribution_total = sum(t["attribution"] for t in totals.values())
    questions_with_noise = sum(1 for r in records if r["noise_candidates"])

    lines = []
    lines.append("# Channel-quality investigation\n")
    lines.append(
        "Diagnostic of where retrieval noise comes from — why "
        "`vector_only` narrowly beats `all_channels` on the live eval "
        "(§9.5 of docs/RAG_QA_PLAN.md). One row per channel; "
        "metrics aggregated over the gold set.\n"
    )
    lines.append("## Per-channel aggregates\n")
    lines.append("| Channel | Total hits | Hits in gold | Hit-rate | Top-8 attribution | Attribution share |")
    lines.append("|---|---|---|---|---|---|")
    for c in channels:
        t = totals[c]
        rate = (t["in_gold"] / t["hits"]) if t["hits"] else 0.0
        share = (t["attribution"] / final_attribution_total) if final_attribution_total else 0.0
        lines.append(
            f"| {c} | {t['hits']} | {t['in_gold']} | {rate:.2%} | "
            f"{t['attribution']} | {share:.1%} |"
        )
    lines.append("")
    lines.append(f"_Final top-8 attribution counts how many surviving entity/rel items "
                 f"each channel contributed to (multi-channel items count once per "
                 f"contributing channel)._\n")

    lines.append("## Noise candidates\n")
    lines.append(
        f"{questions_with_noise} / {len(records)} question(s) have at least one "
        f"item in the final top-8 that came **only** from BM25/Cypher and is **not** "
        f"in gold. These are the items the rerank failed to demote.\n"
    )
    lines.append("| Question | Noise items | Examples |")
    lines.append("|---|---|---|")
    for r in records:
        if not r["noise_candidates"]:
            continue
        examples = "; ".join(
            f"{n['label']} ({'+'.join(n['channels'])})"
            for n in r["noise_candidates"][:2]
        )
        if len(r["noise_candidates"]) > 2:
            examples += f"; +{len(r['noise_candidates']) - 2} more"
        lines.append(f"| {r['question_id']} | {len(r['noise_candidates'])} | {examples} |")
    lines.append("")

    lines.append("## Per-question summary\n")
    lines.append("| Q | Intent | Gold | Top-8 ∩ gold | Channel attribution | Noise |")
    lines.append("|---|---|---|---|---|---|")
    for r in records:
        attr_str = ", ".join(
            f"{c[:3]}={r['final_attribution'].get(c, 0)}"
            for c in channels if r["final_attribution"].get(c)
        ) or "—"
        lines.append(
            f"| {r['question_id']} | {r['intent'] or '-'} | "
            f"{r['n_gold']} | {len(r['final_in_gold'])} | {attr_str} | "
            f"{len(r['noise_candidates'])} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Per-channel diagnostic against the local Neo4j.",
    )
    p.add_argument(
        "--gold", default=str(ROOT / "tests" / "eval" / "rag_gold_local.yaml"),
        help="Gold set YAML to drive the investigation",
    )
    p.add_argument(
        "--out", default=str(ROOT / "tests" / "eval" / "_reports" / "channels"),
        help="Output directory",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p


async def _amain(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _load_dotenv_simple(ROOT / ".env")

    if not (os.environ.get("NEO4J_URI") and os.environ.get("NEO4J_PASSWORD")):
        print("error: NEO4J_URI / NEO4J_PASSWORD required", file=sys.stderr)
        return 2

    gold = load_gold(args.gold)
    logger.info("loaded %d gold questions from %s", len(gold), args.gold)

    orch, driver = await _build_orchestrator()
    try:
        # Skip refusal questions — empty gold sets aren't useful here;
        # the channels will obviously surface "wrong" hits and the
        # signal-to-noise ratio is not what we're measuring.
        records: List[Dict[str, Any]] = []
        for i, q in enumerate(gold, start=1):
            if not q.all_gold_source_ids():
                logger.info("[%d/%d] skipping %s (no gold ids — refusal question)",
                            i, len(gold), q.id)
                continue
            r = await _diagnose_one(orch, q)
            records.append(r)
            logger.info(
                "[%d/%d] %s  gold=%d  top8∩gold=%d  noise=%d  attribution=%s",
                i, len(gold), q.id, r["n_gold"],
                len(r["final_in_gold"]), len(r["noise_candidates"]),
                r["final_attribution"],
            )
    finally:
        await driver.close()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_per_hit_csv(records, out_dir / "per_hit.csv")
    _write_per_question_csv(records, out_dir / "per_question.csv")
    _write_markdown(records, out_dir / "channel_investigation.md")
    logger.info("wrote %s", out_dir / "per_hit.csv")
    logger.info("wrote %s", out_dir / "per_question.csv")
    logger.info("wrote %s", out_dir / "channel_investigation.md")
    return 0


def main() -> int:
    return asyncio.run(_amain(_build_argparser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
