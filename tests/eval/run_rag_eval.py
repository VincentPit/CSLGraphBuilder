#!/usr/bin/env python3
"""Live RAG eval runner — drives the harness against a deployed `/qa/ask`.

Usage:

    python tests/eval/run_rag_eval.py \\
        --api https://chat.staging.local \\
        --api-key "$GRAPHBUILDER_API_KEY" \\
        --gold tests/eval/rag_gold.yaml \\
        --out  tests/eval/_reports/

Writes ``rag_eval.csv`` + ``rag_eval.md`` to the output directory and
exits non-zero if any §9.2 target floor is missed (so this script can
be wired into a nightly CI job that gates on F1).

This is the live-API counterpart of ``test_eval_smoke.py``: same harness
library, different ``ask_fn``. The hermetic test owns CI; this script
owns the staging gate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Make the ``src/`` package importable when this script is run directly
# from the repo root (the CI wrapper sets PYTHONPATH; humans running
# ``python tests/eval/run_rag_eval.py`` get this fallback).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from graphbuilder.core.eval import (  # noqa: E402
    EvalSummary,
    compute_metrics,
    load_gold,
    run_eval,
    write_csv_report,
    write_markdown_report,
)


logger = logging.getLogger("graphbuilder.qa.eval.cli")


# ----------------------------------------------------------------------
# Live AskResult shape — mirrors api/schemas/qa.py::AskResponse so the
# runner's duck-typing hits the right attributes.
# ----------------------------------------------------------------------

@dataclass
class _LiveSource:
    kind: str
    id: str

    @property
    def kind_value(self) -> str:
        return self.kind

    # Runner reads ``s.kind.value`` for enums, ``s.kind`` for strings —
    # an enum-like object satisfies both forms.
    class _KindShim:
        def __init__(self, val: str): self.value = val
        def __str__(self) -> str: return self.value


@dataclass
class _LiveFaithfulness:
    """Mirror of api/schemas/qa.py::FaithfulnessModel for duck-typing."""

    overall_score: Optional[float]
    failed_claims: int = 0
    is_refusal: bool = False


@dataclass
class _LiveAskResult:
    answer: str
    sources: list[Any]
    cited_source_indices: list[int]
    latency_ms: int
    raw: dict
    faithfulness: Optional[_LiveFaithfulness] = None


def _wrap_kind(kind: str) -> Any:
    """Wrap a string kind in an object exposing ``.value`` so the
    runner's ``_kind_value()`` helper succeeds without branching."""
    class _Kind:
        def __init__(self, val: str): self.value = val
    return _Kind(kind)


def _parse_response(payload: dict) -> _LiveAskResult:
    sources = []
    for s in payload.get("sources", []) or []:
        @dataclass
        class _S:
            id: str
            kind: Any
        sources.append(_S(id=s["id"], kind=_wrap_kind(s["kind"])))

    faith_payload = payload.get("faithfulness")
    faithfulness: Optional[_LiveFaithfulness] = None
    if isinstance(faith_payload, dict):
        score = faith_payload.get("overall_score")
        try:
            score_f = float(score) if score is not None else None
        except (TypeError, ValueError):
            score_f = None
        faithfulness = _LiveFaithfulness(
            overall_score=score_f,
            failed_claims=int(faith_payload.get("failed_claims") or 0),
            is_refusal=bool(faith_payload.get("is_refusal")),
        )

    return _LiveAskResult(
        answer=payload.get("answer", "") or "",
        sources=sources,
        cited_source_indices=list(payload.get("cited_source_indices", []) or []),
        latency_ms=int(payload.get("latency_ms", 0) or 0),
        raw=payload,
        faithfulness=faithfulness,
    )


# ----------------------------------------------------------------------
# HTTP client
# ----------------------------------------------------------------------

async def _post_ask(
    *,
    api: str,
    api_key: str,
    user_id: Optional[str],
    query: str,
    top_k: Optional[int],
    timeout_s: float,
    ablation: Optional[dict] = None,
) -> _LiveAskResult:
    """POST /qa/ask using ``aiohttp`` if available, else ``urllib``."""
    body: dict = {"query": query}
    if top_k is not None:
        body["top_k"] = top_k
    if ablation:
        body["ablation"] = ablation

    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    if user_id:
        headers["X-User-Id"] = user_id

    url = f"{api.rstrip('/')}/qa/ask"

    try:
        import aiohttp  # type: ignore[import-not-found]
    except ImportError:
        return await _post_via_urllib(url, body, headers, timeout_s)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, json=body, headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise RuntimeError(f"/qa/ask HTTP {resp.status}: {text[:200]}")
            payload = await resp.json()
    return _parse_response(payload)


async def _post_via_urllib(url: str, body: dict, headers: dict, timeout_s: float) -> _LiveAskResult:
    """Fallback that doesn't pull in aiohttp. Runs the blocking call in
    a worker thread so it doesn't stall the event loop while a real
    request is in flight."""
    import urllib.request

    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST",
    )

    def _do() -> dict:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    payload = await asyncio.to_thread(_do)
    return _parse_response(payload)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the RAG eval harness against a live /qa/ask endpoint.",
    )
    p.add_argument("--api", required=True, help="Base URL of the deployed API")
    p.add_argument(
        "--api-key", default=os.environ.get("GRAPHBUILDER_API_KEY"),
        help="X-API-Key header value (defaults to GRAPHBUILDER_API_KEY env var)",
    )
    p.add_argument(
        "--user-id", default=os.environ.get("GRAPHBUILDER_USER_ID"),
        help="Optional X-User-Id header (eval rows are isolated to this user)",
    )
    p.add_argument(
        "--gold",
        default=str(Path(__file__).parent / "rag_gold.yaml"),
        help="Path to the gold-set YAML/JSON",
    )
    p.add_argument(
        "--out",
        default=str(Path(__file__).parent / "_reports"),
        help="Output directory for CSV + markdown reports",
    )
    p.add_argument(
        "--top-k", type=int, default=None,
        help="Override the QA service's final_top_k (default: server default)",
    )
    p.add_argument(
        "--concurrency", type=int, default=2,
        help="In-flight questions (default 2; bump for staging, keep low for prod)",
    )
    p.add_argument(
        "--timeout", type=float, default=60.0,
        help="Per-request HTTP timeout in seconds",
    )
    p.add_argument(
        "--baseline",
        default=str(Path(__file__).parent / "baselines.json"),
        help="baselines.json path; live_targets in this file gate the exit code",
    )
    p.add_argument(
        "--no-gate", action="store_true",
        help="Don't fail the process when live targets are missed",
    )
    p.add_argument(
        "--ablations",
        default=None,
        help=(
            "Comma-separated list of named ablations to run after the "
            "baseline pass. Recognised names: vector_only, bm25_only, "
            "cypher_only, no_rerank, no_chunks, all_channels (default). "
            "Each ablation runs the gold set through /qa/ask with the "
            "matching channel toggles."
        ),
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p


# Channel/rerank ablation presets — only the listed flags are sent to
# /qa/ask; absent flags fall back to the server's RetrievalConfig.
_ABLATIONS: dict[str, dict] = {
    "all_channels": {},  # baseline — no overrides
    "vector_only": {
        "enable_vector_channel": True,
        "enable_bm25_channel": False,
        "enable_cypher_channel": False,
    },
    "bm25_only": {
        "enable_vector_channel": False,
        "enable_bm25_channel": True,
        "enable_cypher_channel": False,
    },
    "cypher_only": {
        "enable_vector_channel": False,
        "enable_bm25_channel": False,
        "enable_cypher_channel": True,
    },
    "no_rerank": {"enable_cross_encoder": False},
    "no_chunks":  {"emit_chunk_items": False, "chunk_neighbour_radius": 0},
    # Disables the default Person/Document/Organization filter so we
    # can A/B the author/paper noise vs the cleaned baseline. Empty
    # list (not None) is the "include everything" sentinel.
    "with_authors": {"entity_type_blocklist": []},
}


async def _amain(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.api_key:
        logger.error("--api-key is required (or set GRAPHBUILDER_API_KEY)")
        return 2

    gold = load_gold(args.gold)
    logger.info("loaded %d gold questions from %s", len(gold), args.gold)

    def _make_ask_fn(ablation_overrides: Optional[dict]):
        async def ask_fn(query: str):
            return await _post_ask(
                api=args.api,
                api_key=args.api_key,
                user_id=args.user_id,
                query=query,
                top_k=args.top_k,
                timeout_s=args.timeout,
                ablation=ablation_overrides or None,
            )
        return ask_fn

    def _progress(done: int, total: int, record) -> None:
        marker = "✗" if record.error else "✓"
        logger.info(
            "[%d/%d] %s %s  P=%.2f R=%.2f F1=%.2f  (%dms)",
            done, total, marker, record.question_id,
            record.precision_at_k, record.recall_at_k, record.f1_at_k,
            record.latency_ms,
        )

    ask_fn = _make_ask_fn(None)
    records, summary = await run_eval(
        ask_fn=ask_fn, gold=gold, concurrency=args.concurrency,
        on_progress=_progress,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv_report(records, out_dir / "rag_eval.csv")

    # Optional ablation matrix — re-runs the gold set under each named
    # config preset. Each ablation only sends the flags it cares about
    # so the server's ``RetrievalConfig`` defaults fill in the rest.
    ablation_results: list = []
    if args.ablations:
        from graphbuilder.core.eval.metrics import AblationResult  # local import
        for name in [n.strip() for n in args.ablations.split(",") if n.strip()]:
            if name not in _ABLATIONS:
                logger.warning("unknown ablation %r — skipping", name)
                continue
            logger.info("=== running ablation: %s ===", name)
            ab_fn = _make_ask_fn(_ABLATIONS[name])
            _, ab_summary = await run_eval(
                ask_fn=ab_fn, gold=gold, concurrency=args.concurrency,
                on_progress=_progress,
            )
            ablation_results.append(
                AblationResult(name=name, summary=ab_summary,
                               description=_describe_ablation(name))
            )

    md_path = write_markdown_report(
        records, out_dir / "rag_eval.md", summary=summary,
        title="RAG eval report (live)",
        ablations=ablation_results or None,
    )
    logger.info("wrote %s", csv_path)
    logger.info("wrote %s", md_path)

    print()
    print(f"P@k:           {summary.precision_at_k:.3f}")
    print(f"R@k:           {summary.recall_at_k:.3f}")
    print(f"F1@k:          {summary.f1_at_k:.3f}")
    print(f"Context recall:{summary.context_recall:.3f}")
    if summary.answer_coverage is not None:
        print(f"Answer cov:    {summary.answer_coverage:.3f}")
    if summary.answer_faithfulness is not None:
        print(f"Faithfulness:  {summary.answer_faithfulness:.3f}")
    print(f"Latency p95:   {summary.latency_ms_p95:.0f} ms")
    print(f"Errors:        {summary.n_errors} / {summary.n_questions}")

    if ablation_results:
        print("\nAblations:")
        print(
            f"  {'config':<14} {'P':>6} {'R':>6} {'F1':>6} {'ctx':>6} "
            f"{'cov':>6} {'faith':>7} {'p95':>7}"
        )
        for ab in ablation_results:
            s = ab.summary
            cov = f"{s.answer_coverage:.3f}" if s.answer_coverage is not None else "  -  "
            faith = (
                f"{s.answer_faithfulness:.3f}"
                if s.answer_faithfulness is not None else "   -   "
            )
            print(
                f"  {ab.name:<14} "
                f"{s.precision_at_k:>6.3f} {s.recall_at_k:>6.3f} {s.f1_at_k:>6.3f} "
                f"{s.context_recall:>6.3f} {cov:>6} {faith:>7} {s.latency_ms_p95:>6.0f}ms"
            )

    if args.no_gate:
        return 0
    return _gate_against_targets(summary, args.baseline)


def _describe_ablation(name: str) -> str:
    """One-line description of a named ablation, for the markdown report."""
    return {
        "all_channels": "baseline — all channels + rerank + chunk promotion enabled",
        "vector_only":  "vector channel only; cypher + bm25 disabled",
        "bm25_only":    "BM25 channel only; cypher + vector disabled",
        "cypher_only":  "Cypher channel only; vector + bm25 disabled",
        "no_rerank":    "all channels enabled, cross-encoder rerank off",
        "no_chunks":    "no chunk promotion or neighbour expansion (entity-level only)",
        "with_authors": "default Person/Document/Organization blocklist disabled",
    }.get(name, "")


def _gate_against_targets(summary: EvalSummary, baseline_path: str) -> int:
    raw = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    targets = raw.get("live_targets", {})
    misses: list[str] = []

    def _check(name: str, value: float, floor: float) -> None:
        if value + 1e-9 < floor:
            misses.append(f"{name}={value:.3f} < target {floor:.3f}")

    _check("precision_at_k", summary.precision_at_k, targets.get("precision_at_k", 0.0))
    _check("recall_at_k", summary.recall_at_k, targets.get("recall_at_k", 0.0))
    _check("f1_at_k", summary.f1_at_k, targets.get("f1_at_k", 0.0))
    _check("context_recall", summary.context_recall, targets.get("context_recall", 0.0))
    if summary.answer_coverage is not None:
        _check("answer_coverage", summary.answer_coverage,
               targets.get("answer_coverage", 0.0))
    if summary.answer_faithfulness is not None:
        _check("answer_faithfulness", summary.answer_faithfulness,
               targets.get("answer_faithfulness", 0.0))
    p95_max = targets.get("latency_ms_p95_max")
    if p95_max is not None and summary.latency_ms_p95 > p95_max + 1e-6:
        misses.append(f"latency_ms_p95={summary.latency_ms_p95:.0f} > target {p95_max:.0f}")

    if misses:
        print()
        print("Live target floors missed:")
        for m in misses:
            print(f"  - {m}")
        return 1
    print("\nAll live target floors cleared.")
    return 0


def main() -> int:
    args = _build_argparser().parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
