"""Eval runner — drives a callable across a gold set and collects records.

The runner is intentionally transport-agnostic. Callers pass an
``ask_fn(query) -> AskLike`` coroutine; the harness turns each gold
question into one :class:`~graphbuilder.core.eval.metrics.EvalRecord`.

This decoupling is what lets the same code drive:

- **The hermetic CI smoke test** — ``ask_fn`` wraps a local
  :class:`QAService` with stub channels.
- **Ablation runs** — ``ask_fn`` flips ``RetrievalConfig`` knobs between
  runs (see ``run_ablation``).
- **The CLI runner** (``tests/eval/run_rag_eval.py``) — ``ask_fn`` POSTs
  to ``/qa/ask`` over HTTP.

Any object with the duck-typed shape produced by
:class:`graphbuilder.core.retrieval.qa_service.AskResult` works as
``AskLike``: ``.answer``, ``.sources`` (each with ``.kind.value`` +
``.id``), ``.cited_source_indices``, ``.latency_ms``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, List, Optional, Protocol, Sequence

from .gold import GoldQuestion
from .metrics import AblationResult, EvalRecord, EvalSummary, compute_metrics


logger = logging.getLogger("graphbuilder.qa.eval")


class _SourceLike(Protocol):
    id: str
    @property
    def kind(self) -> Any: ...


class AskLike(Protocol):
    """Subset of :class:`AskResult` the runner needs.

    ``faithfulness`` is duck-typed (any object with an
    ``overall_score`` attribute) so the live HTTP runner can pass a
    plain dict-wrapping shim and the in-process AskResult can pass its
    ``FaithfulnessResult`` dataclass. Missing or ``None`` is fine —
    the runner reads the score defensively.
    """

    answer: str
    sources: Sequence[_SourceLike]
    cited_source_indices: Sequence[int]
    latency_ms: int


AskFn = Callable[[str], Awaitable[AskLike]]


# ----------------------------------------------------------------------
# Single-run driver
# ----------------------------------------------------------------------

async def run_eval(
    *,
    ask_fn: AskFn,
    gold: Sequence[GoldQuestion],
    concurrency: int = 1,
    on_progress: Optional[Callable[[int, int, EvalRecord], None]] = None,
) -> tuple[List[EvalRecord], EvalSummary]:
    """Run ``ask_fn`` against every question in ``gold`` and aggregate.

    ``concurrency`` defaults to 1 — sequential runs are deterministic
    and the LLM/Neo4j load on a single test box rarely benefits from
    parallelism. Bump it for live-API ablations against staging.

    The runner never raises on a per-question failure: any exception
    becomes ``EvalRecord(error=...)`` so an aggregate run still completes
    and the report shows which questions broke.
    """
    if not gold:
        raise ValueError("gold set is empty")

    sem = asyncio.Semaphore(max(1, concurrency))
    results: List[Optional[EvalRecord]] = [None] * len(gold)

    async def worker(idx: int, q: GoldQuestion) -> None:
        async with sem:
            record = await _run_one(ask_fn, q)
        results[idx] = record
        if on_progress is not None:
            on_progress(idx + 1, len(gold), record)

    await asyncio.gather(*(worker(i, q) for i, q in enumerate(gold)))
    records: List[EvalRecord] = [r for r in results if r is not None]
    summary = compute_metrics(records)
    return records, summary


async def _run_one(ask_fn: AskFn, question: GoldQuestion) -> EvalRecord:
    t0 = time.perf_counter()
    try:
        result = await ask_fn(question.question)
    except Exception as exc:  # noqa: BLE001 — caught for the report
        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.warning("eval question %s raised: %s", question.id, exc)
        return EvalRecord.from_run(
            question=question,
            retrieved_ids=[],
            answer="",
            cited_indices=[],
            latency_ms=latency_ms,
            error=str(exc),
        )

    # Prefer the latency the service reports (it includes LLM time even
    # when the harness's own stopwatch undercounts due to mocking), but
    # fall back to wall-clock when the field is missing.
    latency_ms = (
        int(getattr(result, "latency_ms", 0))
        or int((time.perf_counter() - t0) * 1000)
    )

    retrieved_ids = [
        f"{_kind_value(s)}:{s.id}" for s in (getattr(result, "sources", []) or [])
    ]
    cited = list(getattr(result, "cited_source_indices", []) or [])
    answer = getattr(result, "answer", "") or ""
    faithfulness_score = _extract_faithfulness_score(result)

    return EvalRecord.from_run(
        question=question,
        retrieved_ids=retrieved_ids,
        answer=answer,
        cited_indices=cited,
        latency_ms=latency_ms,
        answer_faithfulness=faithfulness_score,
    )


def _extract_faithfulness_score(result: Any) -> Optional[float]:
    """Pull the aggregate faithfulness score off whatever shape we got.

    In-process: ``AskResult.faithfulness`` is a dataclass with
    ``overall_score``. Live HTTP: the wrapper exposes a ``raw`` payload
    dict whose ``faithfulness`` key carries the same field. Missing,
    ``None``, or a non-numeric value all collapse to ``None`` so a
    transport quirk never invents a metric.
    """
    fr = getattr(result, "faithfulness", None)
    if fr is None:
        return None
    score = getattr(fr, "overall_score", None)
    if score is None and isinstance(fr, dict):
        score = fr.get("overall_score")
    if score is None:
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def _kind_value(source: Any) -> str:
    """Pull the kind value off either an enum field or a raw string.

    Live API responses come back as Pydantic models with a string
    ``kind``; in-process AskResult objects expose an :class:`ItemKind`
    enum with ``.value``. Handle both without guessing.
    """
    kind = getattr(source, "kind", None)
    if kind is None:
        return "unknown"
    val = getattr(kind, "value", None)
    return val if isinstance(val, str) else str(kind)


# ----------------------------------------------------------------------
# Ablation matrix
# ----------------------------------------------------------------------

async def run_ablation(
    *,
    gold: Sequence[GoldQuestion],
    ask_fn_factory: Callable[[str], AskFn],
    configurations: Sequence[tuple[str, str]],
    concurrency: int = 1,
) -> List[AblationResult]:
    """Run the same gold set under several configurations and return one
    :class:`AblationResult` per config.

    ``configurations`` is a list of ``(name, description)`` pairs; the
    factory is invoked with the *name* and must return an ``ask_fn``
    bound to the corresponding settings. Names are caller-defined
    (``"vector_only"``, ``"all_channels"``, ``"no_rerank"``, …) and
    show up unchanged in the markdown report so the wiring stays in one
    place.
    """
    out: List[AblationResult] = []
    for name, description in configurations:
        ask_fn = ask_fn_factory(name)
        _, summary = await run_eval(
            ask_fn=ask_fn, gold=gold, concurrency=concurrency,
        )
        out.append(AblationResult(name=name, summary=summary, description=description))
    return out


__all__ = ["AskLike", "AskFn", "run_eval", "run_ablation"]
