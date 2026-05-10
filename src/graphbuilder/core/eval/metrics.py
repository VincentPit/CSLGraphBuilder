"""Metric math for the RAG eval harness (§9.2 of docs/RAG_QA_PLAN.md).

All functions here are pure: given the per-question records, they
return aggregated numbers. The runner produces records by calling the
QA service; this module never hits Neo4j or the LLM.

Why micro-/macro-averaging matters: a 100-question gold set with a
handful of "long" queries can be skewed if we just average per-query
P/R. We report **macro-average** (mean over questions) since each
question is curated equally. Latency we summarise as median + p95 to
match the SLO style of §9.2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

from .gold import GoldQuestion


# ----------------------------------------------------------------------
# Per-question record
# ----------------------------------------------------------------------

@dataclass
class EvalRecord:
    """One question's measurement.

    The harness builds these by calling the QA service and reading back
    the retrieved item ids + answer text. ``retrieved_ids`` are
    composite ``"<kind>:<id>"`` strings (matching the orchestrator's
    fusion key) so they line up with :meth:`GoldQuestion.all_gold_source_ids`.
    """

    question_id: str
    intent: Optional[str]
    retrieved_ids: List[str] = field(default_factory=list)
    gold_ids: List[str] = field(default_factory=list)
    answer: str = ""
    answer_substring_hit: Optional[bool] = None  # None = no substrings curated
    cited_indices: List[int] = field(default_factory=list)
    chunk_hit: Optional[bool] = None             # any cited chunk in gold chunks
    latency_ms: int = 0
    error: Optional[str] = None
    # Per-question metrics, computed once at construction time so the
    # CSV writer doesn't have to recompute and the smoke test can
    # assert against them directly.
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    f1_at_k: float = 0.0
    context_recall: float = 0.0   # 1.0 iff at least one retrieved id is gold
    # Faithfulness aggregate from the QA service (P8). ``None`` means
    # the answer had no scorable cited claims (refusals + empty
    # answers also surface as ``None`` so they don't drag the average).
    answer_faithfulness: Optional[float] = None

    @classmethod
    def from_run(
        cls,
        *,
        question: GoldQuestion,
        retrieved_ids: Sequence[str],
        answer: str,
        cited_indices: Sequence[int] = (),
        latency_ms: int = 0,
        error: Optional[str] = None,
        answer_faithfulness: Optional[float] = None,
    ) -> "EvalRecord":
        gold_set = question.all_gold_source_ids()
        retrieved_list = list(retrieved_ids)
        retrieved_set = set(retrieved_list)
        # Precision @ k uses the ACTUAL k that came back, not the
        # configured top_k — if the orchestrator returns fewer items
        # (e.g. small graph) we shouldn't penalise as if it returned
        # zeros to pad. Recall divides by gold size as usual.
        k = max(len(retrieved_list), 1)
        hit = retrieved_set & gold_set
        precision = len(hit) / k if retrieved_list else 0.0
        recall = (len(hit) / len(gold_set)) if gold_set else 0.0
        f1 = _harmonic_mean(precision, recall)

        # Context recall: did we retrieve at least one source the gold
        # marked as relevant? (§9.2 target ≥ 0.9)
        context_recall = 1.0 if hit else 0.0

        # Answer-substring coverage — any-of, case-insensitive.
        if question.gold_answer_substrings:
            ans_lower = (answer or "").lower()
            substring_hit = any(
                s.lower() in ans_lower for s in question.gold_answer_substrings
            )
        else:
            substring_hit = None

        # "Did we cite a gold chunk?" — useful when the gold set pins
        # the chunk but the entity matched. Computed even when no
        # chunks were cited so the column is always meaningful.
        if question.gold_chunk_ids:
            cited_chunk_ids = {
                rid.split(":", 1)[1]
                for idx in cited_indices
                if 1 <= idx <= len(retrieved_list)
                for rid in [retrieved_list[idx - 1]]
                if rid.startswith("chunk:")
            }
            chunk_hit = any(c in cited_chunk_ids for c in question.gold_chunk_ids)
        else:
            chunk_hit = None

        return cls(
            question_id=question.id,
            intent=question.intent,
            retrieved_ids=retrieved_list,
            gold_ids=sorted(gold_set),
            answer=answer or "",
            answer_substring_hit=substring_hit,
            cited_indices=list(cited_indices),
            chunk_hit=chunk_hit,
            latency_ms=int(latency_ms),
            error=error,
            precision_at_k=precision,
            recall_at_k=recall,
            f1_at_k=f1,
            context_recall=context_recall,
            answer_faithfulness=answer_faithfulness,
        )


# ----------------------------------------------------------------------
# Aggregated summary
# ----------------------------------------------------------------------

@dataclass
class EvalSummary:
    """Macro-averaged metrics across a gold-set run."""

    n_questions: int
    n_errors: int
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    context_recall: float
    answer_coverage: Optional[float]      # None when no question has substrings
    latency_ms_p50: float
    latency_ms_p95: float
    # P8 — averaged ``answer_faithfulness`` over records that produced
    # a score. ``None`` until the QA service ships per-claim verdicts
    # *and* at least one question yields a scorable answer.
    answer_faithfulness: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "n_questions": self.n_questions,
            "n_errors": self.n_errors,
            "precision_at_k": round(self.precision_at_k, 4),
            "recall_at_k": round(self.recall_at_k, 4),
            "f1_at_k": round(self.f1_at_k, 4),
            "context_recall": round(self.context_recall, 4),
            "answer_coverage": (
                round(self.answer_coverage, 4) if self.answer_coverage is not None else None
            ),
            "answer_faithfulness": (
                round(self.answer_faithfulness, 4)
                if self.answer_faithfulness is not None else None
            ),
            "latency_ms_p50": round(self.latency_ms_p50, 1),
            "latency_ms_p95": round(self.latency_ms_p95, 1),
        }


@dataclass
class AblationResult:
    """One row in an ablation matrix — the config name + the summary it produced."""

    name: str
    summary: EvalSummary
    description: Optional[str] = None


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------

def compute_metrics(records: Sequence[EvalRecord]) -> EvalSummary:
    """Aggregate per-question records into one :class:`EvalSummary`.

    Macro-average over successful records (records with ``error is None``).
    Erroring records still count toward ``n_questions`` and ``n_errors``
    so a regression that makes 50 % of questions throw can't pretend the
    F1 went up by silently dropping them.
    """
    if not records:
        return EvalSummary(
            n_questions=0,
            n_errors=0,
            precision_at_k=0.0,
            recall_at_k=0.0,
            f1_at_k=0.0,
            context_recall=0.0,
            answer_coverage=None,
            latency_ms_p50=0.0,
            latency_ms_p95=0.0,
            answer_faithfulness=None,
        )

    n = len(records)
    n_errors = sum(1 for r in records if r.error)

    # Errored records contribute zeros so the metric reflects "the
    # system answered correctly" rather than "the system answered
    # correctly *when it didn't crash*".
    precision = sum(r.precision_at_k for r in records) / n
    recall = sum(r.recall_at_k for r in records) / n
    f1 = sum(r.f1_at_k for r in records) / n
    context_recall = sum(r.context_recall for r in records) / n

    sub_records = [r for r in records if r.answer_substring_hit is not None]
    if sub_records:
        coverage = sum(1.0 for r in sub_records if r.answer_substring_hit) / len(sub_records)
    else:
        coverage = None

    # Faithfulness averages over the records that actually produced a
    # score. Refusals + empty/uncited answers are ``None`` and are
    # excluded so an honest "I cannot find this" doesn't drag the
    # number down — context recall already penalises those.
    faith_records = [r for r in records if r.answer_faithfulness is not None]
    if faith_records:
        faithfulness = sum(r.answer_faithfulness for r in faith_records) / len(faith_records)
    else:
        faithfulness = None

    latencies = [r.latency_ms for r in records if r.error is None]
    if latencies:
        p50 = _percentile(latencies, 50.0)
        p95 = _percentile(latencies, 95.0)
    else:
        p50 = 0.0
        p95 = 0.0

    return EvalSummary(
        n_questions=n,
        n_errors=n_errors,
        precision_at_k=precision,
        recall_at_k=recall,
        f1_at_k=f1,
        context_recall=context_recall,
        answer_coverage=coverage,
        latency_ms_p50=p50,
        latency_ms_p95=p95,
        answer_faithfulness=faithfulness,
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _harmonic_mean(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)


def _percentile(values: Iterable[int], pct: float) -> float:
    """Linear-interpolation percentile (matches NumPy's default)."""
    sorted_vals = sorted(int(v) for v in values)
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(sorted_vals[lo])
    frac = rank - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


__all__ = [
    "EvalRecord",
    "EvalSummary",
    "AblationResult",
    "compute_metrics",
]
