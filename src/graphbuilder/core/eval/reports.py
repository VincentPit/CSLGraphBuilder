"""CSV + markdown report writers for the eval harness.

Two outputs per run:

- ``rag_eval.csv``      — one row per question, machine-readable.
- ``rag_eval.md``       — human-readable summary + per-question table.

Both are produced from the same :class:`EvalRecord` list so the numbers
agree by construction. Markdown is the "ship it" artifact for PR
descriptions; CSV is what we diff between runs.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .metrics import AblationResult, EvalRecord, EvalSummary, compute_metrics


_CSV_FIELDS = [
    "question_id",
    "intent",
    "n_gold",
    "n_retrieved",
    "n_hit",
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "context_recall",
    "answer_substring_hit",
    "chunk_hit",
    "latency_ms",
    "error",
]


def write_csv_report(records: Sequence[EvalRecord], path: str | Path) -> Path:
    """Write a per-question CSV. Returns the resolved output path."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for r in records:
            gold_set = set(r.gold_ids)
            retrieved_set = set(r.retrieved_ids)
            writer.writerow({
                "question_id": r.question_id,
                "intent": r.intent or "",
                "n_gold": len(gold_set),
                "n_retrieved": len(retrieved_set),
                "n_hit": len(gold_set & retrieved_set),
                "precision_at_k": f"{r.precision_at_k:.4f}",
                "recall_at_k": f"{r.recall_at_k:.4f}",
                "f1_at_k": f"{r.f1_at_k:.4f}",
                "context_recall": f"{r.context_recall:.4f}",
                "answer_substring_hit": _tribool(r.answer_substring_hit),
                "chunk_hit": _tribool(r.chunk_hit),
                "latency_ms": r.latency_ms,
                "error": r.error or "",
            })
    return out_path


def write_markdown_report(
    records: Sequence[EvalRecord],
    path: str | Path,
    *,
    summary: Optional[EvalSummary] = None,
    title: str = "RAG eval report",
    ablations: Optional[Sequence[AblationResult]] = None,
    baseline: Optional[EvalSummary] = None,
) -> Path:
    """Write a human-readable markdown summary.

    When ``ablations`` is provided, a comparison table is added under
    the headline summary. When ``baseline`` is provided, a delta
    column appears so reviewers can eyeball regressions at a glance.
    """
    if summary is None:
        summary = compute_metrics(records)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    buf.write(f"# {title}\n\n")
    buf.write(f"_Generated {now}_\n\n")
    buf.write(f"- **Questions:** {summary.n_questions}\n")
    buf.write(f"- **Errors:** {summary.n_errors}\n\n")

    buf.write("## Headline metrics\n\n")
    buf.write("| Metric | Value | Target | Status |\n")
    buf.write("|---|---|---|---|\n")
    _write_headline_row(buf, "Precision @ k", summary.precision_at_k, 0.5, baseline_value(baseline, "precision_at_k"))
    _write_headline_row(buf, "Recall @ k", summary.recall_at_k, 0.7, baseline_value(baseline, "recall_at_k"))
    _write_headline_row(buf, "F1 @ k", summary.f1_at_k, 0.55, baseline_value(baseline, "f1_at_k"))
    _write_headline_row(buf, "Context recall", summary.context_recall, 0.9, baseline_value(baseline, "context_recall"))
    if summary.answer_coverage is not None:
        _write_headline_row(
            buf, "Answer coverage", summary.answer_coverage, 0.8,
            baseline_value(baseline, "answer_coverage"),
        )
    buf.write(f"| Latency p50 | {summary.latency_ms_p50:.0f} ms | — | |\n")
    buf.write(f"| Latency p95 | {summary.latency_ms_p95:.0f} ms | ≤ 6000 ms | "
              f"{'✅' if summary.latency_ms_p95 <= 6000 else '⚠️'} |\n\n")

    if ablations:
        buf.write("## Ablations\n\n")
        buf.write("| Config | P@k | R@k | F1@k | Ctx recall | Cov | p95 (ms) |\n")
        buf.write("|---|---|---|---|---|---|---|\n")
        for ab in ablations:
            s = ab.summary
            cov = f"{s.answer_coverage:.3f}" if s.answer_coverage is not None else "—"
            buf.write(
                f"| {ab.name} | {s.precision_at_k:.3f} | {s.recall_at_k:.3f} | "
                f"{s.f1_at_k:.3f} | {s.context_recall:.3f} | {cov} | "
                f"{s.latency_ms_p95:.0f} |\n"
            )
            if ab.description:
                # Inline description as a small italic line so the table
                # stays scannable.
                buf.write(f"|   _{ab.description}_ | | | | | | |\n")
        buf.write("\n")

    buf.write("## Per-question results\n\n")
    buf.write("| ID | Intent | P | R | F1 | Ctx | Sub | Lat (ms) | Note |\n")
    buf.write("|---|---|---|---|---|---|---|---|---|\n")
    for r in records:
        note = r.error if r.error else ""
        if note and len(note) > 60:
            note = note[:57] + "…"
        buf.write(
            f"| {r.question_id} | {r.intent or '-'} | "
            f"{r.precision_at_k:.2f} | {r.recall_at_k:.2f} | {r.f1_at_k:.2f} | "
            f"{r.context_recall:.0f} | {_tribool(r.answer_substring_hit)} | "
            f"{r.latency_ms} | {note} |\n"
        )

    out_path.write_text(buf.getvalue(), encoding="utf-8")
    return out_path


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _write_headline_row(buf: io.StringIO, name: str, value: float, target: float,
                        baseline: Optional[float]) -> None:
    """Write one headline-metric row with a status emoji and optional
    delta against a pinned baseline. Status compares against the §9.2
    target — a green check means we're at or above the documented goal."""
    status = "✅" if value >= target else "⚠️"
    delta = ""
    if baseline is not None:
        diff = value - baseline
        sign = "+" if diff >= 0 else ""
        delta = f" ({sign}{diff:.3f} vs baseline)"
    buf.write(f"| {name} | {value:.3f}{delta} | ≥ {target:.2f} | {status} |\n")


def baseline_value(baseline: Optional[EvalSummary], field: str) -> Optional[float]:
    if baseline is None:
        return None
    return getattr(baseline, field, None)


def _tribool(val: Optional[bool]) -> str:
    if val is None:
        return "-"
    return "yes" if val else "no"


__all__ = ["write_csv_report", "write_markdown_report"]
