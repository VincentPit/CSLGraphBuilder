"""RAG evaluation harness — Precision / Recall / F1 against a gold set
(P13 of docs/RAG_QA_PLAN.md, §9).

Public surface:

- :class:`GoldQuestion`       — one curated question with expected sources.
- :func:`load_gold`           — YAML/JSON loader + validator.
- :class:`EvalRecord`         — per-question result the runner emits.
- :class:`EvalSummary`        — aggregated metrics across the whole gold set.
- :func:`compute_metrics`     — pure metric math given a list of records.
- :func:`run_eval`            — async runner: takes ``ask(query) -> AskLike``
                                and a gold set, returns records + summary.
- :func:`write_csv_report`    — per-question CSV.
- :func:`write_markdown_report` — human-readable summary report.

The runner is intentionally decoupled from FastAPI / Neo4j: callers pass
a ``Callable[[str], Awaitable[AskLike]]`` so the same harness drives the
hermetic CI smoke test, ablation runs against a local QAService, and the
``run_rag_eval.py`` CLI that hits a live ``/qa/ask`` endpoint.
"""

from .gold import GoldQuestion, load_gold
from .metrics import EvalRecord, EvalSummary, AblationResult, compute_metrics
from .reports import write_csv_report, write_markdown_report
from .runner import run_eval

__all__ = [
    "GoldQuestion",
    "load_gold",
    "EvalRecord",
    "EvalSummary",
    "AblationResult",
    "compute_metrics",
    "run_eval",
    "write_csv_report",
    "write_markdown_report",
]
