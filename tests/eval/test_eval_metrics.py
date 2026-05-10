"""Unit tests for the metric math in `graphbuilder.core.eval.metrics`.

These tests are pure: they construct ``GoldQuestion`` + ``EvalRecord``
fixtures by hand and verify ``compute_metrics`` returns the right
aggregates. No QAService, no orchestrator, no I/O.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("LLM_API_KEY", "not-configured")

from graphbuilder.core.eval.gold import GoldQuestion, load_gold  # noqa: E402
from graphbuilder.core.eval.metrics import (  # noqa: E402
    EvalRecord,
    compute_metrics,
    _harmonic_mean,
    _percentile,
)


# ---------------------------------------------------------------- helpers

def _gold(qid: str, **kwargs) -> GoldQuestion:
    """Build a GoldQuestion with sensible defaults for tests."""
    return GoldQuestion(
        id=qid,
        question=kwargs.get("question", f"q? {qid}"),
        intent=kwargs.get("intent"),
        gold_entity_ids=kwargs.get("gold_entity_ids", []),
        gold_relationship_ids=kwargs.get("gold_relationship_ids", []),
        gold_chunk_ids=kwargs.get("gold_chunk_ids", []),
        gold_answer_substrings=kwargs.get("gold_answer_substrings", []),
    )


# ---------------------------------------------------------------- per-record

class TestEvalRecordFromRun:
    def test_perfect_recall_and_precision(self):
        q = _gold("q1", gold_entity_ids=["a", "b"])
        r = EvalRecord.from_run(
            question=q,
            retrieved_ids=["entity:a", "entity:b"],
            answer="alpha and beta",
        )
        assert r.precision_at_k == 1.0
        assert r.recall_at_k == 1.0
        assert r.f1_at_k == 1.0
        assert r.context_recall == 1.0

    def test_partial_overlap(self):
        q = _gold("q2", gold_entity_ids=["a", "b", "c"])
        r = EvalRecord.from_run(
            question=q,
            retrieved_ids=["entity:a", "entity:x"],
            answer="",
        )
        # 1 of 2 retrieved are correct; 1 of 3 gold are recalled.
        assert r.precision_at_k == 0.5
        assert r.recall_at_k == pytest.approx(1 / 3)
        assert r.f1_at_k == pytest.approx(2 * 0.5 * (1 / 3) / (0.5 + 1 / 3))
        assert r.context_recall == 1.0

    def test_no_retrieval(self):
        q = _gold("q3", gold_entity_ids=["a"])
        r = EvalRecord.from_run(question=q, retrieved_ids=[], answer="")
        assert r.precision_at_k == 0.0
        assert r.recall_at_k == 0.0
        assert r.f1_at_k == 0.0
        assert r.context_recall == 0.0

    def test_empty_gold_doesnt_divide_by_zero(self):
        q = _gold("q4")  # no gold ids at all
        r = EvalRecord.from_run(
            question=q, retrieved_ids=["entity:a"], answer="something",
        )
        # Recall is undefined → reported as 0; precision is 0 because
        # nothing in the empty gold set could match.
        assert r.recall_at_k == 0.0
        assert r.precision_at_k == 0.0
        assert r.f1_at_k == 0.0

    def test_kind_namespacing(self):
        """An entity id and a chunk id with the same suffix must not
        collide — composite ``"<kind>:<id>"`` keying is what makes the
        gold contract unambiguous."""
        q = _gold("q5", gold_entity_ids=["x"], gold_chunk_ids=["x"])
        r = EvalRecord.from_run(
            question=q,
            retrieved_ids=["entity:x"],   # only matches the entity, not the chunk
            answer="",
        )
        assert r.precision_at_k == 1.0
        assert r.recall_at_k == 0.5  # 1 of 2 distinct gold sources retrieved

    def test_answer_substring_any_of(self):
        q = _gold("q6", gold_answer_substrings=["BCR-ABL", "KIT"])
        r = EvalRecord.from_run(
            question=q,
            retrieved_ids=["entity:a"],
            answer="Imatinib inhibits the kit receptor.",
        )
        assert r.answer_substring_hit is True  # case-insensitive

    def test_answer_substring_miss(self):
        q = _gold("q7", gold_answer_substrings=["foo"])
        r = EvalRecord.from_run(
            question=q,
            retrieved_ids=[],
            answer="bar baz",
        )
        assert r.answer_substring_hit is False

    def test_answer_substring_none_when_not_curated(self):
        q = _gold("q8")
        r = EvalRecord.from_run(question=q, retrieved_ids=[], answer="anything")
        assert r.answer_substring_hit is None

    def test_chunk_hit_uses_cited_chunks(self):
        q = _gold("q9", gold_chunk_ids=["chunk_42"])
        r = EvalRecord.from_run(
            question=q,
            retrieved_ids=["entity:e1", "chunk:chunk_42", "chunk:chunk_99"],
            answer="X [2] and Y",
            cited_indices=[2],  # only chunk_42 is cited
        )
        assert r.chunk_hit is True

    def test_chunk_hit_false_when_only_cited_chunk_is_wrong(self):
        q = _gold("q10", gold_chunk_ids=["chunk_42"])
        r = EvalRecord.from_run(
            question=q,
            retrieved_ids=["chunk:chunk_42", "chunk:chunk_99"],
            answer="text [2]",
            cited_indices=[2],  # cites the wrong chunk
        )
        assert r.chunk_hit is False


# ---------------------------------------------------------------- aggregation

class TestComputeMetrics:
    def test_macro_average(self):
        q1 = _gold("q1", gold_entity_ids=["a"])
        q2 = _gold("q2", gold_entity_ids=["b", "c"])
        r1 = EvalRecord.from_run(question=q1, retrieved_ids=["entity:a"], answer="", latency_ms=100)
        r2 = EvalRecord.from_run(
            question=q2, retrieved_ids=["entity:b", "entity:x"], answer="", latency_ms=200,
        )
        summary = compute_metrics([r1, r2])
        # Macro: mean of (1.0, 0.5) for precision = 0.75
        assert summary.precision_at_k == pytest.approx(0.75)
        # Macro: mean of (1.0, 0.5) for recall = 0.75
        assert summary.recall_at_k == pytest.approx(0.75)
        assert summary.n_questions == 2
        assert summary.n_errors == 0
        assert summary.latency_ms_p50 == pytest.approx(150.0)
        assert summary.latency_ms_p95 == pytest.approx(195.0)

    def test_errors_count_as_zeros(self):
        q1 = _gold("q1", gold_entity_ids=["a"])
        q2 = _gold("q2", gold_entity_ids=["b"])
        good = EvalRecord.from_run(question=q1, retrieved_ids=["entity:a"], answer="")
        bad = EvalRecord.from_run(
            question=q2, retrieved_ids=[], answer="", error="connection refused",
        )
        summary = compute_metrics([good, bad])
        assert summary.n_errors == 1
        # 1.0 + 0.0 = 0.5 mean precision — the error penalises, doesn't get dropped.
        assert summary.precision_at_k == 0.5

    def test_empty_records(self):
        summary = compute_metrics([])
        assert summary.n_questions == 0
        assert summary.precision_at_k == 0.0
        assert summary.answer_coverage is None

    def test_answer_coverage_excludes_questions_without_substrings(self):
        with_sub = _gold("q1", gold_answer_substrings=["foo"])
        without = _gold("q2")
        r1 = EvalRecord.from_run(question=with_sub, retrieved_ids=[], answer="foo bar")
        r2 = EvalRecord.from_run(question=without, retrieved_ids=[], answer="anything")
        summary = compute_metrics([r1, r2])
        # Coverage averages over the curated subset only — q2 doesn't count.
        assert summary.answer_coverage == 1.0

    def test_answer_coverage_none_when_no_question_has_substrings(self):
        q = _gold("q1")
        r = EvalRecord.from_run(question=q, retrieved_ids=[], answer="")
        summary = compute_metrics([r])
        assert summary.answer_coverage is None

    def test_latency_ignores_errored_records(self):
        q1 = _gold("q1", gold_entity_ids=["a"])
        q2 = _gold("q2", gold_entity_ids=["b"])
        r1 = EvalRecord.from_run(question=q1, retrieved_ids=["entity:a"], answer="", latency_ms=100)
        r2 = EvalRecord.from_run(
            question=q2, retrieved_ids=[], answer="", latency_ms=999_999, error="boom",
        )
        summary = compute_metrics([r1, r2])
        # The 999_999 from the error is excluded so it can't make p95 misleading.
        assert summary.latency_ms_p95 == 100.0


# ---------------------------------------------------------------- helpers

class TestHelpers:
    def test_harmonic_mean(self):
        assert _harmonic_mean(1.0, 1.0) == 1.0
        assert _harmonic_mean(0.5, 0.5) == 0.5
        assert _harmonic_mean(0.0, 1.0) == 0.0
        assert _harmonic_mean(1.0, 0.0) == 0.0

    def test_percentile_single_value(self):
        assert _percentile([42], 95.0) == 42.0
        assert _percentile([], 95.0) == 0.0

    def test_percentile_interpolation(self):
        # 0..100 → p50 should be 50; p95 should be 95.
        vs = list(range(101))
        assert _percentile(vs, 50.0) == 50.0
        assert _percentile(vs, 95.0) == 95.0


# ---------------------------------------------------------------- gold loader

class TestLoadGold:
    def test_loads_seed_yaml(self, tmp_path):
        from pathlib import Path
        seed = Path(__file__).parent / "rag_gold.yaml"
        gold = load_gold(seed)
        assert len(gold) >= 4
        assert all(q.id and q.question for q in gold)

    def test_rejects_unknown_keys(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "- id: q1\n  question: x\n  bogus: 1\n", encoding="utf-8",
        )
        with pytest.raises(ValueError, match="unknown keys"):
            load_gold(bad)

    def test_rejects_duplicate_ids(self, tmp_path):
        bad = tmp_path / "dup.yaml"
        bad.write_text(
            "- id: q1\n  question: a\n- id: q1\n  question: b\n", encoding="utf-8",
        )
        with pytest.raises(ValueError, match="duplicate"):
            load_gold(bad)

    def test_rejects_missing_required(self, tmp_path):
        bad = tmp_path / "missing.yaml"
        bad.write_text("- id: q1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing or empty 'question'"):
            load_gold(bad)

    def test_loads_json(self, tmp_path):
        import json
        path = tmp_path / "g.json"
        path.write_text(
            json.dumps([{"id": "q1", "question": "hi", "gold_entity_ids": ["a"]}]),
            encoding="utf-8",
        )
        gold = load_gold(path)
        assert gold[0].gold_entity_ids == ["a"]
