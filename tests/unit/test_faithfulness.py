"""Unit tests for the answer-faithfulness checker (P8 of docs/RAG_QA_PLAN.md)."""

from __future__ import annotations

import os
from typing import List, Optional

import pytest

os.environ.setdefault("LLM_API_KEY", "not-configured")

from graphbuilder.core.retrieval.faithfulness import (  # noqa: E402
    ClaimSpan,
    ClaimVerification,
    FaithfulnessChecker,
    FaithfulnessConfig,
    FaithfulnessResult,
    extract_claim_spans,
)
from graphbuilder.core.retrieval.models import (  # noqa: E402
    Channel,
    ItemKind,
    RetrievedItem,
)


# ---------------------------------------------------------------- helpers


def _item(label: str, *, chunk: Optional[str] = None, description: Optional[str] = None,
          kind: ItemKind = ItemKind.ENTITY) -> RetrievedItem:
    metadata = {"description": description} if description else {}
    return RetrievedItem(
        kind=kind, id=f"id_{label.lower()}", label=label,
        score_vector=0.9, score_rrf=0.5,
        chunk_preview=chunk,
        contributing_channels=[Channel.VECTOR_ENTITY],
        metadata=metadata,
    )


# ---------------------------------------------------------------- claim extraction


class TestExtractClaimSpans:
    def test_empty_answer(self):
        assert extract_claim_spans("") == []
        assert extract_claim_spans(None) == []

    def test_single_claim_with_one_citation(self):
        spans = extract_claim_spans("Imatinib inhibits BCR-ABL [1].")
        assert len(spans) == 1
        assert spans[0].text == "Imatinib inhibits BCR-ABL"
        assert spans[0].cited_indices == [1]

    def test_multiple_claims_each_cited(self):
        text = "Imatinib targets BCR-ABL [1]. It also hits KIT [2]."
        spans = extract_claim_spans(text)
        # Two cited spans; the trailing "." after [2] is consumed as
        # punctuation belonging to the prior claim.
        cited = [s for s in spans if s.cited_indices]
        assert [s.cited_indices for s in cited] == [[1], [2]]
        assert "BCR-ABL" in cited[0].text
        assert "KIT" in cited[1].text

    def test_clustered_citations_collapse(self):
        # Adjacent markers separated only by whitespace become one span.
        spans = extract_claim_spans("Drug X inhibits target Y [1][2] [3].")
        cited = [s.cited_indices for s in spans if s.cited_indices]
        assert cited == [[1, 2, 3]]

    def test_citations_separated_by_prose_split(self):
        spans = extract_claim_spans("Claim A [1]. And claim B [2].")
        # Spans with citations:
        cited = [s for s in spans if s.cited_indices]
        assert [s.cited_indices for s in cited] == [[1], [2]]

    def test_uncited_tail_kept_with_empty_indices(self):
        spans = extract_claim_spans("Claim A [1]. Uncited tail.")
        tail = [s for s in spans if not s.cited_indices]
        assert len(tail) == 1
        assert tail[0].text == "Uncited tail."

    def test_no_citations_returns_one_uncited_span(self):
        spans = extract_claim_spans("This answer has no citations at all.")
        assert len(spans) == 1
        assert spans[0].cited_indices == []
        assert "no citations" in spans[0].text

    def test_duplicate_indices_in_cluster_dedup(self):
        spans = extract_claim_spans("Claim [1][1][2].")
        cited = [s.cited_indices for s in spans if s.cited_indices]
        assert cited == [[1, 2]]


# ---------------------------------------------------------------- lexical scoring


@pytest.mark.asyncio
class TestLexicalChecker:
    """Lexical-only path — no LLM service supplied."""

    async def test_high_overlap_passes(self):
        checker = FaithfulnessChecker(config=FaithfulnessConfig())
        sources = [
            _item("BCR-ABL", chunk="BCR-ABL is a fusion tyrosine kinase."),
        ]
        result = await checker.check(
            answer="BCR-ABL is a fusion tyrosine kinase [1].",
            sources=sources,
        )
        assert result.overall_score is not None
        assert result.overall_score >= 0.7
        assert result.failed_claims == 0
        assert result.claims[0].method == "text_match"

    async def test_low_overlap_fails(self):
        checker = FaithfulnessChecker(config=FaithfulnessConfig())
        sources = [_item("Aspirin", chunk="Aspirin reduces inflammation.")]
        result = await checker.check(
            answer="Imatinib inhibits BCR-ABL kinase activity [1].",
            sources=sources,
        )
        assert result.overall_score is not None
        assert result.overall_score < 0.5
        assert result.failed_claims == 1

    async def test_uncited_tail_excluded_from_score(self):
        """Trailing prose after the last [n] is uncited and shouldn't
        drag the overall score down.

        Preamble *before* the first citation belongs to the first cited
        claim — that's the more useful split because typical answers
        place context first and the citation marker at the end.
        """
        checker = FaithfulnessChecker(config=FaithfulnessConfig())
        sources = [_item("BCR-ABL", chunk="BCR-ABL is a kinase.")]
        result = await checker.check(
            answer="BCR-ABL is a kinase [1] More uncited prose afterwards",
            sources=sources,
        )
        kinds = {c.method for c in result.claims}
        assert "uncited" in kinds
        assert "text_match" in kinds
        # Overall score reflects only the cited claim.
        assert result.overall_score is not None
        assert result.overall_score >= 0.5

    async def test_out_of_range_citation_treated_as_uncited(self):
        checker = FaithfulnessChecker(config=FaithfulnessConfig())
        sources = [_item("BCR-ABL", chunk="BCR-ABL is a kinase.")]
        result = await checker.check(
            answer="BCR-ABL is a kinase [9].",
            sources=sources,
        )
        assert result.overall_score is None  # no scorable claims
        assert result.claims[0].method == "uncited"
        assert "out of range" in result.claims[0].reasoning.lower()

    async def test_refusal_short_circuits_to_perfect_score(self):
        checker = FaithfulnessChecker(config=FaithfulnessConfig())
        result = await checker.check(
            answer="I cannot find this in the knowledge base.",
            sources=[],
        )
        assert result.overall_score == 1.0
        assert result.is_refusal is True
        assert result.claims[0].method == "refusal"

    async def test_empty_answer(self):
        checker = FaithfulnessChecker(config=FaithfulnessConfig())
        result = await checker.check(answer="", sources=[])
        assert result.overall_score is None
        assert result.claims == []

    async def test_multiple_cited_sources_pooled(self):
        """A claim citing two sources gets context from both."""
        checker = FaithfulnessChecker(config=FaithfulnessConfig())
        sources = [
            _item("Imatinib", chunk="Imatinib is a kinase inhibitor."),
            _item("BCR-ABL", chunk="BCR-ABL is a fusion tyrosine kinase."),
        ]
        result = await checker.check(
            answer="Imatinib inhibits BCR-ABL kinase [1][2].",
            sources=sources,
        )
        assert result.overall_score is not None
        assert result.overall_score >= 0.6
        assert result.claims[0].cited_indices == [1, 2]

    async def test_description_falls_back_when_no_chunk_preview(self):
        """An entity with description but no chunk_preview still gets context."""
        checker = FaithfulnessChecker(config=FaithfulnessConfig())
        sources = [_item("BCR-ABL", description="BCR-ABL is a fusion kinase.")]
        result = await checker.check(
            answer="BCR-ABL is a fusion kinase [1].",
            sources=sources,
        )
        assert result.overall_score is not None
        assert result.overall_score >= 0.7


# ---------------------------------------------------------------- LLM escalation


class _FakeLLM:
    """LLM that returns a scripted JSON verdict."""

    def __init__(self, *, verdict: str, confidence: float, reasoning: str = ""):
        self._payload = (
            f'{{"verdict": "{verdict}", "confidence": {confidence}, '
            f'"reasoning": "{reasoning}"}}'
        )
        self.calls: List[dict] = []

    async def generate_text(self, *, prompt, system_prompt=None,
                            temperature: float = 0.0, max_tokens: int = 200):
        self.calls.append({"prompt": prompt})
        return self._payload


@pytest.mark.asyncio
class TestLLMEscalation:
    async def test_borderline_claim_escalates_when_enabled(self):
        # Lexical overlap ~0.5 (in the inconclusive band).
        checker = FaithfulnessChecker(
            config=FaithfulnessConfig(enable_llm_escalation=True),
            llm_service=_FakeLLM(verdict="supported", confidence=0.95),
        )
        sources = [_item("X", chunk="A short context with one matching word: kinase.")]
        result = await checker.check(
            answer="Drug X inhibits kinase pathway with strong affinity [1].",
            sources=sources,
        )
        # The LLM said supported@0.95 — that should win over the
        # lower lexical signal.
        cited = next(c for c in result.claims if c.method in ("llm", "text_match"))
        assert cited.method == "llm"
        assert cited.confidence == pytest.approx(0.95)
        assert cited.escalated_to_llm is True

    async def test_unsupported_verdict_inverts_score(self):
        checker = FaithfulnessChecker(
            config=FaithfulnessConfig(enable_llm_escalation=True),
            llm_service=_FakeLLM(verdict="unsupported", confidence=0.9),
        )
        sources = [_item("X", chunk="some context")]
        result = await checker.check(
            answer="Wildly fabricated claim about unrelated entities [1].",
            sources=sources,
        )
        assert result.claims[0].confidence <= 0.2

    async def test_llm_off_keeps_lexical(self):
        checker = FaithfulnessChecker(
            config=FaithfulnessConfig(enable_llm_escalation=False),
            llm_service=_FakeLLM(verdict="supported", confidence=0.99),
        )
        sources = [_item("X", chunk="unrelated context bytes")]
        result = await checker.check(
            answer="Imatinib targets BCR-ABL kinase [1].",
            sources=sources,
        )
        # No LLM call.
        assert checker._llm.calls == []
        assert result.claims[0].method == "text_match"

    async def test_llm_failure_falls_back_to_lexical(self):
        class _Boom:
            async def generate_text(self, **kwargs):
                raise RuntimeError("provider outage")

        checker = FaithfulnessChecker(
            config=FaithfulnessConfig(enable_llm_escalation=True),
            llm_service=_Boom(),
        )
        sources = [_item("X", chunk="weak context")]
        result = await checker.check(
            answer="Imatinib targets BCR-ABL kinase [1].",
            sources=sources,
        )
        # Falls back to text_match — better a rough number than no number.
        assert result.claims[0].method == "text_match"
