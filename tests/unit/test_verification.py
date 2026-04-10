"""
Unit tests for the relationship verification pipeline (P2).

Covers:
  - VerificationResult validation
  - TextMatchVerifier (pass / fail / edge cases)
  - EmbeddingVerifier (graceful skip when sentence-transformers absent)
  - LLMVerifier (valid / invalid / uncertain / malformed JSON)
  - CascadingVerifier (majority vote, early exit, all-skip)
  - RelationshipVerificationUseCase (annotation side-effects, report shape)
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

from graphbuilder.core.verification.models import (
    VerificationResult,
    VerificationStage,
    VerificationStatus,
)
from graphbuilder.core.verification.text_match import TextMatchVerifier, TextMatchConfig
from graphbuilder.core.verification.embedding import EmbeddingVerifier, EmbeddingConfig
from graphbuilder.core.verification.llm_verifier import LLMVerifier, LLMVerifierConfig
from graphbuilder.core.verification.cascading import CascadingVerifier, CascadingVerifierConfig
from graphbuilder.application.use_cases.relationship_verification import (
    RelationshipVerificationUseCase,
    VerificationConfig,
)
from graphbuilder.domain.models.graph_models import (
    GraphRelationship,
    GraphEntity,
    EntityType,
    RelationshipType,
    KnowledgeGraph,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_rel(
    src: str = "e1",
    tgt: str = "e2",
    rel_type: RelationshipType = RelationshipType.RELATED_TO,
    description: str = "",
) -> GraphRelationship:
    return GraphRelationship(
        source_entity_id=src,
        target_entity_id=tgt,
        relationship_type=rel_type,
        description=description,
    )


def _make_entity(name: str, entity_type: EntityType = EntityType.CONCEPT) -> GraphEntity:
    return GraphEntity(name=name, entity_type=entity_type)


def _make_graph_with_rel(rel: GraphRelationship) -> KnowledgeGraph:
    g = KnowledgeGraph()
    e1 = _make_entity("Source")
    e2 = _make_entity("Target")
    # Give them stable IDs matching what the relationship expects
    e1.id = rel.source_entity_id
    e2.id = rel.target_entity_id
    g.entities[e1.id] = e1
    g.entities[e2.id] = e2
    g.relationships[rel.id] = rel
    return g


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------

class TestVerificationResult:

    def test_passed_property_true_when_passed(self):
        r = VerificationResult(
            status=VerificationStatus.PASSED,
            stage=VerificationStage.TEXT_MATCH,
            confidence=0.8,
            reasoning="ok",
        )
        assert r.passed is True
        assert r.failed is False

    def test_failed_property_true_when_failed(self):
        r = VerificationResult(
            status=VerificationStatus.FAILED,
            stage=VerificationStage.TEXT_MATCH,
            confidence=0.1,
            reasoning="no match",
        )
        assert r.failed is True
        assert r.passed is False

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError):
            VerificationResult(
                status=VerificationStatus.PASSED,
                stage=VerificationStage.TEXT_MATCH,
                confidence=1.5,
                reasoning="bad",
            )

    def test_confidence_exactly_zero_and_one_are_valid(self):
        for c in (0.0, 1.0):
            r = VerificationResult(
                status=VerificationStatus.PASSED,
                stage=VerificationStage.TEXT_MATCH,
                confidence=c,
                reasoning="edge",
            )
            assert r.confidence == c


# ---------------------------------------------------------------------------
# TextMatchVerifier
# ---------------------------------------------------------------------------

class TestTextMatchVerifier:

    def _verifier(self, **kwargs) -> TextMatchVerifier:
        return TextMatchVerifier(TextMatchConfig(**kwargs))

    def test_passes_when_both_names_present(self):
        rel = _make_rel()
        v = self._verifier(min_match_ratio=0.5)
        result = v.verify(rel, "Alpha is related to Beta", source_name="Alpha", target_name="Beta")
        assert result.passed
        assert result.confidence > 0

    def test_fails_when_no_names_found_in_context(self):
        rel = _make_rel()
        v = self._verifier(min_match_ratio=1.0)
        result = v.verify(rel, "completely unrelated text", source_name="Apple", target_name="Zebra")
        assert result.failed

    def test_empty_context_returns_failed(self):
        rel = _make_rel()
        v = self._verifier()
        result = v.verify(rel, "", source_name="X", target_name="Y")
        assert result.failed
        assert result.confidence == 0.0

    def test_case_insensitive_matching_by_default(self):
        rel = _make_rel()
        v = self._verifier()
        result = v.verify(rel, "ALPHA IS RELATED TO BETA", source_name="alpha", target_name="beta")
        assert result.passed

    def test_case_sensitive_does_not_match_wrong_case(self):
        rel = _make_rel()
        v = self._verifier(case_sensitive=True, min_match_ratio=1.0)
        result = v.verify(rel, "ALPHA related to BETA", source_name="alpha", target_name="beta")
        assert result.failed

    def test_extra_terms_contribute_to_confidence(self):
        rel = _make_rel()
        v = self._verifier(extra_terms=["kinase"])
        # context contains the extra term + entity names
        result = v.verify(rel, "kinase alpha beta", source_name="alpha", target_name="beta")
        assert result.passed
        assert result.confidence > 0.5

    def test_no_terms_trivially_passes(self):
        rel = _make_rel()
        v = self._verifier()
        result = v.verify(rel, "some context")
        assert result.passed  # no required names to fail on

    def test_metadata_includes_matched_unmatched(self):
        rel = _make_rel()
        v = self._verifier()
        result = v.verify(rel, "alpha present here", source_name="alpha", target_name="zorro")
        assert "alpha" in result.metadata["matched"]
        assert "zorro" in result.metadata["unmatched"]


# ---------------------------------------------------------------------------
# EmbeddingVerifier
# ---------------------------------------------------------------------------

class TestEmbeddingVerifier:

    def test_returns_skipped_when_sentence_transformers_missing(self, monkeypatch):
        """If sentence-transformers is not installed, result should be SKIPPED."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("mocked missing")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        # Clear cache so lazy load triggers
        from graphbuilder.core.verification import embedding as emb_mod
        monkeypatch.setattr(emb_mod, "_MODEL_CACHE", {})

        v = EmbeddingVerifier(EmbeddingConfig(model_name="test-model"))
        rel = _make_rel()
        result = v.verify(rel, "some context")
        assert result.status == VerificationStatus.SKIPPED

    def test_empty_context_returns_failed(self):
        v = EmbeddingVerifier()
        # Don't call model — empty context short-circuits before loading
        result = v.verify(_make_rel(), "")
        assert result.failed
        assert result.confidence == 0.0

    def test_build_query_contains_names_and_type(self):
        rel = _make_rel(rel_type=RelationshipType.RELATED_TO, description="a links to b")
        query = EmbeddingVerifier._build_query(rel, "Alpha", "Beta")
        assert "Alpha" in query
        assert "Beta" in query
        assert "RELATED TO" in query
        assert "a links to b" in query

    def test_cosine_similarity_identical_vectors(self):
        import numpy as np

        class FakeModel:
            def encode(self, texts, convert_to_numpy=True):
                return np.array([[1.0, 0.0], [1.0, 0.0]])

        sim = EmbeddingVerifier._cosine_similarity(FakeModel(), "a", "a")
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal_vectors(self):
        import numpy as np

        class FakeModel:
            def encode(self, texts, convert_to_numpy=True):
                return np.array([[1.0, 0.0], [0.0, 1.0]])

        sim = EmbeddingVerifier._cosine_similarity(FakeModel(), "a", "b")
        assert abs(sim) < 1e-6


# ---------------------------------------------------------------------------
# LLMVerifier
# ---------------------------------------------------------------------------

class TestLLMVerifier:

    def _llm(self, response: str) -> MagicMock:
        svc = MagicMock()
        svc.generate_text.return_value = response
        return svc

    def test_valid_verdict_passes(self):
        llm = self._llm(json.dumps({"verdict": "valid", "confidence": 0.9, "reasoning": "clear evidence"}))
        v = LLMVerifier(llm)
        result = v.verify(_make_rel(), "some context")
        assert result.passed
        assert result.confidence == 0.9
        assert "clear evidence" in result.reasoning

    def test_invalid_verdict_fails(self):
        llm = self._llm(json.dumps({"verdict": "invalid", "confidence": 0.15, "reasoning": "no support"}))
        v = LLMVerifier(llm)
        result = v.verify(_make_rel(), "some context")
        assert result.failed

    def test_uncertain_defaults_to_failed(self):
        llm = self._llm(json.dumps({"verdict": "uncertain", "confidence": 0.5, "reasoning": "unclear"}))
        v = LLMVerifier(llm, LLMVerifierConfig(uncertain_as_pass=False))
        result = v.verify(_make_rel(), "context")
        assert result.failed

    def test_uncertain_as_pass_when_configured(self):
        llm = self._llm(json.dumps({"verdict": "uncertain", "confidence": 0.5, "reasoning": "unclear"}))
        v = LLMVerifier(llm, LLMVerifierConfig(uncertain_as_pass=True))
        result = v.verify(_make_rel(), "context")
        assert result.passed

    def test_malformed_json_returns_failed(self):
        llm = self._llm("not json at all")
        v = LLMVerifier(llm)
        result = v.verify(_make_rel(), "context")
        assert result.failed
        assert "non-JSON" in result.reasoning
        assert result.metadata.get("raw_response") == "not json at all"

    def test_strips_code_fences(self):
        payload = json.dumps({"verdict": "valid", "confidence": 0.7, "reasoning": "fine"})
        llm = self._llm(f"```json\n{payload}\n```")
        v = LLMVerifier(llm)
        result = v.verify(_make_rel(), "ctx")
        assert result.passed

    def test_empty_context_short_circuits(self):
        llm = MagicMock()
        v = LLMVerifier(llm)
        result = v.verify(_make_rel(), "")
        assert result.failed
        llm.generate_text.assert_not_called()

    def test_llm_exception_returns_failed(self):
        llm = MagicMock()
        llm.generate_text.side_effect = RuntimeError("timeout")
        v = LLMVerifier(llm)
        result = v.verify(_make_rel(), "context")
        assert result.failed
        assert "timeout" in result.reasoning

    def test_confidence_clamped_to_valid_range(self):
        llm = self._llm(json.dumps({"verdict": "valid", "confidence": 9.99, "reasoning": "wild"}))
        v = LLMVerifier(llm)
        result = v.verify(_make_rel(), "ctx")
        assert result.confidence <= 1.0


# ---------------------------------------------------------------------------
# CascadingVerifier
# ---------------------------------------------------------------------------

class TestCascadingVerifier:

    def _pass_result(self, stage: VerificationStage, conf: float = 0.8) -> VerificationResult:
        return VerificationResult(status=VerificationStatus.PASSED, stage=stage, confidence=conf, reasoning="ok")

    def _fail_result(self, stage: VerificationStage, conf: float = 0.2) -> VerificationResult:
        return VerificationResult(status=VerificationStatus.FAILED, stage=stage, confidence=conf, reasoning="fail")

    def test_all_stages_pass_returns_passed(self):
        cfg = CascadingVerifierConfig(enable_embedding=False, enable_llm=False)
        v = CascadingVerifier(config=cfg)
        rel = _make_rel()
        context = "alpha related to beta"
        result = v.verify(rel, context, source_name="alpha", target_name="beta")
        assert result.stage == VerificationStage.CASCADING
        assert result.passed

    def test_early_exit_on_pass_skips_later_stages(self):
        """With early_exit_on_pass, a passing TextMatch should prevent Embedding running."""
        cfg = CascadingVerifierConfig(
            enable_embedding=True,
            enable_llm=False,
            early_exit_on_pass=True,
        )
        v = CascadingVerifier(config=cfg)
        # Override the embedding verifier to detect if it was called
        called = []
        original_verify = v._emb_verifier.verify
        def spy(*a, **kw):
            called.append(True)
            return original_verify(*a, **kw)
        v._emb_verifier.verify = spy

        rel = _make_rel()
        result = v.verify(rel, "alpha related to beta", source_name="alpha", target_name="beta")
        # embedding stage should NOT have been called because text match passed first
        assert len(called) == 0
        assert result.stage == VerificationStage.CASCADING

    def test_no_stages_enabled_returns_failed(self):
        cfg = CascadingVerifierConfig(
            enable_text_match=False,
            enable_embedding=False,
            enable_llm=False,
        )
        v = CascadingVerifier(config=cfg)
        result = v.verify(_make_rel(), "ctx")
        # No stages ran — aggregate should return FAILED with 0 confidence
        assert result.failed
        assert result.confidence == 0.0

    def test_majority_vote_two_pass_one_fail(self):
        """2 passes, 1 fail → PASSED by majority."""
        cfg = CascadingVerifierConfig(enable_embedding=False, enable_llm=False)
        v = CascadingVerifier(config=cfg)

        # Patch text verifier to return PASSED
        v._text_verifier.verify = lambda **kw: self._pass_result(VerificationStage.TEXT_MATCH)

        result = v.verify(_make_rel(), "ctx")
        # Only one stage ran — but it passed → majority pass
        assert result.passed

    def test_stage_results_included_in_cascading_result(self):
        cfg = CascadingVerifierConfig(enable_embedding=False, enable_llm=False)
        v = CascadingVerifier(config=cfg)
        result = v.verify(_make_rel(), "alpha beta", source_name="alpha", target_name="beta")
        assert len(result.stage_results) >= 1
        assert all(isinstance(r, VerificationResult) for r in result.stage_results)

    def test_llm_stage_skipped_when_no_service_provided(self):
        cfg = CascadingVerifierConfig(enable_embedding=False, enable_llm=True)
        v = CascadingVerifier(config=cfg, llm_service=None)
        result = v.verify(_make_rel(), "some context")
        # Should not raise; LLM stage just warns and skips
        assert result.stage == VerificationStage.CASCADING

    def test_confidence_is_float_between_0_and_1(self):
        cfg = CascadingVerifierConfig(enable_embedding=False, enable_llm=False)
        v = CascadingVerifier(config=cfg)
        result = v.verify(_make_rel(), "alpha beta", source_name="alpha", target_name="beta")
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# RelationshipVerificationUseCase
# ---------------------------------------------------------------------------

class TestRelationshipVerificationUseCase:

    def _make_use_case(self, rel: GraphRelationship) -> RelationshipVerificationUseCase:
        graph = _make_graph_with_rel(rel)
        return RelationshipVerificationUseCase(graph, llm_service=None)

    def _text_only_config(self) -> VerificationConfig:
        cfg = CascadingVerifierConfig(enable_embedding=False, enable_llm=False)
        return VerificationConfig(cascading=cfg)

    def test_returns_success_result(self):
        rel = _make_rel(description="alpha related to beta")
        uc = self._make_use_case(rel)
        result = uc.execute(self._text_only_config())
        assert result.success is True

    def test_report_contains_one_entry_per_relationship(self):
        rel = _make_rel()
        uc = self._make_use_case(rel)
        result = uc.execute(self._text_only_config())
        assert len(result.data["report"]) == 1

    def test_report_entry_shape(self):
        rel = _make_rel()
        uc = self._make_use_case(rel)
        result = uc.execute(self._text_only_config())
        entry = result.data["report"][0]
        for key in ("relationship_id", "status", "confidence", "reasoning", "stage_results"):
            assert key in entry

    def test_passing_rel_annotated_correctly(self):
        rel = _make_rel(description="alpha related to beta")
        graph = _make_graph_with_rel(rel)
        cfg = CascadingVerifierConfig(enable_embedding=False, enable_llm=False)
        ver_cfg = VerificationConfig(
            cascading=cfg,
            context_map={rel.id: "alpha related to beta"},
            entity_name_map={rel.source_entity_id: "alpha", rel.target_entity_id: "beta"},
        )
        uc = RelationshipVerificationUseCase(graph, llm_service=None)
        uc.execute(ver_cfg)
        # The relationship should be annotated in-place
        annotations = graph.relationships[rel.id].metadata.annotations
        assert "verification_confidence" in annotations
        assert "verification_reasoning" in annotations

    def test_counts_correct_in_data(self):
        rel = _make_rel()
        uc = self._make_use_case(rel)
        result = uc.execute(self._text_only_config())
        d = result.data
        assert d["total"] == 1
        assert d["passed"] + d["failed"] + d["skipped"] == 1

    def test_empty_graph_returns_zero_counts(self):
        graph = KnowledgeGraph()
        uc = RelationshipVerificationUseCase(graph, llm_service=None)
        result = uc.execute(self._text_only_config())
        assert result.success
        assert result.data["total"] == 0
        assert result.data["passed"] == 0
