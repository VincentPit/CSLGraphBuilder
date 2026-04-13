"""
CascadingVerifier — orchestrates the three-stage verification pipeline.

Stages run in order of increasing cost.  Each stage only runs when the
previous stage's result is **inconclusive** — i.e. its confidence falls
inside the uncertainty band [``escalation_lower``, ``escalation_upper``].

    1. TextMatchVerifier  — fast, no dependencies
    2. EmbeddingVerifier  — moderate; requires sentence-transformers
    3. LLMVerifier        — most expensive; runs only when earlier stages
                            could not reach a decisive verdict

A stage whose confidence ≥ ``escalation_upper`` is treated as a confident
PASS; one whose confidence < ``escalation_lower`` is a confident FAIL.
Either outcome terminates the cascade early.

When the cascade exhausts all enabled stages (all inconclusive), the final
result is determined by the weighted confidence of all stages vs. the
``pass_threshold``.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from .models import VerificationResult, VerificationStage, VerificationStatus
from .text_match import TextMatchVerifier, TextMatchConfig
from .embedding import EmbeddingVerifier, EmbeddingConfig
from .llm_verifier import LLMVerifier, LLMVerifierConfig
from ...domain.models.graph_models import GraphRelationship

logger = logging.getLogger(__name__)


@dataclass
class CascadingVerifierConfig:
    """
    Configuration for ``CascadingVerifier``.

    All per-stage configs are optional; sensible defaults are used when absent.
    Set ``enable_embedding`` or ``enable_llm`` to ``False`` to disable those
    stages (e.g. in tests or when dependencies are unavailable).
    """

    text_match: TextMatchConfig = field(default_factory=TextMatchConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMVerifierConfig = field(default_factory=LLMVerifierConfig)

    enable_text_match: bool = True
    enable_embedding:  bool = True
    enable_llm:        bool = True

    # Escalation band — results inside this range are inconclusive and
    # trigger the next stage.  Results outside are decisive.
    escalation_lower: float = 0.3
    """Confidence below this → decisive FAIL, stop cascade."""

    escalation_upper: float = 0.7
    """Confidence at or above this → decisive PASS, stop cascade."""

    # Weights for final confidence roll-up (used when all stages run)
    stage_weights: dict = field(default_factory=lambda: {
        VerificationStage.TEXT_MATCH: 0.20,
        VerificationStage.EMBEDDING:  0.35,
        VerificationStage.LLM:        0.45,
    })

    pass_threshold: float = 0.5
    """Weighted confidence must meet or exceed this value to pass."""


class CascadingVerifier:
    """
    Run up to three verifiers in sequence and aggregate their results.

    Parameters
    ----------
    config:
        ``CascadingVerifierConfig``.
    llm_service:
        Required only when ``config.enable_llm`` is ``True``.  Any object with
        a ``generate_text(prompt, system_prompt, temperature) -> str`` method.
    graph_repo:
        Optional graph repository for Neo4j vector search in the embedding
        stage.  When provided, the EmbeddingVerifier will query the vector
        index to find semantically similar entities/relationships.
    """

    def __init__(
        self,
        config: Optional[CascadingVerifierConfig] = None,
        llm_service: Optional[Any] = None,
        graph_repo: Optional[Any] = None,
    ) -> None:
        self._cfg = config or CascadingVerifierConfig()
        self._text_verifier = TextMatchVerifier(self._cfg.text_match)
        self._emb_verifier  = EmbeddingVerifier(self._cfg.embedding, graph_repo=graph_repo)
        self._llm_verifier  = (
            LLMVerifier(llm_service, self._cfg.llm)
            if llm_service is not None
            else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        relationship: GraphRelationship,
        context: str,
        source_name: Optional[str] = None,
        target_name: Optional[str] = None,
    ) -> VerificationResult:
        """
        Run the cascading pipeline for a single relationship.

        Each stage runs only if all previous stages were inconclusive
        (confidence within the escalation band).  A decisive result
        at any stage terminates the cascade early.

        Returns a ``VerificationResult`` with ``stage=CASCADING`` whose
        ``stage_results`` list contains the individual stage outcomes.
        """
        stage_results: List[VerificationResult] = []
        kwargs = dict(
            relationship=relationship,
            context=context,
            source_name=source_name,
            target_name=target_name,
        )

        # --- Stage 1: Text match ---
        if self._cfg.enable_text_match:
            result = self._text_verifier.verify(**kwargs)
            stage_results.append(result)
            if self._is_decisive(result):
                return self._aggregate(stage_results, decisive=result)

        # --- Stage 2: Embedding (only if previous stage was inconclusive) ---
        if self._cfg.enable_embedding:
            result = self._emb_verifier.verify(**kwargs)
            stage_results.append(result)
            if result.status != VerificationStatus.SKIPPED and self._is_decisive(result):
                return self._aggregate(stage_results, decisive=result)

        # --- Stage 3: LLM (only if still inconclusive) ---
        if self._cfg.enable_llm:
            if self._llm_verifier is None:
                logger.warning(
                    "LLM stage enabled but no llm_service was provided; skipping."
                )
            else:
                result = self._llm_verifier.verify(**kwargs)
                stage_results.append(result)
                # LLM is the last stage — its result is always decisive
                return self._aggregate(stage_results, decisive=result)

        # All stages ran and none were decisive → fall back to weighted confidence
        return self._aggregate(stage_results)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_decisive(self, result: VerificationResult) -> bool:
        """Return True if the stage produced a confident enough verdict."""
        return (
            result.confidence >= self._cfg.escalation_upper
            or result.confidence < self._cfg.escalation_lower
        )

    def _aggregate(
        self,
        stage_results: List[VerificationResult],
        decisive: Optional[VerificationResult] = None,
    ) -> VerificationResult:
        if not stage_results:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                stage=VerificationStage.CASCADING,
                confidence=0.0,
                reasoning="No stages were run.",
                stage_results=stage_results,
            )

        # Filter out SKIPPED results for confidence roll-up
        active = [r for r in stage_results if r.status != VerificationStatus.SKIPPED]

        if not active:
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                stage=VerificationStage.CASCADING,
                confidence=0.0,
                reasoning="All stages were skipped.",
                stage_results=stage_results,
            )

        # If a decisive stage ended the cascade, use its verdict directly
        # but report the weighted confidence across all stages that ran.
        weights = self._cfg.stage_weights
        total_weight = 0.0
        weighted_conf = 0.0
        for r in active:
            w = weights.get(r.stage, 1.0 / len(active))
            weighted_conf += w * r.confidence
            total_weight += w

        final_confidence = round(weighted_conf / total_weight if total_weight else 0.0, 4)

        if decisive and decisive.status != VerificationStatus.SKIPPED:
            overall_passed = decisive.passed
            how = f"decided at {decisive.stage.value} stage (confidence {decisive.confidence:.2f})"
        else:
            # No decisive stage — all were inconclusive; fall back to threshold
            overall_passed = final_confidence >= self._cfg.pass_threshold
            how = f"no decisive stage; weighted confidence vs threshold {self._cfg.pass_threshold}"

        stage_summaries = [
            f"{r.stage.value}: {'PASS' if r.passed else ('SKIP' if r.status == VerificationStatus.SKIPPED else 'FAIL')} ({r.confidence:.2f})"
            for r in stage_results
        ]
        reasoning = (
            f"{'PASSED' if overall_passed else 'FAILED'} — {how}. "
            f"Weighted confidence {final_confidence:.4f}. "
            + "; ".join(stage_summaries)
        )

        return VerificationResult(
            status=VerificationStatus.PASSED if overall_passed else VerificationStatus.FAILED,
            stage=VerificationStage.CASCADING,
            confidence=final_confidence,
            reasoning=reasoning,
            stage_results=stage_results,
        )
