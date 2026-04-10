"""
CascadingVerifier — orchestrates the three-stage verification pipeline.

Stages run in order:

    1. TextMatchVerifier  — fast, no dependencies
    2. EmbeddingVerifier  — requires sentence-transformers; skipped gracefully
                            if not installed
    3. LLMVerifier        — most expensive; only runs if embedding stage didn't
                            already produce a decisive result

Short-circuit rules (configurable):
  * ``early_exit_on_pass``  — if any stage passes, skip remaining stages.
  * ``early_exit_on_fail``  — if any stage fails, skip remaining stages.
  * Default: run all stages; take majority verdict.

Final confidence is the weighted mean of per-stage confidence scores.
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

    # Short-circuit behaviour
    early_exit_on_pass: bool = False
    """Stop after the first stage that passes."""

    early_exit_on_fail: bool = False
    """Stop after the first stage that fails."""

    # Weights for final confidence roll-up
    stage_weights: dict = field(default_factory=lambda: {
        VerificationStage.TEXT_MATCH: 0.20,
        VerificationStage.EMBEDDING:  0.35,
        VerificationStage.LLM:        0.45,
    })


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
    """

    def __init__(
        self,
        config: Optional[CascadingVerifierConfig] = None,
        llm_service: Optional[Any] = None,
    ) -> None:
        self._cfg = config or CascadingVerifierConfig()
        self._text_verifier = TextMatchVerifier(self._cfg.text_match)
        self._emb_verifier  = EmbeddingVerifier(self._cfg.embedding)
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
            if self._should_exit(result):
                return self._aggregate(stage_results)

        # --- Stage 2: Embedding ---
        if self._cfg.enable_embedding:
            result = self._emb_verifier.verify(**kwargs)
            stage_results.append(result)
            if result.status != VerificationStatus.SKIPPED and self._should_exit(result):
                return self._aggregate(stage_results)

        # --- Stage 3: LLM ---
        if self._cfg.enable_llm:
            if self._llm_verifier is None:
                logger.warning(
                    "LLM stage enabled but no llm_service was provided; skipping."
                )
            else:
                result = self._llm_verifier.verify(**kwargs)
                stage_results.append(result)

        return self._aggregate(stage_results)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _should_exit(self, result: VerificationResult) -> bool:
        if self._cfg.early_exit_on_pass and result.passed:
            return True
        if self._cfg.early_exit_on_fail and result.failed:
            return True
        return False

    def _aggregate(self, stage_results: List[VerificationResult]) -> VerificationResult:
        if not stage_results:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                stage=VerificationStage.CASCADING,
                confidence=0.0,
                reasoning="No stages were run.",
                stage_results=stage_results,
            )

        # Filter out SKIPPED results for voting and confidence roll-up
        active = [r for r in stage_results if r.status != VerificationStatus.SKIPPED]

        if not active:
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                stage=VerificationStage.CASCADING,
                confidence=0.0,
                reasoning="All stages were skipped.",
                stage_results=stage_results,
            )

        # Majority vote
        passed_count = sum(1 for r in active if r.passed)
        overall_passed = passed_count > len(active) / 2

        # Weighted confidence
        weights = self._cfg.stage_weights
        total_weight = 0.0
        weighted_conf = 0.0
        for r in active:
            w = weights.get(r.stage, 1.0 / len(active))
            weighted_conf += w * r.confidence
            total_weight += w

        final_confidence = round(weighted_conf / total_weight if total_weight else 0.0, 4)

        stage_summaries = [
            f"{r.stage.value}: {'PASS' if r.passed else ('SKIP' if r.status == VerificationStatus.SKIPPED else 'FAIL')} ({r.confidence:.2f})"
            for r in stage_results
        ]
        reasoning = (
            f"{'PASSED' if overall_passed else 'FAILED'} by majority vote "
            f"({passed_count}/{len(active)} stages passed). "
            + "; ".join(stage_summaries)
        )

        return VerificationResult(
            status=VerificationStatus.PASSED if overall_passed else VerificationStatus.FAILED,
            stage=VerificationStage.CASCADING,
            confidence=final_confidence,
            reasoning=reasoning,
            stage_results=stage_results,
        )
