"""
LLMVerifier — Stage 3 of the cascading verification pipeline.

Prompts the configured LLM with a structured verification request and parses
its JSON response to produce a ``VerificationResult`` with a reasoning trace.

The LLM is expected to reply with a JSON object having the following shape:

    {
        "verdict": "valid" | "invalid" | "uncertain",
        "confidence": <float 0–1>,
        "reasoning": "<explanation>"
    }

If the LLM returns malformed JSON or omits required fields the verifier falls
back to a FAILED result with the raw response preserved in ``metadata``.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .models import VerificationResult, VerificationStage, VerificationStatus
from ...domain.models.graph_models import GraphRelationship

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a knowledge-graph relationship verifier.
Given a candidate relationship and a supporting context passage, decide
whether the relationship is factually supported by the context.

Respond ONLY with a single JSON object — no preamble, no markdown:
{
  "verdict": "valid" | "invalid" | "uncertain",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<one or two sentences explaining your decision>"
}
"""

_USER_TEMPLATE = """\
## Relationship
Source entity : {source}
Target entity : {target}
Relationship  : {rel_type}
Description   : {description}

## Context
{context}

Verify whether the context supports the relationship described above.
"""


@dataclass
class LLMVerifierConfig:
    """Configuration for ``LLMVerifier``."""

    uncertain_as_pass: bool = False
    """Treat 'uncertain' verdict as PASSED (default: FAILED)."""

    max_context_chars: int = 3_000
    """Truncate context to this many characters before sending to the LLM."""

    temperature: float = 0.0
    """LLM temperature — keep at 0 for deterministic outputs."""


class LLMVerifier:
    """
    LLM-backed verifier with structured JSON output and reasoning trace.

    Parameters
    ----------
    llm_service:
        Any object that exposes ``generate_text(prompt: str, system_prompt: str,
        temperature: float) -> str``.  Both ``AdvancedLLMService`` and a simple
        mock work.
    config:
        ``LLMVerifierConfig``.  Defaults to ``LLMVerifierConfig()`` if not given.
    """

    def __init__(self, llm_service: Any, config: Optional[LLMVerifierConfig] = None) -> None:
        self._llm = llm_service
        self._cfg = config or LLMVerifierConfig()

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
        Send the relationship + context to an LLM and parse the verdict.
        """
        if not context:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                stage=VerificationStage.LLM,
                confidence=0.0,
                reasoning="Empty context; LLM verification skipped.",
            )

        user_prompt = _USER_TEMPLATE.format(
            source=source_name or relationship.source_entity_id,
            target=target_name or relationship.target_entity_id,
            rel_type=relationship.relationship_type.value,
            description=relationship.description or "(none)",
            context=context[: self._cfg.max_context_chars],
        )

        try:
            raw = self._llm.generate_text(
                prompt=user_prompt,
                system_prompt=_SYSTEM_PROMPT,
                temperature=self._cfg.temperature,
            )
        except Exception as exc:
            logger.error("LLM call failed: %s", exc, exc_info=True)
            return VerificationResult(
                status=VerificationStatus.FAILED,
                stage=VerificationStage.LLM,
                confidence=0.0,
                reasoning=f"LLM call raised an exception: {exc}",
                metadata={"error": str(exc)},
            )

        return self._parse_response(raw)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> VerificationResult:
        try:
            # Strip code fences if the model ignored the system prompt
            clean = raw.strip()
            if clean.startswith("```"):
                clean = "\n".join(clean.split("\n")[1:])
            if clean.endswith("```"):
                clean = "\n".join(clean.split("\n")[:-1])
            parsed: Dict[str, Any] = json.loads(clean)
        except json.JSONDecodeError:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                stage=VerificationStage.LLM,
                confidence=0.0,
                reasoning="LLM returned non-JSON output; treating as failed.",
                metadata={"raw_response": raw},
            )

        verdict = str(parsed.get("verdict", "invalid")).lower()
        confidence = float(parsed.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
        reasoning = str(parsed.get("reasoning", "No reasoning provided."))

        if verdict == "valid":
            status = VerificationStatus.PASSED
        elif verdict == "uncertain":
            status = (
                VerificationStatus.PASSED
                if self._cfg.uncertain_as_pass
                else VerificationStatus.FAILED
            )
        else:
            status = VerificationStatus.FAILED

        return VerificationResult(
            status=status,
            stage=VerificationStage.LLM,
            confidence=confidence,
            reasoning=reasoning,
            metadata={"verdict": verdict, "raw_response": raw},
        )
