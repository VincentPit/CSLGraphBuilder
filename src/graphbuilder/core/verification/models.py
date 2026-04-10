"""
Data models for the relationship verification pipeline.

Each verifier returns a ``VerificationResult`` that records which stage ran,
whether it passed, and a confidence score in [0.0, 1.0].  The *reasoning*
field holds a human-readable explanation so the caller can log or surface it.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class VerificationStatus(Enum):
    """Overall pass/fail outcome of a verification attempt."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"   # stage was short-circuited by an earlier pass


class VerificationStage(Enum):
    """The stage that produced a particular result."""
    TEXT_MATCH = "text_match"
    EMBEDDING  = "embedding"
    LLM        = "llm"
    CASCADING  = "cascading"   # aggregate result from CascadingVerifier


@dataclass
class VerificationResult:
    """
    Outcome of a single verification attempt.

    Attributes
    ----------
    status:
        Whether the relationship passed, failed, or was skipped.
    stage:
        Which verifier produced this result.
    confidence:
        Score in [0.0, 1.0].  Higher = more confident the relationship is valid.
    reasoning:
        Human-readable explanation from the verifier.
    stage_results:
        For ``CASCADING`` stage only — ordered list of per-stage results.
    metadata:
        Arbitrary extra data (e.g. matched terms, cosine similarity value).
    timestamp:
        When the result was produced (UTC).
    """

    status:        VerificationStatus
    stage:         VerificationStage
    confidence:    float
    reasoning:     str
    stage_results: list = field(default_factory=list)  # List[VerificationResult]
    metadata:      Dict[str, Any] = field(default_factory=dict)
    timestamp:     datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    @property
    def passed(self) -> bool:
        return self.status == VerificationStatus.PASSED

    @property
    def failed(self) -> bool:
        return self.status == VerificationStatus.FAILED
