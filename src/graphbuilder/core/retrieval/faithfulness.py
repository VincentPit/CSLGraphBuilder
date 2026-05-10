"""Answer-faithfulness check (P8 of docs/RAG_QA_PLAN.md).

After the LLM generates an answer the QA service runs every claim
through this module to score how well the cited sources actually
support what was said. The output is the per-claim confidence the
frontend renders (yellow underline at < 0.5) and the
``answer_faithfulness`` headline metric on the eval harness.

What "claim" means here
-----------------------
We split the answer at each ``[n]`` citation marker. Every prose span
terminated by one or more ``[n]`` markers is one claim; the indices in
those markers tell us which retrieved sources are supposed to back it.
A trailing span that has no ``[n]`` is *uncited* — kept for visibility
in the result but excluded from the headline score so the model is
penalised for *wrong* citations, not for prose that needs none (the
opening "Here's what I found:" sentence, mostly).

Why a focused checker — not the existing CascadingVerifier
----------------------------------------------------------
The plan's §6.2 sketch suggested "reuse ``CascadingVerifier`` with a
synthesised ``(claim_subject) --MENTIONS--> (claim_object)``". On
paper that lines up; in code the cascade's stages all take a real
``GraphRelationship`` with source/target ids and look at entity-name
overlap. A claim sentence rarely has a clean (subject, object); the
useful signal is "do the cited chunks contain the salient terms of
the claim", which the cascade can't express without contortions.

So we keep the same *shape* (lexical → optional LLM, with explicit
escalation thresholds) but with a checker dedicated to the claim ↔
chunk match. The two-stage default keeps it cheap: lexical alone is
deterministic, latency-free, and good enough for the headline number;
the LLM stage is opt-in for when we want a tighter verdict on the
border cases. Refusal answers ("I cannot find this in the knowledge
base.") short-circuit to a perfect score — declining to answer is the
faithful response when the graph is empty.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .models import RetrievedItem


logger = logging.getLogger("graphbuilder.qa.faithfulness")


# ----------------------------------------------------------------------
# Public dataclasses
# ----------------------------------------------------------------------


@dataclass
class FaithfulnessConfig:
    """Tuning knobs for :class:`FaithfulnessChecker`.

    Defaults are picked so that the lexical-only path produces a
    well-calibrated score on the hermetic gold set without any LLM
    calls. Bump ``enable_llm_escalation`` when you want the cascade's
    stage-3 verdict on borderline claims (cost ≈ 1 LLM call per claim
    that lands in the inconclusive band).
    """

    # Tokens shorter than this are dropped before computing overlap.
    # 3 keeps gene names like "KIT", "MYC" but drops the's, of, and …
    min_token_len: int = 3

    # Lexical match ratio at-or-above this is a confident PASS — no LLM.
    upper_threshold: float = 0.7

    # Below this is a confident FAIL — escalate to LLM if enabled.
    lower_threshold: float = 0.3

    # When the LLM stage is on we send borderline (and failing) claims
    # to the LLM for a verdict. Off by default so the eval harness
    # doesn't pay ≈100 extra LLM calls per gold-set run.
    enable_llm_escalation: bool = False

    # Hard cap on context length sent to the lexical comparator (and
    # to the LLM if escalation runs). Keeps memory bounded on long
    # hydrated chunks.
    max_context_chars: int = 4_000

    # Threshold a per-claim confidence must clear to be considered
    # "passing"; drives the ``failed_claims`` count + the
    # ``qa_faithfulness_failures`` metric. Mirrors the §6.2 yellow-
    # underline cutoff.
    pass_threshold: float = 0.5


@dataclass
class ClaimSpan:
    """One prose span carved out of the answer.

    ``cited_indices`` is empty for the trailing tail (text after the
    last ``[n]``). The checker keeps tail spans in the result for
    debug-pane visibility but excludes them from the overall score.
    """

    text: str
    cited_indices: List[int] = field(default_factory=list)


@dataclass
class ClaimVerification:
    """Per-claim verdict produced by :class:`FaithfulnessChecker`."""

    claim_text: str
    cited_indices: List[int]
    confidence: float                     # 0.0–1.0
    method: str                           # "text_match" | "llm" | "uncited" | "refusal"
    reasoning: str
    escalated_to_llm: bool = False
    matched_terms: List[str] = field(default_factory=list)
    missing_terms: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.confidence >= 0.5

    def to_dict(self) -> dict:
        return {
            "claim_text": self.claim_text,
            "cited_indices": list(self.cited_indices),
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "reasoning": self.reasoning,
            "escalated_to_llm": self.escalated_to_llm,
            "matched_terms": list(self.matched_terms),
            "missing_terms": list(self.missing_terms),
        }


@dataclass
class FaithfulnessResult:
    """Bundle returned by :func:`FaithfulnessChecker.check`."""

    overall_score: Optional[float]        # None when the answer has no scorable claims
    claims: List[ClaimVerification] = field(default_factory=list)
    failed_claims: int = 0
    is_refusal: bool = False

    def to_dict(self) -> dict:
        return {
            "overall_score": (
                round(self.overall_score, 4) if self.overall_score is not None else None
            ),
            "failed_claims": self.failed_claims,
            "is_refusal": self.is_refusal,
            "claims": [c.to_dict() for c in self.claims],
        }


# ----------------------------------------------------------------------
# Claim extraction
# ----------------------------------------------------------------------


_CITATION_RE = re.compile(r"\[(\d+)\]")
_REFUSAL_NEEDLE = "cannot find this in the knowledge base"


def extract_claim_spans(answer: str) -> List[ClaimSpan]:
    """Carve an answer into ``ClaimSpan`` objects at each ``[n]`` boundary.

    Each span owns the text from the previous boundary up to and
    including its citation cluster. A run of adjacent markers
    (``"… BCR-ABL.[1][2]"``) collapses into one span carrying both
    indices — the LLM cited two sources for a single claim, so we
    score against the union.

    The trailing prose after the last marker is appended as a span
    with empty ``cited_indices``. Callers exclude it from the overall
    score; we keep it visible so the debug pane can show the model
    chattering past its citations.
    """
    if not answer:
        return []

    spans: List[ClaimSpan] = []
    cursor = 0
    text_len = len(answer)

    matches = list(_CITATION_RE.finditer(answer))
    if not matches:
        # No citations at all — one tail-only span. Overall score will
        # be ``None`` for this answer; eval treats it as "no signal".
        return [ClaimSpan(text=answer.strip(), cited_indices=[])]

    i = 0
    while i < len(matches):
        # Group runs of adjacent markers separated only by whitespace
        # or punctuation. They share the same prose span.
        run_start = matches[i].start()
        cluster_end = matches[i].end()
        cluster_indices: List[int] = []
        j = i
        while j < len(matches):
            m = matches[j]
            if j > i:
                between = answer[matches[j - 1].end() : m.start()]
                if between.strip():
                    break
            try:
                cluster_indices.append(int(m.group(1)))
            except ValueError:
                pass
            cluster_end = m.end()
            j += 1

        prose = answer[cursor:run_start].strip()
        if prose or cluster_indices:
            # Drop duplicate indices but preserve first-appearance order.
            seen: set[int] = set()
            uniq: List[int] = []
            for idx in cluster_indices:
                if idx not in seen:
                    seen.add(idx)
                    uniq.append(idx)
            spans.append(ClaimSpan(text=prose, cited_indices=uniq))
        # Advance past the trailing punctuation that belongs to *this*
        # claim ("Claim A [1]." → the period closes the cited claim,
        # not the next one). Otherwise the next span inherits a leading
        # ". " which then either pollutes the prose or, if the prose is
        # punctuation-only, becomes a ghost uncited span.
        post = cluster_end
        while post < text_len and answer[post] in ".!?,;:":
            post += 1
        cursor = post
        i = j

    if cursor < text_len:
        tail = answer[cursor:].strip()
        # Drop punctuation-only tails (e.g. the "." after "[1]") — they
        # carry no claim content and would otherwise show up as ghost
        # uncited rows in the debug pane.
        if tail and any(ch.isalnum() for ch in tail):
            spans.append(ClaimSpan(text=tail, cited_indices=[]))

    return spans


# ----------------------------------------------------------------------
# Lexical comparator
# ----------------------------------------------------------------------


_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "has",
    "have", "been", "but", "not", "you", "your", "our", "any", "all", "into",
    "such", "via", "per", "also", "may", "can", "will", "would", "could",
    "than", "then", "thus", "there", "their", "they", "them", "what", "when",
    "which", "who", "whom", "whose", "where", "why", "how", "ref", "see",
    "one", "two", "three", "many", "some", "more", "most", "less", "few",
    "about", "between", "under", "over", "above", "below", "after", "before",
    "during", "into", "onto", "out", "off", "while", "though", "although",
    "because", "since", "due", "however", "therefore", "moreover", "while",
    "associated", "related", "relates", "relating", "involve", "involves",
    "involved", "based", "shown", "show", "shows", "include", "includes",
    "including", "found", "find", "finding", "support", "supports",
    "supported", "answer", "question", "system", "knowledge", "graph",
})


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]*")


def _tokenize(text: str, *, min_len: int) -> List[str]:
    """Split a string into salient lowercase tokens.

    The regex keeps gene-style hyphens (``BCR-ABL``) and digit suffixes
    (``brca1``) intact while dropping punctuation noise. Stopwords and
    sub-``min_len`` tokens are filtered so a short claim like
    "BCR-ABL is a kinase." compares against five real tokens, not
    twelve mostly-empty ones.
    """
    out: List[str] = []
    for raw in _TOKEN_RE.findall(text):
        tok = raw.lower()
        if len(tok) < min_len:
            continue
        if tok in _STOPWORDS:
            continue
        out.append(tok)
    return out


def _lexical_overlap(
    claim_tokens: Sequence[str],
    context: str,
    *,
    min_token_len: int,
) -> tuple[float, List[str], List[str]]:
    """Return (ratio, matched, missing).

    Ratio is ``len(matched) / len(claim_tokens)`` — recall of the
    claim's salient terms in the cited context. We deliberately don't
    use Jaccard: long chunks shouldn't be penalised for containing
    extra unrelated terms.
    """
    if not claim_tokens:
        return 0.0, [], []
    context_tokens = set(_tokenize(context, min_len=min_token_len))
    matched: List[str] = []
    missing: List[str] = []
    seen: set[str] = set()
    for tok in claim_tokens:
        if tok in seen:
            continue
        seen.add(tok)
        if tok in context_tokens:
            matched.append(tok)
        else:
            missing.append(tok)
    denom = len(matched) + len(missing)
    return (len(matched) / denom) if denom else 0.0, matched, missing


# ----------------------------------------------------------------------
# Optional LLM escalation
# ----------------------------------------------------------------------


_LLM_SYSTEM = """\
You are a faithfulness checker for a biomedical retrieval-augmented system.
Given a CLAIM extracted from an LLM's answer and the CONTEXT that was
cited to support it, return a JSON object:
  {"verdict": "supported"|"unsupported"|"partial", "confidence": 0.0-1.0,
   "reasoning": "<one sentence>"}
"supported" means the context contains enough information to back the claim
verbatim; "partial" means some of the claim is supported and some isn't;
"unsupported" means the context does not support the claim. Confidence
should reflect how certain you are of the verdict.
"""

_LLM_USER_TEMPLATE = """\
CLAIM
{claim}

CONTEXT
{context}
"""


async def _ask_llm(
    *,
    llm_service,
    claim: str,
    context: str,
    max_context_chars: int,
) -> tuple[Optional[float], Optional[str]]:
    """Run the LLM stage; return (confidence, reasoning) or (None, error).

    The LLM is expected to return strict JSON; if it doesn't we fall
    back to ``None`` so the caller can keep the lexical confidence.
    Failures here never propagate — this is a soft signal, not a gate.
    """
    import json

    user_prompt = _LLM_USER_TEMPLATE.format(
        claim=claim.strip(),
        context=(context or "")[:max_context_chars],
    )
    try:
        raw = await llm_service.generate_text(
            prompt=user_prompt,
            system_prompt=_LLM_SYSTEM,
            temperature=0.0,
            max_tokens=200,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("faithfulness LLM call failed: %s", exc)
        return None, f"llm error: {exc}"

    body = (raw or "").strip()
    if body.startswith("```"):
        body = "\n".join(body.split("\n")[1:])
        if body.endswith("```"):
            body = "\n".join(body.split("\n")[:-1])
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        logger.debug("faithfulness LLM returned non-JSON: %r", raw[:120])
        return None, "llm returned non-JSON"

    try:
        conf = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None, "llm returned non-numeric confidence"
    conf = max(0.0, min(1.0, conf))
    verdict = str(parsed.get("verdict", "")).lower()
    reasoning = str(parsed.get("reasoning") or "").strip() or f"verdict={verdict}"
    # Treat the LLM's verdict as authoritative — if it says
    # "unsupported" with high confidence we should report a low score,
    # not a high one. Map verdict to a final score.
    if verdict == "unsupported":
        final = max(0.0, 1.0 - conf)
    elif verdict == "partial":
        final = round(conf * 0.5, 4)
    else:
        final = conf
    return final, reasoning


# ----------------------------------------------------------------------
# Faithfulness checker
# ----------------------------------------------------------------------


class FaithfulnessChecker:
    """Score whether each cited claim in an answer is supported by its sources.

    Stateless — safe to reuse across requests. Construct once per
    process; pass ``llm_service`` only when the LLM stage is enabled.
    """

    def __init__(
        self,
        *,
        config: Optional[FaithfulnessConfig] = None,
        llm_service=None,
    ) -> None:
        self._cfg = config or FaithfulnessConfig()
        self._llm = llm_service

    async def check(
        self,
        *,
        answer: str,
        sources: Sequence[RetrievedItem],
    ) -> FaithfulnessResult:
        """Run the per-claim cascade against ``answer``.

        ``sources`` is the same 1-indexed list the LLM was given so the
        ``[n]`` markers in the answer line up. An out-of-range marker
        is dropped (no source to verify against); the claim still gets
        a ``ClaimVerification`` row with the remaining citations or
        ``method='uncited'`` if all of them were bad.
        """
        if not answer or not (answer or "").strip():
            return FaithfulnessResult(overall_score=None, claims=[])

        # Refusals are faithful by definition. Recognise the canonical
        # phrase from the system prompt and short-circuit.
        if _REFUSAL_NEEDLE in answer.lower():
            return FaithfulnessResult(
                overall_score=1.0,
                is_refusal=True,
                claims=[ClaimVerification(
                    claim_text=answer.strip(),
                    cited_indices=[],
                    confidence=1.0,
                    method="refusal",
                    reasoning="Answer is the configured refusal phrase.",
                )],
            )

        spans = extract_claim_spans(answer)
        verifications: List[ClaimVerification] = []
        scorable_total = 0.0
        scorable_count = 0
        failed = 0

        for span in spans:
            verification = await self._verify_span(span, sources)
            verifications.append(verification)
            if verification.method in ("text_match", "llm"):
                scorable_total += verification.confidence
                scorable_count += 1
                if verification.confidence < self._cfg.pass_threshold:
                    failed += 1

        overall = (
            round(scorable_total / scorable_count, 4) if scorable_count else None
        )
        return FaithfulnessResult(
            overall_score=overall,
            claims=verifications,
            failed_claims=failed,
        )

    # ------------------------------------------------------------------
    # Per-span scoring
    # ------------------------------------------------------------------

    async def _verify_span(
        self,
        span: ClaimSpan,
        sources: Sequence[RetrievedItem],
    ) -> ClaimVerification:
        if not span.cited_indices:
            return ClaimVerification(
                claim_text=span.text,
                cited_indices=[],
                confidence=0.0,
                method="uncited",
                reasoning="Span carries no [n] citation; excluded from score.",
            )

        # Build the verification context from every cited source. Bad
        # indices are dropped silently — the LLM may emit ``[9]`` when
        # only 5 sources came back (top-k truncation, etc.).
        contexts: List[str] = []
        valid_indices: List[int] = []
        for idx in span.cited_indices:
            if not (1 <= idx <= len(sources)):
                continue
            valid_indices.append(idx)
            contexts.append(_render_source_context(sources[idx - 1]))
        if not valid_indices:
            return ClaimVerification(
                claim_text=span.text,
                cited_indices=list(span.cited_indices),
                confidence=0.0,
                method="uncited",
                reasoning="All cited indices were out of range.",
            )

        joined_context = "\n".join(c for c in contexts if c)[: self._cfg.max_context_chars]
        claim_tokens = _tokenize(span.text, min_len=self._cfg.min_token_len)
        ratio, matched, missing = _lexical_overlap(
            claim_tokens, joined_context, min_token_len=self._cfg.min_token_len,
        )

        # Decisive ranges short-circuit the LLM. The middle band
        # (lower ≤ ratio < upper) only escalates when LLM is enabled
        # and configured.
        if not claim_tokens:
            return ClaimVerification(
                claim_text=span.text,
                cited_indices=valid_indices,
                confidence=0.0,
                method="text_match",
                reasoning="Claim has no salient tokens after stopword filter.",
                matched_terms=[],
                missing_terms=[],
            )

        if ratio >= self._cfg.upper_threshold:
            return ClaimVerification(
                claim_text=span.text,
                cited_indices=valid_indices,
                confidence=ratio,
                method="text_match",
                reasoning=f"{len(matched)}/{len(claim_tokens)} salient terms in cited context.",
                matched_terms=matched,
                missing_terms=missing,
            )

        # Optionally escalate.
        if self._cfg.enable_llm_escalation and self._llm is not None:
            llm_conf, llm_reason = await _ask_llm(
                llm_service=self._llm,
                claim=span.text,
                context=joined_context,
                max_context_chars=self._cfg.max_context_chars,
            )
            if llm_conf is not None:
                return ClaimVerification(
                    claim_text=span.text,
                    cited_indices=valid_indices,
                    confidence=llm_conf,
                    method="llm",
                    reasoning=(llm_reason or "LLM verdict")[:240],
                    escalated_to_llm=True,
                    matched_terms=matched,
                    missing_terms=missing,
                )
            # Fall through to lexical when the LLM call failed —
            # better a rough number than no number.

        return ClaimVerification(
            claim_text=span.text,
            cited_indices=valid_indices,
            confidence=ratio,
            method="text_match",
            reasoning=f"{len(matched)}/{len(claim_tokens)} salient terms in cited context.",
            matched_terms=matched,
            missing_terms=missing,
        )


def _render_source_context(item: RetrievedItem) -> str:
    """Stitch the bits of a ``RetrievedItem`` that read like supporting text.

    Order matters — the chunk preview is the most direct evidence, then
    the description (which often paraphrases the chunk for entities),
    then the label as a last resort so single-token citations like
    ``[1]`` against a bare entity name still produce *some* context.
    """
    parts: List[str] = []
    if item.chunk_preview:
        parts.append(item.chunk_preview)
    desc = item.metadata.get("description") if item.metadata else None
    if isinstance(desc, str) and desc:
        parts.append(desc)
    if item.label:
        parts.append(item.label)
    return "\n".join(parts)


__all__ = [
    "ClaimSpan",
    "ClaimVerification",
    "FaithfulnessChecker",
    "FaithfulnessConfig",
    "FaithfulnessResult",
    "extract_claim_spans",
]
