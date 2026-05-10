"""Cheap rule-based query intent classifier.

The retrieval orchestrator runs the same channel mix for every query.
The channel-quality investigation (`scripts/investigate_channels.py`)
showed that's wrong: relational questions need Cypher heavily and a
larger top-K (gold avg 8.4/q vs. final_top_k=8), lookup questions don't
need vec_rel at all (it found 0/4 gold there). Knowing the intent up
front lets ``QAService`` pick a per-intent ``RetrievalConfig`` profile
without touching the orchestrator (uses the existing ``config_override``
hook on :meth:`RetrievalOrchestrator.retrieve`).

Three classes, mirroring the gold-set labels:

* ``"lookup"``         bare term / short query, no question word.
                       Examples: ``"olaparib"``, ``"BRCA1"``,
                       ``"PARP inhibitors"``.
* ``"relational"``     asks about associations / interactions / effects.
                       Examples: ``"What is associated with X?"``,
                       ``"what does X target?"``.
* ``"definitional"``   default — ``"What is X?"``, ``"Describe X"``,
                       ``"Tell me about X"``.

Out-of-graph refusal is *not* a class here — text alone can't tell that
apart from a definitional question with no graph coverage. The
retrieve → empty-context → refuse pipeline handles that case downstream.

Rules over LLM: deterministic, latency-free, trivially debuggable. We
can layer in an LLM fallback for ambiguous cases later if eval shows
it's needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Literal, Mapping, Optional

if TYPE_CHECKING:  # pragma: no cover — type-only to avoid import cycle
    from .models import RetrievalConfig


Intent = Literal["lookup", "definitional", "relational"]


# Verb-form-only regex. Matches ``-s``, ``-ed``, ``-ing``, ``-ion``
# suffixes but not nominalisations like ``-or`` / ``-ors`` / ``-er`` /
# ``-ers`` so that "PARP inhibitors" (a noun phrase referring to drug
# entities) doesn't fire the "inhibit" branch — that one's a lookup.
_RELATIONAL_VERBS = re.compile(
    r"\b(?:"
    r"associat(?:e|ed|es|ing|ion|ions)|"
    r"relate(?:d|s)?|"
    r"target(?:s|ed|ing)?|"
    r"inhibit(?:s|ed|ing|ion|ions)?|"
    r"treat(?:s|ed|ing|ment|ments)?|"
    r"caus(?:e|es|ed|ing)|"
    r"affect(?:s|ed|ing)?|"
    r"regulat(?:e|es|ed|ing|ion)|"
    r"interact(?:s|ed|ing|ion|ions)?|"
    r"bind(?:s|ing)?|bound|"
    r"induc(?:e|es|ed|ing|tion)|"
    r"block(?:s|ed|ing)?|"
    r"activat(?:e|es|ed|ing|ion)|"
    r"modulat(?:e|es|ed|ing|ion)|"
    r"between"
    r")\b",
    re.IGNORECASE,
)

# Question/imperative starters. A query that begins with one of these
# is *not* a bare-term lookup, even if it's short.
_QUESTION_STARTERS = frozenset({
    "what", "which", "who", "when", "where", "why", "how",
    "describe", "explain", "define", "list", "find", "show", "tell",
})

# Tuned against the local gold set (`tests/eval/rag_gold_local.yaml`).
# All gold lookup queries are 1–2 tokens; 3 leaves a small margin
# without colliding with the shortest definitional queries (3+ tokens
# starting with "what").
_MAX_LOOKUP_TOKENS = 3

_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]+")


def classify_intent(query: str) -> Intent:
    """Return the predicted intent for ``query``.

    Decision order matters: relational signals beat the lookup
    short-circuit so that "what does BRCA1 target?" classifies as
    relational, not lookup.
    """
    q = (query or "").strip()
    if not q:
        return "lookup"

    if _RELATIONAL_VERBS.search(q):
        return "relational"

    tokens = _TOKEN_RE.findall(q)
    first_lc = tokens[0].lower() if tokens else ""
    if len(tokens) <= _MAX_LOOKUP_TOKENS and first_lc not in _QUESTION_STARTERS:
        return "lookup"

    return "definitional"


# ----------------------------------------------------------------------
# Per-intent retrieval profiles
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class IntentProfile:
    """Per-intent overrides applied on top of a base ``RetrievalConfig``.

    ``None`` means "don't override". The set of fields here is
    deliberately narrow — only the knobs that actually vary by intent
    in :data:`INTENT_PROFILES`. Adding a knob here without a backing
    profile change is a smell.
    """

    final_top_k: Optional[int] = None
    vector_top_k: Optional[int] = None
    enable_vector_relationship: Optional[bool] = None
    vector_relationship_top_k: Optional[int] = None
    bm25_limit: Optional[int] = None
    cypher_top_k: Optional[int] = None


# Profile values come from `tests/eval/_reports/channels/channel_investigation.md`
# (the channel-quality investigation script's most recent run). Each
# profile encodes one finding from that report; comments cite the
# specific number that justifies the override.
INTENT_PROFILES: Mapping[Intent, IntentProfile] = {
    "lookup": IntentProfile(
        # Already 100% recall on the eval — keep the channel mix cheap.
        # vec_rel found 0/4 gold here, so turn it off entirely (saves
        # one round-trip to the relationship vector index per query).
        final_top_k=8,
        vector_top_k=10,
        enable_vector_relationship=False,
        bm25_limit=10,
        cypher_top_k=5,
    ),
    "definitional": IntentProfile(
        # Balanced: vec_ent / bm25 / cypher each contribute ~1 gold/q.
        # vec_rel found 0/9 gold on definitional, so halve its budget
        # rather than disabling it (the rerank can still pick up an
        # occasional useful hit).
        final_top_k=8,
        vector_top_k=20,
        vector_relationship_top_k=10,
        bm25_limit=20,
        cypher_top_k=8,
    ),
    "relational": IntentProfile(
        # Recall bottleneck (24% on the eval). Three changes, each
        # justified by the report:
        #   * final_top_k 8 → 16: gold averages 8.4 items/q so the
        #     trim was clipping legitimate gold.
        #   * cypher_top_k 10 → 20: Cypher finds 5/8.4 gold/q on
        #     relational — the strongest channel for this intent —
        #     but only 3.25 land in the top-K. Doubling its budget
        #     gets more of those gold items into RRF.
        #   * vector_relationship_top_k 20 → 10: vec_rel takes 28% of
        #     top-K seats but contributes 1.6 gold/q. Capping its
        #     pool reduces the noise it contributes to RRF.
        final_top_k=16,
        vector_top_k=15,
        vector_relationship_top_k=10,
        bm25_limit=15,
        cypher_top_k=20,
    ),
}


def apply_profile(
    base: "RetrievalConfig", profile: IntentProfile
) -> "RetrievalConfig":
    """Return a copy of ``base`` with ``profile``'s non-None fields applied.

    Implemented as a free function (rather than a method on
    ``RetrievalConfig``) so ``models.py`` doesn't need to import this
    module — keeps the dependency direction one-way and avoids a
    circular import.
    """
    overrides = {
        f.name: getattr(profile, f.name)
        for f in fields(profile)
        if getattr(profile, f.name) is not None
    }
    if not overrides:
        return base
    return replace(base, **overrides)


def profile_for(query: str) -> IntentProfile:
    """Classify ``query`` and return the matching profile in one step.

    Convenience for the common call site (``QAService.ask``); callers
    that need the raw intent label too can call ``classify_intent``
    directly and look up the profile themselves.
    """
    return INTENT_PROFILES[classify_intent(query)]


__all__ = [
    "Intent",
    "IntentProfile",
    "INTENT_PROFILES",
    "apply_profile",
    "classify_intent",
    "profile_for",
]
