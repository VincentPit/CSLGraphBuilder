"""Unit tests for the rule-based intent classifier.

Coverage:

1. Direct rule checks — one assertion per decision branch.
2. Edge cases that the regex was specifically designed to handle
   (nominalisations like "inhibitors").
3. End-to-end accuracy against the local gold set
   (`tests/eval/rag_gold_local.yaml`) — the classifier must hit 100%
   on the in-domain labels (definitional / relational / lookup).
   Out-of-graph refusal questions are not predicted; we only assert
   the classifier doesn't crash on them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from graphbuilder.core.retrieval.intent import (
    INTENT_PROFILES,
    IntentProfile,
    apply_profile,
    classify_intent,
    profile_for,
)
from graphbuilder.core.retrieval.models import RetrievalConfig


# ---------------------------------------------------------------- direct rules


@pytest.mark.parametrize(
    "query,expected",
    [
        # Lookup — bare terms.
        ("olaparib", "lookup"),
        ("BRCA1", "lookup"),
        ("BRCA2", "lookup"),
        ("PARP inhibitors", "lookup"),                 # nominalisation, not verb
        ("EGFR mutations", "lookup"),
        # Definitional — "what is X" / "describe X".
        ("What is brca1?", "definitional"),
        ("What is homologous recombination?", "definitional"),
        ("Describe atrial fibrillation", "definitional"),
        ("Tell me about pancreatitis", "definitional"),
        # Relational — explicit association / verb language.
        ("What is associated with brca1?", "relational"),
        ("what does BRCA1 target?", "relational"),
        ("How does olaparib inhibit PARP?", "relational"),
        ("drugs that target EGFR", "relational"),
        ("relationship between BRCA1 and BRCA2", "relational"),
        ("how does aspirin treat inflammation", "relational"),
    ],
)
def test_classify_intent_rules(query: str, expected: str) -> None:
    assert classify_intent(query) == expected


def test_empty_query_returns_lookup() -> None:
    assert classify_intent("") == "lookup"
    assert classify_intent("   ") == "lookup"


def test_inhibitor_nominalisation_is_not_relational() -> None:
    """Regression for the verb-vs-noun ambiguity: "inhibitors" as a noun
    must not fire the relational branch — the user is naming a drug
    class, not asking about a relationship."""
    assert classify_intent("PARP inhibitors") == "lookup"
    assert classify_intent("EGFR inhibitor") == "lookup"


def test_inhibits_verb_is_relational() -> None:
    """The verb form *should* fire."""
    assert classify_intent("what inhibits PARP?") == "relational"
    assert classify_intent("BRCA1 inhibits which targets") == "relational"


def test_relational_wins_over_lookup_when_both_apply() -> None:
    """A short query with a relational verb is classified as relational,
    not lookup. Decision-order regression."""
    # 3 tokens, no question word, but the verb fires first.
    assert classify_intent("BRCA1 inhibits TP53") == "relational"


# ---------------------------------------------------------------- gold parity


_GOLD_PATH = Path(__file__).resolve().parents[1] / "eval" / "rag_gold_local.yaml"


def _load_gold_questions() -> list[tuple[str, str, str]]:
    """Parse the gold YAML without taking a yaml dependency.

    The file is mechanically generated and uses a stable shape — flat
    list of records with ``id``, ``question``, ``intent`` fields one
    per line. Returns ``[(id, intent, question), ...]``.
    """
    text = _GOLD_PATH.read_text()
    id_re = re.compile(r"^- id:\s*(\S+)\s*$", re.MULTILINE)
    q_re = re.compile(r"^\s+question:\s*(.+?)\s*$", re.MULTILINE)
    intent_re = re.compile(r"^\s+intent:\s*(\S+)\s*$", re.MULTILINE)
    ids = id_re.findall(text)
    questions = q_re.findall(text)
    intents = intent_re.findall(text)
    assert len(ids) == len(questions) == len(intents), (
        f"gold yaml shape unexpected: {len(ids)=} {len(questions)=} {len(intents)=}"
    )
    return list(zip(ids, intents, questions))


def test_classifier_matches_gold_intents_on_in_domain_questions() -> None:
    """End-to-end accuracy on the local gold set.

    Refusal (``out_of_graph``) questions are skipped — the classifier
    isn't designed to predict that class. Every other gold question
    must be predicted correctly; any drift means the rules need an
    update before we ship intent-aware routing.
    """
    rows = _load_gold_questions()
    in_domain = [r for r in rows if r[1] != "out_of_graph"]
    assert len(in_domain) >= 20, "gold set unexpectedly small"

    mismatches: list[tuple[str, str, str, str]] = []
    for qid, gold, question in in_domain:
        pred = classify_intent(question)
        if pred != gold:
            mismatches.append((qid, gold, pred, question))
    assert not mismatches, (
        "classifier drifted from gold:\n"
        + "\n".join(
            f"  {qid}: gold={g} pred={p} q={q!r}"
            for qid, g, p, q in mismatches
        )
    )


# ---------------------------------------------------------------- profiles


def test_intent_profiles_cover_every_intent_label() -> None:
    """Every intent the classifier can return must have a profile —
    otherwise ``profile_for`` will raise KeyError on a real query."""
    assert set(INTENT_PROFILES.keys()) == {"lookup", "definitional", "relational"}


def test_lookup_profile_disables_vector_relationship() -> None:
    """The investigation showed vec_rel found 0/4 gold on lookup. The
    profile must turn it off — that's the latency win for this intent."""
    profile = INTENT_PROFILES["lookup"]
    assert profile.enable_vector_relationship is False


def test_relational_profile_widens_topk_and_boosts_cypher() -> None:
    """Relational profile must address the two findings from the report:
    final_top_k must beat the gold-avg ceiling (8.4), and Cypher's
    budget must be larger than the default since it carries the intent."""
    base = RetrievalConfig()
    profile = INTENT_PROFILES["relational"]
    assert profile.final_top_k is not None and profile.final_top_k > base.final_top_k
    assert profile.cypher_top_k is not None and profile.cypher_top_k > base.cypher_top_k


def test_apply_profile_overrides_only_non_none_fields() -> None:
    """``None`` profile fields leave the base config alone — they're the
    'don't override' sentinel. Untouched fields must round-trip exactly."""
    base = RetrievalConfig(
        final_top_k=8,
        vector_top_k=20,
        bm25_limit=20,
        cypher_top_k=10,
        rrf_k=60,
        chunk_neighbour_radius=2,
    )
    profile = IntentProfile(final_top_k=16, cypher_top_k=20)  # partial override
    merged = apply_profile(base, profile)
    assert merged.final_top_k == 16          # overridden
    assert merged.cypher_top_k == 20         # overridden
    assert merged.vector_top_k == 20         # untouched
    assert merged.bm25_limit == 20           # untouched
    assert merged.rrf_k == 60                # untouched (not in profile)
    assert merged.chunk_neighbour_radius == 2  # untouched (not in profile)


def test_apply_profile_returns_new_instance_does_not_mutate_base() -> None:
    """``apply_profile`` must be pure — repeated calls with the same
    base must produce identical results, and the base must be unchanged
    so a shared singleton config can be reused per turn."""
    base = RetrievalConfig(final_top_k=8)
    profile = IntentProfile(final_top_k=16)
    merged_a = apply_profile(base, profile)
    merged_b = apply_profile(base, profile)
    assert merged_a is not base
    assert merged_a.final_top_k == 16
    assert merged_b.final_top_k == 16
    assert base.final_top_k == 8  # base unchanged


def test_apply_profile_with_empty_profile_returns_base() -> None:
    """All-None profile is a no-op; the function returns the base
    unchanged (same instance) — small optimisation for the common
    'no override needed' case."""
    base = RetrievalConfig()
    merged = apply_profile(base, IntentProfile())
    assert merged is base


def test_apply_profile_threads_through_step1_knobs() -> None:
    """The new step-1 knobs (``enable_vector_relationship``,
    ``vector_relationship_top_k``) must be reachable from a profile —
    otherwise the lookup profile can't actually disable vec_rel."""
    base = RetrievalConfig()
    merged = apply_profile(base, INTENT_PROFILES["lookup"])
    assert merged.enable_vector_relationship is False


def test_profile_for_routes_classifier_output_to_correct_profile() -> None:
    """End-to-end: ``profile_for(query)`` must equal
    ``INTENT_PROFILES[classify_intent(query)]`` for every gold question."""
    rows = _load_gold_questions()
    for _qid, gold, question in rows:
        if gold == "out_of_graph":
            continue
        assert profile_for(question) is INTENT_PROFILES[gold]


def test_classifier_handles_out_of_graph_questions_without_crash() -> None:
    """Refusal questions can be predicted as anything sensible, but
    must not raise. The downstream pipeline (empty retrieval → LLM
    refusal) handles the actual refusal logic."""
    rows = _load_gold_questions()
    refusals = [r for r in rows if r[1] == "out_of_graph"]
    if not refusals:
        pytest.skip("no out_of_graph questions in gold set")
    for _qid, _intent, question in refusals:
        pred = classify_intent(question)
        assert pred in {"lookup", "definitional", "relational"}
