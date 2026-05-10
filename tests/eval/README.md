# RAG eval harness — `tests/eval/`

P13 of [`docs/RAG_QA_PLAN.md`](../../docs/RAG_QA_PLAN.md). Two
artefacts ship together:

1. **A library** — `src/graphbuilder/core/eval/` — gold-set loader,
   metric math, async runner, CSV/markdown reports. Transport-agnostic:
   the runner takes a callable `ask_fn(query) -> AskLike`, so the same
   code drives the hermetic CI gate, ablation matrices, and the live
   API runner.

2. **An entry-point set** — this directory:
   - `rag_gold.yaml` — the curated gold set (§9.1). Seeded with four
     hand-crafted questions covering the four main intents; grow it
     from the curation review queue as new approved reviews land.
   - `baselines.json` — pinned thresholds (§9.3). Hermetic floors
     enforce the CI gate; live targets mirror §9.2.
   - `test_eval_metrics.py` — unit tests for metric math (P/R/F1,
     percentile, harmonic mean, edge cases).
   - `test_eval_smoke.py` — hermetic CI gate: builds an in-memory graph,
     runs the harness against a tiny embedded gold list, asserts the
     summary clears every floor in `baselines.json`. Fails the build if
     anyone regresses fusion / hydration / rerank pass-through.
   - `run_rag_eval.py` — CLI runner. Hits a live `/qa/ask` over HTTP,
     writes `rag_eval.csv` + `rag_eval.md` next to the gold file.

## Running

```bash
# Hermetic: runs in pytest CI, no Neo4j/LLM required.
pytest tests/eval/

# Live: hits a deployed API.
python tests/eval/run_rag_eval.py \
    --api https://chat.staging.local \
    --api-key "$GRAPHBUILDER_API_KEY" \
    --gold tests/eval/rag_gold.yaml \
    --out tests/eval/_reports/

# Live with an ablation matrix (vector-only, +Cypher, +BM25, all):
python tests/eval/run_rag_eval.py \
    --api https://chat.staging.local --api-key "$KEY" \
    --gold tests/eval/rag_gold.yaml \
    --ablations vector,vector+cypher,vector+bm25,all
```

## Growing the gold set

Per §9.1 of the plan, the source of truth for new questions is the
curation review queue: pick a recently-approved review, write the
question a user might have asked to surface those entities, and pin
the approved entity / relationship / chunk ids as gold. A few rules of
thumb that have held up so far:

- **Bias toward symbol-y queries** ("BCR-ABL", drug codes) — those are
  what the BM25 channel exists to catch and where regressions hide.
- **Mix intents** — the harness reports a single F1 today, but the
  per-intent breakdown in the markdown report is what tells you
  *which* channel regressed.
- **Pin the chunk id** when you can. Context recall (§9.2) only works
  when at least some questions have `gold_chunk_ids` set.
- **Use `gold_answer_substrings` sparingly** but prefer it to a free
  string match — case-insensitive any-of is robust to phrasing churn.

## Updating baselines

The hermetic floors in `baselines.json` are tied to the embedded
in-memory graph in `test_eval_smoke.py`. Bump them only when the
fake graph gets richer (more entities, better relationships, more
chunks); never to chase a regression. The live targets mirror §9.2 of
the plan and should track that document.

## What's not yet here

- **Per-intent breakdowns** — the records carry intent, but the
  markdown summary aggregates them all. Easy follow-up; deferred to
  keep the diff bounded.

## Faithfulness scoring (P8 — shipped)

Every record now carries `answer_faithfulness` and `EvalSummary`
exposes the macro-average. The QA service runs
[`FaithfulnessChecker`](../../src/graphbuilder/core/retrieval/faithfulness.py)
after answer generation: each `[n]`-bounded claim is scored by
lexical overlap against its cited source's chunk preview, with
optional LLM escalation in the inconclusive band (off by default to
keep the eval cost bounded). Refusals short-circuit to 1.0. Hermetic
floor is `0.50`; live target mirrors §9.2 at `0.85`.
