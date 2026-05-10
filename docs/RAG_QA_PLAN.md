# RAG Q&A System — Design Plan

Branch: `feature/chatbot`
Plan drafted: 2026-05-08
Last updated: 2026-05-10

## Status

**Shipped (14 commits on `feature/chatbot`, 334 unit tests passing):**
P0 plan · P1 conversation persistence · P2 observability spine · P3 retrieval orchestrator (Cypher + vector + BM25 + RRF + chunk hydration) · P4 cross-encoder rerank + chunk neighbour expansion · P5 minimal `/qa/ask` endpoint · P6 working memory + rolling summary · P7 episodic recall · P12 `/chat` frontend with per-source confidence + retrieval trace · P13 eval harness (gold loader + metrics + ablation runner + CSV/markdown reports + hermetic CI gate + live `run_rag_eval.py` CLI) · plus an unplanned **lightweight browser identity** (X-User-Id, chat-only, see §14.1) and four eval-driven follow-ups: **re-parse guard**, **chunks-as-first-class-items**, **per-request ablation overrides**, **cross-type entity dedup CLI** (see §13).

**Open / next:**
P8 faithfulness check (now unblocked — eval harness has a slot waiting for it) · P11 SSE streaming · P9/P10 tool-use surface (gated on §14.6 — mutation authority) · P14 cross-session semantic memory.

**First live eval results (23-question grounded gold set, 2026-05-10):** baseline P=0.141, R=0.519, F1=0.188, ctx=0.870, cov=0.870, p95=3.5 s warm. Cross-encoder rerank earns its keep (-21 % F1 without it). Vector-only narrowly beats all-channels on F1 → BM25 + Cypher are bringing in noise the rerank can't fully clean up; templates + term extraction are the next thing to look at. See §9 and `tests/eval/_reports/v2/rag_eval.md`.

See §13 for the full phase table with commit refs.

---

## 0. Context — what we already have

The verification module (`src/graphbuilder/core/verification/`) gives us a 3-stage cascading scorer (text → embedding → LLM) that we can repurpose as the **answer-grounding** layer of a RAG pipeline. The frontend already exposes `/verification` with stage-by-stage confidence + reasoning — we'll mirror that UX so users can see *why* an answer was given.

Reusable building blocks already in the codebase:

| Layer | What exists | Where |
|---|---|---|
| Graph store | Neo4j with `:Entity`, `:RELATES`, `:Document`, `:DocumentChunk`, `:HAS_CHUNK`, `:NEXT_CHUNK`, `:EXTRACTED_FROM` | [graph_repository.py](src/graphbuilder/infrastructure/repositories/graph_repository.py) |
| Vector indexes | `entity_name_vector` (768-d, cosine), `rel_desc_vector` (768-d, cosine) | same file, lines 197-243 |
| Full-text index | `entity_search` on `name` + `description` | same file |
| Embeddings | SapBERT 768-d default, MiniLM fallback, GPU pool, async batched | [embedding_factory.py](src/graphbuilder/infrastructure/services/embedding_factory.py) |
| LLM | `AdvancedLLMService.generate_text(...)`, retries, OpenAI/Azure | [llm_service.py](src/graphbuilder/infrastructure/services/llm_service.py) |
| Chunk lookup | `DocumentRepo.get_chunks_by_ids(...)`, `:NEXT_CHUNK` linked list | [document_repository.py](src/graphbuilder/infrastructure/repositories/document_repository.py) |
| Provenance | `source_chunk_ids` + `source_document_ids` on every entity/relationship | [graph_models.py](src/graphbuilder/domain/models/graph_models.py#L186) |
| Verification cascade | text-match → embedding → LLM with reasoning trace | [cascading.py](src/graphbuilder/core/verification/cascading.py) |
| API style | FastAPI, `X-API-Key` auth, SSE pattern on `/documents/jobs/{id}/stream` | [api/routers/](api/routers/) |
| Frontend | Next.js, react-query + axios, Tailwind-ish CSS vars | [frontend/app/verification/page.tsx](frontend/app/verification/page.tsx) |

**Gaps we must build:**
1. Chunk-level embeddings are **not** stored (only entity-name + relationship-desc). Either add a chunk vector index, or rely on entity/relationship hits → chunk lookup. Plan goes with option B for v1, option A for v2 (see §3.2).
2. No conversation/session storage. We'll add `:ConversationSession` + `:ConversationTurn` nodes.
3. No `/chat` or `/qa` endpoint. To be added.
4. No retrieval-evaluation harness. To be added.

---

## 1. Goals (from the brief, restated)

1. **Precise retrieval** — graph + vector hybrid with explicit ranking, measured on Precision/Recall/F1 against a curated gold set.
2. **Provenance & per-component confidence** — every answer ships with its sources (entities, relationships, chunks, URLs) and a confidence score per source.
3. **Context-window-aware long-term memory** — the bot must stay coherent across long conversations and across sessions, using production-grade memory layering (working / summary / episodic / semantic).
4. **Tool-using assistant** — the bot can mutate the Neo4j graph on user request (insert / update / merge / soft-delete entities and relationships) through a constrained tool layer, and the user can see the result of each action (a focused subgraph of the touched entities + relationships).
5. **Full observability** — every retrieval, tool call, mutation, and LLM call is logged with a correlation id, persisted as an audit trail, and surfaced in metrics + a developer-mode debug pane.

Non-goals for v1: multi-modal input, free-form Cypher generation by the LLM (we ship a fixed tool surface), fine-tuned models.

---

## 2. High-level architecture

```
                             ┌───────────────────┐
  user query  ──────────────►│  Query Planner    │
                             │ (intent + decomp) │
                             └────────┬──────────┘
                                      │
              ┌───────────────────────┼─────────────────────────┐
              ▼                       ▼                         ▼
     ┌──────────────┐         ┌──────────────┐          ┌──────────────┐
     │ Cypher path  │         │ Vector path  │          │ BM25/text    │
     │ (graph-aware │         │ entity+rel   │          │ (substring + │
     │  traversal)  │         │ vec search   │          │  fulltext)   │
     └──────┬───────┘         └──────┬───────┘          └──────┬───────┘
            │                        │                          │
            └──────────► RRF fuse ◄──┴──────────────────────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │  Cross-encoder      │
                │  reranker (top 50→k)│
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ Chunk hydration     │
                │ + neighbour expand  │
                │ (NEXT_CHUNK ±1)     │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ Context packer      │   ← memory layers
                │ (token-budgeted)    │   ←──── working / summary /
                └─────────┬───────────┘        episodic / semantic
                          │
                          ▼
                ┌─────────────────────┐
                │ LLM answer + cite   │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ Faithfulness check  │   ← reuses CascadingVerifier
                │ (per-claim score)   │
                └─────────┬───────────┘
                          │
                          ▼
                  SSE stream back
                          │
                          ▼
              ConversationTurn persisted
              (user_q, llm_a, retrieved_ids, scores, feedback)
```

Two cross-cutting concerns wrap the whole pipeline (detailed in §7 and §8):

- **Tool-use lane.** The LLM can also emit `tool_call`s (read-only `search_graph` / `get_entity` / `verify_claim`, or mutating `propose_*` / `update_*` / `merge_entities` / `soft_delete_*`). Reads execute inline; mutations are parked as `PendingMutation` until the user confirms — at which point they apply through the existing `CurationUseCase` and return a focused subgraph preview to render inline.
- **Observability lane.** Every box above emits structured logs, metrics, and (for mutations) audit rows under one `request_id`. The frontend's debug pane (`?debug=1`) reads back the same `TurnTrace` to show users exactly what the bot did.

---

## 3. Retrieval — precise & multi-signal

### 3.1 Query understanding

A `QueryPlanner` runs **before** retrieval. Cheap LLM call (or rule-based fallback) classifies intent into:

- **Lookup** — "what is X?" → favour entity vector search + entity description.
- **Relational** — "what does X target?" / "what drugs treat Y?" → run **targeted Cypher** based on detected entity names + relationship type vocabulary.
- **Multi-hop** — "what diseases share targets with X?" → 2-hop Cypher with intermediate variable.
- **Definitional/summary** — "tell me about X" → entity + 1-hop neighbourhood.
- **Out-of-graph** — fallback to chunk text only (or refuse with "no evidence in KB").

The planner also extracts:
- Candidate entity mentions (NER via the existing `EntityVerifier` text-match path; cheap).
- Relationship-type hints (matched against `RelationshipType` enum vocabulary).
- Time/quantity constraints (regex; future).

### 3.2 Three retrieval channels (run in parallel)

**Channel A — Graph-aware Cypher** (highest precision when intent is relational)

For relational/multi-hop intents, generate a parameterised Cypher template, e.g.:

```cypher
MATCH (s:Entity)-[r:RELATES]->(t:Entity)
WHERE toLower(s.name) IN $names
  AND r.relationship_type IN $rel_types
RETURN s, r, t, r.strength as score
ORDER BY score DESC LIMIT $k
```

Templates are **hand-written** (not LLM-generated) for v1 — safer, predictable, no Cypher injection risk. ~10 templates cover the main intents.

**Channel B — Dual vector search** (highest recall for fuzzy phrasing)

```python
q_vec = await embed_async(query)
ent_hits = await graph_repo.vector_search_entities(q_vec, top_k=20, min_score=0.5)
rel_hits = await graph_repo.vector_search_relationships(q_vec, top_k=20, min_score=0.5)
```

**Channel C — BM25/keyword fallback** (catches exact terms that vectors miss)

```python
text_hits = await graph_repo.search_entities_by_text(extracted_terms, limit=20)
```

### 3.3 Ranking — RRF + cross-encoder rerank

**Stage 1 — Reciprocal Rank Fusion** across the 3 channels:

```
score(d) = Σ_channels 1 / (k + rank_channel(d))   # k=60 (standard)
```

RRF is rank-based, so it doesn't need score calibration across channels (Cypher returns no score, vector returns cosine, BM25 returns idf). Output: top 50 candidates.

**Stage 2 — Cross-encoder rerank** on top 50 → top `k` (default 8).

For v1, reuse the existing sentence-transformers stack with a small cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~22 MB, CPU-friendly). Each candidate is scored as `(query, candidate.name + " — " + candidate.description)`.

For biomedical work we should swap to a domain cross-encoder later (e.g. `pritamdeka/S-BioBert-snli-multinli-stsb`); this is a config knob.

### 3.4 Chunk hydration

Each ranked entity/relationship carries `source_chunk_ids`. We:

1. Dedup chunk IDs across all top-k items.
2. Fetch via `DocumentRepo.get_chunks_by_ids(...)`.
3. **Expand neighbours**: for each chunk, follow `(:DocumentChunk)-[:NEXT_CHUNK]->(:DocumentChunk)` up to ±1 (configurable) so the LLM sees the surrounding paragraph, not just the matched sentence. This is critical for biomedical text where the claim and the qualifier ("…in mouse models only…") are often in adjacent sentences.

### 3.5 Why this is "precise"

- **Cypher channel** gives us deterministic, schema-aware answers when the question is graph-shaped.
- **Vector channel** uses domain-tuned SapBERT — already proven for biomedical synonym resolution.
- **BM25 channel** catches exact identifiers (gene symbols, drug codes) that embeddings sometimes blur.
- **RRF** removes the need to calibrate scores; **cross-encoder** gives a real semantic re-rank instead of cosine alone.
- **Chunk neighbour expansion** prevents missing context that lives one sentence over.

### 3.6 Settings (new section in `GraphBuilderConfig`)

```python
@dataclass
class RAGRetrievalConfig:
    # channels
    enable_cypher_channel: bool = True
    enable_vector_channel: bool = True
    enable_bm25_channel: bool = True
    # vector
    vector_top_k: int = 20
    vector_min_score: float = 0.5
    # rrf
    rrf_k: int = 60
    rrf_top_n: int = 50
    # rerank
    enable_cross_encoder: bool = True
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    final_top_k: int = 8
    # chunk hydration
    chunk_neighbour_radius: int = 1
    max_chunk_chars: int = 1200
```

---

## 4. Provenance & per-component confidence (for the frontend)

Every retrieved item that ends up in the final answer carries a **provenance bundle**:

```python
@dataclass
class RetrievedItem:
    kind: Literal["entity", "relationship", "chunk"]
    id: str
    label: str                      # entity.name, "X --REL--> Y", or chunk preview
    score_vector: Optional[float]   # cosine, if it came from vector channel
    score_bm25: Optional[float]
    score_rrf: float
    score_rerank: float             # cross-encoder score
    final_confidence: float         # normalised [0,1] for UI
    source_url: Optional[str]
    source_doc_id: Optional[str]
    source_chunk_id: Optional[str]
    chunk_preview: Optional[str]    # first ~200 chars for hover card
    reasoning: str                  # one-line "why retrieved"
```

**Final confidence** for each item is a weighted blend of:
- rerank score (0.5)
- best raw channel score (0.3)
- citation-coverage (0.2) — was this item actually cited by the LLM in its answer? See §6.

The API returns `RetrievedItem[]` alongside the answer text. Frontend renders:

```
┌─ Answer ───────────────────────────────────────────────┐
│ Imatinib targets BCR-ABL [1] and KIT [2]…             │
└────────────────────────────────────────────────────────┘
┌─ Sources ──────────────────────────────────────────────┐
│ [1] Entity: BCR-ABL  ░░░░░░░░░▓▓ 0.92                 │
│     ▸ from PubMed:12345 · "…the BCR-ABL fusion is…"   │
│ [2] Relationship: imatinib --INHIBITS--> KIT  ░░▓▓ 0.81│
│     ▸ from FDA label · "…inhibits KIT signalling…"    │
└────────────────────────────────────────────────────────┘
```

We mirror the verification page's stage-bar component for the confidence bar.

---

## 5. Context-window management — production-grade memory

The single hardest part. Approach:  **four memory layers**, each with a clear retention policy.

### 5.1 Working memory (in-context, verbatim)

The last **N=3** user/assistant turns, full text. No compression. Always included.

### 5.2 Rolling summary (in-context, compressed)

Older turns are summarised into a single rolling paragraph kept under ~500 tokens. Implemented as: when working-memory window slides, the turn that falls out is fed to a cheap summariser:

```
"Summarise the following exchange in 1-2 sentences, preserving any
named entities, drug/gene IDs, and user-stated preferences. Existing
running summary: {old_summary}. New turn: {Q} / {A}"
```

The summary is **regenerated** (not appended) each slide so it stays coherent.

### 5.3 Episodic memory (out-of-context, retrievable)

Every turn is persisted as a `:ConversationTurn` node:

```
(:ConversationSession {id, user_id, started_at, last_active_at, summary})
       -[:HAS_TURN {idx}]->
(:ConversationTurn {id, ts, user_query, llm_answer, query_embedding,
                    answer_embedding, retrieved_item_ids, feedback})
       -[:CITES]->(:Entity | :DocumentChunk | :Relationship)
```

The `query_embedding` lets us do **memory retrieval** before answering: when the user says "what about its side effects?", we vector-search prior turns in the session for the most relevant one, then resolve "it" by looking at the cited entities of that turn. This is more reliable than coreference resolution alone.

### 5.4 Semantic memory (out-of-context, cross-session)

Per user, after a session ends, an LLM produces a **persona/preferences summary** ("user is a clinical pharmacologist, asks at the level of mechanism not indication, prefers structured answers…"). Stored on `(:User {id, semantic_summary, semantic_embedding})`.

On every new session start, this summary is loaded into the system prompt — so the bot is "warm" across sessions.

### 5.5 Token budget (per request)

Hard budget for the LLM context (gpt-4o = 128k, but we **cap at 16k for cost+latency**):

| Slot | Budget | What goes in |
|---|---|---|
| System prompt | 1.5k | task instructions + user semantic summary |
| Working memory | 3k | last 3 turns verbatim |
| Rolling summary | 0.5k | compressed older history |
| Episodic recall | 1k | top-1 prior turn from this session if relevant |
| Retrieved context | 8k | top-k entities/rels + hydrated chunks (priority order) |
| Answer reservation | 2k | reserved for response |
| **Total** | **16k** | |

If retrieved-context overflows, drop in this priority order: chunks furthest from cited entity → low-rerank-score chunks → relationship descriptions → entity descriptions. We **never** drop the cited entities themselves.

### 5.6 Why not just "summarise everything"?

Pure summarisation **loses identifiers** (drug codes, gene symbols, p-values) — which are exactly what biomedical users care about. The episodic-recall layer keeps the precise data retrievable even when the summary has compressed it away. This is the same pattern OpenAI's Memory and Anthropic's projects use under the hood.

---

## 6. Answer generation + faithfulness

### 6.1 Generation prompt (sketch)

```
SYSTEM
You are a biomedical knowledge-graph assistant. Answer ONLY from the
sources provided. Cite each factual claim with [n] where n is the
1-indexed source number. If the sources do not contain the answer,
say "I cannot find this in the knowledge base." — do not guess.

USER SEMANTIC SUMMARY
{user_persona}

ROLLING SUMMARY OF EARLIER CONVERSATION
{rolling_summary}

RECENT TURNS
{working_memory}

POSSIBLY RELEVANT EARLIER TURN
{episodic_hit_if_any}

SOURCES
[1] Entity: BCR-ABL — Tyrosine kinase fusion protein…
    From PubMed:12345 (chunk 7): "…the BCR-ABL fusion is…"
[2] Relationship: imatinib INHIBITS KIT (strength 0.92)
    From FDA label (chunk 3): "…inhibits KIT signalling…"
…

QUESTION
{user_query}
```

Temperature 0.1, JSON mode off (we want prose with `[n]` markers).

### 6.2 Faithfulness check (reuses verification cascade)

After generation, we extract claims from the answer (split on `[n]` boundaries) and run each through the existing `CascadingVerifier`, where:
- **context** = the chunks cited by `[n]`
- **relationship** = a synthesised `(claim_subject) --MENTIONS--> (claim_object)`

A claim that fails verification (confidence < 0.5) is flagged in the UI with a yellow underline. This gives us **per-claim** confidence on top of the per-source confidence in §4.

### 6.3 Citation coverage feeds back into source confidence

If source `[2]` was retrieved but never cited by the LLM, its `final_confidence` drops by 0.2. This makes the source list match what was actually used, not just what was retrieved.

---

## 7. Tool-use — graph mutations from chat

The chatbot can also **act on the graph**, not just read from it. We expose a constrained, well-typed tool surface to the LLM. The model never writes raw Cypher; it picks a tool from a fixed list and fills in a JSON schema. The tool layer validates, executes, audit-logs, and returns a structured result the frontend can render as a focused subgraph preview.

### 7.1 Why a tool layer (not free-form Cypher)

- **Safety** — no Cypher injection; the LLM cannot bypass dedup, verification, or provenance.
- **Validation** — every tool call is a Pydantic model; bad arguments are rejected before they hit Neo4j.
- **Auditability** — every call routes through the existing `CurationUseCase` flow ([curation.py:36](src/graphbuilder/application/use_cases/curation.py#L36)), so mutations land in the same audit log as human-curator actions and inherit the existing review queue.
- **Reuse** — `save_entities_batch`, `save_relationships_batch`, `merge_entities` already exist on the repo; we don't reinvent persistence.

### 7.2 Tool surface (v1)

Exposed to the LLM via OpenAI-style function calling. Each tool has a JSON schema, a Pydantic input model, and an executor.

| Tool | Purpose | Backed by |
|---|---|---|
| `search_graph` | Read-only — find entities/relationships matching a name, type, or vector query. Used liberally; no audit row. | retrieval orchestrator (§3) |
| `get_entity` | Fetch one entity + its 1-hop neighbourhood for inspection. | `graph_repo.get_entity_by_id` + `get_entity_relationships` |
| `propose_entity` | Create a new entity (name, type, description, aliases, external_ids). Goes through the same dedup cascade as ingestion. | `save_entity` ([line 297](src/graphbuilder/infrastructure/repositories/graph_repository.py#L297)) |
| `propose_relationship` | Create a new relationship between two existing entity ids. | `save_relationship` ([line 447](src/graphbuilder/infrastructure/repositories/graph_repository.py#L447)) |
| `update_entity` | Patch fields (description, aliases, external_ids). Diff is recorded. | `save_entity` (upsert) |
| `merge_entities` | Collapse two entities the user identifies as the same concept. | `merge_entities` ([line 1419](src/graphbuilder/infrastructure/repositories/graph_repository.py#L1419)) |
| `soft_delete_entity` | Mark `verification_status="rejected"`; node remains for audit, hidden from default queries. Hard delete is **not** exposed. | `CurationAction.REJECT_ENTITY` |
| `soft_delete_relationship` | Same pattern for relationships. | `CurationAction.REJECT_RELATIONSHIP` |
| `verify_claim` | Run the existing 3-stage cascade against a free-text claim; returns stage-by-stage scores (no mutation). | `CascadingVerifier` |

Hard delete is intentionally absent — `REJECT` + audit retention beats irreversible removal. If a power user needs hard delete we add it later behind a separate role check.

### 7.3 Confirmation policy (the "two-key" rule)

Every **mutating** tool call requires explicit user confirmation before execution. The LLM proposes; the user clicks confirm. This is the single most important safety rule and is enforced server-side, not just in the UI:

```
LLM emits tool_call → server validates schema
                    → server stores PendingMutation { id, tool, args, ttl=15min }
                    → frontend renders "Apply this change?" card with diff preview
                    → on user confirm: POST /qa/mutations/{id}/apply  → executes
                    → on reject:        POST /qa/mutations/{id}/reject → records reason
```

The LLM *cannot* call mutating tools auto-approved, even in agentic loops. Read-only tools (`search_graph`, `get_entity`, `verify_claim`) bypass confirmation.

### 7.4 Result rendering — focused subgraph preview

After a mutation applies, the response includes a `MutationResult`:

```python
@dataclass
class MutationResult:
    tool: str
    target_kind: Literal["entity", "relationship"]
    target_id: str
    operation: Literal["created", "updated", "merged", "soft_deleted"]
    diff: dict                     # before → after, field-by-field
    affected_subgraph: SubgraphSlice  # reuses graph_repo.get_subgraph_slice
    audit_id: str                  # FK into the curation audit log
    confirmation_id: str
```

The `affected_subgraph` is a small slice (touched entity + 1-hop neighbours, capped ~15 nodes) — exactly the shape the existing `/graph/subgraph` endpoint already returns. The frontend renders it inline beneath the chat bubble using the same Cytoscape/D3 component the `/graph` page uses, so users *see* what changed:

```
bot: I've added "GLP-1 receptor agonist" as a new entity (drug class)
     and linked semaglutide → GLP-1R via TARGETS.

     ┌─ Resulting subgraph (touched + neighbours) ────────────┐
     │            (semaglutide) ─TARGETS→ (GLP-1R) NEW       │
     │                  ↓                       ↑            │
     │             (treats T2D)            (modulates GIP)   │
     └────────────────────────────────────────────────────────┘
     [✓ applied · audit #c4f3 · undo within 15min]
```

### 7.5 Undo

Each mutation has a 15-minute reversible window. Undo posts a compensating mutation:
- `created` → `soft_delete`
- `updated` → restore prior field values from the diff
- `soft_deleted` → restore `verification_status`
- `merged` → split is **not** reversible automatically; we keep the merge audit record and require manual curator action (flagged honestly to the user)

### 7.6 LLM prompt for tool-use

The system prompt grows a tool-use section that explicitly forbids guessing IDs:

```
TOOLS
You may call the listed tools. Never invent entity_ids — first call
search_graph or get_entity to confirm a target exists. Mutating tools
require user confirmation; do not assume a previous confirmation
applies to a new call. If a user request is ambiguous (e.g. "delete
that protein"), ask which one rather than picking.
```

### 7.7 Settings

```python
@dataclass
class RAGToolConfig:
    enable_mutations: bool = True
    require_confirmation: bool = True
    confirmation_ttl_seconds: int = 900
    undo_window_seconds: int = 900
    max_tool_calls_per_turn: int = 5     # cap agentic loops
    max_subgraph_preview_nodes: int = 15
```

---

## 8. Observability — logging, metrics, audit

Three layers, each with a different audience: **logs** (engineers debugging), **metrics** (production health), **audit** (compliance + user trust).

### 8.1 Correlation id — the spine

Every chat turn gets a `request_id` (uuid4) at API entry. It propagates through:
- structured log records (`extra={"request_id": ...}`)
- metrics labels
- the persisted `:ConversationTurn.request_id` property
- every audit row tied to that turn
- the SSE stream as the first event

This is what makes the debug pane (§8.5) possible — one id pulls together logs, metrics, audit, and the trace the LLM saw.

### 8.2 Structured logging

Reuse the existing `logging.getLogger("graphbuilder.<module>")` pattern; add a new namespace for the chat path:

| Logger | What it emits |
|---|---|
| `graphbuilder.qa.api` | request received, response sent, status, latency |
| `graphbuilder.qa.planner` | classified intent, extracted entities, chosen retrieval channels |
| `graphbuilder.qa.retrieval` | per-channel hit counts, RRF top-N, rerank top-K, chunk hydration count |
| `graphbuilder.qa.memory` | which memory layers were consulted, episodic-recall hits, token budget breakdown |
| `graphbuilder.qa.llm` | model, prompt-token / completion-token counts, latency, retry count, finish_reason |
| `graphbuilder.qa.tools` | tool name, args (PII-scrubbed), validation outcome, confirmation status |
| `graphbuilder.qa.mutations` | applied operation, diff, audit_id, undo eligibility |
| `graphbuilder.qa.faithfulness` | per-claim verification stage results |

All emit JSON when `LOG_FORMAT=json` (already supported by the project's logging config). Sensitive fields (raw user text, full chunk text) get truncated to a fixed length in logs; full text lives only in the audit store, gated by API key.

### 8.3 Metrics — extend the existing `PipelineMetrics` singleton

Add to [metrics.py](src/graphbuilder/infrastructure/services/metrics.py):

```python
# counters
qa_requests_total{intent, status}
qa_tool_calls_total{tool, outcome}      # outcome ∈ ok|validation_error|denied|undone
qa_mutations_total{tool, operation}
qa_faithfulness_failures_total
qa_memory_overflow_drops_total{layer}

# histograms
qa_latency_seconds{phase}               # phase ∈ planner|retrieval|rerank|llm|verify|total
qa_retrieval_hits{channel}              # channel ∈ cypher|vector|bm25
qa_context_tokens{slot}                 # slot ∈ system|working|summary|episodic|sources
qa_llm_tokens{direction}                # direction ∈ prompt|completion

# gauges
qa_pending_confirmations
qa_active_sessions
```

Surfaced via the existing `GET /health/metrics` endpoint. No new infra needed.

### 8.4 Audit log — the compliance trail

We **extend** the existing curation audit log ([curation.py:34](api/routers/curation.py#L34)) rather than create a parallel store. Every chatbot mutation writes one row using the same shape human curators do:

```json
{
  "request_id": "…",
  "session_id": "…",
  "turn_id": "…",
  "actor": "chatbot",                 // distinguishes from "human"
  "actor_user_id": "user_42",
  "tool": "propose_relationship",
  "action": "approve_relationship",   // mapped onto existing CurationAction
  "target_id": "rel_…",
  "before": {…},
  "after":  {…},
  "reason": "user said: 'add a TARGETS edge from semaglutide to GLP-1R'",
  "confirmation_id": "…",
  "timestamp": "…"
}
```

This means existing tooling (the `/curation/audit` endpoint, the curation review queue UI) shows chatbot actions side-by-side with human ones — invaluable for trust and debugging. Read-only retrievals do **not** write audit rows; they're in logs+metrics only.

### 8.5 Developer debug pane (frontend)

When a `?debug=1` query param is set (or a feature flag is on for staff users), each chat turn renders a collapsible debug pane below the source list:

```
┌─ Debug · request_id 9f3a… ────────────────────────────┐
│ intent: relational  · entities: [imatinib]            │
│ retrieval                                              │
│   cypher: 3 hits · vector: 12 hits · bm25: 5 hits     │
│   rrf_top: 50 · rerank kept: 8                         │
│ memory                                                 │
│   working: 3 turns · summary: 412 tok                 │
│   episodic recall: turn t_88 (sim 0.78)               │
│ llm                                                    │
│   model gpt-4o · prompt 11.2k tok · completion 412 tok│
│   latency 3.4s · retries 0                             │
│ tools                                                  │
│   1× propose_relationship — pending confirmation       │
│ faithfulness: 4/4 claims passed                       │
│ [open trace in audit log →]                            │
└────────────────────────────────────────────────────────┘
```

Powered by the structured `RetrievalTrace` already in §4 + a new `TurnTrace` aggregator. No extra LLM calls — purely data already collected during the turn.

### 8.6 Sampling & retention

- **Logs**: full structured logs for all turns; rotate at the existing log dir defaults.
- **Metrics**: in-memory aggregates only (no external sink in v1); scrape via `/health/metrics`.
- **Audit**: persistent, never auto-deleted. Truncation only for the dev tail buffer ([curation.py:49](api/routers/curation.py#L49)) which is already in place.
- **Token-level LLM traces**: sampled at 5% by default (config `qa_trace_sample_rate`) to keep storage bounded; always-on for failed turns.

### 8.7 Settings

```python
@dataclass
class RAGObservabilityConfig:
    log_format: Literal["text", "json"] = "json"
    log_user_text: bool = False           # keep raw user text out of logs
    audit_chatbot_actor: str = "chatbot"
    trace_sample_rate: float = 0.05
    debug_param_name: str = "debug"
    metrics_namespace: str = "qa"
```

---

## 9. Evaluation — Precision / Recall / F1

### 7.1 Gold dataset (must build)

Curate **~100 question/expected-source triples** seeded from the real graph:

```yaml
- id: q001
  question: "What kinases does imatinib inhibit?"
  intent: relational
  gold_entity_ids: [ent_imatinib, ent_bcr_abl, ent_kit, ent_pdgfr]
  gold_relationship_ids: [rel_imatinib_bcr_abl, rel_imatinib_kit, rel_imatinib_pdgfr]
  gold_chunk_ids: [chunk_42, chunk_91]
  gold_answer_substring: "BCR-ABL"   # must appear in answer
```

Source the gold from existing curated reviews in the `/curation` queue + manual SME review. Store at `tests/eval/rag_gold.yaml`.

### 7.2 Metrics

For each question, compute:

| Metric | Definition | Target |
|---|---|---|
| **Retrieval P@k** | \| retrieved ∩ gold \| / k | ≥ 0.5 @ k=8 |
| **Retrieval R@k** | \| retrieved ∩ gold \| / \| gold \| | ≥ 0.7 @ k=8 |
| **Retrieval F1@k** | harmonic mean | ≥ 0.55 |
| **Context recall** | did we retrieve at least one chunk that contains the answer? | ≥ 0.9 |
| **Answer faithfulness** | fraction of cited claims that pass post-hoc verification | ≥ 0.85 |
| **Answer coverage** | gold_answer_substring appears in answer | ≥ 0.8 |
| **Latency p95** | end-to-end seconds | ≤ 6s |

### 7.3 Harness

`tests/eval/run_rag_eval.py`:
- Iterates the gold set.
- Calls the same `/qa` endpoint we ship to users (no shortcuts).
- Writes a CSV + a markdown report.
- Pinned thresholds in CI; PR fails if F1 drops > 5% absolute.

### 7.4 Ablations to run

- vector channel only vs. +Cypher vs. +BM25 vs. all three
- with/without cross-encoder rerank
- with/without chunk neighbour expansion
- with/without episodic memory recall

This is how we *prove* each piece earns its keep, not just claim it does.

### 9.5 First live results — 2026-05-10

23-question grounded gold set (`tests/eval/rag_gold_local.yaml`)
against the local Neo4j *post* the entity-dedup pass and the
chunks-as-items + ablation-overrides shipped with the eval. 0 errors.

| Config | P@k | R@k | F1@k | Ctx | Cov | p95 (ms) |
|---|---|---|---|---|---|---|
| **all_channels** (baseline) | 0.141 | 0.519 | 0.188 | 0.870 | 0.870 | 3512 |
| vector_only ⭐ | 0.147 | 0.531 | **0.197** | **0.913** | 0.870 | 2421 |
| bm25_only | 0.120 | 0.467 | 0.170 | 0.783 | 0.783 | 1786 |
| cypher_only | 0.125 | 0.489 | 0.168 | 0.783 | 0.826 | 2539 |
| no_rerank | 0.103 | 0.472 | 0.148 | 0.826 | 0.783 | 2868 |
| no_chunks | 0.141 | 0.519 | 0.188 | 0.870 | 0.870 | 3401 |

### 9.6 Channel-quality investigation + fix #1 — 2026-05-10

`scripts/investigate_channels.py` instruments per-channel hit IDs and
their gold overlap to localise where noise enters the top-8. Findings
on the same 23-question gold set (refusal questions skipped):

- **Cypher dominates final top-8 attribution at 35.6 %** but its raw
  hit-rate in gold is only 6.03 % — it's filling slots with
  related-but-not-gold material because it re-anchors on **BM25**
  hits ([channels.py:274](src/graphbuilder/core/retrieval/channels.py#L274)). Two channels compounding the same mistake.
- **Person / Document / Organization entity types** (~35 % of the
  graph; 1569 + 371 + 347 nodes) leak into BM25 hits as authors /
  paper titles / consortium names ("Levin B", "Friedenson B",
  "Scottish/Northern Irish BRCA1/BRCA2 Consortium").
- **Cypher relationship labels are bare UUIDs** like
  `5e75dd33-… --INFLUENCES--> c540d9a-…` — the cross-encoder rerank
  has no readable text to score against, so noise it can't demote
  ([channels.py:366](src/graphbuilder/core/retrieval/channels.py#L366)).
- **16 / 21 questions had at least one noise item** in the final
  top-8 from BM25/Cypher only, not in gold.

**Fix #1 — entity-type blocklist** (`RetrievalConfig.entity_type_blocklist`,
default `("Person", "Document", "Organization")`; per-request override
on `AskRequest.ablation`). Every channel now drops blocked types
pre-fusion; channels over-fetch 2× when a blocklist is set so RRF
input quality stays the same.

What it changed (re-run of `investigate_channels.py`):

| Channel | Hits (before → after) | In-gold rate (before → after) |
|---|---|---|
| BM25 | 369 → 311 (−16 %) | 5.15 % → 6.11 % |
| Cypher | 696 → 533 (−23 %) | 6.03 % → 7.88 % |
| vector_entity | 420 → 420 | 5.24 % → 5.48 % |

Author / paper / consortium noise is **gone** from the final top-8
on every question. Noise candidates dropped 16 → 14 questions.

What it didn't change: **F1 stayed at 0.193** (`with_authors` ablation,
which disables the blocklist, scores identically). The remaining
noise is BRCA1-related *Concepts* (legitimate Concept entities, not
filterable by type) and Cypher's bare-UUID relationship labels —
fixes #2 and #3 in the channel-quality plan target those. The
visible-but-not-yet-measurable improvement is bounded by the gold
set: when retrieval surfaces a *valid but not-pinned* BRCA1
neighbour, gold scores it as 0.

What the numbers say (and don't):

1. **Cross-encoder rerank earns its keep.** Disabling it drops F1
   ~21 % relative (0.188 → 0.148). Validates P4 with data.
2. **Vector-only narrowly beats all-channels** — Cypher and BM25 are
   bringing in noise that the rerank can't fully clean up. The
   Cypher-channel templates and `term_extraction.py` are the next
   thing to revisit.
3. **Chunks-as-items is no-op on F1 here** but only 9 of 23
   questions have `gold_chunk_ids`; chunk-hit measurement on those 9
   went from "structurally impossible" to "actually works".
4. **Precision ~0.14 is structural, not a bug.** Lookup questions
   have 1 gold id at top-k=8 → ceiling P=0.125 (4 of 23). Two
   refusal questions have empty gold → automatic P=0/R=0 (drag the
   macro by ~9 % absolute).
5. **p95 6.1 s is cold-process** (q001 + q002 each ~6.2 s warm-up).
   On a warm run p95 settles at ~3.5 s.

Reports: `tests/eval/_reports/v2/{rag_eval.md,rag_eval.csv}`
(gitignored — regenerate via the CLI in §9.3).

---

## 10. API design

### 10.1 New router: `api/routers/qa.py`

```python
POST /qa/ask
  body: { session_id?: str, query: str, stream?: bool, top_k?: int }
  resp: { answer: str, sources: RetrievedItem[],
          turn_id: str, session_id: str,
          per_claim_confidence: ClaimScore[],
          retrieval_trace: RetrievalTrace,
          pending_mutations: PendingMutation[],   // §7
          turn_trace: TurnTrace }                 // §8

POST /qa/ask/stream                # SSE — same shape, streamed token-by-token
GET  /qa/sessions/{id}             # full history
DEL  /qa/sessions/{id}
POST /qa/turns/{id}/feedback       { rating: -1|0|1, comment?: str }

# tool-use endpoints (§7)
POST /qa/mutations/{id}/apply      # user confirms a pending mutation
POST /qa/mutations/{id}/reject     { reason?: str }
POST /qa/mutations/{id}/undo       # within undo window

# observability (§8)
GET  /qa/turns/{id}/trace          # full TurnTrace for the debug pane
GET  /health/metrics               # existing — extended with qa_* metrics
GET  /curation/audit               # existing — now also returns chatbot rows
```

`RetrievalTrace` is a structured record of which channel returned what, RRF ranks, rerank scores. `TurnTrace` is the superset (retrieval + memory + LLM + tools + faithfulness) used by the frontend's "show your work" / debug pane.

### 10.2 Reuse, don't reinvent

- Auth: existing `Depends(require_api_key)`.
- SSE: copy the pattern from `/documents/jobs/{id}/stream`.
- Embedding: `embed_async()` from `embedding_factory`.
- LLM: `AdvancedLLMService.generate_text(...)` with retry already wired.
- Verification: `CascadingVerifier` for faithfulness check.

---

## 11. Frontend — `/chat` page

### 11.1 Layout

Mirror `/verification`'s look-and-feel for cohesion:

```
┌─ /chat ────────────────────────────────────────────────────┐
│ ┌─ session list ─┐ ┌─ chat thread ─────────────────────┐  │
│ │ • New chat     │ │ user: ...                          │  │
│ │ • Yesterday    │ │ bot:  ...                          │  │
│ │ • Older        │ │       ┌─ Sources [3] ▾ ─────────┐  │  │
│ │                │ │       │ entity bar  0.92  ▒▒▒▒▒ │  │  │
│ │                │ │       │ rel bar     0.81  ▒▒▒▒  │  │  │
│ │                │ │       │ chunk bar   0.75  ▒▒▒   │  │  │
│ │                │ │       └────────────────────────┘  │  │
│ │                │ │       ┌─ Retrieval trace ▾ ────┐  │  │
│ │                │ │       │ cypher: 3 hits         │  │  │
│ │                │ │       │ vector: 12 hits        │  │  │
│ │                │ │       │ bm25:   5 hits         │  │  │
│ │                │ │       │ rerank: kept top 8     │  │  │
│ │                │ │       └────────────────────────┘  │  │
│ │                │ │ [type your question…]    [send]    │  │
│ └────────────────┘ └────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

### 11.2 Components to add

- `frontend/app/chat/page.tsx` — page shell.
- `frontend/components/chat/MessageBubble.tsx` — bubble with citation chips `[1]` `[2]` linking to a source card.
- `frontend/components/chat/SourceCard.tsx` — confidence bar (reuse `verification` page's stage bar) + url + chunk preview on hover.
- `frontend/components/chat/RetrievalTrace.tsx` — collapsible "show your work" panel.
- `frontend/components/chat/SessionSidebar.tsx` — list past sessions, rename, delete.
- `frontend/components/chat/MutationCard.tsx` — pending-mutation card with diff preview, "Apply / Reject" buttons, and the focused subgraph preview after apply (§7.4). Reuses the existing graph component from the `/graph` page.
- `frontend/components/chat/DebugPane.tsx` — collapsible developer pane shown when `?debug=1` is set (§8.5).
- `frontend/lib/api.ts` — add `askQuestion()`, `streamAnswer()` (SSE EventSource), `applyMutation()`, `rejectMutation()`, `undoMutation()`, `getTurnTrace()`.

### 11.3 Streaming UX

Use SSE (existing pattern). Stream answer tokens first, then sources arrive at end-of-stream as a final event — so users see text fast, and sources resolve in place when ready.

---

## 12. Data model additions

### 12.1 New Neo4j node labels & edges

```cypher
// Sessions / turns
CREATE CONSTRAINT IF NOT EXISTS FOR (s:ConversationSession) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (t:ConversationTurn) REQUIRE t.id IS UNIQUE;
CREATE INDEX IF NOT EXISTS FOR (s:ConversationSession) ON (s.user_id);
CREATE INDEX IF NOT EXISTS FOR (t:ConversationTurn) ON (t.ts);

// Vector index for episodic recall
CREATE VECTOR INDEX turn_query_vector IF NOT EXISTS
FOR (t:ConversationTurn) ON (t.query_embedding)
OPTIONS { indexConfig: { `vector.dimensions`: 768, `vector.similarity_function`: 'cosine' } };
```

Edges:
- `(:ConversationSession)-[:HAS_TURN {idx}]->(:ConversationTurn)`
- `(:ConversationTurn)-[:CITES]->(:Entity | :DocumentChunk)`
- `(:ConversationTurn)-[:CITES_REL]->(:Relationship)` (or store rel id as property)
- `(:User)-[:OWNS_SESSION]->(:ConversationSession)` (User node optional v1; can use user_id property)

### 12.2 Pending mutation store (§7)

Pending (unconfirmed) mutations are short-lived state — keep them in-memory + Redis-style TTL semantics inside a singleton, or persist as `:PendingMutation` nodes if we want survivability across API restarts. v1 picks **in-memory with a process-wide dict + asyncio TTL cleanup** (simpler, fine for single-instance deploy); v2 promotes to Neo4j when we scale horizontally.

### 12.3 Audit log

Already exists ([curation.py:34](api/routers/curation.py#L34)). We add an `actor` column (`"chatbot"` vs `"human"`) and a `request_id` column for correlation. No schema migration needed — the audit store is JSON rows.

### 12.4 Repository

Add `ConversationRepositoryInterface` parallel to `GraphRepositoryInterface` and `DocumentRepositoryInterface`:

```python
class ConversationRepositoryInterface(ABC):
    async def create_session(self, user_id: str | None) -> ConversationSession
    async def append_turn(self, session_id: str, turn: ConversationTurn) -> None
    async def get_session(self, session_id: str) -> ConversationSession
    async def list_sessions(self, user_id: str | None, limit: int) -> list[ConversationSession]
    async def vector_search_turns(self, session_id: str, q_vec: list[float],
                                  top_k: int, min_score: float) -> list[tuple[ConversationTurn, float]]
    async def update_session_summary(self, session_id: str, summary: str) -> None
    async def record_feedback(self, turn_id: str, rating: int, comment: str | None) -> None
```

Implementations: `Neo4jConversationRepository` (prod) + `InMemoryConversationRepository` (tests).

---

## 13. Implementation phases

| Phase | Deliverable | Status | Commit |
|---|---|---|---|
| **P0** | This plan | ✅ shipped | `ad4589d` |
| **P1** | `ConversationRepository` + Neo4j schema migration | ✅ shipped | `ad4589d` |
| **P2** | Observability skeleton — `request_id` propagation, `qa.*` loggers, extended `PipelineMetrics`, audit-log `actor` column | ✅ shipped | `ad4589d` |
| **P3** | Retrieval orchestrator: Cypher + vector + BM25 channels, RRF fusion, chunk hydration | ✅ shipped | `43408b4` |
| **P4** | Cross-encoder rerank + chunk neighbour expansion (`NEXT_CHUNK ±1`) | ✅ shipped | `c958dd1` |
| **P5** | `/qa/ask` endpoint (non-streaming), `RetrievedItem` + `RetrievalTrace` shapes | ✅ shipped | `43408b4` |
| **P6** | Working memory + rolling summary | ✅ shipped | `1f0e989` |
| **P7** | Episodic recall via `turn_query_vector` index | ✅ shipped | `1f0e989` |
| **P12** | Frontend `/chat` page with per-source confidence + retrieval trace + sidebar | ✅ shipped | `69524d6` |
| **+ Identity** | Lightweight browser identity (`X-User-Id`, `/users` router, ownership rules) — out-of-plan addition resolving §14.1 | ✅ shipped | `331cdf4` |
| **P13** | Eval harness + gold set + CI gate (`src/graphbuilder/core/eval/`, `tests/eval/`, hermetic smoke + live CLI) | ✅ shipped | `644633c` |
| **+ Re-parse guard** | Stage-0 `source_url` check skips already-ingested documents; `force=True` re-ingests. Stops the duplicate-entity blowups the eval surfaced (BRCA1 Concept + Brca1 Gene from re-parses). | ✅ shipped | `2de92f6` |
| **+ Chunk promotion** | Hydrated chunks emitted as `RetrievedItem(kind=chunk)` so `gold_chunk_ids` can match and the frontend source list exposes chunk rows. | ✅ shipped | `5808540` |
| **+ Ablation overrides** | Per-request `RetrievalConfig` override on `/qa/ask`; eval CLI `--ablations` matrix. | ✅ shipped | `5808540` |
| **+ Cross-type entity dedup** | One-shot CLI (`scripts/dedup_entities.py`) that collapses same-name entities across types via direction-preserving MERGE. Local graph went 7504 → 6610 entities, 4222 rels preserved exactly. | ✅ shipped | `520d3d6` |
| **P8** | Faithfulness check (`CascadingVerifier` on extracted claims) — harness slot reserved (`answer_faithfulness` returns null until wired) | pending | — |
| **P11** | SSE streaming endpoint | pending | — |
| **P9** | Tool-use surface (read-only) — `search_graph`, `get_entity`, `verify_claim` exposed to LLM | pending; gated on §14.6 | — |
| **P10** | Tool-use surface (mutating) — propose/update/merge/soft-delete with PendingMutation + confirmation flow + audit rows | pending; gated on §14.6 | — |
| **P14** | Cross-session semantic memory + user persona summary | pending | — |
| **+ MutationCard / DebugPane in `/chat`** | UI for §7 + §8 (was bundled into P12 originally; now blocked on P10) | deferred | — |

Each phase is independently shippable. P2 (observability) was deliberately first so every later phase emits structured traces from day one. P13 (eval) is recommended next so the remaining quality phases (P4 ablations, P6/P7 tuning, P8) have numbers to chase rather than vibes. P9 ships before P10 so the LLM gets used to the tool schema with zero-risk reads before we open up writes.

### Implementation refinements (vs. the original plan)

A few small details drifted from the §3–§7 sketches during build; recording them here so the plan stays a single source of truth.

- **Rolling-summary freshness signal (§5.2).** Instead of adding a `summarised_through_idx` column to `ConversationSession`, the cached summary itself starts with a `[summary covers N turns]` marker line. Comparing that count against the live "older turns" count is enough to detect staleness — no schema migration needed. Regeneration runs on every `/qa/ask` whose older-turn count has changed since the cache was written; otherwise we reuse.
- **Embedding shared across retrieval + memory (§3.1, §5.3).** `QAService.ask()` embeds the query once at the top, then runs retrieval and memory build in parallel via `asyncio.gather`. The orchestrator's `retrieve()` accepts an optional `query_embedding` kwarg so callers with a pre-computed vector skip internal embedding.
- **Cross-encoder degradation (§3.3).** When the model fails to load (no internet on first run, missing dep, etc.) the reranker passes through the input order and `score_rerank` stays `None`. The pipeline never crashes because of rerank — it just degrades to RRF order.
- **Final-confidence math (§4).** With rerank: `0.7·score_rerank + 0.3·channel_max + multi_channel_bonus`. Without rerank: `channel_max + bonus`. Citation-coverage (§6.3, weight 0.2) layers on later in the QA service after the LLM has cited.
- **Logger-filter placement (P2).** `RequestIdFilter` is attached to each *handler* rather than to the loggers. Python only runs logger-level filters for records emitted directly on that logger, so a filter on root would have skipped child-logger propagation and crashed the `[%(request_id)s]` format string.
- **Test ergonomics for P4.** A module-level autouse fixture in `tests/unit/test_retrieval.py` no-ops `CrossEncoderReranker.rerank` so the 28 channel/RRF/orchestrator tests stay hermetic. Tests that exercise the rerank path inject a fake encoder via the module-level `_MODEL_CACHE`.
- **Eval harness shape (P13, §9).** The harness library lives at `src/graphbuilder/core/eval/` (gold loader, metric math, async runner, CSV/markdown writers) with a transport-agnostic `ask_fn(query) -> AskLike` callback so the same code drives the hermetic CI gate, ablations, and the live API. The CI gate is `tests/eval/test_eval_smoke.py`: it builds a real `RetrievalOrchestrator + QAService` against an in-memory mini-graph and asserts the run clears every floor in `tests/eval/baselines.json::hermetic_floor`. The live counterpart is `tests/eval/run_rag_eval.py`, which posts `/qa/ask` and gates on `live_targets`. The `answer_faithfulness` slot is reserved in `EvalSummary` but reports `null` until P8 lands — pre-wired so the gate just needs threshold edits, not new code.
- **Eval-driven follow-ups (post-P13).** The first live run surfaced four data/architecture issues. All shipped before §9.5's numbers were measured:
  - **Re-parse guard (`2de92f6`).** Stage-0 `source_url` lookup in `DocumentExtractionPipeline.run` short-circuits when a `:Document` with that URL already has chunks. Set `force=True` on `DocumentInput` / `ProcessDocumentRequest` to override. Stops the duplicate-entity blowup that re-ingesting the same paper used to create.
  - **Chunks-as-items (`5808540`).** Hydrated chunks are emitted as `RetrievedItem(kind=chunk)` appended after entity / rel items, with confidence inherited (minus 0.05) so confidence-sorted UIs stay stable. Gold-chunk matching now actually works; toggle via `RetrievalConfig.emit_chunk_items`.
  - **Per-request ablation overrides (`5808540`).** `RetrievalOrchestrator.retrieve(config_override=…)` accepts a per-call `RetrievalConfig`; `AskRequest.ablation` (Pydantic) plumbs it through `/qa/ask`. The eval CLI's `--ablations vector_only,…` runs the matrix without rebuilding the singleton. Disabled channels are skipped entirely so latency ablations stay honest.
  - **Cross-type entity dedup (`520d3d6`).** Ingestion-time dedup tiers in `save_entities_batch` only match within one `entity_type` (BRCA1 Concept and Brca1 Gene escaped). `scripts/dedup_entities.py` plans + applies same-name-across-types merges using a direction-preserving Cypher MERGE (the existing `merge_entities` always flipped incoming edges to outgoing; the script's query splits incoming/outgoing branches and unions `source_chunk_ids` + `source_document_ids`). Local sweep collapsed 7504 → 6610 entities, 4222 rels preserved exactly.

---

## 14. Open questions for the user

1. ~~**User identity**~~ — **resolved 2026-05-09**. Lightweight browser identity, chat-only.

   First visit → frontend prompts for a display name → `POST /users` mints a `user_id` and a `:User` Neo4j node → both stored in `localStorage`. Every `/qa/*` request carries the id in `X-User-Id`. Server validates the header against the user repo on every call; missing → falls through to anonymous (back-compat); present-but-unknown → 401 with "clear localStorage and re-register". `ConversationSession.user_id` is now a real foreign key for new sessions; ownership is enforced on `GET`/`DELETE /qa/sessions/{id}` (mismatch returns 404, not 403, so we don't leak existence). Anonymous sessions stay readable to anyone, matching pre-identity behaviour. Other surfaces (`/graph`, `/curation`, etc.) remain shared. Real auth is a follow-up.
2. **LLM cost ceiling** — OK with gpt-4o for answer generation, or prefer gpt-4o-mini? Affects faithfulness defaults.
3. **Cross-encoder model** — fine to start with `cross-encoder/ms-marco-MiniLM-L-6-v2` (general) and swap to a biomedical cross-encoder later, or want biomedical from day one?
4. ~~**Conversation persistence**~~ — **resolved (implicit, 2026-05-09)**. Shipped on Neo4j per the plan: `:ConversationSession` and `:ConversationTurn` nodes with a `turn_query_vector` cosine index for episodic recall. Keeps everything in one store; no Postgres/Redis dependency added. Re-evaluate when we need horizontal scaling or the audit log grows past disk budget.
5. **Gold set sourcing** — happy to pull seed questions from the existing `curation` review queue, or have a domain SME author from scratch?
6. **Mutation authority** — should *every* user be allowed to confirm graph mutations, or gate `propose_*` / `merge_entities` / `soft_delete_*` behind a curator role? (Affects auth on `/qa/mutations/*/apply`.)
7. **Hard delete** — confirm we ship soft-delete only in v1 (my recommendation), or is there a workflow that needs irreversible removal?
8. **Audit retention** — chatbot audit rows persist forever same as human curator rows, or do we want a retention policy (e.g. 1y) for chat-originated rows?

---

## 15. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Hallucination across long sessions | Faithfulness check + episodic recall + strict "answer only from sources" prompt |
| Vector recall misses exact gene/drug IDs | BM25 channel runs in parallel; not relying on vectors alone |
| Cypher templates become a maintenance burden | Keep them small (≤ 10 templates); intent classifier routes to most-specific template; fallback to vector channel always available |
| Token-budget overflows on long context | Hard cap + priority drop order in §5.5; tested explicitly |
| Cross-encoder latency on CPU | Cap to top 50 candidates; can move to GPU pool (already have it for embeddings) |
| Gold set bias | Ablation studies in §9.4 + periodic SME review; mark eval set as living, not frozen |
| LLM-driven graph corruption | Constrained tool surface (no raw Cypher), mandatory user confirmation for mutations, soft-delete-only, undo window, full audit trail (§7, §8) |
| Prompt injection causing unwanted mutations | Confirmation rule is server-enforced (not just UI); injected text in retrieved chunks cannot bypass it; audit log records `confirmation_id` so injection attempts are visible after the fact |
| Audit log floods from chatty users | Read-only retrievals never write audit rows (§8.4); only confirmed mutations do |
| Loss of pending mutation on API restart | v1 in-memory store is acceptable for single-instance deploy; v2 promotes to `:PendingMutation` Neo4j nodes (§12.2) |
| Sensitive content in logs | `log_user_text=False` default, fixed-length truncation, full text only in audit store gated by API key (§8.2) |
