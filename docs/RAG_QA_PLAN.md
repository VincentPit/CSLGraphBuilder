# RAG Q&A System — Design Plan

Branch: `feature/chatbot`
Plan drafted: 2026-05-08
Last updated: 2026-05-10

## Status

**Shipped (19 commits on `feature/chatbot`, 465 unit + eval tests passing):**
P0 plan · P1 conversation persistence · P2 observability spine · P3 retrieval orchestrator (Cypher + vector + BM25 + RRF + chunk hydration) · P4 cross-encoder rerank + chunk neighbour expansion · P5 minimal `/qa/ask` endpoint · P6 working memory + rolling summary · P7 episodic recall · **P8 answer-faithfulness check** (per-claim lexical scoring with optional LLM escalation; refusal short-circuit; `answer_faithfulness` now flowing through `/qa/ask` + the eval harness + the hermetic gate) · **P9 read-only tool-use** (`search_graph` / `get_entity` / `verify_claim` exposed via OpenAI function-calling; agentic loop in `QAService` capped at `max_tool_calls_per_turn`; opt-in per request via `enable_tools=true`; `tool_calls` recorded on `AskResponse`) · **P10 mutating tool-use** (`propose_entity` / `propose_relationship` / `update_entity` / `merge_entities` / `soft_delete_entity` / `soft_delete_relationship` queue into a process-scoped proposal store; `MutationApplier` runs the actual graph write only after a curator approves via `POST /qa/proposals/{id}/apply`; opt-in per request via `enable_mutations=true`; per §14.6) · **P11 SSE streaming** (`POST /qa/ask/stream` emits `phase` → `retrieval` → `delta` → `done` events; LLM uses `stream=True` when available; graceful fallback to one-shot delta) · P12 `/chat` frontend with per-source confidence + retrieval trace · P13 eval harness (gold loader + metrics + ablation runner + CSV/markdown reports + hermetic CI gate + live `run_rag_eval.py` CLI) · **§9.9 intent-aware retrieval routing** (rule-based classifier + per-intent profiles + override-threading regression test; relational recall 24% → 36%) · plus an unplanned **lightweight browser identity** (X-User-Id, chat-only, see §14.1) and four eval-driven follow-ups: **re-parse guard**, **chunks-as-first-class-items**, **per-request ablation overrides**, **cross-type entity dedup CLI** (see §13).

**Open / next:**
P14 cross-session semantic memory.

**First live eval results (23-question grounded gold set, 2026-05-10):** baseline P=0.141, R=0.519, F1=0.188, ctx=0.870, cov=0.870, p95=3.5 s warm. Cross-encoder rerank earns its keep (-21 % F1 without it). Vector-only narrowly beats all-channels on F1 → BM25 + Cypher are bringing in noise the rerank can't fully clean up; templates + term extraction are the next thing to look at. See §9 and `tests/eval/_reports/v2/rag_eval.md`.

**Post-routing live eval (same gold set, 2026-05-10):** P=0.136, R=**0.582**, F1=**0.205**, ctx=**0.913**, cov=**0.957**, p95=**5.4 s**. Recall +6 pp absolute / +11.5 % relative — the intent-routing change in §9.9 shipped the predicted gain (relational `final_top_k=16` + Cypher boost + vec_rel cap on lookup). Reports: `tests/eval/_reports/intent_routed_v1/`.

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

### 7.3 Confirmation policy — queue into `/curation`

Per the §14.6 resolution (2026-05-10), every **mutating** tool call lands in the existing curation review queue rather than going straight to the database. The chatbot is a *proposal generator*; only a human curator can promote a proposal to a real mutation.

```
LLM emits mutating tool_call
  → server validates Pydantic schema
  → CurationUseCase.create_review(
        proposer="chatbot",
        proposer_user_id=<X-User-Id>,
        target_kind="entity"|"relationship",
        target_id=<existing id or "new">,
        proposed_change={tool, args, diff},
        request_id=<correlation id>,
    )
  → returns review_id
  → chat bubble renders "Proposed for curator review (#<review_id>)"
  → curator opens /curation, sees the row with a "from chatbot" badge,
    Accept → runs the same save_*/merge_*/etc. paths human proposals do
    Reject → records reason, no mutation
```

Read-only tools (`search_graph`, `get_entity`, `verify_claim`) bypass the queue and execute inline — they never write to the graph.

Two things this loses vs. the original "two-key" sketch:

- **Real-time mutation preview in chat.** The subgraph preview from §7.4 is rendered post-acceptance by the existing `/curation` UI, not inline in the chat bubble. The chat bubble shows the proposed diff only.
- **In-chat undo.** Undo is the curator's `Reject` button or a follow-up review; chat doesn't expose a `/qa/mutations/{id}/undo` endpoint in v1.

Both are recoverable once real auth + roles land — the tool surface itself doesn't have to change, only the executor's destination.

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

### 9.7 Fix #3 — relationship label resolution

The investigation also flagged Cypher (and to a lesser extent the
relationship vector channel) emitting hits with bare-UUID labels like
`5e75dd33-… --INFLUENCES--> c540d9a-…` because
[`_relationship_label`](src/graphbuilder/core/retrieval/channels.py#L366-L370)
uses entity ids, not names. The cross-encoder rerank has nothing
readable to score against, so it can't tell the difference between a
genuinely-relevant relationship and a noise one.

**Fix:** the orchestrator does a single batch ``id -> name`` lookup
across every relationship hit's source/target ids
([`get_entity_names_by_ids`](src/graphbuilder/infrastructure/repositories/graph_repository.py#L120))
*before* fusion, then [`_build_item`](src/graphbuilder/core/retrieval/orchestrator.py#L276-L300)
rewrites relationship labels as `"<src_name> --REL--> <tgt_name>"`.
Falls back to the channel's UUID label when neither end resolves
(an entity was deleted between channel emission and resolution) so
nothing crashes the turn — labels are quality, not correctness.

**v3 (fix #1 only) → v4 (fix #1 + #3) deltas, same 23-question gold:**

| Metric | v3 | v4 | Δ |
|---|---|---|---|
| F1 @ k | 0.193 | **0.207** | +0.014 (+7 %) |
| P @ k | 0.147 | 0.158 | +0.011 |
| R @ k | 0.524 | 0.541 | +0.017 |
| Context recall | 0.870 | **0.913** | +0.043 |
| Answer coverage | 0.870 | **0.957** | +0.087 |

The biggest jump is **answer coverage 87 % → 96 %** — the LLM is
producing answers that mention the gold substring on 22 / 23
questions instead of 20 / 23. Mechanism: with readable relationship
labels in the prompt's SOURCES block, the LLM grounds its answer
against actual src/tgt names instead of UUIDs and can faithfully
quote them back. Re-rank quality also improves because the
cross-encoder now has real text to score (no measurable rerank-
latency hit; the batch repo call is one round-trip).

Reports: `tests/eval/_reports/v4/{rag_eval.md,rag_eval.csv}`.

### 9.8 Fix #2 — Cypher anchors on vector, not BM25

The investigation showed the Cypher channel re-uses **BM25 hits** as
its anchor entities ([`channels.py`](src/graphbuilder/core/retrieval/channels.py#L322))
and then expands their 1-hop neighbourhoods. Any noise BM25 surfaced
(e.g. Concept entities containing the gene symbol as a substring,
``"BRCA1-associated genome surveillance complex"``) seeded the Cypher
expansion too — two channels compounding the same mistake.

**Fix:** when an embedding is available, Cypher anchors on
``vector_search_entities`` (semantic), not ``search_entities_by_text``
(substring). Falls back to BM25 when the embedding model failed to
load — better degraded anchors than no anchors.

What it changed (`scripts/investigate_channels.py`, before → after):

| | After #1 | After #1 + #3 + #2 |
|---|---|---|
| Cypher hits | 533 | 709 (+33 %) |
| Cypher in-gold hits | 42 | **53** (+26 %) |
| Cypher top-8 attribution | 113 | **88** (−22 %) |
| Questions with noise | 14 / 21 | **12 / 21** |

Cypher now finds **more** gold relationships (semantic spans more
entities than substring) but contributes **less** to the final top-8
because the rerank — now seeing readable labels (#3) — correctly
demotes its outputs.

**Headline F1 unchanged** (0.207) because the cross-encoder was
already masking the BM25-anchor noise. The visible lift is when
rerank is *off*:

| Config | v4 (#1+#3) | v5 (#1+#3+#2) | Δ |
|---|---|---|---|
| `all_channels` F1 | 0.207 | 0.207 | 0.000 |
| `no_rerank` F1 | 0.159 | **0.190** | +0.031 (+19 %) |

Fix #2 is a robustness improvement: when the cross-encoder model
fails to load (cold start, missing dep, no internet), the system now
degrades much more gracefully because the candidate pool is
intrinsically cleaner. Production-warm metrics don't move.

Reports: `tests/eval/_reports/v5/{rag_eval.md,rag_eval.csv}`.

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

### 9.9 Intent-aware retrieval routing — 2026-05-10

The post-fix-#2 investigation report (now archived as
`tests/eval/_reports/channels/channel_investigation_baseline.md`)
exposed three findings that pointed at one root cause: **the
orchestrator runs the same channel mix for every query**, regardless
of intent. Per the report's per-intent breakdown:

* **Lookup**: vec_rel found 0/4 gold but took 28% of top-K seats.
* **Definitional**: vec_rel found 0/9 gold here too.
* **Relational**: gold averages 8.4 items/q but `final_top_k=8` was
  clipping. Cypher found 5/8.4 gold/q — the strongest channel for
  this intent — but only 3.25 landed in top-K because vec_rel crowded
  it out.

**Fix:** classify each query into `lookup` / `definitional` /
`relational` and apply a per-intent profile over the base
`RetrievalConfig`. Three new files:

* `src/graphbuilder/core/retrieval/intent.py` — rule-based classifier
  (verb-form regex over biomedical action verbs, decision order
  relational > short-bare-term > definitional default), `IntentProfile`
  dataclass, `INTENT_PROFILES` mapping, and `apply_profile()`.
* `RetrievalConfig.enable_vector_relationship` + `vector_relationship_top_k`
  — split sub-toggle for the rel index so profiles can disable it
  without losing entity hits.
* `QAService.ask` — classifies, applies profile, passes via the
  existing `config_override` hook. Explicit `retrieval_override` still
  wins so the eval ablation harness stays honest. The chosen intent is
  stamped on `RetrievalTrace.intent` and the `qa_request` metric label.

The classifier is rule-based on purpose: deterministic, latency-free,
trivially debuggable. 100% match against the 21 in-domain gold-set
labels (validated as a unit test in `tests/unit/test_intent.py`).

**Subtle bug caught by re-running the investigation script:** the
channel objects (`VectorChannel`, `Bm25Channel`, `CypherChannel`) were
reading their construction-time `self._cfg`, so `config_override`
reached only the orchestrator-level toggles
(`enable_*_channel`, `final_top_k`, `rrf_*`, `enable_cross_encoder`,
`hydrate_chunks`, `chunk_neighbour_radius`). Channel-level knobs
(`vector_top_k`, `enable_vector_relationship`,
`vector_relationship_top_k`, `bm25_limit`, `cypher_top_k`,
`entity_type_blocklist`) silently ignored overrides — the first run
showed `vector_relationship: 20` hits on lookup queries despite the
profile setting `enable_vector_relationship=False`. Fix: thread `cfg`
through each channel's `run()` method, falling back to `self._cfg` when
no override is passed. Pinned by
`test_orchestrator_threads_config_override_to_vector_channel`.

#### Profile values (cited by the data)

| Knob | Default | Lookup | Definitional | Relational |
|---|---|---|---|---|
| `final_top_k` | 8 | 8 | 8 | **16** (gold avg 8.4 was clipping) |
| `vector_top_k` | 20 | 10 | 20 | 15 |
| `enable_vector_relationship` | True | **False** (0/4 gold) | True | True |
| `vector_relationship_top_k` | None→20 | n/a (off) | 10 | 10 (caps 28%-share noise) |
| `bm25_limit` | 20 | 10 | 20 | 15 |
| `cypher_top_k` | 10 | 5 | 8 | **20** (Cypher carries it: 5/8.4 gold) |

#### Channel-investigation comparison (post-fixes-#1+#2+#3)

| Metric | Baseline (no routing) | Intent-routed | Δ |
|---|---|---|---|
| Lookup recall | 100% (1.0/1.0) | 100% (1.0/1.0) | unchanged |
| Definitional recall | 53% (1.00/1.89) | 53% (1.00/1.89) | unchanged |
| **Relational recall** | **24% (2.00/8.38)** | **36% (3.00/8.38)** | **+12 pp (~+50% rel.)** |
| vec_ent hit-rate | 5.48% | 6.47% | +1.0 pp |
| vec_rel hit-rate | 3.10% | 4.71% | +1.6 pp |
| bm25 hit-rate | 6.11% | 7.54% | +1.4 pp |
| cypher hit-rate | 7.48% | 6.63% | −0.9 pp |

Reports:
`tests/eval/_reports/channels/channel_investigation.md` (post-routing,
intent-routed) vs.
`tests/eval/_reports/channels/channel_investigation_baseline.md`
(snapshot of the pre-routing report under the same path).

What the numbers say (and don't):

1. **Relational recall lifted 50% relative.** The biggest single jump
   on this branch — from `final_top_k=16` (so the gold avg of 8.4
   stops being clipped) plus `cypher_top_k=20` (so more of Cypher's
   gold lands in RRF). Per-question recall improved on 6/8 relational
   questions (q004, q006, q008, q010, q012, q016).
2. **Lookup unchanged at 100%** — the latency win (one fewer round-trip
   to the rel index) is the bonus, not the recall.
3. **Per-channel hit-rates rose for vec_ent / vec_rel / bm25** because
   the profiles cap their pool sizes — smaller pools mean what stays
   is denser with gold. cypher's hit-rate is slightly down because its
   pool grew (gold density per-hit fell, but absolute gold per-q rose).
4. **Noise count went up** — partly arithmetic for relational
   (`final_top_k` 8→16 means more seats so more non-gold rows fit),
   partly real for lookup (smaller candidate pool means the rerank
   discriminates over a less informative pool, so noise can float up
   into seats 4–8). With recall preserved at the top, this matters
   more for the source list density than for answer correctness.
5. **Headline RAG eval (live `/qa/ask`, same 23-q gold set):**

   | Metric | v5 (no routing) | intent-routed | Δ |
   |---|---|---|---|
   | Precision @ k | 0.147 | 0.136 | −0.011 |
   | **Recall @ k** | **0.522** | **0.582** | **+0.060 (+11.5%)** |
   | F1 @ k | 0.194 | **0.205** | +0.011 (+5.7%) |
   | Context recall | 0.826 | **0.913** | +0.087 |
   | Answer coverage | 0.870 | **0.957** | +0.087 |
   | Latency p95 | 6254 ms | **5415 ms** | −839 ms |

   The recall lift carries through end-to-end: more gold context lands
   in the LLM's prompt, answer-coverage rises with it. Precision dips
   slightly (relational `final_top_k=16` adds non-gold rows). Latency
   improves on lookup (vec_rel disabled = one fewer round-trip).
   Reports: `tests/eval/_reports/intent_routed_v1/{rag_eval.md,rag_eval.csv}`.

#### What changed structurally

* Per-intent retrieval is now a first-class concept of the QA service,
  reachable by the eval harness (`use_intent_routing` flag in
  `scripts/investigate_channels.py`), and observable via
  `RetrievalTrace.intent` + the `qa_request` metric label.
* The `config_override` hook (originally for P13 ablations) now also
  threads through to channel-level knobs, so any future per-request
  config swap (e.g. agent / tool-use loop in P9) inherits this.
* `IntentProfile` is deliberately narrow — only the knobs that actually
  vary by intent. Adding a knob without a backing profile change is
  called out in the docstring as a smell.

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
| **P8** | Faithfulness check — per-claim lexical scoring (optional LLM escalation) on cited sources; `answer_faithfulness` now reported by `QAService` + `EvalSummary` + the hermetic CI gate; refusals short-circuit to 1.0 | ✅ shipped | — |
| **P11** | SSE streaming endpoint — `POST /qa/ask/stream` emits `phase` → `retrieval` → `delta` → `done` events; LLM uses `stream=True` when available; same retrieval/memory/faithfulness wiring as `/qa/ask`; graceful fallback to a single delta when the provider lacks streaming | ✅ shipped | — |
| **P9** | Tool-use surface (read-only) — `search_graph` / `get_entity` / `verify_claim` exposed via OpenAI function-calling; `ToolDispatcher` validates Pydantic schemas + dispatches to existing services; agentic loop in `QAService` (cap = `max_tool_calls_per_turn=5`); opt-in per request via `AskRequest.enable_tools` | ✅ shipped | — |
| **P10** | Tool-use surface (mutating) — six tools (`propose_entity` / `propose_relationship` / `update_entity` / `merge_entities` / `soft_delete_entity` / `soft_delete_relationship`); `MutationToolDispatcher` validates Pydantic schemas + enqueues into `api/proposed_mutation_store`; `MutationApplier` runs the actual `graph_repo.save_*` paths only after a curator approves via `POST /qa/proposals/{id}/apply`; opt-in per request via `AskRequest.enable_mutations` (separate from `enable_tools`) | ✅ shipped | — |
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
- **Mutating tools are a separate dispatcher, not a flag on the read one (P10).** §7's prose treated tool-use as one surface with `enable_mutations` toggling write paths inside the same dispatcher. In code that conflates two distinct concerns: the read dispatcher's executors hit the orchestrator + graph repo + cascading verifier inline, while the write dispatcher *only* validates and enqueues — it never touches the graph. So the implementation splits them: `core/retrieval/tools.py` (read) ships standalone in P9 and stays unchanged; `core/retrieval/mutation_tools.py` (write) is a sibling module, built around an `enqueue_fn` callback so `core/` doesn't import from `api/`. The actual `graph_repo.save_*` paths live in a third module, `core/retrieval/mutation_applier.py`, called by the curator-approval endpoint — keeping the chat code path completely free of repo-write imports. Three flags rule the call: `enable_tools` (read), `enable_mutations` (write), and the agentic loop runs whenever either is on. A model that calls a mutating tool with `enable_mutations=False` gets an error record back instead of silently being ignored, so prompt-injection attempts are visible in the trace.
- **Tool-use is opt-in per request, not a service-wide config (P9).** The plan's §7.7 sketched a `RAGToolConfig.enable_mutations` flag at construction time. In practice the read-only tools are always available to wire up but should fire only when the caller asks — production traffic answers most questions from the upfront retrieval alone, and the agentic loop adds latency + LLM cost. So the toggle moved from `RAGToolConfig` to `AskRequest.enable_tools` (Pydantic, default `False`); the singleton `ToolDispatcher` is constructed once in the router factory and the `QAService` only reaches for it when the request opts in. This keeps the surface zero-cost when off and makes A/B tests trivial — flip the flag, eval same-gold-set, compare. The cap (`max_tool_calls_per_turn=5`) stays on the service. The agentic loop also degrades silently when the LLM service lacks `generate_with_tools` (e.g. legacy fakes in tests, or a provider without function-calling support) by falling through to the single-shot generate path; previous tool calls in the same turn stay on the trace.
- **Streaming event order differs from §11.3 sketch (P11).** The plan's §11.3 said "stream answer tokens first, then sources arrive at end-of-stream as a final event." In code that's not actually possible — the LLM needs the retrieved sources in its prompt before it can emit the first token. So the shipped order is: `phase("retrieving")` → `retrieval` (sources + retrieval_trace + memory_trace, same shape as `/qa/ask` minus the answer/faithfulness) → `phase("generating")` → `delta` × N → `done` (turn_id, session_id, cited_source_indices, faithfulness, latency). The frontend can render source cards as soon as the `retrieval` event arrives — same UX win as the sketch, but the streamed-first thing is the *retrieval result*, not the tokens. Errors (session not found, retrieval failure, mid-stream LLM failure) emit a single `error` event instead of `done`, with a `kind` field (`session_not_found`, `retrieval_failed`, `llm_failed`, `internal_error`) so the client can branch. Mid-stream LLM failures preserve the partial deltas the client already received — the retry helper in `_retryable_llm_call` only wraps stream *open*, not drain, because retrying mid-stream would jumble token order on the client side.
- **Faithfulness — focused checker, not cascade reuse (P8).** §6.2 sketched reusing `CascadingVerifier` with a synthesised `(claim_subject) --MENTIONS--> (claim_object)` relationship. In code the cascade's stages all take a real `GraphRelationship` with source/target ids and look at entity-name overlap; a claim sentence rarely has a clean (subject, object), and the useful signal is "do the cited chunks contain the salient terms of the claim". So `src/graphbuilder/core/retrieval/faithfulness.py` ships a focused `FaithfulnessChecker` that keeps the same *shape* (lexical → optional LLM, explicit escalation thresholds) but with a comparator dedicated to the claim ↔ chunk match. Default is lexical-only (deterministic, latency-free); `enable_llm_escalation=True` opts into stage-3 verdicts on borderline claims. Refusals (the system-prompt phrase "I cannot find this in the knowledge base") short-circuit to 1.0 — declining to answer is the faithful response. Uncited tail prose stays in the trace for the debug pane but is excluded from the headline score so the model is penalised for *wrong* citations, not for missing optional ones. Failures from the checker itself never break the response — `AskResult.faithfulness` drops to `None` and a warning is logged.
- **Smoke-gate bypasses routing (P8 follow-up).** The §9.9 relational profile bumps `final_top_k=16`, which on the hermetic 4-entity graph drops precision below the 0.40 floor without changing the underlying fusion/hydration/rerank wiring the smoke is meant to gate. Routing has its own unit tests (`test_intent.py`, `test_qa_service.py::test_ask_*intent*`); the smoke gate now passes `retrieval_override=svc._cfg` so it stays focused on the channel pipeline + faithfulness wiring. Hermetic floor for `answer_faithfulness` set to 0.50 — the scripted-LLM answer cites entity tokens that match the cited chunks at ≥0.66 lexical overlap, leaving headroom.
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
6. ~~**Mutation authority**~~ — **resolved 2026-05-10**. Hybrid: chatbot mutations queue into the existing `/curation` review queue, never auto-apply.

   The chatbot's tool surface still exposes `propose_entity` / `propose_relationship` / `update_entity` / `merge_entities` / `soft_delete_*` as documented in §7.2, but a mutating call no longer follows the in-memory `PendingMutation` → `POST /qa/mutations/{id}/apply` flow sketched in §7.3. Instead the executor records a `CurationReview` row through the same `CurationUseCase` ([curation.py:36](src/graphbuilder/application/use_cases/curation.py#L36)) that human-proposed edits already use, and the chat bubble surfaces a "Proposed for curator review — track at `/curation`" card with the review id. Reasons:

   - Reuses existing curator UI + audit pipeline; no new role system invented.
   - Auth is currently lightweight browser identity (any `X-User-Id` can be claimed), so a "two-key" auto-apply would be a single-key flow in practice — not safe.
   - Soft-delete and undo from §7.4/§7.5 still apply *post-approval*: the curator's accept-action runs the same `save_entity` / `merge_entities` / etc. paths the original design called for.
   - Read-only tools (`search_graph`, `get_entity`, `verify_claim`) bypass the queue and execute inline; they never write.

   Implications for P9/P10: P9 (read-only tools) is unblocked and ships first. P10 (mutating tools) becomes a "queue-into-curation" surface rather than the §7.3 in-memory store. Revisit if/when real auth + roles land.
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
