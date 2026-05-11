# CSLGraphBuilder

An enterprise biomedical knowledge graph platform for CSL Behring. Two halves that share one Neo4j store:

1. **Ingestion** — pulls documents (URLs, PDFs, JSON, plain text) and external sources (Open Targets, PubMed, web crawl) through a stage-aware pipeline that extracts biomedical entities and relationships via LLM, dedupes, verifies, and persists with full provenance.
2. **RAG Q&A chatbot** — a grounded, citation-emitting biomedical assistant that retrieves from the graph + source chunks via a hybrid Cypher + vector + BM25 + rerank pipeline, runs a per-claim faithfulness check, exposes function-calling tools (read-only + mutating, gated behind curator approval), streams answers over SSE, and remembers users across sessions.

Ships with a FastAPI backend, a Next.js 14 frontend (graph viewer, ingestion stage timeline, curation queue, **and a `/chat` page**), and Docker Compose for one-command deployment.

> **v2.3 — chat latency.** The `/chat` page now consumes `POST /qa/ask/stream` directly: the answer renders **token-by-token** (first token in ~½ s instead of a multi-second "thinking…" stall), and the source cards + retrieval trace paint the moment retrieval lands. Backend: the rolling-summary regeneration is **backgrounded** — a stale summary is refreshed in a detached task and the turn is served the previous (cached) one, so a long conversation never pays a serial second LLM call before answering — and the per-turn **query embedding is kicked off as a task** so the BM25 channel, term extraction, and working-memory load overlap it instead of queuing behind it. See [`docs/RAG_QA_PERF.md`](docs/RAG_QA_PERF.md).
>
> **v2.2 — RAG Q&A chatbot (P0–P14 of [`docs/RAG_QA_PLAN.md`](docs/RAG_QA_PLAN.md))**. A grounded chatbot over the knowledge graph. **Hybrid retrieval** (Cypher + vector + BM25 fused via RRF, then cross-encoder rerank, then `NEXT_CHUNK ±1` neighbour expansion). **Four-layer memory** — verbatim working window, rolling LLM summary, per-session episodic recall via `turn_query_vector`, and per-user persona summary that survives across sessions. **Function-calling tools** — read-only (`search_graph` / `get_entity` / `verify_claim`) and mutating (`propose_entity` / `propose_relationship` / `update_entity` / `merge_entities` / `soft_delete_*`), gated behind the existing curation queue. **Per-claim faithfulness** check + lexical-only default with optional LLM escalation. **SSE streaming** with interleaved `tool_call` events, consumed end-to-end by `/chat`. **Eval harness** with gold set, hermetic CI gate, and ablation matrix. See the dedicated [RAG Q&A Chatbot section](#rag-qa-chatbot) below.
>
> **v2.1 — workflow & UI upgrade.** Document processing runs through a stage-aware pipeline (`fetch → chunk → entities → relationships → finalize`) with bounded **parallel chunk extraction**, process-wide **LLM dedup + embedding caches**, **cooperative cancellation**, structured **per-stage SSE progress events**, and a **`/health/metrics`** endpoint exposing call volume, token usage, latency, and cache hit rates. The frontend renders all of this as a live **stage timeline** with a cancel button, plus a **Job History** page and a **Pipeline Performance** widget on the dashboard.

---

## Table of Contents

1. [Architecture](#architecture)
2. [RAG Q&A Chatbot](#rag-qa-chatbot)
3. [Key Features](#key-features)
4. [Prerequisites](#prerequisites)
5. [Quick Start](#quick-start)
6. [Docker Deployment](#docker-deployment)
7. [Configuration](#configuration)
8. [Verification Policy](#verification-policy)
9. [CLI Usage](#cli-usage)
10. [REST API](#rest-api)
11. [Project Structure](#project-structure)
12. [Module Responsibilities](#module-responsibilities)
13. [Testing](#testing)
14. [Contributing](#contributing)

---

## Architecture

The platform has two halves over one Neo4j store. **Ingestion** (left) writes the graph; the **RAG Q&A chatbot** (right) reads from it.

```
─────────────────────  WRITE PATH (ingestion, v2.1)  ─────────────────────
Input (URL / File / Text / Open Targets API / PubMed / Web Crawl)
        │
        ▼
DocumentExtractionPipeline           ← ordered stages with progress callbacks + cancel
  Stage 1  fetch         ← aiohttp + BeautifulSoup; or use pre-supplied content
  Stage 2  chunk         ← SemanticChunker; FIRST_CHUNK / NEXT_CHUNK linked list
  Stage 3  entities      ← parallel per-chunk LLM extraction (asyncio.Semaphore)
                            └─ vector pre-filter → cache-aware LLM dedup
  Stage 4  relationships ← parallel per-chunk LLM extraction
                            └─ vector pre-filter → cache-aware LLM dedup
  Stage 5  verify        ← cascading verifier on every new relationship
                            └─ confidence + conflict + source_trust → status
  Stage 6  finalize      ← persist counts + status to source document
        │
        ▼
Neo4j Knowledge Graph                  ←─── shared store ───→  READ PATH
  Document → [:FIRST_CHUNK] → Chunk → [:NEXT_CHUNK] → Chunk
  Chunk    → [:HAS_ENTITY]  → Entity
  Entity   → [:REL_TYPE]    → Entity
  Entity.name_embedding       ← 768-d vector index (cosine, SapBERT default)
  Relationship.desc_embedding ← 768-d vector index (cosine)
  ConversationSession → [:HAS_TURN] → ConversationTurn
  ConversationTurn.query_embedding ← `turn_query_vector` (episodic recall)
  User {id, display_name, metadata}     ← persona/embedding for §5.4
        │
        ▼
Relationship Verification (cascading) + Conflict Detection + Curation Queue
─────────────────────────────────────────────────────────────────────────

──────────────────  READ PATH (RAG Q&A chatbot, v2.2)  ──────────────────
User query                                             ┌─────────────────┐
   │  X-User-Id  X-API-Key                            │  /chat (Next.js)│
   ▼                                                  │  · MessageBubble│
POST /qa/ask  |  POST /qa/ask/stream  (SSE)           │  · SourceCard   │
   │                                                  │  · MutationCard │
   ▼                                                  │  · DebugPane    │
QAService.ask                                         └─────────────────┘
  ├── _resolve_session                ← create / load Conversation       ▲
  ├── _load_persona (P14)             ← User.metadata.semantic_summary   │
  ├── _embed_query                                                       │
  ├── asyncio.gather:                                                    │
  │   ├── RetrievalOrchestrator                                          │
  │   │     · Cypher channel (1-hop neighbour by name)                   │
  │   │     · Vector channel (entity + relationship embeddings)          │
  │   │     · BM25 channel (Neo4j fulltext)                              │
  │   │     ──► RRF fusion ──► CrossEncoder rerank ──► chunk hydration   │
  │   │                                                  + NEXT_CHUNK ±1 │
  │   └── MemoryService.build         ← working / summary / episodic     │
  │                                                                       │
  ├── (optional) agentic loop         ← enable_tools / enable_mutations  │
  │     · ToolDispatcher    : search_graph / get_entity / verify_claim    │
  │     · MutationDispatcher: propose_* / update_* / merge_* / soft_del_* │
  │     ──► curator approval queue at POST /qa/proposals/{id}/apply ─────┘
  │
  ├── _generate_answer (system prompt ← persona; user prompt ← memory + sources + question)
  ├── FaithfulnessChecker             ← per-claim lexical + optional LLM escalation
  ├── ConversationRepo.append_turn    ← persists turn + query_embedding
  └── (background) SemanticMemoryService.refresh_persona  ← P14 cross-session

         AskResult: answer, sources, retrieval_trace, memory_trace,
                    cited_source_indices, tool_calls, faithfulness,
                    request_id, latency_ms
─────────────────────────────────────────────────────────────────────────

Cross-cutting
  ├── PipelineMetrics      ← LLM calls, tokens, latency, cache + faithfulness signals
  ├── qa_observability     ← per-request id propagation, qa.* loggers, audit log
  ├── LLMDedupCache        ← skip repeat dedup LLM calls within & across runs
  ├── EmbeddingCache       ← skip repeat sentence-transformer encodes
  ├── Job store            ← stage progress, event log, cancel flag (ingest side)
  └── Eval harness         ← gold YAML + ablation matrix + hermetic CI gate
```

---

## RAG Q&A Chatbot

A grounded, citation-emitting biomedical assistant that answers questions from the knowledge graph. All 15 phases of [`docs/RAG_QA_PLAN.md`](docs/RAG_QA_PLAN.md) (P0–P14), the post-P14 follow-ups (`MutationCard` / `DebugPane` UI, streaming × tool-use combo, gpt-4o-mini QA default), and the chat-latency round (streaming `/chat`, backgrounded summary regen, non-blocking query embedding) have shipped. Companion docs: [`docs/RAG_QA_FOLLOWUPS.md`](docs/RAG_QA_FOLLOWUPS.md) (post-P14 roadmap) and [`docs/RAG_QA_PERF.md`](docs/RAG_QA_PERF.md) (latency work).

### Design pillars

1. **Hybrid retrieval, fused and reranked** — three channels run in parallel:
   - **Cypher channel** — 1-hop neighbourhood lookup by name match (preserves graph structure for relational questions).
   - **Vector channel** — entity-name and relationship-description embeddings (SapBERT default, 768-d cosine).
   - **BM25 channel** — Neo4j fulltext over chunks + entities (catches identifiers vectors miss).

   Channels are fused via **Reciprocal Rank Fusion**, then a **cross-encoder reranker** (`ms-marco-MiniLM-L-6-v2` by default) reorders the top-N. Hydrated chunks are expanded by **`NEXT_CHUNK ±1`** so a citation always lands in a coherent passage. An **intent classifier** routes questions to per-intent profiles (`lookup` / `relational` / `multi_hop` / `definitional` / `out_of_graph`) that tune top-k + channel weights — the relational profile alone moved recall from 24 % to 36 % on the gold set.

2. **Four memory layers** (§5 of the plan):

   | Layer | Where | Contents |
   |---|---|---|
   | Working | in-context, verbatim | Last N=3 user/assistant turns |
   | Rolling summary | in-context, compressed | LLM-compressed older turns. Regenerated **in a detached task** when the window slides; the turn is served the previous (cached) summary — so the answer never blocks on the summariser LLM. (`MemoryConfig.background_summary_refresh`, on by default; flip off for deterministic eval/tests.) |
   | Episodic | out-of-context, retrievable | Vector-search over `turn_query_vector` index — resolves pronouns like *"what about its side effects?"* |
   | Persona (P14) | out-of-context, cross-session | Per-user summary on `User.metadata.semantic_summary`; refreshed on session delete + as a background task on new-session create |

3. **Function-calling tool-use** — opt-in per request, both surfaces independent:
   - **Read-only tools** (`enable_tools=true`): `search_graph`, `get_entity`, `verify_claim`. Validated Pydantic schemas; dispatched against the existing orchestrator + graph repo. Agentic loop capped at `max_tool_calls_per_turn=5`.
   - **Mutating tools** (`enable_mutations=true`): `propose_entity`, `propose_relationship`, `update_entity`, `merge_entities`, `soft_delete_entity`, `soft_delete_relationship`. **Never apply directly** — every call enqueues a `ProposedMutation` (process-scoped store) that a curator must approve via `POST /qa/proposals/{id}/apply`, which then runs the actual `graph_repo.save_*` paths through `MutationApplier`. Closes §14.6 — "hybrid mutation authority": chat surface, human-in-the-loop apply.

4. **Per-claim faithfulness** — after generation, `FaithfulnessChecker` splits the answer into citation-anchored claims and scores each against its cited chunks. Lexical-only by default (deterministic, latency-free); flip `enable_llm_escalation=True` for stage-3 LLM verdicts on borderline claims. Refusals short-circuit to 1.0 — declining to answer *is* faithful. Aggregate score rides on `AskResponse.faithfulness.overall_score`.

5. **Streaming with tool-use** — `POST /qa/ask/stream` emits `phase` → `retrieval` → `phase("tools")` → `tool_call × N` → `phase("generating")` → `delta × N` → `done`. When tools are off, the streaming path is pure token-by-token; when on, the agentic loop runs to completion first (Option A) and the final answer arrives as a single `delta` — same answer as `/qa/ask`, provider-portable. The `/chat` frontend consumes this directly via a `fetch` + `ReadableStream` SSE client (not `EventSource` — the endpoint is a `POST` and needs auth headers): source cards + retrieval trace paint on the `retrieval` event, answer text accretes on each `delta`, and the in-flight stream is cancellable (`AbortController`) on unmount / new submit / session switch.

6. **Observability + eval-gating** — every turn emits `qa.*` logs tagged with `request_id`, records `qa_request` / `qa_latency` / `qa_context_tokens` / `qa_faithfulness_failure` metrics, and stamps a `RetrievalTrace` (channels run, fusion size, rerank order, hydrated chunks) onto the response. A hermetic CI gate ([`tests/eval/test_eval_smoke.py`](tests/eval/test_eval_smoke.py)) runs the full pipeline against an in-memory mini-graph and asserts floors on precision / recall / F1 / faithfulness; the live counterpart ([`tests/eval/run_rag_eval.py`](tests/eval/run_rag_eval.py)) posts `/qa/ask` and gates on per-intent targets.

### Frontend (`/chat`)

- **Composer toggles** for `enable_tools` / `enable_mutations` (persisted to `localStorage`, off by default).
- **`MessageBubble`** renders the answer with inline `[n]` citation chips that link to **`SourceCard`** components (per-channel confidence bars + hydrated chunk preview). While the SSE stream is live the answer fills in incrementally with a blinking caret; citation chips light up once the `done` event carries `cited_source_indices`.
- **`RetrievalTracePane`** (collapsible) shows extracted terms, per-channel hits, RRF→rerank→kept funnel.
- **`MutationCard`** (inline alongside the bubble whenever a turn proposes a mutation): plain-English summary, pretty-printed args, status pill, inline Approve / Reject buttons, deep-link to `/curation`.
- **`DebugPane`** (toggled via `?debug=1`): `request_id`, intent, memory trace, tool-calls table, per-claim faithfulness chips. Production users never see it.
- **`SessionSidebar`** with per-user session list, rename, delete; lightweight browser identity (`X-User-Id` + display name in `localStorage`).

### Quick try

```bash
# Anonymous one-shot ask
curl -X POST http://localhost:8000/qa/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"query": "What does Imatinib target?"}'

# Streamed (SSE) with tool-use opted in
curl -N -X POST http://localhost:8000/qa/ask/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -H "X-User-Id: user_abc123def456" \
  -d '{
        "query": "Are there proposed edits to BRCA1?",
        "enable_tools": true,
        "enable_mutations": false,
        "model": "gpt-4o"
      }'

# A/B the QA LLM against the gold set
python tests/eval/run_rag_eval.py \
  --base-url http://localhost:8000 \
  --gold tests/eval/rag_gold.yaml \
  --ablations relational_only,vector_only,no_rerank
```

### Memory + persona refresh triggers

- **Synchronous** on `DELETE /qa/sessions/{id}` — the just-ended session's turns flow into the persona before the session is dropped (`force=True`, `include_session_ids=[...]`).
- **Background** on new-session creation — `asyncio.create_task` runs `refresh_persona(user_id)`; the freshness gate (`min_new_sessions_to_refresh` + cached `META_COVERS` count) makes it a fast no-op when nothing has changed.

Persona lives at `User.metadata.semantic_summary` (string) + `semantic_embedding` (768-d) + `semantic_summary_covers_sessions` (count) + `semantic_summary_updated_at` (ISO). No schema migration — the `metadata` bag on `:User` was reserved for this in [`user_models.py`](src/graphbuilder/domain/models/user_models.py).

### Phase map

| Phase | Deliverable |
|---|---|
| **P0–P2** | Plan + `ConversationRepository` + observability spine |
| **P3** | Retrieval orchestrator (Cypher + vector + BM25 + RRF + chunk hydration) |
| **P4** | Cross-encoder rerank + `NEXT_CHUNK ±1` neighbour expansion |
| **P5** | `/qa/ask` endpoint |
| **P6** | Working memory + rolling summary |
| **P7** | Episodic recall via `turn_query_vector` |
| **P8** | Per-claim faithfulness check + eval plumbing |
| **P9** | Read-only tool-use (`search_graph` / `get_entity` / `verify_claim`) |
| **P10** | Mutating tool-use queued into `/curation` |
| **P11** | SSE streaming endpoint |
| **P12** | `/chat` page with citations + per-source confidence + retrieval trace |
| **P13** | Eval harness — gold set, metrics, ablations, hermetic CI gate, live CLI |
| **P14** | Cross-session semantic memory + per-user persona summary |
| **+ §9.9** | Intent-aware retrieval routing (relational recall 24 % → 36 %) |
| **+ Identity** | Lightweight browser identity (`X-User-Id`, ownership rules) — §14.1 |
| **+ MutationCard / DebugPane** | `/chat` UI for §7 + §8.5 (post-P14 follow-up §1) |
| **+ Streaming × tool-use** | `/qa/ask/stream` honours `enable_tools` / `enable_mutations` (post-P14 follow-up §3) |
| **+ Q2 QA model split** | `gpt-4o-mini` default for QA flow + per-request `AskRequest.model` override (post-P14 follow-up §2) |
| **+ Chat latency** | `/chat` consumes `/qa/ask/stream` (token-by-token render), backgrounded rolling-summary regen, non-blocking query embedding ([`docs/RAG_QA_PERF.md`](docs/RAG_QA_PERF.md)) |

See [`docs/RAG_QA_PLAN.md`](docs/RAG_QA_PLAN.md) for the full design rationale + the "Implementation refinements" log explaining where the shipped code drifted from the original sketch.

---

## Key Features

- **Stage-aware extraction pipeline** — Each document flows through six named stages (`fetch`, `chunk`, `entities`, `relationships`, `verify`, `finalize`) with structured per-stage progress callbacks. The frontend renders this as a live timeline.
- **Auto-verification + curation triage** — After extraction, the cascading verifier runs on every new relationship and tags it `verified`, `flagged`, or `rejected` per a configurable confidence × source-trust matrix. Most items skip the human queue; only the uncertain ones reach a curator. See **Verification Policy** below.
- **Biomedical embeddings (SapBERT)** — Default sentence-embedding model is `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, fine-tuned for biomedical entity linking. Override with `EMBEDDING_MODEL` env var; falls back to `all-MiniLM-L6-v2` if SapBERT can't load.
- **Parallel chunk processing** — Per-document entity & relationship extraction runs chunks concurrently with a bounded `asyncio.Semaphore` (capped by `parallel_workers`). Multi-chunk documents extract roughly N× faster.
- **LLM dedup cache** — Process-wide LRU keyed by the `(new entities, candidate entities)` signature. Repeat dedup calls within and across runs become free.
- **Embedding cache** — Process-wide LRU on entity-name embeddings; eliminates redundant sentence-transformer encodes during vector pre-filtering.
- **Cooperative cancellation** — `POST /documents/jobs/{id}/cancel` flips a flag; the pipeline polls between chunk batches and aborts cleanly with status `cancelled`.
- **Pipeline metrics endpoint** — `GET /health/metrics` exposes LLM call volume by type, prompt/completion tokens, average latency, cache hit rate, embedding hit rate, and graph throughput.
- **Structured SSE event stream** — `GET /documents/jobs/{id}/stream` emits typed `progress` and `done` events containing the full job snapshot (status, current stage, per-stage status map, recent event log).
- **Unified job model** — Documents, web crawls, PubMed, and Open Targets all share one `Job` shape; the same UI timeline renders for any kind.
- **Multi-source ingestion** — URLs, files, raw text, **Open Targets API (any entity kind: disease/target/drug/variant/study)**, PubMed, and web crawling with configurable depth and domain restrictions.
- **Batched embeddings + cascading ingest dedup** — `save_entities_batch` encodes all entity-name embeddings in one `model.encode` call (10–50× faster than per-entity), and runs each new entity through a 3-tier dedup cascade against the existing graph (external-ID → case-insensitive name/alias → SapBERT vector ≥ 0.92), with an external-ID-contradiction gate that prevents merging entities with conflicting IDs. Re-ingesting the same source becomes a no-op.
- **Async, non-blocking embedding** — `embed_async` / `embed_batch_async` run on a background thread executor under an `asyncio.Lock`, so a long encode no longer freezes `/health`, frontend polling, or other in-flight ingest jobs.
- **GPU embedding pool** — When `torch.cuda.is_available()`, a `GPUEmbeddingPool` activates automatically: N model copies, each on its own dedicated `torch.cuda.Stream`, sized from free GPU memory. CPU stays single-worker (per-architecture decision: multi-worker on CPU is no-throughput-gain because torch's MKL already saturates cores).
- **LLM entity & relationship extraction** — GPT-4 powered extraction with configurable schema constraints (`allowed_nodes`, `allowed_relationships`, `strict_mode`).
- **Two-stage LLM deduplication** — Vector pre-filter (low threshold, cheap) followed by LLM confirmation for domain-aware synonym resolution across abbreviations, alternate names, and scientific notation.
- **Neo4j vector search** — Native vector indexes on entity names and relationship descriptions for fast approximate nearest-neighbour queries.
- **Cascading verification pipeline** — Three-stage (text match → embedding → LLM) verification with confidence-based escalation; cheap stages run first and expensive stages only fire when earlier results are inconclusive.
- **Conflict detection** — Automatic identification of contradictory relationships (e.g. INHIBITS vs ACTIVATES between the same entity pair) with severity scoring.
- **Provenance tracking** — Every entity and relationship links back to source documents, chunks, and extraction metadata.
- **Source trust** — Configurable trust levels per source; higher-trust sources win in merge conflicts.
- **Curation workflow** — Queue-based human review with approve / reject / correct actions and full audit trail.
- **Web crawler with cache** — Crawls web pages with domain restrictions, page limits, and disk-based cache to avoid re-fetching.
- **Export** — JSON, Cytoscape, GraphML, and interactive HTML graph exports.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | venv recommended |
| Neo4j | 5.x | required for graph storage and vector search |
| OpenAI **or** Azure OpenAI | — | GPT-4o recommended |
| Node.js | 18+ | for local frontend development |
| Docker + Docker Compose | — | for containerised deployment |

---

## Quick Start

**1. Clone and install**

```bash
git clone <repo-url>
cd CSLGraphBuilder

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

**2. Configure environment**

```bash
cp .env.example .env
# Edit .env with your Neo4j and LLM credentials
```

**3. Process a document**

```bash
# Single URL
graphbuilder process --url https://example.com/article --title "My Article"

# Local PDF
graphbuilder process --file /path/to/paper.pdf --title "Research Paper"

# Restrict what the LLM may extract
graphbuilder process --url https://... \
  --allowed-nodes Gene Disease Drug \
  --allowed-relationships ASSOCIATED_WITH TREATS
```

**4. Verify relationships**

```bash
graphbuilder verify --context-file context.txt
```

---

## Docker Deployment

```bash
# Start Neo4j + API + Frontend (default)
docker compose up -d

# Add nginx reverse proxy
docker compose --profile nginx up -d
```

| Service | Port | Description |
|---|---|---|
| `neo4j` | 7474 / 7687 | Neo4j database |
| `api` | 8000 | FastAPI backend |
| `frontend` | 3000 | Next.js frontend |
| `nginx` *(optional)* | 80 | Reverse proxy (`/api/*` → api, `/*` → frontend) |

Health check: `GET http://localhost:8000/health` → `{"status":"ok"}`

---

## Configuration

All settings are read from environment variables (loaded from `.env` via `python-dotenv`).

### Neo4j

| Variable | Default | Description |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | Connection URI |
| `NEO4J_USER` | `neo4j` | Username |
| `NEO4J_PASSWORD` | *(required)* | Password |
| `NEO4J_DATABASE` | `neo4j` | Database name |

### LLM

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `azure_openai` | `openai` \| `azure_openai` |
| `LLM_MODEL_NAME` | `gpt-4o` | Default model — used by ingestion (entity/relationship extraction, dedup) where quality matters most |
| `QA_LLM_MODEL_NAME` | `gpt-4o-mini` | **§14 Q2 resolution** — cheaper model used by the QA flow (answer generation, faithfulness escalation, persona summarisation). Per-request override via `AskRequest.model` (e.g. opt into `gpt-4o` for a single ask). |
| `LLM_API_KEY` | *(required)* | API key |
| `LLM_API_ENDPOINT` | *(required for Azure)* | `https://<resource>.openai.azure.com` |
| `LLM_API_VERSION` | `2024-02-01` | Azure API version |
| `LLM_TEMPERATURE` | `0.1` | Generation temperature |
| `LLM_MAX_TOKENS` | `4096` | Max output tokens |

### Embeddings

| Variable | Default | Description |
|---|---|---|
| `IS_EMBEDDING` | `false` | Enable chunk embedding persistence |
| `EMBEDDING_MODEL` | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | Model name; falls back to `all-MiniLM-L6-v2` if SapBERT can't load |

When `IS_EMBEDDING=true`, each `Chunk` node gets an `embedding` float-array property and Neo4j creates a native `VECTOR INDEX` (cosine similarity) over it.

**GPU embedding pool** (auto-activated when CUDA-enabled torch is installed):

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_GPU_WORKERS` | *(auto from free GPU mem)* | Exact worker count; bypasses sizing math |
| `EMBEDDING_GPU_MIN_WORKERS` | `1` | Floor on worker count |
| `EMBEDDING_GPU_MAX_WORKERS` | `8` | Hard ceiling for safety |
| `EMBEDDING_GPU_MEMORY_FRACTION` | `0.7` | Usable share of free GPU memory |
| `EMBEDDING_GPU_DEVICE` | `cuda:0` | Target CUDA device |

The pool sizes itself by loading + warming up one model, measuring its
GPU footprint, and dividing the available memory budget. Each worker
holds its own model copy + dedicated `torch.cuda.Stream`, so encodes
genuinely run in parallel on the GPU's SMs (sub-linear scaling — N=4
≈ 2.5×, N=8 ≈ 3×; bottleneck is shared SMs and memory bandwidth, not
queue depth). On CPU the pool stays disabled and the existing
single-worker `asyncio.Lock` + executor path handles requests.

### Processing

| Variable | Default | Description |
|---|---|---|
| `PROCESSING_CHUNK_SIZE` | `512` | Token chunk size |
| `PROCESSING_CHUNK_OVERLAP` | `50` | Token overlap between chunks |
| `DATABASE_PROVIDER` | `in_memory` | `in_memory` \| `neo4j` |

### API Server

| Variable | Default | Description |
|---|---|---|
| `API_KEY` | *(unset = open)* | When set, all requests must include `X-API-Key: <value>` |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |

### Minimal `.env` template

```dotenv
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=

# LLM (Azure OpenAI)
LLM_PROVIDER=azure_openai
LLM_MODEL_NAME=gpt-4o
LLM_API_KEY=
LLM_API_ENDPOINT=https://<resource>.openai.azure.com
LLM_API_VERSION=2024-02-01

# Processing
DATABASE_PROVIDER=neo4j
IS_EMBEDDING=true
EMBEDDING_MODEL=all-MiniLM-L6-v2
LOG_LEVEL=INFO
```

---

## Verification Policy

After extraction, every new relationship runs through the cascading
verifier (text-match → embedding → LLM, with the LLM stage skipped in
batch mode by default). The aggregated confidence + conflict signal +
source-trust level then map to a `verification_status` annotation that
drives the curation queue.

### Default thresholds

| `aggregated_confidence` | `conflict?` | `source_trust` | → `verification_status` |
|---|---|---|---|
| ≥ **0.90** | no | any | **`verified`** (auto-approved, skips queue) |
| 0.60 – 0.90 | no | `reviewed` | **`verified`** (trusted-source bias) |
| 0.60 – 0.90 | no | `extracted` | **`unverified`** (low priority in queue) |
| < 0.60 | no | any | **`flagged`** (mid priority — needs human eye) |
| any | **yes** | any | **`rejected`** (top of queue — conflicts with trusted data) |
| verifier crashed / disabled | — | — | **`unverified`** (default) |

Entities are auto-verified too via a lightweight two-stage check (no
LLM ever): **text-match** against this run's chunk text + **embedding
similarity** to existing graph entities. Confidence maps via the
entity-specific knobs below. Entities that don't reach `auto_approve`
but are above `flag_below` land in `unverified` (the middle bucket —
not flagged as bad, but waiting for human approval).

### Tuning

All thresholds are env-driven via `VerificationConfiguration`:

| Env var | Default | Effect |
|---|---|---|
| `VERIFY_ENABLED` | `true` | Run the verify stage at all |
| `VERIFY_BATCH_SKIP_LLM` | `true` | Skip the (slow, expensive) LLM stage during batch verify |
| `VERIFY_PARALLEL_WORKERS` | `4` | Bounded concurrency for in-pipeline verification |
| `VERIFY_ENTITY_AUTO` | `0.85` | Auto-approve threshold for entities |
| `VERIFY_ENTITY_FLAG` | `0.50` | Below this, entities go to `flagged` |
| `VERIFY_REL_AUTO` | `0.90` | Auto-approve threshold for relationships |
| `VERIFY_REL_FLAG` | `0.60` | Below this, relationships go to `flagged` |
| `VERIFY_TRUSTED_AUTO` | `0.60` | Trusted-source bias auto-approve threshold |
| `VERIFY_CONFLICT_AS` | `rejected` | What to do when a conflict is detected |

### Migrating to a different embedding model

If you change `EMBEDDING_MODEL` (e.g. MiniLM → SapBERT), existing
embeddings in Neo4j are stored at the old dim and won't compare
against new ones. One call recreates everything:

```bash
curl -X POST http://localhost:8000/dev/reembed
```

Drops `entity_name_vector` + `rel_desc_vector` indexes, re-creates them
at the new dimension via `_initialize_schema`, then re-embeds every
entity + relationship with the current model **through the batched
path** (one `model.encode` call per chunk of texts, not one per item —
4-5k entities reembed in ~60 s on CPU). Idempotent; safe to re-run.

### Startup warm-up

The embedding model (~440 MB for SapBERT) preloads in a background
task on FastAPI startup, so the first Process request after a fresh
boot doesn't pay the download/load cost. The API is reachable
immediately; if a request hits before warm-up finishes, the lazy
load path still works (just slower for that one call).

A typical Wikipedia extraction (50–100 relationships, mostly
`extracted` source trust) produces roughly 60–70% auto-verified, 5–10%
flagged, and 0–3% rejected — collapsing a 100-item review queue into
~10 items that actually need a human.

---

## CLI Usage

```
graphbuilder [OPTIONS] COMMAND [ARGS]

Commands:
  process    Process a single document (URL, file, or raw text)
  ingest     Ingest from external sources (Open Targets, PubMed)
  verify     Run relationship verification pipeline
  curate     Apply manual curation events to the graph
  visualize  Export graph to HTML / JSON / GraphML / Cytoscape
```

### `process`

```bash
graphbuilder process --url <url> --title <title>
graphbuilder process --file <path> --title <title>
graphbuilder process --text "raw text content" --title <title>

# Optional schema constraints
--allowed-nodes Gene Disease Drug            # repeatable
--allowed-relationships ASSOCIATED_WITH      # repeatable
--chunk-size 512
--chunk-overlap 50
```

### `ingest`

```bash
# Open Targets — accepts any entity kind. Kind is auto-detected from the
# ID prefix (EFO_/MONDO_/Orphanet_ → disease, ENSG → target, CHEMBL →
# drug, rs… → variant, GCST → study). Pass --entity-type to override.
graphbuilder ingest --source open-targets --entity-id EFO_0000400      # disease
graphbuilder ingest --source open-targets --entity-id ENSG00000048462  # target / gene
graphbuilder ingest --source open-targets --entity-id CHEMBL941        # drug

# Legacy --disease-id is still accepted (assumes --entity-type=disease).

# PubMed
graphbuilder ingest --source pubmed --query "FVIII hemophilia" --max-results 50
```

### `verify`

```bash
# Full cascading pipeline (default: text-match → embedding → LLM)
graphbuilder verify --context-file context.txt

# Disable individual stages
graphbuilder verify --no-llm
graphbuilder verify --no-embedding --no-llm          # text-match only

# Tune escalation band (controls when later stages are triggered)
graphbuilder verify --escalation-lower 0.2 --escalation-upper 0.8

# Tune embedding threshold
graphbuilder verify --threshold 0.6
```

### `visualize`

```bash
graphbuilder visualize --format html --output graph.html
graphbuilder visualize --format json --output graph.json
graphbuilder visualize --format graphml --output graph.graphml
```

---

## REST API

Base URL: `http://localhost:8000`

All endpoints accept/return JSON. Protect with `X-API-Key` header when `API_KEY` env var is set.

### Health & Metrics

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/health/ready` | Readiness — surfaces configured DB + LLM provider/model |
| GET | `/health/metrics` | Process-wide pipeline metrics (LLM calls, tokens, latency, cache hit rates, throughput, cache sizes) |

### Documents & Jobs

| Method | Path | Description |
|---|---|---|
| POST | `/documents/process` | Kick off the extraction pipeline (returns a job envelope with ordered stages) |
| GET | `/documents/jobs` | List recent jobs across all kinds (document / web-crawl / pubmed / open-targets) |
| GET | `/documents/jobs/{id}` | Full job snapshot — status, current stage, per-stage status map, event log, result |
| POST | `/documents/jobs/{id}/cancel` | Cooperative cancel; pipeline aborts at the next chunk boundary |
| GET | `/documents/jobs/{id}/stream` | SSE — typed `progress` + `done` events with the full snapshot |
| GET | `/documents` | List persisted source documents |

The job envelope:

```json
{
  "job_id": "…",
  "kind": "document",
  "status": "running",
  "stages": ["fetch", "chunk", "entities", "relationships", "finalize"],
  "current_stage": "entities",
  "stage_progress": {"fetch": "completed", "chunk": "completed", "entities": "running", ...},
  "progress": 0.42,
  "events": [{"ts": "…", "stage": "entities", "level": "info", "message": "Processed 4/12 chunks", "data": {...}}],
  "cancel_requested": false,
  "result": null
}
```

**`POST /documents/process` body:**
```json
{
  "url": "https://...",
  "title": "Optional title",
  "source_label": "Alternative to title (frontend uses this)",
  "tags": ["biomedical"],
  "chunk_size": 512,
  "chunk_overlap": 50
}
```

### Graph

| Method | Path | Description |
|---|---|---|
| GET | `/graph/stats` | Entity and relationship counts |
| GET | `/graph/entities` | List entities (filterable by type) |
| GET | `/graph/relationships` | List relationships |

### Ingest

| Method | Path | Description |
|---|---|---|
| POST | `/ingest/open-targets` | Open Targets ingestion — any entity kind (disease/target/drug/variant/study), neighbors materialized as separate entities/relationships |
| POST | `/ingest/pubmed` | PubMed article ingestion |
| POST | `/ingest/crawl` | Web crawl — seed URLs with depth/domain control |

**`POST /ingest/open-targets` body:**
```json
{
  "entity_id": "ENSG00000048462",          // any OT ID — disease/target/drug/variant/study
  "entity_type": "target",                  // optional; auto-detected from prefix when omitted
  "max_associations": 100,                  // caps disease↔target / drug↔target neighbor edges
  "min_association_score": 0.0,             // OT score floor (disease/target only)
  "tag": "cancer-2026"
}
```
Legacy `disease_id` field is still accepted as an alias for `entity_id`
(implies `entity_type="disease"`). Materialized edges per root kind:
disease→targets + known drugs; target→diseases + pathways + known drugs;
drug→mechanisms-of-action + indications; variant→transcript consequences;
study→trait diseases. Re-ingesting the same root is a no-op thanks to
the cascading dedup in `save_entities_batch`.

### RAG Q&A Chatbot

| Method | Path | Description |
|---|---|---|
| POST | `/qa/ask` | Single-turn chat. Body: `query`, optional `session_id`, `top_k`, `enable_tools`, `enable_mutations`, `model`, `ablation`. Returns `answer`, `sources`, `cited_source_indices`, `retrieval_trace`, `memory_trace`, `tool_calls`, `faithfulness`, `request_id`, `latency_ms`. |
| POST | `/qa/ask/stream` | SSE counterpart. Events: `phase` → `retrieval` → (`phase("tools")` → `tool_call × N`) → `phase("generating")` → `delta × N` → `done`. Honours `enable_tools` + `enable_mutations`. Error path emits a single `error` event with a `kind`. |
| GET | `/qa/sessions` | List sessions (filterable by `user_id`, falls back to header `X-User-Id`). |
| GET | `/qa/sessions/{id}` | Session + ordered turns. Ownership enforced (mismatch returns 404 to avoid leaking existence). |
| DELETE | `/qa/sessions/{id}` | Detach-delete. **Synchronously refreshes the user's persona** before drop (P14). |
| POST | `/qa/turns/{id}/feedback` | Per-turn thumbs (`rating: -1/0/1`, optional `comment`). |
| GET | `/qa/proposals` | List chatbot-proposed mutations (filterable by status). |
| POST | `/qa/proposals/{id}/apply` | Curator approves and applies via `MutationApplier`. |
| POST | `/qa/proposals/{id}/reject` | Curator rejects with optional notes. |
| POST | `/users` | Mint a new `:User` (lightweight browser identity, §14.1). |
| GET | `/users/{id}` | Fetch user (id + display_name + metadata incl. persona fields). |
| PATCH | `/users/{id}` | Rename + metadata patch. |

### Curation

| Method | Path | Description |
|---|---|---|
| POST | `/curation/events` | Submit batch curation events (approve / reject / correct) |
| GET | `/curation/queue` | View items pending review (filterable by status) |
| GET | `/curation/queue/counts` | Per-status counts (rejected / flagged / unverified) for entities + relationships |
| GET | `/curation/audit` | Recent audit log entries (chat-originated rows tagged `actor.kind="chat"`) |

### Verification

| Method | Path | Description |
|---|---|---|
| POST | `/verification/run` | Run cascading verification on selected relationships |
| POST | `/verification/text` | Verify a free-text claim against the knowledge graph |
| POST | `/verification/conflicts` | Detect contradictions for new claims against existing graph |
| GET | `/verification/reviews` | List pending conflict reviews |
| POST | `/verification/reviews/decide` | Approve or reject a flagged conflict |

### Export

| Method | Path | Description |
|---|---|---|
| GET | `/export?format=json` | Export graph (json \| cytoscape \| graphml \| html) |

---

## Project Structure

```
CSLGraphBuilder/
├── api/                                # FastAPI application
│   ├── main.py                         # App factory, CORS, router registration
│   ├── auth.py                         # X-API-Key guard
│   ├── dependencies.py                 # FastAPI Depends() factories (+ user_repo, conversation_repo singletons)
│   ├── job_store.py                    # Job model w/ stages, events, cancel flag
│   ├── review_store.py                 # In-memory conflict review store
│   ├── proposed_mutation_store.py      # Process-scoped chatbot-mutation queue (P10)
│   ├── routers/                        # health(+metrics), graph, documents, ingest, curation, verification, export, qa, users, dev
│   │   ├── qa.py                       # /qa/ask, /qa/ask/stream, /qa/sessions, /qa/proposals/*  (P5 + P9-P11 + P14)
│   │   └── users.py                    # /users — lightweight browser identity (§14.1)
│   └── schemas/                        # Pydantic request/response models (incl. AskRequest, AskResponse, ToolCallModel, FaithfulnessModel, ProposedMutationModel)
├── frontend/                           # Next.js 14 frontend
│   ├── app/
│   │   ├── page.tsx                    # Dashboard (graph stats + Pipeline Performance widget + recent jobs)
│   │   ├── graph/                      # Interactive graph viewer (react-force-graph-2d)
│   │   ├── process/                    # Document ingestion (stage timeline + cancel + result summary)
│   │   ├── ingest/                     # Open Targets / PubMed / Web Crawl, all rendered with the shared timeline
│   │   ├── documents/                  # Job History — split-pane list + live stage timeline
│   │   ├── curation/                   # Manual curation queue (incl. chatbot-proposed mutations)
│   │   ├── verification/               # Verification + conflict detection + pending reviews
│   │   ├── chat/                       # RAG Q&A — composer with Tools/Mutations toggles + ?debug=1 pane
│   │   └── export/                     # Graph export
│   ├── components/
│   │   ├── Nav.tsx                     # Sidebar navigation
│   │   ├── JobTimeline.tsx             # Reusable stage timeline + event log + cancel
│   │   ├── chat/
│   │   │   ├── MessageBubble.tsx       # Assistant answer + [n] citation chips
│   │   │   ├── SourceCard.tsx          # Per-channel confidence + chunk preview
│   │   │   ├── RetrievalTracePane.tsx  # Collapsible "show your work" panel
│   │   │   ├── MutationCard.tsx        # Inline card for chatbot-proposed mutations (P10)
│   │   │   ├── DebugPane.tsx           # ?debug=1 developer pane (§8.5)
│   │   │   ├── SessionSidebar.tsx      # Per-user session list
│   │   │   └── IdentityPrompt.tsx      # First-visit display-name prompt
│   │   └── Providers.tsx               # React Query provider
│   └── lib/
│       ├── api.ts                      # Typed API client (Job, AskResponse, ToolCall, FaithfulnessResult, ProposedMutation, ChatUser)
│       ├── identity.ts                 # localStorage user_id + display_name
│       └── useJobStream.ts             # SSE subscription with polling fallback
├── src/graphbuilder/                   # Installable Python package
│   ├── cli/main.py                     # Click CLI entry point
│   ├── application/use_cases/
│   │   ├── document_pipeline.py        # Stage-aware orchestrator with caches + cancel + parallel chunks
│   │   ├── document_processing.py      # Legacy task-state-machine (still covered by tests)
│   │   ├── pubmed_ingestion.py
│   │   ├── open_targets_ingestion.py
│   │   ├── relationship_verification.py
│   │   ├── text_verification.py
│   │   ├── conflict_detection.py
│   │   ├── curation.py
│   │   └── graph_visualization.py
│   ├── core/
│   │   ├── graph/                      # Chunking, transformer, schema extraction
│   │   ├── verification/               # Cascading verifier (text → embedding → LLM)
│   │   ├── retrieval/                  # RAG Q&A stack (P3-P14)
│   │   │   ├── orchestrator.py         # Hybrid retrieval: cypher + vector + BM25 + RRF + hydration
│   │   │   ├── channels.py             # Per-channel implementations
│   │   │   ├── rrf.py                  # Reciprocal Rank Fusion
│   │   │   ├── reranker.py             # Cross-encoder rerank (ms-marco-MiniLM-L-6-v2)
│   │   │   ├── intent.py               # Rule-based classifier + per-intent profiles (§9.9)
│   │   │   ├── memory.py               # Working / rolling-summary / episodic layers (P6 + P7)
│   │   │   ├── semantic_memory.py      # Cross-session persona summary (P14)
│   │   │   ├── faithfulness.py         # Per-claim lexical + optional LLM escalation (P8)
│   │   │   ├── tools.py                # Read-only ToolDispatcher (P9)
│   │   │   ├── mutation_tools.py       # Mutating dispatcher — enqueues into proposal store (P10)
│   │   │   ├── mutation_applier.py     # Curator-approved apply path
│   │   │   ├── term_extraction.py      # Query-term extraction for BM25 + cypher anchors
│   │   │   ├── models.py               # RetrievalConfig, RetrievedItem, RetrievalTrace
│   │   │   └── qa_service.py           # ask() + ask_stream() glue — memory + retrieval + LLM + faithfulness + persistence
│   │   └── eval/                       # P13 eval harness (gold loader, metrics, async runner, CSV/markdown reports)
│   ├── domain/
│   │   ├── models/
│   │   │   ├── graph_models.py
│   │   │   ├── conversation_models.py  # ConversationSession + ConversationTurn (P1)
│   │   │   ├── user_models.py          # :User with metadata bag for persona (§14.1 + P14)
│   │   │   └── processing_models.py
│   │   └── repository interfaces       # ConversationRepository + UserRepository + GraphRepository
│   └── infrastructure/
│       ├── config/settings.py          # GraphBuilderConfig (env-var driven; LLMConfiguration.qa_model_name)
│       ├── crawlers/                   # web crawler (with cache), sync, json, file crawlers
│       ├── database/neo4j_client.py
│       ├── external/                   # open_targets_client, pubmed_client
│       ├── repositories/
│       │   ├── graph_repository.py     # Neo4j + in-memory entity/relationship repos with vector search
│       │   ├── document_repository.py  # Document + chunk persistence
│       │   ├── conversation_repository.py  # Sessions + turns + turn_query_vector (P1+P7)
│       │   └── user_repository.py      # :User CRUD (lightweight browser identity)
│       └── services/
│           ├── llm_service.py          # generate_text / generate_text_stream / generate_with_tools — per-call `model` kwarg (Q2)
│           ├── qa_observability.py     # request_id propagation, qa.* loggers, audit log
│           ├── metrics.py              # process-wide PipelineMetrics singleton (incl. qa_request, qa_latency, qa_faithfulness_failure)
│           ├── cache.py                # LLMDedupCache + EmbeddingCache (async LRU)
│           ├── embedding_factory.py    # Single source of truth for the embedder; sync + async + batched APIs
│           └── gpu_embedding_pool.py   # Multi-worker GPU pool (per-worker model + CUDA stream)
├── scripts/
│   ├── dedup_entities.py               # Cross-type entity dedup CLI (BRCA1 Concept + Brca1 Gene → merge)
│   ├── investigate_channels.py         # Per-channel diagnostic over the gold set
│   └── seed_gold_from_curation.py      # §14 Q5 — mine approved curation rows into a YAML gold draft
├── tests/
│   ├── unit/                           # Unit tests (incl. test_memory.py, test_semantic_memory.py, test_qa_service.py)
│   ├── integration/                    # Stage-aware pipeline coverage
│   ├── eval/                           # Gold set (rag_gold.yaml) + hermetic smoke gate (test_eval_smoke.py) + live runner (run_rag_eval.py)
│   └── e2e/                            # FastAPI TestClient + in-memory graph (incl. test_curation_queue.py for chatbot proposals)
├── Dockerfile.api
├── Dockerfile.frontend
├── docker-compose.yml
├── nginx.conf
├── pyproject.toml
└── requirements.txt
```

## Workflow Upgrade Highlights (v2.1)

| Concern | Before | After |
|---|---|---|
| Per-chunk LLM extraction | Sequential `for` loop | Bounded parallel `asyncio.gather` (capped by `parallel_workers`) |
| Repeat dedup calls | Always hit the LLM | Hashed (`new`, `candidates`) signature → in-process LRU; subsequent identical calls are free |
| Repeat embeddings | Re-encoded every call | Text-keyed LRU |
| Progress reporting | Single `progress` float updated start/end | Per-stage status map + append-only event log + weighted global progress |
| SSE | Polled state at 0.5 s, raw dict | Snapshot only when state changes; typed `progress` / `done` events |
| Cancellation | Not supported | Cooperative — `POST /documents/jobs/{id}/cancel` flips a flag the pipeline polls between chunks |
| Observability | Logs only | `GET /health/metrics` — calls by type, tokens, avg latency, cache hit rate, throughput |
| Document-pipeline contract | API-side `SourceDocument(url=..., content=...)` failed at import time | New `DocumentInput` shape; pipeline accepts pre-fetched content or fetches the URL itself |
| In-memory document repo | `save_chunks_with_links` was abstract | Implemented; pipeline now runs end-to-end without Neo4j |
| Frontend progress UI | Plain log lines | Shared `JobTimeline` component (stage rail, weighted bar, live event tail, cancel) used by Process / Ingest / Job History |
| Frontend dashboard | Stats only | + Pipeline Performance widget (auto-refresh 5s) and Recent Jobs panel |

---

## Module Responsibilities

| Layer | Responsibility | Allowed dependencies |
|---|---|---|
| `cli/` | Argument parsing and output only. No business logic. | Click, Rich |
| `api/` | HTTP transport, request validation, async job dispatch. The `qa.py` router owns the `QAService` singleton and injects the read + mutating tool dispatchers via setters — `core/` never imports `api/`. | FastAPI, Pydantic |
| `application/use_cases/` | Orchestrates the full ingestion pipeline via interfaces. No direct I/O. | None (calls domain interfaces) |
| `core/` | Pure domain algorithms: chunking, graph transformation, schema extraction, verification, **retrieval orchestration + RAG Q&A glue (`core/retrieval/`)**, **eval harness (`core/eval/`)**. Stateless. | LangChain (transformer only), sentence-transformers (rerank model) |
| `domain/` | Data models and repository interfaces. No implementation. | Pydantic |
| `infrastructure/` | All external integrations: Neo4j, LLM APIs, crawlers, file parsers, embeddings, **conversation + user repositories**. | All external libs |

---

## Testing

The project has **578 tests** across four tiers:

```bash
# Run all tests
python -m pytest tests/ -v

# Run by tier
python -m pytest tests/unit/ -v         # Fast, no external deps
python -m pytest tests/integration/ -v  # Mocked repos/services
python -m pytest tests/e2e/ -v          # FastAPI TestClient + in-memory graph
python -m pytest tests/eval/ -v         # Hermetic eval smoke gate + metric maths

# Live eval (requires running API + Neo4j with ingested gold-set sources)
python tests/eval/run_rag_eval.py --base-url http://localhost:8000 \
    --gold tests/eval/rag_gold.yaml --ablations relational_only,vector_only
```

| Tier | What it covers |
|---|---|
| **Unit** | Verification pipeline (text match / embedding / LLM / cascading), graph transformer, processor, LLM dedup methods, embedding helpers, **`MemoryService` (P6+P7)**, **`SemanticMemoryService` (P14)**, **`QAService.ask` + `ask_stream` (P5+P11)**, **`FaithfulnessChecker` (P8)**, **`ToolDispatcher` + `MutationToolDispatcher` (P9+P10)**, intent classifier (§9.9), user identity (§14.1), per-call LLM model override (§14 Q2). |
| **Integration** | Legacy document processing use case, LLM service, entity extraction with dedup, relationship extraction with entity resolution, `DocumentExtractionPipeline` (stage emission, cooperative cancel, dedup-cache reuse). |
| **E2E** | Full API pipeline (health → graph → curation → export), PubMed / OpenTargets ingest, extraction pipeline with dedup, **chatbot curation queue (`test_curation_queue.py` — proposal lifecycle from `/qa/ask` through `/qa/proposals/{id}/apply`)**. |
| **Eval** | Hermetic smoke gate ([`test_eval_smoke.py`](tests/eval/test_eval_smoke.py)) runs the full retrieval + memory + LLM stub + faithfulness pipeline against an in-memory mini-graph and asserts floors on precision / recall / F1 / faithfulness. Metric maths covered by [`test_eval_metrics.py`](tests/eval/test_eval_metrics.py). |

All external dependencies (Neo4j, LLM APIs) are mocked. Tests use `asyncio_mode = "auto"` via pytest-asyncio.

---

## Contributing

1. Fork the repo and create a feature branch from `main`.
2. Run `pip install -e ".[dev]"` and confirm `python -m pytest tests/ -q` passes.
3. All new business logic in `core/` and `application/` must have unit tests in `tests/unit/`.
4. Integration tests for use cases go in `tests/integration/`; API-level tests in `tests/e2e/`.
5. Follow the module responsibility boundaries above — no direct DB or API calls from `core/`.
6. Secrets must never be committed; use `.env` (gitignored).
