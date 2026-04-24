# CSLGraphBuilder

An enterprise knowledge graph construction pipeline for CSL Behring. Ingests documents (URLs, PDFs, JSON, plain text), extracts biomedical entities and relationships via LLM, and persists them into a Neo4j graph database with LLM-powered deduplication, cascading verification, conflict detection, and provenance tracking. Ships with a FastAPI backend, a Next.js 14 frontend, and Docker Compose for one-command deployment.

> **v2.1 — workflow & UI upgrade.** Document processing now runs through a stage-aware pipeline (`fetch → chunk → entities → relationships → finalize`) with bounded **parallel chunk extraction**, process-wide **LLM dedup + embedding caches**, **cooperative cancellation**, structured **per-stage SSE progress events**, and a **`/health/metrics`** endpoint exposing call volume, token usage, latency, and cache hit rates. The frontend renders all of this as a live **stage timeline** with a cancel button, plus a new **Job History** page and a **Pipeline Performance** widget on the dashboard.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Key Features](#key-features)
3. [Prerequisites](#prerequisites)
4. [Quick Start](#quick-start)
5. [Docker Deployment](#docker-deployment)
6. [Configuration](#configuration)
7. [CLI Usage](#cli-usage)
8. [REST API](#rest-api)
9. [Project Structure](#project-structure)
10. [Module Responsibilities](#module-responsibilities)
11. [Testing](#testing)
12. [Contributing](#contributing)

---

## Architecture

```
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
Neo4j Knowledge Graph
  Document → [:FIRST_CHUNK] → Chunk → [:NEXT_CHUNK] → Chunk
  Chunk    → [:HAS_ENTITY]  → Entity
  Entity   → [:REL_TYPE]    → Entity
  Entity.name_embedding      ← 384-d float[] vector index (cosine)
  Relationship.desc_embedding← 384-d float[] vector index (cosine)
        │
        ▼
Relationship Verification Pipeline (cascading with escalation bands)
  Stage 1 — TextMatchVerifier     ← regex / exact string pattern match
  Stage 2 — EmbeddingVerifier     ← cosine similarity + Neo4j vector search
  Stage 3 — LLMVerifier           ← structured LLM prompt with reasoning trace
        │
        ▼
Conflict Detection + Curation
  ├── Automatic contradiction detection (INHIBITS vs ACTIVATES)
  ├── Source trust scoring
  ├── Pending review queue
  └── Human-in-the-loop approve / reject / correct
        │
        ▼
Cross-cutting
  ├── PipelineMetrics      ← LLM calls, tokens, latency, cache hit rate
  ├── LLMDedupCache        ← skip repeat dedup LLM calls within & across runs
  ├── EmbeddingCache       ← skip repeat sentence-transformer encodes
  └── Job store            ← stage progress, append-only event log, cancel flag
        │
        ▼
REST API (FastAPI, SSE)  ←→  Next.js 14 Frontend (stage timeline, metrics widget)
```

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
- **Multi-source ingestion** — URLs, files, raw text, Open Targets API, PubMed, and web crawling with configurable depth and domain restrictions.
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
| `LLM_MODEL_NAME` | `gpt-4o` | Model or Azure deployment name |
| `LLM_API_KEY` | *(required)* | API key |
| `LLM_API_ENDPOINT` | *(required for Azure)* | `https://<resource>.openai.azure.com` |
| `LLM_API_VERSION` | `2024-02-01` | Azure API version |
| `LLM_TEMPERATURE` | `0.1` | Generation temperature |
| `LLM_MAX_TOKENS` | `4096` | Max output tokens |

### Embeddings

| Variable | Default | Description |
|---|---|---|
| `IS_EMBEDDING` | `false` | Enable chunk embedding persistence |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Model name, or `openai` / `vertexai` |

When `IS_EMBEDDING=true`, each `Chunk` node gets an `embedding` float-array property and Neo4j creates a native `VECTOR INDEX` (cosine similarity) over it.

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
curl -X POST http://localhost:8001/dev/reembed
```

Drops `entity_name_vector` + `rel_desc_vector` indexes, re-embeds
every entity + relationship with the current model, recreates the
indexes with the right dim. Idempotent; safe to re-run.

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
# Open Targets
graphbuilder ingest --source open-targets --disease-id EFO_0000400

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
| POST | `/ingest/open-targets` | Open Targets disease/gene enrichment |
| POST | `/ingest/pubmed` | PubMed article ingestion |
| POST | `/ingest/crawl` | Web crawl — seed URLs with depth/domain control |

### Curation

| Method | Path | Description |
|---|---|---|
| POST | `/curation/events` | Submit batch curation events (approve / reject / correct) |
| GET | `/curation/queue` | View items pending review (filterable by status) |

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
│   ├── dependencies.py                 # FastAPI Depends() factories
│   ├── job_store.py                    # Job model w/ stages, events, cancel flag
│   ├── review_store.py                 # In-memory conflict review store
│   ├── routers/                        # health (+metrics), graph, documents, ingest, curation, verification, export
│   └── schemas/                        # Pydantic request/response models
├── frontend/                           # Next.js 14 frontend
│   ├── app/
│   │   ├── page.tsx                    # Dashboard (graph stats + Pipeline Performance widget + recent jobs)
│   │   ├── graph/                      # Interactive graph viewer (react-force-graph-2d)
│   │   ├── process/                    # Document ingestion (stage timeline + cancel + result summary)
│   │   ├── ingest/                     # Open Targets / PubMed / Web Crawl, all rendered with the shared timeline
│   │   ├── documents/                  # Job History — split-pane list + live stage timeline
│   │   ├── curation/                   # Manual curation queue
│   │   ├── verification/               # Verification + conflict detection + pending reviews
│   │   └── export/                     # Graph export
│   ├── components/
│   │   ├── Nav.tsx                     # Sidebar navigation
│   │   ├── JobTimeline.tsx             # Reusable stage timeline + event log + cancel
│   │   └── Providers.tsx               # React Query provider
│   └── lib/
│       ├── api.ts                      # Typed API client (incl. Job, JobSummary, PipelineMetrics)
│       └── useJobStream.ts             # SSE subscription with polling fallback
├── src/graphbuilder/                   # Installable Python package
│   ├── cli/main.py                     # Click CLI entry point
│   ├── application/use_cases/
│   │   ├── document_pipeline.py        # NEW: lean stage-aware orchestrator with caches + cancel + parallel chunks
│   │   ├── document_processing.py      # Legacy task-state-machine (still covered by tests)
│   │   ├── pubmed_ingestion.py
│   │   ├── open_targets_ingestion.py
│   │   ├── relationship_verification.py
│   │   ├── text_verification.py
│   │   ├── conflict_detection.py
│   │   ├── curation.py
│   │   └── graph_visualization.py
│   ├── core/                           # Pure domain algorithms (chunking, transformer, verification cascade)
│   ├── domain/                         # Models + repository interfaces
│   └── infrastructure/
│       ├── config/settings.py          # GraphBuilderConfig (env-var driven)
│       ├── crawlers/                   # web crawler (with cache), sync, json, file crawlers
│       ├── database/neo4j_client.py
│       ├── external/                   # open_targets_client, pubmed_client
│       ├── repositories/               # Neo4j + in-memory document/graph repositories (vector search)
│       └── services/
│           ├── llm_service.py          # LLM extraction + dedup; records to PipelineMetrics
│           ├── content_extractor.py
│           ├── metrics.py              # NEW: process-wide PipelineMetrics singleton
│           └── cache.py                # NEW: LLMDedupCache + EmbeddingCache (async LRU)
├── tests/
│   ├── unit/                           # Unit tests
│   ├── integration/                    # Includes test_document_pipeline.py covering the new orchestrator
│   └── e2e/                            # FastAPI TestClient + in-memory graph
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
| `api/` | HTTP transport, request validation, async job dispatch. | FastAPI, Pydantic |
| `application/use_cases/` | Orchestrates the full pipeline via interfaces. No direct I/O. | None (calls domain interfaces) |
| `core/` | Pure domain algorithms: chunking, graph transformation, schema extraction, verification. Stateless. | LangChain (transformer only) |
| `domain/` | Data models and repository interfaces. No implementation. | Pydantic |
| `infrastructure/` | All external integrations: Neo4j, LLM APIs, crawlers, file parsers, embeddings. | All external libs |

---

## Testing

The project has **196 tests** across three tiers:

```bash
# Run all tests
python -m pytest tests/ -v

# Run by tier
python -m pytest tests/unit/ -v         # Fast, no external deps
python -m pytest tests/integration/ -v  # Mocked repos/services
python -m pytest tests/e2e/ -v          # FastAPI TestClient + in-memory graph
```

| Tier | What it covers |
|---|---|
| **Unit** | Verification pipeline (text match, embedding, LLM, cascading), graph transformer, processor, LLM dedup methods, embedding helpers |
| **Integration** | Legacy document processing use case, LLM service, entity extraction with dedup, relationship extraction with entity resolution, **new `DocumentExtractionPipeline`** (stage emission, cooperative cancel, dedup-cache reuse) |
| **E2E** | Full API pipeline (health → graph → curation → export), PubMed/OpenTargets ingest, extraction pipeline with dedup |

All external dependencies (Neo4j, LLM APIs) are mocked. Tests use `asyncio_mode = "auto"` via pytest-asyncio.

---

## Contributing

1. Fork the repo and create a feature branch from `main`.
2. Run `pip install -e ".[dev]"` and confirm `python -m pytest tests/ -q` passes.
3. All new business logic in `core/` and `application/` must have unit tests in `tests/unit/`.
4. Integration tests for use cases go in `tests/integration/`; API-level tests in `tests/e2e/`.
5. Follow the module responsibility boundaries above — no direct DB or API calls from `core/`.
6. Secrets must never be committed; use `.env` (gitignored).
