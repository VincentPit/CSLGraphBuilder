# CSLGraphBuilder

An enterprise knowledge graph construction pipeline for CSL Behring. Ingests documents (URLs, PDFs, JSON, plain text), extracts biomedical entities and relationships via LLM, and persists them into a Neo4j graph database. Ships with a FastAPI backend, a Next.js 14 frontend, and Docker Compose for one-command deployment.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Docker Deployment](#docker-deployment)
5. [Configuration](#configuration)
6. [CLI Usage](#cli-usage)
7. [REST API](#rest-api)
8. [Project Structure](#project-structure)
9. [Module Responsibilities](#module-responsibilities)
10. [Contributing](#contributing)

---

## Architecture

```
Input (URL / File / Text / Open Targets API / PubMed)
        │
        ▼
ContentExtractorService          ← aiohttp, BeautifulSoup, PyMuPDF, UnstructuredFileLoader
        │
        ▼
Chunking (TokenTextSplitter)     ← configurable chunk_size + overlap
        │
        ▼
Chunk Graph (Neo4j)              ← SHA-1 IDs; FIRST_CHUNK / NEXT_CHUNK linked list
        │
        ▼
LLM Extraction (LLMGraphTransformer)
  ├── allowed_nodes filter        ← restrict extracted node types (optional)
  ├── allowed_relationships filter← restrict extracted relationship types (optional)
  └── strict_mode                 ← reject entities not in allowed lists
        │
        ▼
Neo4j Knowledge Graph
  Document → [:FIRST_CHUNK] → Chunk → [:NEXT_CHUNK] → Chunk
  Chunk    → [:HAS_ENTITY]  → Entity
  Entity   → [:REL_TYPE]    → Entity
  Chunk.embedding            ← float[] vector, native Neo4j vector index (cosine)
        │
        ▼
Relationship Verification Pipeline (cascading, stops at first pass)
  Stage 1 — TextMatchVerifier     ← regex / exact string pattern match
  Stage 2 — EmbeddingVerifier     ← cosine similarity (sentence-transformers)
  Stage 3 — LLMVerifier           ← structured LLM prompt with reasoning trace
        │
        ▼
REST API (FastAPI)  ←→  Next.js 14 Frontend (React, Tailwind, react-force-graph-2d)
```

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
graphbuilder verify --method cascading --context-file context.txt
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
graphbuilder verify --method cascading --context-file context.txt
graphbuilder verify --method text-match
graphbuilder verify --method embedding --threshold 0.6
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

### Health

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/health/ready` | Readiness check (config loaded) |

### Documents

| Method | Path | Description |
|---|---|---|
| POST | `/documents/process` | Ingest a document (background job, returns job ID) |
| GET | `/documents/jobs/{id}` | Poll job status |
| GET | `/documents/jobs/{id}/stream` | SSE stream of job progress |
| GET | `/documents` | List ingested documents |

**`POST /documents/process` body:**
```json
{
  "url": "https://...",
  "title": "Optional title",
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

### Curation

| Method | Path | Description |
|---|---|---|
| POST | `/curation/events` | Submit batch curation events |
| GET | `/curation/queue` | View pending curation events |

### Verification

| Method | Path | Description |
|---|---|---|
| POST | `/verification/run` | Run cascading verification on all relationships |

### Export

| Method | Path | Description |
|---|---|---|
| GET | `/export?format=json` | Export graph (json \| cytoscape \| graphml \| html) |

---

## Project Structure

```
CSLGraphBuilder/
├── api/                           # FastAPI application
│   ├── main.py                    # App factory, CORS, router registration
│   ├── auth.py                    # X-API-Key guard
│   ├── dependencies.py            # FastAPI Depends() factories
│   ├── job_store.py               # In-memory background job tracker
│   ├── routers/                   # health, graph, documents, ingest, curation, verification, export
│   └── schemas/                   # Pydantic request/response models
├── frontend/                      # Next.js 14 frontend
│   ├── app/
│   │   ├── page.tsx               # Dashboard
│   │   ├── graph/                 # Interactive graph viewer (react-force-graph-2d)
│   │   ├── process/               # Document ingestion form
│   │   ├── ingest/                # Open Targets / PubMed ingestion
│   │   ├── curation/              # Manual curation queue
│   │   ├── verification/          # Verification report viewer
│   │   └── export/                # Graph export
│   ├── components/
│   │   ├── Nav.tsx                # Sidebar navigation
│   │   └── Providers.tsx          # React Query provider
│   └── lib/api.ts                 # Typed API client
├── src/graphbuilder/              # Installable Python package
│   ├── cli/main.py                # Click CLI entry point
│   ├── application/use_cases/     # ProcessDocument, Ingest, Verify, Curate, Visualize
│   ├── core/
│   │   ├── graph/transformer.py   # LLMGraphTransformer (allowed_nodes/rels, strict_mode)
│   │   ├── processing/processor.py# Chunking, FIRST/NEXT_CHUNK graph, vector index
│   │   ├── schema/extraction.py   # Structured schema extraction from LLM
│   │   ├── verification/          # TextMatch → Embedding → LLM cascading verifiers
│   │   └── utils/                 # common_functions, constants, visualization
│   ├── domain/
│   │   ├── entities/              # SourceNode, UserCredential
│   │   └── models/                # GraphEntity, GraphRelationship, KnowledgeGraph, ProcessingResult
│   └── infrastructure/
│       ├── config/settings.py     # GraphBuilderConfig (env-var driven)
│       ├── crawlers/              # web, sync, json, file crawlers
│       ├── database/neo4j_client.py
│       ├── external/              # open_targets_client, pubmed_client
│       ├── repositories/          # Neo4j + in-memory document/graph repositories
│       └── services/              # llm_service, content_extractor, legacy_llm
├── tests/
│   ├── unit/                      # 130 unit tests (all passing)
│   ├── integration/
│   └── e2e/
├── Dockerfile.api
├── Dockerfile.frontend
├── docker-compose.yml
├── nginx.conf
├── pyproject.toml
└── requirements.txt
```

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

## Contributing

1. Fork the repo and create a feature branch from `main`.
2. Run `pip install -e ".[dev]"` and confirm `python -m pytest tests/ -q` passes.
3. All new business logic in `core/` and `application/` must have unit tests in `tests/unit/`.
4. Follow the module responsibility boundaries above — no direct DB or API calls from `core/`.
5. Secrets must never be committed; use `.env` (gitignored).
