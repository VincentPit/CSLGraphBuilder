# CSLGraphBuilder

A knowledge graph construction pipeline for CSL Behring. Ingests documents (URLs, PDFs, JSON, plain text), extracts biomedical entities and relationships via LLM, and persists them into a Neo4j graph database. Designed to eventually layer in Open Targets API enrichment and a cascading relationship-verification pipeline.

> **Status:** Core pipeline is functional via the legacy execution path. The new async architecture (`src/graphbuilder/`) is structurally complete but has broken imports that must be resolved before it can run end-to-end. Tests do not yet exist. See [Known Issues](#known-issues) and [Roadmap](#roadmap).

---

## Table of Contents

1. [Architecture](#architecture)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [CLI Usage](#cli-usage)
6. [Project Structure](#project-structure)
7. [Module Responsibilities](#module-responsibilities)
8. [Execution Paths](#execution-paths)
9. [Known Issues](#known-issues)
10. [Roadmap](#roadmap)
11. [Contributing](#contributing)

---

## Architecture

```
Input (URL / File / Text)
        │
        ▼
ContentExtractorService          ← aiohttp, BeautifulSoup, PyMuPDF, UnstructuredFileLoader
        │
        ▼
Chunking (TokenTextSplitter)     ← chunk_size=200, overlap=20
        │
        ▼
Chunk Graph (Neo4j)              ← SHA-1 IDs; FIRST_CHUNK / NEXT_CHUNK linked list
        │
        ▼
LLM Extraction                   ← AzureOpenAI / OpenAI (JSON-mode)
  ├── Entity extraction           ← type, name, description, confidence
  └── Relationship extraction     ← source, target, type, confidence
        │
        ▼
Neo4j Knowledge Graph
  Document → [:FIRST_CHUNK] → Chunk → [:NEXT_CHUNK] → Chunk
  Chunk    → [:HAS_ENTITY]  → Entity
  Entity   → [:REL_TYPE]    → Entity

── Planned ──────────────────────────────────────────
Verification Pipeline (not yet implemented)
  1. Text/pattern match
  2. Embedding similarity (sentence-transformers)
  3. LLM reasoning
  → Cascading: stop at first pass; reject only if all three fail

Open Targets API Enrichment (not yet implemented)
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | Tested on 3.10.16 (conda env `graph`) |
| Neo4j | 5.x | Local or remote; Bolt protocol |
| Azure OpenAI **or** OpenAI | — | GPT-4o recommended |
| conda **or** pip | — | `environment.yml` or `requirements.txt` |

---

## Quick Start

**1. Clone and create the environment**

```bash
git clone <repo-url>
cd CSLGraphBuilder

# Option A — conda
conda env create -f environment.yml
conda activate graph

# Option B — pip + venv
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**2. Configure environment variables**

```bash
cp .env.example .env   # template provided below; create this file manually for now
```

Edit `.env` — at minimum you need Neo4j connection and an LLM key (see [Configuration](#configuration)).

**3. Verify Neo4j is reachable**

```bash
python -c "
from graphbuilder.infrastructure.config.settings import GraphBuilderConfig
from graphbuilder.core.utils.common_functions import create_graph_database_connection
cfg = GraphBuilderConfig()
g = create_graph_database_connection(cfg.database.uri, cfg.database.username, cfg.database.password)
print('Connected:', g)
"
```

**4. Process your first document**

```bash
# Process a URL
graphbuilder process --url https://example.com/article --title "My Article"

# Process a local PDF
graphbuilder process --file /path/to/document.pdf --title "Research Paper"

# Batch-process a directory
graphbuilder batch --input-dir ./data/docs --pattern "*.pdf"
```

---

## Configuration

All settings are read from environment variables (loaded from `.env` via `python-dotenv`). A config file (`--config`) in JSON/YAML format can override any value.

### Neo4j

| Variable | Default | Description |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | Connection URI |
| `NEO4J_USER` | `neo4j` | Username |
| `NEO4J_PASSWORD` | *(required)* | Password |
| `NEO4J_DATABASE` | `neo4j` | Database name |
| `NEO4J_MAX_POOL_SIZE` | `50` | Connection pool size |
| `NEO4J_CONNECTION_TIMEOUT` | `30` | Seconds |
| `NEO4J_ENCRYPTED` | `false` | Enable TLS |

### LLM

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `azure_openai` | `openai` \| `azure_openai` \| `huggingface` \| `local` |
| `LLM_MODEL_NAME` | `gpt-4o` | Model or Azure deployment name |
| `LLM_API_KEY` | *(required)* | API key |
| `LLM_API_ENDPOINT` | *(required for Azure)* | `https://<resource>.openai.azure.com` |
| `LLM_API_VERSION` | `2024-02-01` | Azure API version |
| `LLM_TEMPERATURE` | `0.1` | Generation temperature |
| `LLM_MAX_TOKENS` | `4096` | Max output tokens |
| `LLM_MAX_RETRIES` | `3` | Retry attempts on failure |
| `LLM_REQUESTS_PER_MINUTE` | `60` | Rate limit |

### Processing

| Variable | Default | Description |
|---|---|---|
| `PROCESSING_CHUNK_SIZE` | `512` | Token chunk size (new path) |
| `PROCESSING_CHUNK_OVERLAP` | `50` | Token overlap between chunks |
| `NUMBER_OF_CHUNKS_TO_COMBINE` | `5` | Chunks per LLM extraction call (legacy) |
| `NUMBER_OF_CHUNKS_ALLOWED` | `1000` | Max chunks per document |

### Crawler

| Variable | Default | Description |
|---|---|---|
| `CRAWLER_MAX_URLS` | `100` | Max URLs per crawl job |
| `CRAWLER_MAX_WORKERS` | `10` | Concurrent crawl workers |
| `CRAWLER_REQUEST_DELAY` | `1.0` | Seconds between requests |

### Other

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model name, or `openai` / `vertexai` |
| `APP_ENVIRONMENT` | `development` | `development` \| `production` |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

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
EMBEDDING_MODEL=all-MiniLM-L6-v2
LOG_LEVEL=INFO
```

---

## CLI Usage

```
graphbuilder [OPTIONS] COMMAND [ARGS]

Options:
  --config PATH     Path to JSON/YAML config file
  --verbose         Enable verbose logging
  --log-file PATH   Write logs to file

Commands:
  process    Process a single document (URL, file, or raw text)
  batch      Batch-process all documents in a directory
  crawl      Recursively crawl a website and ingest pages
  status     Check the processing status of a document
  optimize   Run graph optimization pass (deduplication, index tuning)
```

**Examples**

```bash
# Single URL
graphbuilder process --url https://example.com --title "Source Article"

# Local file, extract entities only
graphbuilder process --file report.pdf --no-extract-relationships

# Batch PDF ingest, 5 concurrent workers
graphbuilder batch --input-dir ./data --pattern "*.pdf" --max-concurrent 5

# Check status of a named document
graphbuilder status --title "Source Article"
```

---

## Project Structure

```
CSLGraphBuilder/
├── src/graphbuilder/              # Installable package (v2.0.0)
│   ├── cli/main.py                # CLI entry point (Click + Rich)
│   ├── application/
│   │   ├── cli/                   # Migrated legacy entry-point scripts
│   │   ├── dto/                   # (empty — DTOs not yet created)
│   │   └── use_cases/
│   │       └── document_processing.py   # ProcessDocumentUseCase, BatchProcessDocumentsUseCase
│   ├── core/
│   │   ├── graph/transformer.py         # LLMGraphTransformer (few-shot → GraphDocument)
│   │   ├── processing/processor.py      # Chunking + FIRST/NEXT_CHUNK graph construction
│   │   ├── schema/extraction.py         # Structured LLM output for node labels + rel types
│   │   └── utils/
│   │       ├── common_functions.py      # DB connection, embedding loader, graph write helpers
│   │       └── constants.py             # MODEL_VERSIONS, prompt templates, thresholds
│   ├── domain/
│   │   ├── entities/source_node.py      # SourceNode dataclass (status/type enums)
│   │   └── models/
│   │       ├── graph_models.py          # GraphEntity, GraphRelationship, KnowledgeGraph
│   │       └── processing_models.py     # ProcessingTask, ProcessingPipeline, ProcessingResult
│   └── infrastructure/
│       ├── config/settings.py           # GraphBuilderConfig (7 sub-configs, env-var driven)
│       ├── crawlers/                    # web_crawler, sync_crawler, json_crawler, file_crawler
│       ├── database/neo4j_client.py     # graphDBdataAccess (legacy LangChain wrapper)
│       ├── repositories/                # Neo4jDocumentRepository, Neo4jGraphRepository (async)
│       └── services/
│           ├── llm_service.py           # AdvancedLLMService (async, OpenAI + Azure)
│           ├── legacy_llm.py            # generate_graphDocuments (AzureChatOpenAI, LangChain)
│           └── content_extractor.py     # AdvancedContentExtractorService (multi-format)
├── legacy/                        # Pre-migration flat-file codebase (reference only)
├── tests/
│   ├── unit/                      # (empty)
│   ├── integration/               # (empty)
│   └── e2e/                       # (empty)
├── data/                          # Sample and test data (empty)
├── docs/                          # Documentation
├── pyproject.toml
├── requirements.txt
└── environment.yml
```

---

## Module Responsibilities

| Module | Responsibility | External deps allowed? |
|---|---|---|
| `cli/` | Argument parsing and output formatting only. No business logic. | Click, Rich |
| `application/use_cases/` | Orchestrates the full pipeline. Calls services and repositories via interfaces. No direct DB or API calls. | None |
| `core/` | Pure domain algorithms: chunking, graph transformation, schema extraction. Stateless. | LangChain (graph transformer only) |
| `domain/` | Data models and repository interfaces. No implementation. | Pydantic |
| `infrastructure/` | All external integrations: Neo4j, LLM APIs, crawlers, file parsers, embeddings. Implements domain interfaces. | All external libs |

---

## Execution Paths

There are currently two working execution paths. The intent is to converge on the **new path** once broken imports are fixed.

### Legacy path (battle-tested, functional)

Entry points: `legacy/scripts/main_*.py` or `application/cli/legacy_*_main.py`

```
Input → sync/async crawlers → CreateChunksofDocument
      → generate_graphDocuments() [LLMGraphTransformer + AzureChatOpenAI]
      → save_graphDocuments_in_neo4j() [graphDBdataAccess]
```

Limitations: synchronous in places, hardcoded Azure OpenAI only, no retry/rate-limit logic.

### New path (async, partially broken)

Entry point: `graphbuilder process` / `graphbuilder batch` CLI

```
Input → AdvancedContentExtractorService
      → ProcessDocumentUseCase
      → AdvancedLLMService [async OpenAI / Azure]
      → Neo4jDocumentRepository + Neo4jGraphRepository
```

Currently blocked by three broken import paths (see [Known Issues](#known-issues)).

---

## Known Issues

These must be resolved before the new execution path can run:

1. **`core/schema/extraction.py`** — imports `from src.llm import get_llm`. Should import from `graphbuilder.infrastructure.services.legacy_llm`.

2. **`core/processing/processor.py`** — bare import `from local_file import ...`. Should be `from graphbuilder.infrastructure.crawlers.file_crawler import ...`.

3. **`infrastructure/services/legacy_llm.py`** — imports `from graphTransformer import LLMGraphTransformer`. Should be `from graphbuilder.core.graph.transformer import LLMGraphTransformer`.

4. **`infrastructure/crawlers/web_crawler.py`** — domain filter is hardcoded to `dfrobot`. Must be made configurable before it can be used on any other source.

5. **`tests/`** — all test directories are empty. There is no test coverage.

6. **`application/dto/`** and **`domain/repositories/`** — empty; repository interfaces intended for `domain/` were placed directly in `infrastructure/` instead.

---

## Roadmap

Priorities are ordered by blocking impact.

### P0 — Fix broken imports (unblocks new execution path)

- Fix the three import errors in `core/schema/extraction.py`, `core/processing/processor.py`, and `infrastructure/services/legacy_llm.py`
- Add a smoke test to CI that imports all package modules

### P1 — Test coverage

- Unit tests for `core/processing/processor.py` (chunking logic, chunk-link graph)
- Unit tests for `core/graph/transformer.py` (LLM output → GraphDocument)
- Integration test for `ProcessDocumentUseCase` (document → Neo4j write) with a mock Neo4j or test instance
- Integration test for `AdvancedLLMService` with a mock OpenAI client

### P2 — Relationship verification pipeline

Implement the cascading verifier in `core/` with a clean interface:

```
verify_relationship(rel: GraphRelationship, context: str) -> VerificationResult
  1. TextMatchVerifier     — regex / exact string match
  2. EmbeddingVerifier     — cosine similarity via sentence-transformers; configurable threshold
  3. LLMVerifier           — structured LLM prompt with reasoning trace
```

- Each verifier is independently testable
- `VerificationResult` includes which stage passed and a confidence score
- Thresholds configurable via `infrastructure/config/settings.py`

### P3 — Open Targets API ingestion

- Add `infrastructure/external/open_targets_client.py` (GraphQL API wrapper)
- Create `application/use_cases/open_targets_ingestion.py`
- Add `graphbuilder ingest --source open-targets --disease-id EFO_XXXX` CLI command

### P4 — Configurability and hardening

- Make `web_crawler.py` domain-agnostic (pass allowed domains via config)
- Add structured JSON logging to `infrastructure/logging/`
- Add Dockerfile and docker-compose (Neo4j + app)
- Add `.env.example` to repo

### P5 — Future directions

- PubMed and internal dataset ingestion
- Manual curation / feedback loop
- Visualization tooling

---

## Contributing

1. Fork the repo and create a feature branch from `main`.
2. Run `pip install -e ".[dev]"` and ensure `black`, `isort`, and `mypy` pass before submitting.
3. All new business logic in `core/` and `application/` must have unit tests in `tests/unit/`.
4. Follow the module responsibility boundaries in the table above — no direct DB or API calls from `core/`.
5. Secrets must never be committed; use `.env` (gitignored).
