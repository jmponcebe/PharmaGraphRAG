# PharmaGraphRAG -- Copilot Instructions

## Project Overview
PharmaGraphRAG is a fully functional GraphRAG system for querying drug interactions and adverse events using FDA data. It combines a Neo4j knowledge graph with vector search (ChromaDB) and LLM-powered answers (Gemini API / Ollama), served via FastAPI and a Streamlit chat interface.

## Author
- **Name**: Jose Maria Ponce Bernabe
- **Background**: Biotechnology + Bioinformatics + Knowledge Engineering (BASF, NTT DATA) + MLOps (DengueMLOps TFM)
- **Goal**: Portfolio project to demonstrate GenAI/RAG skills and bridge KG experience with LLM integration

## Current Status (All Weeks Complete)

All three development phases are finished. The system is fully operational end-to-end.

| Component | Status | Module |
| --- | --- | --- |
| Data Pipeline | Complete | `src/pharmagraphrag/data/` |
| Knowledge Graph | Complete | `src/pharmagraphrag/graph/` |
| Vector Store | Complete | `src/pharmagraphrag/vectorstore/` |
| Query Engine | Complete | `src/pharmagraphrag/engine/` |
| LLM Integration | Complete | `src/pharmagraphrag/llm/` |
| REST API | Complete | `src/pharmagraphrag/api/` |
| Streamlit UI | Complete | `src/pharmagraphrag/ui/` |
| Agent Mode | Complete | `src/pharmagraphrag/agent/` (LangGraph ReAct + multi-agent supervisor) |
| Observability | Complete | `src/pharmagraphrag/observability.py` (Langfuse tracing) |
| Docker Compose | Complete | `docker-compose.yml` + `docker/` |
| CI/CD | Complete | `.github/workflows/ci.yml` + `deploy.yml` |
| Evaluation | Complete | `src/pharmagraphrag/evaluation/` (RAGAS metrics, agent eval, curated testset) |
| Tests | 263 passing | `tests/` |
| Cloud Deployment | Live | Streamlit Cloud + Cloud Run + Neo4j Aura |

### Data at a Glance
- **FAERS**: 2 quarters (2024Q3, 2024Q4) -- 816K reports, 3.9M drug entries, 2.8M reactions
- **DailyMed**: 88 drugs via openFDA API -- JSON labels
- **Knowledge Graph**: 4,998 Drug + 6,863 AdverseEvent + 7 Outcome + 32 DrugCategory nodes; 365,360 CAUSES + 15,759 HAS_OUTCOME + 193 INTERACTS_WITH + 47 BELONGS_TO relationships
- **Vector Store**: 5,654 text chunks, 384-dim embeddings (all-MiniLM-L6-v2), cosine similarity

## Architecture

```
FDA FAERS (CSV) + DailyMed (API)
        |
    Data Pipeline (ingestion + cleaning)
        |
+-------------------+  +------------------+
|  Neo4j (KG)       |  |  ChromaDB        |
|  Drug, Adverse    |  |  Drug label      |
|  Event, Category  |  |  embeddings      |
|  relationships    |  |  (chunks)        |
+--------+----------+  +--------+---------+
         |      GraphRAG        |
         +----------+-----------+
                    |
         Query Engine
         (entity extraction + graph traversal
          + vector search + context merging)
                    |
         LLM (Gemini API / Ollama + fallback)
                    |
         FastAPI (REST) + Streamlit (Chat UI)
```

### Query Flow (end-to-end)

#### Classic Mode (`POST /query`)
1. **Entity Extraction** (`engine/entity_extractor.py`): exact substring match + fuzzy matching (rapidfuzz, threshold=80) against known drug names from Neo4j or disk cache.
2. **Graph Retrieval** (`engine/retriever.py` -> `graph/queries.py`): for each drug, fetch info, adverse events (top-N), interactions, outcomes from Neo4j.
3. **Vector Retrieval** (`engine/retriever.py` -> `vectorstore/store.py`): semantic search in ChromaDB filtered by extracted drug names.
4. **Prompt Assembly** (`engine/query_engine.py`): merge graph + vector context into a structured prompt with `SYSTEM_PROMPT` + `USER_PROMPT`.
5. **LLM Generation** (`llm/client.py`): call Gemini API (primary) or Ollama (fallback). Auto-fallback on error.
6. **API Response** (`api/main.py`): return answer + sources via FastAPI `POST /query`.
7. **UI Display** (`ui/app.py` + `ui/components.py`): Streamlit chat with graph visualization, source evidence, pipeline steps expander (classic), nested sub-agent reasoning (multi), clickable follow-up buttons, and confidence tooltips.

#### Agent Mode (`POST /agent/query`)
1. **User question** arrives at the LangGraph ReAct agent (`agent/graph.py`).
2. **Tool selection**: the LLM (Gemini 2.5 Flash via `langchain-google-genai`) autonomously decides which tools to call based on the question.
3. **Tool execution**: 9 tools available (`agent/tools.py`): `search_drug_info`, `find_drugs_for_adverse_event`, `search_drug_labels`, `list_drug_interactions`, `search_drugs_by_name`, `search_adverse_events`, `get_drug_outcomes`, `compare_drugs`, `find_drugs_by_category`. Each wraps existing graph/vector services. `find_drugs_for_adverse_event` includes fuzzy fallback: if exact MedDRA term not found, suggests similar events via substring search.
4. **Iterative reasoning**: the agent can call multiple tools in sequence, refining its understanding. Has conversation memory via LangGraph `MemorySaver` checkpointer.
5. **Structured output**: the agent returns a `StructuredResponse` (Pydantic model) with answer, drugs_mentioned, adverse_events_mentioned, confidence level, and follow-up suggestions.
6. **Structured data collection** (`_collect_structured_data`): after agent execution, re-fetches structured graph/vector data based on which tools were called (same Neo4j/ChromaDB connections, no extra HTTP calls).
7. **Response**: returns `AgentQueryResponse` with answer, tool calls, tool results, `graph_data` (structured Neo4j context per drug), `vector_data` (ChromaDB search results), structured metadata (drugs, AEs, confidence, follow-ups). UI renders Sources + Graph tabs identically to classic mode.

#### Multi-Agent Mode (`POST /agent/multi`)
1. **User question** arrives at the supervisor agent (`agent/multi.py`).
2. **Delegation**: supervisor has 3 tool-wrappers: `ask_drug_expert`, `ask_safety_analyst`, `ask_literature_researcher`.
3. **Sub-agent execution**: each sub-agent is a `create_react_agent` with a specialized prompt and tool subset (Drug Expert: 4 tools, Safety Analyst: 4 tools, Literature Researcher: 2 tools). Each sub-agent's inner tool calls are captured and propagated to the response.
4. **Synthesis**: supervisor collects sub-agent responses and generates a final coherent answer with `response_format=StructuredResponse` (confidence, drugs_mentioned, adverse_events_mentioned, follow_up_suggestions).
5. **Structured data collection** (`_collect_structured_from_results`): uses `extract_entities()` (fuzzy matching) to identify drugs from sub-agent questions and responses. Fetches graph context via `get_drug_full_context()` and vector data via `store.search()` for up to 3 sub-agent questions.
6. **Response**: returns `AgentQueryResponse` with answer, tool calls (including nested `inner_tool_calls` per sub-agent), tool results, `graph_data`, `vector_data`, structured metadata (drugs, AEs, confidence, follow-ups). UI renders Sources + Graph tabs identically to other modes, plus nested sub-agent reasoning hierarchy.

## Tech Stack
- **Language**: Python 3.13 (runtime), compatible with 3.11+
- **Package Manager**: uv (fast, Rust-based)
- **Knowledge Graph**: Neo4j 5 Community (Docker container `pharmagraphrag-neo4j`)
- **Vector Store**: ChromaDB (embedded, persisted at `data/chroma/`)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, 384 dimensions)
- **LLM Primary**: Google Gemini API (Tier 1, `google-genai` SDK >= 1.64.0)
- **LLM Backup**: Ollama + Llama 3 / Mistral (local, `ollama` SDK >= 0.4)
- **Agent Framework**: LangGraph + LangChain (ReAct agent with tool calling)
- **Observability**: Langfuse (LLM tracing, token usage, latency monitoring)
- **Entity Matching**: rapidfuzz >= 3.14.3 (fuzzy string matching)
- **API**: FastAPI >= 0.115 with Pydantic v2
- **UI**: Streamlit 1.54+ with streamlit-agraph, pyvis, plotly
- **Containers**: Docker Compose (Neo4j + API + UI + optional Ollama)
- **CI/CD**: GitHub Actions (ci.yml: lint+test on push; deploy.yml: CD on v* tags via Cloud Build)
- **Evaluation**: RAGAS 0.4.3 (Faithfulness, Relevancy, Precision, Recall, Correctness) + custom agent tool accuracy
- **Testing**: pytest (261 tests passing)
- **CI/CD**: GitHub Actions (ci.yml: lint + test matrix 3.11/3.13; deploy.yml: v* tags → Cloud Build → Cloud Run)
- **Cloud Build**: Google Cloud Build (cloudbuild.yaml) — downloads ChromaDB from GCS, builds Docker, deploys
- **Object Storage**: Google Cloud Storage (gs://pharmagraphrag-data for ChromaDB snapshots)
- **Linting/Formatting**: ruff (check + format)
- **Logging**: loguru
- **Data formats**: Parquet (processed FAERS), JSON (DailyMed labels)

## Data Sources
1. **FDA FAERS**: https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
   - Quarterly CSV files with drug adverse event reports
   - Key tables: DRUG, REAC (reactions), OUTC (outcomes), DEMO (demographics)
2. **DailyMed**: https://dailymed.nlm.nih.gov/dailymed/
   - Drug label information (interactions, warnings, contraindications)
   - REST API via openFDA

## Project Structure

```
PharmaGraphRAG/
+-- .github/
|   +-- copilot-instructions.md    # This file
|   +-- workflows/
|       +-- ci.yml                 # CI: lint + test matrix (3.11/3.13), reusable via workflow_call
|       +-- deploy.yml             # CD: v* tags → Cloud Build → Cloud Run deploy
+-- data/
|   +-- raw/                       # Downloaded FAERS CSVs, DailyMed JSONs (gitignored)
|   |   +-- faers/                 # {2024Q3,2024Q4}/ with $-delimited .txt files
|   |   +-- dailymed/              # 88 JSON files (one per drug)
|   +-- processed/                 # Cleaned Parquet files (gitignored)
|   |   +-- faers/                 # {2024Q3,2024Q4}/ with DEMO/DRUG/REAC/OUTC/INDI.parquet
|   +-- chroma/                    # ChromaDB persistent storage (gitignored)
+-- src/pharmagraphrag/
|   +-- __init__.py                # Package root, version 0.1.0
|   +-- config.py                  # Pydantic BaseSettings (Neo4j, LLM, ChromaDB, FAERS, Langfuse) + model constants (FLASH_MODELS, PRO_MODELS)
|   +-- observability.py           # Langfuse tracing: init, callbacks, decorators, trace_generation, flush
|   +-- data/
|   |   +-- __init__.py
|   |   +-- download_faers.py      # Download FAERS quarterly ZIPs from FDA
|   |   +-- clean_faers.py         # Clean FAERS CSVs -> Parquet (normalize, dedup)
|   |   +-- ingest_dailymed.py     # Fetch drug labels from openFDA API -> JSON
|   +-- graph/
|   |   +-- __init__.py
|   |   +-- schema.py              # Neo4j constraints + indexes
|   |   +-- loader.py              # Load FAERS + DailyMed into Neo4j (batch upserts)
|   |   +-- queries.py             # Cypher query functions for GraphRAG retrieval
|   +-- vectorstore/
|   |   +-- __init__.py
|   |   +-- chunker.py             # Text chunking (1000 chars, 200 overlap)
|   |   +-- embedder.py            # Embedding generation (all-MiniLM-L6-v2)
|   |   +-- store.py               # ChromaDB operations (add, search, format_context)
|   +-- engine/
|   |   +-- __init__.py
|   |   +-- entity_extractor.py    # Extract drug names (exact + fuzzy match)
|   |   +-- retriever.py           # Dual retrieval (graph + vector)
|   |   +-- query_engine.py        # Orchestrator: extract -> retrieve -> prompt
|   +-- llm/
|   |   +-- __init__.py
|   |   +-- client.py              # Unified LLM client (Gemini + Ollama + fallback)
|   +-- agent/
|   |   +-- __init__.py
|   |   +-- tools.py               # 9 LangChain tools wrapping graph/vector services
|   |   +-- graph.py               # LangGraph ReAct agent (create_react_agent + Gemini, per-model caching)
|   |   +-- multi.py               # Multi-agent supervisor with 3 specialized sub-agents (dual model support)
|   +-- api/
|   |   +-- __init__.py
|   |   +-- main.py                # FastAPI app: POST /query, POST /agent/query, POST /agent/multi, GET /drug/{name}, GET /health
|   |   +-- models.py              # Pydantic v2 request/response schemas (incl. AgentQueryRequest/Response)
|   +-- evaluation/
|   |   +-- __init__.py
|   |   +-- metrics.py             # RAGAS metric wrappers (Faithfulness, Relevancy, Precision, Recall, Correctness)
|   |   +-- dataset.py             # Curated testset loader, EvalSample/EvalDataset
|   |   +-- runner.py              # Batch evaluation runner (calls API, computes RAGAS scores, exports CSV)
|   |   +-- agent_eval.py          # Agent tool selection accuracy (precision/recall/F1)
|   +-- ui/
|       +-- __init__.py
|       +-- app.py                 # Streamlit chat: clickable follow-ups, confidence tooltips, pipeline steps (classic), nested sub-agent reasoning (multi)
|       +-- components.py          # Graph viz, sources panel, drug explorer
+-- data/
|   +-- evaluation/
|       +-- testset.json           # 25 curated evaluation questions (8 types, ground truth, expected tools)
+-- tests/
|   +-- __init__.py
|   +-- test_download_faers.py     # 2 tests
|   +-- test_clean_faers.py        # 13 tests
|   +-- test_ingest_dailymed.py    # 12 tests (mocked HTTP)
|   +-- test_vectorstore.py        # 35 tests (chunker + embedder + ChromaDB store)
|   +-- test_engine.py             # 37 tests (entity extractor + retriever + query engine)
|   +-- test_llm.py                # 14 tests (Gemini + Ollama + fallback, mocked)
|   +-- test_api.py                # 18 tests (FastAPI endpoints, TestClient, drug search, fallback)
|   +-- test_ui.py                 # 14 tests (Streamlit components + session state)
|   +-- test_agent.py              # 61 tests (9 tools, AgentResponse, StructuredResponse, multi-agent, endpoints)
|   +-- test_observability.py      # 13 tests (Langfuse init, callbacks, decorator, graceful degradation)
|   +-- test_evaluation.py         # 40 tests (dataset, metrics, runner, agent eval, all mocked)
+-- scripts/
|   +-- load_vectorstore.py        # One-off: populate ChromaDB
|   +-- validate_search.py         # One-off: test semantic search queries
|   +-- run_evaluation.py          # Batch eval: --mode classic|agent|multi|all, exports CSV reports
|   +-- setup_demo.py              # Demo setup: load graph + embeddings (~3 min)
|   +-- migrate_neo4j.py           # Migrate data between Neo4j instances
+-- docker/
|   +-- Dockerfile.api             # Multi-stage build, non-root, healthcheck
|   +-- Dockerfile.ui              # Multi-stage build, non-root, healthcheck
|   +-- Dockerfile.cloudrun        # Cloud Run: CPU-only PyTorch, baked-in ChromaDB
+-- docs/                          # Private didactic docs (gitignored)
|   +-- plan.md                    # Project plan
|   +-- 01_architecture_and_concepts.md
|   +-- 02_data_pipeline.md
|   +-- 03_knowledge_graphs_neo4j.md
|   +-- 04_embeddings_and_vector_search.md
|   +-- 05_python_modern_tooling.md
|   +-- 06_query_engine_and_llm.md
|   +-- 07_api_and_ui.md
|   +-- 08_cloud_deployment.md      # Free-tier cloud architecture
+-- .dockerignore
+-- .env.example
+-- .gitignore
+-- .pre-commit-config.yaml
+-- cloudbuild.yaml                # Cloud Build: GCS download → Docker build → GCR push → Cloud Run deploy
+-- docker-compose.yml             # Neo4j + API + UI + optional Ollama
+-- pyproject.toml
+-- uv.lock
+-- README.md
```

## Code Style & Conventions

### Python
- Use type hints everywhere (PEP 484)
- Pydantic v2 for all data models and settings
- f-strings for formatting
- Use pathlib.Path for file paths
- Docstrings: Google style
- Max line length: 88 (ruff default)
- Linting + Formatting: ruff (replaces black, isort, flake8)
- Type checking: mypy (continue-on-error in CI)

### Naming
- Modules: snake_case
- Classes: PascalCase
- Functions/variables: snake_case
- Constants: UPPER_SNAKE_CASE
- Neo4j labels: PascalCase (Drug, AdverseEvent)
- Neo4j relationships: UPPER_SNAKE_CASE (CAUSES, INTERACTS_WITH)

### Architecture Patterns
- Config via environment variables (.env file, Pydantic BaseSettings)
- Dependency injection for Neo4j driver, ChromaDB client, LLM client
- Each module is independently testable
- Separate retrieval (graph + vector) from generation (LLM)
- Synchronous FastAPI endpoints (Neo4j driver is sync)
- LLM fallback chain: Gemini -> Ollama -> error response

### Neo4j Schema
```cypher
// Nodes
(:Drug {name: string, pharmacologic_class: string?, source: string?})
(:AdverseEvent {name: string})
(:Outcome {code: string, name: string})
(:DrugCategory {name: string})

// Relationships
(:Drug)-[:CAUSES {report_count: int}]->(:AdverseEvent)
(:Drug)-[:INTERACTS_WITH {source: string, description: string?}]->(:Drug)
(:Drug)-[:HAS_OUTCOME {report_count: int}]->(:Outcome)
(:Drug)-[:BELONGS_TO]->(:DrugCategory)
```

### ChromaDB Schema
- **Collection**: `drug_labels` (cosine distance)
- **Embedding model**: all-MiniLM-L6-v2 (384 dimensions)
- **Chunk size**: 1000 chars with 200 overlap
- **Metadata per chunk**: drug_name, section, chunk_index, generic_names, brand_names, route
- **12 label sections**: drug_interactions, adverse_reactions, warnings_and_cautions, contraindications, boxed_warning, indications_and_usage, dosage_and_administration, clinical_pharmacology, mechanism_of_action, pharmacodynamics, overdosage, warnings

### API Endpoints
- `POST /query` -- Classic GraphRAG pipeline: question -> answer + sources
- `POST /agent/query` -- Agent Mode: LangGraph ReAct agent autonomously selects tools
- `GET /drug/{name}` -- Graph lookup: drug info, adverse events, interactions
- `GET /drugs/search?q=` -- Search drugs by name prefix (autocomplete)
- `GET /health` -- Service health: Neo4j + ChromaDB status

### Docker
- Multi-stage builds (builder + runtime) for API and UI images
- Non-root user (`appuser:1000`) in all containers
- Health checks for all services (Neo4j, API, UI)
- Volume mounts for Neo4j data persistence
- Optional Ollama service via Docker Compose profiles
- .env file for configuration (never committed)

### Git
- Conventional commits (feat:, fix:, docs:, refactor:, test:, ci:)
- Branch: main (protected) + feature branches
- .gitignore: data/raw/, data/processed/, data/chroma/, .env, __pycache__, .pytest_cache
- **Deploy rule**: NEVER create version tags or trigger deployments without explicit user confirmation. Commits and pushes to main are fine; tags (v*) require user approval.

### Testing (261 tests)
- pytest with fixtures for sample data and mocked services
- Mock Neo4j driver for graph tests
- Mock LLM API calls (never call real API in tests)
- `_DictLike(dict)` helper for Streamlit session state mocking
- `patch.dict("sys.modules", ...)` for streamlit-agraph component mocking
- Test matrix: Python 3.11 + 3.13 in GitHub Actions

| Test File | Count | Coverage |
| --- | --- | --- |
| test_download_faers.py | 2 | FAERS download URLs, skip existing |
| test_clean_faers.py | 13 | Normalization, dedup, outcome mapping |
| test_ingest_dailymed.py | 12 | API parsing, JSON save, error handling |
| test_vectorstore.py | 35 | Chunking, embeddings, ChromaDB CRUD |
| test_engine.py | 37 | Entity extraction, retrieval, prompt assembly |
| test_llm.py | 14 | Gemini, Ollama, fallback chain |
| test_api.py | 18 | FastAPI endpoints, TestClient, drug search, fallback |
| test_ui.py | 14 | Streamlit components, session state |
| test_agent.py | 61 | 9 tools, AgentResponse, StructuredResponse, multi-agent supervisor, model selector, endpoints |
| test_observability.py | 13 | Langfuse init, callback handler, config builder, decorator, trace generation, flush |
| test_evaluation.py | 42 | RAGAS metrics, dataset loading, runner, agent tool eval, call_agent parsing, CSV export |
| **Total** | **263** | |

### Evaluation (RAGAS)
- **Framework**: RAGAS 0.4.3 with Gemini via OpenAI-compatible endpoint
- **Curated testset**: 25 questions across 8 types (drug_info, interaction, adverse_event, outcome, category, comparison, multi_drug, label_search)
- **Reference-free metrics**: Faithfulness, Answer Relevancy
- **Reference-based metrics**: Context Precision, Context Recall, Answer Correctness
- **Agent evaluation**: Custom tool selection accuracy (precision/recall/F1), goal achievement tracking
- **Batch runner**: Calls API endpoints (classic/agent/multi), computes metrics, exports CSV
- **Script**: `scripts/run_evaluation.py --mode all --api-url http://localhost:8000`

## Key Design Decisions

1. **Neo4j over RDFLib**: Learning new skill (more marketable). Graph database provides native traversal.
2. **ChromaDB over Pinecone/Qdrant**: Embedded (no extra infra), SQLite-backed, good enough for portfolio scale.
3. **Gemini API over OpenAI**: Tier 1 ($10/month) gives generous quotas (Flash 10K RPD, Pro 1K RPD). Ollama as local backup removes vendor lock-in.
4. **google-genai over google-generativeai**: The `google-generativeai` SDK is deprecated. We use `google-genai >= 1.64.0` with `google.genai.Client` and `types.GenerateContentConfig`.
5. **Dual retrieval (graph + vector)**: The core differentiator. Graph provides structured context (relationships), vector provides unstructured context (text chunks). Merging both gives better answers than either alone.
6. **sentence-transformers over OpenAI embeddings**: Free, local, fast. all-MiniLM-L6-v2 is the standard baseline.
7. **rapidfuzz for entity extraction**: Fuzzy matching (threshold=80) catches misspellings and partial drug names without requiring an LLM call.
8. **Synchronous FastAPI**: Neo4j Python driver is synchronous; async endpoints would add complexity without benefit.
9. **LangGraph ReAct over manual StateGraph**: `create_react_agent` implements the ReAct loop natively. Tools wrap existing services (graph queries, vector search) — zero duplication. Agent Mode is opt-in (toggle in UI, separate endpoint).
10. **Singleton Neo4j driver**: Single driver instance with connection pool instead of driver-per-query. Critical for Neo4j Aura where each new connection has high latency.

## LLM Configuration

### Gemini API
- Default model: gemini-2.5-flash (configurable per query via UI/API)
- Flash models: gemini-2.5-flash, gemini-3-flash-preview, gemini-2.5-flash-lite
- Pro models (supervisor only): gemini-2.5-pro, gemini-3.1-pro-preview
- Model selector: UI sidebar dropdown per mode. Multi-agent has dual selectors (supervisor + sub-agents)
- Per-model agent caching: agents cached by model string in dicts, not singletons
- API key via GEMINI_API_KEY env var
- SDK: google-genai (>= 1.64.0)
- Temperature: 0.3, max_output_tokens: 2048

### Ollama (backup)
- Model: llama3:8b (default) or mistral:7b
- Run in Docker (profile: ollama) or host
- Base URL via OLLAMA_BASE_URL env var

### Fallback Chain
1. Try configured provider (gemini or ollama)
2. If Gemini fails, automatically try Ollama
3. If both fail and context available, return graph/vector data as formatted answer (provider="fallback")
4. If no context available, return error in LLMResponse (ok=False)

**Agent fallback**: If agent hits rate limit, auto-falls back to classic pipeline → tries LLM (Flash, separate quota) → if LLM also fails → returns raw graph/vector context.

**Response cache**: Agent responses cached in-memory (max 50 entries, LRU) to avoid wasting RPD on repeated questions.

### System Prompt (actual)
```
You are a pharmaceutical knowledge assistant specializing in drug
interactions, adverse events, and safety information. Answer the
user's question based ONLY on the provided context from FDA FAERS
reports and DailyMed drug labels.

Rules:
- Only use information from the provided context
- Cite specific drugs, adverse events, and report counts when available
- If the context does not contain enough information, say so explicitly
- Be precise with medical terminology
- Structure your answer clearly with sections if needed
```

## Environment Variables (.env.example)
```
# LLM
GEMINI_API_KEY=your-key-here
OLLAMA_BASE_URL=http://ollama:11434    # Docker: http://ollama:11434
LLM_PROVIDER=gemini                    # gemini or ollama
LLM_MODEL=gemini-2.5-flash

# Neo4j
NEO4J_URI=bolt://localhost:7687        # Docker: bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=pharmagraphrag

# ChromaDB
CHROMA_PERSIST_DIR=./data/chroma

# App
API_HOST=0.0.0.0
API_PORT=8000
STREAMLIT_PORT=8501
```

## Cloud Deployment (Live)

The system is deployed on a distributed free-tier architecture:

| Service | Platform | URL | Cost |
| --- | --- | --- | --- |
| Chat UI | Streamlit Community Cloud | https://pharmagraphrag.streamlit.app | $0 |
| API + ChromaDB | Google Cloud Run | pharmagraphrag-api-893694384146.us-central1.run.app | $0 (free tier) |
| Knowledge Graph | Neo4j Aura Free | Managed instance (11.9K nodes, 381K rels) | $0 (200K nodes limit) |

### Deployment Architecture
- **Streamlit Cloud**: reads `API_URL` from `st.secrets`, switches to HTTP mode (calls Cloud Run API instead of local imports). Uses `uv sync` from `uv.lock`. Main file: `src/pharmagraphrag/ui/app.py`.
- **Cloud Run**: Docker image with CPU-only PyTorch + baked-in ChromaDB + pre-cached embedding model. `Dockerfile.cloudrun` uses multi-stage build. Min instances=0 (scale to zero), max=2. Cold start ~50s, warm ~4.5s.
- **Neo4j Aura**: Free tier (200K nodes, 400K rels). Data migrated via `scripts/migrate_neo4j.py`. Auto-pauses after 3 days inactivity.
- **Gemini API**: `GEMINI_API_KEY` set as Cloud Run env var. Tier 1: Flash 10K RPD, Pro 1K RPD.

### Deployment Pipeline (CD)
- **Trigger**: GitHub Actions `deploy.yml` on version tags (`v*`)
- **Flow**: `deploy.yml` reuses CI workflow for tests → authenticates GCP → runs `gcloud builds submit --config=cloudbuild.yaml --substitutions=_TAG={tag}`
- **Cloud Build steps** (`cloudbuild.yaml`):
  1. Download ChromaDB snapshot from `gs://pharmagraphrag-data/chroma/chroma/`
  2. Multi-stage Docker build (`docker/Dockerfile.cloudrun`)
  3. Push image to GCR (`gcr.io/pharmagraphrag/api:{tag}` + `latest`)
  4. Deploy to Cloud Run (min_instances=0, max_instances=2)
- **Service Account**: `github-cd@pharmagraphrag.iam.gserviceaccount.com` (roles: run.admin, storage.admin, iam.serviceAccountUser, cloudbuild.builds.editor, logging.viewer, viewer)
- **GCS Bucket**: `gs://pharmagraphrag-data` (us-central1) — stores ChromaDB embeddings (99.6 MiB)
- **Versions**: v1.0.0 (initial), v1.0.3 (current)

### Key Deployment Files
- `cloudbuild.yaml` -- Cloud Build config: GCS download → Docker build → deploy
- `.github/workflows/deploy.yml` -- CD workflow: v* tags → Cloud Build → Cloud Run
- `docker/Dockerfile.cloudrun` -- Cloud Run image (CPU-only, baked ChromaDB)
- `scripts/migrate_neo4j.py` -- Migrate data between Neo4j instances
- `scripts/setup_demo.py` -- Load demo data into any Neo4j instance
- `requirements.txt` -- Minimal pip deps for Streamlit Cloud (fallback to uv.lock)

## Related Projects
- **DengueMLOps**: https://github.com/jmponcebe/DengueMLOps -- MLOps pipeline (same author)
- **Microsoft GraphRAG**: https://github.com/microsoft/graphrag -- Reference implementation
- **LlamaIndex Knowledge Graph**: https://docs.llamaindex.ai/en/stable/examples/index_structs/knowledge_graph/
