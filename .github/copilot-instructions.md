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
| Docker Compose | Complete | `docker-compose.yml` + `docker/` |
| CI/CD | Complete | `.github/workflows/ci.yml` |
| Tests | 142 passing | `tests/` |

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
1. **Entity Extraction** (`engine/entity_extractor.py`): exact substring match + fuzzy matching (rapidfuzz, threshold=80) against known drug names from Neo4j or disk cache.
2. **Graph Retrieval** (`engine/retriever.py` -> `graph/queries.py`): for each drug, fetch info, adverse events (top-N), interactions, outcomes from Neo4j.
3. **Vector Retrieval** (`engine/retriever.py` -> `vectorstore/store.py`): semantic search in ChromaDB filtered by extracted drug names.
4. **Prompt Assembly** (`engine/query_engine.py`): merge graph + vector context into a structured prompt with `SYSTEM_PROMPT` + `USER_PROMPT`.
5. **LLM Generation** (`llm/client.py`): call Gemini API (primary) or Ollama (fallback). Auto-fallback on error.
6. **API Response** (`api/main.py`): return answer + sources via FastAPI `POST /query`.
7. **UI Display** (`ui/app.py` + `ui/components.py`): Streamlit chat with graph visualization and source evidence.

## Tech Stack
- **Language**: Python 3.13 (runtime), compatible with 3.11+
- **Package Manager**: uv (fast, Rust-based)
- **Knowledge Graph**: Neo4j 5 Community (Docker container `pharmagraphrag-neo4j`)
- **Vector Store**: ChromaDB (embedded, persisted at `data/chroma/`)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, 384 dimensions)
- **LLM Primary**: Google Gemini API (free tier, `google-genai` SDK >= 1.64.0)
- **LLM Backup**: Ollama + Llama 3 / Mistral (local, `ollama` SDK >= 0.4)
- **Entity Matching**: rapidfuzz >= 3.14.3 (fuzzy string matching)
- **API**: FastAPI >= 0.115 with Pydantic v2
- **UI**: Streamlit 1.54+ with streamlit-agraph, pyvis, plotly
- **Containers**: Docker Compose (Neo4j + API + UI + optional Ollama)
- **CI/CD**: GitHub Actions (lint + test matrix 3.11/3.13 + Docker build with Buildx)
- **Testing**: pytest (142 tests passing)
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
|       +-- ci.yml                 # GitHub Actions: lint + test matrix + Docker build
+-- data/
|   +-- raw/                       # Downloaded FAERS CSVs, DailyMed JSONs (gitignored)
|   |   +-- faers/                 # {2024Q3,2024Q4}/ with $-delimited .txt files
|   |   +-- dailymed/              # 88 JSON files (one per drug)
|   +-- processed/                 # Cleaned Parquet files (gitignored)
|   |   +-- faers/                 # {2024Q3,2024Q4}/ with DEMO/DRUG/REAC/OUTC/INDI.parquet
|   +-- chroma/                    # ChromaDB persistent storage (gitignored)
+-- src/pharmagraphrag/
|   +-- __init__.py                # Package root, version 0.1.0
|   +-- config.py                  # Pydantic BaseSettings (Neo4j, LLM, ChromaDB, FAERS)
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
|   +-- api/
|   |   +-- __init__.py
|   |   +-- main.py                # FastAPI app: POST /query, GET /drug/{name}, GET /health
|   |   +-- models.py              # Pydantic v2 request/response schemas
|   +-- ui/
|       +-- __init__.py
|       +-- app.py                 # Streamlit chat interface
|       +-- components.py          # Graph viz, sources panel, drug explorer
+-- tests/
|   +-- __init__.py
|   +-- test_download_faers.py     # 2 tests
|   +-- test_clean_faers.py        # 13 tests
|   +-- test_ingest_dailymed.py    # 12 tests (mocked HTTP)
|   +-- test_vectorstore.py        # 35 tests (chunker + embedder + ChromaDB store)
|   +-- test_engine.py             # 37 tests (entity extractor + retriever + query engine)
|   +-- test_llm.py                # 14 tests (Gemini + Ollama + fallback, mocked)
|   +-- test_api.py                # 13 tests (FastAPI endpoints, TestClient)
|   +-- test_ui.py                 # 14 tests (Streamlit components + session state)
+-- scripts/
|   +-- load_vectorstore.py        # One-off: populate ChromaDB
|   +-- validate_search.py         # One-off: test semantic search queries
+-- docker/
|   +-- Dockerfile.api             # Multi-stage build, non-root, healthcheck
|   +-- Dockerfile.ui              # Multi-stage build, non-root, healthcheck
+-- docs/
|   +-- plan.md                    # Project plan
|   +-- 01_architecture_and_concepts.md
|   +-- 02_data_pipeline.md
|   +-- 03_knowledge_graphs_neo4j.md
|   +-- 04_embeddings_and_vector_search.md
|   +-- 05_python_modern_tooling.md
|   +-- 06_query_engine_and_llm.md
|   +-- 07_api_and_ui.md
+-- .dockerignore
+-- .env.example
+-- .gitignore
+-- .pre-commit-config.yaml
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
- `POST /query` -- Full GraphRAG pipeline: question -> answer + sources
- `GET /drug/{name}` -- Graph lookup: drug info, adverse events, interactions
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

### Testing (142 tests)
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
| test_api.py | 13 | FastAPI endpoints, TestClient |
| test_ui.py | 14 | Streamlit components, session state |
| **Total** | **142** | |

## Key Design Decisions

1. **Neo4j over RDFLib**: Learning new skill (more marketable). Graph database provides native traversal.
2. **ChromaDB over Pinecone/Qdrant**: Embedded (no extra infra), SQLite-backed, good enough for portfolio scale.
3. **Gemini API over OpenAI**: Free tier is generous. Ollama as local backup removes vendor lock-in.
4. **google-genai over google-generativeai**: The `google-generativeai` SDK is deprecated. We use `google-genai >= 1.64.0` with `google.genai.Client` and `types.GenerateContentConfig`.
5. **Dual retrieval (graph + vector)**: The core differentiator. Graph provides structured context (relationships), vector provides unstructured context (text chunks). Merging both gives better answers than either alone.
6. **sentence-transformers over OpenAI embeddings**: Free, local, fast. all-MiniLM-L6-v2 is the standard baseline.
7. **rapidfuzz for entity extraction**: Fuzzy matching (threshold=80) catches misspellings and partial drug names without requiring an LLM call.
8. **Synchronous FastAPI**: Neo4j Python driver is synchronous; async endpoints would add complexity without benefit.

## LLM Configuration

### Gemini API
- Model: gemini-2.0-flash (fast, free tier)
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
3. If both fail, return error in LLMResponse (ok=False)

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
LLM_MODEL=gemini-2.0-flash

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

## Related Projects
- **DengueMLOps**: https://github.com/jmponcebe/DengueMLOps -- MLOps pipeline (same author)
- **Microsoft GraphRAG**: https://github.com/microsoft/graphrag -- Reference implementation
- **LlamaIndex Knowledge Graph**: https://docs.llamaindex.ai/en/stable/examples/index_structs/knowledge_graph/
