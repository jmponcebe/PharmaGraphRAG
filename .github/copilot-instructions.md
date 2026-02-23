# PharmaGraphRAG — Copilot Instructions

## Project Overview
PharmaGraphRAG is a GraphRAG system for querying drug interactions and adverse events using FDA data. It combines a Neo4j knowledge graph with vector search (ChromaDB) and LLM-powered answers (Gemini API / Ollama).

## Author
- **Name**: Jose María Ponce Bernabé
- **Background**: Biotechnology + Bioinformatics + Knowledge Engineering (BASF, NTT DATA) + MLOps (DengueMLOps TFM)
- **Goal**: Portfolio project to demonstrate GenAI/RAG skills and bridge KG experience with LLM integration

## Current Status (Week 1 Complete)

### What's built and working:
1. **Data Pipeline** (fully operational):
   - FAERS download: 2 quarters (2024Q3, 2024Q4) → ~135MB of CSVs
   - FAERS cleaning: 816K reports, 3.9M drug entries, 2.8M reactions → Parquet
   - DailyMed ingestion: 88 drugs via openFDA API → JSON labels

2. **Knowledge Graph** (Neo4j, running in Docker):
   - 4,998 Drug nodes
   - 6,863 AdverseEvent nodes
   - 7 Outcome nodes (Death, Hospitalization, Life-Threatening, etc.)
   - 32 DrugCategory nodes
   - 365,360 CAUSES relationships (drug → adverse event with report_count)
   - 15,759 HAS_OUTCOME relationships
   - 193 INTERACTS_WITH relationships (from DailyMed labels)
   - 47 BELONGS_TO relationships (drug → pharmacologic class)

3. **Vector Store** (ChromaDB, embedded):
   - 5,654 text chunks from 88 drugs across 12 label sections
   - 384-dimensional embeddings via all-MiniLM-L6-v2
   - Semantic search validated with drug interaction queries
   - Cosine similarity search with metadata filtering by drug name

4. **Tests**: 64 tests passing (29 data pipeline + 35 vector store)

### What's NOT built yet (Week 2-3):
- Query engine (entity extraction, graph + vector retrieval, context merging)
- LLM integration (Gemini API + Ollama fallback)
- FastAPI endpoints
- Streamlit UI
- Docker Compose full stack
- Full CI/CD pipeline

## Architecture

```
FDA FAERS (CSV) + DailyMed (API)
        ↓
    Data Pipeline (ingestion + cleaning)
        ↓
┌───────────────────┐  ┌──────────────────┐
│  Neo4j (KG)       │  │  ChromaDB        │
│  Drug, Adverse    │  │  Drug label      │
│  Event, Category  │  │  embeddings      │
│  relationships    │  │  (chunks)        │
└────────┬──────────┘  └────────┬─────────┘
         │      GraphRAG        │
         └──────────┬───────────┘
                    ↓
         Query Engine (entity extraction
         + graph traversal + vector search
         + context merging)
                    ↓
         LLM (Gemini API / Ollama)
                    ↓
         FastAPI + Streamlit
```

## Tech Stack
- **Language**: Python 3.13 (runtime), compatible with 3.11+
- **Package Manager**: uv (fast, Rust-based — installed at `C:\Users\ponce\.local\bin`)
- **Knowledge Graph**: Neo4j 5 Community (Docker container `pharmagraphrag-neo4j`)
- **Vector Store**: ChromaDB (embedded, persisted at `data/chroma/`)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, 384 dimensions)
- **LLM Primary**: Google Gemini API (free tier, google-generativeai SDK)
- **LLM Backup**: Ollama + Llama 3 / Mistral (local)
- **API**: FastAPI with Pydantic v2
- **UI**: Streamlit
- **Containers**: Docker Compose (Neo4j + app + optional Ollama)
- **CI/CD**: GitHub Actions
- **Testing**: pytest (64 tests passing)
- **Logging**: loguru
- **Data formats**: Parquet (processed FAERS), JSON (DailyMed labels)

## Data Sources
1. **FDA FAERS**: https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
   - Quarterly CSV files with drug adverse event reports
   - Key tables: DRUG, REAC (reactions), OUTC (outcomes), DEMO (demographics)
2. **DailyMed**: https://dailymed.nlm.nih.gov/dailymed/
   - Drug label information (interactions, warnings, contraindications)
   - REST API available

## Project Structure (actual)

```
PharmaGraphRAG/
├── .github/
│   ├── copilot-instructions.md    # This file
│   └── workflows/
│       └── ci.yml                 # GitHub Actions: lint + test + build
├── data/
│   ├── raw/                       # Downloaded FAERS CSVs, DailyMed JSONs (gitignored)
│   │   ├── faers/                 # {2024Q3,2024Q4}/ with $-delimited .txt files
│   │   └── dailymed/              # 88 JSON files (one per drug)
│   ├── processed/                 # Cleaned Parquet files (gitignored)
│   │   └── faers/                 # {2024Q3,2024Q4}/ with DEMO/DRUG/REAC/OUTC/INDI.parquet
│   ├── chroma/                    # ChromaDB persistent storage (gitignored)
│   └── sample/                    # Small sample for testing (committed)
├── src/pharmagraphrag/
│   ├── __init__.py                # Package root, version 0.1.0
│   ├── config.py                  # Pydantic BaseSettings (Neo4j, LLM, FAERS, etc.)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download_faers.py      # Download FAERS quarterly ZIPs from FDA
│   │   ├── clean_faers.py         # Clean FAERS CSVs → Parquet (normalize, dedup)
│   │   └── ingest_dailymed.py     # Fetch drug labels from openFDA API → JSON
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── schema.py              # Neo4j constraints + indexes (4 constraints, 5 indexes)
│   │   ├── loader.py              # Load FAERS + DailyMed into Neo4j (batch upserts)
│   │   └── queries.py             # Cypher query functions for GraphRAG retrieval
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── chunker.py             # Text chunking (1000 char chunks, 200 overlap)
│   │   ├── embedder.py            # Embedding generation (all-MiniLM-L6-v2, 384 dims)
│   │   └── store.py               # ChromaDB operations (add, search, format_context)
│   ├── engine/                    # TODO: GraphRAG query engine
│   │   └── __init__.py
│   ├── llm/                       # TODO: LLM integration (Gemini + Ollama)
│   │   └── __init__.py
│   ├── api/                       # TODO: FastAPI endpoints
│   │   └── __init__.py
│   └── ui/                        # TODO: Streamlit dashboard
│       └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── test_download_faers.py     # 2 tests
│   ├── test_clean_faers.py        # 13 tests
│   ├── test_ingest_dailymed.py    # 12 tests (mocked HTTP)
│   └── test_vectorstore.py        # 35 tests (chunker + embedder + ChromaDB store)
├── scripts/
│   ├── download_faers.py          # One-off: download FAERS data
│   ├── clean_faers.py             # One-off: clean FAERS → Parquet
│   ├── ingest_dailymed.py         # One-off: fetch DailyMed labels
│   ├── load_graph.py              # One-off: populate Neo4j
│   ├── load_vectorstore.py        # One-off: populate ChromaDB
│   └── validate_search.py         # One-off: test semantic search queries
├── docker/
│   ├── Dockerfile.api             # API container (Python 3.13-slim + uv)
│   └── Dockerfile.ui              # UI container (Streamlit)
├── .env.example                   # Environment variables template
├── .gitignore
├── .pre-commit-config.yaml        # Pre-commit hooks (ruff, trailing whitespace)
├── docker-compose.yml             # Neo4j service (API/UI commented out for now)
├── pyproject.toml                 # Project config (dependencies, ruff, pytest, mypy)
├── uv.lock                        # Lockfile for reproducible installs
├── README.md
└── docs/
    └── plan.md                    # Project plan
```

## Code Style & Conventions

### Python
- Use type hints everywhere (PEP 484)
- Pydantic v2 for all data models and settings
- f-strings for formatting
- Use pathlib.Path for file paths
- Docstrings: Google style
- Max line length: 88 (black default)
- Linting: ruff
- Formatting: black
- Import sorting: isort (profile=black)

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
- Each module should be independently testable
- Separate retrieval (graph + vector) from generation (LLM)
- Async FastAPI endpoints where I/O bound

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

### Docker
- Multi-stage builds for production image
- Non-root user in containers
- Health checks for all services
- Volume mounts for Neo4j data persistence
- .env file for configuration (never committed)

### Git
- Conventional commits (feat:, fix:, docs:, refactor:, test:, ci:)
- Branch: main (protected) + feature branches
- PR required for main (even self-merge is fine)
- .gitignore: data/raw/, data/processed/, .env, __pycache__, .pytest_cache

### Testing
- pytest with fixtures for sample data and mocked services
- Test Neo4j with testcontainers or mock
- Test LLM with mock responses (don't call real API in tests)
- Minimum coverage target: 80%

## Key Design Decisions

1. **Neo4j over RDFLib**: Learning new skill (more marketable). Can fallback to RDFLib if Neo4j proves too complex in the timeline.
2. **ChromaDB over Pinecone/Qdrant**: Embedded (no extra infra), SQLite-backed, good enough for portfolio scale.
3. **Gemini API over OpenAI**: Free tier is generous. Ollama as local backup removes vendor lock-in.
4. **Dual retrieval (graph + vector)**: The core differentiator. Graph provides structured context (relationships), vector provides unstructured context (text chunks). Merging both gives better answers than either alone.
5. **sentence-transformers over OpenAI embeddings**: Free, local, fast. all-MiniLM-L6-v2 is the standard baseline.

## LLM Configuration

### Gemini API
- Model: gemini-2.0-flash (fast, free tier)
- API key via GEMINI_API_KEY env var
- SDK: google-generativeai

### Ollama (backup)
- Model: llama3:8b or mistral:7b
- Run in Docker or host
- Base URL via OLLAMA_BASE_URL env var

### Prompt Template (draft)
```
You are a pharmaceutical knowledge assistant. Answer the user's question about drug interactions and adverse events based ONLY on the provided context.

GRAPH CONTEXT (structured relationships):
{graph_context}

TEXT CONTEXT (from drug labels):
{text_context}

USER QUESTION: {question}

Provide a clear, accurate answer. Cite specific drugs and adverse events from the context. If the context doesn't contain enough information to answer, say so explicitly.
```

## Environment Variables (.env.example)
```
# LLM
GEMINI_API_KEY=your-key-here
OLLAMA_BASE_URL=http://ollama:11434
LLM_PROVIDER=gemini  # gemini or ollama
LLM_MODEL=gemini-2.0-flash

# Neo4j
NEO4J_URI=bolt://neo4j:7687
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
- **DengueMLOps**: https://github.com/jmponcebe/DengueMLOps — MLOps pipeline (same author)
- **Microsoft GraphRAG**: https://github.com/microsoft/graphrag — Reference implementation
- **LlamaIndex Knowledge Graph**: https://docs.llamaindex.ai/en/stable/examples/index_structs/knowledge_graph/
