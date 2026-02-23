# PharmaGraphRAG — Copilot Instructions

## Project Overview
PharmaGraphRAG is a GraphRAG system for querying drug interactions and adverse events using FDA data. It combines a Neo4j knowledge graph with vector search (ChromaDB) and LLM-powered answers (Gemini API / Ollama).

## Author
- **Name**: Jose María Ponce Bernabé
- **Background**: Biotechnology + Bioinformatics + Knowledge Engineering (BASF, NTT DATA) + MLOps (DengueMLOps TFM)
- **Goal**: Portfolio project to demonstrate GenAI/RAG skills and bridge KG experience with LLM integration

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
- **Language**: Python 3.11+
- **Knowledge Graph**: Neo4j (Docker container)
- **Vector Store**: ChromaDB (embedded)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM Primary**: Google Gemini API (free tier, google-generativeai SDK)
- **LLM Backup**: Ollama + Llama 3 / Mistral (local)
- **API**: FastAPI with Pydantic v2
- **UI**: Streamlit
- **Containers**: Docker Compose (Neo4j + app + optional Ollama)
- **CI/CD**: GitHub Actions
- **Testing**: pytest
- **Data formats**: Parquet (processed FAERS), JSON (DailyMed labels)

## Data Sources
1. **FDA FAERS**: https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
   - Quarterly CSV files with drug adverse event reports
   - Key tables: DRUG, REAC (reactions), OUTC (outcomes), DEMO (demographics)
2. **DailyMed**: https://dailymed.nlm.nih.gov/dailymed/
   - Drug label information (interactions, warnings, contraindications)
   - REST API available

## Project Structure

```
PharmaGraphRAG/
├── .github/
│   ├── copilot-instructions.md    # This file
│   └── workflows/
│       └── ci.yml                 # GitHub Actions: lint + test + build
├── data/
│   ├── raw/                       # Downloaded FAERS CSVs, DailyMed JSONs (gitignored)
│   ├── processed/                 # Cleaned Parquet files (gitignored)
│   └── sample/                    # Small sample for testing (committed)
├── src/
│   ├── __init__.py
│   ├── config.py                  # Settings (Pydantic BaseSettings, env vars)
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── faers.py               # Download + parse FAERS data
│   │   └── dailymed.py            # Fetch drug labels from DailyMed API
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── schema.py              # Neo4j schema definition (constraints, indexes)
│   │   ├── loader.py              # Load data into Neo4j
│   │   └── queries.py             # Cypher query functions
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── chunker.py             # Text chunking for drug labels
│   │   ├── embedder.py            # Embedding generation
│   │   └── store.py               # ChromaDB operations
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── entity_extractor.py    # Extract drug/condition entities from query
│   │   ├── graph_retriever.py     # Neo4j context retrieval
│   │   ├── vector_retriever.py    # ChromaDB context retrieval
│   │   ├── context_merger.py      # Combine graph + vector contexts
│   │   └── generator.py           # LLM answer generation (Gemini + Ollama)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI app
│   │   ├── routes.py              # API endpoints
│   │   └── models.py              # Pydantic request/response models
│   └── ui/
│       └── app.py                 # Streamlit dashboard
├── tests/
│   ├── conftest.py                # Fixtures (sample data, mock Neo4j)
│   ├── test_ingestion/
│   ├── test_graph/
│   ├── test_vectorstore/
│   ├── test_rag/
│   └── test_api/
├── docker/
│   ├── Dockerfile                 # App container
│   └── Dockerfile.ollama          # Ollama container (optional)
├── scripts/
│   ├── download_faers.py          # One-off data download
│   ├── load_graph.py              # Populate Neo4j from processed data
│   └── load_vectorstore.py        # Populate ChromaDB from drug labels
├── notebooks/                     # Exploration notebooks (gitignored or sample only)
├── .env.example                   # Environment variables template
├── .gitignore
├── docker-compose.yml             # Neo4j + ChromaDB + App + Streamlit (+ Ollama)
├── pyproject.toml                 # Project config (dependencies, tools)
├── README.md
└── docs/
    └── plan.md                    # Project plan (moved from cv repo)
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
(:Drug {name: string, drugbank_id: string?, category: string?})
(:AdverseEvent {name: string, meddra_code: string?})
(:DrugCategory {name: string})

// Relationships
(:Drug)-[:CAUSES {report_count: int, severity: string?}]->(:AdverseEvent)
(:Drug)-[:INTERACTS_WITH {source: string, description: string?}]->(:Drug)
(:Drug)-[:BELONGS_TO]->(:DrugCategory)
```

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
