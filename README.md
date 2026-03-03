# PharmaGraphRAG

[![CI](https://github.com/jmponcebe/PharmaGraphRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/jmponcebe/PharmaGraphRAG/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-142%20passing-brightgreen.svg)](#testing)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/jmponcebe/PharmaGraphRAG?quickstart=1)

> **GraphRAG system for querying drug interactions and adverse events using FDA data.**

A production-ready question-answering system that combines a **pharmaceutical knowledge graph** (Neo4j) with **Retrieval-Augmented Generation** (RAG) to answer natural language questions about drug interactions and adverse events — grounded in real **FDA FAERS** data and **DailyMed** drug labels.

### Key Highlights

- **Dual retrieval**: structured graph queries (Neo4j) + semantic vector search (ChromaDB) merged into a single LLM prompt
- **Real FDA data**: 816K adverse event reports, 4,998 drugs, 365K causal relationships, 88 drug labels
- **142 tests** with CI/CD on GitHub Actions (Python 3.11 + 3.13 matrix)
- **Full stack**: data pipeline → knowledge graph → vector store → query engine → REST API → chat UI
- **One-click Codespaces**: try it instantly from your browser

## Status

| Component | Status | Details |
|---|---|---|
| Data Pipeline | ✅ Complete | FAERS (2024Q3+Q4): 816K reports, 3.9M drug entries. DailyMed: 88 drugs |
| Knowledge Graph | ✅ Complete | 4,998 Drugs, 6,863 AdverseEvents, 365K CAUSES, 193 INTERACTS_WITH |
| Vector Store | ✅ Complete | 5,654 text chunks, 384-dim embeddings, cosine similarity search |
| Query Engine | ✅ Complete | Entity extraction (exact + fuzzy), dual retrieval, prompt assembly |
| LLM Integration | ✅ Complete | Gemini API + Ollama with automatic fallback |
| REST API | ✅ Complete | FastAPI: POST /query, GET /drug/{name}, GET /health |
| Chat UI | ✅ Complete | Streamlit: chat, graph visualization, sources panel, drug explorer |
| Docker Compose | ✅ Complete | Neo4j + API + UI + Ollama (optional profile) |
| CI/CD | ✅ Complete | GitHub Actions: lint, test matrix (3.11/3.13), Docker build |
| Tests | ✅ 142 passing | Data pipeline (27) + vectors (35) + engine (37) + LLM (14) + API (13) + UI (14) |

## Example Questions

- *"What are the side effects of ibuprofen?"*
- *"Does metformin interact with other drugs?"*
- *"What adverse events are associated with warfarin?"*
- *"Compare the safety profile of aspirin and clopidogrel"*
- *"What drugs cause liver damage?"*

## Example API Response

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the side effects of ibuprofen?"}'
```

```json
{
  "answer": "Based on FDA FAERS data, ibuprofen is associated with several adverse events...",
  "drugs_extracted": ["IBUPROFEN"],
  "drugs_found": ["IBUPROFEN"],
  "sources": {
    "graph": {
      "adverse_events": ["DRUG INEFFECTIVE (1,245)", "NAUSEA (892)", "PAIN (654)", "..."],
      "interactions": ["ASPIRIN", "NAPROXEN", "..."],
      "outcomes": ["Hospitalization (2,341)", "Death (156)", "..."]
    },
    "vector": ["drug_interactions (similarity: 94%)", "adverse_reactions (similarity: 91%)", "..."]
  },
  "llm_provider": "gemini",
  "llm_model": "gemini-2.5-flash"
}
```

## Architecture

```
FDA FAERS (CSV) ──→ Cleaning ──→ Neo4j Knowledge Graph
DailyMed (API) ──→ Extraction ──→ ChromaDB Vector Store
                                          │
         User Question ──→ Query Engine ──┤
                           (Entity NER +  │
                            Graph Query + │
                            Vector Search)│
                                          ▼
                                    LLM (Gemini / Ollama)
                                          │
                                    Grounded Answer
                                    + Sources/Evidence
```

### Query Flow

1. **Entity extraction** — Identifies drug names using exact substring matching (word-boundary aware) + fuzzy matching (rapidfuzz, threshold=80)
2. **Graph retrieval** — Queries Neo4j for adverse events, interactions, outcomes, and categories per drug
3. **Vector retrieval** — Searches ChromaDB for relevant drug label text chunks (per-drug + global)
4. **Context assembly** — Merges graph + vector context into a structured LLM prompt
5. **LLM generation** — Sends to Gemini API (or Ollama fallback) with pharmaceutical system prompt
6. **Response** — Returns answer + sources/evidence for transparency

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.13 (compatible 3.11+) |
| Package Manager | [uv](https://docs.astral.sh/uv/) (Rust-based, fast) |
| Knowledge Graph | Neo4j 5 Community (Docker) |
| Vector Store | ChromaDB (embedded, SQLite-backed) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2, 384 dims) |
| LLM Primary | Google Gemini API (gemini-2.5-flash, free tier) |
| LLM Backup | Ollama + Llama 3 / Mistral (local) |
| NLP | rapidfuzz (fuzzy entity matching) |
| API | FastAPI + Pydantic v2 |
| UI | Streamlit + streamlit-agraph (graph visualization) |
| Containers | Docker Compose (multi-stage, non-root, healthchecks) |
| CI/CD | GitHub Actions (lint + test matrix + Docker build) |
| Testing | pytest (142 tests, mocked services) |
| Linting | ruff (check + format) |

## Data Sources

| Source | Content | Scale |
|---|---|---|
| [FDA FAERS](https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html) | Adverse event reports (drugs, reactions, outcomes) | 816K reports, 2 quarters |
| [DailyMed](https://dailymed.nlm.nih.gov/dailymed/) | Drug labels (interactions, warnings, contraindications) | 88 drugs, 12 label sections |

## Quick Start

### Option 1: GitHub Codespaces (Fastest — zero setup)

Click the button below to open a fully configured environment in your browser:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/jmponcebe/PharmaGraphRAG?quickstart=1)

Everything happens automatically:
1. Dependencies install, Neo4j starts
2. Demo data loads into the knowledge graph + vector store (~3 min)
3. API + Streamlit UI start automatically
4. **Streamlit opens in a new browser tab — ready to chat!**

> **LLM note**: Without a Gemini API key, the system auto-detects and falls back to Ollama. For best results, set `GEMINI_API_KEY` in `.env` ([free key](https://aistudio.google.com/apikey)).

### Option 2: Local Quick Demo (~5 min)

```bash
# Clone and install
git clone https://github.com/jmponcebe/PharmaGraphRAG.git
cd PharmaGraphRAG
uv sync --extra dev --extra ui

# Configure
cp .env.example .env
# Optional: set GEMINI_API_KEY in .env (free: https://aistudio.google.com/apikey)

# Start Neo4j + load demo data
docker compose up -d neo4j
uv run python scripts/setup_demo.py   # ~3 min: loads graph + embeddings

# Start the app
uv run uvicorn pharmagraphrag.api.main:app --host 0.0.0.0 &
uv run streamlit run src/pharmagraphrag/ui/app.py
```

Demo includes 88 drugs with adverse events, interactions, and full label embeddings.

> **⚠️ Demo data disclaimer**: The demo dataset is a representative subset — it includes the top 15 adverse events per drug (by report count) and 88 drug labels. Results may differ from the full dataset (816K reports, 4,998 drugs). For production-grade analysis, run the [complete data pipeline](#option-3-full-data-pipeline-complete-dataset). This project is for educational/portfolio purposes only and should **not** be used for clinical decision-making.

### Option 3: Full Data Pipeline (complete dataset)

For the complete dataset (816K reports, 4,998 drugs):

#### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Docker and Docker Compose
- Gemini API key ([free tier](https://aistudio.google.com/apikey)) or Ollama installed locally

#### Installation

```bash
# Clone the repo
git clone https://github.com/jmponcebe/PharmaGraphRAG.git
cd PharmaGraphRAG

# Install all dependencies
uv sync --extra dev --extra ui

# Copy and configure environment variables
cp .env.example .env
# Edit .env: set GEMINI_API_KEY, adjust NEO4J_PASSWORD if needed
```

### Data Pipeline

```bash
# Start Neo4j
docker compose up -d neo4j

# 1. Download FAERS data (~135MB)
uv run python scripts/download_faers.py

# 2. Clean FAERS → Parquet
uv run python scripts/clean_faers.py

# 3. Fetch DailyMed drug labels
uv run python scripts/ingest_dailymed.py

# 4. Load knowledge graph into Neo4j
uv run python scripts/load_graph.py

# 5. Build vector store (ChromaDB)
uv run python scripts/load_vectorstore.py

# 6. Validate semantic search
uv run python scripts/validate_search.py
```

### Running the Application

#### Option A: Local (development)

```bash
# Start API (port 8000)
uv run uvicorn pharmagraphrag.api.main:app --reload

# Start UI (port 8501, in another terminal)
uv run streamlit run src/pharmagraphrag/ui/app.py
```

#### Option B: Docker Compose (production)

```bash
# All services (Neo4j + API + UI)
docker compose up --build -d

# With local Ollama LLM
docker compose --profile ollama up --build -d
```

| Service | URL | Description |
|---|---|---|
| Streamlit UI | <http://localhost:8501> | Chat interface with graph visualization |
| FastAPI | <http://localhost:8000> | REST API (Swagger at /docs) |
| FastAPI docs | <http://localhost:8000/docs> | Interactive API documentation |
| Neo4j Browser | <http://localhost:7474> | Knowledge graph browser |

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/query` | Ask a question, get a RAG-powered answer |
| `GET` | `/drug/{name}` | Get graph data for a specific drug |
| `GET` | `/health` | Health check (Neo4j + ChromaDB status) |

#### Example: Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the side effects of ibuprofen?"}'
```

#### Example: Drug Info

```bash
curl http://localhost:8000/drug/IBUPROFEN
```

## Development

### Testing

```bash
# Run all tests (142 tests)
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_engine.py -v

# Run with coverage
uv run pytest --cov=pharmagraphrag --cov-report=html
```

### Linting and Formatting

```bash
# Check lint
uv run ruff check src/ tests/

# Auto-fix lint issues
uv run ruff check src/ tests/ --fix

# Check formatting
uv run ruff format --check src/ tests/

# Auto-format
uv run ruff format src/ tests/

# Type check
uv run mypy src/
```

## Cloud Deployment (Free Tier)

The project supports a **distributed free-tier deployment**:

| Service | Platform | Cost |
|---|---|---|
| Knowledge Graph | [Neo4j Aura Free](https://neo4j.com/cloud/aura-free/) | $0 (200K nodes) |
| API + Vector Store | [Google Cloud Run](https://cloud.google.com/run) | $0 (free tier) |
| Chat UI | [Streamlit Community Cloud](https://streamlit.io/cloud) | $0 |

### Step 1: Neo4j Aura Free

1. Create a free instance at [console.neo4j.io](https://console.neo4j.io)
2. Save the connection URI and password
3. Migrate data from your local Neo4j:

```bash
uv run python scripts/migrate_neo4j.py \
    --source bolt://localhost:7687 \
    --source-password pharmagraphrag \
    --target neo4j+s://<id>.databases.neo4j.io \
    --target-password <aura-password>
```

Or load demo data directly into Aura:

```bash
NEO4J_URI=neo4j+s://<id>.databases.neo4j.io \
NEO4J_PASSWORD=<aura-password> \
uv run python scripts/setup_demo.py
```

### Step 2: Google Cloud Run (API)

```bash
# Build the image (requires data/chroma/ to be populated)
docker build -f docker/Dockerfile.cloudrun -t gcr.io/<project>/pharmagraphrag-api .

# Push to Google Container Registry
docker push gcr.io/<project>/pharmagraphrag-api

# Deploy
gcloud run deploy pharmagraphrag-api \
    --image gcr.io/<project>/pharmagraphrag-api \
    --platform managed --region us-central1 \
    --allow-unauthenticated \
    --set-env-vars NEO4J_URI=neo4j+s://<id>.databases.neo4j.io,NEO4J_USER=neo4j,NEO4J_PASSWORD=<pw>,GEMINI_API_KEY=<key>
```

### Step 3: Streamlit Community Cloud (UI)

1. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub repo
2. Configure:
   - **Main file**: `src/pharmagraphrag/ui/app.py`
   - **Requirements file**: `requirements-streamlit.txt`
3. Add secret in Streamlit Cloud settings: `API_URL = "https://pharmagraphrag-api-xxx.run.app"`

The Streamlit app auto-detects `API_URL` and switches to API mode (HTTP calls to Cloud Run instead of local imports).

## Project Structure

```
src/pharmagraphrag/
├── config.py               # Pydantic BaseSettings (Neo4j, LLM, ChromaDB, etc.)
├── data/                   # Data download, cleaning, ingestion
│   ├── download_faers.py       # Download FAERS quarterly ZIPs from FDA
│   ├── clean_faers.py          # Clean FAERS CSVs → Parquet (normalize, dedup)
│   └── ingest_dailymed.py      # Fetch drug labels from openFDA API → JSON
├── graph/                  # Neo4j schema, loading, queries
│   ├── schema.py               # Constraints + indexes (4 constraints, 5 indexes)
│   ├── loader.py               # Load FAERS + DailyMed into Neo4j (batch upserts)
│   └── queries.py              # Cypher query functions for retrieval
├── vectorstore/            # Embeddings, ChromaDB operations
│   ├── chunker.py              # Text chunking (1000 chars, 200 overlap)
│   ├── embedder.py             # sentence-transformers (all-MiniLM-L6-v2, 384 dims)
│   └── store.py                # ChromaDB add, search, format_context
├── engine/                 # GraphRAG query engine
│   ├── entity_extractor.py     # Drug name extraction (exact + fuzzy matching)
│   ├── retriever.py            # Dual retrieval (Neo4j graph + ChromaDB vector)
│   └── query_engine.py         # Orchestrator: extract → retrieve → prompt assembly
├── llm/                    # LLM integration
│   └── client.py               # Unified client: Gemini + Ollama with fallback
├── api/                    # REST API
│   ├── main.py                 # FastAPI app (POST /query, GET /drug, GET /health)
│   └── models.py               # Pydantic v2 request/response schemas
└── ui/                     # Chat interface
    ├── app.py                  # Streamlit app (chat, sidebar, settings)
    └── components.py           # Graph visualization, sources panel, drug explorer
```

## Knowledge Graph Schema

```cypher
(:Drug)-[:CAUSES {report_count}]->(:AdverseEvent)
(:Drug)-[:INTERACTS_WITH {source, description}]->(:Drug)
(:Drug)-[:HAS_OUTCOME {report_count}]->(:Outcome)
(:Drug)-[:BELONGS_TO]->(:DrugCategory)
```

| Node | Count | Source |
|---|---|---|
| Drug | 4,998 | FAERS + DailyMed |
| AdverseEvent | 6,863 | FAERS |
| Outcome | 7 | FAERS (Death, Hospitalization, etc.) |
| DrugCategory | 32 | DailyMed |

| Relationship | Count |
|---|---|
| CAUSES | 365,360 |
| HAS_OUTCOME | 15,759 |
| INTERACTS_WITH | 193 |
| BELONGS_TO | 47 |

## License

MIT

## Author

**Jose María Ponce Bernabé** — [GitHub](https://github.com/jmponcebe) · [LinkedIn](https://linkedin.com/in/jmponcebe)
