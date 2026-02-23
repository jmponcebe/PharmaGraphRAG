# PharmaGraphRAG

> **GraphRAG system for querying drug interactions and adverse events using FDA data.**

A question-answering system that combines a pharmaceutical knowledge graph (Neo4j) with Retrieval-Augmented Generation (RAG) to answer natural language questions about drug interactions and adverse events, grounded in real FDA data.

## Current Status

| Component | Status | Details |
|-----------|--------|---------|
| Data Pipeline | ✅ Complete | FAERS (2024Q3+Q4): 816K reports, 3.9M drug entries. DailyMed: 88 drugs |
| Knowledge Graph | ✅ Complete | 4,998 Drugs, 6,863 AdverseEvents, 365K CAUSES, 193 INTERACTS_WITH |
| Vector Store | ✅ Complete | 5,654 text chunks, 384-dim embeddings, cosine similarity search |
| Query Engine | 🔲 Pending | Entity extraction, graph+vector retrieval, context merging |
| LLM Integration | 🔲 Pending | Gemini API + Ollama fallback |
| API | 🔲 Pending | FastAPI endpoints |
| UI | 🔲 Pending | Streamlit dashboard |
| Tests | ✅ 64 passing | 29 data pipeline + 35 vector store |

## Example Questions

- *"¿Qué efectos adversos tiene el ibuprofeno?"*
- *"¿Qué fármacos interactúan con metformina?"*
- *"¿Hay reportes de eventos adversos cuando se combina warfarina con aspirina?"*

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

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.13 |
| Knowledge Graph | Neo4j (Docker) |
| Vector Store | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | Google Gemini API / Ollama (local fallback) |
| API | FastAPI |
| UI | Streamlit |
| Containers | Docker Compose |
| CI/CD | GitHub Actions |

## Data Sources

| Source | Content | Format |
|--------|---------|--------|
| [FDA FAERS](https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html) | Adverse event reports (drugs, reactions, outcomes) | CSV (quarterly) |
| [DailyMed](https://dailymed.nlm.nih.gov/dailymed/) | Drug labels (interactions, contraindications, warnings) | XML/API |

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Docker & Docker Compose
- Google Gemini API key (free tier) or Ollama installed locally

### Setup

```bash
# Clone the repo
git clone https://github.com/jmponcebe/PharmaGraphRAG.git
cd PharmaGraphRAG

# Install dependencies (requires uv: https://docs.astral.sh/uv/)
uv sync --extra dev --extra ui

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys

# Start infrastructure (Neo4j)
docker compose up -d neo4j

# 1. Download & clean FAERS data
uv run python scripts/download_faers.py
uv run python scripts/clean_faers.py

# 2. Fetch DailyMed drug labels
uv run python scripts/ingest_dailymed.py

# 3. Load knowledge graph
uv run python scripts/load_graph.py

# 4. Build vector store
uv run python scripts/load_vectorstore.py

# 5. Validate semantic search
uv run python scripts/validate_search.py
```

### Running (coming soon)

```bash
# Start API
uv run uvicorn pharmagraphrag.api.main:app --reload

# Start UI (in another terminal)
uv run streamlit run src/pharmagraphrag/ui/app.py
```

### Docker Compose (full stack)

```bash
docker compose up --build
```

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Lint & format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type check
uv run mypy src/
```

## Project Structure

```
src/pharmagraphrag/
├── config.py           # Pydantic BaseSettings (Neo4j, LLM, FAERS config)
├── data/               # Data download, cleaning, ingestion
│   ├── download_faers.py   # Download FAERS quarterly ZIPs from FDA
│   ├── clean_faers.py      # Clean FAERS CSVs → Parquet
│   └── ingest_dailymed.py  # Fetch drug labels from openFDA API
├── graph/              # Neo4j schema, loading, queries
│   ├── schema.py           # Constraints + indexes
│   ├── loader.py           # Load FAERS + DailyMed into Neo4j
│   └── queries.py          # Cypher query functions for retrieval
├── vectorstore/        # Embeddings, ChromaDB operations
│   ├── chunker.py          # Text chunking (1000 chars, 200 overlap)
│   ├── embedder.py         # sentence-transformers embeddings (384 dims)
│   └── store.py            # ChromaDB add, search, format_context
├── engine/             # TODO: GraphRAG query engine
├── llm/                # TODO: LLM integration (Gemini, Ollama)
├── api/                # TODO: FastAPI endpoints
└── ui/                 # TODO: Streamlit dashboard
```

## Knowledge Graph Schema

```
(:Drug)-[:CAUSES {report_count}]->(:AdverseEvent)
(:Drug)-[:INTERACTS_WITH {source, description}]->(:Drug)
(:Drug)-[:HAS_OUTCOME {report_count}]->(:Outcome)
(:Drug)-[:BELONGS_TO]->(:DrugCategory)
```

| Node | Count | Source |
|------|-------|--------|
| Drug | 4,998 | FAERS + DailyMed |
| AdverseEvent | 6,863 | FAERS |
| Outcome | 7 | FAERS (Death, Hospitalization, etc.) |
| DrugCategory | 32 | DailyMed |

| Relationship | Count |
|-------------|-------|
| CAUSES | 365,360 |
| HAS_OUTCOME | 15,759 |
| INTERACTS_WITH | 193 |
| BELONGS_TO | 47 |

## License

MIT

## Author

**Jose María Ponce Bernabé** — [GitHub](https://github.com/jmponcebe) · [LinkedIn](https://linkedin.com/in/jmponcebe)
