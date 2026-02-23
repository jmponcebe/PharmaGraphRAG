# PharmaGraphRAG

> **GraphRAG system for querying drug interactions and adverse events using FDA data.**

A question-answering system that combines a pharmaceutical knowledge graph (Neo4j) with Retrieval-Augmented Generation (RAG) to answer natural language questions about drug interactions and adverse events, grounded in real FDA data.

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

# Install dependencies
uv sync

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys

# Start infrastructure (Neo4j)
docker compose up -d neo4j

# Run data pipeline
uv run python -m pharmagraphrag.data.download_faers
uv run python -m pharmagraphrag.data.clean_faers
uv run python -m pharmagraphrag.data.ingest_dailymed

# Load knowledge graph
uv run python -m pharmagraphrag.graph.load

# Build vector store
uv run python -m pharmagraphrag.vectorstore.build

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
├── data/           # Data download, cleaning, ingestion
├── graph/          # Neo4j schema, loading, queries
├── vectorstore/    # Embeddings, ChromaDB operations
├── engine/         # GraphRAG query engine
├── llm/            # LLM integration (Gemini, Ollama)
├── api/            # FastAPI endpoints
└── ui/             # Streamlit dashboard
```

## License

MIT

## Author

**Jose María Ponce Bernabé** — [GitHub](https://github.com/jmponcebe) · [LinkedIn](https://linkedin.com/in/jmponcebe)
