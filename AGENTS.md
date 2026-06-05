# PharmaGraphRAG — Project Instructions

## Project Overview

GraphRAG system for querying drug interactions and adverse events using FDA data. Combines Neo4j knowledge graph with vector search (ChromaDB) and LLM-powered answers (Gemini API / Ollama), served via FastAPI and Streamlit chat UI.

**Demo**: <https://pharmagraphrag.streamlit.app> | **Repo**: <https://github.com/jmponcebe/PharmaGraphRAG>

## Data at a Glance

- **FAERS**: 2 quarters (2024Q3, 2024Q4) — 816K reports, 3.9M drug entries, 2.8M reactions
- **DailyMed**: 88 drugs via openFDA API — JSON labels
- **Knowledge Graph**: 4,998 Drug + 6,863 AdverseEvent + 7 Outcome + 32 DrugCategory nodes; 365K CAUSES + 15.8K HAS_OUTCOME + 193 INTERACTS_WITH + 47 BELONGS_TO relationships
- **Vector Store**: 5,654 text chunks, 384-dim embeddings (all-MiniLM-L6-v2), cosine similarity

## Architecture

Data flows: FDA FAERS (CSV) + DailyMed (API) → Data Pipeline (ingestion + cleaning) → Neo4j (KG, 11.9K nodes, 381K rels) + ChromaDB (vector, 5.6K chunks, 384-dim embeddings) → Query Engine (entity extraction + dual retrieval: graph traversal + vector search + context merging) → LLM (Gemini API primary, Ollama fallback) → FastAPI (REST) + Streamlit (Chat UI)

### Query Flow

**Classic mode** (`POST /query`): entity extraction (rapidfuzz) → graph retrieval (Cypher) + vector retrieval (ChromaDB) → prompt assembly → LLM generation → answer + sources.

**Agent mode** (`POST /agent/query`): LangGraph ReAct agent with 9 tools, autonomously selects tools, iterative reasoning, conversation memory via MemorySaver. Returns StructuredResponse (confidence, drugs_mentioned, follow_up_suggestions).

**Multi-agent mode** (`POST /agent/multi`): supervisor delegates to 3 sub-agents (Drug Expert: 4 tools, Safety Analyst: 4 tools, Literature Researcher: 2 tools). Supervisor synthesizes final answer.

## Tech Stack

Python 3.13, uv (package manager), Neo4j 5 Community, ChromaDB (embedded), sentence-transformers (all-MiniLM-L6-v2), Gemini API (Tier 1, `google-genai` SDK), Ollama (local fallback), LangGraph + LangChain, Langfuse (LLM tracing), RAGAS 0.4.3 (evaluation), rapidfuzz (entity matching), FastAPI + Pydantic v2, Streamlit, Docker multi-stage builds, GitHub Actions CI/CD, ruff (lint/format).

## Neo4j Schema

```cypher
(:Drug {name: string, pharmacologic_class: string?, source: string?})
(:AdverseEvent {name: string})
(:Outcome {code: string, name: string})
(:DrugCategory {name: string})

(:Drug)-[:CAUSES {report_count: int}]->(:AdverseEvent)
(:Drug)-[:INTERACTS_WITH {source: string, description: string?}]->(:Drug)
(:Drug)-[:HAS_OUTCOME {report_count: int}]->(:Outcome)
(:Drug)-[:BELONGS_TO]->(:DrugCategory)
```

## ChromaDB Schema

- Collection: `drug_labels` (cosine distance), embedding: all-MiniLM-L6-v2 (384 dims)
- Chunk size: 1000 chars, 200 overlap
- Metadata: drug_name, section, chunk_index, generic_names, brand_names, route
- 12 label sections: drug_interactions, adverse_reactions, warnings, contraindications, etc.

## Testing

263 tests, pytest, mocked LLM/Neo4j (never call real API in tests). CI matrix: Python 3.11 + 3.13.

## Evaluation (RAGAS)

RAGAS 0.4.3 with Gemini via OpenAI-compatible endpoint. 25-question curated testset across 8 types. Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall, Answer Correctness. Agent: tool selection accuracy (P/R/F1). Batch runner exports CSV.

## Key Design Decisions

1. **Neo4j over RDFLib**: more marketable, native graph traversal
2. **ChromaDB over Pinecone/Qdrant**: embedded, no extra infra, SQLite-backed
3. **Gemini API over OpenAI**: Tier 1 generous quotas (Flash 10K RPD). Ollama as local backup removes vendor lock-in
4. **Dual retrieval (graph + vector)**: graph gives structured relationships, vector gives semantic context. Combined > either alone
5. **rapidfuzz for entity extraction**: fuzzy matching (threshold=80) catches misspellings without LLM call
6. **LangGraph ReAct over manual StateGraph**: `create_react_agent` implements ReAct natively. Tools wrap existing services
7. **Singleton Neo4j driver**: single instance with connection pool. Critical for Aura where each new connection has high latency
8. **Synchronous FastAPI**: Neo4j driver is sync, async would add complexity without benefit
9. **sentence-transformers over OpenAI embeddings**: free, local, fast
10. **Per-model agent caching**: agents cached by model string in dicts, not singletons

## LLM Configuration

- Default: `gemini-2.5-flash`. Pro models (supervisor only): `gemini-2.5-pro`
- Temperature: 0.3, max_output_tokens: 2048
- Fallback chain: Gemini → Ollama → raw context → error
- Agent fallback: if agent hits rate limit → classic pipeline → Flash (separate quota) → raw context
- Response cache: in-memory LRU (max 50 entries)

## Cloud Deployment

| Service | Platform | URL |
|---|---|---|
| Chat UI | Streamlit Community Cloud | <https://pharmagraphrag.streamlit.app> |
| API + ChromaDB | Google Cloud Run | us-central1.run.app |
| Knowledge Graph | Neo4j Aura Free | Managed instance |

- **CD pipeline**: v* tags → GitHub Actions → Cloud Build → GCR → Cloud Run
- **Cloud Build**: downloads ChromaDB from GCS bucket → Docker build → deploy
- **Image**: CPU-only PyTorch, baked-in ChromaDB, pre-cached embedding model (~2.5 GB). Cold start ~50s, warm ~4.5s
- **Service account**: see `docs/deployment-reference.md` (gitignored)

### Deploy Safety

NEVER create version tags or trigger deployments without explicit user confirmation. Commits and pushes to main are fine; tags (v*) require user approval.

### gcloud PowerShell Gotchas

- `--set-env-vars KEY1=val1,KEY2=val2` does NOT work in PowerShell — commas get misinterpreted
- Use `--env-vars-file=file.yaml` instead (YAML with key: "value" pairs)
- `--set-env-vars` REPLACES all env vars. Use `--update-env-vars` to add/update without removing existing ones
- Always verify with `gcloud run revisions describe <rev> --format="yaml(spec.containers[0].env)"` after changes
- Delete the YAML file after use if it contains secrets

## Project-Specific Code Style

- Neo4j labels: PascalCase (Drug, AdverseEvent). Relationships: UPPER_SNAKE_CASE (CAUSES, INTERACTS_WITH)
- Max line length: 88 (ruff default)
- Type hints everywhere, Pydantic v2 for all data models, f-strings, pathlib.Path
- Docstrings: Google style
