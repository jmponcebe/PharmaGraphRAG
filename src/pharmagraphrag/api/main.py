"""FastAPI application — routes for the GraphRAG API.

Endpoints:
    POST /query     — Ask a question, get a RAG-powered answer.
    GET  /drug/{name} — Get graph data for a specific drug.
    GET  /health    — Health check.

Usage:
    uvicorn pharmagraphrag.api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from loguru import logger

from pharmagraphrag import __version__
from pharmagraphrag.api.models import (
    DrugInfoResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceInfo,
)

app = FastAPI(
    title="PharmaGraphRAG",
    description=(
        "GraphRAG API for querying drug interactions and adverse events "
        "using FDA FAERS data, DailyMed labels, and LLM-powered answers."
    ),
    version=__version__,
)


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Process a natural-language question through the GraphRAG pipeline.

    1. Extract drug entities from the question.
    2. Retrieve context from Neo4j graph + ChromaDB vectors.
    3. (Optionally) Generate an LLM answer.
    """
    from pharmagraphrag.engine.query_engine import process_query
    from pharmagraphrag.llm.client import generate_answer

    try:
        # Run the query engine (entity extraction + retrieval)
        result = process_query(
            req.question,
            use_graph=req.use_graph,
            use_vector=req.use_vector,
            n_vector_results=req.n_results,
        )

        # Build sources list
        sources: list[SourceInfo] = []

        # Graph sources
        for drug in result.context.drugs_found:
            sources.append(SourceInfo(
                type="graph",
                drug=drug,
                section="",
                snippet=f"Knowledge graph data for {drug}",
            ))

        # Vector sources
        for vr in result.context.vector_raw:
            meta = vr.get("metadata", {})
            sources.append(SourceInfo(
                type="vector",
                drug=meta.get("drug_name", ""),
                section=meta.get("section", ""),
                snippet=vr.get("text", "")[:200],
            ))

        # LLM answer (optional)
        answer = ""
        llm_model = ""
        llm_provider = ""
        error = None

        if req.use_llm:
            llm_resp = generate_answer(
                system_prompt=result.system_prompt,
                user_prompt=result.user_prompt,
            )
            answer = llm_resp.text
            llm_model = llm_resp.model
            llm_provider = llm_resp.provider
            if not llm_resp.ok:
                error = llm_resp.error

        return QueryResponse(
            question=req.question,
            answer=answer,
            drugs_extracted=result.entities.drugs,
            drugs_found_in_graph=result.context.drugs_found,
            has_graph_context=result.context.has_graph,
            has_vector_context=result.context.has_vector,
            sources=sources,
            llm_model=llm_model,
            llm_provider=llm_provider,
            error=error,
        )

    except Exception as exc:
        logger.error("Query failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# GET /drug/{name}
# ---------------------------------------------------------------------------


@app.get("/drug/{name}", response_model=DrugInfoResponse)
def get_drug(name: str) -> DrugInfoResponse:
    """Get complete graph information about a drug."""
    from pharmagraphrag.graph.queries import get_drug_full_context

    try:
        ctx = get_drug_full_context(name)

        drug_info = ctx.get("drug_info") or {}
        if not drug_info:
            raise HTTPException(
                status_code=404,
                detail=f"Drug '{name}' not found in the knowledge graph.",
            )

        return DrugInfoResponse(
            name=drug_info.get("name", name.upper()),
            generic_names=drug_info.get("generic_names") or [],
            brand_names=drug_info.get("brand_names") or [],
            category=drug_info.get("category", ""),
            route=drug_info.get("route", ""),
            adverse_events=ctx.get("adverse_events", []),
            interactions=ctx.get("interactions", []),
            outcomes=ctx.get("outcomes", []),
            categories=ctx.get("categories", []),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Drug lookup failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check — reports status of Neo4j and ChromaDB."""
    neo4j_status = "unknown"
    chroma_status = "unknown"

    # Check Neo4j
    try:
        from pharmagraphrag.graph.queries import search_drugs

        result = search_drugs("ASPIRIN", limit=1)
        neo4j_status = "ok" if result else "empty"
    except Exception as exc:  # noqa: BLE001
        neo4j_status = f"error: {exc}"

    # Check ChromaDB
    try:
        from pharmagraphrag.vectorstore.store import get_collection

        coll = get_collection()
        count = coll.count()
        chroma_status = f"ok ({count} docs)"
    except Exception as exc:  # noqa: BLE001
        chroma_status = f"error: {exc}"

    return HealthResponse(
        status="ok",
        version=__version__,
        neo4j=neo4j_status,
        chromadb=chroma_status,
    )
