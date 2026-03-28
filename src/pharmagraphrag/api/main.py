"""FastAPI application — routes for the GraphRAG API.

Endpoints:
    POST /query       — Ask a question, get a RAG-powered answer.
    POST /agent/query — Ask a question using the ReAct agent.
    GET  /drug/{name} — Get graph data for a specific drug.
    GET  /health      — Health check.

Usage:
    uvicorn pharmagraphrag.api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from loguru import logger

from pharmagraphrag import __version__
from pharmagraphrag.api.models import (
    AgentQueryRequest,
    AgentQueryResponse,
    DrugInfoResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceInfo,
    ToolCallInfo,
    ToolResultInfo,
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
            sources.append(
                SourceInfo(
                    type="graph",
                    drug=drug,
                    section="",
                    snippet=f"Knowledge graph data for {drug}",
                )
            )

        # Vector sources
        for vr in result.context.vector_raw:
            meta = vr.get("metadata", {})
            sources.append(
                SourceInfo(
                    type="vector",
                    drug=meta.get("drug_name", ""),
                    section=meta.get("section", ""),
                    snippet=vr.get("text", "")[:200],
                )
            )

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
                # Fallback: present retrieved context as the answer
                if result.has_context:
                    answer = (
                        "⚠️ *LLM unavailable — showing retrieved data directly.*\n\n"
                        + result.context.graph_context
                    )
                    if result.context.has_vector:
                        answer += "\n\n---\n**Relevant drug label excerpts:**\n"
                        answer += result.context.vector_context[:2000]
                    llm_provider = "fallback"

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
# GET /drugs/search
# ---------------------------------------------------------------------------


@app.get("/drugs/search")
def search_drugs_endpoint(q: str, limit: int = 10) -> list[str]:
    """Search for drugs whose name contains the query string.

    Args:
        q: Partial drug name (min 2 characters).
        limit: Maximum number of results (default 10).

    Returns:
        List of matching drug names.
    """
    if len(q) < 2:
        return []

    from pharmagraphrag.graph.queries import search_drugs

    try:
        return search_drugs(q, limit=limit)
    except Exception as exc:
        logger.error("Drug search failed: {}", exc)
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
# POST /agent/query
# ---------------------------------------------------------------------------


@app.post("/agent/query", response_model=AgentQueryResponse)
def agent_query(req: AgentQueryRequest) -> AgentQueryResponse:
    """Process a question using the LangGraph ReAct agent.

    The agent autonomously decides which tools to call (graph queries,
    vector search, etc.) and generates a final answer.
    """
    from pharmagraphrag.agent.graph import run_agent

    try:
        result = run_agent(req.question, thread_id=req.session_id)

        # If the agent hit rate limits, fall back to classic pipeline
        if not result.ok and result.error and "rate limit" in result.error.lower():
            logger.info("Agent rate-limited, falling back to classic pipeline")
            return _agent_fallback_to_classic(req.question)

        return AgentQueryResponse(
            question=req.question,
            answer=result.answer,
            tool_calls=[ToolCallInfo(**tc) for tc in result.tool_calls],
            tool_results=[ToolResultInfo(**tr) for tr in result.tool_results],
            graph_data=result.graph_data,
            vector_data=result.vector_data,
            error=result.error,
        )
    except Exception as exc:
        logger.error("Agent query failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /agent/multi — Multi-agent supervisor
# ---------------------------------------------------------------------------


@app.post("/agent/multi", response_model=AgentQueryResponse)
def multi_agent_query(req: AgentQueryRequest) -> AgentQueryResponse:
    """Process a question using the multi-agent supervisor system.

    A supervisor agent delegates to specialized sub-agents (Drug Expert,
    Safety Analyst, Literature Researcher) and synthesizes their findings.
    """
    from pharmagraphrag.agent.multi import run_multi_agent

    try:
        result = run_multi_agent(req.question, thread_id=req.session_id)

        if not result.ok and result.error and "rate limit" in result.error.lower():
            logger.info("Multi-agent rate-limited, falling back to classic pipeline")
            return _agent_fallback_to_classic(req.question)

        return AgentQueryResponse(
            question=req.question,
            answer=result.answer,
            tool_calls=[ToolCallInfo(**tc) for tc in result.tool_calls],
            tool_results=[ToolResultInfo(**tr) for tr in result.tool_results],
            graph_data=result.graph_data,
            vector_data=result.vector_data,
            error=result.error,
        )
    except Exception as exc:
        logger.error("Multi-agent query failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Agent → Classic fallback
# ---------------------------------------------------------------------------


def _agent_fallback_to_classic(question: str) -> AgentQueryResponse:
    """Run the classic GraphRAG pipeline as a fallback when the agent is unavailable.

    Extracts entities, retrieves graph/vector context, and tries the LLM.
    If the LLM also fails, returns the raw retrieved context.
    """
    from pharmagraphrag.engine.query_engine import process_query
    from pharmagraphrag.llm.client import generate_answer

    result = process_query(question)

    # Try to generate an LLM answer
    answer = ""
    llm_failed = False
    if result.has_context:
        llm_resp = generate_answer(
            system_prompt=result.system_prompt,
            user_prompt=result.user_prompt,
        )
        if llm_resp.ok:
            answer = llm_resp.text
        else:
            llm_failed = True

    # Build fallback answer from context
    if not answer and result.has_context:
        answer = result.context.graph_context
        if result.context.has_vector:
            answer += "\n\n---\n**Relevant drug label excerpts:**\n"
            answer += result.context.vector_context[:2000]

    disclaimer = (
        "⚠️ *Agent unavailable (rate limit) — answered via classic pipeline.*\n\n"
        if not llm_failed
        else "⚠️ *LLM unavailable — showing retrieved data directly.*\n\n"
    )

    # Build graph_data for UI visualization
    graph_data: dict = {}
    for drug, ctx in result.context.graph_raw.items():
        if ctx and ctx.get("drug_info"):
            graph_data[drug] = ctx

    # Build vector_data for UI sources
    vector_data = result.context.vector_raw

    return AgentQueryResponse(
        question=question,
        answer=disclaimer + answer
        if answer
        else disclaimer + "No relevant data found for this query.",
        tool_calls=[],
        tool_results=[],
        graph_data=graph_data,
        vector_data=vector_data,
        error=None,  # Clear the error since we recovered
    )


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
    except Exception as exc:
        neo4j_status = f"error: {exc}"

    # Check ChromaDB
    try:
        from pharmagraphrag.vectorstore.store import get_collection

        coll = get_collection()
        count = coll.count()
        chroma_status = f"ok ({count} docs)"
    except Exception as exc:
        chroma_status = f"error: {exc}"

    return HealthResponse(
        status="ok",
        version=__version__,
        neo4j=neo4j_status,
        chromadb=chroma_status,
    )
