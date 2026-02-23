"""Pydantic models for the FastAPI layer.

Defines request and response schemas for the REST API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Body for POST /query."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural-language question about drugs, interactions, or adverse events.",
        examples=["What are the side effects of ibuprofen?"],
    )
    use_graph: bool = Field(True, description="Include knowledge graph context.")
    use_vector: bool = Field(True, description="Include vector store context.")
    use_llm: bool = Field(True, description="Generate an LLM answer (False = retrieval only).")
    n_results: int = Field(5, ge=1, le=20, description="Max vector search results.")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class SourceInfo(BaseModel):
    """A single source used to build the answer."""

    type: str = Field(..., description="'graph' or 'vector'.")
    drug: str = Field("", description="Drug name this source relates to.")
    section: str = Field("", description="Label section name (vector only).")
    snippet: str = Field("", description="Short text excerpt.")


class QueryResponse(BaseModel):
    """Response for POST /query."""

    question: str
    answer: str = Field("", description="LLM-generated answer (empty if use_llm=False).")
    drugs_extracted: list[str] = Field(default_factory=list)
    drugs_found_in_graph: list[str] = Field(default_factory=list)
    has_graph_context: bool = False
    has_vector_context: bool = False
    sources: list[SourceInfo] = Field(default_factory=list)
    llm_model: str = ""
    llm_provider: str = ""
    error: str | None = None


class DrugInfoResponse(BaseModel):
    """Response for GET /drug/{name}."""

    name: str
    generic_names: list[str] = Field(default_factory=list)
    brand_names: list[str] = Field(default_factory=list)
    category: str = ""
    route: str = ""
    adverse_events: list[dict] = Field(default_factory=list)
    interactions: list[dict] = Field(default_factory=list)
    outcomes: list[dict] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = "ok"
    version: str = ""
    neo4j: str = "unknown"
    chromadb: str = "unknown"
