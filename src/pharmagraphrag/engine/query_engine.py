"""GraphRAG query engine — main orchestrator.

Coordinates entity extraction, dual retrieval (graph + vector), and
prompt assembly. The result is a :class:`QueryResult` ready to be
fed to the LLM module.

Flow:
    User question
        → Entity extraction   (entity_extractor)
        → Graph retrieval      (retriever → Neo4j)
        → Vector retrieval     (retriever → ChromaDB)
        → Prompt assembly      (merge graph + vector context)
        → QueryResult          (ready for LLM)

Usage:
    from pharmagraphrag.engine.query_engine import process_query
    result = process_query("What are the side effects of ibuprofen?")
    print(result.prompt)  # ready for the LLM
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from pharmagraphrag.engine.entity_extractor import ExtractedEntities, extract_entities
from pharmagraphrag.engine.retriever import RetrievedContext, retrieve_context

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a pharmaceutical knowledge assistant. Answer the user's question "
    "about drug interactions and adverse events based ONLY on the provided "
    "context.\n\n"
    "Rules:\n"
    "1. Be factual — cite specific drugs, adverse events, and report counts "
    "from the context.\n"
    "2. If the context does not contain enough information to answer, say so "
    "explicitly.\n"
    "3. Organise your answer with bullet points or numbered lists when "
    "appropriate.\n"
    "4. Mention the data source (FAERS reports, DailyMed label) when "
    "relevant.\n"
)

CONTEXT_TEMPLATE = (
    "GRAPH CONTEXT (structured relationships from the knowledge graph):\n"
    "{graph_context}\n\n"
    "TEXT CONTEXT (from drug labels via semantic search):\n"
    "{text_context}\n"
)

USER_TEMPLATE = "USER QUESTION: {question}\n"


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    """Output of the query engine, ready for LLM consumption."""

    question: str
    """Original user question."""

    entities: ExtractedEntities = field(default_factory=ExtractedEntities)
    """Entities extracted from the question."""

    context: RetrievedContext = field(default_factory=RetrievedContext)
    """Retrieved context from graph + vector."""

    system_prompt: str = ""
    """System prompt for the LLM."""

    user_prompt: str = ""
    """User prompt (context + question) for the LLM."""

    @property
    def prompt(self) -> str:
        """Full prompt as a single string (system + user)."""
        return f"{self.system_prompt}\n\n{self.user_prompt}"

    @property
    def has_context(self) -> bool:
        """True if any context (graph or vector) was retrieved."""
        return not self.context.is_empty

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a dictionary (for API responses)."""
        return {
            "question": self.question,
            "drugs_extracted": self.entities.drugs,
            "adverse_events_extracted": self.entities.adverse_events,
            "drugs_found_in_graph": self.context.drugs_found,
            "has_graph_context": self.context.has_graph,
            "has_vector_context": self.context.has_vector,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
        }


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def _build_context_text(context: RetrievedContext) -> str:
    """Build the context section of the prompt."""
    graph_text = context.graph_context or "No structured graph data available."
    vector_text = context.vector_context or "No relevant text context found."
    return CONTEXT_TEMPLATE.format(
        graph_context=graph_text,
        text_context=vector_text,
    )


def _build_user_prompt(question: str, context: RetrievedContext) -> str:
    """Build the user prompt with context and question."""
    context_text = _build_context_text(context)
    user_q = USER_TEMPLATE.format(question=question)
    return f"{context_text}\n{user_q}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process_query(
    question: str,
    *,
    use_graph: bool = True,
    use_vector: bool = True,
    use_neo4j_entities: bool = True,
    fuzzy_match: bool = True,
    n_vector_results: int = 5,
    max_vector_chars: int = 4000,
) -> QueryResult:
    """Process a user question through the full GraphRAG pipeline.

    Steps:
        1. Extract entities (drugs, adverse events) from the question.
        2. Retrieve graph context from Neo4j for identified drugs.
        3. Retrieve vector context from ChromaDB.
        4. Assemble the LLM prompt.

    Args:
        question: Natural-language user question.
        use_graph: Enable graph retrieval.
        use_vector: Enable vector retrieval.
        use_neo4j_entities: Use Neo4j for entity catalogue.
        fuzzy_match: Enable fuzzy entity matching.
        n_vector_results: Number of vector search results.
        max_vector_chars: Max characters for vector context.

    Returns:
        QueryResult with assembled prompt and metadata.
    """
    logger.info("Processing query: '{}'", question[:100])

    # 1. Entity extraction
    entities = extract_entities(
        question,
        use_neo4j=use_neo4j_entities,
        fuzzy=fuzzy_match,
    )
    logger.info(
        "Entities — drugs: {}, adverse_events: {}",
        entities.drugs,
        entities.adverse_events,
    )

    # 2. Dual retrieval
    context = retrieve_context(
        drugs=entities.drugs,
        query=question,
        n_vector_results=n_vector_results,
        max_vector_chars=max_vector_chars,
        use_graph=use_graph,
        use_vector=use_vector,
    )

    # 3. Prompt assembly
    system_prompt = SYSTEM_PROMPT
    user_prompt = _build_user_prompt(question, context)

    result = QueryResult(
        question=question,
        entities=entities,
        context=context,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    logger.info(
        "Query result — context: {} chars, drugs found: {}",
        len(result.user_prompt),
        context.drugs_found,
    )

    return result
