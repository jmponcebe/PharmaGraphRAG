"""Dual retriever: knowledge graph + vector store.

Given extracted entities and/or a raw query, retrieves context from
both Neo4j (graph relationships) and ChromaDB (text chunks), then
formats each into text blocks ready for the LLM prompt.

Usage:
    from pharmagraphrag.engine.retriever import retrieve_context
    ctx = retrieve_context(drugs=["IBUPROFEN"], query="side effects of ibuprofen")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class RetrievedContext:
    """Combined context from graph and vector retrieval."""

    graph_context: str = ""
    """Formatted text from the knowledge graph."""

    vector_context: str = ""
    """Formatted text from the vector store."""

    graph_raw: dict[str, Any] = field(default_factory=dict)
    """Raw graph data (per drug) for programmatic access."""

    vector_raw: list[dict[str, Any]] = field(default_factory=list)
    """Raw vector search results."""

    drugs_found: list[str] = field(default_factory=list)
    """Drugs that had graph data."""

    @property
    def has_graph(self) -> bool:
        """True if graph context is non-empty."""
        return bool(self.graph_context.strip())

    @property
    def has_vector(self) -> bool:
        """True if vector context is non-empty."""
        return bool(self.vector_context.strip())

    @property
    def is_empty(self) -> bool:
        """True if neither source returned useful context."""
        return not self.has_graph and not self.has_vector


# ---------------------------------------------------------------------------
# Graph retrieval
# ---------------------------------------------------------------------------


def _retrieve_graph(drugs: list[str]) -> tuple[str, dict[str, Any], list[str]]:
    """Retrieve and format graph context for a list of drugs.

    Args:
        drugs: List of uppercase drug names.

    Returns:
        Tuple of (formatted_text, raw_data_per_drug, drugs_with_data).
    """
    try:
        from pharmagraphrag.graph.queries import (
            format_graph_context,
            get_drug_full_context,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Graph module unavailable: {}", exc)
        return "", {}, []

    parts: list[str] = []
    raw: dict[str, Any] = {}
    found: list[str] = []

    for drug in drugs:
        try:
            ctx = get_drug_full_context(drug)
            # Check if any data was returned
            has_data = any([
                ctx.get("drug_info"),
                ctx.get("adverse_events"),
                ctx.get("interactions"),
                ctx.get("outcomes"),
                ctx.get("categories"),
            ])
            if has_data:
                formatted = format_graph_context(ctx)
                parts.append(formatted)
                raw[drug] = ctx
                found.append(drug)
                logger.debug("Graph context for {}: {} chars", drug, len(formatted))
            else:
                logger.debug("No graph data for {}", drug)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error retrieving graph context for {}: {}", drug, exc)

    return "\n\n---\n\n".join(parts), raw, found


# ---------------------------------------------------------------------------
# Vector retrieval
# ---------------------------------------------------------------------------


def _retrieve_vector(
    query: str,
    drugs: list[str] | None = None,
    n_results: int = 5,
    max_chars: int = 4000,
) -> tuple[str, list[dict[str, Any]]]:
    """Retrieve and format vector context.

    If specific drugs are provided, searches per-drug and merges results.
    Otherwise, does a global search.

    Args:
        query: User question.
        drugs: Optional list of drug names to scope the search.
        n_results: Max results per search.
        max_chars: Max total chars in formatted context.

    Returns:
        Tuple of (formatted_text, raw_results).
    """
    try:
        from pharmagraphrag.vectorstore.store import (
            format_vector_context,
            search,
            search_by_drug,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Vector store unavailable: {}", exc)
        return "", []

    all_results: list[dict[str, Any]] = []

    try:
        if drugs:
            # Search per-drug and merge results
            per_drug = max(2, n_results // len(drugs)) if drugs else n_results
            for drug in drugs:
                results = search_by_drug(query, drug, n_results=per_drug)
                all_results.extend(results)
                logger.debug(
                    "Vector search for '{}' scoped to {}: {} results",
                    query[:40], drug, len(results),
                )

            # Also do a global search to catch cross-drug info
            global_results = search(query, n_results=max(2, n_results // 2))
            # Add global results not already in per-drug results
            existing_ids = {r["id"] for r in all_results}
            for r in global_results:
                if r["id"] not in existing_ids:
                    all_results.append(r)
        else:
            # No drugs — pure semantic search
            all_results = search(query, n_results=n_results)
            logger.debug(
                "Global vector search for '{}': {} results",
                query[:40], len(all_results),
            )

        # Sort by distance (lower = better)
        all_results.sort(key=lambda r: r.get("distance", 999))

        # Deduplicate by id
        seen_ids: set[str] = set()
        unique: list[dict[str, Any]] = []
        for r in all_results:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                unique.append(r)

        # Trim to n_results
        unique = unique[:n_results]

        formatted = format_vector_context(unique, max_chars=max_chars)
        return formatted, unique

    except Exception as exc:  # noqa: BLE001
        logger.warning("Vector search failed: {}", exc)
        return "", []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def retrieve_context(
    *,
    drugs: list[str] | None = None,
    query: str = "",
    n_vector_results: int = 5,
    max_vector_chars: int = 4000,
    use_graph: bool = True,
    use_vector: bool = True,
) -> RetrievedContext:
    """Retrieve context from both graph and vector stores.

    Args:
        drugs: Extracted drug names (uppercase).
        query: Original user question.
        n_vector_results: Number of vector search results.
        max_vector_chars: Max characters for vector context text.
        use_graph: Enable graph retrieval.
        use_vector: Enable vector retrieval.

    Returns:
        RetrievedContext with both graph and vector context.
    """
    drugs = drugs or []
    ctx = RetrievedContext()

    # Graph retrieval (needs drug names)
    if use_graph and drugs:
        graph_text, graph_raw, found = _retrieve_graph(drugs)
        ctx.graph_context = graph_text
        ctx.graph_raw = graph_raw
        ctx.drugs_found = found

    # Vector retrieval (can work with or without drug names)
    if use_vector and query:
        vector_text, vector_raw = _retrieve_vector(
            query=query,
            drugs=drugs if drugs else None,
            n_results=n_vector_results,
            max_chars=max_vector_chars,
        )
        ctx.vector_context = vector_text
        ctx.vector_raw = vector_raw

    logger.info(
        "Retrieved context — graph: {} chars, vector: {} chars, drugs found: {}",
        len(ctx.graph_context),
        len(ctx.vector_context),
        ctx.drugs_found,
    )

    return ctx
