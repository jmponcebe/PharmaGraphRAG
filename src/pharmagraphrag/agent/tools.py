"""LangChain tools wrapping existing PharmaGraphRAG services.

Each tool exposes a specific capability of the existing engine
(graph queries, vector search) so the ReAct agent can decide
which to call and in what order.
"""

from __future__ import annotations

from langchain_core.tools import tool

from pharmagraphrag.graph import queries
from pharmagraphrag.vectorstore import store


@tool
def search_drug_info(drug_name: str) -> str:
    """Look up a drug in the Neo4j knowledge graph.

    Returns structured information: pharmacologic class, adverse events
    (with report counts), drug interactions, clinical outcomes, and
    categories. Use this when the user asks about a specific drug.

    Args:
        drug_name: Exact drug name (e.g. "ASPIRIN", "METFORMIN").
    """
    ctx = queries.get_drug_full_context(drug_name.upper())
    if not ctx or not ctx.get("drug_info"):
        return f"No information found for '{drug_name}' in the knowledge graph."
    return queries.format_graph_context(ctx)


@tool
def find_drugs_for_adverse_event(event_name: str, limit: int = 15) -> str:
    """Find which drugs are associated with a specific adverse event.

    Searches the FAERS knowledge graph for drugs reported to cause
    a given adverse event, ranked by report count.

    Args:
        event_name: Adverse event name (e.g. "HEPATOTOXICITY", "NAUSEA").
        limit: Max number of drugs to return (default 15).
    """
    results = queries.get_adverse_event_drugs(event_name.upper(), limit=limit)
    if not results:
        return f"No drugs found for adverse event '{event_name}'."
    lines = [f"Drugs associated with {event_name.upper()}:"]
    for r in results:
        lines.append(f"  - {r['drug']}: {r['report_count']} reports")
    return "\n".join(lines)


@tool
def search_drug_labels(query: str, drug_name: str = "", n_results: int = 5) -> str:
    """Search DailyMed drug label text using semantic (vector) search.

    Finds relevant passages from FDA drug labels about interactions,
    warnings, adverse reactions, contraindications, etc. Optionally
    filter results to a specific drug.

    Args:
        query: Natural language search query.
        drug_name: Optional drug name to filter results. Leave empty to search all.
        n_results: Number of results to return (default 5).
    """
    if drug_name:
        results = store.search_by_drug(query, drug_name.upper(), n_results=n_results)
    else:
        results = store.search(query, n_results=n_results)

    if not results:
        return "No relevant drug label passages found."

    return store.format_vector_context(results, max_chars=4000)


@tool
def list_drug_interactions(drug_name: str) -> str:
    """List known drug-drug interactions from the knowledge graph.

    Returns interaction partners with source and description from
    DailyMed labels.

    Args:
        drug_name: Drug name to look up interactions for.
    """
    interactions = queries.get_drug_interactions(drug_name.upper())
    if not interactions:
        return f"No known interactions found for '{drug_name}'."
    lines = [f"Known interactions for {drug_name.upper()}:"]
    for ix in interactions:
        partner = ix.get("drug", "unknown")
        source = ix.get("source", "")
        desc = ix.get("description", "")
        line = f"  - {partner}"
        if desc:
            line += f": {desc[:200]}"
        if source:
            line += f" (source: {source})"
        lines.append(line)
    return "\n".join(lines)


@tool
def search_drugs_by_name(query: str, limit: int = 10) -> str:
    """Search for drug names matching a partial query.

    Useful when the user mentions a drug but the exact name is unclear.
    Returns a list of matching drug names from the knowledge graph.

    Args:
        query: Partial drug name to search for.
        limit: Max results (default 10).
    """
    results = queries.search_drugs(query.upper(), limit=limit)
    if not results:
        return f"No drugs found matching '{query}'."
    return "Matching drugs: " + ", ".join(results)


# All tools for the agent
ALL_TOOLS = [
    search_drug_info,
    find_drugs_for_adverse_event,
    search_drug_labels,
    list_drug_interactions,
    search_drugs_by_name,
]
