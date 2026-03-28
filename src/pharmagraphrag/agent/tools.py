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

    IMPORTANT: Adverse events use MedDRA terminology (e.g. "HEPATOTOXICITY"
    not "liver damage", "RHABDOMYOLYSIS" not "muscle breakdown"). If you're
    unsure of the exact MedDRA term, use search_adverse_events first to find
    matching event names.

    Args:
        event_name: Adverse event name in MedDRA terminology (e.g. "HEPATOTOXICITY", "NAUSEA").
        limit: Max number of drugs to return (default 15).
    """
    results = queries.get_adverse_event_drugs(event_name.upper(), limit=limit)
    if not results:
        # Try substring search as fallback
        similar = queries.search_adverse_events(event_name.upper(), limit=5)
        if similar:
            suggestions = ", ".join(e["name"] for e in similar)
            return (
                f"No exact match for '{event_name}'. "
                f"Similar adverse events found: {suggestions}. "
                f"Try searching with one of these exact names."
            )
        return f"No drugs found for adverse event '{event_name}'."
    lines = [f"Drugs associated with {event_name.upper()}:"]
    for r in results:
        lines.append(f"  - {r['drug_name']}: {r['report_count']} reports")
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
        partner = ix.get("interacting_drug", "unknown")
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


@tool
def search_adverse_events(query: str, limit: int = 10) -> str:
    """Search for adverse event names matching a partial query.

    Use this to find the correct MedDRA terminology for an adverse event.
    For example, searching "LIVER" returns events like "HEPATOTOXICITY",
    "LIVER INJURY", "LIVER DISORDER", etc.

    Args:
        query: Partial event name to search for (e.g. "LIVER", "CARDIAC", "RENAL").
        limit: Max results (default 10).
    """
    results = queries.search_adverse_events(query.upper(), limit=limit)
    if not results:
        return f"No adverse events found matching '{query}'."
    lines = [f"Adverse events matching '{query}':"]
    for r in results:
        lines.append(f"  - {r['name']} ({r['total_reports']} total reports)")
    return "\n".join(lines)


@tool
def get_drug_outcomes(drug_name: str) -> str:
    """Get patient outcomes (hospitalization, death, disability, etc.) associated with a drug.

    Returns outcomes from FAERS reports ranked by frequency. Useful for
    assessing drug safety severity.

    Args:
        drug_name: Drug name (e.g. "WARFARIN", "METFORMIN").
    """
    outcomes = queries.get_drug_outcomes(drug_name.upper())
    if not outcomes:
        return f"No outcome data found for '{drug_name}'."
    lines = [f"Patient outcomes for {drug_name.upper()}:"]
    for o in outcomes:
        desc = o.get("outcome_description", o.get("outcome_code", "unknown"))
        code = o.get("outcome_code", "")
        count = o.get("report_count", 0)
        lines.append(f"  - {desc} ({code}): {count} reports")
    return "\n".join(lines)


@tool
def compare_drugs(drug_name_1: str, drug_name_2: str) -> str:
    """Compare two drugs side-by-side: adverse events, outcomes, interactions, and categories.

    Useful when the user asks which drug is safer, or wants to compare
    side effect profiles between alternatives.

    Args:
        drug_name_1: First drug name (e.g. "ASPIRIN").
        drug_name_2: Second drug name (e.g. "IBUPROFEN").
    """
    ctx1 = queries.get_drug_full_context(drug_name_1.upper())
    ctx2 = queries.get_drug_full_context(drug_name_2.upper())

    if not ctx1 or not ctx1.get("drug_info"):
        return f"Drug '{drug_name_1}' not found in the knowledge graph."
    if not ctx2 or not ctx2.get("drug_info"):
        return f"Drug '{drug_name_2}' not found in the knowledge graph."

    parts = []

    # Header
    parts.append(f"=== Comparison: {drug_name_1.upper()} vs {drug_name_2.upper()} ===\n")

    # Categories
    cat1 = ctx1.get("categories", [])
    cat2 = ctx2.get("categories", [])
    parts.append(f"{drug_name_1.upper()} categories: {', '.join(cat1) if cat1 else 'N/A'}")
    parts.append(f"{drug_name_2.upper()} categories: {', '.join(cat2) if cat2 else 'N/A'}\n")

    # Top adverse events
    parts.append("Top adverse events:")
    for name, ctx in [(drug_name_1.upper(), ctx1), (drug_name_2.upper(), ctx2)]:
        events = ctx.get("adverse_events", [])[:10]
        parts.append(f"  {name}:")
        for e in events:
            parts.append(f"    - {e['adverse_event']}: {e['report_count']} reports")
        if not events:
            parts.append("    (none)")

    # Outcomes
    parts.append("\nPatient outcomes:")
    for name, ctx in [(drug_name_1.upper(), ctx1), (drug_name_2.upper(), ctx2)]:
        outcomes = ctx.get("outcomes", [])
        parts.append(f"  {name}:")
        for o in outcomes:
            desc = o.get("outcome_description", o.get("outcome_code", ""))
            parts.append(f"    - {desc}: {o['report_count']} reports")
        if not outcomes:
            parts.append("    (none)")

    # Interactions between the two
    ix1 = ctx1.get("interactions", [])
    mutual = [i for i in ix1 if i.get("interacting_drug", "").upper() == drug_name_2.upper()]
    if mutual:
        parts.append(f"\nDirect interaction: {drug_name_1.upper()} ↔ {drug_name_2.upper()}")
        for i in mutual:
            desc = i.get("description", "No description")
            parts.append(f"  {desc[:300]}")
    else:
        parts.append(
            f"\nNo direct interaction found between {drug_name_1.upper()} and {drug_name_2.upper()}."
        )

    return "\n".join(parts)


@tool
def find_drugs_by_category(category: str, limit: int = 15) -> str:
    """Find all drugs belonging to a pharmacologic category.

    Useful when the user asks about drug classes (e.g. "NSAIDs",
    "beta blockers", "statins").

    Args:
        category: Category name or partial match (e.g. "NSAID", "STATIN").
        limit: Max drugs to return (default 15).
    """
    results = queries.get_drugs_by_category(category.upper(), limit=limit)
    if not results:
        return f"No drugs found in category matching '{category}'."
    lines = [f"Drugs in category matching '{category}':"]
    for r in results:
        lines.append(f"  - {r['drug_name']} (category: {r['category']})")
    return "\n".join(lines)


# All tools for the agent
ALL_TOOLS = [
    search_drug_info,
    find_drugs_for_adverse_event,
    search_drug_labels,
    list_drug_interactions,
    search_drugs_by_name,
    search_adverse_events,
    get_drug_outcomes,
    compare_drugs,
    find_drugs_by_category,
]
