"""Cypher query functions for the Neo4j knowledge graph.

Provides reusable query functions for the GraphRAG query engine.
Each function returns structured data from the knowledge graph.

Usage:
    from pharmagraphrag.graph.queries import get_drug_adverse_events
    events = get_drug_adverse_events("IBUPROFEN")
"""

from __future__ import annotations

from neo4j import Driver, GraphDatabase

from pharmagraphrag.config import get_settings


def _get_driver() -> Driver:
    """Create a Neo4j driver from settings."""
    settings = get_settings()
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


def get_drug_info(drug_name: str) -> dict | None:
    """Get comprehensive information about a drug.

    Args:
        drug_name: Drug name (case-insensitive).

    Returns:
        Dictionary with drug properties, or None if not found.
    """
    driver = _get_driver()
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (d:Drug)
                WHERE toUpper(d.name) = toUpper($name)
                RETURN d {
                    .name, .generic_names, .brand_names,
                    .category, .route, .role_cod
                } AS drug
                """,
                name=drug_name,
            )
            record = result.single()
            return dict(record["drug"]) if record else None
    finally:
        driver.close()


def get_drug_adverse_events(drug_name: str, limit: int = 20) -> list[dict]:
    """Get adverse events caused by a drug, ordered by report count.

    Args:
        drug_name: Drug name (case-insensitive).
        limit: Maximum number of results.

    Returns:
        List of dicts with adverse event name and report count.
    """
    driver = _get_driver()
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (d:Drug)-[r:CAUSES]->(ae:AdverseEvent)
                WHERE toUpper(d.name) = toUpper($name)
                RETURN ae.name AS adverse_event,
                       r.report_count AS report_count
                ORDER BY r.report_count DESC
                LIMIT $limit
                """,
                name=drug_name,
                limit=limit,
            )
            return [dict(record) for record in result]
    finally:
        driver.close()


def get_drug_interactions(drug_name: str) -> list[dict]:
    """Get drugs that interact with the given drug.

    Args:
        drug_name: Drug name (case-insensitive).

    Returns:
        List of dicts with interacting drug name and description.
    """
    driver = _get_driver()
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (d:Drug)-[r:INTERACTS_WITH]-(other:Drug)
                WHERE toUpper(d.name) = toUpper($name)
                RETURN other.name AS interacting_drug,
                       r.description AS description,
                       r.source AS source
                """,
                name=drug_name,
            )
            return [dict(record) for record in result]
    finally:
        driver.close()


def get_drug_outcomes(drug_name: str) -> list[dict]:
    """Get patient outcomes associated with a drug.

    Args:
        drug_name: Drug name (case-insensitive).

    Returns:
        List of dicts with outcome code, description, and report count.
    """
    driver = _get_driver()
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (d:Drug)-[r:HAS_OUTCOME]->(o:Outcome)
                WHERE toUpper(d.name) = toUpper($name)
                RETURN o.code AS outcome_code,
                       o.description AS outcome_description,
                       r.report_count AS report_count
                ORDER BY r.report_count DESC
                """,
                name=drug_name,
            )
            return [dict(record) for record in result]
    finally:
        driver.close()


def get_drug_category(drug_name: str) -> list[str]:
    """Get pharmacologic categories for a drug.

    Args:
        drug_name: Drug name (case-insensitive).

    Returns:
        List of category names.
    """
    driver = _get_driver()
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (d:Drug)-[:BELONGS_TO]->(dc:DrugCategory)
                WHERE toUpper(d.name) = toUpper($name)
                RETURN dc.name AS category
                """,
                name=drug_name,
            )
            return [record["category"] for record in result]
    finally:
        driver.close()


def get_adverse_event_drugs(event_name: str, limit: int = 20) -> list[dict]:
    """Get drugs that cause a specific adverse event.

    Args:
        event_name: Adverse event name (case-insensitive).
        limit: Maximum number of results.

    Returns:
        List of dicts with drug name and report count.
    """
    driver = _get_driver()
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (d:Drug)-[r:CAUSES]->(ae:AdverseEvent)
                WHERE toUpper(ae.name) = toUpper($name)
                RETURN d.name AS drug_name,
                       r.report_count AS report_count
                ORDER BY r.report_count DESC
                LIMIT $limit
                """,
                name=event_name,
                limit=limit,
            )
            return [dict(record) for record in result]
    finally:
        driver.close()


def search_drugs(query: str, limit: int = 10) -> list[str]:
    """Search for drugs by partial name match.

    Args:
        query: Search query (partial name).
        limit: Maximum results.

    Returns:
        List of matching drug names.
    """
    driver = _get_driver()
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (d:Drug)
                WHERE toUpper(d.name) CONTAINS toUpper($query)
                RETURN d.name AS name
                ORDER BY d.name
                LIMIT $limit
                """,
                query=query,
                limit=limit,
            )
            return [record["name"] for record in result]
    finally:
        driver.close()


def get_drug_full_context(drug_name: str) -> dict:
    """Get complete graph context for a drug (for RAG).

    Combines adverse events, interactions, outcomes, and categories
    into a single structured context.

    Args:
        drug_name: Drug name (case-insensitive).

    Returns:
        Dictionary with all graph context for the drug.
    """
    return {
        "drug_info": get_drug_info(drug_name),
        "adverse_events": get_drug_adverse_events(drug_name, limit=15),
        "interactions": get_drug_interactions(drug_name),
        "outcomes": get_drug_outcomes(drug_name),
        "categories": get_drug_category(drug_name),
    }


def format_graph_context(context: dict) -> str:
    """Format graph context as human-readable text for LLM prompt.

    Args:
        context: Output from get_drug_full_context().

    Returns:
        Formatted text string.
    """
    parts = []

    drug_info = context.get("drug_info")
    if drug_info:
        name = drug_info.get("name", "Unknown")
        brands = drug_info.get("brand_names", [])
        route = drug_info.get("route", "")
        parts.append(f"Drug: {name}")
        if brands:
            parts.append(f"Brand names: {', '.join(brands)}")
        if route:
            parts.append(f"Route: {route}")

    categories = context.get("categories", [])
    if categories:
        parts.append(f"Pharmacologic class: {', '.join(categories)}")

    events = context.get("adverse_events", [])
    if events:
        parts.append("\nAdverse events (by report count):")
        for e in events:
            parts.append(f"  - {e['adverse_event']}: {e['report_count']} reports")

    interactions = context.get("interactions", [])
    if interactions:
        parts.append("\nDrug interactions:")
        for i in interactions:
            desc = i.get("description", "")
            parts.append(f"  - Interacts with {i['interacting_drug']}: {desc[:200]}")

    outcomes = context.get("outcomes", [])
    if outcomes:
        parts.append("\nPatient outcomes:")
        for o in outcomes:
            parts.append(
                f"  - {o['outcome_description']} ({o['outcome_code']}): {o['report_count']} reports"
            )

    return "\n".join(parts)
