"""Neo4j knowledge graph schema definition.

Defines the graph schema with constraints, indexes, and node/relationship types.

Schema:
    (:Drug {name, generic_names, brand_names, category, route})
    (:AdverseEvent {name})
    (:Outcome {code, description})
    (:DrugCategory {name})

    (:Drug)-[:CAUSES {report_count, role_cod}]->(:AdverseEvent)
    (:Drug)-[:INTERACTS_WITH {description, source}]->(:Drug)
    (:Drug)-[:BELONGS_TO]->(:DrugCategory)
    (:Drug)-[:HAS_OUTCOME {report_count}]->(:Outcome)

Usage:
    uv run python -m pharmagraphrag.graph.schema
"""

from __future__ import annotations

import sys

from loguru import logger
from neo4j import GraphDatabase

from pharmagraphrag.config import get_settings

# Constraints ensure uniqueness and create implicit indexes
CONSTRAINTS = [
    "CREATE CONSTRAINT drug_name IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE",
    "CREATE CONSTRAINT adverse_event_name IF NOT EXISTS FOR (ae:AdverseEvent) REQUIRE ae.name IS UNIQUE",
    "CREATE CONSTRAINT outcome_code IF NOT EXISTS FOR (o:Outcome) REQUIRE o.code IS UNIQUE",
    "CREATE CONSTRAINT drug_category_name IF NOT EXISTS FOR (dc:DrugCategory) REQUIRE dc.name IS UNIQUE",
]

# Additional indexes for common query patterns
INDEXES = [
    "CREATE INDEX drug_category_idx IF NOT EXISTS FOR (d:Drug) ON (d.category)",
    "CREATE INDEX drug_route_idx IF NOT EXISTS FOR (d:Drug) ON (d.route)",
    "CREATE INDEX adverse_event_name_idx IF NOT EXISTS FOR (ae:AdverseEvent) ON (ae.name)",
    "CREATE TEXT INDEX drug_name_text_idx IF NOT EXISTS FOR (d:Drug) ON (d.name)",
    "CREATE TEXT INDEX adverse_event_text_idx IF NOT EXISTS FOR (ae:AdverseEvent) ON (ae.name)",
]


def create_schema(uri: str | None = None, user: str | None = None,
                  password: str | None = None) -> None:
    """Create the Neo4j schema with constraints and indexes.

    Args:
        uri: Neo4j bolt URI. Defaults to settings.
        user: Neo4j username. Defaults to settings.
        password: Neo4j password. Defaults to settings.
    """
    settings = get_settings()
    uri = uri or settings.neo4j_uri
    user = user or settings.neo4j_user
    password = password or settings.neo4j_password

    logger.info(f"Connecting to Neo4j at {uri}")

    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        driver.verify_connectivity()
        logger.success("Connected to Neo4j")

        with driver.session() as session:
            # Create constraints
            for constraint in CONSTRAINTS:
                session.run(constraint)
                logger.info(f"  Created constraint: {constraint.split('FOR')[0].strip()}")

            # Create indexes
            for index in INDEXES:
                session.run(index)
                logger.info(f"  Created index: {index.split('FOR')[0].strip()}")

        logger.success("Schema created successfully")

    finally:
        driver.close()


def drop_all_data(uri: str | None = None, user: str | None = None,
                  password: str | None = None) -> None:
    """Drop all nodes and relationships (for development/testing).

    Args:
        uri: Neo4j bolt URI.
        user: Neo4j username.
        password: Neo4j password.
    """
    settings = get_settings()
    uri = uri or settings.neo4j_uri
    user = user or settings.neo4j_user
    password = password or settings.neo4j_password

    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session() as session:
            # Delete in batches to avoid memory issues
            result = session.run(
                "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(*) AS deleted"
            )
            total = 0
            deleted = result.single()["deleted"]
            total += deleted

            while deleted > 0:
                result = session.run(
                    "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(*) AS deleted"
                )
                deleted = result.single()["deleted"]
                total += deleted

            logger.warning(f"Deleted {total} nodes from Neo4j")

    finally:
        driver.close()


def get_schema_info(uri: str | None = None, user: str | None = None,
                    password: str | None = None) -> dict:
    """Get current schema information from Neo4j.

    Returns:
        Dictionary with node labels, relationship types, and counts.
    """
    settings = get_settings()
    uri = uri or settings.neo4j_uri
    user = user or settings.neo4j_user
    password = password or settings.neo4j_password

    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session() as session:
            # Node counts by label
            node_counts = {}
            result = session.run("CALL db.labels() YIELD label RETURN label")
            for record in result:
                label = record["label"]
                count_result = session.run(
                    f"MATCH (n:`{label}`) RETURN count(n) AS count"
                )
                node_counts[label] = count_result.single()["count"]

            # Relationship counts by type
            rel_counts = {}
            result = session.run(
                "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
            )
            for record in result:
                rel_type = record["relationshipType"]
                count_result = session.run(
                    f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) AS count"
                )
                rel_counts[rel_type] = count_result.single()["count"]

            return {"nodes": node_counts, "relationships": rel_counts}

    finally:
        driver.close()


def main() -> None:
    """CLI entry point for schema creation."""
    import argparse

    parser = argparse.ArgumentParser(description="Create Neo4j schema")
    parser.add_argument("--drop", action="store_true", help="Drop all data first")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    if args.drop:
        logger.warning("Dropping all existing data...")
        drop_all_data()

    create_schema()

    info = get_schema_info()
    logger.info(f"Schema info: {info}")


if __name__ == "__main__":
    main()
