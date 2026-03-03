#!/usr/bin/env python3
"""Migrate Neo4j data between instances (e.g. local → Aura Free).

Exports all nodes and relationships from a source Neo4j instance and
imports them into a target instance via batched Cypher UNWIND statements.

Usage:
    # Migrate local → Aura Free
    uv run python scripts/migrate_neo4j.py \
        --source bolt://localhost:7687 \
        --source-user neo4j \
        --source-password pharmagraphrag \
        --target neo4j+s://<id>.databases.neo4j.io \
        --target-user neo4j \
        --target-password <aura-password>

    # Alternatively, load demo data directly into Aura (no source needed)
    NEO4J_URI=neo4j+s://<id>.databases.neo4j.io \
    NEO4J_USER=neo4j \
    NEO4J_PASSWORD=<aura-password> \
    uv run python scripts/setup_demo.py
"""

from __future__ import annotations

import argparse
import time

from loguru import logger
from neo4j import GraphDatabase

BATCH_SIZE = 500


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _export_nodes(session, label: str) -> list[dict]:
    """Export all nodes of a given label."""
    result = session.run(f"MATCH (n:{label}) RETURN properties(n) AS props")
    nodes = [record["props"] for record in result]
    logger.info(f"  Exported {len(nodes):,} {label} nodes")
    return nodes


def _export_relationships(
    session, rel_type: str, src_label: str, tgt_label: str, src_key: str, tgt_key: str
) -> list[dict]:
    """Export all relationships of a given type."""
    query = f"""
    MATCH (a:{src_label})-[r:{rel_type}]->(b:{tgt_label})
    RETURN a.{src_key} AS src, b.{tgt_key} AS tgt, properties(r) AS props
    """
    result = session.run(query)
    rels = [{"src": r["src"], "tgt": r["tgt"], "props": r["props"]} for r in result]
    logger.info(f"  Exported {len(rels):,} {rel_type} relationships")
    return rels


def export_graph(driver) -> dict:
    """Export the full graph from a Neo4j instance."""
    logger.info("Exporting graph data…")
    data: dict = {"nodes": {}, "relationships": {}}

    with driver.session() as session:
        # Nodes
        data["nodes"]["Drug"] = _export_nodes(session, "Drug")
        data["nodes"]["AdverseEvent"] = _export_nodes(session, "AdverseEvent")
        data["nodes"]["Outcome"] = _export_nodes(session, "Outcome")
        data["nodes"]["DrugCategory"] = _export_nodes(session, "DrugCategory")

        # Relationships
        data["relationships"]["CAUSES"] = _export_relationships(
            session, "CAUSES", "Drug", "AdverseEvent", "name", "name"
        )
        data["relationships"]["INTERACTS_WITH"] = _export_relationships(
            session, "INTERACTS_WITH", "Drug", "Drug", "name", "name"
        )
        data["relationships"]["HAS_OUTCOME"] = _export_relationships(
            session, "HAS_OUTCOME", "Drug", "Outcome", "name", "code"
        )
        data["relationships"]["BELONGS_TO"] = _export_relationships(
            session, "BELONGS_TO", "Drug", "DrugCategory", "name", "name"
        )

    total_nodes = sum(len(v) for v in data["nodes"].values())
    total_rels = sum(len(v) for v in data["relationships"].values())
    logger.success(f"Export complete: {total_nodes:,} nodes, {total_rels:,} relationships")
    return data


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def _import_nodes_batch(session, label: str, nodes: list[dict], key: str) -> None:
    """Import nodes in batches using MERGE."""
    for i in range(0, len(nodes), BATCH_SIZE):
        batch = nodes[i : i + BATCH_SIZE]
        query = f"""
        UNWIND $batch AS props
        MERGE (n:{label} {{{key}: props.{key}}})
        SET n += props
        """
        session.run(query, batch=batch)
    logger.info(f"  Imported {len(nodes):,} {label} nodes")


def _import_rels_batch(
    session,
    rel_type: str,
    src_label: str,
    tgt_label: str,
    src_key: str,
    tgt_key: str,
    rels: list[dict],
) -> None:
    """Import relationships in batches using MERGE."""
    for i in range(0, len(rels), BATCH_SIZE):
        batch = rels[i : i + BATCH_SIZE]
        query = f"""
        UNWIND $batch AS rel
        MATCH (a:{src_label} {{{src_key}: rel.src}})
        MATCH (b:{tgt_label} {{{tgt_key}: rel.tgt}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r += rel.props
        """
        session.run(query, batch=batch)
    logger.info(f"  Imported {len(rels):,} {rel_type} relationships")


def import_graph(driver, data: dict) -> None:
    """Import the exported graph into a Neo4j instance."""
    logger.info("Importing graph data…")

    with driver.session() as session:
        # Create schema (constraints + indexes)
        from pharmagraphrag.graph.schema import CONSTRAINTS, INDEXES

        for stmt in CONSTRAINTS + INDEXES:
            try:
                session.run(stmt)
            except Exception as exc:
                logger.warning(f"  Schema statement skipped: {exc}")

        # Nodes (order matters: DrugCategory before Drug)
        _import_nodes_batch(session, "DrugCategory", data["nodes"]["DrugCategory"], "name")
        _import_nodes_batch(session, "Outcome", data["nodes"]["Outcome"], "code")
        _import_nodes_batch(session, "AdverseEvent", data["nodes"]["AdverseEvent"], "name")
        _import_nodes_batch(session, "Drug", data["nodes"]["Drug"], "name")

        # Relationships
        _import_rels_batch(
            session,
            "CAUSES",
            "Drug",
            "AdverseEvent",
            "name",
            "name",
            data["relationships"]["CAUSES"],
        )
        _import_rels_batch(
            session,
            "INTERACTS_WITH",
            "Drug",
            "Drug",
            "name",
            "name",
            data["relationships"]["INTERACTS_WITH"],
        )
        _import_rels_batch(
            session,
            "HAS_OUTCOME",
            "Drug",
            "Outcome",
            "name",
            "code",
            data["relationships"]["HAS_OUTCOME"],
        )
        _import_rels_batch(
            session,
            "BELONGS_TO",
            "Drug",
            "DrugCategory",
            "name",
            "name",
            data["relationships"]["BELONGS_TO"],
        )

    total_nodes = sum(len(v) for v in data["nodes"].values())
    total_rels = sum(len(v) for v in data["relationships"].values())
    logger.success(f"Import complete: {total_nodes:,} nodes, {total_rels:,} relationships")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate PharmaGraphRAG Neo4j data between instances."
    )
    parser.add_argument(
        "--source", required=True, help="Source Neo4j URI (e.g. bolt://localhost:7687)"
    )
    parser.add_argument("--source-user", default="neo4j", help="Source username")
    parser.add_argument("--source-password", default="pharmagraphrag", help="Source password")
    parser.add_argument(
        "--target", required=True, help="Target Neo4j URI (e.g. neo4j+s://xxx.databases.neo4j.io)"
    )
    parser.add_argument("--target-user", default="neo4j", help="Target username")
    parser.add_argument("--target-password", required=True, help="Target password")
    args = parser.parse_args()

    t0 = time.time()

    # Connect source
    logger.info(f"Connecting to source: {args.source}")
    src_driver = GraphDatabase.driver(args.source, auth=(args.source_user, args.source_password))
    src_driver.verify_connectivity()
    logger.success("Source connected")

    # Export
    data = export_graph(src_driver)
    src_driver.close()

    # Check Aura Free limits
    total_nodes = sum(len(v) for v in data["nodes"].values())
    total_rels = sum(len(v) for v in data["relationships"].values())

    if total_nodes > 200_000:
        logger.warning(f"⚠️  {total_nodes:,} nodes exceeds Aura Free limit of 200,000!")
    if total_rels > 400_000:
        logger.warning(f"⚠️  {total_rels:,} relationships exceeds Aura Free limit of 400,000!")

    # Connect target
    logger.info(f"Connecting to target: {args.target}")
    tgt_driver = GraphDatabase.driver(args.target, auth=(args.target_user, args.target_password))
    tgt_driver.verify_connectivity()
    logger.success("Target connected")

    # Import
    import_graph(tgt_driver, data)
    tgt_driver.close()

    elapsed = time.time() - t0
    logger.success(f"Migration completed in {elapsed:.1f}s")
    logger.info(f"  {total_nodes:,} nodes, {total_rels:,} relationships migrated")


if __name__ == "__main__":
    main()
