"""One-click demo setup — load demo data into Neo4j + ChromaDB.

Loads pre-aggregated FAERS data and DailyMed labels so the system is
ready to answer queries immediately. Takes ~2-3 minutes.

Usage:
    uv run python scripts/setup_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from loguru import logger

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = PROJECT_ROOT / "data" / "demo"
DEMO_FAERS = DEMO_DIR / "faers_graph.json"
DEMO_DAILYMED = DEMO_DIR / "dailymed"


def _print_step(step: int, total: int, msg: str) -> None:
    """Print a formatted step message."""
    logger.info(f"[{step}/{total}] {msg}")


def load_demo_faers(faers_json: Path | None = None) -> dict[str, int]:
    """Load pre-aggregated FAERS demo data into Neo4j.

    Args:
        faers_json: Path to the demo FAERS JSON file.

    Returns:
        Dictionary with counts of loaded entities.
    """
    from neo4j import GraphDatabase

    from pharmagraphrag.config import get_settings

    faers_json = faers_json or DEMO_FAERS
    settings = get_settings()

    with open(faers_json, encoding="utf-8") as f:
        data = json.load(f)

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    counts = {
        "drugs": 0,
        "adverse_events": 0,
        "outcomes": 0,
        "causes": 0,
        "has_outcome": 0,
    }

    try:
        with driver.session() as session:
            # Load drug nodes
            for drug in data["drugs"]:
                session.run("MERGE (d:Drug {name: $name})", name=drug)
            counts["drugs"] = len(data["drugs"])

            # Load adverse event nodes
            for ae in data["adverse_events"]:
                session.run("MERGE (ae:AdverseEvent {name: $name})", name=ae)
            counts["adverse_events"] = len(data["adverse_events"])

            # Load outcome nodes
            for outcome in data["outcomes"]:
                session.run(
                    """
                    MERGE (o:Outcome {code: $code})
                    ON CREATE SET o.description = $description
                    """,
                    code=outcome["code"],
                    description=outcome["description"],
                )
            counts["outcomes"] = len(data["outcomes"])

            # Load CAUSES relationships
            for rel in data["causes"]:
                session.run(
                    """
                    MATCH (d:Drug {name: $drug_name})
                    MATCH (ae:AdverseEvent {name: $event_name})
                    MERGE (d)-[r:CAUSES]->(ae)
                    ON CREATE SET r.report_count = $report_count
                    """,
                    drug_name=rel["drug_name"],
                    event_name=rel["event_name"],
                    report_count=rel["report_count"],
                )
            counts["causes"] = len(data["causes"])

            # Load HAS_OUTCOME relationships
            for rel in data["has_outcome"]:
                session.run(
                    """
                    MATCH (d:Drug {name: $drug_name})
                    MATCH (o:Outcome {code: $outc_cod})
                    MERGE (d)-[r:HAS_OUTCOME]->(o)
                    ON CREATE SET r.report_count = $report_count
                    """,
                    drug_name=rel["drug_name"],
                    outc_cod=rel["outc_cod"],
                    report_count=rel["report_count"],
                )
            counts["has_outcome"] = len(data["has_outcome"])

    finally:
        driver.close()

    return counts


def load_demo_dailymed(dailymed_dir: Path | None = None) -> dict[str, int]:
    """Load DailyMed labels into Neo4j (interactions + categories).

    Args:
        dailymed_dir: Path to the demo DailyMed directory.

    Returns:
        Dictionary with counts.
    """
    from pharmagraphrag.graph.loader import load_dailymed_to_neo4j

    dailymed_dir = dailymed_dir or DEMO_DAILYMED
    return load_dailymed_to_neo4j(raw_dir=dailymed_dir)


def load_demo_vectorstore(dailymed_dir: Path | None = None) -> int:
    """Generate embeddings and load into ChromaDB.

    Args:
        dailymed_dir: Path to the demo DailyMed directory.

    Returns:
        Number of chunks added.
    """
    from pharmagraphrag.vectorstore.chunker import chunk_all_labels
    from pharmagraphrag.vectorstore.store import add_chunks, reset_collection

    dailymed_dir = dailymed_dir or DEMO_DAILYMED

    chunks = chunk_all_labels(dailymed_dir=dailymed_dir, chunk_size=1000, chunk_overlap=200)
    if not chunks:
        logger.error("No chunks created from DailyMed labels")
        return 0

    collection = reset_collection()
    added = add_chunks(chunks, collection=collection, batch_size=50)
    return added


def main() -> None:
    """Run the full demo setup pipeline."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="  {message}")

    total_steps = 5
    start = time.time()

    print()
    print("=" * 50)
    print("  PharmaGraphRAG — Demo Setup")
    print("=" * 50)
    print()

    # Step 1: Verify demo data exists
    _print_step(1, total_steps, "Verifying demo data...")
    if not DEMO_FAERS.exists():
        logger.error(f"Demo FAERS data not found: {DEMO_FAERS}")
        sys.exit(1)
    if not DEMO_DAILYMED.exists():
        logger.error(f"Demo DailyMed data not found: {DEMO_DAILYMED}")
        sys.exit(1)
    dm_count = len(list(DEMO_DAILYMED.glob("*.json")))
    logger.info(f"  Found FAERS graph data + {dm_count} DailyMed labels ✓")

    # Step 2: Create Neo4j schema
    _print_step(2, total_steps, "Creating Neo4j schema...")
    from pharmagraphrag.graph.schema import create_schema

    create_schema()
    logger.info("  Schema created ✓")

    # Step 3: Load FAERS demo data into Neo4j
    _print_step(3, total_steps, "Loading FAERS data into knowledge graph...")
    t0 = time.time()
    faers_counts = load_demo_faers()
    logger.info(
        f"  Loaded {faers_counts['drugs']} drugs, "
        f"{faers_counts['adverse_events']} adverse events, "
        f"{faers_counts['causes']} relationships ({time.time() - t0:.1f}s) ✓"
    )

    # Step 4: Load DailyMed into Neo4j (interactions + categories)
    _print_step(4, total_steps, "Loading DailyMed interactions + categories...")
    t0 = time.time()
    dm_counts = load_demo_dailymed()
    logger.info(
        f"  Added {dm_counts.get('interactions', 0)} interactions, "
        f"{dm_counts.get('categories', 0)} categories ({time.time() - t0:.1f}s) ✓"
    )

    # Step 5: Load embeddings into ChromaDB
    _print_step(5, total_steps, "Generating embeddings for vector search...")
    t0 = time.time()
    chunks_added = load_demo_vectorstore()
    logger.info(f"  Added {chunks_added} text chunks to ChromaDB ({time.time() - t0:.1f}s) ✓")

    elapsed = time.time() - start
    print()
    print("=" * 50)
    print(f"  ✅ Demo setup complete! ({elapsed:.0f}s)")
    print("=" * 50)
    print()
    print("  ⚠️  DEMO DATA NOTICE:")
    print("  The demo uses a representative subset (88 drugs,")
    print("  top 15 adverse events per drug). Results may differ")
    print("  from the full dataset (816K reports, 4,998 drugs).")
    print("  For complete data, run the full pipeline (see README).")
    print("  This project is for educational purposes only.")
    print()
    print("  Start the app:")
    print("    uv run uvicorn pharmagraphrag.api.main:app --host 0.0.0.0 &")
    print("    uv run streamlit run src/pharmagraphrag/ui/app.py")
    print()
    print('  Try asking: "What are the side effects of metformin?"')
    print()


if __name__ == "__main__":
    main()
