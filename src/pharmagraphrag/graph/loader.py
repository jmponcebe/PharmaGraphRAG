"""Load processed data into Neo4j knowledge graph.

Loads FAERS Parquet files and DailyMed JSON labels into the Neo4j graph,
creating nodes (Drug, AdverseEvent, Outcome, DrugCategory) and relationships
(CAUSES, INTERACTS_WITH, BELONGS_TO, HAS_OUTCOME).

Usage:
    uv run python -m pharmagraphrag.graph.loader
    uv run python -m pharmagraphrag.graph.loader --skip-faers
    uv run python -m pharmagraphrag.graph.loader --skip-dailymed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger
from neo4j import Driver, GraphDatabase, ManagedTransaction

from pharmagraphrag.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, get_settings
from pharmagraphrag.data.clean_faers import OUTCOME_DESCRIPTIONS

# Batch size for Neo4j imports
BATCH_SIZE = 5000


def _get_driver(
    uri: str | None = None, user: str | None = None, password: str | None = None
) -> Driver:
    """Create a Neo4j driver instance."""
    settings = get_settings()
    return GraphDatabase.driver(
        uri or settings.neo4j_uri,
        auth=(user or settings.neo4j_user, password or settings.neo4j_password),
    )


# ---------------------------------------------------------------------------
# FAERS loading
# ---------------------------------------------------------------------------


def _load_drugs_batch(tx: ManagedTransaction, drugs: list[dict]) -> None:
    """Load a batch of drug nodes via MERGE."""
    tx.run(
        """
        UNWIND $drugs AS drug
        MERGE (d:Drug {name: drug.name})
        ON CREATE SET d.role_cod = drug.role_cod
        """,
        drugs=drugs,
    )


def _load_adverse_events_batch(tx: ManagedTransaction, events: list[dict]) -> None:
    """Load a batch of adverse event nodes via MERGE."""
    tx.run(
        """
        UNWIND $events AS event
        MERGE (ae:AdverseEvent {name: event.name})
        """,
        events=events,
    )


def _load_outcomes_batch(tx: ManagedTransaction, outcomes: list[dict]) -> None:
    """Load outcome nodes."""
    tx.run(
        """
        UNWIND $outcomes AS outcome
        MERGE (o:Outcome {code: outcome.code})
        ON CREATE SET o.description = outcome.description
        """,
        outcomes=outcomes,
    )


def _load_causes_batch(tx: ManagedTransaction, relations: list[dict]) -> None:
    """Load CAUSES relationships (Drug -> AdverseEvent)."""
    tx.run(
        """
        UNWIND $rels AS rel
        MATCH (d:Drug {name: rel.drug_name})
        MATCH (ae:AdverseEvent {name: rel.event_name})
        MERGE (d)-[r:CAUSES]->(ae)
        ON CREATE SET r.report_count = rel.report_count
        ON MATCH SET r.report_count = r.report_count + rel.report_count
        """,
        rels=relations,
    )


def _load_has_outcome_batch(tx: ManagedTransaction, relations: list[dict]) -> None:
    """Load HAS_OUTCOME relationships (Drug -> Outcome)."""
    tx.run(
        """
        UNWIND $rels AS rel
        MATCH (d:Drug {name: rel.drug_name})
        MATCH (o:Outcome {code: rel.outc_cod})
        MERGE (d)-[r:HAS_OUTCOME]->(o)
        ON CREATE SET r.report_count = rel.report_count
        ON MATCH SET r.report_count = r.report_count + rel.report_count
        """,
        rels=relations,
    )


def load_faers_to_neo4j(
    processed_dir: Path | None = None, quarters: list[str] | None = None
) -> dict[str, int]:
    """Load FAERS data from Parquet into Neo4j.

    Creates Drug and AdverseEvent nodes and CAUSES relationships,
    aggregated by drug-reaction pair with report counts.

    Args:
        processed_dir: Directory with processed Parquet files.
        quarters: Quarters to load.

    Returns:
        Dictionary with counts of loaded entities.
    """
    settings = get_settings()
    processed_dir = processed_dir or DATA_PROCESSED_DIR
    quarters = quarters or settings.faers_quarters

    driver = _get_driver()
    counts = {"drugs": 0, "events": 0, "outcomes": 0, "causes": 0, "has_outcome": 0}

    try:
        for quarter in quarters:
            quarter_dir = processed_dir / "faers" / quarter
            if not quarter_dir.exists():
                logger.warning(f"No processed data for {quarter}")
                continue

            logger.info(f"Loading FAERS {quarter} into Neo4j")

            # --- Load Outcome nodes (static) ---
            with driver.session() as session:
                outcomes = [
                    {"code": code, "description": desc}
                    for code, desc in OUTCOME_DESCRIPTIONS.items()
                ]
                session.execute_write(_load_outcomes_batch, outcomes)
                counts["outcomes"] = len(outcomes)
                logger.info(f"  Loaded {len(outcomes)} outcome types")

            # --- Load Drug → AdverseEvent relationships ---
            drug_path = quarter_dir / "drug.parquet"
            reac_path = quarter_dir / "reac.parquet"

            if not drug_path.exists() or not reac_path.exists():
                logger.warning(f"  Missing drug or reac parquet for {quarter}")
                continue

            # Load and merge drug + reaction data
            df_drug = pd.read_parquet(drug_path, columns=["primaryid", "drugname", "role_cod"])
            df_reac = pd.read_parquet(reac_path, columns=["primaryid", "pt"])

            # Focus on primary/secondary suspect drugs
            df_drug = df_drug[df_drug["role_cod"].isin(["PS", "SS"])]

            # Join drug and reaction on primaryid
            df_merged = df_drug.merge(df_reac, on="primaryid", how="inner")

            # Aggregate: count reports per drug-reaction pair
            df_agg = (
                df_merged.groupby(["drugname", "pt"])
                .agg(report_count=("primaryid", "nunique"))
                .reset_index()
            )

            # Filter: keep only pairs with >= 3 reports (noise reduction)
            df_agg = df_agg[df_agg["report_count"] >= 3]

            logger.info(
                f"  {quarter}: {df_agg['drugname'].nunique():,} drugs, "
                f"{df_agg['pt'].nunique():,} adverse events, "
                f"{len(df_agg):,} drug-event pairs"
            )

            # Load drug nodes in batches
            unique_drugs = df_agg["drugname"].unique().tolist()
            with driver.session() as session:
                for i in range(0, len(unique_drugs), BATCH_SIZE):
                    batch = [
                        {"name": d, "role_cod": "PS"} for d in unique_drugs[i : i + BATCH_SIZE]
                    ]
                    session.execute_write(_load_drugs_batch, batch)
                counts["drugs"] += len(unique_drugs)
                logger.info(f"  Loaded {len(unique_drugs):,} drug nodes")

            # Load adverse event nodes in batches
            unique_events = df_agg["pt"].unique().tolist()
            with driver.session() as session:
                for i in range(0, len(unique_events), BATCH_SIZE):
                    batch = [{"name": e} for e in unique_events[i : i + BATCH_SIZE]]
                    session.execute_write(_load_adverse_events_batch, batch)
                counts["events"] += len(unique_events)
                logger.info(f"  Loaded {len(unique_events):,} adverse event nodes")

            # Load CAUSES relationships in batches
            causes_data = df_agg.rename(
                columns={"drugname": "drug_name", "pt": "event_name"}
            ).to_dict("records")

            with driver.session() as session:
                for i in range(0, len(causes_data), BATCH_SIZE):
                    batch = causes_data[i : i + BATCH_SIZE]
                    session.execute_write(_load_causes_batch, batch)
                counts["causes"] += len(causes_data)
                logger.info(f"  Loaded {len(causes_data):,} CAUSES relationships")

            # --- Load Drug → Outcome relationships ---
            outc_path = quarter_dir / "outc.parquet"
            if outc_path.exists():
                df_outc = pd.read_parquet(outc_path, columns=["primaryid", "outc_cod"])
                df_drug_outc = df_drug.merge(df_outc, on="primaryid", how="inner")
                df_outc_agg = (
                    df_drug_outc.groupby(["drugname", "outc_cod"])
                    .agg(report_count=("primaryid", "nunique"))
                    .reset_index()
                )
                df_outc_agg = df_outc_agg[df_outc_agg["report_count"] >= 3]

                outc_data = df_outc_agg.rename(columns={"drugname": "drug_name"}).to_dict("records")

                with driver.session() as session:
                    for i in range(0, len(outc_data), BATCH_SIZE):
                        batch = outc_data[i : i + BATCH_SIZE]
                        session.execute_write(_load_has_outcome_batch, batch)
                    counts["has_outcome"] += len(outc_data)
                    logger.info(f"  Loaded {len(outc_data):,} HAS_OUTCOME relationships")

    finally:
        driver.close()

    return counts


# ---------------------------------------------------------------------------
# DailyMed loading (drug interactions + categories)
# ---------------------------------------------------------------------------


def _load_interacts_with_batch(tx: ManagedTransaction, interactions: list[dict]) -> None:
    """Load INTERACTS_WITH relationships between drugs."""
    tx.run(
        """
        UNWIND $interactions AS interaction
        MERGE (d1:Drug {name: interaction.drug1})
        MERGE (d2:Drug {name: interaction.drug2})
        MERGE (d1)-[r:INTERACTS_WITH]->(d2)
        ON CREATE SET r.description = interaction.description,
                      r.source = 'DailyMed'
        """,
        interactions=interactions,
    )


def _load_drug_categories_batch(tx: ManagedTransaction, relations: list[dict]) -> None:
    """Load BELONGS_TO relationships (Drug -> DrugCategory)."""
    tx.run(
        """
        UNWIND $rels AS rel
        MERGE (d:Drug {name: rel.drug_name})
        MERGE (dc:DrugCategory {name: rel.category})
        MERGE (d)-[:BELONGS_TO]->(dc)
        """,
        rels=relations,
    )


def _extract_interacting_drugs(
    drug_name: str, interaction_text: str, known_drugs: set[str]
) -> list[dict]:
    """Extract drug interaction pairs from label text.

    Simple approach: check if any known drug names appear in the interaction text.

    Args:
        drug_name: The drug whose label we're reading.
        interaction_text: Text from the drug_interactions section.
        known_drugs: Set of known drug names (uppercase).

    Returns:
        List of interaction dicts with drug1, drug2, description.
    """
    text_upper = interaction_text.upper()
    interactions = []

    for other_drug in known_drugs:
        if other_drug == drug_name:
            continue
        if other_drug in text_upper and len(other_drug) > 3:
            # Extract a snippet around the mention for context
            idx = text_upper.find(other_drug)
            start = max(0, idx - 100)
            end = min(len(interaction_text), idx + len(other_drug) + 200)
            snippet = interaction_text[start:end].strip()

            interactions.append(
                {
                    "drug1": drug_name,
                    "drug2": other_drug,
                    "description": snippet[:500],  # Cap at 500 chars
                }
            )

    return interactions


def load_dailymed_to_neo4j(raw_dir: Path | None = None) -> dict[str, int]:
    """Load DailyMed drug label data into Neo4j.

    Creates INTERACTS_WITH relationships based on drug interaction section text,
    and BELONGS_TO relationships for pharmacologic classes.

    Args:
        raw_dir: Directory with DailyMed JSON files.

    Returns:
        Dictionary with counts of loaded entities.
    """
    raw_dir = raw_dir or (DATA_RAW_DIR / "dailymed")
    counts = {"interactions": 0, "categories": 0, "drugs_enriched": 0}

    if not raw_dir.exists():
        logger.warning(f"DailyMed data directory not found: {raw_dir}")
        return counts

    # Load all drug labels
    labels = []
    for json_file in sorted(raw_dir.glob("*.json")):
        if json_file.name.startswith("_"):
            continue
        with open(json_file, encoding="utf-8") as f:
            labels.append(json.load(f))

    if not labels:
        logger.warning("No DailyMed labels found")
        return counts

    logger.info(f"Processing {len(labels)} drug labels for Neo4j")

    # Build set of known drug names
    known_drugs = {label["drug_name"] for label in labels}

    driver = _get_driver()

    try:
        # --- Enrich Drug nodes with DailyMed metadata ---
        with driver.session() as session:
            for label in labels:
                session.run(
                    """
                    MERGE (d:Drug {name: $name})
                    SET d.generic_names = $generic_names,
                        d.brand_names = $brand_names,
                        d.route = $route,
                        d.category = $category
                    """,
                    name=label["drug_name"],
                    generic_names=label.get("generic_names", []),
                    brand_names=label.get("brand_names", []),
                    route=", ".join(label.get("route", [])),
                    category=", ".join(label.get("pharm_class_epc", [])),
                )
                counts["drugs_enriched"] += 1

        logger.info(f"  Enriched {counts['drugs_enriched']} drug nodes with DailyMed metadata")

        # --- Load INTERACTS_WITH relationships ---
        all_interactions = []
        for label in labels:
            interaction_text = label.get("sections", {}).get("drug_interactions", "")
            if not interaction_text:
                continue

            interactions = _extract_interacting_drugs(
                label["drug_name"], interaction_text, known_drugs
            )
            all_interactions.extend(interactions)

        # Deduplicate (A->B and B->A are the same)
        seen = set()
        unique_interactions = []
        for inter in all_interactions:
            pair = tuple(sorted([inter["drug1"], inter["drug2"]]))
            if pair not in seen:
                seen.add(pair)
                unique_interactions.append(inter)

        with driver.session() as session:
            for i in range(0, len(unique_interactions), BATCH_SIZE):
                batch = unique_interactions[i : i + BATCH_SIZE]
                session.execute_write(_load_interacts_with_batch, batch)

        counts["interactions"] = len(unique_interactions)
        logger.info(f"  Loaded {len(unique_interactions)} INTERACTS_WITH relationships")

        # --- Load Drug → DrugCategory relationships ---
        category_rels = []
        for label in labels:
            drug_name = label["drug_name"]
            for cat in label.get("pharm_class_epc", []):
                category_rels.append({"drug_name": drug_name, "category": cat.strip()})

        with driver.session() as session:
            for i in range(0, len(category_rels), BATCH_SIZE):
                batch = category_rels[i : i + BATCH_SIZE]
                session.execute_write(_load_drug_categories_batch, batch)

        counts["categories"] = len(set(r["category"] for r in category_rels))
        logger.info(
            f"  Loaded {len(category_rels)} BELONGS_TO relationships "
            f"({counts['categories']} unique categories)"
        )

    finally:
        driver.close()

    return counts


def main() -> None:
    """CLI entry point for Neo4j data loading."""
    import argparse

    parser = argparse.ArgumentParser(description="Load data into Neo4j")
    parser.add_argument("--skip-faers", action="store_true", help="Skip FAERS loading")
    parser.add_argument("--skip-dailymed", action="store_true", help="Skip DailyMed loading")
    parser.add_argument("--drop", action="store_true", help="Drop all data first")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Import schema module
    from pharmagraphrag.graph.schema import create_schema, drop_all_data, get_schema_info

    if args.drop:
        logger.warning("Dropping all existing data...")
        drop_all_data()

    # Create schema first
    create_schema()

    # Load data
    total_counts: dict[str, int] = {}

    if not args.skip_faers:
        faers_counts = load_faers_to_neo4j()
        total_counts.update({f"faers_{k}": v for k, v in faers_counts.items()})

    if not args.skip_dailymed:
        dm_counts = load_dailymed_to_neo4j()
        total_counts.update({f"dailymed_{k}": v for k, v in dm_counts.items()})

    # Print summary
    logger.success("Loading complete!")
    for key, value in total_counts.items():
        logger.info(f"  {key}: {value:,}")

    # Final schema info
    info = get_schema_info()
    logger.info("Final graph stats:")
    for label, count in info["nodes"].items():
        logger.info(f"  :{label} — {count:,} nodes")
    for rel_type, count in info["relationships"].items():
        logger.info(f"  :{rel_type} — {count:,} relationships")


if __name__ == "__main__":
    main()
