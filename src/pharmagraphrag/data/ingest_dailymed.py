"""Ingest drug label information from DailyMed and openFDA APIs.

This module:
1. Uses a curated list of common drugs (or queries DailyMed for drug names)
2. Fetches drug label text sections from openFDA Drug Label API
3. Extracts key sections: interactions, adverse reactions, warnings, contraindications
4. Saves structured JSON for later embedding and graph construction

Data sources:
- DailyMed API: https://dailymed.nlm.nih.gov/dailymed/services/v2/
- openFDA API: https://api.fda.gov/drug/label.json

Usage:
    uv run python -m pharmagraphrag.data.ingest_dailymed
    uv run python -m pharmagraphrag.data.ingest_dailymed --top-n 100
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx
from loguru import logger
from tqdm import tqdm

from pharmagraphrag.config import DATA_RAW_DIR, get_settings

# API endpoints
DAILYMED_BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"

# Label sections we want to extract (openFDA field names)
LABEL_SECTIONS = [
    "drug_interactions",
    "adverse_reactions",
    "warnings",
    "warnings_and_cautions",
    "contraindications",
    "boxed_warning",
    "indications_and_usage",
    "dosage_and_administration",
    "clinical_pharmacology",
    "mechanism_of_action",
    "pharmacodynamics",
    "overdosage",
]

# Curated list of common drugs to start with
# These are among the most prescribed drugs globally, covering diverse categories
TOP_DRUGS: list[str] = [
    # Pain / Anti-inflammatory
    "ibuprofen",
    "acetaminophen",
    "aspirin",
    "naproxen",
    "celecoxib",
    "diclofenac",
    "meloxicam",
    # Cardiovascular
    "lisinopril",
    "amlodipine",
    "metoprolol",
    "atorvastatin",
    "simvastatin",
    "losartan",
    "valsartan",
    "hydrochlorothiazide",
    "furosemide",
    "warfarin",
    "clopidogrel",
    "apixaban",
    "rivaroxaban",
    # Diabetes
    "metformin",
    "glipizide",
    "sitagliptin",
    "empagliflozin",
    "insulin glargine",
    "liraglutide",
    "semaglutide",
    # Respiratory
    "albuterol",
    "fluticasone",
    "montelukast",
    "tiotropium",
    "budesonide",
    # Antibiotics
    "amoxicillin",
    "azithromycin",
    "ciprofloxacin",
    "doxycycline",
    "metronidazole",
    "levofloxacin",
    "trimethoprim",
    "cephalexin",
    "clindamycin",
    # Psychiatric / Neurological
    "sertraline",
    "fluoxetine",
    "escitalopram",
    "duloxetine",
    "venlafaxine",
    "bupropion",
    "trazodone",
    "alprazolam",
    "lorazepam",
    "diazepam",
    "gabapentin",
    "pregabalin",
    "lamotrigine",
    "carbamazepine",
    "levetiracetam",
    "quetiapine",
    "aripiprazole",
    "olanzapine",
    "risperidone",
    # Gastrointestinal
    "omeprazole",
    "pantoprazole",
    "esomeprazole",
    "ranitidine",
    "ondansetron",
    # Thyroid
    "levothyroxine",
    # Steroids
    "prednisone",
    "dexamethasone",
    "methylprednisolone",
    # Opioids
    "tramadol",
    "oxycodone",
    "morphine",
    "hydrocodone",
    # Muscle relaxants
    "cyclobenzaprine",
    "baclofen",
    # Other common
    "sildenafil",
    "tadalafil",
    "finasteride",
    "tamsulosin",
    "allopurinol",
    "colchicine",
    "hydroxychloroquine",
    "methotrexate",
    "zolpidem",
    "melatonin",
    "cetirizine",
    "loratadine",
    "diphenhydramine",
]


def fetch_drug_names_from_dailymed(top_n: int = 200, timeout: float = 30.0) -> list[str]:
    """Fetch drug names from DailyMed API.

    Args:
        top_n: Maximum number of drug names to fetch.
        timeout: Request timeout in seconds.

    Returns:
        List of generic drug names.
    """
    logger.info(f"Fetching drug names from DailyMed (up to {top_n})")

    drug_names: list[str] = []
    page = 1
    page_size = 100

    with httpx.Client(timeout=timeout) as client:
        while len(drug_names) < top_n:
            url = f"{DAILYMED_BASE_URL}/drugnames.json"
            params = {"pagesize": page_size, "page": page}

            try:
                response = client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
            except (httpx.HTTPError, json.JSONDecodeError) as e:
                logger.warning(f"Error fetching page {page}: {e}")
                break

            names = data.get("data", [])
            if not names:
                break

            for item in names:
                name = item.get("drug_name", "").strip().lower()
                if name and name not in drug_names:
                    drug_names.append(name)

            page += 1
            time.sleep(0.2)  # Rate limiting

    logger.info(f"Fetched {len(drug_names)} drug names from DailyMed")
    return drug_names[:top_n]


def fetch_drug_label(drug_name: str, timeout: float = 30.0) -> dict | None:
    """Fetch drug label information from openFDA API.

    Args:
        drug_name: Generic drug name.
        timeout: Request timeout in seconds.

    Returns:
        Dictionary with extracted label sections, or None if not found.
    """
    # Try generic name first, then brand name
    search_queries = [
        f'openfda.generic_name:"{drug_name}"',
        f'openfda.brand_name:"{drug_name}"',
    ]

    with httpx.Client(timeout=timeout) as client:
        for query in search_queries:
            try:
                response = client.get(
                    OPENFDA_LABEL_URL,
                    params={"search": query, "limit": 1},
                )

                if response.status_code == 404:
                    continue
                response.raise_for_status()

                data = response.json()
                results = data.get("results", [])
                if results:
                    return _extract_label_data(drug_name, results[0])

            except (httpx.HTTPError, json.JSONDecodeError) as e:
                logger.debug(f"Error fetching label for {drug_name}: {e}")
                continue

    return None


def _extract_label_data(drug_name: str, result: dict) -> dict:
    """Extract relevant sections from an openFDA label result.

    Args:
        drug_name: The drug name used for the query.
        result: Single result from openFDA API.

    Returns:
        Structured dictionary with drug info and label sections.
    """
    # Extract openFDA metadata
    openfda = result.get("openfda", {})

    label_data = {
        "drug_name": drug_name.upper(),
        "generic_names": openfda.get("generic_name", []),
        "brand_names": openfda.get("brand_name", []),
        "manufacturer": openfda.get("manufacturer_name", []),
        "product_type": openfda.get("product_type", []),
        "route": openfda.get("route", []),
        "substance_name": openfda.get("substance_name", []),
        "pharm_class_epc": openfda.get("pharm_class_epc", []),
        "pharm_class_moa": openfda.get("pharm_class_moa", []),
        "sections": {},
    }

    # Extract text sections
    for section in LABEL_SECTIONS:
        text_list = result.get(section, [])
        if text_list:
            # openFDA returns lists of strings; join them
            label_data["sections"][section] = "\n\n".join(text_list)

    return label_data


def ingest_drug_labels(
    drug_names: list[str] | None = None, output_dir: Path | None = None
) -> list[dict]:
    """Fetch and save drug labels for a list of drugs.

    Args:
        drug_names: List of drug names. Defaults to TOP_DRUGS.
        output_dir: Output directory. Defaults to DATA_RAW_DIR/dailymed.

    Returns:
        List of successfully fetched drug label data.
    """
    drug_names = drug_names or TOP_DRUGS
    output_dir = output_dir or (DATA_RAW_DIR / "dailymed")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Ingesting labels for {len(drug_names)} drugs")

    labels: list[dict] = []
    failed: list[str] = []

    for drug_name in tqdm(drug_names, desc="Fetching drug labels"):
        # Check if already downloaded
        safe_name = drug_name.replace(" ", "_").replace("/", "_").lower()
        label_path = output_dir / f"{safe_name}.json"

        if label_path.exists():
            with open(label_path) as f:
                label_data = json.load(f)
            labels.append(label_data)
            continue

        label_data = fetch_drug_label(drug_name)

        if label_data is None:
            failed.append(drug_name)
            logger.debug(f"No label found for: {drug_name}")
            continue

        if not label_data["sections"]:
            failed.append(drug_name)
            logger.debug(f"No relevant sections for: {drug_name}")
            continue

        # Save individual label
        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(label_data, f, indent=2, ensure_ascii=False)

        labels.append(label_data)

        # Rate limiting for openFDA (max 240 requests/minute without API key)
        time.sleep(0.3)

    # Save summary
    summary = {
        "total_drugs_queried": len(drug_names),
        "labels_found": len(labels),
        "labels_failed": len(failed),
        "failed_drugs": failed,
        "sections_available": {
            section: sum(1 for lbl in labels if section in lbl.get("sections", {}))
            for section in LABEL_SECTIONS
        },
    }

    summary_path = output_dir / "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.success(f"Ingested {len(labels)}/{len(drug_names)} drug labels ({len(failed)} failed)")
    if failed:
        logger.info(f"Failed drugs: {', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}")

    return labels


def main() -> None:
    """CLI entry point for DailyMed ingestion."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest drug labels from DailyMed/openFDA")
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Number of top drugs to ingest. Defaults to the curated list (~85 drugs).",
    )
    parser.add_argument(
        "--use-dailymed-list",
        action="store_true",
        help="Fetch drug names from DailyMed API instead of using the curated list.",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    settings = get_settings()

    if args.use_dailymed_list:
        top_n = args.top_n or settings.dailymed_top_n_drugs
        drug_names = fetch_drug_names_from_dailymed(top_n)
    else:
        drug_names = TOP_DRUGS
        if args.top_n:
            drug_names = drug_names[: args.top_n]

    labels = ingest_drug_labels(drug_names)

    # Print summary stats
    if labels:
        logger.info("Section coverage:")
        for section in LABEL_SECTIONS:
            count = sum(1 for lbl in labels if section in lbl.get("sections", {}))
            if count > 0:
                logger.info(f"  {section}: {count}/{len(labels)} drugs")


if __name__ == "__main__":
    main()
