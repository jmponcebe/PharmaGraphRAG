"""Clean and transform FAERS raw data into analysis-ready Parquet files.

FAERS raw files are pipe-delimited (`$`) text files with inconsistent casing,
duplicates, and missing values. This module:
1. Loads raw FAERS text files (DEMO, DRUG, REAC, OUTC, INDI)
2. Standardizes column names and drug names
3. Removes duplicates and cleans data types
4. Saves cleaned data as Parquet files

Usage:
    uv run python -m pharmagraphrag.data.clean_faers
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from pharmagraphrag.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, get_settings

# FAERS files use '$' as delimiter
FAERS_DELIMITER = "$"

# Tables to process and their key columns
FAERS_TABLE_CONFIG: dict[str, dict] = {
    "DEMO": {
        "key_cols": ["primaryid", "caseid", "age", "age_cod", "sex", "wt", "wt_cod",
                     "occr_country", "event_dt", "init_fda_dt", "rept_cod", "mfr_sndr"],
        "dtype_overrides": {"primaryid": str, "caseid": str},
    },
    "DRUG": {
        "key_cols": ["primaryid", "caseid", "drug_seq", "drugname", "prod_ai",
                     "route", "dose_vbm", "dose_amt", "dose_unit", "dose_form",
                     "role_cod"],
        "dtype_overrides": {"primaryid": str, "caseid": str, "drug_seq": str},
    },
    "REAC": {
        "key_cols": ["primaryid", "caseid", "pt", "drug_rec_act"],
        "dtype_overrides": {"primaryid": str, "caseid": str},
    },
    "OUTC": {
        "key_cols": ["primaryid", "caseid", "outc_cod"],
        "dtype_overrides": {"primaryid": str, "caseid": str},
    },
    "INDI": {
        "key_cols": ["primaryid", "caseid", "drug_seq", "indi_pt"],
        "dtype_overrides": {"primaryid": str, "caseid": str, "drug_seq": str},
    },
}

# Outcome code descriptions
OUTCOME_DESCRIPTIONS = {
    "DE": "Death",
    "LT": "Life-Threatening",
    "HO": "Hospitalization",
    "DS": "Disability",
    "CA": "Congenital Anomaly",
    "RI": "Required Intervention",
    "OT": "Other Serious",
}

# Drug role codes
ROLE_DESCRIPTIONS = {
    "PS": "Primary Suspect",
    "SS": "Secondary Suspect",
    "C": "Concomitant",
    "I": "Interacting",
}


def find_faers_file(quarter_dir: Path, table_name: str) -> Path | None:
    """Find a FAERS file in a quarter directory, handling naming variations.

    FAERS files can be named like DRUG24Q3.txt, DRUG24Q3.TXT, etc.

    Args:
        quarter_dir: Path to the extracted quarter directory.
        table_name: FAERS table name (DEMO, DRUG, REAC, etc.).

    Returns:
        Path to the file, or None if not found.
    """
    # Try common patterns
    for ext in [".txt", ".TXT", ".csv", ".CSV"]:
        for f in quarter_dir.iterdir():
            if f.name.upper().startswith(table_name.upper()) and f.suffix.lower() == ext.lower():
                return f
    return None


def load_faers_table(filepath: Path, table_name: str) -> pd.DataFrame:
    """Load a single FAERS table from a raw text file.

    Args:
        filepath: Path to the FAERS text file.
        table_name: Table name for config lookup.

    Returns:
        DataFrame with raw data.
    """
    config = FAERS_TABLE_CONFIG[table_name]

    logger.info(f"Loading {table_name} from {filepath.name}")

    df = pd.read_csv(
        filepath,
        sep=FAERS_DELIMITER,
        dtype=config.get("dtype_overrides", {}),
        low_memory=False,
        on_bad_lines="skip",
        encoding="latin-1",
    )

    # Standardize column names to lowercase
    df.columns = [col.strip().lower() for col in df.columns]

    logger.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def normalize_drug_name(name: str | float) -> str:
    """Normalize a drug name for consistency.

    - Uppercase
    - Remove extra whitespace
    - Remove trailing punctuation
    - Strip common suffixes like dosage info in parentheses

    Args:
        name: Raw drug name.

    Returns:
        Normalized drug name.
    """
    if pd.isna(name) or not isinstance(name, str):
        return ""

    name = name.strip().upper()
    # Remove multiple spaces
    name = re.sub(r"\s+", " ", name)
    # Remove trailing dots, commas
    name = name.rstrip(".,;:")
    # Remove content in parentheses at the end (often dosage)
    name = re.sub(r"\s*\(.*\)\s*$", "", name)

    return name


def clean_demo(df: pd.DataFrame) -> pd.DataFrame:
    """Clean demographics table."""
    # Keep only relevant columns that exist
    available_cols = [c for c in FAERS_TABLE_CONFIG["DEMO"]["key_cols"] if c in df.columns]
    df = df[available_cols].copy()

    # Deduplicate by primaryid (keep latest report)
    df = df.drop_duplicates(subset=["primaryid"], keep="last")

    # Clean sex
    if "sex" in df.columns:
        df["sex"] = df["sex"].map({"M": "Male", "F": "Female", "UNK": "Unknown"}).fillna("Unknown")

    # Parse event date
    if "event_dt" in df.columns:
        df["event_dt"] = pd.to_datetime(df["event_dt"], format="%Y%m%d", errors="coerce")

    logger.info(f"  DEMO cleaned: {len(df):,} unique reports")
    return df


def clean_drug(df: pd.DataFrame) -> pd.DataFrame:
    """Clean drug table."""
    available_cols = [c for c in FAERS_TABLE_CONFIG["DRUG"]["key_cols"] if c in df.columns]
    df = df[available_cols].copy()

    # Normalize drug names
    if "drugname" in df.columns:
        df["drugname"] = df["drugname"].apply(normalize_drug_name)
        df = df[df["drugname"] != ""]

    if "prod_ai" in df.columns:
        df["prod_ai"] = df["prod_ai"].apply(normalize_drug_name)

    # Normalize role codes
    if "role_cod" in df.columns:
        df["role_cod"] = df["role_cod"].str.strip().str.upper()

    # Deduplicate
    dedup_cols = ["primaryid", "drug_seq"]
    dedup_cols = [c for c in dedup_cols if c in df.columns]
    df = df.drop_duplicates(subset=dedup_cols, keep="last")

    logger.info(f"  DRUG cleaned: {len(df):,} drug entries")
    return df


def clean_reac(df: pd.DataFrame) -> pd.DataFrame:
    """Clean reactions table."""
    available_cols = [c for c in FAERS_TABLE_CONFIG["REAC"]["key_cols"] if c in df.columns]
    df = df[available_cols].copy()

    # Normalize reaction terms (MedDRA Preferred Terms)
    if "pt" in df.columns:
        df["pt"] = df["pt"].str.strip().str.upper()
        df = df[df["pt"].notna() & (df["pt"] != "")]

    # Deduplicate
    dedup_cols = ["primaryid", "pt"]
    dedup_cols = [c for c in dedup_cols if c in df.columns]
    df = df.drop_duplicates(subset=dedup_cols, keep="last")

    logger.info(f"  REAC cleaned: {len(df):,} reaction entries")
    return df


def clean_outc(df: pd.DataFrame) -> pd.DataFrame:
    """Clean outcomes table."""
    available_cols = [c for c in FAERS_TABLE_CONFIG["OUTC"]["key_cols"] if c in df.columns]
    df = df[available_cols].copy()

    if "outc_cod" in df.columns:
        df["outc_cod"] = df["outc_cod"].str.strip().str.upper()
        df["outc_desc"] = df["outc_cod"].map(OUTCOME_DESCRIPTIONS).fillna("Unknown")

    # Deduplicate
    dedup_cols = ["primaryid", "outc_cod"]
    dedup_cols = [c for c in dedup_cols if c in df.columns]
    df = df.drop_duplicates(subset=dedup_cols, keep="last")

    logger.info(f"  OUTC cleaned: {len(df):,} outcome entries")
    return df


def clean_indi(df: pd.DataFrame) -> pd.DataFrame:
    """Clean indications table."""
    available_cols = [c for c in FAERS_TABLE_CONFIG["INDI"]["key_cols"] if c in df.columns]
    df = df[available_cols].copy()

    if "indi_pt" in df.columns:
        df["indi_pt"] = df["indi_pt"].str.strip().str.upper()

    # Deduplicate
    dedup_cols = ["primaryid", "drug_seq", "indi_pt"]
    dedup_cols = [c for c in dedup_cols if c in df.columns]
    df = df.drop_duplicates(subset=dedup_cols, keep="last")

    logger.info(f"  INDI cleaned: {len(df):,} indication entries")
    return df


# Mapping of table names to cleaning functions
CLEAN_FUNCTIONS = {
    "DEMO": clean_demo,
    "DRUG": clean_drug,
    "REAC": clean_reac,
    "OUTC": clean_outc,
    "INDI": clean_indi,
}


def process_quarter(quarter: str, raw_dir: Path | None = None,
                    output_dir: Path | None = None) -> dict[str, Path]:
    """Process all tables for a single FAERS quarter.

    Args:
        quarter: Quarter string like '2024Q3'.
        raw_dir: Base directory for raw data.
        output_dir: Directory for processed Parquet files.

    Returns:
        Dictionary mapping table names to output Parquet paths.
    """
    raw_dir = raw_dir or DATA_RAW_DIR
    output_dir = output_dir or DATA_PROCESSED_DIR

    quarter_raw = raw_dir / "faers" / quarter
    quarter_out = output_dir / "faers" / quarter
    quarter_out.mkdir(parents=True, exist_ok=True)

    if not quarter_raw.exists():
        logger.error(f"Raw data not found: {quarter_raw}")
        return {}

    logger.info(f"Processing FAERS {quarter}")

    output_files = {}
    for table_name, clean_fn in CLEAN_FUNCTIONS.items():
        filepath = find_faers_file(quarter_raw, table_name)
        if filepath is None:
            logger.warning(f"  {table_name} file not found in {quarter_raw}")
            continue

        # Load raw data
        df = load_faers_table(filepath, table_name)

        # Clean
        df = clean_fn(df)

        # Save as Parquet
        out_path = quarter_out / f"{table_name.lower()}.parquet"
        df.to_parquet(out_path, index=False, engine="pyarrow")
        output_files[table_name] = out_path

        logger.success(
            f"  {table_name} → {out_path.name} "
            f"({len(df):,} rows, {out_path.stat().st_size / 1e6:.1f} MB)"
        )

    return output_files


def process_all_quarters(quarters: list[str] | None = None) -> dict[str, dict[str, Path]]:
    """Process all configured FAERS quarters.

    Args:
        quarters: List of quarter strings. Defaults to settings.

    Returns:
        Nested dict: {quarter: {table: parquet_path}}.
    """
    settings = get_settings()
    quarters = quarters or settings.faers_quarters

    all_outputs = {}
    for quarter in quarters:
        all_outputs[quarter] = process_quarter(quarter)

    return all_outputs


def main() -> None:
    """CLI entry point for FAERS cleaning."""
    import argparse

    parser = argparse.ArgumentParser(description="Clean FAERS data and save as Parquet")
    parser.add_argument(
        "--quarters",
        nargs="+",
        default=None,
        help="Quarters to process (e.g., 2024Q3 2024Q4). Defaults to config.",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    results = process_all_quarters(args.quarters)

    total_tables = sum(len(tables) for tables in results.values())
    logger.success(f"Processed {total_tables} tables across {len(results)} quarters")


if __name__ == "__main__":
    main()
