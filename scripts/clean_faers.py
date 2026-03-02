"""Clean FAERS data and convert to Parquet format.

Wrapper script for the data pipeline module.

Usage:
    uv run python scripts/clean_faers.py
    uv run python scripts/clean_faers.py --quarters 2024Q3 2024Q4
"""

from pharmagraphrag.data.clean_faers import main

if __name__ == "__main__":
    main()
