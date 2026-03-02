"""Fetch drug labels from DailyMed / openFDA API.

Wrapper script for the data pipeline module.

Usage:
    uv run python scripts/ingest_dailymed.py
    uv run python scripts/ingest_dailymed.py --top-n 20
"""

from pharmagraphrag.data.ingest_dailymed import main

if __name__ == "__main__":
    main()
