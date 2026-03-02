"""Download FAERS quarterly data from FDA.

Wrapper script for the data pipeline module.

Usage:
    uv run python scripts/download_faers.py
    uv run python scripts/download_faers.py --quarters 2024Q3 2024Q4
"""

from pharmagraphrag.data.download_faers import main

if __name__ == "__main__":
    main()
