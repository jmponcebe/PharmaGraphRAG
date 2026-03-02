"""Load FAERS + DailyMed data into Neo4j knowledge graph.

Wrapper script for the graph loader module.

Usage:
    uv run python scripts/load_graph.py
    uv run python scripts/load_graph.py --skip-faers
    uv run python scripts/load_graph.py --drop
"""

from pharmagraphrag.graph.loader import main

if __name__ == "__main__":
    main()
