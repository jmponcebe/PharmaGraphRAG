#!/usr/bin/env bash
# =============================================================
# PharmaGraphRAG — Codespaces / devcontainer setup
# =============================================================
# This script runs automatically when a Codespace is created.
# It installs dependencies and starts Neo4j so the project is
# ready to use immediately.
# =============================================================

set -euo pipefail

echo "============================================="
echo " PharmaGraphRAG — Setting up environment"
echo "============================================="

# 1. Install uv (fast Python package manager)
echo ">>> Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. Install Python dependencies
echo ">>> Installing Python dependencies..."
uv sync --extra dev --extra ui

# 3. Create .env from example if not present
if [ ! -f .env ]; then
    echo ">>> Creating .env from .env.example..."
    cp .env.example .env
    echo ""
    echo "  ⚠️  IMPORTANT: Set your GEMINI_API_KEY in .env"
    echo "     Get a free key at: https://aistudio.google.com/apikey"
    echo ""
fi

# 4. Start Neo4j via Docker Compose
echo ">>> Starting Neo4j..."
docker compose up -d neo4j

# 5. Wait for Neo4j to be healthy
echo ">>> Waiting for Neo4j to be ready..."
RETRIES=30
until docker compose exec neo4j wget --quiet --spider http://localhost:7474 2>/dev/null || [ $RETRIES -eq 0 ]; do
    echo "    Waiting... ($RETRIES attempts remaining)"
    sleep 3
    RETRIES=$((RETRIES - 1))
done

if [ $RETRIES -eq 0 ]; then
    echo "  ⚠️  Neo4j didn't start in time. Run: docker compose up -d neo4j"
else
    echo "  ✅ Neo4j is ready!"
fi

echo ""
echo "============================================="
echo " ✅ Setup complete!"
echo "============================================="
echo ""
echo " Quick start:"
echo "   1. Set GEMINI_API_KEY in .env (free: https://aistudio.google.com/apikey)"
echo "   2. Run the data pipeline:"
echo "      uv run python scripts/download_faers.py"
echo "      uv run python scripts/clean_faers.py"
echo "      uv run python scripts/ingest_dailymed.py"
echo "      uv run python scripts/load_graph.py"
echo "      uv run python scripts/load_vectorstore.py"
echo "   3. Start the app:"
echo "      uv run uvicorn pharmagraphrag.api.main:app --reload &"
echo "      uv run streamlit run src/pharmagraphrag/ui/app.py"
echo ""
echo " Or run tests: uv run pytest"
echo ""
