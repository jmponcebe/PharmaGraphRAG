#!/usr/bin/env bash
# =============================================================
# PharmaGraphRAG — Codespaces / devcontainer setup
# =============================================================
# This script runs automatically when a Codespace is created.
# It installs dependencies, loads demo data, and starts the app
# so it's ready to use immediately.
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
    exit 1
fi
echo "  ✅ Neo4j is ready!"

# 6. Load demo data (knowledge graph + vector store)
echo ">>> Loading demo data (~2-3 minutes)..."
uv run python scripts/setup_demo.py

# 7. Start API server in background
echo ">>> Starting API server..."
nohup uv run uvicorn pharmagraphrag.api.main:app --host 0.0.0.0 --port 8000 > /tmp/api.log 2>&1 &
sleep 2
echo "  ✅ API running on port 8000"

# 8. Start Streamlit UI in background
echo ">>> Starting Streamlit UI..."
nohup uv run streamlit run src/pharmagraphrag/ui/app.py --server.port 8501 --server.headless true > /tmp/ui.log 2>&1 &
sleep 2
echo "  ✅ Streamlit running on port 8501"

echo ""
echo "============================================="
echo " ✅ Everything is ready!"
echo "============================================="
echo ""
echo " 🌐 Streamlit will open automatically in a new tab."
echo "    If not, click the 'Ports' tab and open port 8501."
echo ""
echo " 💬 Try asking: \"What are the side effects of metformin?\""
echo ""
echo " 🔑 Optional: Set GEMINI_API_KEY in .env for better answers"
echo "    Get a free key at: https://aistudio.google.com/apikey"
echo "    Without it, the system uses Ollama (if available) as fallback."
echo ""
echo " 🧪 Run tests: uv run pytest"
echo ""
