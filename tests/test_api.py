"""Tests for the FastAPI REST API.

Uses FastAPI TestClient with mocked backends.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from pharmagraphrag.api.main import app
from pharmagraphrag.api.models import (
    DrugInfoResponse,
    HealthResponse,
    QueryResponse,
    SourceInfo,
)
from pharmagraphrag.engine.entity_extractor import ExtractedEntities
from pharmagraphrag.engine.query_engine import QueryResult
from pharmagraphrag.engine.retriever import RetrievedContext
from pharmagraphrag.llm.client import LLMResponse

client = TestClient(app)


# ===========================================================================
# Model tests
# ===========================================================================


class TestApiModels:
    """Tests for Pydantic request/response models."""

    def test_query_response_defaults(self):
        r = QueryResponse(question="test?")
        assert r.question == "test?"
        assert r.answer == ""
        assert r.sources == []

    def test_source_info(self):
        s = SourceInfo(type="graph", drug="ASPIRIN", section="", snippet="data")
        assert s.type == "graph"

    def test_health_response_defaults(self):
        h = HealthResponse()
        assert h.status == "ok"

    def test_drug_info_response(self):
        d = DrugInfoResponse(name="ASPIRIN")
        assert d.adverse_events == []


# ===========================================================================
# POST /query
# ===========================================================================


class TestQueryEndpoint:
    """Tests for the /query endpoint."""

    @patch("pharmagraphrag.llm.client.generate_answer")
    @patch("pharmagraphrag.engine.query_engine.process_query")
    def test_successful_query(self, mock_process, mock_llm):
        mock_process.return_value = QueryResult(
            question="What are the side effects of ibuprofen?",
            entities=ExtractedEntities(drugs=["IBUPROFEN"]),
            context=RetrievedContext(
                graph_context="Drug: IBUPROFEN\nAdverse events: NAUSEA",
                vector_context="[Source 1] text...",
                drugs_found=["IBUPROFEN"],
                vector_raw=[
                    {
                        "id": "1",
                        "text": "Ibuprofen may cause nausea...",
                        "metadata": {"drug_name": "IBUPROFEN", "section": "adverse_reactions"},
                        "distance": 0.3,
                    }
                ],
            ),
            system_prompt="SYS",
            user_prompt="USR",
        )
        mock_llm.return_value = LLMResponse(
            text="Ibuprofen can cause nausea and headache.",
            model="gemini-2.0-flash",
            provider="gemini",
        )

        resp = client.post("/query", json={"question": "What are the side effects of ibuprofen?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["question"] == "What are the side effects of ibuprofen?"
        assert "nausea" in data["answer"].lower()
        assert "IBUPROFEN" in data["drugs_extracted"]
        assert data["has_graph_context"] is True
        assert len(data["sources"]) >= 1
        assert data["llm_provider"] == "gemini"

    @patch("pharmagraphrag.llm.client.generate_answer")
    @patch("pharmagraphrag.engine.query_engine.process_query")
    def test_query_without_llm(self, mock_process, mock_llm):
        mock_process.return_value = QueryResult(
            question="test?",
            entities=ExtractedEntities(drugs=["ASPIRIN"]),
            context=RetrievedContext(
                graph_context="Drug: ASPIRIN",
                drugs_found=["ASPIRIN"],
            ),
            system_prompt="SYS",
            user_prompt="USR",
        )

        resp = client.post("/query", json={"question": "test?", "use_llm": False})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == ""
        mock_llm.assert_not_called()

    def test_query_validation_short_question(self):
        resp = client.post("/query", json={"question": "ab"})
        assert resp.status_code == 422

    def test_query_validation_missing_question(self):
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    @patch("pharmagraphrag.llm.client.generate_answer")
    @patch("pharmagraphrag.engine.query_engine.process_query")
    def test_query_llm_error(self, mock_process, mock_llm):
        mock_process.return_value = QueryResult(
            question="test?",
            entities=ExtractedEntities(drugs=[]),
            context=RetrievedContext(),
            system_prompt="SYS",
            user_prompt="USR",
        )
        mock_llm.return_value = LLMResponse(
            text="",
            model="gemini-2.0-flash",
            provider="gemini",
            error="API key invalid",
        )

        resp = client.post("/query", json={"question": "test query?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] == "API key invalid"
        assert data["answer"] == ""


# ===========================================================================
# GET /drug/{name}
# ===========================================================================


class TestDrugEndpoint:
    """Tests for the /drug/{name} endpoint."""

    @patch("pharmagraphrag.graph.queries.get_drug_full_context")
    def test_drug_found(self, mock_ctx):
        mock_ctx.return_value = {
            "drug_info": {
                "name": "ASPIRIN",
                "generic_names": ["ASPIRIN"],
                "brand_names": ["BAYER"],
                "category": "NSAID",
                "route": "ORAL",
            },
            "adverse_events": [
                {"adverse_event": "NAUSEA", "report_count": 100},
            ],
            "interactions": [],
            "outcomes": [],
            "categories": ["NSAID"],
        }

        resp = client.get("/drug/aspirin")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "ASPIRIN"
        assert len(data["adverse_events"]) == 1
        assert data["categories"] == ["NSAID"]

    @patch("pharmagraphrag.graph.queries.get_drug_full_context")
    def test_drug_not_found(self, mock_ctx):
        mock_ctx.return_value = {
            "drug_info": None,
            "adverse_events": [],
            "interactions": [],
            "outcomes": [],
            "categories": [],
        }

        resp = client.get("/drug/nonexistent")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


# ===========================================================================
# GET /health
# ===========================================================================


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    @patch("pharmagraphrag.vectorstore.store.get_collection")
    @patch("pharmagraphrag.graph.queries.search_drugs")
    def test_healthy(self, mock_search, mock_coll):
        mock_search.return_value = ["ASPIRIN"]
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5000
        mock_coll.return_value = mock_collection

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "ok" in data["neo4j"]
        assert "5000" in data["chromadb"]

    @patch("pharmagraphrag.vectorstore.store.get_collection")
    @patch("pharmagraphrag.graph.queries.search_drugs")
    def test_neo4j_down(self, mock_search, mock_coll):
        mock_search.side_effect = ConnectionError("connection refused")
        mock_collection = MagicMock()
        mock_collection.count.return_value = 100
        mock_coll.return_value = mock_collection

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data["neo4j"]
