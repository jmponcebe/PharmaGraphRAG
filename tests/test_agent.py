"""Tests for the LangGraph agent module.

All external dependencies (Neo4j, ChromaDB, LLM) are mocked.
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from pharmagraphrag.agent.graph import AgentResponse
from pharmagraphrag.agent.tools import ALL_TOOLS
from pharmagraphrag.api.main import app
from pharmagraphrag.api.models import AgentQueryResponse, ToolCallInfo

client = TestClient(app)


# ===========================================================================
# Tool definitions
# ===========================================================================


class TestToolDefinitions:
    """Verify tools are correctly defined for the agent."""

    def test_all_tools_count(self):
        assert len(ALL_TOOLS) == 5

    def test_tools_have_names(self):
        names = {t.name for t in ALL_TOOLS}
        assert "search_drug_info" in names
        assert "find_drugs_for_adverse_event" in names
        assert "search_drug_labels" in names
        assert "list_drug_interactions" in names
        assert "search_drugs_by_name" in names

    def test_tools_have_descriptions(self):
        for tool in ALL_TOOLS:
            assert tool.description, f"Tool {tool.name} missing description"


# ===========================================================================
# Tool execution (mocked backends)
# ===========================================================================


class TestToolExecution:
    """Test individual tool functions with mocked graph/vector backends."""

    @patch("pharmagraphrag.agent.tools.queries.get_drug_full_context")
    @patch("pharmagraphrag.agent.tools.queries.format_graph_context")
    def test_search_drug_info_found(self, mock_format, mock_ctx):
        mock_ctx.return_value = {
            "drug_info": {"name": "ASPIRIN"},
            "adverse_events": [],
        }
        mock_format.return_value = "ASPIRIN: analgesic"

        from pharmagraphrag.agent.tools import search_drug_info

        result = search_drug_info.invoke({"drug_name": "aspirin"})
        assert "ASPIRIN" in result
        mock_ctx.assert_called_once_with("ASPIRIN")

    @patch("pharmagraphrag.agent.tools.queries.get_drug_full_context")
    def test_search_drug_info_not_found(self, mock_ctx):
        mock_ctx.return_value = {}

        from pharmagraphrag.agent.tools import search_drug_info

        result = search_drug_info.invoke({"drug_name": "FAKE_DRUG"})
        assert "No information" in result

    @patch("pharmagraphrag.agent.tools.queries.get_adverse_event_drugs")
    def test_find_drugs_for_event(self, mock_ae):
        mock_ae.return_value = [
            {"drug": "ASPIRIN", "report_count": 100},
            {"drug": "IBUPROFEN", "report_count": 50},
        ]

        from pharmagraphrag.agent.tools import find_drugs_for_adverse_event

        result = find_drugs_for_adverse_event.invoke({"event_name": "nausea"})
        assert "ASPIRIN" in result
        assert "100" in result

    @patch("pharmagraphrag.agent.tools.queries.get_adverse_event_drugs")
    def test_find_drugs_for_event_empty(self, mock_ae):
        mock_ae.return_value = []

        from pharmagraphrag.agent.tools import find_drugs_for_adverse_event

        result = find_drugs_for_adverse_event.invoke({"event_name": "UNKNOWN"})
        assert "No drugs found" in result

    @patch("pharmagraphrag.agent.tools.store.search")
    @patch("pharmagraphrag.agent.tools.store.format_vector_context")
    def test_search_drug_labels_global(self, mock_fmt, mock_search):
        mock_search.return_value = [{"text": "some text"}]
        mock_fmt.return_value = "Formatted context"

        from pharmagraphrag.agent.tools import search_drug_labels

        result = search_drug_labels.invoke({"query": "liver damage"})
        assert result == "Formatted context"
        mock_search.assert_called_once()

    @patch("pharmagraphrag.agent.tools.store.search_by_drug")
    @patch("pharmagraphrag.agent.tools.store.format_vector_context")
    def test_search_drug_labels_filtered(self, mock_fmt, mock_search):
        mock_search.return_value = [{"text": "aspirin text"}]
        mock_fmt.return_value = "Aspirin context"

        from pharmagraphrag.agent.tools import search_drug_labels

        result = search_drug_labels.invoke({"query": "side effects", "drug_name": "aspirin"})
        assert result == "Aspirin context"
        mock_search.assert_called_once_with("side effects", "ASPIRIN", n_results=5)

    @patch("pharmagraphrag.agent.tools.queries.get_drug_interactions")
    def test_list_interactions(self, mock_ix):
        mock_ix.return_value = [
            {"drug": "WARFARIN", "source": "dailymed", "description": "increased bleeding risk"}
        ]

        from pharmagraphrag.agent.tools import list_drug_interactions

        result = list_drug_interactions.invoke({"drug_name": "aspirin"})
        assert "WARFARIN" in result
        assert "bleeding" in result

    @patch("pharmagraphrag.agent.tools.queries.get_drug_interactions")
    def test_list_interactions_empty(self, mock_ix):
        mock_ix.return_value = []

        from pharmagraphrag.agent.tools import list_drug_interactions

        result = list_drug_interactions.invoke({"drug_name": "water"})
        assert "No known interactions" in result

    @patch("pharmagraphrag.agent.tools.queries.search_drugs")
    def test_search_drugs_by_name(self, mock_search):
        mock_search.return_value = ["ASPIRIN", "ASPIRIN COMPLEX"]

        from pharmagraphrag.agent.tools import search_drugs_by_name

        result = search_drugs_by_name.invoke({"query": "aspir"})
        assert "ASPIRIN" in result


# ===========================================================================
# AgentResponse model
# ===========================================================================


class TestAgentResponse:
    """Test the AgentResponse dataclass."""

    def test_ok_with_answer(self):
        r = AgentResponse(answer="Some answer")
        assert r.ok

    def test_not_ok_with_error(self):
        r = AgentResponse(error="Something failed")
        assert not r.ok

    def test_not_ok_empty(self):
        r = AgentResponse()
        assert not r.ok

    def test_tool_calls_default(self):
        r = AgentResponse(answer="test")
        assert r.tool_calls == []


# ===========================================================================
# API models
# ===========================================================================


class TestAgentApiModels:
    """Test agent-related Pydantic models."""

    def test_agent_query_response_defaults(self):
        r = AgentQueryResponse(question="test?")
        assert r.answer == ""
        assert r.tool_calls == []
        assert r.error is None

    def test_tool_call_info(self):
        t = ToolCallInfo(tool="search_drug_info", args={"drug_name": "ASPIRIN"})
        assert t.tool == "search_drug_info"


# ===========================================================================
# POST /agent/query endpoint
# ===========================================================================


class TestAgentEndpoint:
    """Test the /agent/query endpoint with mocked agent."""

    @patch("pharmagraphrag.agent.graph.run_agent")
    def test_agent_query_success(self, mock_agent):
        mock_agent.return_value = AgentResponse(
            answer="Aspirin is an analgesic.",
            tool_calls=[{"tool": "search_drug_info", "args": {"drug_name": "ASPIRIN"}}],
        )

        resp = client.post("/agent/query", json={"question": "What is aspirin?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "analgesic" in data["answer"]
        assert len(data["tool_calls"]) == 1
        assert data["error"] is None

    @patch("pharmagraphrag.agent.graph.run_agent")
    def test_agent_query_error(self, mock_agent):
        mock_agent.return_value = AgentResponse(error="LLM unavailable")

        resp = client.post("/agent/query", json={"question": "test question?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] == "LLM unavailable"

    def test_agent_query_validation_error(self):
        resp = client.post("/agent/query", json={"question": "ab"})
        assert resp.status_code == 422

    @patch("pharmagraphrag.agent.graph.run_agent")
    def test_agent_query_exception(self, mock_agent):
        mock_agent.side_effect = RuntimeError("boom")

        resp = client.post("/agent/query", json={"question": "test question?"})
        assert resp.status_code == 500
