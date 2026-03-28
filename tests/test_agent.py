"""Tests for the LangGraph agent module.

All external dependencies (Neo4j, ChromaDB, LLM) are mocked.
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from pharmagraphrag.agent.graph import AgentResponse, StructuredResponse
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
        assert len(ALL_TOOLS) == 9

    def test_tools_have_names(self):
        names = {t.name for t in ALL_TOOLS}
        assert "search_drug_info" in names
        assert "find_drugs_for_adverse_event" in names
        assert "search_drug_labels" in names
        assert "list_drug_interactions" in names
        assert "search_drugs_by_name" in names
        assert "search_adverse_events" in names
        assert "get_drug_outcomes" in names
        assert "compare_drugs" in names
        assert "find_drugs_by_category" in names

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
            {"drug_name": "ASPIRIN", "report_count": 100},
            {"drug_name": "IBUPROFEN", "report_count": 50},
        ]

        from pharmagraphrag.agent.tools import find_drugs_for_adverse_event

        result = find_drugs_for_adverse_event.invoke({"event_name": "nausea"})
        assert "ASPIRIN" in result
        assert "100" in result

    @patch("pharmagraphrag.agent.tools.queries.search_adverse_events")
    @patch("pharmagraphrag.agent.tools.queries.get_adverse_event_drugs")
    def test_find_drugs_for_event_empty_no_similar(self, mock_ae, mock_search):
        mock_ae.return_value = []
        mock_search.return_value = []

        from pharmagraphrag.agent.tools import find_drugs_for_adverse_event

        result = find_drugs_for_adverse_event.invoke({"event_name": "UNKNOWN"})
        assert "No drugs found" in result

    @patch("pharmagraphrag.agent.tools.queries.search_adverse_events")
    @patch("pharmagraphrag.agent.tools.queries.get_adverse_event_drugs")
    def test_find_drugs_for_event_suggests_similar(self, mock_ae, mock_search):
        mock_ae.return_value = []
        mock_search.return_value = [
            {"name": "HEPATOTOXICITY", "total_reports": 500},
            {"name": "LIVER INJURY", "total_reports": 300},
        ]

        from pharmagraphrag.agent.tools import find_drugs_for_adverse_event

        result = find_drugs_for_adverse_event.invoke({"event_name": "LIVER DAMAGE"})
        assert "No exact match" in result
        assert "HEPATOTOXICITY" in result
        assert "LIVER INJURY" in result

    @patch("pharmagraphrag.agent.tools.queries.search_adverse_events")
    def test_search_adverse_events_found(self, mock_search):
        mock_search.return_value = [
            {"name": "HEPATOTOXICITY", "total_reports": 500},
            {"name": "LIVER DISORDER", "total_reports": 200},
        ]

        from pharmagraphrag.agent.tools import search_adverse_events

        result = search_adverse_events.invoke({"query": "LIVER"})
        assert "HEPATOTOXICITY" in result
        assert "500" in result

    @patch("pharmagraphrag.agent.tools.queries.search_adverse_events")
    def test_search_adverse_events_empty(self, mock_search):
        mock_search.return_value = []

        from pharmagraphrag.agent.tools import search_adverse_events

        result = search_adverse_events.invoke({"query": "XYZNOTFOUND"})
        assert "No adverse events found" in result

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
            {
                "interacting_drug": "WARFARIN",
                "source": "DailyMed",
                "description": "increased bleeding risk",
            }
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

    @patch("pharmagraphrag.agent.tools.queries.get_drug_outcomes")
    def test_get_drug_outcomes_found(self, mock_outcomes):
        mock_outcomes.return_value = [
            {"outcome_code": "HO", "outcome_description": "Hospitalization", "report_count": 200},
            {"outcome_code": "DE", "outcome_description": "Death", "report_count": 50},
        ]

        from pharmagraphrag.agent.tools import get_drug_outcomes

        result = get_drug_outcomes.invoke({"drug_name": "warfarin"})
        assert "Hospitalization" in result
        assert "200" in result
        assert "Death" in result

    @patch("pharmagraphrag.agent.tools.queries.get_drug_outcomes")
    def test_get_drug_outcomes_empty(self, mock_outcomes):
        mock_outcomes.return_value = []

        from pharmagraphrag.agent.tools import get_drug_outcomes

        result = get_drug_outcomes.invoke({"drug_name": "UNKNOWN"})
        assert "No outcome data" in result

    @patch("pharmagraphrag.agent.tools.queries.get_drug_full_context")
    def test_compare_drugs_both_found(self, mock_ctx):
        mock_ctx.side_effect = [
            {
                "drug_info": {"name": "ASPIRIN"},
                "adverse_events": [{"adverse_event": "NAUSEA", "report_count": 100}],
                "outcomes": [
                    {
                        "outcome_code": "HO",
                        "outcome_description": "Hospitalization",
                        "report_count": 50,
                    }
                ],
                "interactions": [],
                "categories": ["NSAID"],
            },
            {
                "drug_info": {"name": "IBUPROFEN"},
                "adverse_events": [{"adverse_event": "HEADACHE", "report_count": 80}],
                "outcomes": [],
                "interactions": [],
                "categories": ["NSAID"],
            },
        ]

        from pharmagraphrag.agent.tools import compare_drugs

        result = compare_drugs.invoke({"drug_name_1": "aspirin", "drug_name_2": "ibuprofen"})
        assert "ASPIRIN" in result
        assert "IBUPROFEN" in result
        assert "NAUSEA" in result
        assert "HEADACHE" in result

    @patch("pharmagraphrag.agent.tools.queries.get_drug_full_context")
    def test_compare_drugs_one_not_found(self, mock_ctx):
        mock_ctx.side_effect = [
            {
                "drug_info": {"name": "ASPIRIN"},
                "adverse_events": [],
                "outcomes": [],
                "interactions": [],
                "categories": [],
            },
            {},
        ]

        from pharmagraphrag.agent.tools import compare_drugs

        result = compare_drugs.invoke({"drug_name_1": "aspirin", "drug_name_2": "FAKE"})
        assert "not found" in result

    @patch("pharmagraphrag.agent.tools.queries.get_drugs_by_category")
    def test_find_drugs_by_category_found(self, mock_cat):
        mock_cat.return_value = [
            {"drug_name": "ASPIRIN", "category": "NSAID"},
            {"drug_name": "IBUPROFEN", "category": "NSAID"},
        ]

        from pharmagraphrag.agent.tools import find_drugs_by_category

        result = find_drugs_by_category.invoke({"category": "nsaid"})
        assert "ASPIRIN" in result
        assert "IBUPROFEN" in result

    @patch("pharmagraphrag.agent.tools.queries.get_drugs_by_category")
    def test_find_drugs_by_category_empty(self, mock_cat):
        mock_cat.return_value = []

        from pharmagraphrag.agent.tools import find_drugs_by_category

        result = find_drugs_by_category.invoke({"category": "XYZNOTFOUND"})
        assert "No drugs found" in result


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

    def test_structured_output_fields_default(self):
        r = AgentResponse(answer="test")
        assert r.drugs_mentioned == []
        assert r.adverse_events_mentioned == []
        assert r.confidence == ""
        assert r.follow_up_suggestions == []

    def test_structured_output_fields_populated(self):
        r = AgentResponse(
            answer="Aspirin is an NSAID.",
            drugs_mentioned=["ASPIRIN"],
            adverse_events_mentioned=["NAUSEA"],
            confidence="high",
            follow_up_suggestions=["What are the interactions of aspirin?"],
        )
        assert r.drugs_mentioned == ["ASPIRIN"]
        assert r.adverse_events_mentioned == ["NAUSEA"]
        assert r.confidence == "high"
        assert len(r.follow_up_suggestions) == 1


class TestStructuredResponse:
    """Test the StructuredResponse Pydantic model used for agent output format."""

    def test_defaults(self):
        s = StructuredResponse(answer="Test answer")
        assert s.drugs_mentioned == []
        assert s.adverse_events_mentioned == []
        assert s.confidence == "medium"
        assert s.follow_up_suggestions == []

    def test_full(self):
        s = StructuredResponse(
            answer="Aspirin causes nausea.",
            drugs_mentioned=["ASPIRIN"],
            adverse_events_mentioned=["NAUSEA"],
            confidence="high",
            follow_up_suggestions=["What is the dose?"],
        )
        assert s.answer == "Aspirin causes nausea."
        assert len(s.drugs_mentioned) == 1
        assert s.confidence == "high"


# ===========================================================================
# API models
# ===========================================================================


class TestAgentApiModels:
    """Test agent-related Pydantic models."""

    def test_agent_query_response_defaults(self):
        r = AgentQueryResponse(question="test?")
        assert r.answer == ""
        assert r.drugs_mentioned == []
        assert r.adverse_events_mentioned == []
        assert r.confidence == ""
        assert r.follow_up_suggestions == []
        assert r.tool_calls == []
        assert r.error is None

    def test_tool_call_info(self):
        t = ToolCallInfo(tool="search_drug_info", args={"drug_name": "ASPIRIN"})
        assert t.tool == "search_drug_info"

    def test_agent_query_request_with_session_id(self):
        """Verify AgentQueryRequest accepts session_id."""
        from pharmagraphrag.api.models import AgentQueryRequest

        req = AgentQueryRequest(question="test question", session_id="abc-123")
        assert req.session_id == "abc-123"

    def test_agent_query_request_without_session_id(self):
        """session_id defaults to None."""
        from pharmagraphrag.api.models import AgentQueryRequest

        req = AgentQueryRequest(question="test question")
        assert req.session_id is None


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

    @patch("pharmagraphrag.agent.graph.run_agent")
    def test_agent_query_with_session_id(self, mock_agent):
        """Verify session_id is passed as thread_id to run_agent."""
        mock_agent.return_value = AgentResponse(answer="Answer with memory.")

        resp = client.post(
            "/agent/query",
            json={"question": "follow up question", "session_id": "sess-123"},
        )
        assert resp.status_code == 200
        mock_agent.assert_called_once_with("follow up question", thread_id="sess-123")

    @patch("pharmagraphrag.agent.graph.run_agent")
    def test_agent_query_without_session_id(self, mock_agent):
        """Without session_id, thread_id should be None."""
        mock_agent.return_value = AgentResponse(answer="Stateless answer.")

        resp = client.post(
            "/agent/query",
            json={"question": "stateless question"},
        )
        assert resp.status_code == 200
        mock_agent.assert_called_once_with("stateless question", thread_id=None)


# ===========================================================================
# Multi-agent supervisor
# ===========================================================================


class TestMultiAgentDefinitions:
    """Verify multi-agent supervisor tools and prompts."""

    def test_supervisor_tools_count(self):
        from pharmagraphrag.agent.multi import SUPERVISOR_TOOLS

        assert len(SUPERVISOR_TOOLS) == 3

    def test_supervisor_tool_names(self):
        from pharmagraphrag.agent.multi import SUPERVISOR_TOOLS

        names = {t.name for t in SUPERVISOR_TOOLS}
        assert names == {"ask_drug_expert", "ask_safety_analyst", "ask_literature_researcher"}

    def test_sub_agent_tool_assignments(self):
        from pharmagraphrag.agent.multi import (
            DRUG_EXPERT_TOOLS,
            LITERATURE_RESEARCHER_TOOLS,
            SAFETY_ANALYST_TOOLS,
        )

        assert len(DRUG_EXPERT_TOOLS) == 4
        assert len(SAFETY_ANALYST_TOOLS) == 4
        assert len(LITERATURE_RESEARCHER_TOOLS) == 2


class TestCollectStructuredFromResults:
    """Test the _collect_structured_from_results helper."""

    @patch("pharmagraphrag.graph.queries.get_drug_full_context")
    def test_extracts_drug_from_drug_prefix(self, mock_ctx):
        from pharmagraphrag.agent.multi import _collect_structured_from_results

        mock_ctx.return_value = {"drug_info": {"name": "ASPIRIN"}}

        results = [{"tool": "ask_drug_expert", "content": "Drug: ASPIRIN\nSome details."}]
        graph, vector = _collect_structured_from_results(results)

        assert "ASPIRIN" in graph
        assert vector == []

    @patch("pharmagraphrag.graph.queries.get_drug_full_context")
    def test_extracts_drug_from_reports_pattern(self, mock_ctx):
        from pharmagraphrag.agent.multi import _collect_structured_from_results

        mock_ctx.return_value = {"drug_info": {"name": "WARFARIN"}}

        results = [{"tool": "ask_safety_analyst", "content": "- WARFARIN: 150 reports"}]
        graph, _vector = _collect_structured_from_results(results)

        assert "WARFARIN" in graph

    @patch("pharmagraphrag.graph.queries.get_drug_full_context")
    def test_extracts_drugs_from_comparison(self, mock_ctx):
        from pharmagraphrag.agent.multi import _collect_structured_from_results

        mock_ctx.return_value = {"drug_info": {"name": "test"}}

        results = [
            {"tool": "ask_safety_analyst", "content": "=== Comparison: ASPIRIN vs IBUPROFEN ==="}
        ]
        graph, _vector = _collect_structured_from_results(results)

        assert "ASPIRIN" in graph
        assert "IBUPROFEN" in graph

    def test_empty_results(self):
        from pharmagraphrag.agent.multi import _collect_structured_from_results

        graph, vector = _collect_structured_from_results([])
        assert graph == {}
        assert vector == []


class TestMultiAgentEndpoint:
    """Test the /agent/multi endpoint with mocked multi-agent."""

    @patch("pharmagraphrag.agent.multi.run_multi_agent")
    def test_multi_query_success(self, mock_multi):
        mock_multi.return_value = AgentResponse(
            answer="The Drug Expert reports that aspirin is an NSAID.",
            tool_calls=[{"tool": "ask_drug_expert", "args": {"question": "What is aspirin?"}}],
        )

        resp = client.post("/agent/multi", json={"question": "What is aspirin?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "NSAID" in data["answer"]
        assert len(data["tool_calls"]) == 1

    @patch("pharmagraphrag.agent.multi.run_multi_agent")
    def test_multi_query_error(self, mock_multi):
        mock_multi.return_value = AgentResponse(error="Sub-agent failed")

        resp = client.post("/agent/multi", json={"question": "test multi question?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] == "Sub-agent failed"

    @patch("pharmagraphrag.agent.multi.run_multi_agent")
    def test_multi_query_with_session_id(self, mock_multi):
        mock_multi.return_value = AgentResponse(answer="Memory answer.")

        resp = client.post(
            "/agent/multi",
            json={"question": "follow up about aspirin", "session_id": "multi-sess"},
        )
        assert resp.status_code == 200
        mock_multi.assert_called_once_with("follow up about aspirin", thread_id="multi-sess")

    @patch("pharmagraphrag.agent.multi.run_multi_agent")
    def test_multi_query_exception(self, mock_multi):
        mock_multi.side_effect = RuntimeError("supervisor crash")

        resp = client.post("/agent/multi", json={"question": "test crash scenario?"})
        assert resp.status_code == 500

    def test_multi_query_validation_error(self):
        resp = client.post("/agent/multi", json={"question": "ab"})
        assert resp.status_code == 422
