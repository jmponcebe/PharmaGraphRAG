"""Tests for the Streamlit UI components.

Tests the data-handling logic in components.py and app.py
without requiring a running Streamlit server.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# ChatMessage dataclass
# ---------------------------------------------------------------------------


def test_chat_message_defaults():
    """ChatMessage should have sensible defaults."""
    from pharmagraphrag.ui.app import ChatMessage

    msg = ChatMessage(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"
    assert msg.sources_graph == {}
    assert msg.sources_vector == []
    assert msg.drugs_extracted == []
    assert msg.drugs_found == []
    assert msg.llm_provider == ""
    assert msg.llm_model == ""
    assert msg.error is None


def test_chat_message_with_sources():
    """ChatMessage should store graph and vector sources."""
    from pharmagraphrag.ui.app import ChatMessage

    msg = ChatMessage(
        role="assistant",
        content="answer",
        sources_graph={"IBUPROFEN": {"drug_info": {"name": "IBUPROFEN"}}},
        sources_vector=[{"id": "1", "text": "chunk", "metadata": {}}],
        drugs_extracted=["IBUPROFEN"],
        drugs_found=["IBUPROFEN"],
        llm_provider="gemini",
        llm_model="gemini-2.0-flash",
    )
    assert msg.sources_graph["IBUPROFEN"]["drug_info"]["name"] == "IBUPROFEN"
    assert len(msg.sources_vector) == 1
    assert msg.drugs_extracted == ["IBUPROFEN"]
    assert msg.llm_provider == "gemini"


# ---------------------------------------------------------------------------
# Components — render_sources (logic tests)
# ---------------------------------------------------------------------------


class TestRenderSources:
    """Test render_sources component data handling."""

    def test_empty_sources(self):
        """render_sources with no data should call st.info."""
        from pharmagraphrag.ui.components import render_sources

        with patch("pharmagraphrag.ui.components.st") as mock_st:
            render_sources({}, [])
            mock_st.info.assert_called_once()

    def test_graph_sources_only(self):
        """render_sources with graph data should render expander."""
        from pharmagraphrag.ui.components import render_sources

        graph = {
            "ASPIRIN": {
                "drug_info": {"name": "ASPIRIN", "brand_names": ["Bayer"], "route": "oral"},
                "adverse_events": [{"adverse_event": "NAUSEA", "report_count": 100}],
                "interactions": [],
                "outcomes": [],
                "categories": [],
            }
        }

        with patch("pharmagraphrag.ui.components.st") as mock_st:
            # Mock expander context manager
            mock_expander = MagicMock()
            mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
            mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)

            render_sources(graph, [])

            # Should have called expander for graph
            mock_st.expander.assert_called()

    def test_vector_sources_only(self):
        """render_sources with vector data should render expander."""
        from pharmagraphrag.ui.components import render_sources

        vector = [
            {
                "id": "1",
                "text": "Ibuprofen may cause stomach bleeding.",
                "distance": 0.2,
                "metadata": {"drug_name": "IBUPROFEN", "section": "warnings"},
            }
        ]

        with patch("pharmagraphrag.ui.components.st") as mock_st:
            mock_expander = MagicMock()
            mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
            mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)

            render_sources({}, vector)

            mock_st.expander.assert_called()


# ---------------------------------------------------------------------------
# Components — render_graph (logic tests)
# ---------------------------------------------------------------------------


class TestRenderGraph:
    """Test render_graph component data handling."""

    def test_empty_graph(self):
        """render_graph with no data should show info message."""
        from pharmagraphrag.ui.components import render_graph

        with patch("pharmagraphrag.ui.components.st") as mock_st:
            render_graph({})
            mock_st.info.assert_called_once()

    @patch("pharmagraphrag.ui.components.st")
    def test_graph_with_data_creates_nodes(self, mock_st):
        """render_graph should create nodes and edges from graph data."""
        graph_raw = {
            "WARFARIN": {
                "drug_info": {"name": "WARFARIN"},
                "adverse_events": [
                    {"adverse_event": "BLEEDING", "report_count": 5000},
                    {"adverse_event": "BRUISING", "report_count": 1200},
                ],
                "interactions": [
                    {"interacting_drug": "ASPIRIN", "description": "Increased bleeding risk"},
                ],
                "outcomes": [
                    {"outcome_code": "DE", "outcome_description": "Death", "report_count": 300},
                ],
                "categories": ["Anticoagulant"],
            }
        }

        mock_agraph = MagicMock()
        mock_module = MagicMock()
        mock_module.agraph = mock_agraph
        mock_module.Config = MagicMock()
        mock_module.Node = MagicMock(side_effect=lambda **kw: kw)
        mock_module.Edge = MagicMock(side_effect=lambda **kw: kw)

        with patch.dict("sys.modules", {"streamlit_agraph": mock_module}):
            # Re-import to pick up mocked module
            import importlib

            import pharmagraphrag.ui.components as comp_mod

            importlib.reload(comp_mod)
            comp_mod.render_graph(graph_raw)

            # agraph should have been called
            mock_agraph.assert_called_once()

            call_kwargs = mock_agraph.call_args
            nodes = call_kwargs.kwargs.get("nodes", [])
            edges = call_kwargs.kwargs.get("edges", [])

            # Nodes: WARFARIN, BLEEDING, BRUISING, ASPIRIN, Death, Anticoagulant = 6
            assert len(nodes) == 6
            # Edges: 2 adverse events + 1 interaction + 1 outcome + 1 category = 5
            assert len(edges) == 5


# ---------------------------------------------------------------------------
# Components — render_drug_explorer
# ---------------------------------------------------------------------------


class TestDrugExplorer:
    """Test drug explorer sidebar component."""

    @patch("pharmagraphrag.ui.components.st")
    def test_short_query_returns_none(self, mock_st):
        """Explorer should return None for short queries."""
        from pharmagraphrag.ui.components import render_drug_explorer

        mock_st.sidebar.text_input.return_value = "a"
        result = render_drug_explorer()
        assert result is None

    @patch("pharmagraphrag.ui.components.st")
    def test_empty_query_returns_none(self, mock_st):
        """Explorer should return None for empty queries."""
        from pharmagraphrag.ui.components import render_drug_explorer

        mock_st.sidebar.text_input.return_value = ""
        result = render_drug_explorer()
        assert result is None


# ---------------------------------------------------------------------------
# App — _process_question
# ---------------------------------------------------------------------------


class TestProcessQuestion:
    """Test the query processing function."""

    @patch("pharmagraphrag.ui.app.st")
    def test_process_question_success(self, mock_st):
        """_process_question should return a ChatMessage on success."""
        mock_st.session_state = MagicMock()
        mock_st.session_state.settings = {
            "use_graph": True,
            "use_vector": True,
            "use_llm": True,
            "n_results": 5,
            "llm_provider": "gemini",
        }

        from pharmagraphrag.engine.entity_extractor import ExtractedEntities
        from pharmagraphrag.engine.query_engine import QueryResult
        from pharmagraphrag.engine.retriever import RetrievedContext
        from pharmagraphrag.llm.client import LLMResponse

        mock_result = QueryResult(
            question="test",
            entities=ExtractedEntities(drugs=["ASPIRIN"]),
            context=RetrievedContext(
                graph_context="graph data",
                graph_raw={"ASPIRIN": {}},
                drugs_found=["ASPIRIN"],
            ),
            system_prompt="sys",
            user_prompt="user",
        )

        mock_llm = LLMResponse(
            text="Aspirin is an NSAID.",
            model="gemini-2.0-flash",
            provider="gemini",
        )

        with (
            patch(
                "pharmagraphrag.engine.query_engine.process_query",
                return_value=mock_result,
            ),
            patch(
                "pharmagraphrag.llm.client.generate_answer",
                return_value=mock_llm,
            ),
        ):
            from pharmagraphrag.ui.app import _process_question

            msg = _process_question("What is aspirin?")

        assert msg.role == "assistant"
        assert "Aspirin" in msg.content
        assert msg.llm_provider == "gemini"
        assert msg.drugs_extracted == ["ASPIRIN"]

    @patch("pharmagraphrag.ui.app.st")
    def test_process_question_no_llm(self, mock_st):
        """_process_question with use_llm=False should return context only."""
        mock_st.session_state = MagicMock()
        mock_st.session_state.settings = {
            "use_graph": True,
            "use_vector": True,
            "use_llm": False,
            "n_results": 5,
            "llm_provider": "gemini",
        }

        from pharmagraphrag.engine.entity_extractor import ExtractedEntities
        from pharmagraphrag.engine.query_engine import QueryResult
        from pharmagraphrag.engine.retriever import RetrievedContext

        mock_result = QueryResult(
            question="test",
            entities=ExtractedEntities(drugs=["METFORMIN"]),
            context=RetrievedContext(
                vector_context="some vector text",
            ),
            system_prompt="sys",
            user_prompt="context here",
        )

        with patch(
            "pharmagraphrag.engine.query_engine.process_query",
            return_value=mock_result,
        ):
            from pharmagraphrag.ui.app import _process_question

            msg = _process_question("Tell me about metformin")

        assert msg.role == "assistant"
        assert "solo recuperación" in msg.content.lower() or "context" in msg.content.lower()
        assert msg.llm_provider == ""

    @patch("pharmagraphrag.ui.app.st")
    def test_process_question_error(self, mock_st):
        """_process_question should handle errors gracefully."""
        mock_st.session_state = MagicMock()
        mock_st.session_state.settings = {
            "use_graph": True,
            "use_vector": True,
            "use_llm": True,
            "n_results": 5,
            "llm_provider": "gemini",
        }

        with patch(
            "pharmagraphrag.engine.query_engine.process_query",
            side_effect=RuntimeError("Neo4j down"),
        ):
            from pharmagraphrag.ui.app import _process_question

            msg = _process_question("broken query")

        assert msg.role == "assistant"
        assert "Error" in msg.content or "error" in msg.content.lower()
        assert msg.error is not None


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


class _DictLike(dict):
    """Dict that also supports attribute access (like st.session_state)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as err:
            raise AttributeError(key) from err

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as err:
            raise AttributeError(key) from err


class TestSessionInit:
    """Test session state initialization."""

    @patch("pharmagraphrag.ui.app.st")
    def test_init_session_creates_defaults(self, mock_st):
        """_init_session should set default values."""
        mock_st.session_state = _DictLike()

        from pharmagraphrag.ui.app import _init_session

        _init_session()

        assert "messages" in mock_st.session_state
        assert "settings" in mock_st.session_state
        assert mock_st.session_state["settings"]["use_graph"] is True
        assert mock_st.session_state["settings"]["use_vector"] is True
        assert mock_st.session_state["settings"]["use_llm"] is True

    @patch("pharmagraphrag.ui.app.st")
    def test_init_session_preserves_existing(self, mock_st):
        """_init_session should not overwrite existing state."""
        existing_msgs = [{"role": "user", "content": "hi"}]
        mock_st.session_state = _DictLike({"messages": existing_msgs})

        from pharmagraphrag.ui.app import _init_session

        _init_session()

        assert mock_st.session_state["messages"] is existing_msgs
