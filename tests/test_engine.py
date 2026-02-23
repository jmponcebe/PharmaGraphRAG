"""Tests for the GraphRAG query engine module.

Covers entity extraction, retrieval, and query orchestration.
All external dependencies (Neo4j, ChromaDB) are mocked.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from pharmagraphrag.engine.entity_extractor import (
    ExtractedEntities,
    _exact_match,
    _fuzzy_match,
    _normalize_query,
    extract_entities,
    reset_cache,
)
from pharmagraphrag.engine.query_engine import (
    QueryResult,
    _build_user_prompt,
    process_query,
)
from pharmagraphrag.engine.retriever import (
    RetrievedContext,
    retrieve_context,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DRUGS: set[str] = {
    "IBUPROFEN",
    "ASPIRIN",
    "WARFARIN",
    "METFORMIN",
    "ATORVASTATIN",
    "LISINOPRIL",
    "METOPROLOL",
    "INSULIN GLARGINE",
    "ACETAMINOPHEN",
    "AMOXICILLIN",
    "GABAPENTIN",
    "HYDROCHLOROTHIAZIDE",
}


@pytest.fixture(autouse=True)
def _reset_entity_cache():
    """Reset the drug-name cache before each test."""
    reset_cache()
    yield
    reset_cache()


# ===========================================================================
# entity_extractor tests
# ===========================================================================


class TestNormalizeQuery:
    """Tests for query normalisation."""

    def test_uppercase(self):
        assert _normalize_query("hello world") == "HELLO WORLD"

    def test_collapse_whitespace(self):
        assert _normalize_query("  hello   world  ") == "HELLO WORLD"

    def test_mixed_case(self):
        assert _normalize_query("What about Ibuprofen?") == "WHAT ABOUT IBUPROFEN?"


class TestExactMatch:
    """Tests for exact substring matching."""

    def test_single_drug(self):
        result = _exact_match("WHAT ARE SIDE EFFECTS OF IBUPROFEN", SAMPLE_DRUGS)
        assert "IBUPROFEN" in result

    def test_two_drugs(self):
        result = _exact_match(
            "DOES WARFARIN INTERACT WITH ASPIRIN",
            SAMPLE_DRUGS,
        )
        assert "WARFARIN" in result
        assert "ASPIRIN" in result

    def test_no_match(self):
        result = _exact_match("WHAT IS THE WEATHER TODAY", SAMPLE_DRUGS)
        assert result == []

    def test_multi_word_drug(self):
        result = _exact_match("TELL ME ABOUT INSULIN GLARGINE", SAMPLE_DRUGS)
        assert "INSULIN GLARGINE" in result

    def test_case_insensitive_via_normalisation(self):
        query = _normalize_query("tell me about aspirin")
        result = _exact_match(query, SAMPLE_DRUGS)
        assert "ASPIRIN" in result

    def test_no_partial_word_match(self):
        """A drug name embedded inside another word should not match."""
        # "ASPIRING" should not match "ASPIRIN"
        result = _exact_match("I AM ASPIRING TO BE HEALTHY", SAMPLE_DRUGS)
        assert "ASPIRIN" not in result


class TestFuzzyMatch:
    """Tests for fuzzy token matching."""

    def test_misspelled_drug(self):
        result = _fuzzy_match(["IBUPROFN"], SAMPLE_DRUGS, threshold=75)
        assert "IBUPROFEN" in result

    def test_no_match_below_threshold(self):
        result = _fuzzy_match(["XYZABC"], SAMPLE_DRUGS, threshold=80)
        assert result == []

    def test_exact_token(self):
        result = _fuzzy_match(["ASPIRIN"], SAMPLE_DRUGS)
        assert "ASPIRIN" in result

    def test_bigram_match(self):
        result = _fuzzy_match(
            ["INSULIN", "GLARGINE"],
            SAMPLE_DRUGS,
            threshold=80,
        )
        assert "INSULIN GLARGINE" in result

    def test_stop_word_skipped(self):
        """Stop words should be ignored even if they fuzzy-match."""
        result = _fuzzy_match(["DRUG", "THE", "EFFECTS"], SAMPLE_DRUGS)
        assert not result  # None of these should match actual drug names


class TestExtractEntities:
    """Tests for the full extract_entities pipeline."""

    @patch(
        "pharmagraphrag.engine.entity_extractor.get_known_drugs",
        return_value=SAMPLE_DRUGS,
    )
    def test_basic_extraction(self, _mock):
        result = extract_entities("What are the side effects of ibuprofen?")
        assert "IBUPROFEN" in result.drugs

    @patch(
        "pharmagraphrag.engine.entity_extractor.get_known_drugs",
        return_value=SAMPLE_DRUGS,
    )
    def test_multiple_drugs(self, _mock):
        result = extract_entities(
            "Does warfarin interact with aspirin?",
        )
        assert "WARFARIN" in result.drugs
        assert "ASPIRIN" in result.drugs

    @patch(
        "pharmagraphrag.engine.entity_extractor.get_known_drugs",
        return_value=SAMPLE_DRUGS,
    )
    def test_no_drugs_found(self, _mock):
        result = extract_entities("What is the meaning of life?")
        assert result.drugs == []
        assert result.is_empty()

    @patch(
        "pharmagraphrag.engine.entity_extractor.get_known_drugs",
        return_value=SAMPLE_DRUGS,
    )
    def test_fuzzy_fallback(self, _mock):
        result = extract_entities("Side effects of ibuprofn?", fuzzy=True)
        # Fuzzy should kick in since exact match fails for "ibuprofn"
        assert "IBUPROFEN" in result.drugs

    @patch(
        "pharmagraphrag.engine.entity_extractor.get_known_drugs",
        return_value=SAMPLE_DRUGS,
    )
    def test_raw_query_preserved(self, _mock):
        q = "Does metformin cause nausea?"
        result = extract_entities(q)
        assert result.raw_query == q

    @patch(
        "pharmagraphrag.engine.entity_extractor.get_known_drugs",
        return_value=SAMPLE_DRUGS,
    )
    def test_deduplication(self, _mock):
        result = extract_entities("Ibuprofen ibuprofen IBUPROFEN side effects")
        assert result.drugs.count("IBUPROFEN") == 1

    def test_extracted_entities_dataclass(self):
        e = ExtractedEntities(drugs=["ASPIRIN"], adverse_events=["NAUSEA"])
        assert not e.is_empty()
        assert e.drugs == ["ASPIRIN"]

    def test_extracted_entities_empty(self):
        e = ExtractedEntities()
        assert e.is_empty()


# ===========================================================================
# retriever tests
# ===========================================================================


class TestRetrievedContext:
    """Tests for the RetrievedContext data class."""

    def test_empty(self):
        ctx = RetrievedContext()
        assert ctx.is_empty
        assert not ctx.has_graph
        assert not ctx.has_vector

    def test_graph_only(self):
        ctx = RetrievedContext(graph_context="Drug: ASPIRIN\nAdverse events...")
        assert ctx.has_graph
        assert not ctx.has_vector
        assert not ctx.is_empty

    def test_vector_only(self):
        ctx = RetrievedContext(vector_context="[Source 1: ASPIRIN]...")
        assert not ctx.has_graph
        assert ctx.has_vector
        assert not ctx.is_empty

    def test_both(self):
        ctx = RetrievedContext(
            graph_context="Drug: ASPIRIN",
            vector_context="[Source 1]...",
        )
        assert ctx.has_graph
        assert ctx.has_vector


class TestRetrieveContext:
    """Tests for the retrieve_context function with mocked backends."""

    @patch("pharmagraphrag.engine.retriever._retrieve_vector", return_value=("", []))
    @patch(
        "pharmagraphrag.engine.retriever._retrieve_graph",
        return_value=("Drug: ASPIRIN\nAdverse events: NAUSEA", {"ASPIRIN": {}}, ["ASPIRIN"]),
    )
    def test_graph_retrieval(self, _mock_graph, _mock_vector):
        ctx = retrieve_context(drugs=["ASPIRIN"], query="aspirin side effects")
        assert ctx.has_graph
        assert "ASPIRIN" in ctx.graph_context

    @patch(
        "pharmagraphrag.engine.retriever._retrieve_graph",
        return_value=("", {}, []),
    )
    @patch(
        "pharmagraphrag.engine.retriever._retrieve_vector",
        return_value=(
            "[Source 1: ASPIRIN — Drug Interactions]\nText...",
            [{"id": "1", "text": "..."}],
        ),
    )
    def test_vector_retrieval(self, _mock_vector, _mock_graph):
        ctx = retrieve_context(drugs=["ASPIRIN"], query="aspirin interactions")
        assert ctx.has_vector
        assert "Source 1" in ctx.vector_context

    @patch("pharmagraphrag.engine.retriever._retrieve_vector", return_value=("", []))
    @patch(
        "pharmagraphrag.engine.retriever._retrieve_graph",
        return_value=("", {}, []),
    )
    def test_empty_results(self, _mock_graph, _mock_vector):
        ctx = retrieve_context(drugs=["UNKNOWNDRUG"], query="unknown thing")
        assert ctx.is_empty

    def test_no_drugs_no_graph(self):
        """Without drugs, graph retrieval is skipped."""
        with patch("pharmagraphrag.engine.retriever._retrieve_vector", return_value=("text", [])):
            ctx = retrieve_context(drugs=[], query="some query")
            assert not ctx.has_graph  # no drugs → no graph call


# ===========================================================================
# query_engine tests
# ===========================================================================


class TestQueryResult:
    """Tests for the QueryResult data class."""

    def test_prompt_assembly(self):
        qr = QueryResult(
            question="test?",
            system_prompt="SYS",
            user_prompt="USR",
        )
        assert qr.prompt == "SYS\n\nUSR"
        assert qr.has_context is False  # empty context

    def test_to_dict(self):
        qr = QueryResult(
            question="test?",
            entities=ExtractedEntities(drugs=["ASPIRIN"]),
        )
        d = qr.to_dict()
        assert d["question"] == "test?"
        assert "ASPIRIN" in d["drugs_extracted"]

    def test_has_context_with_graph(self):
        ctx = RetrievedContext(graph_context="some data")
        qr = QueryResult(question="test?", context=ctx)
        assert qr.has_context


class TestBuildUserPrompt:
    """Tests for prompt building."""

    def test_includes_graph_and_vector(self):
        ctx = RetrievedContext(
            graph_context="Drug: ASPIRIN",
            vector_context="[Source 1]...",
        )
        prompt = _build_user_prompt("What about aspirin?", ctx)
        assert "GRAPH CONTEXT" in prompt
        assert "TEXT CONTEXT" in prompt
        assert "Drug: ASPIRIN" in prompt
        assert "[Source 1]" in prompt
        assert "What about aspirin?" in prompt

    def test_empty_context_placeholders(self):
        ctx = RetrievedContext()
        prompt = _build_user_prompt("question?", ctx)
        assert "No structured graph data available" in prompt
        assert "No relevant text context found" in prompt


class TestProcessQuery:
    """Tests for the full process_query pipeline."""

    @patch(
        "pharmagraphrag.engine.query_engine.retrieve_context",
        return_value=RetrievedContext(
            graph_context="Drug: IBUPROFEN\nAdverse events: NAUSEA",
            vector_context="[Source 1: IBUPROFEN]...",
            drugs_found=["IBUPROFEN"],
        ),
    )
    @patch(
        "pharmagraphrag.engine.query_engine.extract_entities",
        return_value=ExtractedEntities(
            drugs=["IBUPROFEN"],
            raw_query="What are the side effects of ibuprofen?",
        ),
    )
    def test_full_pipeline(self, _mock_extract, _mock_retrieve):
        result = process_query("What are the side effects of ibuprofen?")
        assert result.question == "What are the side effects of ibuprofen?"
        assert "IBUPROFEN" in result.entities.drugs
        assert result.has_context
        assert "GRAPH CONTEXT" in result.user_prompt
        assert "TEXT CONTEXT" in result.user_prompt
        assert "pharmaceutical knowledge assistant" in result.system_prompt

    @patch(
        "pharmagraphrag.engine.query_engine.retrieve_context",
        return_value=RetrievedContext(),
    )
    @patch(
        "pharmagraphrag.engine.query_engine.extract_entities",
        return_value=ExtractedEntities(drugs=[], raw_query="hello"),
    )
    def test_no_entities_found(self, _mock_extract, _mock_retrieve):
        result = process_query("hello")
        assert result.entities.is_empty()
        assert not result.has_context
        # Prompt is still assembled (with placeholder context)
        assert "USER QUESTION: hello" in result.user_prompt
