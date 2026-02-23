"""Tests for the vectorstore package — chunker, embedder, and store."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

    import chromadb

# ===================================================================
# chunker tests
# ===================================================================
from pharmagraphrag.vectorstore.chunker import (
    TextChunk,
    _clean_text,
    chunk_all_labels,
    chunk_drug_label,
    chunk_text,
    load_drug_label,
)


class TestCleanText:
    """Tests for _clean_text helper."""

    def test_removes_leading_section_number(self) -> None:
        assert _clean_text("7 DRUG INTERACTIONS some text") == "DRUG INTERACTIONS some text"

    def test_removes_subsection_number(self) -> None:
        assert _clean_text("12.1 Mechanism of Action blah") == "Mechanism of Action blah"

    def test_collapses_whitespace(self) -> None:
        assert _clean_text("hello   world\n\nfoo") == "hello world foo"

    def test_empty_string(self) -> None:
        assert _clean_text("") == ""


class TestChunkText:
    """Tests for chunk_text."""

    def test_empty_text_returns_empty(self) -> None:
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_returns_single_chunk(self) -> None:
        result = chunk_text("Hello world", chunk_size=1000)
        assert len(result) == 1
        assert "Hello world" in result[0]

    def test_long_text_produces_multiple_chunks(self) -> None:
        text = "word " * 500  # ~2500 chars
        result = chunk_text(text, chunk_size=500, chunk_overlap=100)
        assert len(result) > 1

    def test_overlap_between_chunks(self) -> None:
        # Create text that will definitely need splitting
        text = ". ".join([f"Sentence number {i}" for i in range(100)])
        result = chunk_text(text, chunk_size=200, chunk_overlap=50)
        assert len(result) >= 2
        # The end of chunk N should overlap with the start of chunk N+1
        # (at the character level, overlap means shared content)

    def test_respects_max_chunk_size(self) -> None:
        text = "x " * 1000
        result = chunk_text(text, chunk_size=300, chunk_overlap=50)
        for chunk in result:
            # Allow some tolerance for sentence boundary breaks
            assert len(chunk) <= 350


class TestTextChunk:
    """Tests for TextChunk dataclass."""

    def test_doc_id_format(self) -> None:
        tc = TextChunk(
            text="some text",
            drug_name="ASPIRIN",
            section="drug_interactions",
            chunk_index=2,
        )
        assert tc.doc_id == "ASPIRIN__drug_interactions__2"


class TestLoadDrugLabel:
    """Tests for load_drug_label."""

    def test_loads_valid_json(self, tmp_path: Path) -> None:
        data = {"drug_name": "TEST", "sections": {"warnings": "Be careful"}}
        filepath = tmp_path / "test.json"
        filepath.write_text(json.dumps(data), encoding="utf-8")

        result = load_drug_label(filepath)
        assert result is not None
        assert result["drug_name"] == "TEST"

    def test_returns_none_for_invalid_json(self, tmp_path: Path) -> None:
        filepath = tmp_path / "bad.json"
        filepath.write_text("not json at all", encoding="utf-8")

        result = load_drug_label(filepath)
        assert result is None

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        result = load_drug_label(tmp_path / "nonexistent.json")
        assert result is None


class TestChunkDrugLabel:
    """Tests for chunk_drug_label."""

    @pytest.fixture()
    def sample_label(self) -> dict:
        return {
            "drug_name": "WARFARIN",
            "generic_names": ["WARFARIN SODIUM"],
            "brand_names": ["Coumadin"],
            "route": ["ORAL"],
            "sections": {
                "drug_interactions": "Warfarin interacts with many drugs. " * 20,
                "adverse_reactions": "Bleeding is a common side effect. " * 10,
                "warnings": "Be careful with warfarin. " * 5,
            },
        }

    def test_produces_chunks(self, sample_label: dict) -> None:
        chunks = chunk_drug_label(sample_label, chunk_size=500, chunk_overlap=100)
        assert len(chunks) > 0
        assert all(isinstance(c, TextChunk) for c in chunks)

    def test_metadata_preserved(self, sample_label: dict) -> None:
        chunks = chunk_drug_label(sample_label)
        for c in chunks:
            assert c.drug_name == "WARFARIN"
            assert c.metadata["drug_name"] == "WARFARIN"
            assert c.metadata["generic_names"] == "WARFARIN SODIUM"
            assert c.metadata["brand_names"] == "Coumadin"

    def test_filter_sections(self, sample_label: dict) -> None:
        chunks = chunk_drug_label(sample_label, sections=["drug_interactions"])
        assert all(c.section == "drug_interactions" for c in chunks)

    def test_empty_sections(self) -> None:
        label = {"drug_name": "EMPTY", "sections": {}}
        chunks = chunk_drug_label(label)
        assert chunks == []

    def test_prefix_added(self, sample_label: dict) -> None:
        chunks = chunk_drug_label(sample_label)
        for c in chunks:
            assert c.text.startswith("Drug: WARFARIN |")


class TestChunkAllLabels:
    """Tests for chunk_all_labels."""

    def test_processes_directory(self, tmp_path: Path) -> None:
        # Create two sample JSON files
        for name in ["drug_a", "drug_b"]:
            data = {
                "drug_name": name.upper(),
                "generic_names": [],
                "brand_names": [],
                "route": [],
                "sections": {
                    "warnings": f"Warning for {name}. " * 10,
                },
            }
            (tmp_path / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")

        chunks = chunk_all_labels(dailymed_dir=tmp_path, chunk_size=500)
        assert len(chunks) > 0
        drug_names = {c.drug_name for c in chunks}
        assert "DRUG_A" in drug_names
        assert "DRUG_B" in drug_names

    def test_empty_directory(self, tmp_path: Path) -> None:
        chunks = chunk_all_labels(dailymed_dir=tmp_path)
        assert chunks == []


# ===================================================================
# embedder tests
# ===================================================================
from pharmagraphrag.vectorstore.embedder import (  # noqa: E402
    embed_single,
    embed_texts,
    get_embedding_dimension,
    get_model,
)


class TestEmbedder:
    """Tests for the embedder module.

    Uses real model for integration-style tests (model is small ~80MB).
    """

    def test_get_model_returns_model(self) -> None:
        model = get_model("all-MiniLM-L6-v2")
        assert model is not None

    def test_embedding_dimension(self) -> None:
        dim = get_embedding_dimension("all-MiniLM-L6-v2")
        assert dim == 384

    def test_embed_texts_returns_correct_shape(self) -> None:
        texts = ["hello world", "drug interactions with warfarin"]
        embeddings = embed_texts(texts, show_progress=False)
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384
        assert all(isinstance(v, float) for v in embeddings[0])

    def test_embed_empty_list(self) -> None:
        assert embed_texts([]) == []

    def test_embed_single(self) -> None:
        vec = embed_single("aspirin side effects", model_name="all-MiniLM-L6-v2")
        assert len(vec) == 384

    def test_similar_texts_have_close_embeddings(self) -> None:
        """Sanity check: similar texts should have higher cosine similarity."""
        import numpy as np

        texts = [
            "warfarin drug interactions",
            "warfarin interacts with aspirin",
            "the weather is sunny today",
        ]
        vecs = embed_texts(texts, normalize=True, show_progress=False)

        # Cosine similarity (vectors are normalised, so dot product = cosine)
        sim_related = float(np.dot(vecs[0], vecs[1]))
        sim_unrelated = float(np.dot(vecs[0], vecs[2]))

        assert sim_related > sim_unrelated


# ===================================================================
# store tests
# ===================================================================
from pharmagraphrag.vectorstore.store import (  # noqa: E402
    add_chunks,
    format_vector_context,
    get_collection,
    reset_collection,
    search,
    search_by_drug,
)


class TestChromaStore:
    """Tests for ChromaDB store operations.

    Uses a temporary directory for each test to avoid interference.
    """

    @pytest.fixture()
    def temp_client(self, tmp_path: Path) -> chromadb.ClientAPI:
        """Create a fresh ChromaDB client in a temp directory."""
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        return chromadb.PersistentClient(
            path=str(tmp_path / "chroma_test"),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    @pytest.fixture()
    def sample_chunks(self) -> list[TextChunk]:
        """Create sample TextChunk objects for testing."""
        return [
            TextChunk(
                text="Drug: ASPIRIN | Section: Adverse Reactions\nAspirin can cause stomach bleeding and ulcers.",
                drug_name="ASPIRIN",
                section="adverse_reactions",
                chunk_index=0,
                metadata={
                    "drug_name": "ASPIRIN",
                    "section": "adverse_reactions",
                    "chunk_index": "0",
                    "generic_names": "ASPIRIN",
                    "brand_names": "Bayer",
                    "route": "ORAL",
                },
            ),
            TextChunk(
                text="Drug: WARFARIN | Section: Drug Interactions\nWarfarin interacts with many drugs including aspirin and NSAIDs.",
                drug_name="WARFARIN",
                section="drug_interactions",
                chunk_index=0,
                metadata={
                    "drug_name": "WARFARIN",
                    "section": "drug_interactions",
                    "chunk_index": "0",
                    "generic_names": "WARFARIN SODIUM",
                    "brand_names": "Coumadin",
                    "route": "ORAL",
                },
            ),
            TextChunk(
                text="Drug: WARFARIN | Section: Adverse Reactions\nBleeding is the main risk of warfarin therapy.",
                drug_name="WARFARIN",
                section="adverse_reactions",
                chunk_index=0,
                metadata={
                    "drug_name": "WARFARIN",
                    "section": "adverse_reactions",
                    "chunk_index": "0",
                    "generic_names": "WARFARIN SODIUM",
                    "brand_names": "Coumadin",
                    "route": "ORAL",
                },
            ),
        ]

    def test_get_collection_creates(self, temp_client: chromadb.ClientAPI) -> None:
        coll = get_collection(client=temp_client, collection_name="test_coll")
        assert coll.count() == 0

    def test_reset_collection(self, temp_client: chromadb.ClientAPI) -> None:
        coll = get_collection(client=temp_client, collection_name="test_reset")
        # Add something via raw API
        coll.add(ids=["x"], documents=["hello"], embeddings=[[0.0] * 384])
        assert coll.count() == 1

        # Reset
        coll2 = reset_collection(client=temp_client, collection_name="test_reset")
        assert coll2.count() == 0

    def test_add_chunks(
        self,
        temp_client: chromadb.ClientAPI,
        sample_chunks: list[TextChunk],
    ) -> None:
        coll = reset_collection(client=temp_client, collection_name="test_add")
        added = add_chunks(sample_chunks, collection=coll, batch_size=2)
        assert added == 3
        assert coll.count() == 3

    def test_search_returns_results(
        self,
        temp_client: chromadb.ClientAPI,
        sample_chunks: list[TextChunk],
    ) -> None:
        coll = reset_collection(client=temp_client, collection_name="test_search")
        add_chunks(sample_chunks, collection=coll)

        results = search("warfarin bleeding risk", n_results=2, collection=coll)
        assert len(results) == 2
        assert all("text" in r for r in results)
        assert all("metadata" in r for r in results)
        assert all("distance" in r for r in results)

    def test_search_by_drug_filters(
        self,
        temp_client: chromadb.ClientAPI,
        sample_chunks: list[TextChunk],
    ) -> None:
        coll = reset_collection(client=temp_client, collection_name="test_drug_filter")
        add_chunks(sample_chunks, collection=coll)

        results = search_by_drug("bleeding", drug_name="WARFARIN", n_results=5, collection=coll)
        assert len(results) > 0
        assert all(r["metadata"]["drug_name"] == "WARFARIN" for r in results)

    def test_search_empty_collection(self, temp_client: chromadb.ClientAPI) -> None:
        coll = reset_collection(client=temp_client, collection_name="test_empty_search")
        results = search("anything", n_results=5, collection=coll)
        assert results == []


class TestFormatVectorContext:
    """Tests for format_vector_context."""

    def test_empty_results(self) -> None:
        assert format_vector_context([]) == "No relevant text context found."

    def test_formats_results(self) -> None:
        results = [
            {
                "text": "Warfarin interacts with aspirin.",
                "metadata": {"drug_name": "WARFARIN", "section": "drug_interactions"},
                "distance": 0.1234,
            }
        ]
        formatted = format_vector_context(results)
        assert "WARFARIN" in formatted
        assert "Drug Interactions" in formatted
        assert "0.1234" in formatted

    def test_respects_max_chars(self) -> None:
        results = [
            {
                "text": "x" * 2000,
                "metadata": {"drug_name": "TEST", "section": "warnings"},
                "distance": 0.5,
            },
            {
                "text": "y" * 2000,
                "metadata": {"drug_name": "TEST2", "section": "warnings"},
                "distance": 0.6,
            },
        ]
        formatted = format_vector_context(results, max_chars=500)
        assert len(formatted) <= 600  # some tolerance for headers
