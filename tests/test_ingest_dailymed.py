"""Tests for DailyMed/openFDA ingestion module."""

from pharmagraphrag.data.ingest_dailymed import (
    TOP_DRUGS,
    LABEL_SECTIONS,
    _extract_label_data,
)


class TestTopDrugs:
    """Tests for the curated drug list."""

    def test_top_drugs_not_empty(self) -> None:
        assert len(TOP_DRUGS) > 50

    def test_top_drugs_are_lowercase(self) -> None:
        for drug in TOP_DRUGS:
            assert drug == drug.lower(), f"Drug '{drug}' is not lowercase"

    def test_common_drugs_present(self) -> None:
        common = ["ibuprofen", "metformin", "aspirin", "warfarin", "omeprazole"]
        for drug in common:
            assert drug in TOP_DRUGS, f"Expected '{drug}' in TOP_DRUGS"

    def test_no_duplicates(self) -> None:
        assert len(TOP_DRUGS) == len(set(TOP_DRUGS))


class TestExtractLabelData:
    """Tests for label data extraction from openFDA results."""

    def test_extracts_drug_name_uppercase(self) -> None:
        result = {"openfda": {"generic_name": ["Ibuprofen"]}}
        data = _extract_label_data("ibuprofen", result)
        assert data["drug_name"] == "IBUPROFEN"

    def test_extracts_openfda_metadata(self) -> None:
        result = {
            "openfda": {
                "generic_name": ["Ibuprofen"],
                "brand_name": ["Advil", "Motrin"],
                "manufacturer_name": ["Pfizer"],
                "route": ["ORAL"],
            }
        }
        data = _extract_label_data("ibuprofen", result)
        assert data["brand_names"] == ["Advil", "Motrin"]
        assert data["manufacturer"] == ["Pfizer"]
        assert data["route"] == ["ORAL"]

    def test_extracts_text_sections(self) -> None:
        result = {
            "openfda": {},
            "drug_interactions": ["Do not take with alcohol.", "May interact with warfarin."],
            "adverse_reactions": ["Common: headache, nausea."],
        }
        data = _extract_label_data("ibuprofen", result)
        assert "drug_interactions" in data["sections"]
        assert "warfarin" in data["sections"]["drug_interactions"]
        assert "adverse_reactions" in data["sections"]

    def test_joins_multiple_text_items(self) -> None:
        result = {
            "openfda": {},
            "warnings": ["Warning 1.", "Warning 2."],
        }
        data = _extract_label_data("test_drug", result)
        text = data["sections"]["warnings"]
        assert "Warning 1." in text
        assert "Warning 2." in text

    def test_missing_sections_not_included(self) -> None:
        result = {"openfda": {}}
        data = _extract_label_data("test_drug", result)
        assert data["sections"] == {}

    def test_empty_openfda_defaults(self) -> None:
        result = {"openfda": {}}
        data = _extract_label_data("test_drug", result)
        assert data["generic_names"] == []
        assert data["brand_names"] == []


class TestLabelSections:
    """Tests for the label sections configuration."""

    def test_key_sections_present(self) -> None:
        assert "drug_interactions" in LABEL_SECTIONS
        assert "adverse_reactions" in LABEL_SECTIONS
        assert "warnings" in LABEL_SECTIONS
        assert "contraindications" in LABEL_SECTIONS

    def test_sections_are_strings(self) -> None:
        for section in LABEL_SECTIONS:
            assert isinstance(section, str)
