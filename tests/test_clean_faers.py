"""Tests for FAERS cleaning module."""

import pandas as pd

from pharmagraphrag.data.clean_faers import (
    normalize_drug_name,
    clean_drug,
    clean_reac,
    clean_outc,
    OUTCOME_DESCRIPTIONS,
)


class TestNormalizeDrugName:
    """Tests for drug name normalization."""

    def test_basic_normalization(self) -> None:
        assert normalize_drug_name("Ibuprofen") == "IBUPROFEN"

    def test_strips_whitespace(self) -> None:
        assert normalize_drug_name("  aspirin  ") == "ASPIRIN"

    def test_removes_extra_spaces(self) -> None:
        assert normalize_drug_name("sodium   chloride") == "SODIUM CHLORIDE"

    def test_removes_trailing_punctuation(self) -> None:
        assert normalize_drug_name("metformin.") == "METFORMIN"
        assert normalize_drug_name("warfarin,") == "WARFARIN"

    def test_removes_parenthetical_dosage(self) -> None:
        assert normalize_drug_name("IBUPROFEN (200MG)") == "IBUPROFEN"

    def test_handles_nan(self) -> None:
        assert normalize_drug_name(float("nan")) == ""

    def test_handles_none_like(self) -> None:
        assert normalize_drug_name(None) == ""  # type: ignore[arg-type]

    def test_handles_empty_string(self) -> None:
        assert normalize_drug_name("") == ""


class TestCleanDrug:
    """Tests for drug table cleaning."""

    def test_deduplicates_by_primaryid_and_seq(self) -> None:
        df = pd.DataFrame({
            "primaryid": ["1", "1", "2"],
            "drug_seq": ["1", "1", "1"],
            "drugname": ["aspirin", "ASPIRIN", "ibuprofen"],
            "role_cod": ["PS", "PS", "SS"],
        })
        result = clean_drug(df)
        assert len(result) == 2

    def test_normalizes_drug_names(self) -> None:
        df = pd.DataFrame({
            "primaryid": ["1"],
            "drug_seq": ["1"],
            "drugname": ["  ibuprofen (200mg)  "],
            "role_cod": ["PS"],
        })
        result = clean_drug(df)
        assert result.iloc[0]["drugname"] == "IBUPROFEN"

    def test_removes_empty_drug_names(self) -> None:
        df = pd.DataFrame({
            "primaryid": ["1", "2"],
            "drug_seq": ["1", "1"],
            "drugname": ["aspirin", ""],
            "role_cod": ["PS", "PS"],
        })
        result = clean_drug(df)
        assert len(result) == 1


class TestCleanReac:
    """Tests for reactions table cleaning."""

    def test_normalizes_reaction_terms(self) -> None:
        df = pd.DataFrame({
            "primaryid": ["1"],
            "pt": ["  headache  "],
        })
        result = clean_reac(df)
        assert result.iloc[0]["pt"] == "HEADACHE"

    def test_removes_empty_reactions(self) -> None:
        df = pd.DataFrame({
            "primaryid": ["1", "2"],
            "pt": ["nausea", None],
        })
        result = clean_reac(df)
        assert len(result) == 1


class TestCleanOutc:
    """Tests for outcomes table cleaning."""

    def test_maps_outcome_descriptions(self) -> None:
        df = pd.DataFrame({
            "primaryid": ["1", "2"],
            "outc_cod": ["DE", "HO"],
        })
        result = clean_outc(df)
        assert result.iloc[0]["outc_desc"] == "Death"
        assert result.iloc[1]["outc_desc"] == "Hospitalization"

    def test_all_outcome_codes_have_descriptions(self) -> None:
        """Ensure all known codes are mapped."""
        for code in ["DE", "LT", "HO", "DS", "CA", "RI", "OT"]:
            assert code in OUTCOME_DESCRIPTIONS
