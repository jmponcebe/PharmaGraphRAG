"""Tests for FAERS download module."""

from pharmagraphrag.data.download_faers import build_download_url


def test_build_download_url() -> None:
    """Test that download URLs are built correctly."""
    url = build_download_url("2024Q3")
    assert url == "https://fis.fda.gov/content/Exports/faers_ascii_2024Q3.zip"


def test_build_download_url_different_quarter() -> None:
    """Test URL construction for different quarters."""
    url = build_download_url("2023Q1")
    assert "faers_ascii_2023Q1.zip" in url
    assert url.startswith("https://fis.fda.gov/content/Exports/")
