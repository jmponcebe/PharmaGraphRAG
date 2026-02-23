"""Tests for the LLM client module.

All API calls are mocked — no real LLM calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pharmagraphrag.llm.client import (
    LLMResponse,
    _generate_gemini,
    _generate_ollama,
    generate_answer,
)


# ===========================================================================
# LLMResponse data class
# ===========================================================================


class TestLLMResponse:
    """Tests for the LLMResponse data class."""

    def test_ok_with_text(self):
        r = LLMResponse(text="Hello", model="m", provider="gemini")
        assert r.ok

    def test_not_ok_empty_text(self):
        r = LLMResponse(text="", model="m", provider="gemini")
        assert not r.ok

    def test_not_ok_with_error(self):
        r = LLMResponse(text="Hello", model="m", provider="gemini", error="fail")
        assert not r.ok

    def test_default_values(self):
        r = LLMResponse()
        assert r.text == ""
        assert r.model == ""
        assert r.provider == ""
        assert r.usage == {}
        assert r.error is None
        assert not r.ok


# ===========================================================================
# Gemini
# ===========================================================================


class TestGenerateGemini:
    """Tests for Gemini generation (mocked)."""

    @patch("pharmagraphrag.llm.client.get_settings")
    def test_no_api_key(self, mock_settings):
        """Returns error when GEMINI_API_KEY is not set."""
        mock_settings.return_value.gemini_api_key = ""
        result = _generate_gemini("sys", "user")
        assert not result.ok
        assert "GEMINI_API_KEY" in result.error
        assert result.provider == "gemini"

    @patch("pharmagraphrag.llm.client.get_settings")
    @patch("google.genai.Client")
    def test_successful_generation(self, mock_client_cls, mock_settings):
        """Successful Gemini response."""
        mock_settings.return_value.gemini_api_key = "test-key"

        # Set up mock response
        mock_response = MagicMock()
        mock_response.text = "Ibuprofen can cause nausea."
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.total_token_count = 150

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = _generate_gemini("sys prompt", "user prompt")
        assert result.ok
        assert "Ibuprofen" in result.text
        assert result.provider == "gemini"
        assert result.usage["total_tokens"] == 150

    @patch("pharmagraphrag.llm.client.get_settings")
    @patch("google.genai.Client")
    def test_api_error(self, mock_client_cls, mock_settings):
        """Handles API errors gracefully."""
        mock_settings.return_value.gemini_api_key = "test-key"

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("rate limit")
        mock_client_cls.return_value = mock_client

        result = _generate_gemini("sys", "user")
        assert not result.ok
        assert "rate limit" in result.error


# ===========================================================================
# Ollama
# ===========================================================================


class TestGenerateOllama:
    """Tests for Ollama generation (mocked)."""

    @patch("pharmagraphrag.llm.client.get_settings")
    @patch("ollama.Client")
    def test_successful_generation(self, mock_client_cls, mock_settings):
        """Successful Ollama response."""
        mock_settings.return_value.ollama_base_url = "http://localhost:11434"

        mock_response = MagicMock()
        mock_response.message.content = "Aspirin interacts with warfarin."
        mock_response.prompt_eval_count = 80
        mock_response.eval_count = 40

        mock_client = MagicMock()
        mock_client.chat.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = _generate_ollama("sys", "user")
        assert result.ok
        assert "Aspirin" in result.text
        assert result.provider == "ollama"

    @patch("pharmagraphrag.llm.client.get_settings")
    @patch("ollama.Client")
    def test_connection_error(self, mock_client_cls, mock_settings):
        """Handles Ollama connection errors."""
        mock_settings.return_value.ollama_base_url = "http://localhost:11434"

        mock_client = MagicMock()
        mock_client.chat.side_effect = ConnectionError("Ollama not running")
        mock_client_cls.return_value = mock_client

        result = _generate_ollama("sys", "user")
        assert not result.ok
        assert "not running" in result.error


# ===========================================================================
# Unified generate_answer
# ===========================================================================


class TestGenerateAnswer:
    """Tests for the unified generate_answer function."""

    @patch("pharmagraphrag.llm.client._generate_gemini")
    @patch("pharmagraphrag.llm.client.get_settings")
    def test_uses_configured_provider(self, mock_settings, mock_gemini):
        """Uses the provider from settings."""
        mock_settings.return_value.llm_provider = "gemini"
        mock_gemini.return_value = LLMResponse(
            text="answer", model="gemini-2.0-flash", provider="gemini",
        )

        result = generate_answer("sys", "user")
        assert result.ok
        assert result.provider == "gemini"
        mock_gemini.assert_called_once()

    @patch("pharmagraphrag.llm.client._generate_ollama")
    @patch("pharmagraphrag.llm.client.get_settings")
    def test_uses_ollama_provider(self, mock_settings, mock_ollama):
        """Uses Ollama when configured."""
        mock_settings.return_value.llm_provider = "ollama"
        mock_ollama.return_value = LLMResponse(
            text="answer", model="llama3:8b", provider="ollama",
        )

        result = generate_answer("sys", "user")
        assert result.ok
        assert result.provider == "ollama"

    @patch("pharmagraphrag.llm.client._generate_ollama")
    @patch("pharmagraphrag.llm.client._generate_gemini")
    @patch("pharmagraphrag.llm.client.get_settings")
    def test_fallback_on_failure(self, mock_settings, mock_gemini, mock_ollama):
        """Falls back to the other provider when primary fails."""
        mock_settings.return_value.llm_provider = "gemini"
        mock_gemini.return_value = LLMResponse(
            model="gemini-2.0-flash", provider="gemini", error="API key invalid",
        )
        mock_ollama.return_value = LLMResponse(
            text="fallback answer", model="llama3:8b", provider="ollama",
        )

        result = generate_answer("sys", "user")
        assert result.ok
        assert result.provider == "ollama"

    @patch("pharmagraphrag.llm.client._generate_gemini")
    def test_explicit_provider_override(self, mock_gemini):
        """Explicit provider overrides config."""
        mock_gemini.return_value = LLMResponse(
            text="answer", model="gemini-2.0-flash", provider="gemini",
        )

        result = generate_answer("sys", "user", provider="gemini")
        assert result.ok
        mock_gemini.assert_called_once()

    @patch("pharmagraphrag.llm.client.get_settings")
    def test_unknown_provider(self, mock_settings):
        """Returns error for unknown provider."""
        mock_settings.return_value.llm_provider = "unknown"
        result = generate_answer("sys", "user")
        assert not result.ok
        assert "Unknown" in result.error
