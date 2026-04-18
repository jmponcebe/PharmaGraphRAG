"""Tests for the Langfuse observability module.

Verifies graceful degradation (no-op) when Langfuse is not configured,
and correct callback creation when it is.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset the module-level singleton between tests."""
    import pharmagraphrag.observability as obs

    obs._langfuse_initialized = False
    yield
    obs._langfuse_initialized = False


def _settings_disabled(**overrides):
    """Return a mock Settings with Langfuse disabled."""
    s = MagicMock()
    s.langfuse_enabled = overrides.get("langfuse_enabled", False)
    s.langfuse_public_key = overrides.get("langfuse_public_key", "")
    s.langfuse_secret_key = overrides.get("langfuse_secret_key", "")
    s.langfuse_base_url = overrides.get("langfuse_base_url", "https://cloud.langfuse.com")
    return s


def _settings_enabled():
    return _settings_disabled(
        langfuse_enabled=True,
        langfuse_public_key="pk-lf-test",
        langfuse_secret_key="sk-lf-test",
    )


# ---------------------------------------------------------------------------
# is_enabled / _ensure_initialized
# ---------------------------------------------------------------------------


class TestIsEnabled:
    def test_disabled_by_default(self):
        with patch("pharmagraphrag.observability.get_settings", return_value=_settings_disabled()):
            from pharmagraphrag.observability import is_enabled

            assert is_enabled() is False

    def test_disabled_when_no_keys(self):
        settings = _settings_disabled(langfuse_enabled=True)
        with patch("pharmagraphrag.observability.get_settings", return_value=settings):
            from pharmagraphrag.observability import is_enabled

            assert is_enabled() is False

    def test_enabled_with_keys(self):
        with (
            patch("pharmagraphrag.observability.get_settings", return_value=_settings_enabled()),
            patch("pharmagraphrag.observability.Langfuse", create=True),
            patch("langfuse.Langfuse", MagicMock()),
        ):
            from pharmagraphrag.observability import is_enabled

            assert is_enabled() is True

    def test_enabled_cached_after_init(self):
        """Once initialized, subsequent calls don't re-check settings."""
        import pharmagraphrag.observability as obs

        obs._langfuse_initialized = True
        # Even with disabled settings, returns True because already init'd
        with patch("pharmagraphrag.observability.get_settings", return_value=_settings_disabled()):
            assert obs.is_enabled() is True


# ---------------------------------------------------------------------------
# get_langfuse_callback
# ---------------------------------------------------------------------------


class TestGetLangfuseCallback:
    def test_returns_none_when_disabled(self):
        with patch("pharmagraphrag.observability.get_settings", return_value=_settings_disabled()):
            from pharmagraphrag.observability import get_langfuse_callback

            assert get_langfuse_callback() is None

    def test_returns_handler_when_enabled(self):
        import sys

        import pharmagraphrag.observability as obs

        obs._langfuse_initialized = True
        mock_handler = MagicMock()
        mock_lf_langchain = MagicMock()
        mock_lf_langchain.CallbackHandler.return_value = mock_handler
        with patch.dict(sys.modules, {"langfuse.langchain": mock_lf_langchain}):
            handler = obs.get_langfuse_callback(session_id="s1", user_id="u1")
            assert handler is mock_handler


# ---------------------------------------------------------------------------
# build_callback_config
# ---------------------------------------------------------------------------


class TestBuildCallbackConfig:
    def test_noop_when_disabled(self):
        with patch("pharmagraphrag.observability.get_settings", return_value=_settings_disabled()):
            from pharmagraphrag.observability import build_callback_config

            existing = {"configurable": {"thread_id": "t1"}}
            result = build_callback_config(existing_config=existing)
            assert result == existing

    def test_adds_callbacks_when_enabled(self):
        import sys

        import pharmagraphrag.observability as obs

        obs._langfuse_initialized = True
        mock_handler = MagicMock()
        mock_lf_langchain = MagicMock()
        mock_lf_langchain.CallbackHandler.return_value = mock_handler
        with patch.dict(sys.modules, {"langfuse.langchain": mock_lf_langchain}):
            result = obs.build_callback_config(
                session_id="s1",
                tags=["test"],
                existing_config={"configurable": {"thread_id": "t1"}},
            )
            assert "callbacks" in result
            assert mock_handler in result["callbacks"]
            assert result["configurable"]["thread_id"] == "t1"
            assert result["metadata"]["langfuse_session_id"] == "s1"


# ---------------------------------------------------------------------------
# observe_fn decorator
# ---------------------------------------------------------------------------


class TestObserveFn:
    def test_passthrough_when_disabled(self):
        with patch("pharmagraphrag.observability.get_settings", return_value=_settings_disabled()):
            from pharmagraphrag.observability import observe_fn

            @observe_fn()
            def add(a, b):
                return a + b

            assert add(2, 3) == 5

    def test_preserves_function_name(self):
        from pharmagraphrag.observability import observe_fn

        @observe_fn(name="custom_name")
        def my_func():
            pass

        assert my_func.__name__ == "my_func"


# ---------------------------------------------------------------------------
# trace_generation
# ---------------------------------------------------------------------------


class TestTraceGeneration:
    def test_noop_when_disabled(self):
        """Should not raise even when Langfuse is off."""
        with patch("pharmagraphrag.observability.get_settings", return_value=_settings_disabled()):
            from pharmagraphrag.observability import trace_generation

            trace_generation(
                name="test",
                model="gemini-2.5-flash",
                input_text="hello",
                output_text="world",
                usage={"prompt_tokens": 10, "completion_tokens": 20},
            )


# ---------------------------------------------------------------------------
# flush
# ---------------------------------------------------------------------------


class TestFlush:
    def test_noop_when_not_initialized(self):
        from pharmagraphrag.observability import flush

        flush()  # should not raise

    def test_calls_client_flush(self):
        import pharmagraphrag.observability as obs

        obs._langfuse_initialized = True
        mock_client = MagicMock()
        with patch("langfuse.get_client", return_value=mock_client):
            obs.flush()
            mock_client.flush.assert_called_once()
