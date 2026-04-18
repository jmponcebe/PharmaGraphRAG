"""Langfuse observability integration for PharmaGraphRAG.

Provides centralized tracing for all LLM calls, agent tool usage,
and query pipeline execution. Gracefully degrades when Langfuse is
not configured (no API keys → tracing silently disabled).

Usage:
    from pharmagraphrag.observability import get_langfuse_callback, observe_fn

    # LangChain/LangGraph agent tracing
    handler = get_langfuse_callback(session_id="...", user_id="...")
    agent.invoke(..., config={"callbacks": [handler]})

    # Custom function tracing
    @observe_fn()
    def my_pipeline_step(...):
        ...
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

from loguru import logger

from pharmagraphrag.config import get_settings

_langfuse_initialized = False
_langfuse_disabled = False


def _ensure_initialized() -> bool:
    """Initialize the Langfuse singleton client if not done yet.

    Returns True if Langfuse is available and configured.
    """
    global _langfuse_initialized, _langfuse_disabled
    if _langfuse_initialized:
        return True
    if _langfuse_disabled:
        return False

    settings = get_settings()
    if not settings.langfuse_enabled:
        _langfuse_disabled = True
        return False

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning("Langfuse enabled but API keys not set — tracing disabled")
        _langfuse_disabled = True
        return False

    try:
        from langfuse import Langfuse

        Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_base_url,
        )
        _langfuse_initialized = True
        logger.info("Langfuse tracing initialized (host={})", settings.langfuse_base_url)
        return True
    except Exception as exc:
        logger.warning("Failed to initialize Langfuse: {}", exc)
        return False


def is_enabled() -> bool:
    """Check if Langfuse tracing is active."""
    return _ensure_initialized()


def get_langfuse_callback(
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, str] | None = None,
) -> Any | None:
    """Create a Langfuse CallbackHandler for LangChain/LangGraph tracing.

    Returns None if Langfuse is not configured, so callers can safely
    filter it out of callback lists.
    """
    if not _ensure_initialized():
        return None

    try:
        from langfuse.langchain import CallbackHandler

        handler = CallbackHandler()
        # Trace-level attributes go via metadata in config
        handler._langfuse_metadata = {
            "session_id": session_id,
            "user_id": user_id,
            "tags": tags or [],
            "metadata": metadata or {},
        }
        return handler
    except Exception as exc:
        logger.debug("Could not create Langfuse callback: {}", exc)
        return None


def build_callback_config(
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    existing_config: dict | None = None,
) -> dict:
    """Build a LangChain config dict with Langfuse callbacks and metadata.

    Merges with existing config if provided. If Langfuse is disabled,
    returns the existing config unchanged.
    """
    config = dict(existing_config or {})

    handler = get_langfuse_callback(
        session_id=session_id,
        user_id=user_id,
        tags=tags,
    )
    if handler is None:
        return config

    callbacks = config.get("callbacks", [])
    callbacks = [*list(callbacks), handler]
    config["callbacks"] = callbacks

    # Add langfuse metadata for trace attributes (copy to avoid mutation)
    meta = dict(config.get("metadata", {}))
    if session_id:
        meta["langfuse_session_id"] = session_id
    if user_id:
        meta["langfuse_user_id"] = user_id
    if tags:
        meta["langfuse_tags"] = tags
    config["metadata"] = meta

    return config


def observe_fn(
    *,
    name: str | None = None,
) -> Callable:
    """Decorator that wraps a function with Langfuse @observe tracing.

    Falls back to a no-op decorator when Langfuse is not configured,
    so production code works with zero overhead.
    """

    def decorator(func: Callable) -> Callable:
        traced_func: Callable | None = None

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal traced_func

            if traced_func is not None:
                return traced_func(*args, **kwargs)

            if not _ensure_initialized():
                return func(*args, **kwargs)

            try:
                from langfuse import observe

                traced_func = observe(name=name or func.__name__)(func)
                return traced_func(*args, **kwargs)
            except Exception:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def trace_generation(
    *,
    name: str,
    model: str,
    input_text: str,
    output_text: str,
    usage: dict[str, int] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Log a standalone LLM generation to Langfuse.

    Used for the classic pipeline where we call the LLM directly
    (not through LangChain).
    """
    if not _ensure_initialized():
        return

    try:
        from langfuse import get_client

        langfuse = get_client()
        with langfuse.start_as_current_observation(
            as_type="generation",
            name=name,
            model=model,
            input=input_text,
        ) as gen:
            gen.update(
                output=output_text,
                usage_details={
                    "input_tokens": (usage or {}).get("prompt_tokens", 0),
                    "output_tokens": (usage or {}).get("completion_tokens", 0),
                }
                if usage
                else None,
                metadata=metadata,
            )
    except Exception as exc:
        logger.debug("Langfuse trace_generation failed: {}", exc)


def flush() -> None:
    """Flush pending Langfuse events (call before shutdown)."""
    if not _langfuse_initialized:
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        pass
