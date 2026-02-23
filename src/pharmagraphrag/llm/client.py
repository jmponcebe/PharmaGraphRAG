"""LLM client abstraction — Gemini API + Ollama fallback.

Provides a unified interface for generating text answers with either
Google Gemini (via ``google-genai``) or a local Ollama instance.

The active provider is determined by the ``LLM_PROVIDER`` env-var
(``gemini`` | ``ollama``).

Usage:
    from pharmagraphrag.llm.client import generate_answer
    answer = generate_answer(system_prompt="...", user_prompt="...")
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from pharmagraphrag.config import get_settings

# Default models per provider
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_OLLAMA_MODEL = "llama3:8b"


@dataclass
class LLMResponse:
    """Wrapper around a raw LLM generation result."""

    text: str = ""
    """The generated text."""

    model: str = ""
    """Model identifier that produced the answer."""

    provider: str = ""
    """Provider name ('gemini' or 'ollama')."""

    usage: dict[str, int] = field(default_factory=dict)
    """Token usage statistics (if available)."""

    error: str | None = None
    """Error message if generation failed."""

    @property
    def ok(self) -> bool:
        """True if generation was successful."""
        return self.error is None and bool(self.text)


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


def _generate_gemini(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    api_key: str | None = None,
    **kwargs,
) -> LLMResponse:
    """Call Google Gemini via the google-genai SDK.

    Args:
        system_prompt: System instruction for the model.
        user_prompt: User message.
        model: Model name (defaults to ``gemini-2.0-flash``).
        api_key: Gemini API key (falls back to settings).

    Returns:
        LLMResponse with generated text.
    """
    settings = get_settings()
    api_key = api_key or settings.gemini_api_key
    model = model or DEFAULT_GEMINI_MODEL

    if not api_key:
        return LLMResponse(
            model=model,
            provider="gemini",
            error="GEMINI_API_KEY is not set",
        )

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
                max_output_tokens=2048,
            ),
        )

        text = response.text or ""
        usage: dict[str, int] = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", 0) or 0,
                "completion_tokens": getattr(um, "candidates_token_count", 0) or 0,
                "total_tokens": getattr(um, "total_token_count", 0) or 0,
            }

        logger.info(
            "Gemini response: {} chars, model={}, tokens={}",
            len(text),
            model,
            usage.get("total_tokens", "?"),
        )

        return LLMResponse(
            text=text,
            model=model,
            provider="gemini",
            usage=usage,
        )

    except Exception as exc:
        logger.error("Gemini generation failed: {}", exc)
        return LLMResponse(model=model, provider="gemini", error=str(exc))


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


def _generate_ollama(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    base_url: str | None = None,
    **kwargs,
) -> LLMResponse:
    """Call a local Ollama instance.

    Args:
        system_prompt: System prompt.
        user_prompt: User message.
        model: Ollama model name (defaults to ``llama3:8b``).
        base_url: Ollama server URL (falls back to settings).

    Returns:
        LLMResponse with generated text.
    """
    settings = get_settings()
    model = model or DEFAULT_OLLAMA_MODEL
    base_url = base_url or settings.ollama_base_url

    try:
        import ollama as ollama_sdk

        client = ollama_sdk.Client(host=base_url)

        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0.3},
        )

        text = response.message.content or ""
        usage: dict[str, int] = {}
        if hasattr(response, "prompt_eval_count"):
            usage = {
                "prompt_tokens": response.prompt_eval_count or 0,
                "completion_tokens": response.eval_count or 0,
                "total_tokens": (response.prompt_eval_count or 0) + (response.eval_count or 0),
            }

        logger.info(
            "Ollama response: {} chars, model={}",
            len(text),
            model,
        )

        return LLMResponse(
            text=text,
            model=model,
            provider="ollama",
            usage=usage,
        )

    except Exception as exc:
        logger.error("Ollama generation failed: {}", exc)
        return LLMResponse(model=model, provider="ollama", error=str(exc))


# ---------------------------------------------------------------------------
# Unified public API
# ---------------------------------------------------------------------------


def generate_answer(
    system_prompt: str,
    user_prompt: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    **kwargs,
) -> LLMResponse:
    """Generate an LLM answer using the configured provider.

    Tries the primary provider first. If it fails and the other provider
    is available, falls back automatically.

    Args:
        system_prompt: System prompt / instruction.
        user_prompt: User message with context + question.
        provider: Force a provider ('gemini' | 'ollama'). Uses config if None.
        model: Override the default model for the provider.

    Returns:
        LLMResponse with the generated answer.
    """
    settings = get_settings()
    provider = provider or settings.llm_provider

    logger.info("Generating answer with provider='{}', model='{}'", provider, model or "default")

    # Primary attempt
    if provider == "gemini":
        response = _generate_gemini(system_prompt, user_prompt, model=model, **kwargs)
    elif provider == "ollama":
        response = _generate_ollama(system_prompt, user_prompt, model=model, **kwargs)
    else:
        return LLMResponse(error=f"Unknown LLM provider: {provider}")

    # Automatic fallback
    if not response.ok:
        fallback = "ollama" if provider == "gemini" else "gemini"
        logger.warning(
            "Primary provider '{}' failed ({}), trying fallback '{}'",
            provider,
            response.error,
            fallback,
        )
        if fallback == "gemini":
            response = _generate_gemini(system_prompt, user_prompt, **kwargs)
        else:
            response = _generate_ollama(system_prompt, user_prompt, **kwargs)

    return response
