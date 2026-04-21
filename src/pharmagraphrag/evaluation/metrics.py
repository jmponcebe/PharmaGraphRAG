"""RAGAS metric wrappers for PharmaGraphRAG.

Configures RAGAS metrics with Gemini as the evaluator LLM.
Provides both reference-free metrics (faithfulness, answer relevancy)
and reference-based metrics (context precision, context recall,
answer correctness) for comprehensive RAG evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class MetricResult:
    """Result of a single metric evaluation."""

    name: str
    score: float
    details: dict[str, Any] | None = None


@dataclass
class EvalResult:
    """Aggregated evaluation result for a single sample."""

    question: str
    answer: str
    metrics: list[MetricResult]
    contexts: list[str] | None = None
    reference: str | None = None

    @property
    def scores(self) -> dict[str, float]:
        return {m.name: m.score for m in self.metrics}


def _get_evaluator_llm(model: str = "gemini-2.5-flash"):
    """Create a RAGAS-compatible LLM wrapper using Gemini via OpenAI compatibility.

    Uses LangChain's ChatOpenAI pointed at the Gemini OpenAI-compatible
    endpoint, wrapped in LangchainLLMWrapper. Bumps max_tokens to avoid
    truncation on multi-statement classification prompts (ContextRecall).
    """
    import os

    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY env var is required for RAGAS evaluation")

    chat = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        max_tokens=8192,
        temperature=0.0,
    )
    return LangchainLLMWrapper(chat)


def _get_evaluator_embeddings(model: str = "models/gemini-embedding-001"):
    """Create RAGAS-compatible embeddings using Gemini native API.

    The Gemini OpenAI-compatibility endpoint does not support embeddings
    (returns 501 UNIMPLEMENTED), so we use the native Google Generative AI
    embeddings via langchain-google-genai.
    """
    import os

    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY env var is required for RAGAS evaluation")

    embeddings = GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)
    return LangchainEmbeddingsWrapper(embeddings)


def get_reference_free_metrics(
    llm=None,
    embeddings=None,
) -> list:
    """Get metrics that don't require ground truth.

    Returns faithfulness and answer relevancy metrics.
    """
    from ragas.metrics import AnswerRelevancy, Faithfulness

    llm = llm or _get_evaluator_llm()
    embeddings = embeddings or _get_evaluator_embeddings()

    return [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
    ]


def get_reference_metrics(
    llm=None,
    embeddings=None,
) -> list:
    """Get metrics that require ground truth reference.

    Returns context precision, context recall, and answer correctness.
    """
    from ragas.metrics import (
        AnswerCorrectness,
        AnswerSimilarity,
        ContextPrecision,
        ContextRecall,
    )

    llm = llm or _get_evaluator_llm()
    embeddings = embeddings or _get_evaluator_embeddings()

    # AnswerCorrectness needs an explicit AnswerSimilarity (embeddings-based)
    answer_similarity = AnswerSimilarity(embeddings=embeddings)

    return [
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
        AnswerCorrectness(llm=llm, embeddings=embeddings, answer_similarity=answer_similarity),
    ]


def get_all_metrics(llm=None, embeddings=None) -> list:
    """Get all available metrics (reference-free + reference-based)."""
    llm = llm or _get_evaluator_llm()
    embeddings = embeddings or _get_evaluator_embeddings()

    return get_reference_free_metrics(llm, embeddings) + get_reference_metrics(llm, embeddings)


def score_sample(
    question: str,
    answer: str,
    contexts: list[str],
    reference: str | None = None,
    llm=None,
    embeddings=None,
) -> EvalResult:
    """Evaluate a single RAG response using RAGAS metrics.

    Uses reference-free metrics always; adds reference-based metrics
    if a ground truth reference is provided.
    """
    llm = llm or _get_evaluator_llm()
    embeddings = embeddings or _get_evaluator_embeddings()

    import asyncio

    from ragas.dataset_schema import SingleTurnSample

    metrics = get_reference_free_metrics(llm, embeddings)
    if reference:
        metrics.extend(get_reference_metrics(llm, embeddings))

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=reference,
    )

    async def _score_all() -> list[MetricResult]:
        out: list[MetricResult] = []
        for metric in metrics:
            try:
                score = await metric.single_turn_ascore(sample)
                out.append(MetricResult(name=type(metric).__name__, score=float(score)))
            except Exception as exc:
                logger.warning("Metric {} failed: {}", type(metric).__name__, exc)
                out.append(MetricResult(name=type(metric).__name__, score=-1.0))
        return out

    results = asyncio.run(_score_all())

    return EvalResult(
        question=question,
        answer=answer,
        metrics=results,
        contexts=contexts,
        reference=reference,
    )
