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

    Uses the Gemini OpenAI-compatible endpoint to avoid instructor/google-genai
    SDK conflicts with safety settings.
    """
    import os

    from openai import OpenAI
    from ragas.llms import llm_factory

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY env var is required for RAGAS evaluation")

    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    return llm_factory(model, provider="openai", client=client)


def _get_evaluator_embeddings(model: str = "text-embedding-004"):
    """Create RAGAS-compatible embeddings using Gemini via OpenAI compatibility."""
    import os

    from openai import OpenAI
    from ragas.embeddings import OpenAIEmbeddings

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY env var is required for RAGAS evaluation")

    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    return OpenAIEmbeddings(client=client, model=model)


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
    from ragas.metrics import AnswerCorrectness, ContextPrecision, ContextRecall

    llm = llm or _get_evaluator_llm()
    embeddings = embeddings or _get_evaluator_embeddings()

    return [
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
        AnswerCorrectness(llm=llm, embeddings=embeddings),
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

    metrics = get_reference_free_metrics(llm, embeddings)
    if reference:
        metrics.extend(get_reference_metrics(llm, embeddings))

    results = []
    for metric in metrics:
        try:
            score = metric.single_score(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=reference,
            )
            results.append(MetricResult(name=type(metric).__name__, score=score))
        except Exception as exc:
            logger.warning("Metric {} failed: {}", type(metric).__name__, exc)
            results.append(MetricResult(name=type(metric).__name__, score=-1.0))

    return EvalResult(
        question=question,
        answer=answer,
        metrics=results,
        contexts=contexts,
        reference=reference,
    )
