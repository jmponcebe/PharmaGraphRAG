"""Evaluation runner for PharmaGraphRAG.

Executes the curated testset against the classic pipeline, agent mode,
and multi-agent mode, then computes RAGAS metrics for each response.
Results are exported as CSV for analysis.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from pharmagraphrag.evaluation.dataset import EvalDataset, EvalSample
from pharmagraphrag.evaluation.metrics import EvalResult, MetricResult, score_sample


@dataclass
class RunConfig:
    """Configuration for an evaluation run."""

    api_url: str = "http://localhost:8000"
    mode: str = "classic"  # "classic", "agent", "multi"
    model: str | None = None
    timeout: float = 120.0
    include_reference: bool = True


@dataclass
class PipelineResponse:
    """Raw response from a PharmaGraphRAG pipeline call."""

    answer: str = ""
    contexts: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def _call_classic(question: str, config: RunConfig) -> PipelineResponse:
    """Call the classic pipeline via POST /query."""
    url = f"{config.api_url}/query"
    body: dict[str, Any] = {"question": question}
    if config.model:
        body["model"] = config.model

    start = time.perf_counter()
    try:
        resp = httpx.post(url, json=body, timeout=config.timeout)
        latency = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        data = resp.json()

        contexts = []
        for src in data.get("sources", []):
            snippet = src.get("snippet", "")
            if snippet:
                contexts.append(snippet)

        return PipelineResponse(
            answer=data.get("answer", ""),
            contexts=contexts,
            latency_ms=latency,
            raw=data,
        )
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000
        return PipelineResponse(error=str(exc), latency_ms=latency)


def _call_agent(
    question: str, config: RunConfig, endpoint: str = "/agent/query"
) -> PipelineResponse:
    """Call the agent or multi-agent pipeline."""
    url = f"{config.api_url}{endpoint}"
    body: dict[str, Any] = {"question": question}
    if config.model:
        body["model"] = config.model

    start = time.perf_counter()
    try:
        resp = httpx.post(url, json=body, timeout=config.timeout)
        latency = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        data = resp.json()

        # Extract contexts from tool results and graph/vector data
        contexts = []
        for tr in data.get("tool_results", []):
            content = tr.get("content", "")
            if content:
                contexts.append(content[:2000])

        # Also use vector data snippets
        for vd in data.get("vector_data", []):
            snippet = vd.get("text", vd.get("snippet", ""))
            if snippet:
                contexts.append(snippet)

        tools = [tc.get("tool", "") for tc in data.get("tool_calls", [])]

        return PipelineResponse(
            answer=data.get("answer", ""),
            contexts=contexts,
            tool_calls=tools,
            latency_ms=latency,
            raw=data,
        )
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000
        return PipelineResponse(error=str(exc), latency_ms=latency)


def run_pipeline(question: str, config: RunConfig) -> PipelineResponse:
    """Route a question to the configured pipeline mode."""
    if config.mode == "classic":
        return _call_classic(question, config)
    elif config.mode == "agent":
        return _call_agent(question, config, endpoint="/agent/query")
    elif config.mode == "multi":
        return _call_agent(question, config, endpoint="/agent/multi")
    else:
        return PipelineResponse(error=f"Unknown mode: {config.mode}")


def evaluate_sample(
    sample: EvalSample,
    config: RunConfig,
    llm=None,
    embeddings=None,
) -> EvalResult:
    """Run a single sample through the pipeline and evaluate with RAGAS."""
    logger.info("[{}] Evaluating: {}", sample.id, sample.question[:60])

    # Call pipeline
    response = run_pipeline(sample.question, config)

    if response.error:
        logger.warning("[{}] Pipeline error: {}", sample.id, response.error)
        return EvalResult(
            question=sample.question,
            answer=f"ERROR: {response.error}",
            metrics=[MetricResult(name="error", score=-1.0)],
            contexts=[],
            reference=sample.reference if config.include_reference else None,
        )

    # Populate sample with response data
    sample.answer = response.answer
    sample.contexts = response.contexts
    sample.tool_calls = response.tool_calls

    # Compute RAGAS metrics
    reference = sample.reference if config.include_reference else None
    result = score_sample(
        question=sample.question,
        answer=response.answer,
        contexts=response.contexts,
        reference=reference,
        llm=llm,
        embeddings=embeddings,
    )

    # Add latency as a pseudo-metric
    result.metrics.append(MetricResult(name="latency_ms", score=response.latency_ms))

    return result


def evaluate_dataset(
    dataset: EvalDataset,
    config: RunConfig,
    llm=None,
    embeddings=None,
) -> list[EvalResult]:
    """Evaluate all samples in the dataset.

    Args:
        dataset: Loaded evaluation dataset.
        config: Pipeline configuration (mode, API URL, model).
        llm: RAGAS evaluator LLM (created if not provided).
        embeddings: RAGAS evaluator embeddings (created if not provided).

    Returns:
        List of EvalResult, one per sample.
    """
    from pharmagraphrag.evaluation.metrics import _get_evaluator_embeddings, _get_evaluator_llm

    llm = llm or _get_evaluator_llm()
    embeddings = embeddings or _get_evaluator_embeddings()

    results = []
    total = len(dataset)

    for i, sample in enumerate(dataset, 1):
        logger.info("[{}/{}] {}", i, total, sample.id)
        result = evaluate_sample(sample, config, llm=llm, embeddings=embeddings)
        results.append(result)

    return results


def export_results(
    results: list[EvalResult],
    output_path: Path | str,
    config: RunConfig | None = None,
) -> Path:
    """Export evaluation results to CSV.

    Args:
        results: List of EvalResult from evaluate_dataset.
        output_path: Path for the output CSV file.
        config: Optional config for metadata in the output.

    Returns:
        Path to the written CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all unique metric names
    all_metric_names = sorted({m.name for r in results for m in r.metrics})

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        header = ["question", "answer_preview", "reference_preview", *all_metric_names]
        writer.writerow(header)

        # Rows
        for result in results:
            scores = result.scores
            row = [
                result.question,
                (result.answer or "")[:200],
                (result.reference or "")[:200],
                *[scores.get(m, "") for m in all_metric_names],
            ]
            writer.writerow(row)

    logger.info("Exported {} results to {}", len(results), output_path)
    return output_path


def compute_summary(results: list[EvalResult]) -> dict[str, float]:
    """Compute average scores across all results for each metric."""
    from collections import defaultdict

    totals: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for metric in result.metrics:
            if metric.score >= 0:  # Exclude errors (-1)
                totals[metric.name].append(metric.score)

    return {name: sum(scores) / len(scores) for name, scores in totals.items() if scores}
