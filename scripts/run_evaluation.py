"""Batch evaluation script for PharmaGraphRAG.

Runs the curated testset against one or more pipeline modes (classic, agent, multi)
and generates RAGAS metric reports + agent tool evaluation results.

Usage:
    # Evaluate classic pipeline (requires local or Cloud Run API)
    python scripts/run_evaluation.py --mode classic --api-url http://localhost:8000

    # Evaluate agent mode
    python scripts/run_evaluation.py --mode agent --api-url http://localhost:8000

    # Evaluate all modes
    python scripts/run_evaluation.py --mode all --api-url http://localhost:8000

    # Evaluate against production
    python scripts/run_evaluation.py --mode all --api-url https://pharmagraphrag-api-893694384146.us-central1.run.app

    # Custom testset
    python scripts/run_evaluation.py --testset data/evaluation/custom.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from loguru import logger  # noqa: E402

from pharmagraphrag.evaluation.agent_eval import evaluate_agent_dataset  # noqa: E402
from pharmagraphrag.evaluation.dataset import load_testset  # noqa: E402
from pharmagraphrag.evaluation.runner import (  # noqa: E402
    RunConfig,
    compute_summary,
    evaluate_dataset,
    export_results,
)


def run_evaluation(
    mode: str,
    api_url: str,
    testset_path: str | None,
    output_dir: str,
    model: str | None,
    limit: int | None = None,
) -> None:
    """Run evaluation for a single pipeline mode."""
    dataset = load_testset(testset_path)
    if limit is not None and limit > 0:
        dataset.samples = dataset.samples[:limit]
    logger.info("Loaded {} samples, evaluating in '{}' mode", len(dataset), mode)

    config = RunConfig(api_url=api_url, mode=mode, model=model)

    start = time.perf_counter()
    results = evaluate_dataset(dataset, config)
    elapsed = time.perf_counter() - start

    # Export RAGAS metrics
    output_path = Path(output_dir) / f"ragas_{mode}.csv"
    export_results(results, output_path, config)

    # Summary
    summary = compute_summary(results)
    logger.info("=== {} mode summary ({:.0f}s) ===", mode.upper(), elapsed)
    for metric_name, avg_score in sorted(summary.items()):
        logger.info("  {}: {:.3f}", metric_name, avg_score)

    # Agent-specific evaluation (tool selection)
    if mode in ("agent", "multi"):
        agent_summary = evaluate_agent_dataset(dataset.samples)
        logger.info("=== Agent tool selection ===")
        logger.info("  Precision: {:.3f}", agent_summary.avg_tool_precision)
        logger.info("  Recall: {:.3f}", agent_summary.avg_tool_recall)
        logger.info("  F1: {:.3f}", agent_summary.avg_tool_f1)
        logger.info("  Goal accuracy: {:.1%}", agent_summary.goal_accuracy)

        # Export agent results
        agent_path = Path(output_dir) / f"agent_tools_{mode}.csv"
        _export_agent_results(agent_summary.results, agent_path)


def _export_agent_results(results, output_path: Path) -> None:
    """Export agent tool evaluation to CSV."""
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_id",
                "question",
                "expected_tools",
                "actual_tools",
                "precision",
                "recall",
                "f1",
                "goal_achieved",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.sample_id,
                    r.question[:100],
                    "|".join(r.expected_tools),
                    "|".join(r.actual_tools),
                    r.tool_precision,
                    r.tool_recall,
                    r.tool_f1,
                    r.goal_achieved,
                ]
            )
    logger.info("Exported agent tool evaluation to {}", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="PharmaGraphRAG RAGAS Evaluation")
    parser.add_argument(
        "--mode",
        choices=["classic", "agent", "multi", "all"],
        default="classic",
        help="Pipeline mode to evaluate (default: classic)",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the PharmaGraphRAG API",
    )
    parser.add_argument(
        "--testset",
        default=None,
        help="Path to testset JSON (default: data/evaluation/testset.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/evaluation/results",
        help="Output directory for CSV reports",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model to use for the pipeline (e.g. 'gemini-2.5-flash')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of evaluation samples (useful for quick subset runs)",
    )
    args = parser.parse_args()

    modes = ["classic", "agent", "multi"] if args.mode == "all" else [args.mode]

    for mode in modes:
        logger.info("Starting evaluation: mode={}, api={}", mode, args.api_url)
        run_evaluation(mode, args.api_url, args.testset, args.output_dir, args.model, args.limit)
        logger.info("Completed evaluation for mode={}", mode)


if __name__ == "__main__":
    main()
