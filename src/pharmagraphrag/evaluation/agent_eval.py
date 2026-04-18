"""Agent-specific evaluation metrics.

Evaluates agent tool selection accuracy by comparing expected tools
from the curated testset against the tools actually called by the agent.
Also provides agent goal accuracy (did the agent answer the question?).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from pharmagraphrag.evaluation.dataset import EvalSample


@dataclass
class AgentEvalResult:
    """Evaluation result for agent tool selection and goal accuracy."""

    sample_id: str
    question: str
    expected_tools: list[str]
    actual_tools: list[str]
    tool_precision: float = 0.0
    tool_recall: float = 0.0
    tool_f1: float = 0.0
    goal_achieved: bool = False

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "expected_tools": self.expected_tools,
            "actual_tools": self.actual_tools,
            "tool_precision": self.tool_precision,
            "tool_recall": self.tool_recall,
            "tool_f1": self.tool_f1,
            "goal_achieved": self.goal_achieved,
        }


def evaluate_tool_selection(sample: EvalSample) -> AgentEvalResult:
    """Evaluate whether the agent called the right tools.

    Computes precision, recall, and F1 for tool selection by comparing
    expected_tools (from testset) vs actual tool_calls (from agent response).

    Args:
        sample: EvalSample with expected_tools and tool_calls populated.

    Returns:
        AgentEvalResult with tool selection metrics.
    """
    expected = set(sample.expected_tools)
    actual = set(sample.tool_calls)

    if not expected and not actual:
        return AgentEvalResult(
            sample_id=sample.id,
            question=sample.question,
            expected_tools=sample.expected_tools,
            actual_tools=sample.tool_calls,
            tool_precision=1.0,
            tool_recall=1.0,
            tool_f1=1.0,
            goal_achieved=bool(sample.answer),
        )

    true_positives = len(expected & actual)
    precision = true_positives / len(actual) if actual else 0.0
    recall = true_positives / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return AgentEvalResult(
        sample_id=sample.id,
        question=sample.question,
        expected_tools=sample.expected_tools,
        actual_tools=sample.tool_calls,
        tool_precision=round(precision, 3),
        tool_recall=round(recall, 3),
        tool_f1=round(f1, 3),
        goal_achieved=bool(sample.answer and "ERROR" not in sample.answer),
    )


@dataclass
class AgentEvalSummary:
    """Aggregated agent evaluation metrics."""

    total_samples: int = 0
    avg_tool_precision: float = 0.0
    avg_tool_recall: float = 0.0
    avg_tool_f1: float = 0.0
    goal_accuracy: float = 0.0
    results: list[AgentEvalResult] = field(default_factory=list)


def evaluate_agent_dataset(samples: list[EvalSample]) -> AgentEvalSummary:
    """Evaluate agent tool selection across all samples.

    Args:
        samples: List of EvalSample with tool_calls populated (after running agent pipeline).

    Returns:
        AgentEvalSummary with averaged metrics.
    """
    results = []
    for sample in samples:
        result = evaluate_tool_selection(sample)
        results.append(result)
        logger.debug(
            "[{}] P={:.2f} R={:.2f} F1={:.2f} goal={}",
            sample.id,
            result.tool_precision,
            result.tool_recall,
            result.tool_f1,
            result.goal_achieved,
        )

    n = len(results)
    if n == 0:
        return AgentEvalSummary()

    return AgentEvalSummary(
        total_samples=n,
        avg_tool_precision=round(sum(r.tool_precision for r in results) / n, 3),
        avg_tool_recall=round(sum(r.tool_recall for r in results) / n, 3),
        avg_tool_f1=round(sum(r.tool_f1 for r in results) / n, 3),
        goal_accuracy=round(sum(1 for r in results if r.goal_achieved) / n, 3),
        results=results,
    )
