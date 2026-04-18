"""Evaluation dataset loader.

Loads curated question-answer pairs from ``data/evaluation/testset.json``
and converts them to RAGAS ``EvaluationDataset`` format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

DEFAULT_TESTSET_PATH = Path(__file__).resolve().parents[3] / "data" / "evaluation" / "testset.json"


@dataclass
class EvalSample:
    """A single evaluation sample."""

    id: str
    question: str
    reference: str = ""
    question_type: str = ""
    expected_tools: list[str] = field(default_factory=list)

    # Populated after running the pipeline
    answer: str = ""
    contexts: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)


@dataclass
class EvalDataset:
    """Collection of evaluation samples."""

    samples: list[EvalSample]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def by_type(self, question_type: str) -> list[EvalSample]:
        return [s for s in self.samples if s.question_type == question_type]

    @property
    def question_types(self) -> list[str]:
        return sorted({s.question_type for s in self.samples})


def load_testset(path: Path | str | None = None) -> EvalDataset:
    """Load evaluation testset from JSON file.

    Args:
        path: Path to the testset JSON file. Uses default if not provided.

    Returns:
        EvalDataset with loaded samples.
    """
    path = Path(path) if path else DEFAULT_TESTSET_PATH

    if not path.exists():
        raise FileNotFoundError(f"Testset not found: {path}")

    with open(path) as f:
        data = json.load(f)

    samples = []
    for item in data.get("samples", []):
        samples.append(
            EvalSample(
                id=item["id"],
                question=item["question"],
                reference=item.get("reference", ""),
                question_type=item.get("question_type", ""),
                expected_tools=item.get("expected_tools", []),
            )
        )

    logger.info("Loaded {} evaluation samples from {}", len(samples), path.name)
    return EvalDataset(samples=samples, metadata=data.get("metadata", {}))


def to_ragas_dataset(dataset: EvalDataset) -> list[dict[str, Any]]:
    """Convert EvalDataset to RAGAS-compatible format.

    Each sample becomes a dict with keys expected by RAGAS:
    user_input, response, retrieved_contexts, reference.

    Only includes samples where answer and contexts have been populated
    (i.e., after running the pipeline).
    """
    ragas_samples = []
    for sample in dataset.samples:
        if not sample.answer:
            continue
        entry = {
            "user_input": sample.question,
            "response": sample.answer,
            "retrieved_contexts": sample.contexts,
        }
        if sample.reference:
            entry["reference"] = sample.reference
        ragas_samples.append(entry)
    return ragas_samples
