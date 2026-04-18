"""Tests for the evaluation module.

Covers dataset loading, RAGAS metric wrappers, agent evaluation,
and the evaluation runner. All external dependencies (LLM, API) are mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pharmagraphrag.evaluation.agent_eval import (
    evaluate_agent_dataset,
    evaluate_tool_selection,
)
from pharmagraphrag.evaluation.dataset import (
    EvalDataset,
    EvalSample,
    load_testset,
    to_ragas_dataset,
)
from pharmagraphrag.evaluation.metrics import EvalResult, MetricResult, score_sample
from pharmagraphrag.evaluation.runner import (
    PipelineResponse,
    RunConfig,
    _call_agent,
    _call_classic,
    compute_summary,
    evaluate_sample,
    export_results,
    run_pipeline,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TESTSET = {
    "metadata": {"version": "1.0.0", "description": "test"},
    "samples": [
        {
            "id": "q01",
            "question": "What are the side effects of aspirin?",
            "reference": "Aspirin causes GI bleeding and nausea.",
            "question_type": "drug_info",
            "expected_tools": ["search_drug_info"],
        },
        {
            "id": "q02",
            "question": "Does warfarin interact with aspirin?",
            "reference": "Yes, warfarin interacts with aspirin.",
            "question_type": "interaction",
            "expected_tools": ["search_drug_info", "list_drug_interactions"],
        },
    ],
}


@pytest.fixture
def testset_path(tmp_path: Path) -> Path:
    path = tmp_path / "testset.json"
    path.write_text(json.dumps(SAMPLE_TESTSET))
    return path


@pytest.fixture
def sample_dataset() -> EvalDataset:
    return EvalDataset(
        samples=[
            EvalSample(
                id="q01",
                question="What are the side effects of aspirin?",
                reference="Aspirin causes GI bleeding.",
                question_type="drug_info",
                expected_tools=["search_drug_info"],
            ),
            EvalSample(
                id="q02",
                question="Does warfarin interact with aspirin?",
                reference="Yes, they interact.",
                question_type="interaction",
                expected_tools=["search_drug_info", "list_drug_interactions"],
            ),
        ],
    )


@pytest.fixture
def eval_sample_with_response() -> EvalSample:
    sample = EvalSample(
        id="q01",
        question="What are the side effects of aspirin?",
        reference="Aspirin causes GI bleeding.",
        question_type="drug_info",
        expected_tools=["search_drug_info"],
    )
    sample.answer = "Aspirin can cause gastrointestinal bleeding."
    sample.contexts = ["ASPIRIN CAUSES GASTROINTESTINAL HAEMORRHAGE (report_count: 1523)"]
    sample.tool_calls = ["search_drug_info"]
    return sample


# ===========================================================================
# Dataset tests
# ===========================================================================


class TestLoadTestset:
    def test_load_from_file(self, testset_path: Path):
        dataset = load_testset(testset_path)
        assert len(dataset) == 2
        assert dataset.samples[0].id == "q01"
        assert dataset.samples[0].question_type == "drug_info"

    def test_load_preserves_expected_tools(self, testset_path: Path):
        dataset = load_testset(testset_path)
        assert dataset.samples[0].expected_tools == ["search_drug_info"]
        assert "list_drug_interactions" in dataset.samples[1].expected_tools

    def test_load_metadata(self, testset_path: Path):
        dataset = load_testset(testset_path)
        assert dataset.metadata["version"] == "1.0.0"

    def test_load_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_testset(tmp_path / "nonexistent.json")


class TestEvalDataset:
    def test_length(self, sample_dataset: EvalDataset):
        assert len(sample_dataset) == 2

    def test_iter(self, sample_dataset: EvalDataset):
        items = list(sample_dataset)
        assert len(items) == 2

    def test_by_type(self, sample_dataset: EvalDataset):
        drug_info = sample_dataset.by_type("drug_info")
        assert len(drug_info) == 1
        assert drug_info[0].id == "q01"

    def test_question_types(self, sample_dataset: EvalDataset):
        types = sample_dataset.question_types
        assert "drug_info" in types
        assert "interaction" in types

    def test_by_type_empty(self, sample_dataset: EvalDataset):
        assert sample_dataset.by_type("unknown") == []


class TestToRagasDataset:
    def test_converts_populated_samples(self, sample_dataset: EvalDataset):
        sample_dataset.samples[0].answer = "GI bleeding"
        sample_dataset.samples[0].contexts = ["some context"]
        ragas = to_ragas_dataset(sample_dataset)
        assert len(ragas) == 1  # only q01 has answer
        assert ragas[0]["user_input"] == sample_dataset.samples[0].question
        assert ragas[0]["response"] == "GI bleeding"

    def test_skips_unanswered_samples(self, sample_dataset: EvalDataset):
        ragas = to_ragas_dataset(sample_dataset)
        assert len(ragas) == 0

    def test_includes_reference(self, sample_dataset: EvalDataset):
        sample_dataset.samples[0].answer = "test"
        sample_dataset.samples[0].contexts = ["ctx"]
        ragas = to_ragas_dataset(sample_dataset)
        assert "reference" in ragas[0]
        assert ragas[0]["reference"] == "Aspirin causes GI bleeding."


# ===========================================================================
# Metrics tests
# ===========================================================================


class TestMetricResult:
    def test_creation(self):
        mr = MetricResult(name="faithfulness", score=0.85)
        assert mr.name == "faithfulness"
        assert mr.score == 0.85

    def test_with_details(self):
        mr = MetricResult(name="test", score=0.5, details={"key": "val"})
        assert mr.details["key"] == "val"


class TestEvalResult:
    def test_scores_property(self):
        result = EvalResult(
            question="test?",
            answer="answer",
            metrics=[
                MetricResult(name="faithfulness", score=0.9),
                MetricResult(name="relevancy", score=0.8),
            ],
        )
        assert result.scores == {"faithfulness": 0.9, "relevancy": 0.8}


class TestScoreSample:
    @patch("pharmagraphrag.evaluation.metrics._get_evaluator_embeddings")
    @patch("pharmagraphrag.evaluation.metrics._get_evaluator_llm")
    @patch("pharmagraphrag.evaluation.metrics.get_reference_free_metrics")
    def test_score_without_reference(self, mock_free, mock_llm, mock_emb):
        mock_metric = MagicMock()
        mock_metric.single_score.return_value = 0.85
        type(mock_metric).__name__ = "Faithfulness"
        mock_free.return_value = [mock_metric]

        result = score_sample(
            question="test?",
            answer="answer",
            contexts=["context"],
            reference=None,
        )
        assert len(result.metrics) == 1
        assert result.metrics[0].score == 0.85

    @patch("pharmagraphrag.evaluation.metrics._get_evaluator_embeddings")
    @patch("pharmagraphrag.evaluation.metrics._get_evaluator_llm")
    @patch("pharmagraphrag.evaluation.metrics.get_reference_metrics")
    @patch("pharmagraphrag.evaluation.metrics.get_reference_free_metrics")
    def test_score_with_reference(self, mock_free, mock_ref, mock_llm, mock_emb):
        mock_metric1 = MagicMock()
        mock_metric1.single_score.return_value = 0.9
        type(mock_metric1).__name__ = "Faithfulness"
        mock_free.return_value = [mock_metric1]

        mock_metric2 = MagicMock()
        mock_metric2.single_score.return_value = 0.7
        type(mock_metric2).__name__ = "ContextRecall"
        mock_ref.return_value = [mock_metric2]

        result = score_sample(
            question="test?",
            answer="answer",
            contexts=["context"],
            reference="ground truth",
        )
        assert len(result.metrics) == 2
        assert result.scores["Faithfulness"] == 0.9
        assert result.scores["ContextRecall"] == 0.7

    @patch("pharmagraphrag.evaluation.metrics._get_evaluator_embeddings")
    @patch("pharmagraphrag.evaluation.metrics._get_evaluator_llm")
    @patch("pharmagraphrag.evaluation.metrics.get_reference_free_metrics")
    def test_score_handles_metric_error(self, mock_free, mock_llm, mock_emb):
        mock_metric = MagicMock()
        mock_metric.single_score.side_effect = RuntimeError("API error")
        type(mock_metric).__name__ = "Faithfulness"
        mock_free.return_value = [mock_metric]

        result = score_sample(
            question="test?",
            answer="answer",
            contexts=["context"],
        )
        assert result.metrics[0].score == -1.0


# ===========================================================================
# Runner tests
# ===========================================================================


class TestRunConfig:
    def test_defaults(self):
        config = RunConfig()
        assert config.mode == "classic"
        assert config.api_url == "http://localhost:8000"
        assert config.timeout == 120.0


class TestPipelineResponse:
    def test_defaults(self):
        resp = PipelineResponse()
        assert resp.answer == ""
        assert resp.contexts == []
        assert resp.error is None


class TestCallClassic:
    @patch("pharmagraphrag.evaluation.runner.httpx.post")
    def test_successful_call(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "answer": "Aspirin causes GI bleeding.",
            "sources": [
                {"snippet": "ASPIRIN CAUSES GI_HAEMORRHAGE", "type": "graph"},
                {"snippet": "Aspirin label text", "type": "vector"},
            ],
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        config = RunConfig(api_url="http://test:8000")
        resp = _call_classic("test question?", config)

        assert resp.answer == "Aspirin causes GI bleeding."
        assert len(resp.contexts) == 2
        assert resp.error is None

    @patch("pharmagraphrag.evaluation.runner.httpx.post")
    def test_handles_error(self, mock_post):
        mock_post.side_effect = Exception("Connection refused")

        config = RunConfig(api_url="http://test:8000")
        resp = _call_classic("test?", config)

        assert resp.error is not None
        assert "Connection refused" in resp.error


class TestCallAgent:
    @patch("pharmagraphrag.evaluation.runner.httpx.post")
    def test_parses_agent_response(self, mock_post):
        """Verify _call_agent correctly parses tool_results, vector_data, and tool_calls."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "answer": "Aspirin interacts with Warfarin.",
            "tool_calls": [
                {"tool": "search_drug_info", "args": {}},
                {"tool": "list_drug_interactions", "args": {}},
            ],
            "tool_results": [
                {"tool": "search_drug_info", "content": "Drug info context"},
                {"tool": "list_drug_interactions", "content": "Interaction context"},
            ],
            "vector_data": [
                {"text": "Vector snippet about aspirin"},
                {"snippet": "Another vector snippet"},
            ],
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        config = RunConfig(api_url="http://test:8000")
        resp = _call_agent("test question?", config)

        assert resp.answer == "Aspirin interacts with Warfarin."
        assert resp.tool_calls == ["search_drug_info", "list_drug_interactions"]
        assert len(resp.contexts) == 4  # 2 tool_results + 2 vector_data
        assert "Drug info context" in resp.contexts
        assert "Vector snippet about aspirin" in resp.contexts
        assert resp.error is None

    @patch("pharmagraphrag.evaluation.runner.httpx.post")
    def test_handles_error(self, mock_post):
        mock_post.side_effect = Exception("Timeout")

        config = RunConfig(api_url="http://test:8000")
        resp = _call_agent("test?", config)

        assert resp.error is not None
        assert "Timeout" in resp.error


class TestRunPipeline:
    @patch("pharmagraphrag.evaluation.runner._call_classic")
    def test_routes_to_classic(self, mock_classic):
        mock_classic.return_value = PipelineResponse(answer="test")
        config = RunConfig(mode="classic")
        resp = run_pipeline("question?", config)
        mock_classic.assert_called_once()
        assert resp.answer == "test"

    @patch("pharmagraphrag.evaluation.runner._call_agent")
    def test_routes_to_agent(self, mock_agent):
        mock_agent.return_value = PipelineResponse(answer="agent answer")
        config = RunConfig(mode="agent")
        run_pipeline("question?", config)
        mock_agent.assert_called_once_with("question?", config, endpoint="/agent/query")

    @patch("pharmagraphrag.evaluation.runner._call_agent")
    def test_routes_to_multi(self, mock_agent):
        mock_agent.return_value = PipelineResponse(answer="multi answer")
        config = RunConfig(mode="multi")
        run_pipeline("question?", config)
        mock_agent.assert_called_once_with("question?", config, endpoint="/agent/multi")

    def test_unknown_mode(self):
        config = RunConfig(mode="unknown")
        resp = run_pipeline("question?", config)
        assert resp.error is not None


class TestEvaluateSample:
    @patch("pharmagraphrag.evaluation.runner.score_sample")
    @patch("pharmagraphrag.evaluation.runner.run_pipeline")
    def test_successful_evaluation(self, mock_pipeline, mock_score):
        mock_pipeline.return_value = PipelineResponse(
            answer="GI bleeding",
            contexts=["context"],
            tool_calls=["search_drug_info"],
        )
        mock_score.return_value = EvalResult(
            question="test?",
            answer="GI bleeding",
            metrics=[MetricResult(name="faithfulness", score=0.9)],
        )

        sample = EvalSample(id="q01", question="test?")
        config = RunConfig()
        result = evaluate_sample(sample, config)

        assert result.metrics[0].score == 0.9
        assert sample.answer == "GI bleeding"
        assert sample.tool_calls == ["search_drug_info"]

    @patch("pharmagraphrag.evaluation.runner.run_pipeline")
    def test_pipeline_error(self, mock_pipeline):
        mock_pipeline.return_value = PipelineResponse(error="timeout")

        sample = EvalSample(id="q01", question="test?")
        config = RunConfig()
        result = evaluate_sample(sample, config)

        assert result.metrics[0].name == "error"
        assert result.metrics[0].score == -1.0


class TestExportResults:
    def test_exports_csv(self, tmp_path: Path):
        results = [
            EvalResult(
                question="test?",
                answer="answer",
                metrics=[
                    MetricResult(name="faithfulness", score=0.9),
                    MetricResult(name="relevancy", score=0.8),
                ],
                reference="ground truth",
            ),
        ]
        path = export_results(results, tmp_path / "results.csv")
        assert path.exists()

        content = path.read_text()
        assert "faithfulness" in content
        assert "0.9" in content


class TestComputeSummary:
    def test_averages(self):
        results = [
            EvalResult(
                question="q1",
                answer="a1",
                metrics=[
                    MetricResult(name="faithfulness", score=0.8),
                    MetricResult(name="relevancy", score=0.6),
                ],
            ),
            EvalResult(
                question="q2",
                answer="a2",
                metrics=[
                    MetricResult(name="faithfulness", score=0.9),
                    MetricResult(name="relevancy", score=0.7),
                ],
            ),
        ]
        summary = compute_summary(results)
        assert summary["faithfulness"] == pytest.approx(0.85)
        assert summary["relevancy"] == pytest.approx(0.65)

    def test_excludes_errors(self):
        results = [
            EvalResult(
                question="q1",
                answer="a1",
                metrics=[
                    MetricResult(name="faithfulness", score=0.8),
                    MetricResult(name="error", score=-1.0),
                ],
            ),
        ]
        summary = compute_summary(results)
        assert "error" not in summary
        assert summary["faithfulness"] == 0.8


# ===========================================================================
# Agent evaluation tests
# ===========================================================================


class TestEvaluateToolSelection:
    def test_perfect_match(self, eval_sample_with_response: EvalSample):
        result = evaluate_tool_selection(eval_sample_with_response)
        assert result.tool_precision == 1.0
        assert result.tool_recall == 1.0
        assert result.tool_f1 == 1.0
        assert result.goal_achieved is True

    def test_partial_match(self):
        sample = EvalSample(
            id="q02",
            question="test?",
            expected_tools=["search_drug_info", "list_drug_interactions"],
        )
        sample.answer = "Some answer"
        sample.tool_calls = ["search_drug_info", "search_adverse_events"]

        result = evaluate_tool_selection(sample)
        assert result.tool_precision == 0.5  # 1/2
        assert result.tool_recall == 0.5  # 1/2
        assert result.tool_f1 == 0.5

    def test_no_tools_expected_or_called(self):
        sample = EvalSample(id="q03", question="test?", expected_tools=[])
        sample.tool_calls = []

        result = evaluate_tool_selection(sample)
        assert result.tool_precision == 1.0
        assert result.tool_recall == 1.0

    def test_no_answer_means_goal_not_achieved(self):
        sample = EvalSample(
            id="q04",
            question="test?",
            expected_tools=["search_drug_info"],
        )
        sample.answer = ""
        sample.tool_calls = ["search_drug_info"]

        result = evaluate_tool_selection(sample)
        assert result.goal_achieved is False

    def test_error_answer_means_goal_not_achieved(self):
        sample = EvalSample(
            id="q05",
            question="test?",
            expected_tools=["search_drug_info"],
        )
        sample.answer = "ERROR: timeout"
        sample.tool_calls = ["search_drug_info"]

        result = evaluate_tool_selection(sample)
        assert result.goal_achieved is False

    def test_to_dict(self, eval_sample_with_response: EvalSample):
        result = evaluate_tool_selection(eval_sample_with_response)
        d = result.to_dict()
        assert d["sample_id"] == "q01"
        assert d["tool_f1"] == 1.0


class TestEvaluateAgentDataset:
    def test_summary_metrics(self):
        samples = [
            EvalSample(id="q01", question="q1", expected_tools=["tool_a"]),
            EvalSample(id="q02", question="q2", expected_tools=["tool_a", "tool_b"]),
        ]
        samples[0].answer = "answer1"
        samples[0].tool_calls = ["tool_a"]
        samples[1].answer = "answer2"
        samples[1].tool_calls = ["tool_a"]

        summary = evaluate_agent_dataset(samples)
        assert summary.total_samples == 2
        assert summary.avg_tool_precision == 1.0  # both called tools were correct
        assert summary.avg_tool_recall == 0.75  # q01: 1/1, q02: 1/2

    def test_empty_dataset(self):
        summary = evaluate_agent_dataset([])
        assert summary.total_samples == 0


# ===========================================================================
# Integration: default testset loads
# ===========================================================================


class TestDefaultTestset:
    def test_default_testset_exists(self):
        """Verify the curated testset file exists and loads."""
        from pharmagraphrag.evaluation.dataset import DEFAULT_TESTSET_PATH

        assert DEFAULT_TESTSET_PATH.exists(), (
            f"Expected curated default testset at {DEFAULT_TESTSET_PATH}"
        )
        dataset = load_testset()
        assert len(dataset) >= 20
        assert all(s.question for s in dataset.samples)
        assert all(s.reference for s in dataset.samples)
