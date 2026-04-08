from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from casecrawler.cli import cli
from casecrawler.models.case import (
    Complication,
    DecisionChoice,
    DifficultyLevel,
    GeneratedCase,
    GroundTruth,
    Patient,
    ReviewResult,
)


def _fake_case() -> GeneratedCase:
    return GeneratedCase(
        case_id="test-1",
        topic="SAH",
        difficulty=DifficultyLevel.RESIDENT,
        specialty=["neurosurgery"],
        patient=Patient(age=42, sex="female", demographics="Healthy"),
        vignette="A 42-year-old woman...",
        decision_prompt="What would you do next?",
        ground_truth=GroundTruth(
            diagnosis="SAH", optimal_next_step="CT",
            rationale="Sensitive", key_findings=["headache"],
        ),
        decision_tree=[
            DecisionChoice(
                choice="CT", is_correct=True, error_type=None,
                reasoning="Right", outcome="Confirmed",
                consequence=None, next_decision=None,
            ),
        ],
        complications=[Complication(trigger="delay", detail="6h", event="Rebleed", outcome="Death")],
        review=ReviewResult(accuracy_score=0.95, pedagogy_score=0.9, bias_score=0.92, approved=True, notes=[]),
        sources=[],
        metadata={"generated_at": "2026-04-08", "model": "test", "retry_count": 0},
    )


def test_generate_command():
    runner = CliRunner()
    with patch("casecrawler.cli.get_provider") as mock_get_provider, \
         patch("casecrawler.cli.GenerationPipeline") as MockPipeline, \
         patch("casecrawler.cli.Retriever"), \
         patch("casecrawler.cli.Store"), \
         patch("casecrawler.cli.CaseStore") as MockCaseStore:
        mock_pipeline = MockPipeline.return_value
        mock_pipeline.generate_batch = AsyncMock(return_value={
            "cases": [_fake_case()],
            "generated": 1,
            "failed": 0,
            "total_input_tokens": 1000,
            "total_output_tokens": 500,
        })
        mock_case_store = MockCaseStore.return_value
        mock_case_store.save = MagicMock()

        result = runner.invoke(cli, ["generate", "SAH", "--count", "1"])
        assert result.exit_code == 0
        assert "1" in result.output


def test_cases_command():
    runner = CliRunner()
    with patch("casecrawler.cli.CaseStore") as MockCaseStore:
        mock_store = MockCaseStore.return_value
        mock_store.list_cases.return_value = [_fake_case()]

        result = runner.invoke(cli, ["cases"])
        assert result.exit_code == 0
        assert "SAH" in result.output
