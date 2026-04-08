import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from casecrawler.generation.case_generator import CaseGeneratorAgent
from casecrawler.generation.clinical_reviewer import ClinicalReviewerAgent
from casecrawler.generation.decision_tree_builder import DecisionTreeBuilderAgent
from casecrawler.llm.base import StructuredGenerationResult
from casecrawler.models.case import (
    Complication,
    DecisionChoice,
    DifficultyLevel,
    GroundTruth,
    Patient,
    ReviewResult,
)


@pytest.fixture
def mock_provider():
    return AsyncMock()


# --- Case Generator ---


@pytest.mark.asyncio
async def test_case_generator(mock_provider):
    mock_provider.generate_structured.return_value = StructuredGenerationResult(
        data=MagicMock(
            patient=Patient(age=42, sex="female", demographics="Healthy"),
            vignette="A 42-year-old woman...",
            decision_prompt="What would you do next?",
            ground_truth=GroundTruth(
                diagnosis="SAH", optimal_next_step="CT head",
                rationale="Most sensitive", key_findings=["thunderclap headache"],
            ),
            specialty=["neurosurgery"],
        ),
        input_tokens=500,
        output_tokens=300,
        model="test-model",
    )

    agent = CaseGeneratorAgent(provider=mock_provider)
    result = await agent.generate(
        topic="subarachnoid hemorrhage",
        difficulty="resident",
        context="SAH requires immediate CT...",
    )
    assert result.data.vignette == "A 42-year-old woman..."
    assert mock_provider.generate_structured.called


# --- Decision Tree Builder ---


@pytest.mark.asyncio
async def test_decision_tree_builder(mock_provider):
    mock_provider.generate_structured.return_value = StructuredGenerationResult(
        data=MagicMock(
            decision_tree=[
                DecisionChoice(
                    choice="CT head", is_correct=True, error_type=None,
                    reasoning="Correct", outcome="SAH confirmed",
                    consequence=None, next_decision=None,
                ),
                DecisionChoice(
                    choice="MRI", is_correct=False, error_type="common_mistake",
                    reasoning="Takes too long", outcome="Delay",
                    consequence="Risk of rebleed", next_decision=None,
                ),
            ],
            complications=[
                Complication(trigger="delayed_diagnosis", detail="6h", event="Rebleed", outcome="Death"),
            ],
        ),
        input_tokens=600,
        output_tokens=400,
        model="test-model",
    )

    agent = DecisionTreeBuilderAgent(provider=mock_provider)
    result = await agent.build(
        vignette="A 42-year-old woman...",
        ground_truth_json='{"diagnosis": "SAH"}',
        difficulty="resident",
        context="SAH management guidelines...",
    )
    assert len(result.data.decision_tree) == 2
    assert result.data.decision_tree[0].is_correct is True


# --- Clinical Reviewer ---


@pytest.mark.asyncio
async def test_clinical_reviewer_approves(mock_provider):
    mock_provider.generate_structured.return_value = StructuredGenerationResult(
        data=ReviewResult(
            accuracy_score=0.95, pedagogy_score=0.9, bias_score=0.92,
            approved=True, notes=[],
        ),
        input_tokens=800,
        output_tokens=100,
        model="test-model",
    )

    agent = ClinicalReviewerAgent(provider=mock_provider, threshold=0.7)
    result = await agent.review(case_json="{}", context="source material")
    assert result.data.approved is True


@pytest.mark.asyncio
async def test_clinical_reviewer_rejects(mock_provider):
    mock_provider.generate_structured.return_value = StructuredGenerationResult(
        data=ReviewResult(
            accuracy_score=0.5, pedagogy_score=0.9, bias_score=0.92,
            approved=False, notes=["Diagnosis is incorrect"],
        ),
        input_tokens=800,
        output_tokens=100,
        model="test-model",
    )

    agent = ClinicalReviewerAgent(provider=mock_provider, threshold=0.7)
    result = await agent.review(case_json="{}", context="source material")
    assert result.data.approved is False
    assert "Diagnosis is incorrect" in result.data.notes
