from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from casecrawler.generation.pipeline import GenerationPipeline
from casecrawler.llm.base import StructuredGenerationResult
from casecrawler.models.case import (
    Complication,
    DecisionChoice,
    DifficultyLevel,
    GeneratedCase,
    GroundTruth,
    Patient,
    ReviewResult,
)


def _mock_case_gen_result():
    return StructuredGenerationResult(
        data=MagicMock(
            patient=Patient(age=42, sex="female", demographics="Healthy"),
            vignette="A 42-year-old woman presents with thunderclap headache.",
            decision_prompt="What would you do next?",
            ground_truth=GroundTruth(
                diagnosis="SAH", optimal_next_step="CT head",
                rationale="Most sensitive", key_findings=["thunderclap headache"],
            ),
            specialty=["neurosurgery"],
        ),
        input_tokens=500, output_tokens=300, model="test-model",
    )


def _mock_tree_result():
    return StructuredGenerationResult(
        data=MagicMock(
            decision_tree=[
                DecisionChoice(
                    choice="CT head", is_correct=True, error_type=None,
                    reasoning="Correct", outcome="Confirmed",
                    consequence=None, next_decision=None,
                ),
            ],
            complications=[
                Complication(trigger="delayed_diagnosis", detail="6h", event="Rebleed", outcome="Death"),
            ],
        ),
        input_tokens=600, output_tokens=400, model="test-model",
    )


def _mock_review_approved():
    return StructuredGenerationResult(
        data=ReviewResult(
            accuracy_score=0.95, pedagogy_score=0.9, bias_score=0.92,
            approved=True, notes=[],
        ),
        input_tokens=800, output_tokens=100, model="test-model",
    )


def _mock_review_rejected():
    return StructuredGenerationResult(
        data=ReviewResult(
            accuracy_score=0.5, pedagogy_score=0.9, bias_score=0.92,
            approved=False, notes=["Diagnosis incorrect"],
        ),
        input_tokens=800, output_tokens=100, model="test-model",
    )


@pytest.fixture
def mock_retriever():
    r = MagicMock()
    r.retrieve.return_value = [
        {"chunk_id": "c1", "text": "SAH content", "credibility": "guideline",
         "credibility_rank": 0, "source_document_id": "pubmed:1", "source": "pubmed",
         "score": 0.9, "specialty": "", "doi": "", "url": ""},
    ]
    r.format_context.return_value = "[Source 1] (guideline, pubmed)\nSAH content"
    return r


@pytest.fixture
def mock_provider():
    return AsyncMock()


@pytest.mark.asyncio
async def test_pipeline_generates_case(mock_retriever, mock_provider):
    mock_provider.generate_structured.side_effect = [
        _mock_case_gen_result(),
        _mock_tree_result(),
        _mock_review_approved(),
    ]

    pipeline = GenerationPipeline(
        provider=mock_provider,
        retriever=mock_retriever,
        max_retries=3,
        review_threshold=0.7,
    )

    result = await pipeline.generate_one(topic="SAH", difficulty="resident")
    assert result is not None
    assert isinstance(result, GeneratedCase)
    assert result.topic == "SAH"
    assert result.review.approved is True
    assert mock_provider.generate_structured.call_count == 3


@pytest.mark.asyncio
async def test_pipeline_retries_on_rejection(mock_retriever, mock_provider):
    mock_provider.generate_structured.side_effect = [
        _mock_case_gen_result(),
        _mock_tree_result(),
        _mock_review_rejected(),    # first review rejects
        _mock_case_gen_result(),     # retry case gen
        _mock_tree_result(),         # retry tree
        _mock_review_approved(),     # second review approves
    ]

    pipeline = GenerationPipeline(
        provider=mock_provider,
        retriever=mock_retriever,
        max_retries=3,
        review_threshold=0.7,
    )

    result = await pipeline.generate_one(topic="SAH", difficulty="resident")
    assert result is not None
    assert result.review.approved is True
    assert result.metadata["retry_count"] == 1


@pytest.mark.asyncio
async def test_pipeline_returns_none_after_max_retries(mock_retriever, mock_provider):
    mock_provider.generate_structured.side_effect = [
        _mock_case_gen_result(), _mock_tree_result(), _mock_review_rejected(),
        _mock_case_gen_result(), _mock_tree_result(), _mock_review_rejected(),
        _mock_case_gen_result(), _mock_tree_result(), _mock_review_rejected(),
    ]

    pipeline = GenerationPipeline(
        provider=mock_provider,
        retriever=mock_retriever,
        max_retries=3,
        review_threshold=0.7,
    )

    result = await pipeline.generate_one(topic="SAH", difficulty="resident")
    assert result is None
