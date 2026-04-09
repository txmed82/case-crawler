from unittest.mock import AsyncMock, MagicMock

import pytest

from casecrawler.generation.blueprint_reviewer import BlueprintReviewResult
from casecrawler.generation.case_planner import CasePlannerOutput
from casecrawler.generation.consistency_checker import ConsistencyCheckerOutput
from casecrawler.generation.multi_step_pipeline import MultiStepPipeline
from casecrawler.generation.phase_renderer import PhaseRendererOutput
from casecrawler.llm.base import StructuredGenerationResult
from casecrawler.models.blueprint import CaseBlueprint, PhaseBlueprint
from casecrawler.models.case import (
    DifficultyLevel,
    GeneratedCase,
    GroundTruth,
    Patient,
    ReviewResult,
)
from casecrawler.models.diagnostics import VitalSigns
from casecrawler.models.phase import CasePhase, PhaseDecision, PhaseOutcome


def _blueprint():
    return CaseBlueprint(
        diagnosis="SAH",
        clinical_arc="SAH via CT, surgical management",
        phase_count=3,
        phases=[
            PhaseBlueprint(
                phase_number=i, time_offset=f"T+{i*30}min",
                clinical_context=f"Phase {i}", available_diagnostics=[], pending_diagnostics=[],
                decision_type="order_workup", correct_action=f"Action {i}",
                key_reasoning=f"Reasoning {i}",
            )
            for i in range(1, 4)
        ],
        branching_points=[],
        expected_complications=["vasospasm"],
    )


def _phase(n: int):
    return CasePhase(
        phase_number=n, time_offset=f"T+{n*30}min",
        narrative=f"Phase {n} narrative",
        vitals=VitalSigns(hr=80+n, bp_systolic=120, bp_diastolic=80, rr=16, spo2=98.0, temp_c=37.0),
        lab_results=[], imaging_results=[],
        decisions=[
            PhaseDecision(
                action=f"Action {n}", is_optimal=True, quality="optimal",
                reasoning="Correct", clinical_outcome="Good outcome",
            ),
        ],
        phase_outcome=PhaseOutcome(
            optimal_next_phase=n+1 if n < 3 else None,
            patient_status="stable",
            narrative_transition=f"Transition from phase {n}",
        ),
    )


def _mock_planner():
    return StructuredGenerationResult(
        data=CasePlannerOutput(
            blueprint=_blueprint(),
            patient=Patient(age=42, sex="female", demographics="No significant PMH"),
            specialty=["neurosurgery"],
        ),
        input_tokens=500, output_tokens=400, model="test-model",
    )


def _mock_blueprint_review_approved():
    return StructuredGenerationResult(
        data=BlueprintReviewResult(approved=True, notes=[]),
        input_tokens=300, output_tokens=50, model="test-model",
    )


def _mock_phase_render(n: int):
    return StructuredGenerationResult(
        data=PhaseRendererOutput(phase=_phase(n)),
        input_tokens=800, output_tokens=600, model="test-model",
    )


def _mock_consistency_ok():
    return StructuredGenerationResult(
        data=ConsistencyCheckerOutput(issues=[]),
        input_tokens=500, output_tokens=50, model="test-model",
    )


def _mock_clinical_review_approved():
    return StructuredGenerationResult(
        data=ReviewResult(
            accuracy_score=0.95, pedagogy_score=0.9, bias_score=0.92,
            approved=True, notes=[],
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


@pytest.mark.asyncio
async def test_multi_step_pipeline_generates_case(mock_retriever):
    provider = AsyncMock()
    # Call sequence: planner, blueprint_review, render×3, consistency, clinical_review
    provider.generate_structured.side_effect = [
        _mock_planner(),
        _mock_blueprint_review_approved(),
        _mock_phase_render(1),
        _mock_phase_render(2),
        _mock_phase_render(3),
        _mock_consistency_ok(),
        _mock_clinical_review_approved(),
    ]

    pipeline = MultiStepPipeline(
        provider=provider, retriever=mock_retriever,
        max_retries=3, review_threshold=0.7,
    )
    result = await pipeline.generate_one(topic="SAH", difficulty="resident")

    assert result is not None
    assert isinstance(result, GeneratedCase)
    assert result.is_multi_step()
    assert len(result.phases) == 3
    assert result.blueprint is not None
    assert result.blueprint.diagnosis == "SAH"
    assert result.review.approved is True


@pytest.mark.asyncio
async def test_multi_step_pipeline_retries_on_blueprint_rejection(mock_retriever):
    provider = AsyncMock()
    provider.generate_structured.side_effect = [
        _mock_planner(),
        StructuredGenerationResult(
            data=BlueprintReviewResult(approved=False, notes=["Too few phases"]),
            input_tokens=300, output_tokens=80, model="test-model",
        ),
        _mock_planner(),
        _mock_blueprint_review_approved(),
        _mock_phase_render(1),
        _mock_phase_render(2),
        _mock_phase_render(3),
        _mock_consistency_ok(),
        _mock_clinical_review_approved(),
    ]

    pipeline = MultiStepPipeline(
        provider=provider, retriever=mock_retriever,
        max_retries=3, review_threshold=0.7,
    )
    result = await pipeline.generate_one(topic="SAH", difficulty="resident")

    assert result is not None
    assert result.is_multi_step()


@pytest.mark.asyncio
async def test_multi_step_pipeline_returns_none_after_max_retries(mock_retriever):
    provider = AsyncMock()
    rejected_review = StructuredGenerationResult(
        data=ReviewResult(
            accuracy_score=0.5, pedagogy_score=0.9, bias_score=0.92,
            approved=False, notes=["Inaccurate"],
        ),
        input_tokens=800, output_tokens=100, model="test-model",
    )

    def side_effects():
        for _ in range(3):
            yield _mock_planner()
            yield _mock_blueprint_review_approved()
            yield _mock_phase_render(1)
            yield _mock_phase_render(2)
            yield _mock_phase_render(3)
            yield _mock_consistency_ok()
            yield rejected_review

    provider.generate_structured.side_effect = list(side_effects())

    pipeline = MultiStepPipeline(
        provider=provider, retriever=mock_retriever,
        max_retries=3, review_threshold=0.7,
    )
    result = await pipeline.generate_one(topic="SAH", difficulty="resident")

    assert result is None
