import pytest

from casecrawler.llm.base import StructuredGenerationResult
from casecrawler.models.blueprint import (
    BlueprintEvidence,
    BlueprintGenerationRequest,
    ClinicalBlueprint,
    CohortArchetype,
    CohortPlan,
    GenerationAttemptStatus,
    GenerationRole,
    GenerationRolePolicy,
)
from casecrawler.models.synthetic import Modality
from casecrawler.storage.dataset_store import DatasetStore


class FakeBlueprintProvider:
    def __init__(self, blueprint: ClinicalBlueprint) -> None:
        self.blueprint = blueprint
        self.calls = []

    async def generate_structured(self, prompt, schema, system="", **kwargs):
        self.calls.append(
            {
                "prompt": prompt,
                "schema": schema,
                "system": system,
                "kwargs": kwargs,
            }
        )
        return StructuredGenerationResult(
            data=self.blueprint,
            input_tokens=200,
            output_tokens=120,
            model="blueprint-model",
        )


class FailingBlueprintProvider:
    async def generate_structured(self, prompt, schema, system="", **kwargs):
        raise RuntimeError("blueprint boom")


def _archetype() -> CohortArchetype:
    return CohortArchetype(
        name="anticoagulation decision",
        organ_system="cardiovascular",
        setting="outpatient",
        target_count=2,
        acuity_mix={"routine": 0.5, "urgent": 0.5},
        difficulty_mix={"moderate": 1.0},
        required_modalities=[
            Modality.STRUCTURED_EHR,
            Modality.CLINICAL_TEXT,
        ],
        task_targets=["medication_reconciliation"],
    )


def _plan() -> CohortPlan:
    return CohortPlan(
        plan_id="plan-1",
        request="Generate outpatient anticoagulation decision cases.",
        target_count=2,
        domains=["cardiology"],
        settings=["outpatient"],
        archetypes=[_archetype()],
        required_grounding=True,
        created_by=GenerationRole.PLANNER,
    )


def _request() -> BlueprintGenerationRequest:
    return BlueprintGenerationRequest(
        request="Generate outpatient anticoagulation decision cases.",
        target_count=2,
        role_policies=[
            GenerationRolePolicy(
                role=GenerationRole.BLUEPRINT_GENERATOR,
                provider="openrouter",
                model="anthropic/claude-sonnet-4-6",
                temperature=0.3,
            )
        ],
    )


def _raw_blueprint() -> ClinicalBlueprint:
    return ClinicalBlueprint(
        blueprint_id="model-controlled-id",
        dataset_id="wrong-dataset",
        cohort_plan_id="wrong-plan",
        archetype_name="wrong archetype",
        organ_system="wrong system",
        setting="wrong setting",
        patient={"age": 72, "sex": "female"},
        chief_concern="Atrial fibrillation anticoagulation follow-up.",
        diagnoses=[
            {
                "name": "atrial fibrillation",
                "supporting_findings": ["ECG confirms AF"],
            }
        ],
        clinical_reasoning_targets=["Review renal dosing and bleeding risk."],
        evidence=BlueprintEvidence(
            supported_claims=["AF anticoagulation requires renal-dose review."],
            citations=[{"source": "dailymed", "claim": "renal-dose review"}],
        ),
    )


@pytest.mark.asyncio
async def test_blueprint_generator_uses_role_policy_and_persists_blueprint(tmp_path):
    from casecrawler.generation.blueprint_generator import ClinicalBlueprintGenerator

    store = DatasetStore(db_path=str(tmp_path / "datasets.db"))
    provider = FakeBlueprintProvider(_raw_blueprint())
    generator = ClinicalBlueprintGenerator(
        provider_factory=lambda provider_name, model: provider
    )

    blueprint = await generator.generate_for_archetype(
        _request(),
        plan=_plan(),
        archetype=_archetype(),
        dataset_id="ds-1",
        sequence_index=0,
        store=store,
    )

    assert blueprint.blueprint_id.startswith("bp-")
    assert blueprint.blueprint_id != "model-controlled-id"
    assert blueprint.dataset_id == "ds-1"
    assert blueprint.cohort_plan_id == "plan-1"
    assert blueprint.archetype_name == "anticoagulation decision"
    assert blueprint.organ_system == "cardiovascular"
    assert blueprint.setting == "outpatient"
    assert store.get_blueprint(blueprint.blueprint_id) == blueprint
    attempts = store.list_generation_attempts(dataset_id="ds-1")
    assert len(attempts) == 1
    assert attempts[0].role == GenerationRole.BLUEPRINT_GENERATOR
    assert attempts[0].status == GenerationAttemptStatus.SUCCEEDED
    assert attempts[0].provider == "openrouter"
    assert attempts[0].model == "anthropic/claude-sonnet-4-6"
    assert attempts[0].artifact_id == blueprint.blueprint_id
    assert attempts[0].total_tokens == 320
    assert attempts[0].prompt_hash
    assert provider.calls[0]["schema"] is ClinicalBlueprint
    assert provider.calls[0]["kwargs"]["temperature"] == 0.3
    assert "anticoagulation decision" in provider.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_blueprint_generator_requires_blueprint_generator_policy():
    from casecrawler.generation.blueprint_generator import ClinicalBlueprintGenerator

    request = BlueprintGenerationRequest(
        request="Generate broad internal medicine cases.",
        target_count=2,
    )

    with pytest.raises(ValueError, match="blueprint_generator role policy"):
        await ClinicalBlueprintGenerator(
            provider_factory=lambda provider_name, model: None
        ).generate_for_archetype(
            request,
            plan=_plan(),
            archetype=_archetype(),
            dataset_id="ds-1",
        )


@pytest.mark.asyncio
async def test_blueprint_generator_persists_failed_attempt(tmp_path):
    from casecrawler.generation.blueprint_generator import ClinicalBlueprintGenerator

    store = DatasetStore(db_path=str(tmp_path / "datasets.db"))
    generator = ClinicalBlueprintGenerator(
        provider_factory=lambda provider_name, model: FailingBlueprintProvider()
    )

    with pytest.raises(RuntimeError, match="blueprint boom"):
        await generator.generate_for_archetype(
            _request(),
            plan=_plan(),
            archetype=_archetype(),
            dataset_id="ds-1",
            store=store,
        )

    attempts = store.list_generation_attempts(dataset_id="ds-1")
    assert len(attempts) == 1
    assert attempts[0].role == GenerationRole.BLUEPRINT_GENERATOR
    assert attempts[0].status == GenerationAttemptStatus.FAILED
    assert attempts[0].artifact_id is None
    assert attempts[0].errors == ["blueprint boom"]
    assert attempts[0].total_tokens == 0
    assert attempts[0].prompt_hash
