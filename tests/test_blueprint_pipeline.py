import pytest

from casecrawler.llm.base import StructuredGenerationResult
from casecrawler.models.blueprint import (
    BlueprintEvidence,
    BlueprintGenerationRequest,
    ClinicalBlueprint,
    CohortArchetype,
    CohortPlan,
    GenerationRole,
    GenerationRolePolicy,
)
from casecrawler.models.synthetic import Modality
from casecrawler.storage.dataset_store import DatasetStore


class RoutingProviderFactory:
    def __init__(self, plan: CohortPlan, blueprint: ClinicalBlueprint) -> None:
        self.plan = plan
        self.blueprint = blueprint

    def __call__(self, provider_name, model):
        if model == "planner-model":
            return FakeStructuredProvider(self.plan, input_tokens=20, output_tokens=10)
        return FakeStructuredProvider(self.blueprint, input_tokens=30, output_tokens=15)


class FakeStructuredProvider:
    def __init__(self, data, *, input_tokens: int, output_tokens: int) -> None:
        self.data = data
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    async def generate_structured(self, prompt, schema, system="", **kwargs):
        return StructuredGenerationResult(
            data=self.data,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            model="fake",
        )


def _request() -> BlueprintGenerationRequest:
    return BlueprintGenerationRequest(
        request="Generate outpatient anticoagulation decision cases.",
        target_count=2,
        role_policies=[
            GenerationRolePolicy(
                role=GenerationRole.PLANNER,
                provider="openrouter",
                model="planner-model",
            ),
            GenerationRolePolicy(
                role=GenerationRole.BLUEPRINT_GENERATOR,
                provider="openrouter",
                model="blueprint-model",
            ),
        ],
    )


def _archetype() -> CohortArchetype:
    return CohortArchetype(
        name="anticoagulation decision",
        organ_system="cardiovascular",
        setting="outpatient",
        target_count=2,
        acuity_mix={"routine": 1.0},
        difficulty_mix={"moderate": 1.0},
        required_modalities=[Modality.STRUCTURED_EHR, Modality.CLINICAL_TEXT],
        task_targets=["medication_reconciliation"],
    )


def _plan() -> CohortPlan:
    return CohortPlan(
        plan_id="model-plan",
        request="Generate outpatient anticoagulation decision cases.",
        target_count=2,
        archetypes=[_archetype()],
        created_by=GenerationRole.PLANNER,
    )


def _blueprint() -> ClinicalBlueprint:
    return ClinicalBlueprint(
        blueprint_id="model-blueprint",
        dataset_id="model-dataset",
        cohort_plan_id="model-plan",
        archetype_name="model archetype",
        organ_system="model organ system",
        setting="model setting",
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
        ),
    )


@pytest.mark.asyncio
async def test_blueprint_pipeline_generates_and_persists_planned_cohort(tmp_path):
    from casecrawler.generation.blueprint_pipeline import BlueprintPipeline

    store = DatasetStore(db_path=str(tmp_path / "datasets.db"))
    pipeline = BlueprintPipeline(
        provider_factory=RoutingProviderFactory(_plan(), _blueprint())
    )

    result = await pipeline.generate(_request(), dataset_id="ds-1", store=store)

    assert result.dataset_id == "ds-1"
    assert result.generated_count == 2
    assert result.plan.plan_id.startswith("plan-")
    assert len(result.blueprints) == 2
    assert {blueprint.dataset_id for blueprint in result.blueprints} == {"ds-1"}
    assert {blueprint.cohort_plan_id for blueprint in result.blueprints} == {
        result.plan.plan_id
    }
    assert len(store.list_blueprints(dataset_id="ds-1")) == 2
    assert len(store.list_generation_attempts(dataset_id="ds-1")) == 3
