import pytest

from casecrawler.models.blueprint import (
    BlueprintGenerationRequest,
    CohortArchetype,
    CohortPlan,
    GenerationAttemptStatus,
    GenerationRole,
    GenerationRolePolicy,
)
from casecrawler.models.synthetic import Modality
from casecrawler.llm.base import StructuredGenerationResult
from casecrawler.storage.dataset_store import DatasetStore


class FakePlannerProvider:
    def __init__(self, plan: CohortPlan) -> None:
        self.plan = plan
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
            data=self.plan,
            input_tokens=123,
            output_tokens=45,
            model="planner-model",
        )


class FailingPlannerProvider:
    async def generate_structured(self, prompt, schema, system="", **kwargs):
        raise RuntimeError("planner boom")


def _request() -> BlueprintGenerationRequest:
    return BlueprintGenerationRequest(
        request="Generate outpatient anticoagulation decision cases.",
        target_count=2,
        domains=["cardiology"],
        settings=["outpatient"],
        role_policies=[
            GenerationRolePolicy(
                role=GenerationRole.PLANNER,
                provider="openrouter",
                model="anthropic/claude-sonnet-4-6",
                temperature=0.2,
            )
        ],
        required_grounding=True,
    )


def _plan() -> CohortPlan:
    return CohortPlan(
        plan_id="plan-1",
        request="Generate outpatient anticoagulation decision cases.",
        target_count=2,
        domains=["cardiology"],
        settings=["outpatient"],
        archetypes=[
            CohortArchetype(
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
        ],
        required_grounding=True,
        created_by=GenerationRole.PLANNER,
    )


@pytest.mark.asyncio
async def test_cohort_planner_uses_byok_policy_and_persists_attempt(tmp_path):
    from casecrawler.generation.blueprint_planner import CohortPlanner

    store = DatasetStore(db_path=str(tmp_path / "datasets.db"))
    provider = FakePlannerProvider(_plan())
    planner = CohortPlanner(provider_factory=lambda provider_name, model: provider)

    plan = await planner.plan(_request(), dataset_id="ds-1", store=store)

    assert plan.plan_id.startswith("plan-")
    assert plan.plan_id != "plan-1"
    assert plan.request == "Generate outpatient anticoagulation decision cases."
    assert plan.target_count == 2
    assert plan.required_grounding is True
    assert store.get_cohort_plan(plan.plan_id) == plan
    attempts = store.list_generation_attempts(dataset_id="ds-1")
    assert len(attempts) == 1
    assert attempts[0].role == GenerationRole.PLANNER
    assert attempts[0].status == GenerationAttemptStatus.SUCCEEDED
    assert attempts[0].provider == "openrouter"
    assert attempts[0].model == "anthropic/claude-sonnet-4-6"
    assert attempts[0].artifact_id == plan.plan_id
    assert attempts[0].total_tokens == 168
    assert attempts[0].prompt_hash
    assert provider.calls[0]["schema"] is CohortPlan
    assert provider.calls[0]["kwargs"]["temperature"] == 0.2
    assert "anticoagulation decision cases" in provider.calls[0]["prompt"]
    assert "recipe" not in provider.calls[0]["prompt"].lower()


@pytest.mark.asyncio
async def test_cohort_planner_requires_planner_role_policy():
    from casecrawler.generation.blueprint_planner import CohortPlanner

    request = BlueprintGenerationRequest(
        request="Generate broad internal medicine cases.",
        target_count=2,
    )

    with pytest.raises(ValueError, match="planner role policy"):
        await CohortPlanner(provider_factory=lambda provider_name, model: None).plan(
            request,
            dataset_id="ds-1",
        )


@pytest.mark.asyncio
async def test_cohort_planner_persists_failed_attempt(tmp_path):
    from casecrawler.generation.blueprint_planner import CohortPlanner

    store = DatasetStore(db_path=str(tmp_path / "datasets.db"))
    planner = CohortPlanner(
        provider_factory=lambda provider_name, model: FailingPlannerProvider()
    )

    with pytest.raises(RuntimeError, match="planner boom"):
        await planner.plan(_request(), dataset_id="ds-1", store=store)

    attempts = store.list_generation_attempts(dataset_id="ds-1")
    assert len(attempts) == 1
    assert attempts[0].role == GenerationRole.PLANNER
    assert attempts[0].status == GenerationAttemptStatus.FAILED
    assert attempts[0].provider == "openrouter"
    assert attempts[0].model == "anthropic/claude-sonnet-4-6"
    assert attempts[0].artifact_id is None
    assert attempts[0].errors == ["planner boom"]
    assert attempts[0].total_tokens == 0
    assert attempts[0].prompt_hash
