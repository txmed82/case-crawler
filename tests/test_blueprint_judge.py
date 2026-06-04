import pytest

from casecrawler.llm.base import StructuredGenerationResult
from casecrawler.models.blueprint import (
    BlueprintEvidence,
    BlueprintGenerationRequest,
    ClinicalBlueprint,
    GenerationAttemptStatus,
    GenerationRole,
    GenerationRolePolicy,
    JudgeReport,
)
from casecrawler.storage.dataset_store import DatasetStore


class FakeJudgeProvider:
    def __init__(self, report: JudgeReport) -> None:
        self.report = report
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
            data=self.report,
            input_tokens=80,
            output_tokens=30,
            model="judge-model",
        )


class FailingJudgeProvider:
    async def generate_structured(self, prompt, schema, system="", **kwargs):
        raise RuntimeError("judge boom")


def _request() -> BlueprintGenerationRequest:
    return BlueprintGenerationRequest(
        request="Judge generated cardiology blueprints before release.",
        target_count=1,
        role_policies=[
            GenerationRolePolicy(
                role=GenerationRole.JUDGE,
                provider="openai",
                model="gpt-4.1-mini",
                temperature=0.0,
            )
        ],
    )


def _blueprint() -> ClinicalBlueprint:
    return ClinicalBlueprint(
        blueprint_id="bp-1",
        dataset_id="ds-1",
        cohort_plan_id="plan-1",
        archetype_name="anticoagulation decision",
        organ_system="cardiovascular",
        setting="outpatient",
        patient={"age": 72, "sex": "female"},
        chief_concern="Atrial fibrillation anticoagulation follow-up.",
        diagnoses=[
            {
                "name": "atrial fibrillation",
                "supporting_findings": ["ECG confirms AF"],
            }
        ],
        clinical_reasoning_targets=["Review renal dosing and bleeding risk."],
        safety_constraints=["Review bleeding risk before anticoagulation."],
        evidence=BlueprintEvidence(
            supported_claims=["AF anticoagulation requires renal-dose review."],
            citations=[{"source": "dailymed", "claim": "renal-dose review"}],
        ),
    )


def _raw_report() -> JudgeReport:
    return JudgeReport(
        report_id="model-controlled-id",
        dataset_id="wrong-dataset",
        artifact_id="wrong-artifact",
        role=GenerationRole.REPAIR,
        score=0.91,
        passed=True,
        rubric="blueprint_plausibility",
        findings=[{"criterion": "diagnostic_support", "passed": True}],
    )


@pytest.mark.asyncio
async def test_blueprint_judge_uses_role_policy_and_persists_report(tmp_path):
    from casecrawler.generation.blueprint_judge import BlueprintJudge

    store = DatasetStore(db_path=str(tmp_path / "datasets.db"))
    provider = FakeJudgeProvider(_raw_report())
    judge = BlueprintJudge(provider_factory=lambda provider_name, model: provider)

    report = await judge.evaluate(_request(), _blueprint(), store=store)

    assert report.report_id.startswith("judge-")
    assert report.report_id != "model-controlled-id"
    assert report.dataset_id == "ds-1"
    assert report.artifact_id == "bp-1"
    assert report.role == GenerationRole.JUDGE
    assert report.score == 0.91
    assert report.passed is True
    assert report.rubric == "blueprint_plausibility"
    assert store.list_judge_reports(artifact_id="bp-1") == [report]

    attempts = store.list_generation_attempts(dataset_id="ds-1")
    assert len(attempts) == 1
    assert attempts[0].role == GenerationRole.JUDGE
    assert attempts[0].status == GenerationAttemptStatus.SUCCEEDED
    assert attempts[0].provider == "openai"
    assert attempts[0].model == "gpt-4.1-mini"
    assert attempts[0].artifact_id == "bp-1"
    assert attempts[0].total_tokens == 110
    assert attempts[0].prompt_hash
    assert provider.calls[0]["schema"] is JudgeReport
    assert provider.calls[0]["kwargs"]["temperature"] == 0.0
    assert "bp-1" in provider.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_blueprint_judge_requires_judge_policy():
    from casecrawler.generation.blueprint_judge import BlueprintJudge

    request = BlueprintGenerationRequest(
        request="Judge generated cardiology blueprints before release.",
        target_count=1,
    )

    with pytest.raises(ValueError, match="judge role policy"):
        await BlueprintJudge(
            provider_factory=lambda provider_name, model: None
        ).evaluate(request, _blueprint())


@pytest.mark.asyncio
async def test_blueprint_judge_persists_failed_attempt(tmp_path):
    from casecrawler.generation.blueprint_judge import BlueprintJudge

    store = DatasetStore(db_path=str(tmp_path / "datasets.db"))
    judge = BlueprintJudge(
        provider_factory=lambda provider_name, model: FailingJudgeProvider()
    )

    with pytest.raises(RuntimeError, match="judge boom"):
        await judge.evaluate(_request(), _blueprint(), store=store)

    attempts = store.list_generation_attempts(dataset_id="ds-1")
    assert len(attempts) == 1
    assert attempts[0].role == GenerationRole.JUDGE
    assert attempts[0].status == GenerationAttemptStatus.FAILED
    assert attempts[0].artifact_id == "bp-1"
    assert attempts[0].errors == ["judge boom"]
    assert attempts[0].total_tokens == 0
    assert attempts[0].prompt_hash
