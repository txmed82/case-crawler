from collections import Counter

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


class SequenceProvider:
    def __init__(self, results) -> None:
        self.results = list(results)
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
        data = self.results.pop(0)
        return StructuredGenerationResult(
            data=data,
            input_tokens=50,
            output_tokens=25,
            model="fake",
        )


class RoutingProviderFactory:
    def __init__(self, *, judge_results, repair_results=()) -> None:
        self.judge_provider = SequenceProvider(judge_results)
        self.repair_provider = SequenceProvider(repair_results)

    def __call__(self, provider_name, model):
        if model == "judge-model":
            return self.judge_provider
        if model == "repair-model":
            return self.repair_provider
        raise ValueError(f"Unexpected model {model}")


def _request(max_repair_rounds: int = 1) -> BlueprintGenerationRequest:
    return BlueprintGenerationRequest(
        request="Judge and repair outpatient anticoagulation decision blueprints.",
        target_count=1,
        max_repair_rounds=max_repair_rounds,
        role_policies=[
            GenerationRolePolicy(
                role=GenerationRole.JUDGE,
                provider="openai",
                model="judge-model",
            ),
            GenerationRolePolicy(
                role=GenerationRole.REPAIR,
                provider="openai",
                model="repair-model",
                temperature=0.1,
            ),
        ],
    )


def _blueprint(**overrides) -> ClinicalBlueprint:
    payload = {
        "blueprint_id": "bp-1",
        "dataset_id": "ds-1",
        "cohort_plan_id": "plan-1",
        "archetype_name": "anticoagulation decision",
        "organ_system": "cardiovascular",
        "setting": "outpatient",
        "patient": {"age": 72, "sex": "female"},
        "chief_concern": "Atrial fibrillation anticoagulation follow-up.",
        "diagnoses": [
            {
                "name": "atrial fibrillation",
                "supporting_findings": ["ECG confirms AF"],
            }
        ],
        "clinical_reasoning_targets": ["Review renal dosing and bleeding risk."],
        "evidence": BlueprintEvidence(
            supported_claims=["AF anticoagulation requires renal-dose review."],
            citations=[{"source": "dailymed", "claim": "renal-dose review"}],
        ),
    }
    payload.update(overrides)
    return ClinicalBlueprint(**payload)


def _judge_report(*, passed: bool, score: float = 0.8) -> JudgeReport:
    return JudgeReport(
        report_id="model-controlled",
        dataset_id="wrong-dataset",
        artifact_id="wrong-artifact",
        role=GenerationRole.REPAIR,
        score=score,
        passed=passed,
        rubric="blueprint_plausibility",
        findings=[{"criterion": "diagnostic_support", "passed": passed}],
    )


@pytest.mark.asyncio
async def test_blueprint_repair_loop_repairs_failed_judge_report(tmp_path):
    from casecrawler.generation.blueprint_repair import BlueprintRepairLoop

    store = DatasetStore(db_path=str(tmp_path / "datasets.db"))
    original = _blueprint()
    repaired_raw = _blueprint(
        blueprint_id="model-repaired-id",
        dataset_id="wrong-dataset",
        chief_concern="Atrial fibrillation follow-up with renal-dose review.",
        evidence=BlueprintEvidence(
            supported_claims=[
                "AF anticoagulation requires renal-dose review.",
                "Renal function changes anticoagulant dosing.",
            ],
            citations=[
                {"source": "dailymed", "claim": "renal-dose review"},
                {"source": "dailymed", "claim": "renal dosing"},
            ],
        ),
    )
    factory = RoutingProviderFactory(
        judge_results=[
            _judge_report(passed=False, score=0.42),
            _judge_report(passed=True, score=0.93),
        ],
        repair_results=[repaired_raw],
    )

    result = await BlueprintRepairLoop(provider_factory=factory).run(
        _request(),
        original,
        store=store,
    )

    assert result.passed is True
    assert result.repair_rounds == 1
    assert result.original_blueprint == original
    assert result.final_blueprint.blueprint_id.startswith("bp-")
    assert result.final_blueprint.blueprint_id != original.blueprint_id
    assert result.final_blueprint.dataset_id == "ds-1"
    assert result.final_blueprint.metadata["parent_blueprint_id"] == "bp-1"
    assert result.final_blueprint.metadata["repair_round"] == 1
    assert len(result.judge_reports) == 2
    assert len(result.repaired_blueprints) == 1
    assert store.get_blueprint(result.final_blueprint.blueprint_id) == (
        result.final_blueprint
    )
    attempts = store.list_generation_attempts(dataset_id="ds-1")
    assert Counter((attempt.role, attempt.status) for attempt in attempts) == Counter(
        {
            (GenerationRole.JUDGE, GenerationAttemptStatus.SUCCEEDED): 2,
            (GenerationRole.REPAIR, GenerationAttemptStatus.REPAIR_REQUESTED): 1,
            (GenerationRole.REPAIR, GenerationAttemptStatus.SUCCEEDED): 1,
        }
    )
    repair_attempts = [
        attempt for attempt in attempts if attempt.role == GenerationRole.REPAIR
    ]
    assert {attempt.metadata["repair_round"] for attempt in repair_attempts} == {1}
    assert factory.repair_provider.calls[0]["kwargs"]["temperature"] == 0.1
    assert "bp-1" in factory.repair_provider.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_blueprint_repair_loop_does_not_repair_passing_blueprint(tmp_path):
    from casecrawler.generation.blueprint_repair import BlueprintRepairLoop

    store = DatasetStore(db_path=str(tmp_path / "datasets.db"))
    original = _blueprint()
    factory = RoutingProviderFactory(
        judge_results=[_judge_report(passed=True, score=0.95)]
    )

    result = await BlueprintRepairLoop(provider_factory=factory).run(
        _request(),
        original,
        store=store,
    )

    assert result.passed is True
    assert result.repair_rounds == 0
    assert result.final_blueprint == original
    assert result.repaired_blueprints == []
    assert factory.repair_provider.calls == []
    assert len(store.list_judge_reports(artifact_id="bp-1")) == 1
    assert len(store.list_generation_attempts(dataset_id="ds-1")) == 1


@pytest.mark.asyncio
async def test_blueprint_repair_loop_respects_zero_repair_rounds(tmp_path):
    from casecrawler.generation.blueprint_repair import BlueprintRepairLoop

    store = DatasetStore(db_path=str(tmp_path / "datasets.db"))
    original = _blueprint()
    factory = RoutingProviderFactory(
        judge_results=[_judge_report(passed=False, score=0.35)]
    )

    result = await BlueprintRepairLoop(provider_factory=factory).run(
        _request(max_repair_rounds=0),
        original,
        store=store,
    )

    assert result.passed is False
    assert result.repair_rounds == 0
    assert result.final_blueprint == original
    assert result.repaired_blueprints == []
    assert factory.repair_provider.calls == []
    attempts = store.list_generation_attempts(dataset_id="ds-1")
    assert len(attempts) == 1
    assert attempts[0].role == GenerationRole.JUDGE
