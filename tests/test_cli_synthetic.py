import json

from click.testing import CliRunner

from casecrawler.cli import cli
from casecrawler.generation import blueprint_pipeline as blueprint_pipeline_module
from casecrawler.generation.blueprint_pipeline import BlueprintPipelineResult
from casecrawler.models.blueprint import (
    BlueprintEvidence,
    ClinicalBlueprint,
    CohortArchetype,
    CohortPlan,
    GenerationRole,
    JudgeReport,
)
from casecrawler.models.synthetic import Modality
from casecrawler.storage.dataset_store import DatasetStore


def _blueprint_cli_result(dataset_id: str) -> BlueprintPipelineResult:
    archetype = CohortArchetype(
        name="anticoagulation decision",
        organ_system="cardiovascular",
        setting="outpatient",
        target_count=1,
        required_modalities=[Modality.STRUCTURED_EHR, Modality.CLINICAL_TEXT],
    )
    plan = CohortPlan(
        plan_id="plan-1",
        request="Generate anticoagulation decision cases.",
        target_count=1,
        archetypes=[archetype],
        created_by=GenerationRole.PLANNER,
    )
    blueprint = ClinicalBlueprint(
        blueprint_id="bp-1",
        dataset_id=dataset_id,
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
        evidence=BlueprintEvidence(
            supported_claims=["AF anticoagulation requires renal-dose review."],
            citations=[{"source": "dailymed", "claim": "renal-dose review"}],
        ),
    )
    return BlueprintPipelineResult(dataset_id=dataset_id, plan=plan, blueprints=[blueprint])


def test_generate_dataset_command_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(cli, ["generate-dataset", "sepsis", "--count", "1"])

    assert result.exit_code == 0
    assert "Generated: 1" in result.output
    assert "Approved: 1" in result.output


def test_generate_dataset_invalid_complexity_fails():
    runner = CliRunner()

    result = runner.invoke(cli, ["generate-dataset", "sepsis", "--complexity", "bogus"])

    assert result.exit_code != 0
    assert "Invalid value for '--complexity'" in result.output


def test_generate_blueprints_command_uses_model_driven_request(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    captured = []

    class FakeBlueprintPipeline:
        async def generate(self, req, *, dataset_id, store):
            captured.append((req, dataset_id, store))
            store.save_cohort_plan(_blueprint_cli_result(dataset_id).plan)
            for blueprint in _blueprint_cli_result(dataset_id).blueprints:
                store.save_blueprint(blueprint)
            return _blueprint_cli_result(dataset_id)

    monkeypatch.setattr(
        blueprint_pipeline_module,
        "BlueprintPipeline",
        FakeBlueprintPipeline,
    )
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "generate-blueprints",
            "Generate anticoagulation decision cases.",
            "--count",
            "1",
            "--planner-provider",
            "openrouter",
            "--planner-model",
            "planner-model",
            "--blueprint-provider",
            "openrouter",
            "--blueprint-model",
            "blueprint-model",
        ],
    )

    assert result.exit_code == 0
    assert "Dataset: blueprint-ds-" in result.output
    assert "Plan: plan-1" in result.output
    assert "Blueprints: 1" in result.output
    assert captured[0][0].request == "Generate anticoagulation decision cases."
    assert captured[0][0].target_count == 1
    assert captured[0][0].policy_for(GenerationRole.PLANNER).model == "planner-model"
    assert (
        captured[0][0].policy_for(GenerationRole.BLUEPRINT_GENERATOR).model
        == "blueprint-model"
    )
    assert isinstance(captured[0][2], DatasetStore)


def test_export_blueprints_command_writes_jsonl(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = DatasetStore()
    result = _blueprint_cli_result("ds-blueprint")
    store.save_cohort_plan(result.plan)
    for blueprint in result.blueprints:
        store.save_blueprint(blueprint)
    runner = CliRunner()

    exported = runner.invoke(
        cli,
        [
            "export-blueprints",
            "--dataset-id",
            "ds-blueprint",
            "--output",
            "blueprints.jsonl",
        ],
    )

    payload = json.loads((tmp_path / "blueprints.jsonl").read_text().splitlines()[0])
    assert exported.exit_code == 0
    assert "Exported 1 blueprint artifact(s)" in exported.output
    assert payload["artifact_type"] == "casecrawler_clinical_blueprint"
    assert payload["blueprint"]["blueprint_id"] == "bp-1"
    assert payload["cohort_plan"]["plan_id"] == "plan-1"


def test_validate_blueprints_command_persists_validation_reports(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = DatasetStore()
    result = _blueprint_cli_result("ds-blueprint")
    store.save_cohort_plan(result.plan)
    for blueprint in result.blueprints:
        store.save_blueprint(blueprint)
    runner = CliRunner()

    validated = runner.invoke(
        cli,
        [
            "validate-blueprints",
            "--dataset-id",
            "ds-blueprint",
        ],
    )

    report = DatasetStore().get_blueprint_validation_report("bp-1")
    assert validated.exit_code == 0
    assert "Validated 1 blueprint artifact(s)" in validated.output
    assert report is not None
    assert report.blueprint_id == "bp-1"
    assert report.clinically_plausible is True


def test_blueprint_cli_materializes_and_exports_release_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = DatasetStore()
    result = _blueprint_cli_result("ds-blueprint")
    store.save_cohort_plan(result.plan)
    for blueprint in result.blueprints:
        store.save_blueprint(blueprint)
        store.save_judge_report(
            JudgeReport(
                report_id="judge-1",
                dataset_id="ds-blueprint",
                artifact_id=blueprint.blueprint_id,
                role=GenerationRole.JUDGE,
                score=0.93,
                passed=True,
                rubric="blueprint_plausibility",
            )
        )
    runner = CliRunner()

    validated = runner.invoke(
        cli,
        [
            "validate-blueprints",
            "--dataset-id",
            "ds-blueprint",
        ],
    )
    materialized = runner.invoke(
        cli,
        [
            "materialize-blueprints",
            "--dataset-id",
            "ds-blueprint",
            "--require-release-ready",
        ],
    )
    exported = runner.invoke(
        cli,
        [
            "export-blueprint-release-summary",
            "--dataset-id",
            "ds-blueprint",
            "--output",
            "blueprint-release-summary.json",
        ],
    )

    summary = json.loads((tmp_path / "blueprint-release-summary.json").read_text())
    records = DatasetStore().list_records(dataset_id="ds-blueprint")
    assert validated.exit_code == 0
    assert materialized.exit_code == 0
    assert "Materialized 1 blueprint record(s)" in materialized.output
    assert exported.exit_code == 0
    assert "Wrote blueprint release summary" in exported.output
    assert len(records) == 1
    assert records[0].metadata["blueprint_id"] == "bp-1"
    assert summary["blueprint_count"] == 1
    assert summary["research_release_ready_count"] == 1
    assert summary["materialized_record_count"] == 1
