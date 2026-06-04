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
        evidence=BlueprintEvidence(
            supported_claims=["AF anticoagulation requires renal-dose review."],
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
