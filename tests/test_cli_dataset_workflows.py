import json
import re

from click.testing import CliRunner

from casecrawler.cli import cli
from casecrawler.integrations.huggingface import import_reference_rows
from casecrawler.models.dataset import GenerationRequest
from casecrawler.models.synthetic import (
    ComplexityProfile,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
    ValidationReport,
)
from casecrawler.storage.dataset_store import DatasetStore


def test_dataset_cli_list_validate_and_export(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    generate = runner.invoke(cli, ["generate-dataset", "sepsis", "--count", "1"])
    assert generate.exit_code == 0
    match = re.search(r"Dataset: (ds-[0-9a-f-]+)", generate.output)
    assert match, f"Failed to find dataset id in output: {generate.output}"
    dataset_id = match.group(1)
    listed = runner.invoke(cli, ["datasets", "list"])
    quality = runner.invoke(cli, ["datasets", "quality", dataset_id])
    validated = runner.invoke(cli, ["validate", "--dataset-id", dataset_id])
    exported = runner.invoke(
        cli,
        [
            "export-dataset",
            "--dataset-id",
            dataset_id,
            "--output",
            "synthetic.jsonl",
            "--format",
            "sft_jsonl",
        ],
    )

    assert listed.exit_code == 0
    assert "sepsis" in listed.output
    assert quality.exit_code == 0
    assert '"export_ready": true' in quality.output
    assert validated.exit_code == 0
    assert "Validated:" in validated.output
    assert exported.exit_code == 0
    assert "Exported" in exported.output
    assert (tmp_path / "synthetic.jsonl").exists()

    fhir_exported = runner.invoke(
        cli,
        [
            "export-dataset",
            "--dataset-id",
            dataset_id,
            "--output",
            "synthetic.fhir.ndjson",
            "--format",
            "fhir_ndjson",
        ],
    )
    assert fhir_exported.exit_code == 0
    assert "Exported" in fhir_exported.output
    assert "Bundle" in (tmp_path / "synthetic.fhir.ndjson").read_text()
    fhir_verified = runner.invoke(cli, ["verify-fhir-export", "synthetic.fhir.ndjson"])
    assert fhir_verified.exit_code == 0
    assert json.loads(fhir_verified.output)["valid"] is True

    note_fact_exported = runner.invoke(
        cli,
        [
            "export-dataset",
            "--dataset-id",
            dataset_id,
            "--output",
            "synthetic.note-facts.jsonl",
            "--format",
            "note_fact_sft_jsonl",
        ],
    )
    note_fact_lines = [
        json.loads(line)
        for line in (tmp_path / "synthetic.note-facts.jsonl").read_text().splitlines()
    ]
    assert note_fact_exported.exit_code == 0
    assert len(note_fact_lines) > 1
    assert {line["task"] for line in note_fact_lines} == {"extract_clinical_facts_from_note"}


def test_dataset_cli_exports_split_fine_tuning_package(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    generate = runner.invoke(cli, ["generate-dataset", "sepsis", "--count", "3"])
    dataset_id = re.search(r"Dataset: (ds-[0-9a-f-]+)", generate.output).group(1)
    exported = runner.invoke(
        cli,
        [
            "export-dataset-splits",
            "--dataset-id",
            dataset_id,
            "--output-dir",
            "split-package",
            "--format",
            "sft_jsonl",
            "--train-ratio",
            "0.34",
            "--validation-ratio",
            "0.33",
            "--test-ratio",
            "0.33",
            "--seed",
            "unit-test",
        ],
    )

    manifest = json.loads((tmp_path / "split-package" / "manifest.json").read_text())
    exports = DatasetStore().list_export_manifests(dataset_id=dataset_id)

    assert exported.exit_code == 0
    assert "Exported split package" in exported.output
    assert manifest["record_count"] == 3
    assert manifest["example_count"] == 3
    assert manifest["splits"]["train"]["record_count"] == 1
    assert manifest["splits"]["validation"]["record_count"] == 1
    assert manifest["splits"]["test"]["record_count"] == 1
    assert set(manifest["audit_artifacts"]) == {
        "benchmark_profile.json",
        "dataset_card.md",
        "model_card.md",
        "quality_report.json",
    }
    assert (tmp_path / "split-package" / "train.jsonl").exists()
    assert (tmp_path / "split-package" / "dataset_card.md").exists()
    assert json.loads((tmp_path / "split-package" / "quality_report.json").read_text())[
        "export_ready"
    ] is True
    benchmark_profile = json.loads(
        (tmp_path / "split-package" / "benchmark_profile.json").read_text()
    )
    assert benchmark_profile["artifact_type"] == "casecrawler_benchmark_profile"
    assert benchmark_profile["profile"]["dataset_id"] == dataset_id
    assert exports[0].metadata["split_package"] is True
    assert exports[0].metadata["seed"] == "unit-test"
    assert exports[0].metadata["multimodal_release_ready"] is False
    assert "benchmark_reference" in exports[0].metadata["multimodal_release_missing"]
    assert exports[0].metadata["core_artifact_coverage"]["records"] is True
    assert "benchmark_profile.json" in exports[0].metadata["audit_artifacts"]
    verified = runner.invoke(cli, ["verify-split-package", "split-package"])
    release_verified = runner.invoke(
        cli,
        ["verify-split-package", "--require-multimodal-release", "split-package"],
    )
    assert verified.exit_code == 0
    verify_report = json.loads(verified.output)
    assert verify_report["valid"] is True
    assert verify_report["quality_report"]["multimodal_release_ready"] is False
    assert verify_report["splits"]["train"]["example_count"] == 1
    assert release_verified.exit_code != 0
    assert "not multimodal-release-ready" in release_verified.output


def test_dataset_cli_blocks_profile_specific_export_when_artifacts_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    generate = runner.invoke(
        cli,
        [
            "generate-dataset",
            "sepsis",
            "--count",
            "1",
            "--modalities",
            "clinical_text",
        ],
    )
    dataset_id = re.search(r"Dataset: (ds-[0-9a-f-]+)", generate.output).group(1)
    blocked = runner.invoke(
        cli,
        [
            "export-dataset",
            "--dataset-id",
            dataset_id,
            "--format",
            "medication_reconciliation_jsonl",
            "--output",
            "meds.jsonl",
        ],
    )
    allowed = runner.invoke(
        cli,
        [
            "export-dataset",
            "--dataset-id",
            dataset_id,
            "--format",
            "medication_reconciliation_jsonl",
            "--output",
            "meds.jsonl",
            "--allow-blocked",
        ],
    )

    assert blocked.exit_code != 0
    assert "Export profile medication_reconciliation_jsonl is not ready" in blocked.output
    assert "medications" in blocked.output
    assert allowed.exit_code == 0


def test_dataset_cli_blocks_profile_specific_split_export_when_artifacts_missing(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    generate = runner.invoke(
        cli,
        [
            "generate-dataset",
            "sepsis",
            "--count",
            "1",
            "--modalities",
            "clinical_text",
        ],
    )
    dataset_id = re.search(r"Dataset: (ds-[0-9a-f-]+)", generate.output).group(1)
    blocked = runner.invoke(
        cli,
        [
            "export-dataset-splits",
            "--dataset-id",
            dataset_id,
            "--format",
            "clinical_observation_jsonl",
            "--output-dir",
            "obs-package",
        ],
    )

    assert blocked.exit_code != 0
    assert "Export profile clinical_observation_jsonl is not ready" in blocked.output
    assert "labs_or_vitals" in blocked.output


def test_dataset_cli_can_require_multimodal_release_for_split_export(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    generate = runner.invoke(cli, ["generate-dataset", "sepsis", "--count", "1"])
    dataset_id = re.search(r"Dataset: (ds-[0-9a-f-]+)", generate.output).group(1)

    blocked = runner.invoke(
        cli,
        [
            "export-dataset-splits",
            "--dataset-id",
            dataset_id,
            "--output-dir",
            "release-package",
            "--require-multimodal-release",
        ],
    )

    assert blocked.exit_code != 0
    assert "not ready for multimodal release package export" in blocked.output
    assert "benchmark_reference" in blocked.output


def test_dataset_cli_generates_release_package_with_fixture_references(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    generated = runner.invoke(
        cli,
        [
            "generate-release-package",
            "sepsis",
            "--count",
            "1",
            "--output-dir",
            "release-package",
            "--seed",
            "unit-test",
        ],
    )

    assert generated.exit_code == 0, generated.output
    body = json.loads(generated.output)
    manifest = json.loads((tmp_path / "release-package" / "manifest.json").read_text())
    quality = json.loads((tmp_path / "release-package" / "quality_report.json").read_text())
    benchmark = json.loads(
        (tmp_path / "release-package" / "benchmark_report.json").read_text()
    )
    benchmark_suite = json.loads(
        (tmp_path / "release-package" / "benchmark_suite_report.json").read_text()
    )
    release_verified = runner.invoke(
        cli,
        ["verify-split-package", "--require-multimodal-release", "release-package"],
    )
    exports = DatasetStore().list_export_manifests(dataset_id=body["dataset_id"])

    assert body["verification"]["valid"] is True
    assert body["quality_report"]["multimodal_release_ready"] is True
    assert body["benchmark"]["passed"] is True
    assert body["seeded_references"]["imported"]
    assert body["manifest"]["export_format"] == "multimodal_jsonl"
    assert manifest["record_count"] == 1
    assert set(manifest["audit_artifacts"]) == {
        "benchmark_profile.json",
        "benchmark_report.json",
        "benchmark_suite_report.json",
        "dataset_card.md",
        "model_card.md",
        "quality_report.json",
    }
    assert quality["multimodal_release_ready"] is True
    assert benchmark["passed"] is True
    assert benchmark_suite["passed"] is True
    assert benchmark_suite["reference_count"] >= 1
    assert body["benchmark_suite"]["passed"] is True
    assert body["benchmark_suite"]["reference_count"] == benchmark_suite["reference_count"]
    assert release_verified.exit_code == 0, release_verified.output
    assert exports[0].metadata["release_package"] is True
    assert exports[0].metadata["multimodal_release_ready"] is True
    assert exports[0].metadata["benchmark_passed"] is True
    assert exports[0].metadata["benchmark_suite_passed"] is True


def test_dataset_cli_split_export_can_require_benchmark_gate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    generated = runner.invoke(cli, ["generate-dataset", "sepsis", "--count", "3"])
    reference = runner.invoke(cli, ["generate-dataset", "sepsis", "--count", "3"])
    dataset_id = re.search(r"Dataset: (ds-[0-9a-f-]+)", generated.output).group(1)
    reference_dataset_id = re.search(r"Dataset: (ds-[0-9a-f-]+)", reference.output).group(1)

    exported = runner.invoke(
        cli,
        [
            "export-dataset-splits",
            "--dataset-id",
            dataset_id,
            "--reference-dataset-id",
            reference_dataset_id,
            "--min-overall-score",
            "0",
            "--min-metric-score",
            "0",
            "--output-dir",
            "benchmark-split-package",
        ],
    )

    manifest = json.loads((tmp_path / "benchmark-split-package" / "manifest.json").read_text())
    exports = DatasetStore().list_export_manifests(dataset_id=dataset_id)

    assert exported.exit_code == 0
    assert "benchmark_report.json" in manifest["audit_artifacts"]
    assert (tmp_path / "benchmark-split-package" / "benchmark_report.json").exists()
    assert exports[0].metadata["benchmark_reference_dataset_id"] == reference_dataset_id
    assert exports[0].metadata["benchmark_passed"] is True


def test_dataset_cli_blocks_export_until_required_human_review(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    generate = runner.invoke(
        cli,
        ["generate-dataset", "sepsis", "--count", "1", "--require-human-review"],
    )
    assert generate.exit_code == 0
    match = re.search(r"Dataset: (ds-[0-9a-f-]+)", generate.output)
    assert match, f"Failed to find dataset id in output: {generate.output}"
    dataset_id = match.group(1)
    record_id = DatasetStore().list_records(dataset_id=dataset_id)[0].record_id

    blocked = runner.invoke(
        cli,
        [
            "export-dataset",
            "--dataset-id",
            dataset_id,
            "--output",
            "synthetic.jsonl",
            "--format",
            "sft_jsonl",
        ],
    )
    marked = runner.invoke(
        cli,
        [
            "reviews",
            "mark",
            record_id,
            "--status",
            "approved",
            "--reviewer",
            "clinical-reviewer",
        ],
    )
    exported = runner.invoke(
        cli,
        [
            "export-dataset",
            "--dataset-id",
            dataset_id,
            "--output",
            "synthetic.jsonl",
            "--format",
            "sft_jsonl",
        ],
    )

    assert blocked.exit_code != 0
    assert "human_review.missing" in blocked.output
    assert marked.exit_code == 0
    assert exported.exit_code == 0
    assert (tmp_path / "synthetic.jsonl").exists()


def test_dataset_cli_lists_generation_recipes():
    result = CliRunner().invoke(cli, ["generation-recipes"])

    assert result.exit_code == 0
    assert "full_multimodal_acute_care" in result.output
    assert "icu_timeseries_notes" in result.output
    assert "references=" in result.output


def test_dataset_cli_lists_imaging_model_use_policies():
    result = CliRunner().invoke(cli, ["imaging-models"])

    assert result.exit_code == 0
    assert "medisyn" in result.output
    assert "use_policy=non_commercial_no_derivatives_review_before_release" in result.output
    assert "roentgen_v2_gated" in result.output
    assert "gated=True" in result.output


def test_dataset_cli_reports_unknown_generation_recipe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        cli,
        ["generate-dataset", "sepsis", "--recipe", "missing"],
    )

    assert result.exit_code != 0
    assert "Unknown generation recipe" in result.output


def test_dataset_cli_export_can_require_benchmark_gate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    generated = runner.invoke(cli, ["generate-dataset", "sepsis", "--count", "1"])
    reference = runner.invoke(cli, ["generate-dataset", "sepsis", "--count", "1"])
    dataset_id = re.search(r"Dataset: (ds-[0-9a-f-]+)", generated.output).group(1)
    reference_dataset_id = re.search(r"Dataset: (ds-[0-9a-f-]+)", reference.output).group(1)

    exported = runner.invoke(
        cli,
        [
            "export-dataset",
            "--dataset-id",
            dataset_id,
            "--reference-dataset-id",
            reference_dataset_id,
            "--min-overall-score",
            "0",
            "--min-metric-score",
            "0",
            "--output",
            "benchmark-gated.jsonl",
        ],
    )

    assert exported.exit_code == 0
    assert "Exported" in exported.output
    assert (tmp_path / "benchmark-gated.jsonl").exists()


def test_dataset_cli_export_can_auto_select_recipe_benchmark_reference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    generated = runner.invoke(
        cli,
        ["generate-dataset", "sepsis", "--count", "1", "--recipe", "icu_timeseries_notes"],
    )
    reference = runner.invoke(cli, ["generate-dataset", "sepsis", "--count", "1"])
    dataset_id = re.search(r"Dataset: (ds-[0-9a-f-]+)", generated.output).group(1)
    reference_dataset_id = re.search(r"Dataset: (ds-[0-9a-f-]+)", reference.output).group(1)
    store = DatasetStore()
    for record in store.list_records(dataset_id=reference_dataset_id):
        store.save_record(
            record.model_copy(
                update={
                    "metadata": {
                        **record.metadata,
                        "reference_key": "synthclinicalnotes",
                        "reference_dataset": "IntelLabs/SynthClinicalNotes",
                    }
                }
            )
        )

    exported = runner.invoke(
        cli,
        [
            "export-dataset",
            "--dataset-id",
            dataset_id,
            "--auto-benchmark",
            "--allow-blocked",
            "--output",
            "auto-benchmark.jsonl",
        ],
    )

    assert exported.exit_code == 0
    assert "Exported" in exported.output
    manifest = store.list_export_manifests(dataset_id=dataset_id)[0]
    assert manifest.metadata["benchmark_reference_dataset_id"] == reference_dataset_id
    assert manifest.metadata["benchmark_reference_key"] == "synthclinicalnotes"
    assert manifest.metadata["benchmark_auto_selected"] is True


def test_dataset_cli_reports_recipe_benchmark_plan_readiness(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    generated = runner.invoke(
        cli,
        ["generate-dataset", "sepsis", "--count", "1", "--recipe", "icu_timeseries_notes"],
    )
    reference = runner.invoke(cli, ["generate-dataset", "sepsis", "--count", "1"])
    dataset_id = re.search(r"Dataset: (ds-[0-9a-f-]+)", generated.output).group(1)
    reference_dataset_id = re.search(r"Dataset: (ds-[0-9a-f-]+)", reference.output).group(1)
    store = DatasetStore()
    for record in store.list_records(dataset_id=reference_dataset_id):
        store.save_record(
            record.model_copy(
                update={
                    "metadata": {
                        **record.metadata,
                        "reference_key": "synthclinicalnotes",
                        "reference_dataset": "IntelLabs/SynthClinicalNotes",
                    }
                }
            )
        )

    result = runner.invoke(cli, ["datasets", "benchmark-plan", dataset_id])

    assert result.exit_code == 0
    body = json.loads(result.output)
    assert body["dataset_id"] == dataset_id
    assert body["resolved_reference_dataset_id"] == reference_dataset_id
    assert body["resolved_reference_key"] == "synthclinicalnotes"
    assert body["ready"] is True
    assert body["task_export_reference_readiness"]["note_fact_sft_jsonl"][
        "available_reference_keys"
    ] == ["synthclinicalnotes"]


def test_dataset_cli_seeds_recipe_reference_fixtures(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    generated = runner.invoke(
        cli,
        ["generate-dataset", "sepsis", "--count", "1", "--recipe", "icu_timeseries_notes"],
    )
    dataset_id = re.search(r"Dataset: (ds-[0-9a-f-]+)", generated.output).group(1)

    seeded = runner.invoke(
        cli,
        [
            "datasets",
            "seed-reference-fixtures",
            dataset_id,
            "--dataset-id-prefix",
            "fixture-ref",
        ],
    )
    plan = runner.invoke(cli, ["datasets", "benchmark-plan", dataset_id])

    assert seeded.exit_code == 0
    body = json.loads(seeded.output)
    assert body["unavailable"] == []
    assert {item["reference_key"] for item in body["imported"]} == {
        "synthea_fhir",
        "synthclinicalnotes",
        "augmented_clinical_notes",
        "medsynth_dialogue_note",
        "clinical_notes_to_fhir",
        "technetium_i",
    }
    plan_body = json.loads(plan.output)
    assert plan_body["ready"] is True
    assert plan_body["missing_reference_keys"] == []


def test_dataset_cli_runs_recipe_benchmark_suite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    generated = runner.invoke(
        cli,
        ["generate-dataset", "sepsis", "--count", "1", "--recipe", "icu_timeseries_notes"],
    )
    first_reference = runner.invoke(cli, ["generate-dataset", "sepsis", "--count", "1"])
    second_reference = runner.invoke(cli, ["generate-dataset", "sepsis", "--count", "1"])
    dataset_id = re.search(r"Dataset: (ds-[0-9a-f-]+)", generated.output).group(1)
    first_reference_id = re.search(r"Dataset: (ds-[0-9a-f-]+)", first_reference.output).group(1)
    second_reference_id = re.search(r"Dataset: (ds-[0-9a-f-]+)", second_reference.output).group(1)
    store = DatasetStore()
    for dataset_id_to_mark, reference_key in [
        (first_reference_id, "synthclinicalnotes"),
        (second_reference_id, "clinical_notes_to_fhir"),
    ]:
        for record in store.list_records(dataset_id=dataset_id_to_mark):
            store.save_record(
                record.model_copy(
                    update={
                        "metadata": {
                            **record.metadata,
                            "reference_key": reference_key,
                            "reference_dataset": reference_key,
                        }
                    }
                )
            )

    result = runner.invoke(cli, ["datasets", "benchmark-suite", dataset_id])

    assert result.exit_code == 0
    body = json.loads(result.output)
    assert body["dataset_id"] == dataset_id
    assert body["reference_count"] == 2
    assert {
        item["reference_key"] for item in body["results"]
    } == {"synthclinicalnotes", "clinical_notes_to_fhir"}
    assert body["task_export_results"]["medication_reconciliation_jsonl"][
        "missing_reference_keys"
    ] == ["synthea_fhir", "medsynth_dialogue_note"]


def test_dataset_cli_export_blocks_failed_benchmark_gate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    generated = runner.invoke(cli, ["generate-dataset", "sepsis", "--count", "1"])
    reference = runner.invoke(cli, ["generate-dataset", "heart failure", "--count", "1"])
    dataset_id = re.search(r"Dataset: (ds-[0-9a-f-]+)", generated.output).group(1)
    reference_dataset_id = re.search(r"Dataset: (ds-[0-9a-f-]+)", reference.output).group(1)

    blocked = runner.invoke(
        cli,
        [
            "export-dataset",
            "--dataset-id",
            dataset_id,
            "--reference-dataset-id",
            reference_dataset_id,
            "--min-overall-score",
            "1",
            "--min-metric-score",
            "1",
            "--output",
            "blocked.jsonl",
        ],
    )
    allowed = runner.invoke(
        cli,
        [
            "export-dataset",
            "--dataset-id",
            dataset_id,
            "--reference-dataset-id",
            reference_dataset_id,
            "--min-overall-score",
            "1",
            "--min-metric-score",
            "1",
            "--allow-blocked",
            "--output",
            "allowed.jsonl",
        ],
    )

    assert blocked.exit_code != 0
    assert "failed benchmark gate" in blocked.output
    assert allowed.exit_code == 0
    assert (tmp_path / "allowed.jsonl").exists()


def test_dataset_cli_passes_imaging_model_options(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    captured = []

    class FakePipeline:
        async def generate(self, req: GenerationRequest):
            captured.append(req)
            return {
                "dataset_id": "ds-test",
                "generated": 0,
                "approved": 0,
                "records": [],
            }

    monkeypatch.setattr("casecrawler.generation.synthetic_pipeline.SyntheticPipeline", FakePipeline)
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "generate-dataset",
            "pneumonia",
            "--modalities",
            "imaging",
            "--imaging-backend",
            "diffusers",
            "--imaging-model-profile",
            "cxr_pneumonia_dreambooth",
            "--diffusers-model-id",
            "hf/test-cxr",
        ],
    )

    assert result.exit_code == 0
    assert captured[0].modalities == [Modality.IMAGING]
    assert captured[0].imaging_backend == "diffusers"
    assert captured[0].imaging_model_profile == "cxr_pneumonia_dreambooth"
    assert captured[0].diffusers_model_id == "hf/test-cxr"


def test_dataset_cli_passes_time_series_model_options(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    captured = []

    class FakePipeline:
        async def generate(self, req: GenerationRequest):
            captured.append(req)
            return {
                "dataset_id": "ds-test",
                "generated": 0,
                "approved": 0,
                "records": [],
            }

    monkeypatch.setattr("casecrawler.generation.synthetic_pipeline.SyntheticPipeline", FakePipeline)
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "generate-dataset",
            "sepsis",
            "--modalities",
            "time_series",
            "--time-series-backend",
            "external",
            "--time-series-model-profile",
            "timediff",
            "--time-series-command",
            "timediff-sample,--checkpoint,local.pt",
        ],
    )

    assert result.exit_code == 0
    assert captured[0].modalities == [Modality.TIME_SERIES]
    assert captured[0].time_series_backend == "external"
    assert captured[0].time_series_model_profile == "timediff"
    assert captured[0].time_series_command == [
        "timediff-sample",
        "--checkpoint",
        "local.pt",
    ]


def test_dataset_cli_passes_clinical_text_model_options(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    captured = []

    class FakePipeline:
        async def generate(self, req: GenerationRequest):
            captured.append(req)
            return {
                "dataset_id": "ds-test",
                "generated": 0,
                "approved": 0,
                "records": [],
            }

    monkeypatch.setattr("casecrawler.generation.synthetic_pipeline.SyntheticPipeline", FakePipeline)
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "generate-dataset",
            "sepsis",
            "--modalities",
            "clinical_text",
            "--clinical-text-backend",
            "llm",
            "--llm-provider",
            "ollama",
            "--llm-model",
            "medgemma-local",
            "--ollama-base-url",
            "http://localhost:11434",
        ],
    )

    assert result.exit_code == 0
    assert captured[0].modalities == [Modality.CLINICAL_TEXT]
    assert captured[0].clinical_text_backend == "llm"
    assert captured[0].llm_provider == "ollama"
    assert captured[0].llm_model == "medgemma-local"
    assert captured[0].ollama_base_url == "http://localhost:11434"


def test_dataset_cli_export_blocks_unready_dataset_without_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = DatasetStore()
    store.save_record(
        SyntheticRecord(
            record_id="rec-blocked",
            dataset_id="ds-blocked",
            topic="sepsis",
            complexity=ComplexityProfile.MODERATE,
            modalities=[Modality.CLINICAL_TEXT],
            patient=SyntheticPatient(patient_id="pat-1", age=64, sex="male"),
            encounters=[],
            provenance=Provenance(generator="unit-test", created_at="2026-01-01T00:00:00"),
            validation=ValidationReport(
                schema_score=1.0,
                clinical_consistency_score=1.0,
                privacy_score=1.0,
                utility_score=1.0,
                approved=True,
            ),
        )
    )
    runner = CliRunner()

    blocked = runner.invoke(
        cli,
        [
            "export-dataset",
            "--dataset-id",
            "ds-blocked",
            "--output",
            "blocked.jsonl",
        ],
    )
    allowed = runner.invoke(
        cli,
        [
            "export-dataset",
            "--dataset-id",
            "ds-blocked",
            "--output",
            "blocked.jsonl",
            "--allow-blocked",
        ],
    )

    assert blocked.exit_code != 0
    assert "not ready for fine-tuning export" in blocked.output
    assert "clinical_text.missing_artifacts" in blocked.output
    assert allowed.exit_code == 0
    assert (tmp_path / "blocked.jsonl").exists()


def test_dataset_cli_generates_modalities_and_cohort_constraints(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    generate = runner.invoke(
        cli,
        [
            "generate-dataset",
            "pulmonary embolism",
            "--count",
            "2",
            "--modalities",
            "structured_ehr,imaging,time_series",
            "--age-min",
            "50",
            "--age-max",
            "51",
            "--sexes",
            "female,male",
            "--topic-mix",
            "pulmonary embolism:2,sepsis:1",
            "--base-time",
            "2026-02-03T04:05:06",
            "--encounter-count",
            "2",
        ],
    )
    match = re.search(r"Dataset: (ds-[0-9a-f-]+)", generate.output)
    assert generate.exit_code == 0
    assert match, f"Failed to find dataset id in output: {generate.output}"
    records = DatasetStore().list_records(dataset_id=match.group(1), limit=10)

    assert sorted(record.patient.age for record in records) == [50, 51]
    assert sorted(record.patient.sex for record in records) == ["female", "male"]
    assert sorted(record.topic for record in records) == [
        "pulmonary embolism",
        "pulmonary embolism",
    ]
    assert records[0].metadata["cohort_constraints"]["topic_mix"] == [
        "pulmonary embolism:2",
        "sepsis:1",
    ]
    assert records[0].metadata["cohort_constraints"]["encounter_count"] == 2
    assert len(records[0].encounters) == 2
    assert Modality.IMAGING in records[0].modalities
    assert {record.imaging[0].modality for record in records} == {"CTA"}
    assert records[0].time_series


def test_dataset_cli_accepts_time_series_export_profile(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    generated = runner.invoke(cli, ["generate-dataset", "sepsis", "--count", "1"])
    assert generated.exit_code == 0
    match = re.search(r"Dataset: (ds-[0-9a-f-]+)", generated.output)
    assert match, f"Failed to find dataset id in output: {generated.output}"

    result = runner.invoke(
        cli,
        [
            "export-dataset",
            "--dataset-id",
            match.group(1),
            "--format",
            "time_series_jsonl",
            "--output",
            "time-series.jsonl",
            "--allow-blocked",
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "time-series.jsonl").exists()


def test_dataset_cli_benchmark_against_reference_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    generated = runner.invoke(cli, ["generate-dataset", "sepsis", "--count", "1"])
    assert generated.exit_code == 0
    match = re.search(r"Dataset: (ds-[0-9a-f-]+)", generated.output)
    assert match, f"Failed to find dataset id in output: {generated.output}"
    dataset_id = match.group(1)
    store = DatasetStore()
    for record in import_reference_rows(
        [
            {
                "patient_id": "ref-1",
                "note": "Progress Note: 60-year-old male with sepsis.",
                "question": "Summarize.",
                "answer": "Sepsis.",
                "task": "Summarization",
            }
        ],
        dataset_id="ds-reference",
    ):
        store.save_record(record)

    result = runner.invoke(
        cli,
        [
            "benchmark-dataset",
            "--dataset-id",
            dataset_id,
            "--reference-dataset-id",
            "ds-reference",
            "--min-overall-score",
            "0.2",
            "--min-metric-score",
            "0",
            "--output",
            "benchmark.json",
        ],
    )

    assert result.exit_code == 0
    assert "Overall score:" in result.output
    assert "Passed:" in result.output
    assert (tmp_path / "benchmark.json").exists()
    report = json.loads((tmp_path / "benchmark.json").read_text())
    assert report["thresholds"] == {"min_overall_score": 0.2, "min_metric_score": 0.0}


def test_dataset_cli_exports_and_compares_benchmark_profiles(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    generated = runner.invoke(cli, ["generate-dataset", "sepsis", "--count", "2"])
    reference = runner.invoke(cli, ["generate-dataset", "sepsis", "--count", "2"])
    assert generated.exit_code == 0
    assert reference.exit_code == 0
    dataset_id = re.search(r"Dataset: (ds-[0-9a-f-]+)", generated.output).group(1)
    reference_id = re.search(r"Dataset: (ds-[0-9a-f-]+)", reference.output).group(1)

    generated_profile = runner.invoke(
        cli,
        [
            "export-benchmark-profile",
            "--dataset-id",
            dataset_id,
            "--output",
            "generated-profile.json",
        ],
    )
    reference_profile = runner.invoke(
        cli,
        [
            "export-benchmark-profile",
            "--dataset-id",
            reference_id,
            "--output",
            "reference-profile.json",
        ],
    )
    compared = runner.invoke(
        cli,
        [
            "benchmark-profile",
            "--profile",
            "generated-profile.json",
            "--reference-profile",
            "reference-profile.json",
            "--min-overall-score",
            "0.2",
            "--min-metric-score",
            "0",
            "--output",
            "profile-benchmark.json",
        ],
    )

    assert generated_profile.exit_code == 0
    assert reference_profile.exit_code == 0
    assert compared.exit_code == 0
    profile_payload = json.loads((tmp_path / "generated-profile.json").read_text())
    assert profile_payload["artifact_type"] == "casecrawler_benchmark_profile"
    report = json.loads((tmp_path / "profile-benchmark.json").read_text())
    assert report["generated_dataset_id"] == dataset_id
    assert report["reference_dataset_id"] == reference_id


def test_dataset_cli_benchmark_reports_failing_gate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    generated = runner.invoke(cli, ["generate-dataset", "sepsis", "--count", "1"])
    reference = runner.invoke(cli, ["generate-dataset", "heart failure", "--count", "1"])
    assert generated.exit_code == 0
    assert reference.exit_code == 0
    dataset_id = re.search(r"Dataset: (ds-[0-9a-f-]+)", generated.output).group(1)
    reference_dataset_id = re.search(r"Dataset: (ds-[0-9a-f-]+)", reference.output).group(1)

    result = runner.invoke(
        cli,
        [
            "benchmark-dataset",
            "--dataset-id",
            dataset_id,
            "--reference-dataset-id",
            reference_dataset_id,
            "--min-overall-score",
            "1",
            "--min-metric-score",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "Passed: false" in result.output
    assert "Failing metrics:" in result.output


def test_dataset_cli_benchmark_rejects_invalid_threshold(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "benchmark-dataset",
            "--dataset-id",
            "ds-one",
            "--reference-dataset-id",
            "ds-reference",
            "--min-overall-score",
            "1.1",
        ],
    )

    assert result.exit_code != 0
    assert "Invalid value for '--min-overall-score'" in result.output


def test_dataset_cli_imports_hf_reference_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    def fake_load_reference_dataset(key, *, split=None, streaming=True):
        assert key == "asclepius"
        assert split == "validation"
        assert streaming is True
        return [
            {
                "patient_id": "ref-1",
                "note": "Progress Note: 60-year-old male with sepsis.",
                "question": "Summarize.",
                "answer": "Sepsis.",
                "task": "Summarization",
            }
        ]

    monkeypatch.setattr(
        "casecrawler.integrations.huggingface.load_reference_dataset",
        fake_load_reference_dataset,
    )

    listed = runner.invoke(cli, ["reference-datasets"])
    imported = runner.invoke(
        cli,
        [
            "import-reference-dataset",
            "asclepius",
            "--dataset-id",
            "ds-hf-reference",
            "--split",
            "validation",
            "--limit",
            "1",
        ],
    )
    store = DatasetStore()

    assert listed.exit_code == 0
    assert "synthea_fhir" in listed.output
    assert "import-synthea-fhir" in listed.output
    assert "asclepius" in listed.output
    assert "augmented_clinical_notes" in listed.output
    assert "clinical_notes_to_fhir" in listed.output
    assert "radiology_report_consistency" in listed.output
    assert imported.exit_code == 0
    assert "Imported 1 reference record(s)" in imported.output
    assert store.dataset_exists("ds-hf-reference")
    assert store.get_manifest("ds-hf-reference").metadata["record_ids"]


def test_dataset_cli_imports_bundled_reference_fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    listed = runner.invoke(cli, ["reference-datasets"])
    imported = runner.invoke(
        cli,
        [
            "import-reference-fixture",
            "clinical_notes_to_fhir",
            "--dataset-id",
            "ds-fixture-reference",
        ],
    )
    store = DatasetStore()
    manifest = store.get_manifest("ds-fixture-reference")
    record = store.list_records(dataset_id="ds-fixture-reference")[0]

    assert listed.exit_code == 0
    assert "bundled_fixtures" in listed.output
    assert imported.exit_code == 0
    assert "Imported 1 bundled reference fixture record" in imported.output
    assert manifest.metadata["primary_reference_key"] == "clinical_notes_to_fhir"
    assert record.labs[0].name == "Lactate"
    assert record.medication_history[0].name == "Ceftriaxone"


def test_dataset_cli_imports_synthea_fhir_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    bundle_dir = tmp_path / "synthea"
    bundle_dir.mkdir()
    bundle = {
        "resourceType": "Bundle",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "pat-1",
                    "gender": "female",
                    "birthDate": "1970-01-01",
                }
            },
            {
                "resource": {
                    "resourceType": "Encounter",
                    "id": "enc-1",
                    "period": {"start": "2026-01-01T00:00:00"},
                    "reasonCode": [{"text": "sepsis"}],
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"text": "Lactate"},
                    "valueQuantity": {"value": 3.4, "unit": "mmol/L"},
                    "effectiveDateTime": "2026-01-01T01:00:00",
                }
            },
        ],
    }
    (bundle_dir / "patient.json").write_text(json.dumps(bundle))

    result = runner.invoke(
        cli,
        [
            "import-synthea-fhir",
            str(bundle_dir),
            "--dataset-id",
            "ds-synthea",
        ],
    )

    assert result.exit_code == 0
    assert "Imported 1 Synthea FHIR record(s) into ds-synthea" in result.output
    record = DatasetStore().list_records(dataset_id="ds-synthea")[0]
    assert record.patient.patient_id == "pat-1"
    assert record.labs[0].name == "Lactate"


def test_dataset_cli_runs_synthea_and_imports_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    output_dir = tmp_path / "synthea-output"
    output_dir.mkdir()

    def fake_run_and_import(self, *, executable, output_dir, dataset_id, population):
        assert executable == "/opt/synthea/run_synthea"
        assert population == 2
        return [
            SyntheticRecord(
                record_id="synthea-pat-1",
                dataset_id=dataset_id,
                topic="synthea import",
                complexity=ComplexityProfile.MODERATE,
                modalities=[Modality.STRUCTURED_EHR],
                patient=SyntheticPatient(patient_id="pat-1", age=50, sex="female"),
                encounters=[],
                provenance=Provenance(
                    generator="synthea-fhir-import",
                    created_at="2026-01-01T00:00:00",
                ),
            )
        ]

    monkeypatch.setattr(
        "casecrawler.integrations.synthea.SyntheaAdapter.run_and_import",
        fake_run_and_import,
    )

    result = runner.invoke(
        cli,
        [
            "run-synthea",
            "--dataset-id",
            "ds-synthea",
            "--output-dir",
            str(output_dir),
            "--population",
            "2",
            "--synthea-executable",
            "/opt/synthea/run_synthea",
        ],
    )

    assert result.exit_code == 0
    assert "Ran Synthea and imported 1 record(s) into ds-synthea" in result.output
    assert DatasetStore().get_record("synthea-pat-1").patient.patient_id == "pat-1"


def test_dataset_cli_run_synthea_requires_executable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "run-synthea",
            "--dataset-id",
            "ds-synthea",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert "Provide --synthea-executable" in result.output


def test_dataset_cli_imports_custom_hf_reference_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    def fake_load_huggingface_dataset(repo_id, *, split, streaming=True):
        assert repo_id == "org/custom-synthetic-notes"
        assert split == "eval"
        assert streaming is True
        return [
            {
                "subject_id": "ref-1",
                "clinical_note": "Progress Note: 57-year-old female with COPD.",
                "prompt": "Extract diagnosis.",
                "completion": "COPD.",
                "task_name": "extraction",
                "labs": [
                    {
                        "name": "PaCO2",
                        "value": 51,
                        "unit": "mmHg",
                        "flag": "H",
                        "effective_time": "2026-01-01T00:00:00",
                    }
                ],
                "vitals": [
                    {
                        "name": "SpO2",
                        "value": 91,
                        "unit": "%",
                        "effective_time": "2026-01-01T00:05:00",
                    }
                ],
                "medications": [{"name": "Albuterol", "route": "inhaled"}],
                "signals": [
                    {
                        "name": "respiratory_rate",
                        "unit": "/min",
                        "points": [
                            {
                                "timestamp": "2026-01-01T00:05:00",
                                "values": {"value": 24},
                            }
                        ],
                    }
                ],
            }
        ]

    monkeypatch.setattr(
        "casecrawler.integrations.huggingface.load_huggingface_dataset",
        fake_load_huggingface_dataset,
    )

    result = runner.invoke(
        cli,
        [
            "import-reference-dataset",
            "--repo-id",
            "org/custom-synthetic-notes",
            "--dataset-id",
            "ds-custom-reference",
            "--split",
            "eval",
            "--license",
            "cc-by-4.0",
            "--note-field",
            "clinical_note",
            "--question-field",
            "prompt",
            "--answer-field",
            "completion",
            "--task-field",
            "task_name",
            "--patient-id-field",
            "subject_id",
            "--lab-values-field",
            "labs",
            "--vital-values-field",
            "vitals",
            "--medications-field",
            "medications",
            "--time-series-field",
            "signals",
            "--limit",
            "1",
        ],
    )
    store = DatasetStore()

    assert result.exit_code == 0
    assert "Imported 1 reference record(s) from org/custom-synthetic-notes" in result.output
    record = store.list_records(dataset_id="ds-custom-reference")[0]
    assert record.metadata["reference_key"] == "org/custom-synthetic-notes"
    assert record.metadata["reference_dataset"] == "org/custom-synthetic-notes"
    assert record.documents[0].extracted_facts["instruction"] == "Extract diagnosis."
    assert record.labs[0].name == "PaCO2"
    assert record.vitals[0].name == "SpO2"
    assert record.medication_history[0].name == "Albuterol"
    assert record.time_series[0].name == "respiratory_rate"


def test_dataset_cli_benchmark_reports_missing_reference_cleanly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    generated = runner.invoke(cli, ["generate-dataset", "sepsis", "--count", "1"])
    assert generated.exit_code == 0
    match = re.search(r"Dataset: (ds-[0-9a-f-]+)", generated.output)
    assert match, f"Failed to find dataset id in output: {generated.output}"

    result = runner.invoke(
        cli,
        [
            "benchmark-dataset",
            "--dataset-id",
            match.group(1),
            "--reference-dataset-id",
            "ds-missing",
        ],
    )

    assert result.exit_code != 0
    assert "Reference dataset ds-missing not found." in result.output


def test_dataset_cli_benchmark_reports_output_write_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    generated = runner.invoke(cli, ["generate-dataset", "sepsis", "--count", "1"])
    assert generated.exit_code == 0
    match = re.search(r"Dataset: (ds-[0-9a-f-]+)", generated.output)
    assert match, f"Failed to find dataset id in output: {generated.output}"
    dataset_id = match.group(1)
    store = DatasetStore()
    for record in import_reference_rows(
        [
            {
                "patient_id": "ref-1",
                "note": "Progress Note: 60-year-old male with sepsis.",
                "question": "Summarize.",
                "answer": "Sepsis.",
                "task": "Summarization",
            }
        ],
        dataset_id="ds-reference",
    ):
        store.save_record(record)

    result = runner.invoke(
        cli,
        [
            "benchmark-dataset",
            "--dataset-id",
            dataset_id,
            "--reference-dataset-id",
            "ds-reference",
            "--output",
            "missing-dir/benchmark.json",
        ],
    )

    assert result.exit_code != 0
    assert "Failed to write benchmark report" in result.output


def test_dataset_cli_review_queue_and_mark(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = DatasetStore()
    store.save_record(
        SyntheticRecord(
            record_id="rec-review",
            dataset_id="ds-review",
            topic="sepsis",
            complexity=ComplexityProfile.MODERATE,
            modalities=[Modality.CLINICAL_TEXT],
            patient=SyntheticPatient(patient_id="pat-1", age=64, sex="male"),
            encounters=[],
            provenance=Provenance(
                generator="unit-test",
                created_at="2026-05-06T10:00:00",
            ),
        )
    )
    runner = CliRunner()

    queue = runner.invoke(cli, ["reviews", "queue", "--dataset-id", "ds-review"])
    marked = runner.invoke(
        cli,
        [
            "reviews",
            "mark",
            "rec-review",
            "--status",
            "approved",
            "--reviewer",
            "clinical-reviewer",
            "--note",
            "Approved after manual chart review.",
        ],
    )
    queue_after = runner.invoke(cli, ["reviews", "queue", "--dataset-id", "ds-review"])

    assert queue.exit_code == 0
    assert "rec-review" in queue.output
    assert marked.exit_code == 0
    assert "effective_approved=True" in marked.output
    assert queue_after.exit_code == 0
    assert "No records need human review." in queue_after.output


def test_dataset_cli_generates_dataset_and_model_cards(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    generated = runner.invoke(cli, ["generate-dataset", "sepsis", "--count", "1"])
    assert generated.exit_code == 0
    match = re.search(r"Dataset: (ds-[0-9a-f-]+)", generated.output)
    assert match, f"Failed to find dataset id in output: {generated.output}"
    dataset_id = match.group(1)

    dataset_card = runner.invoke(
        cli,
        [
            "document-dataset",
            "--dataset-id",
            dataset_id,
            "--output",
            "DATASET_CARD.md",
        ],
    )
    model_card = runner.invoke(
        cli,
        [
            "document-dataset",
            "--dataset-id",
            dataset_id,
            "--output",
            "MODEL_CARD.md",
            "--kind",
            "model",
        ],
    )

    assert dataset_card.exit_code == 0
    assert model_card.exit_code == 0
    assert "# Dataset Card:" in (tmp_path / "DATASET_CARD.md").read_text()
    assert "# Model Card:" in (tmp_path / "MODEL_CARD.md").read_text()
