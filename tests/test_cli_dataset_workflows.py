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


def test_dataset_cli_lists_generation_recipes():
    result = CliRunner().invoke(cli, ["generation-recipes"])

    assert result.exit_code == 0
    assert "full_multimodal_acute_care" in result.output
    assert "icu_timeseries_notes" in result.output


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
    assert Modality.IMAGING in records[0].modalities
    assert {record.imaging[0].modality for record in records} == {"CTA"}
    assert records[0].time_series


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
    assert "asclepius" in listed.output
    assert "augmented_clinical_notes" in listed.output
    assert "clinical_notes_to_fhir" in listed.output
    assert "radiology_report_consistency" in listed.output
    assert imported.exit_code == 0
    assert "Imported 1 reference record(s)" in imported.output
    assert store.dataset_exists("ds-hf-reference")
    assert store.get_manifest("ds-hf-reference").metadata["record_ids"]


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
            "--limit",
            "1",
        ],
    )
    store = DatasetStore()

    assert result.exit_code == 0
    assert "Imported 1 reference record(s) from org/custom-synthetic-notes" in result.output
    record = store.list_records(dataset_id="ds-custom-reference")[0]
    assert record.metadata["reference_dataset"] == "org/custom-synthetic-notes"
    assert record.documents[0].extracted_facts["instruction"] == "Extract diagnosis."


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
