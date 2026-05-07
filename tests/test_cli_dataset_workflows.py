import re

from click.testing import CliRunner

from casecrawler.cli import cli
from casecrawler.integrations.huggingface import import_reference_rows
from casecrawler.models.synthetic import (
    ComplexityProfile,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
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
            "--output",
            "benchmark.json",
        ],
    )

    assert result.exit_code == 0
    assert "Overall score:" in result.output
    assert (tmp_path / "benchmark.json").exists()


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
    assert imported.exit_code == 0
    assert "Imported 1 reference record(s)" in imported.output
    assert store.dataset_exists("ds-hf-reference")
    assert store.get_manifest("ds-hf-reference").metadata["record_ids"]


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
