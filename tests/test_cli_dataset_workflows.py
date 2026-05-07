import re

from click.testing import CliRunner

from casecrawler.cli import cli
from casecrawler.integrations.huggingface import import_reference_rows
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
