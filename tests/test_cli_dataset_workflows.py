import re

from click.testing import CliRunner

from casecrawler.cli import cli


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
