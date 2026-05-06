from click.testing import CliRunner

from casecrawler.cli import cli


def test_generate_dataset_command_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(cli, ["generate-dataset", "sepsis", "--count", "1"])

    assert result.exit_code == 0
    assert "Generated: 1" in result.output
    assert "Approved: 1" in result.output

