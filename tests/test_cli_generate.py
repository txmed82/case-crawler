from click.testing import CliRunner

from casecrawler.cli import cli


def test_legacy_generate_command_is_not_registered():
    runner = CliRunner()

    result = runner.invoke(cli, ["generate", "SAH", "--count", "1"])

    assert result.exit_code != 0
    assert "No such command" in result.output


def test_legacy_cases_command_is_not_registered():
    runner = CliRunner()

    result = runner.invoke(cli, ["cases"])

    assert result.exit_code != 0
    assert "No such command" in result.output


def test_legacy_case_export_command_is_not_registered():
    runner = CliRunner()

    result = runner.invoke(cli, ["export", "cases.jsonl"])

    assert result.exit_code != 0
    assert "No such command" in result.output
