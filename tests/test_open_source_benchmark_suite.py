import json

from click.testing import CliRunner

from casecrawler.cli import cli
from casecrawler.evaluation.open_source_suite import (
    OpenSourceBenchmarkSuite,
    default_open_source_requests,
)
from casecrawler.models.dataset import GenerationRequest
from casecrawler.models.synthetic import Modality


def test_open_source_benchmark_suite_runs_offline():
    result = OpenSourceBenchmarkSuite().run_generation_smoke(
        requests=[
            GenerationRequest(
                topic="sepsis",
                count=2,
                modalities=[
                    Modality.STRUCTURED_EHR,
                    Modality.CLINICAL_TEXT,
                    Modality.LABS,
                    Modality.VITALS,
                    Modality.TIME_SERIES,
                ],
            )
        ]
    )

    report = result.to_report()
    assert report["artifact_type"] == "casecrawler_open_source_benchmark"
    assert report["passed"] is True
    assert report["generated"] == 2
    assert report["approved"] == 2
    assert report["blocking_issues"] == 0
    assert report["scenarios"][0]["topic"] == "sepsis"
    assert report["scenarios"][0]["artifact_counts"]["documents"] > 0
    assert report["scenarios"][0]["artifact_counts"]["time_series_channels"] > 0


def test_default_open_source_requests_cover_core_scenarios():
    requests = default_open_source_requests(count=3)

    assert [request.topic for request in requests] == [
        "sepsis",
        "heart failure exacerbation",
    ]
    assert all(request.count == 3 for request in requests)
    assert all(Modality.TIME_SERIES in request.modalities for request in requests)


def test_cli_benchmark_open_source_outputs_json_report(tmp_path):
    output = tmp_path / "open-source-benchmark.json"
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "benchmark-open-source",
            "--topic",
            "sepsis",
            "--count",
            "1",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    stdout_payload = json.loads(result.output)
    file_payload = json.loads(output.read_text())
    assert stdout_payload == file_payload
    assert file_payload["passed"] is True
    assert file_payload["scenarios"][0]["topic"] == "sepsis"
