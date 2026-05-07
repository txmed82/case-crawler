import json
import subprocess

from casecrawler.generation.structured_generator import StructuredGenerator
from casecrawler.generation import timeseries_generator
from casecrawler.generation.timeseries_generator import TimeSeriesGenerator
from casecrawler.models.dataset import GenerationRequest
from casecrawler.models.synthetic import Modality


def test_timeseries_generator_adds_longitudinal_channels():
    req = GenerationRequest(
        topic="sepsis",
        modalities=[Modality.TIME_SERIES],
        cohort_constraints={"base_time": "2026-01-01T00:00:00"},
    )
    record = StructuredGenerator().generate("ds-1", req, 0)

    updated = TimeSeriesGenerator().add_time_series(record)

    assert updated.time_series
    assert {channel.name for channel in updated.time_series} >= {
        "heart_rate",
        "systolic_bp",
        "spo2",
        "lactate",
        "ecg_lead_ii",
        "pleth",
    }
    heart_rate = next(channel for channel in updated.time_series if channel.name == "heart_rate")
    assert len(heart_rate.points) == 6
    assert heart_rate.points[0].timestamp == "2026-01-01T00:00:00"


def test_timeseries_generator_adds_numeric_lab_trajectories():
    req = GenerationRequest(
        topic="acute pancreatitis",
        modalities=[Modality.TIME_SERIES],
        cohort_constraints={"base_time": "2026-01-01T00:00:00"},
    )
    record = StructuredGenerator().generate("ds-1", req, 0)

    updated = TimeSeriesGenerator().add_time_series(record)
    channels = {channel.name: channel for channel in updated.time_series}

    assert "lab_lipase" in channels
    assert "lab_wbc" in channels
    assert channels["lab_lipase"].unit == "U/L"
    assert channels["lab_lipase"].points[0].values["value"] == 1240
    assert channels["lab_lipase"].points[-1].values["value"] < 1240
    assert channels["lab_wbc"].points[-1].values["value"] < channels["lab_wbc"].points[0].values["value"]


def test_timeseries_generator_adds_waveform_like_channels():
    req = GenerationRequest(
        topic="sepsis",
        modalities=[Modality.TIME_SERIES],
        cohort_constraints={"base_time": "2026-01-01T00:00:00"},
    )
    record = StructuredGenerator().generate("ds-1", req, 0)

    updated = TimeSeriesGenerator().add_time_series(
        record,
        channels=["ecg_lead_ii", "pleth"],
        points=8,
    )

    ecg = next(channel for channel in updated.time_series if channel.name == "ecg_lead_ii")
    pleth = next(channel for channel in updated.time_series if channel.name == "pleth")
    assert ecg.sampling_rate_hz == 125
    assert pleth.sampling_rate_hz == 25
    assert len(ecg.points) >= 125
    assert len(pleth.points) >= 100
    assert {"millivolts", "phase"} <= set(ecg.points[0].values)
    assert {"amplitude", "phase"} <= set(pleth.points[0].values)
    assert ecg.points[1].timestamp == "2026-01-01T00:00:00.008000"


def test_timeseries_generator_honors_requested_channels():
    req = GenerationRequest(
        topic="sepsis",
        modalities=[Modality.TIME_SERIES],
        cohort_constraints={"base_time": "2026-01-01T00:00:00"},
    )
    record = StructuredGenerator().generate("ds-1", req, 0)

    updated = TimeSeriesGenerator().add_time_series(record, channels=["heart_rate"])

    assert [channel.name for channel in updated.time_series] == ["heart_rate"]


def test_timeseries_generator_can_use_external_backend():
    req = GenerationRequest(
        topic="sepsis",
        modalities=[Modality.TIME_SERIES],
        cohort_constraints={"base_time": "2026-01-01T00:00:00"},
    )
    record = StructuredGenerator().generate("ds-1", req, 0)
    calls = []

    def fake_runner(command, payload):
        calls.append((command, payload))
        return (
            "[{\"name\":\"heart_rate\",\"unit\":\"/min\",\"points\":["
            "{\"timestamp\":\"2026-01-01T00:00:00\",\"values\":{\"value\":101.0}}"
            "]}]"
        )

    updated = TimeSeriesGenerator(
        external_command=["timediff-sample"],
        external_runner=fake_runner,
    ).add_time_series(record, channels=["heart_rate"], points=1)

    assert calls[0][0] == ["timediff-sample"]
    payload = json.loads(calls[0][1])
    assert payload["channels"] == ["heart_rate"]
    assert payload["points"] == 1
    assert updated.time_series[0].name == "heart_rate"
    assert updated.time_series[0].points[0].values["value"] == 101.0


def test_timeseries_generator_rejects_empty_external_command():
    try:
        TimeSeriesGenerator(external_command=[])
    except ValueError as exc:
        assert "external_command must not be empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty external command.")


def test_external_timeseries_runner_normalizes_process_failures(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=2,
            cmd=["timediff-sample"],
            output="bad output",
            stderr="bad error",
        )

    monkeypatch.setattr(timeseries_generator.subprocess, "run", fake_run)

    try:
        timeseries_generator._run_external_command(["timediff-sample"], "{}")
    except RuntimeError as exc:
        assert "exit code 2" in str(exc)
        assert "bad error" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for backend process failure.")
