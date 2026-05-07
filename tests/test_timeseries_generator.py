from casecrawler.generation.structured_generator import StructuredGenerator
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
    }
    heart_rate = next(channel for channel in updated.time_series if channel.name == "heart_rate")
    assert len(heart_rate.points) == 6
    assert heart_rate.points[0].timestamp == "2026-01-01T00:00:00"


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
    assert '"channels": ["heart_rate"]' in calls[0][1]
    assert updated.time_series[0].name == "heart_rate"
    assert updated.time_series[0].points[0].values["value"] == 101.0
