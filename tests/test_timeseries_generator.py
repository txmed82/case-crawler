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
