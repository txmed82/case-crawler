from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeSeriesModelProfile:
    name: str
    adapter_type: str
    reference: str
    notes: str


TIME_SERIES_MODEL_PROFILES: dict[str, TimeSeriesModelProfile] = {
    "timediff": TimeSeriesModelProfile(
        name="timediff",
        adapter_type="external_command",
        reference="https://github.com/MuhangTian/TimeDiff",
        notes=(
            "Diffusion model reference for mixed-type EHR time-series generation; "
            "wrap a trained sampler with synthetic record JSON in/stdout."
        ),
    ),
    "rawmed": TimeSeriesModelProfile(
        name="rawmed",
        adapter_type="external_command",
        reference="https://github.com/eunbyeol-cho/RawMed",
        notes=(
            "Research reference for multi-table time-series EHR synthesis; wrap "
            "exported sampler output into TimeSeriesChannel JSON."
        ),
    ),
}


def list_time_series_model_profiles() -> list[TimeSeriesModelProfile]:
    return list(TIME_SERIES_MODEL_PROFILES.values())


def resolve_time_series_model_profile(name: str | None) -> TimeSeriesModelProfile | None:
    if name is None:
        return None
    try:
        return TIME_SERIES_MODEL_PROFILES[name]
    except KeyError as exc:
        available = ", ".join(sorted(TIME_SERIES_MODEL_PROFILES))
        raise ValueError(
            f"Unknown time-series model profile '{name}'. Available profiles: {available}."
        ) from exc
