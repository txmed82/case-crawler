from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TimeSeriesModelProfile:
    name: str
    adapter_type: str
    reference: str
    notes: str
    model_id: str | None = None
    license: str | None = None
    gated: bool = False
    use_policy: str = "review_license_before_use"
    command_template: list[str] = field(default_factory=list)
    input_contract: dict[str, object] = field(default_factory=dict)
    output_contract: dict[str, object] = field(default_factory=dict)
    validation_requirements: list[str] = field(default_factory=list)


TIME_SERIES_MODEL_PROFILES: dict[str, TimeSeriesModelProfile] = {
    "timediff": TimeSeriesModelProfile(
        name="timediff",
        adapter_type="external_command",
        reference="https://github.com/MuhangTian/TimeDiff",
        model_id="MuhangTian/TimeDiff",
        license="mit",
        gated=False,
        use_policy="wrap_external_sampler_validate_outputs",
        command_template=["timediff-sample", "--checkpoint", "<checkpoint>"],
        input_contract={
            "transport": "stdin",
            "stdin_json": ["record", "channels", "points"],
            "record_schema": "SyntheticRecord.model_dump()",
            "channels": "requested channel names or null",
            "points": "requested point count",
        },
        output_contract={
            "transport": "stdout",
            "stdout_json": "TimeSeriesChannel[] or {'channels': TimeSeriesChannel[]}",
            "channel_fields": ["name", "unit", "points"],
            "point_fields": ["timestamp", "values"],
            "optional_fields": ["sampling_rate_hz", "generation_backend"],
        },
        validation_requirements=[
            "generation_backend",
            "schema_valid_TimeSeriesChannel",
            "monotonic_or_parseable_timestamps",
            "non_empty_points",
        ],
        notes=(
            "Diffusion model reference for mixed-type EHR time-series generation; "
            "wrap a trained sampler with synthetic record JSON in/stdout."
        ),
    ),
    "rawmed": TimeSeriesModelProfile(
        name="rawmed",
        adapter_type="external_command",
        reference="https://github.com/eunbyeol-cho/RawMed",
        model_id="eunbyeol-cho/RawMed",
        license=None,
        gated=False,
        use_policy="research_reference_validate_outputs",
        command_template=["rawmed-sample", "--config", "<config.yaml>"],
        input_contract={
            "transport": "stdin",
            "stdin_json": ["record", "channels", "points"],
            "record_schema": "SyntheticRecord.model_dump()",
            "channels": "requested table/channel names or null",
            "points": "requested point count",
        },
        output_contract={
            "transport": "stdout",
            "stdout_json": "TimeSeriesChannel[] or {'channels': TimeSeriesChannel[]}",
            "channel_fields": ["name", "unit", "points"],
            "point_fields": ["timestamp", "values"],
            "optional_fields": ["sampling_rate_hz", "generation_backend"],
        },
        validation_requirements=[
            "generation_backend",
            "schema_valid_TimeSeriesChannel",
            "unit_consistency",
            "non_empty_points",
        ],
        notes=(
            "Research reference for multi-table time-series EHR synthesis; wrap "
            "exported sampler output into TimeSeriesChannel JSON."
        ),
    ),
    "mira": TimeSeriesModelProfile(
        name="mira",
        adapter_type="external_command",
        reference="https://huggingface.co/MIRA-Mode/MIRA",
        model_id="MIRA-Mode/MIRA",
        license="mit",
        gated=False,
        use_policy="forecasting_backbone_validate_synthetic_rollouts",
        command_template=["mira-rollout", "--model", "MIRA-Mode/MIRA"],
        input_contract={
            "transport": "stdin",
            "stdin_json": ["record", "channels", "points"],
            "record_schema": "SyntheticRecord.model_dump()",
            "channels": "requested irregular clinical series channels or null",
            "points": "requested rollout horizon",
        },
        output_contract={
            "transport": "stdout",
            "stdout_json": "TimeSeriesChannel[] or {'channels': TimeSeriesChannel[]}",
            "channel_fields": ["name", "unit", "points"],
            "point_fields": ["timestamp", "values"],
            "optional_fields": ["sampling_rate_hz", "generation_backend"],
        },
        validation_requirements=[
            "generation_backend",
            "schema_valid_TimeSeriesChannel",
            "clinical_range_review",
            "non_empty_points",
        ],
        notes=(
            "Medical time-series foundation model profile for irregular, "
            "heterogeneous clinical forecasting; wrap as a backbone or critic "
            "for synthetic rollouts and validate generated channels before export."
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
