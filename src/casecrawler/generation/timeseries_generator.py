from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta
from typing import Protocol

from casecrawler.generation._external_subprocess import run_external_command
from casecrawler.models.synthetic import (
    SyntheticRecord,
    TimeSeriesChannel,
    TimeSeriesPoint,
)


EXTERNAL_TIME_SERIES_TIMEOUT_SECONDS = 120.0


class ExternalTimeSeriesRunner(Protocol):
    def __call__(self, command: list[str], payload: str) -> str: ...


class TimeSeriesGenerator:
    def __init__(
        self,
        external_command: list[str] | None = None,
        external_runner: ExternalTimeSeriesRunner | None = None,
    ) -> None:
        if external_command is not None and not external_command:
            raise ValueError("external_command must not be empty when provided.")
        self._external_command = external_command
        self._external_runner = external_runner or _run_external_command

    def add_time_series(
        self,
        record: SyntheticRecord,
        channels: list[str] | None = None,
        points: int = 6,
    ) -> SyntheticRecord:
        if record.time_series:
            return record
        if self._external_command is not None:
            return self._add_external_time_series(record, channels=channels, points=points)

        start = datetime.fromisoformat(record.provenance.created_at.replace("Z", "+00:00"))
        base_values = {
            "heart_rate": _first_vital(record, "HR", 96.0),
            "systolic_bp": _first_vital(record, "SBP", 118.0),
            "spo2": _first_vital(record, "SpO2", 96.0),
            "lactate": _first_lab(record, "Lactate", 1.4),
        }
        lab_channels = _numeric_lab_channels(record)
        observed_channels = _observation_channels(record)
        units = {
            "heart_rate": "/min",
            "systolic_bp": "mmHg",
            "spo2": "%",
            "lactate": "mmol/L",
            **{name: unit for name, (_, unit, _) in lab_channels.items()},
        }
        waveform_specs = {
            "ecg_lead_ii": {"unit": "mV", "sampling_rate_hz": 125.0, "minimum_points": 125},
            "pleth": {"unit": "relative", "sampling_rate_hz": 25.0, "minimum_points": 100},
        }

        generated_channels = []
        selected_channels = channels if channels else [
            *base_values,
            *lab_channels,
            *waveform_specs,
        ]
        for name in selected_channels:
            if name in waveform_specs:
                generated_channels.append(
                    _waveform_channel(
                        name=name,
                        start=start,
                        heart_rate=base_values["heart_rate"],
                        spo2=base_values["spo2"],
                        points=points,
                        spec=waveform_specs[name],
                    )
                )
                continue
            observed_points = observed_channels.get(name)
            if observed_points and len(observed_points) > 1:
                generated_channels.append(
                    TimeSeriesChannel(
                        name=name,
                        unit=units[name],
                        generation_backend="deterministic:structured-observations",
                        sampling_rate_hz=None,
                        points=observed_points,
                    )
                )
                continue
            if name not in base_values:
                lab_channel = lab_channels.get(name)
                if lab_channel is None:
                    continue
                base, _unit, lab_target = lab_channel
                is_lab_channel = True
            else:
                base = base_values[name]
                lab_target = base
                is_lab_channel = False
            series_points = []
            for offset in range(points):
                timestamp = (start + timedelta(hours=offset)).isoformat()
                drift = (
                    _lab_drift(base, lab_target, offset, points)
                    if is_lab_channel
                    else _drift(name, offset)
                )
                series_points.append(
                    TimeSeriesPoint(
                        timestamp=timestamp,
                        values={"value": round(max(base + drift, 0.0), 3)},
                    )
                )
            generated_channels.append(
                TimeSeriesChannel(
                    name=name,
                    unit=units[name],
                    generation_backend="deterministic",
                    sampling_rate_hz=None,
                    points=series_points,
                )
            )

        return record.model_copy(
            update={"time_series": [*record.time_series, *generated_channels]}
        )

    def _add_external_time_series(
        self,
        record: SyntheticRecord,
        channels: list[str] | None,
        points: int,
    ) -> SyntheticRecord:
        assert self._external_command is not None
        payload = json.dumps(
            {
                "record": record.model_dump(),
                "channels": channels,
                "points": points,
            },
            sort_keys=True,
        )
        output = self._external_runner(self._external_command, payload)
        try:
            raw_channels = json.loads(output)
        except json.JSONDecodeError as exc:
            raise RuntimeError("External time-series backend returned invalid JSON.") from exc
        if isinstance(raw_channels, dict):
            raw_channels = raw_channels.get("channels")
        if not isinstance(raw_channels, list):
            raise RuntimeError(
                "External time-series backend must return a JSON list or "
                "an object with a channels list."
            )
        backend = f"external:{' '.join(self._external_command)}"
        generated_channels = [
            _external_channel(channel, backend=backend)
            for channel in raw_channels
        ]
        return record.model_copy(
            update={"time_series": [*record.time_series, *generated_channels]}
        )


def _first_vital(record: SyntheticRecord, name: str, fallback: float) -> float:
    for vital in record.vitals:
        if vital.name == name:
            return float(vital.value)
    return fallback


def _first_lab(record: SyntheticRecord, name: str, fallback: float) -> float:
    for lab in record.labs:
        if lab.name == name and isinstance(lab.value, (int, float)):
            return float(lab.value)
    return fallback


def _drift(name: str, offset: int) -> float:
    if name in {"heart_rate", "lactate"}:
        return -1.5 * offset
    if name == "systolic_bp":
        return 1.2 * offset
    if name == "spo2":
        return 0.4 * offset
    return 0.0


def _numeric_lab_channels(record: SyntheticRecord) -> dict[str, tuple[float, str, float]]:
    channels: dict[str, tuple[float, str, float]] = {}
    for lab in record.labs:
        if not isinstance(lab.value, (int, float)):
            continue
        channel_name = f"lab_{_slug(lab.name)}"
        if channel_name in channels:
            continue
        target = _lab_target(float(lab.value), lab.reference_low, lab.reference_high)
        channels[channel_name] = (float(lab.value), lab.unit, target)
    return channels


def _observation_channels(record: SyntheticRecord) -> dict[str, list[TimeSeriesPoint]]:
    channels: dict[str, list[TimeSeriesPoint]] = {}
    for vital in record.vitals:
        channel_name = _vital_channel_name(vital.name)
        if channel_name is None:
            continue
        channels.setdefault(channel_name, []).append(
            TimeSeriesPoint(
                timestamp=vital.effective_time,
                values={"value": round(float(vital.value), 3)},
            )
        )
    for lab in record.labs:
        if not isinstance(lab.value, (int, float)):
            continue
        channels.setdefault(f"lab_{_slug(lab.name)}", []).append(
            TimeSeriesPoint(
                timestamp=lab.effective_time,
                values={"value": round(float(lab.value), 3)},
            )
        )
    return {
        name: sorted(points, key=lambda point: point.timestamp)
        for name, points in channels.items()
    }


def _vital_channel_name(name: str) -> str | None:
    normalized = _slug(name)
    if normalized in {"hr", "heart_rate"}:
        return "heart_rate"
    if normalized in {"sbp", "systolic_bp", "systolic_blood_pressure"}:
        return "systolic_bp"
    if normalized in {"spo2", "oxygen_saturation"}:
        return "spo2"
    return None


def _lab_target(
    value: float,
    reference_low: float | None,
    reference_high: float | None,
) -> float:
    if reference_low is not None and value < reference_low:
        return reference_low
    if reference_high is not None and value > reference_high:
        return reference_high
    return value


def _lab_drift(base: float, target: float, offset: int, points: int) -> float:
    if points <= 1:
        return 0.0
    progress = min(1.0, offset / (points - 1))
    return (target - base) * 0.45 * progress


def _slug(value: str) -> str:
    return re.sub(r"\W+", "_", value.lower()).strip("_")


def _waveform_channel(
    *,
    name: str,
    start: datetime,
    heart_rate: float,
    spo2: float,
    points: int,
    spec: dict,
) -> TimeSeriesChannel:
    sampling_rate_hz = float(spec["sampling_rate_hz"])
    point_count = max(points, int(spec["minimum_points"]))
    series_points = []
    for sample_index in range(point_count):
        timestamp = (start + timedelta(seconds=sample_index / sampling_rate_hz)).isoformat()
        phase = (sample_index / sampling_rate_hz) * (heart_rate / 60.0)
        if name == "ecg_lead_ii":
            values = {
                "millivolts": round(_ecg_sample(phase), 4),
                "phase": round(phase % 1.0, 4),
            }
        else:
            values = {
                "amplitude": round(_pleth_sample(phase, spo2), 4),
                "phase": round(phase % 1.0, 4),
            }
        series_points.append(TimeSeriesPoint(timestamp=timestamp, values=values))
    return TimeSeriesChannel(
        name=name,
        unit=spec["unit"],
        generation_backend="deterministic",
        sampling_rate_hz=sampling_rate_hz,
        points=series_points,
    )


def _external_channel(channel: object, *, backend: str) -> TimeSeriesChannel:
    if not isinstance(channel, dict):
        return TimeSeriesChannel.model_validate(channel)
    return TimeSeriesChannel.model_validate(
        {"generation_backend": backend, **channel}
    )


def _ecg_sample(phase: float) -> float:
    cycle = phase % 1.0
    baseline = 0.03 * math.sin(2 * math.pi * cycle)
    p_wave = 0.08 * _gaussian(cycle, 0.18, 0.035)
    q_wave = -0.12 * _gaussian(cycle, 0.37, 0.012)
    r_wave = 1.05 * _gaussian(cycle, 0.40, 0.01)
    s_wave = -0.25 * _gaussian(cycle, 0.43, 0.014)
    t_wave = 0.28 * _gaussian(cycle, 0.68, 0.08)
    return baseline + p_wave + q_wave + r_wave + s_wave + t_wave


def _pleth_sample(phase: float, spo2: float) -> float:
    cycle = phase % 1.0
    oxygen_scale = max(0.75, min(1.05, spo2 / 96.0))
    upstroke = 1.1 * _gaussian(cycle, 0.18, 0.08)
    dicrotic_notch = -0.18 * _gaussian(cycle, 0.38, 0.025)
    runoff = 0.45 * _gaussian(cycle, 0.55, 0.18)
    respiratory_variation = 0.04 * math.sin(2 * math.pi * phase / 4)
    return max(0.0, oxygen_scale * (upstroke + dicrotic_notch + runoff) + respiratory_variation)


def _gaussian(value: float, mean: float, sigma: float) -> float:
    return math.exp(-0.5 * ((value - mean) / sigma) ** 2)


def _run_external_command(command: list[str], payload: str) -> str:
    return run_external_command(
        command,
        payload,
        backend_label="time-series",
        timeout_seconds=EXTERNAL_TIME_SERIES_TIMEOUT_SECONDS,
    )
