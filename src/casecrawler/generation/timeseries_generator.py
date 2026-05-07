from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta
from typing import Protocol

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
        units = {
            "heart_rate": "/min",
            "systolic_bp": "mmHg",
            "spo2": "%",
            "lactate": "mmol/L",
        }

        generated_channels = []
        selected_channels = channels if channels else list(base_values)
        for name in selected_channels:
            if name not in base_values:
                continue
            base = base_values[name]
            series_points = []
            for offset in range(points):
                timestamp = (start + timedelta(hours=offset)).isoformat()
                drift = _drift(name, offset)
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
        if not isinstance(raw_channels, list):
            raise RuntimeError("External time-series backend must return a JSON list.")
        generated_channels = [
            TimeSeriesChannel.model_validate(channel) for channel in raw_channels
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


def _run_external_command(command: list[str], payload: str) -> str:
    try:
        result = subprocess.run(
            command,
            input=payload,
            capture_output=True,
            check=True,
            text=True,
            timeout=EXTERNAL_TIME_SERIES_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "External time-series backend timed out after "
            f"{EXTERNAL_TIME_SERIES_TIMEOUT_SECONDS:.0f}s: {command!r}."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "External time-series backend failed with exit code "
            f"{exc.returncode}: {command!r}. stdout={exc.stdout!r} stderr={exc.stderr!r}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"External time-series backend could not be executed: {command!r}."
        ) from exc
    return result.stdout
