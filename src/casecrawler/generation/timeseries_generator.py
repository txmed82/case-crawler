from __future__ import annotations

from datetime import datetime, timedelta

from casecrawler.models.synthetic import (
    SyntheticRecord,
    TimeSeriesChannel,
    TimeSeriesPoint,
)


class TimeSeriesGenerator:
    def add_time_series(
        self,
        record: SyntheticRecord,
        channels: list[str] | None = None,
        points: int = 6,
    ) -> SyntheticRecord:
        if record.time_series:
            return record

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

        channels = []
        selected_channels = channels or list(base_values)
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
            channels.append(
                TimeSeriesChannel(
                    name=name,
                    unit=units[name],
                    sampling_rate_hz=None,
                    points=series_points,
                )
            )

        return record.model_copy(update={"time_series": [*record.time_series, *channels]})


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
