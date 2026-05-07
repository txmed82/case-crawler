from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CohortProfile(BaseModel):
    dataset_id: str
    record_count: int
    modality_counts: dict[str, int] = Field(default_factory=dict)
    mean_age: float | None = None
    sex_counts: dict[str, int] = Field(default_factory=dict)
    mean_document_chars: float | None = None
    note_type_counts: dict[str, int] = Field(default_factory=dict)
    lab_name_counts: dict[str, int] = Field(default_factory=dict)
    vital_name_counts: dict[str, int] = Field(default_factory=dict)
    medication_name_counts: dict[str, int] = Field(default_factory=dict)
    time_series_channel_counts: dict[str, int] = Field(default_factory=dict)
    mean_time_series_points: float | None = None
    mean_time_series_duration_hours: float | None = None
    imaging_modality_counts: dict[str, int] = Field(default_factory=dict)
    imaging_body_region_counts: dict[str, int] = Field(default_factory=dict)
    approved_rate: float | None = None


class BenchmarkMetric(BaseModel):
    name: str
    score: float
    generated_value: float | int | str | None
    reference_value: float | int | str | None
    details: dict[str, Any] = Field(default_factory=dict)


class BenchmarkReport(BaseModel):
    generated_dataset_id: str
    reference_dataset_id: str
    overall_score: float
    generated_profile: CohortProfile
    reference_profile: CohortProfile
    metrics: list[BenchmarkMetric]
    warnings: list[str] = Field(default_factory=list)
