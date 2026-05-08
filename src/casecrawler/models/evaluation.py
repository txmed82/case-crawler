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
    document_author_role_counts: dict[str, int] = Field(default_factory=dict)
    messy_document_rate: float | None = None
    extracted_fact_key_counts: dict[str, int] = Field(default_factory=dict)
    extracted_fact_density: dict[str, float] = Field(default_factory=dict)
    artifact_counts: dict[str, int] = Field(default_factory=dict)
    artifact_density: dict[str, float] = Field(default_factory=dict)
    modality_artifact_coverage: dict[str, float] = Field(default_factory=dict)
    lab_name_counts: dict[str, int] = Field(default_factory=dict)
    lab_flag_counts: dict[str, int] = Field(default_factory=dict)
    lab_numeric_summaries: dict[str, dict[str, float | int]] = Field(default_factory=dict)
    vital_name_counts: dict[str, int] = Field(default_factory=dict)
    vital_numeric_summaries: dict[str, dict[str, float | int]] = Field(default_factory=dict)
    medication_name_counts: dict[str, int] = Field(default_factory=dict)
    medication_route_counts: dict[str, int] = Field(default_factory=dict)
    medication_status_counts: dict[str, int] = Field(default_factory=dict)
    time_series_channel_counts: dict[str, int] = Field(default_factory=dict)
    time_series_backend_counts: dict[str, int] = Field(default_factory=dict)
    mean_time_series_points: float | None = None
    mean_time_series_duration_hours: float | None = None
    imaging_modality_counts: dict[str, int] = Field(default_factory=dict)
    imaging_body_region_counts: dict[str, int] = Field(default_factory=dict)
    imaging_backend_counts: dict[str, int] = Field(default_factory=dict)
    imaging_label_counts: dict[str, int] = Field(default_factory=dict)
    imaging_label_pair_counts: dict[str, int] = Field(default_factory=dict)
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
    passed: bool = False
    failing_metrics: list[str] = Field(default_factory=list)
    thresholds: dict[str, float] = Field(default_factory=dict)
    generated_profile: CohortProfile
    reference_profile: CohortProfile
    metrics: list[BenchmarkMetric]
    warnings: list[str] = Field(default_factory=list)


class DatasetQualityReport(BaseModel):
    dataset_id: str
    record_count: int
    approved_count: int
    approval_rate: float
    export_ready: bool
    benchmark_ready: bool | None = None
    recommended_reference_keys: list[str] = Field(default_factory=list)
    resolved_reference_dataset_id: str | None = None
    missing_reference_keys: list[str] = Field(default_factory=list)
    benchmark_thresholds: dict[str, float] = Field(default_factory=dict)
    modality_counts: dict[str, int] = Field(default_factory=dict)
    artifact_counts: dict[str, int] = Field(default_factory=dict)
    note_type_counts: dict[str, int] = Field(default_factory=dict)
    blocking_issue_count: int = 0
    warning_issue_count: int = 0
    issue_counts_by_field: dict[str, int] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)
