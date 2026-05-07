from __future__ import annotations

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
    approved_rate: float | None = None


class BenchmarkMetric(BaseModel):
    name: str
    score: float
    generated_value: float | int | str | None
    reference_value: float | int | str | None
    details: dict = Field(default_factory=dict)


class BenchmarkReport(BaseModel):
    generated_dataset_id: str
    reference_dataset_id: str
    overall_score: float
    generated_profile: CohortProfile
    reference_profile: CohortProfile
    metrics: list[BenchmarkMetric]
    warnings: list[str] = Field(default_factory=list)

