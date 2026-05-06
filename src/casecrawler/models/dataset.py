from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from casecrawler.models.synthetic import ComplexityProfile, Modality


class ExportFormat(str, Enum):
    RAW_JSONL = "raw_jsonl"
    SFT_JSONL = "sft_jsonl"
    CHAT_JSONL = "chat_jsonl"
    MULTIMODAL_JSONL = "multimodal_jsonl"
    RL_JSONL = "rl_jsonl"
    FHIR_NDJSON = "fhir_ndjson"
    PARQUET = "parquet"


class GenerationRequest(BaseModel):
    topic: str
    count: int = Field(default=1, ge=1)
    complexity: ComplexityProfile = ComplexityProfile.MODERATE
    modalities: list[Modality] = Field(
        default_factory=lambda: [
            Modality.STRUCTURED_EHR,
            Modality.CLINICAL_TEXT,
            Modality.LABS,
            Modality.VITALS,
        ]
    )
    cohort_constraints: dict = Field(default_factory=dict)
    export_formats: list[ExportFormat] = Field(
        default_factory=lambda: [ExportFormat.SFT_JSONL]
    )
    ingest_first: bool = False
    validation_threshold: float = Field(default=0.8, ge=0.0, le=1.0)


class DatasetManifest(BaseModel):
    dataset_id: str
    name: str
    topic: str
    requested_count: int
    generated_count: int
    approved_count: int
    modalities: list[Modality]
    export_formats: list[ExportFormat]
    created_at: str
    metadata: dict = Field(default_factory=dict)


class ExportManifest(BaseModel):
    dataset_id: str
    export_format: ExportFormat
    file_path: str
    record_count: int
    created_at: str
    metadata: dict = Field(default_factory=dict)
