from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from casecrawler.models.synthetic import ComplexityProfile, Modality


class ExportFormat(str, Enum):
    RAW_JSONL = "raw_jsonl"
    SFT_JSONL = "sft_jsonl"
    NOTE_FACT_SFT_JSONL = "note_fact_sft_jsonl"
    CLINICAL_OBSERVATION_JSONL = "clinical_observation_jsonl"
    MEDICATION_RECONCILIATION_JSONL = "medication_reconciliation_jsonl"
    CHAT_JSONL = "chat_jsonl"
    TOOL_CALL_JSONL = "tool_call_jsonl"
    MULTIMODAL_JSONL = "multimodal_jsonl"
    TIME_SERIES_JSONL = "time_series_jsonl"
    DPO_JSONL = "dpo_jsonl"
    RL_JSONL = "rl_jsonl"
    FHIR_NDJSON = "fhir_ndjson"
    PARQUET = "parquet"


class HumanReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


class HumanReviewDecision(BaseModel):
    status: HumanReviewStatus
    reviewer: str = "human"
    notes: list[str] = Field(default_factory=list)
    reviewed_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReviewQueueItem(BaseModel):
    record_id: str
    dataset_id: str
    topic: str
    complexity: ComplexityProfile
    modalities: list[Modality]
    validation_approved: bool | None = None
    human_review: HumanReviewDecision | None = None
    issue_count: int = 0
    blocking_issue_count: int = 0


class GenerationRequest(BaseModel):
    topic: str
    count: int = Field(default=1, ge=1)
    recipe: str | None = Field(default=None, min_length=1)
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
    clinical_text_backend: Literal["deterministic", "llm"] | None = None
    llm_provider: str | None = Field(default=None, min_length=1)
    llm_model: str | None = Field(default=None, min_length=1)
    ollama_base_url: str | None = Field(default=None, min_length=1)
    imaging_backend: Literal["placeholder", "diffusers"] | None = None
    imaging_model_profile: str | None = Field(default=None, min_length=1)
    diffusers_model_id: str | None = Field(default=None, min_length=1)
    time_series_backend: Literal["deterministic", "external"] | None = None
    time_series_model_profile: str | None = Field(default=None, min_length=1)
    time_series_command: list[str] | None = Field(default=None, min_length=1)
    require_human_review: bool = False
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
