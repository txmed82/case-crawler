from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from casecrawler.models.dataset import ExportFormat
from casecrawler.models.synthetic import ComplexityProfile, Modality


class SourcesConfig(BaseModel):
    priority: list[str] = [
        "pubmed", "glass", "openfda", "annas_archive",
        "dailymed", "rxnorm", "medrxiv", "clinicaltrials", "firecrawl",
    ]
    disabled: list[str] = []


class IngestionConfig(BaseModel):
    default_limit_per_source: int = 20
    sources: SourcesConfig = SourcesConfig()


class ChunkingConfig(BaseModel):
    default_chunk_size: int = 500
    overlap: int = 50


class EmbeddingConfig(BaseModel):
    model: str = "all-MiniLM-L6-v2"


class StorageConfig(BaseModel):
    chroma_persist_dir: str = "./data/chroma"


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class LlmConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    ollama_base_url: str = "http://localhost:11434"


class JudgeConfig(BaseModel):
    """LLM judge used during preference / DPO data construction.

    Open-source guidance:
    - Pick a *different* provider from the generator. Self-judging biases
      the resulting preference pairs (the Anthropic-on-Anthropic case is
      particularly bad). The pipeline emits a runtime warning when the
      provider matches the generator.
    - Cost: judges should be cheap models. The defaults below are the
      lightest tier each provider currently offers.
    """

    provider: str | None = None
    model: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class GroundingConfig(BaseModel):
    """RAG grounding configuration for synthetic generation.

    When ``enabled`` is true, ``SyntheticPipeline`` attaches a
    :class:`GroundingBundle` to every record's ``metadata["grounding"]``
    and the validator requires at least one citation. When the Chroma
    collection has no chunks for the topic the ``fallback`` setting
    decides:

    - ``"template"``: warn, mark the bundle as fallback, allow approval
      via the offline / template path. Useful for CI and local dev
      without an ingested knowledge base.
    - ``"fail"``: refuse to generate the record.
    """

    enabled: bool = False
    k: int = Field(default=8, ge=1, le=50)
    fallback: Literal["template", "fail"] = "template"


class SyntheticConfig(BaseModel):
    default_complexity: ComplexityProfile = ComplexityProfile.MODERATE
    default_modalities: list[Modality] = [
        Modality.STRUCTURED_EHR,
        Modality.CLINICAL_TEXT,
        Modality.LABS,
        Modality.VITALS,
    ]
    validation_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    clinical_text_backend: str = "deterministic"
    clinical_text_noise_profile: Literal["standard", "message", "ocr", "heavy"] = "standard"
    clinical_text_model_profile: str | None = None
    clinical_text_command: list[str] | None = None
    imaging_backend: str = "placeholder"
    imaging_model_profile: str | None = None
    diffusers_model_id: str = "stabilityai/stable-diffusion-2-1"
    imaging_command: list[str] | None = None
    time_series_backend: str = "deterministic"
    time_series_model_profile: str | None = None
    time_series_command: list[str] | None = None
    synthea_executable: str | None = None
    image_output_dir: str = "./data/images"
    export_formats: list[ExportFormat] = Field(
        default_factory=lambda: [
            ExportFormat.RAW_JSONL,
            ExportFormat.SFT_JSONL,
            ExportFormat.NOTE_FACT_SFT_JSONL,
            ExportFormat.CLINICAL_OBSERVATION_JSONL,
            ExportFormat.MEDICATION_RECONCILIATION_JSONL,
            ExportFormat.CHAT_JSONL,
            ExportFormat.TOOL_CALL_JSONL,
            ExportFormat.MULTIMODAL_JSONL,
            ExportFormat.TIME_SERIES_JSONL,
            ExportFormat.DPO_JSONL,
            ExportFormat.RL_JSONL,
            ExportFormat.FHIR_NDJSON,
            ExportFormat.PARQUET,
        ]
    )
    max_api_generation_count: int = Field(default=100, ge=1)
    max_api_returned_records: int = Field(default=25, ge=0)
    grounding: GroundingConfig = GroundingConfig()
    judge: JudgeConfig = JudgeConfig()


class AppConfig(BaseModel):
    ingestion: IngestionConfig = IngestionConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    storage: StorageConfig = StorageConfig()
    api: ApiConfig = ApiConfig()
    llm: LlmConfig = LlmConfig()
    synthetic: SyntheticConfig = SyntheticConfig()
