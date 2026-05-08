import pytest
from pydantic import ValidationError

from casecrawler.models.dataset import (
    DatasetManifest,
    ExportFormat,
    GenerationRequest,
    HumanReviewDecision,
    HumanReviewStatus,
)
from casecrawler.models.synthetic import ComplexityProfile, Modality


def test_generation_request_defaults():
    req = GenerationRequest(topic="sepsis", count=25)

    assert req.count == 25
    assert req.complexity == ComplexityProfile.MODERATE
    assert Modality.CLINICAL_TEXT in req.modalities
    assert Modality.LABS in req.modalities
    assert req.imaging_backend is None
    assert req.imaging_model_profile is None


def test_generation_request_accepts_imaging_model_overrides():
    req = GenerationRequest(
        topic="pneumonia",
        modalities=[Modality.IMAGING],
        imaging_backend="external",
        imaging_model_profile="cxr_pneumonia_dreambooth",
        diffusers_model_id="hf/test-cxr",
        imaging_command=["hf-image-sample"],
    )

    assert req.imaging_backend == "external"
    assert req.imaging_model_profile == "cxr_pneumonia_dreambooth"
    assert req.diffusers_model_id == "hf/test-cxr"
    assert req.imaging_command == ["hf-image-sample"]


def test_generation_request_accepts_clinical_text_model_overrides():
    req = GenerationRequest(
        topic="sepsis",
        modalities=[Modality.CLINICAL_TEXT],
        clinical_text_backend="llm",
        llm_provider="ollama",
        llm_model="medgemma-local",
        ollama_base_url="http://localhost:11434",
    )

    assert req.clinical_text_backend == "llm"
    assert req.llm_provider == "ollama"
    assert req.llm_model == "medgemma-local"
    assert req.ollama_base_url == "http://localhost:11434"


def test_generation_request_accepts_time_series_model_overrides():
    req = GenerationRequest(
        topic="sepsis",
        modalities=[Modality.TIME_SERIES],
        time_series_backend="external",
        time_series_model_profile="timediff",
        time_series_command=["timediff-sample", "--checkpoint", "local.pt"],
    )

    assert req.time_series_backend == "external"
    assert req.time_series_model_profile == "timediff"
    assert req.time_series_command == ["timediff-sample", "--checkpoint", "local.pt"]


def test_generation_request_rejects_invalid_validation_threshold():
    with pytest.raises(ValidationError):
        GenerationRequest(topic="sepsis", validation_threshold=1.5)


def test_dataset_manifest_records_validation_summary():
    manifest = DatasetManifest(
        dataset_id="ds-1",
        name="sepsis-multimodal-v1",
        topic="sepsis",
        requested_count=100,
        generated_count=97,
        approved_count=91,
        modalities=[Modality.CLINICAL_TEXT, Modality.LABS],
        export_formats=[ExportFormat.SFT_JSONL, ExportFormat.PARQUET],
        created_at="2026-05-06T12:00:00",
    )

    assert manifest.approved_count == 91
    assert ExportFormat.SFT_JSONL in manifest.export_formats


def test_human_review_decision_defaults_to_timestamped_human_gate():
    decision = HumanReviewDecision(
        status=HumanReviewStatus.NEEDS_REVISION,
        notes=["Temporal contradiction in labs."],
    )

    assert decision.reviewer == "human"
    assert decision.reviewed_at
    assert decision.notes == ["Temporal contradiction in labs."]
