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
        imaging_backend="diffusers",
        imaging_model_profile="cxr_pneumonia_dreambooth",
        diffusers_model_id="hf/test-cxr",
    )

    assert req.imaging_backend == "diffusers"
    assert req.imaging_model_profile == "cxr_pneumonia_dreambooth"
    assert req.diffusers_model_id == "hf/test-cxr"


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
