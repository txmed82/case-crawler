from casecrawler.models.dataset import DatasetManifest, ExportFormat, GenerationRequest
from casecrawler.models.synthetic import ComplexityProfile, Modality


def test_generation_request_defaults():
    req = GenerationRequest(topic="sepsis", count=25)

    assert req.count == 25
    assert req.complexity == ComplexityProfile.MODERATE
    assert Modality.CLINICAL_TEXT in req.modalities
    assert Modality.LABS in req.modalities


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

