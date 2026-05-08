from casecrawler.export.cards import build_dataset_card, build_model_card
from casecrawler.models.dataset import DatasetManifest, ExportFormat
from casecrawler.models.synthetic import (
    ComplexityProfile,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
    ValidationReport,
)


def test_build_dataset_card_includes_validation_and_use_limits():
    record = _record()
    manifest = _manifest()

    card = build_dataset_card(manifest, [record])

    assert "# Dataset Card: sepsis-synthetic" in card
    assert "- clinical_text: 1" in card
    assert "- Mean clinical consistency score: 0.900" in card
    assert "Records are synthetic" in card
    assert "- sft_jsonl" in card
    assert "- imaging_backend=diffusers: 1" in card


def test_build_model_card_documents_generator_and_validation_gates():
    record = _record()
    manifest = _manifest()

    card = build_model_card(manifest, [record])

    assert "# Model Card: sepsis-synthetic synthetic generation pipeline" in card
    assert "- unit-test-generator: 1" in card
    assert "- unit-test-model: 1" in card
    assert "- imaging_model_profile=cxr_pneumonia_dreambooth: 1" in card
    assert "PHI-like privacy scanning" in card


def _record() -> SyntheticRecord:
    return SyntheticRecord(
        record_id="rec-1",
        dataset_id="ds-1",
        topic="sepsis",
        complexity=ComplexityProfile.MODERATE,
        modalities=[Modality.CLINICAL_TEXT],
        patient=SyntheticPatient(patient_id="pat-1", age=64, sex="male"),
        encounters=[],
        provenance=Provenance(
            generator="unit-test-generator",
            model="unit-test-model",
            created_at="2026-05-06T10:00:00",
        ),
        validation=ValidationReport(
            schema_score=1.0,
            clinical_consistency_score=0.9,
            privacy_score=1.0,
            utility_score=1.0,
            approved=True,
        ),
        metadata={
            "generation_overrides": {
                "imaging_backend": "diffusers",
                "imaging_model_profile": "cxr_pneumonia_dreambooth",
            }
        },
    )


def _manifest() -> DatasetManifest:
    return DatasetManifest(
        dataset_id="ds-1",
        name="sepsis-synthetic",
        topic="sepsis",
        requested_count=1,
        generated_count=1,
        approved_count=1,
        modalities=[Modality.CLINICAL_TEXT],
        export_formats=[ExportFormat.SFT_JSONL],
        created_at="2026-05-06T10:00:00",
    )
