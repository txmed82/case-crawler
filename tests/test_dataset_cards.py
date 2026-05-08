import struct
import zlib

from casecrawler.export.cards import build_dataset_card, build_model_card
from casecrawler.models.dataset import DatasetManifest, ExportFormat
from casecrawler.models.synthetic import (
    ClinicalDocument,
    ComplexityProfile,
    Modality,
    Provenance,
    Code,
    Encounter,
    ImagingAsset,
    LabObservation,
    MedicationStatement,
    SyntheticPatient,
    SyntheticRecord,
    TimeSeriesChannel,
    TimeSeriesPoint,
    VitalObservation,
    ValidationReport,
)


def test_build_dataset_card_includes_validation_and_use_limits(tmp_path):
    record = _record(tmp_path)
    manifest = _manifest()

    card = build_dataset_card(manifest, [record])

    assert "# Dataset Card: sepsis-synthetic" in card
    assert "- clinical_text: 1" in card
    assert "- Mean clinical consistency score: 0.900" in card
    assert "- Mean modality alignment score: 0.850" in card
    assert "## Multimodal Release Readiness" in card
    assert "- Ready: False" in card
    assert "- nursing_notes: False" in card
    assert "- radiology_reports: False" in card
    assert "- benchmark_reference: True" in card
    assert "Records are synthetic" in card
    assert "- sft_jsonl" in card
    assert "- imaging_backend=diffusers: 1" in card
    assert "benchmark_passed=True" in card
    assert "reference=ds-ref" in card
    assert "## Recommended Benchmark Plan" in card
    assert "synthchex_75k" in card
    assert "## Task-Specific Export References" in card
    assert "clinical_observation_jsonl: synthea_fhir, clinical_notes_to_fhir" in card
    assert "medication_reconciliation_jsonl: synthea_fhir, medsynth_dialogue_note" in card
    assert "## Extracted Fact Targets" in card
    assert "- lab_values: 1" in card
    assert "- medications: 1" in card
    assert "## Procedures" in card
    assert "- Central venous catheter placement: 1" in card
    assert "## Clinical Units" in card
    assert "- lab:mmol/L: 1" in card
    assert "- vital:/min: 1" in card
    assert "## Medication Regimens" in card
    assert "- dose=1 g: 1" in card
    assert "- frequency=daily: 1" in card
    assert "- route=IV: 1" in card
    assert "- status=active: 1" in card
    assert "## Diagnosis Coding Signals" in card
    assert "- ICD-9-CM: 2" in card
    assert "- ICD-9-CM:401.9: 1" in card
    assert "- ICD-9-CM:428.0: 1" in card
    assert "## PHI Annotation Signals" in card
    assert "- AGE: 1" in card
    assert "- NAME: 1" in card


def test_build_model_card_documents_generator_and_validation_gates(tmp_path):
    record = _record(tmp_path)
    manifest = _manifest()

    card = build_model_card(manifest, [record])

    assert "# Model Card: sepsis-synthetic synthetic generation pipeline" in card
    assert "- unit-test-generator: 1" in card
    assert "- unit-test-model: 1" in card
    assert "- imaging_model_profile=cxr_pneumonia_dreambooth: 1" in card
    assert "## Imaging Model Policies" in card
    assert (
        "- profile=cxr_pneumonia_dreambooth license=openrail++ "
        "gated=False use_policy=openrail_review_outputs_before_release: 1"
    ) in card
    assert "## Time-Series Backends" in card
    assert "- external:timediff-sample: 1" in card
    assert "## Time-Series Units" in card
    assert "- /min: 1" in card
    assert "## Imaging Dimensions" in card
    assert "- Mean width: 96.0 px" in card
    assert "- Mean height: 64.0 px" in card
    assert "## Procedure Coverage" in card
    assert "- Central venous catheter placement: 1" in card
    assert "## Diagnosis Coding Signals" in card
    assert "- ICD-9-CM: 2" in card
    assert "## PHI Annotation Signals" in card
    assert "- NAME: 1" in card
    assert "PHI-like privacy scanning" in card
    assert "## Multimodal Release Readiness" in card
    assert "- Ready: False" in card
    assert "nursing_notes" in card


def _record(tmp_path) -> SyntheticRecord:
    image_path = tmp_path / "xray.png"
    image_path.write_bytes(_png_bytes(width=96, height=64))
    return SyntheticRecord(
        record_id="rec-1",
        dataset_id="ds-1",
        topic="sepsis",
        complexity=ComplexityProfile.MODERATE,
        modalities=[Modality.CLINICAL_TEXT],
        patient=SyntheticPatient(patient_id="pat-1", age=64, sex="male"),
        encounters=[
            Encounter(
                encounter_id="enc-1",
                start="2026-05-06T10:00:00",
                setting="emergency_department",
                reason="sepsis",
                diagnoses=[
                    Code(system="ICD-9-CM", code="428.0", display="Heart failure"),
                    Code(system="ICD-9-CM", code="401.9", display="Hypertension"),
                ],
                procedures=[
                    Code(
                        system="http://snomed.info/sct",
                        code="232717009",
                        display="Central venous catheter placement",
                    )
                ],
            )
        ],
        labs=[
            LabObservation(
                name="Lactate",
                value=3.4,
                unit="mmol/L",
                reference_low=0.5,
                reference_high=2.0,
                flag="H",
                effective_time="2026-05-06T10:00:00",
            )
        ],
        vitals=[
            VitalObservation(
                name="HR",
                value=110,
                unit="/min",
                effective_time="2026-05-06T10:00:00",
            )
        ],
        medication_history=[
            MedicationStatement(
                name="Ceftriaxone",
                dose="1 g",
                route="IV",
                frequency="daily",
                status="active",
            )
        ],
        documents=[
            ClinicalDocument(
                document_id="doc-1",
                note_type="progress_note",
                author_role="physician",
                timestamp="2026-05-06T10:00:00",
                clean_text="Progress note with extracted facts.",
                extracted_facts={
                    "lab_values": [{"name": "Lactate", "value": 3.4, "unit": "mmol/L"}],
                    "medications": ["Ceftriaxone"],
                    "phi_annotations": [
                        {
                            "entity_type": "NAME",
                            "text": "Smith",
                            "start": 0,
                            "end": 5,
                        },
                        {
                            "entity_type": "AGE",
                            "text": "64-year-old",
                            "start": 6,
                            "end": 17,
                        },
                    ],
                    "empty_target": [],
                },
            )
        ],
        time_series=[
            TimeSeriesChannel(
                name="heart_rate",
                unit="/min",
                generation_backend="external:timediff-sample",
                sampling_rate_hz=1.0,
                points=[
                    TimeSeriesPoint(
                        timestamp="2026-05-06T10:00:00",
                        values={"value": 110},
                    )
                ],
            )
        ],
        imaging=[
            ImagingAsset(
                image_id="img-1",
                modality="XR",
                body_region="chest",
                prompt="portable chest x-ray with sepsis evaluation",
                file_path=str(image_path),
                report_text="Portable chest radiograph with no focal opacity.",
                labels=[],
                generation_backend="diffusers:cxr_pneumonia_dreambooth",
            )
        ],
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
            modality_alignment_score=0.85,
            approved=True,
        ),
        metadata={
            "generation_overrides": {
                "imaging_backend": "diffusers",
                "imaging_model_profile": "cxr_pneumonia_dreambooth",
            },
            "imaging_model_policy": {
                "profile": "cxr_pneumonia_dreambooth",
                "model_id": "chimbiwide/cxr-pneumonia-dreambooth",
                "license": "openrail++",
                "gated": False,
                "use_policy": "openrail_review_outputs_before_release",
            },
        },
    )


def _png_bytes(*, width: int, height: int) -> bytes:
    raw = b"".join(b"\x00" + (b"\x80" * width) for _ in range(height))
    chunks = [
        b"\x89PNG\r\n\x1a\n",
        _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)),
        _png_chunk(b"IDAT", zlib.compress(raw)),
        _png_chunk(b"IEND", b""),
    ]
    return b"".join(chunks)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
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
        metadata={
            "primary_recipe": "radiology_cxr_report",
            "recommended_reference_keys": [
                "synthchex_75k",
                "rexgradient_160k",
            ],
            "benchmark_thresholds": {
                "min_overall_score": 0.7,
                "min_metric_score": 0.45,
            },
            "task_export_reference_keys": {
                "clinical_observation_jsonl": [
                    "synthea_fhir",
                    "clinical_notes_to_fhir",
                ],
                "medication_reconciliation_jsonl": [
                    "synthea_fhir",
                    "medsynth_dialogue_note",
                ],
            },
            "latest_exports": [
                {
                    "dataset_id": "ds-1",
                    "export_format": "sft_jsonl",
                    "file_path": "train.jsonl",
                    "record_count": 1,
                    "created_at": "2026-05-06T11:00:00",
                    "metadata": {
                        "benchmark_passed": True,
                        "benchmark_reference_dataset_id": "ds-ref",
                    },
                }
            ]
        },
    )
