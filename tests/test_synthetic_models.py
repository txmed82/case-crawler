import pytest
from pydantic import ValidationError

from casecrawler.models.synthetic import (
    ClinicalDocument,
    Code,
    ComplexityProfile,
    Encounter,
    LabObservation,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
    ValidationReport,
    VitalObservation,
)


def test_synthetic_record_with_text_labs_and_vitals():
    record = SyntheticRecord(
        record_id="rec-1",
        dataset_id="ds-1",
        topic="heart failure exacerbation",
        complexity=ComplexityProfile.MODERATE,
        modalities=[
            Modality.STRUCTURED_EHR,
            Modality.CLINICAL_TEXT,
            Modality.LABS,
            Modality.VITALS,
        ],
        patient=SyntheticPatient(patient_id="pat-1", age=72, sex="female"),
        encounters=[
            Encounter(
                encounter_id="enc-1",
                start="2026-05-06T08:00:00",
                setting="emergency_department",
                reason="dyspnea",
                diagnoses=[
                    Code(
                        system="ICD-10-CM",
                        code="I50.9",
                        display="Heart failure, unspecified",
                    )
                ],
            )
        ],
        labs=[
            LabObservation(
                name="BNP",
                loinc="30934-4",
                value=1220.0,
                unit="pg/mL",
                reference_low=0.0,
                reference_high=100.0,
                flag="H",
                effective_time="2026-05-06T08:45:00",
            )
        ],
        vitals=[
            VitalObservation(
                name="SpO2",
                value=89.0,
                unit="%",
                effective_time="2026-05-06T08:05:00",
            )
        ],
        documents=[
            ClinicalDocument(
                document_id="doc-1",
                note_type="ed_note",
                author_role="physician",
                timestamp="2026-05-06T09:15:00",
                clean_text="Patient presents with progressive dyspnea and edema.",
                messy_text="pt w/ prog dyspnea + edema, BNP 1220",
            )
        ],
        provenance=Provenance(
            generator="unit-test",
            created_at="2026-05-06T09:30:00",
        ),
        validation=ValidationReport(
            schema_score=1.0,
            clinical_consistency_score=0.95,
            privacy_score=1.0,
            utility_score=0.9,
            approved=True,
        ),
    )

    assert record.record_id == "rec-1"
    assert record.complexity == ComplexityProfile.MODERATE
    assert record.labs[0].flag == "H"
    assert record.validation.approved is True


def test_synthetic_models_reject_unknown_fields():
    with pytest.raises(ValidationError):
        SyntheticPatient(
            patient_id="pat-1",
            age=72,
            sex="female",
            real_name="Jane Doe",
        )
