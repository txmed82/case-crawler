from casecrawler.models.synthetic import (
    ClinicalDocument,
    ComplexityProfile,
    LabObservation,
    MedicationStatement,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
    ValidationReport,
    VitalObservation,
)
from casecrawler.validation.benchmark import DatasetBenchmark, profile_records


def _record(
    record_id: str,
    dataset_id: str,
    *,
    age: int = 60,
    sex: str = "male",
    note_type: str = "progress_note",
    topic: str = "sepsis",
) -> SyntheticRecord:
    return SyntheticRecord(
        record_id=record_id,
        dataset_id=dataset_id,
        topic=topic,
        complexity=ComplexityProfile.MODERATE,
        modalities=[Modality.CLINICAL_TEXT, Modality.LABS, Modality.VITALS],
        patient=SyntheticPatient(patient_id=f"pat-{record_id}", age=age, sex=sex),
        encounters=[],
        labs=[
            LabObservation(
                name="WBC",
                value=12.0,
                unit="K/uL",
                reference_low=4.5,
                reference_high=11.0,
                flag="H",
                effective_time="2026-01-01T00:00:00",
            )
        ],
        vitals=[
            VitalObservation(
                name="HR",
                value=110,
                unit="/min",
                effective_time="2026-01-01T00:00:00",
            )
        ],
        medication_history=[
            MedicationStatement(name="Ceftriaxone", status="active")
        ],
        documents=[
            ClinicalDocument(
                document_id=f"doc-{record_id}",
                note_type=note_type,
                author_role="physician",
                timestamp="2026-01-01T00:00:00",
                clean_text="Synthetic clinical note with labs and vitals.",
            )
        ],
        provenance=Provenance(generator="unit-test", created_at="2026-01-01T00:00:00"),
        validation=ValidationReport(
            schema_score=1.0,
            clinical_consistency_score=1.0,
            privacy_score=1.0,
            utility_score=1.0,
            approved=True,
        ),
    )


def test_profile_records_summarizes_multimodal_cohort():
    profile = profile_records([
        _record("rec-1", "ds-gen", age=60, sex="male"),
        _record("rec-2", "ds-gen", age=70, sex="female"),
    ])

    assert profile.dataset_id == "ds-gen"
    assert profile.record_count == 2
    assert profile.mean_age == 65
    assert profile.sex_counts == {"female": 1, "male": 1}
    assert profile.lab_name_counts == {"WBC": 2}
    assert profile.approved_rate == 1.0


def test_dataset_benchmark_compares_generated_to_reference_records():
    generated = [
        _record("rec-1", "ds-gen", age=60, sex="male"),
        _record("rec-2", "ds-gen", age=62, sex="female"),
    ]
    reference = [
        _record("ref-1", "ds-ref", age=61, sex="male"),
        _record("ref-2", "ds-ref", age=63, sex="female"),
    ]

    report = DatasetBenchmark().compare(generated, reference)

    assert report.generated_dataset_id == "ds-gen"
    assert report.reference_dataset_id == "ds-ref"
    assert report.overall_score > 0.8
    assert {metric.name for metric in report.metrics} >= {
        "modality_overlap",
        "mean_age",
        "lab_name_overlap",
        "vital_name_overlap",
        "medication_name_overlap",
    }


def test_dataset_benchmark_flags_modality_mismatch():
    generated = [_record("rec-1", "ds-gen")]
    reference = [
        _record("ref-1", "ds-ref").model_copy(update={"modalities": [Modality.IMAGING]})
    ]

    report = DatasetBenchmark().compare(generated, reference)
    modality_metric = next(
        metric for metric in report.metrics if metric.name == "modality_overlap"
    )

    assert modality_metric.score == 0.0
    assert any("modality_overlap" in warning for warning in report.warnings)
