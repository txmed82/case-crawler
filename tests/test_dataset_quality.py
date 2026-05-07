from casecrawler.models.synthetic import (
    ClinicalDocument,
    ComplexityProfile,
    ImagingAsset,
    LabObservation,
    Modality,
    MedicationStatement,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
    TimeSeriesChannel,
    TimeSeriesPoint,
    ValidationIssue,
    ValidationReport,
)
from casecrawler.validation.quality import build_dataset_quality_report


def _record(record_id: str, *, approved: bool = True, issues=None) -> SyntheticRecord:
    return SyntheticRecord(
        record_id=record_id,
        dataset_id="ds-quality",
        topic="sepsis",
        complexity=ComplexityProfile.MODERATE,
        modalities=[Modality.CLINICAL_TEXT, Modality.LABS],
        patient=SyntheticPatient(patient_id=f"pat-{record_id}", age=64, sex="male"),
        encounters=[],
        documents=[
            ClinicalDocument(
                document_id=f"doc-{record_id}-ed",
                note_type="ed_note",
                author_role="physician",
                timestamp="2026-01-01T00:00:00",
                clean_text="ED note.",
            ),
            ClinicalDocument(
                document_id=f"doc-{record_id}-progress",
                note_type="progress_note",
                author_role="physician",
                timestamp="2026-01-01T00:00:00",
                clean_text="Progress note.",
            ),
            ClinicalDocument(
                document_id=f"doc-{record_id}-nursing",
                note_type="nursing_note",
                author_role="nurse",
                timestamp="2026-01-01T00:00:00",
                clean_text="Nursing note.",
            ),
            ClinicalDocument(
                document_id=f"doc-{record_id}-discharge",
                note_type="discharge_summary",
                author_role="physician",
                timestamp="2026-01-01T00:00:00",
                clean_text="Discharge summary.",
            )
        ],
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
        provenance=Provenance(generator="unit-test", created_at="2026-01-01T00:00:00"),
        validation=ValidationReport(
            schema_score=1.0,
            clinical_consistency_score=1.0 if approved else 0.5,
            privacy_score=1.0,
            utility_score=1.0,
            approved=approved,
            issues=issues or [],
        ),
    )


def test_quality_report_marks_fully_approved_dataset_export_ready():
    report = build_dataset_quality_report(
        "ds-quality",
        [_record("rec-1"), _record("rec-2")],
    )

    assert report.export_ready is True
    assert report.approval_rate == 1.0
    assert report.modality_counts == {"clinical_text": 2, "labs": 2}
    assert report.recommendations == []


def test_quality_report_requires_declared_modality_artifacts():
    record = _record("rec-1").model_copy(
        update={
            "modalities": [
                Modality.CLINICAL_TEXT,
                Modality.LABS,
                Modality.VITALS,
                Modality.TIME_SERIES,
                Modality.IMAGING,
            ],
            "documents": [
                ClinicalDocument(
                    document_id="doc-1",
                    note_type="ed_note",
                    author_role="physician",
                    timestamp="2026-01-01T00:00:00",
                    clean_text="ED note.",
                    messy_text="ed note",
                )
            ],
            "labs": [
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
            "vitals": [],
            "time_series": [],
            "imaging": [],
        }
    )

    report = build_dataset_quality_report("ds-quality", [record])

    assert report.export_ready is False
    assert report.artifact_counts["documents"] == 1
    assert report.artifact_counts["labs"] == 1
    assert report.artifact_counts["vitals"] == 0
    assert report.artifact_counts["time_series_channels"] == 0
    assert report.artifact_counts["imaging_assets"] == 0
    assert "vitals.missing_artifacts" in report.issue_counts_by_field
    assert "time_series.missing_artifacts" in report.issue_counts_by_field
    assert "imaging.missing_artifacts" in report.issue_counts_by_field
    assert any("missing modality artifacts" in item for item in report.recommendations)


def test_quality_report_requires_expected_clinical_document_types():
    record = _record("rec-1").model_copy(
        update={
            "modalities": [Modality.CLINICAL_TEXT, Modality.IMAGING],
            "documents": [
                ClinicalDocument(
                    document_id="doc-ed",
                    note_type="ed_note",
                    author_role="physician",
                    timestamp="2026-01-01T00:00:00",
                    clean_text="ED physician note.",
                    messy_text="ed note",
                )
            ],
            "imaging": [
                ImagingAsset(
                    image_id="img-1",
                    modality="XR",
                    body_region="chest",
                    prompt="portable chest x-ray pneumonia",
                    report_text="Pneumonia.",
                    generation_backend="placeholder",
                )
            ],
        }
    )

    report = build_dataset_quality_report("ds-quality", [record])

    assert report.export_ready is False
    assert report.issue_counts_by_field["documents.progress_note.missing"] == 1
    assert report.issue_counts_by_field["documents.nursing_note.missing"] == 1
    assert report.issue_counts_by_field["documents.discharge_summary.missing"] == 1
    assert report.issue_counts_by_field["documents.radiology_report.missing"] == 1
    assert any("expected clinical document types" in item for item in report.recommendations)


def test_quality_report_requires_expected_clinical_document_author_roles():
    record = _record("rec-1").model_copy(
        update={
            "modalities": [Modality.CLINICAL_TEXT, Modality.IMAGING],
            "documents": [
                ClinicalDocument(
                    document_id="doc-ed",
                    note_type="ed_note",
                    author_role="nurse",
                    timestamp="2026-01-01T00:00:00",
                    clean_text="ED note.",
                ),
                ClinicalDocument(
                    document_id="doc-progress",
                    note_type="progress_note",
                    author_role="nurse",
                    timestamp="2026-01-01T00:00:00",
                    clean_text="Progress note.",
                ),
                ClinicalDocument(
                    document_id="doc-nursing",
                    note_type="nursing_note",
                    author_role="physician",
                    timestamp="2026-01-01T00:00:00",
                    clean_text="Nursing note.",
                ),
                ClinicalDocument(
                    document_id="doc-discharge",
                    note_type="discharge_summary",
                    author_role="nurse",
                    timestamp="2026-01-01T00:00:00",
                    clean_text="Discharge summary.",
                ),
                ClinicalDocument(
                    document_id="doc-rad",
                    note_type="radiology_report",
                    author_role="physician",
                    timestamp="2026-01-01T00:00:00",
                    clean_text="Radiology report.",
                ),
            ],
            "imaging": [
                ImagingAsset(
                    image_id="img-1",
                    modality="XR",
                    body_region="chest",
                    prompt="portable chest x-ray pneumonia",
                    report_text="Pneumonia.",
                    generation_backend="placeholder",
                )
            ],
        }
    )

    report = build_dataset_quality_report("ds-quality", [record])

    assert report.export_ready is False
    assert report.issue_counts_by_field["documents.ed_note.author_role"] == 1
    assert report.issue_counts_by_field["documents.progress_note.author_role"] == 1
    assert report.issue_counts_by_field["documents.nursing_note.author_role"] == 1
    assert report.issue_counts_by_field["documents.discharge_summary.author_role"] == 1
    assert report.issue_counts_by_field["documents.radiology_report.author_role"] == 1
    assert any("expected clinical document author roles" in item for item in report.recommendations)


def test_quality_report_summarizes_multimodal_training_artifacts():
    record = _record("rec-1").model_copy(
        update={
            "modalities": [
                Modality.CLINICAL_TEXT,
                Modality.LABS,
                Modality.VITALS,
                Modality.TIME_SERIES,
                Modality.IMAGING,
            ],
            "documents": [
                ClinicalDocument(
                    document_id="doc-ed",
                    note_type="ed_note",
                    author_role="physician",
                    timestamp="2026-01-01T00:00:00",
                    clean_text="ED note.",
                    messy_text="ed note",
                ),
                ClinicalDocument(
                    document_id="doc-rad",
                    note_type="radiology_report",
                    author_role="radiologist",
                    timestamp="2026-01-01T00:00:00",
                    clean_text="Radiology report.",
                ),
            ],
            "labs": [
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
            "vitals": [],
            "medication_history": [
                MedicationStatement(name="Ceftriaxone", route="IV", status="active")
            ],
            "time_series": [
                TimeSeriesChannel(
                    name="heart_rate",
                    unit="/min",
                    points=[
                        TimeSeriesPoint(
                            timestamp="2026-01-01T00:00:00",
                            values={"value": 100},
                        )
                    ],
                ),
                TimeSeriesChannel(
                    name="ecg_lead_ii",
                    unit="mV",
                    sampling_rate_hz=125,
                    points=[
                        TimeSeriesPoint(
                            timestamp="2026-01-01T00:00:00",
                            values={"millivolts": 0.1, "phase": 0.1},
                        )
                    ],
                ),
            ],
            "imaging": [
                ImagingAsset(
                    image_id="img-1",
                    modality="XR",
                    body_region="chest",
                    prompt="portable chest x-ray pneumonia",
                    report_text="Pneumonia.",
                    generation_backend="placeholder",
                )
            ],
        }
    )

    report = build_dataset_quality_report("ds-quality", [record])

    assert report.artifact_counts["documents"] == 2
    assert report.artifact_counts["messy_documents"] == 1
    assert report.artifact_counts["medications"] == 1
    assert report.artifact_counts["time_series_waveform_channels"] == 1
    assert report.artifact_counts["imaging_assets"] == 1
    assert report.note_type_counts == {"ed_note": 1, "radiology_report": 1}
    assert report.export_ready is False
    assert "vitals.missing_artifacts" in report.issue_counts_by_field


def test_quality_report_counts_blocking_issues_and_recommendations():
    issue = ValidationIssue(
        severity="error",
        modality=Modality.LABS,
        field="labs.flag",
        message="Lab flag missing.",
    )
    report = build_dataset_quality_report(
        "ds-quality",
        [_record("rec-1"), _record("rec-2", approved=False, issues=[issue])],
    )

    assert report.export_ready is False
    assert report.approved_count == 1
    assert report.blocking_issue_count == 1
    assert report.issue_counts_by_field == {"labs.flag": 1}
    assert "Resolve blocking validation issues before marking the dataset ready." in report.recommendations


def test_quality_report_treats_missing_validation_as_blocking():
    missing = _record("rec-1").model_copy(update={"validation": None})

    report = build_dataset_quality_report("ds-quality", [missing])

    assert report.export_ready is False
    assert report.blocking_issue_count == 1
    assert report.issue_counts_by_field == {"validation.missing": 1}
