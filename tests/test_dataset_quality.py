from casecrawler.models.synthetic import (
    ComplexityProfile,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
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
