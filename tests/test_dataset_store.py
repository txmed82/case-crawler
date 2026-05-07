from casecrawler.models.dataset import ExportFormat, HumanReviewDecision, HumanReviewStatus
from casecrawler.models.synthetic import (
    ComplexityProfile,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
    ValidationIssue,
    ValidationReport,
)
from casecrawler.storage.dataset_store import DatasetStore


def test_dataset_store_round_trips_record(tmp_path):
    store = DatasetStore(db_path=str(tmp_path / "datasets.db"))
    record = SyntheticRecord(
        record_id="rec-1",
        dataset_id="ds-1",
        topic="sepsis",
        complexity=ComplexityProfile.MODERATE,
        modalities=[Modality.CLINICAL_TEXT],
        patient=SyntheticPatient(patient_id="pat-1", age=64, sex="male"),
        encounters=[],
        provenance=Provenance(
            generator="unit-test",
            created_at="2026-05-06T10:00:00",
        ),
    )

    store.save_record(record)

    assert store.get_record("rec-1").record_id == "rec-1"
    assert len(store.list_records(dataset_id="ds-1")) == 1


def test_dataset_store_builds_manifest_and_export_manifest(tmp_path):
    store = DatasetStore(db_path=str(tmp_path / "datasets.db"))
    record = SyntheticRecord(
        record_id="rec-1",
        dataset_id="ds-1",
        topic="sepsis",
        complexity=ComplexityProfile.MODERATE,
        modalities=[Modality.CLINICAL_TEXT, Modality.LABS],
        patient=SyntheticPatient(patient_id="pat-1", age=64, sex="male"),
        encounters=[],
        provenance=Provenance(
            generator="unit-test",
            created_at="2026-05-06T10:00:00",
        ),
    )

    store.save_record(record)
    manifest = store.get_manifest("ds-1")
    export_manifest = store.save_export_manifest(
        dataset_id="ds-1",
        export_format="sft_jsonl",
        file_path=str(tmp_path / "export.jsonl"),
        record_count=1,
    )

    assert manifest.dataset_id == "ds-1"
    assert manifest.generated_count == 1
    assert manifest.modalities == [Modality.CLINICAL_TEXT, Modality.LABS]
    assert manifest.export_formats == list(ExportFormat)
    assert export_manifest.dataset_id == "ds-1"
    assert export_manifest.record_count == 1


def test_dataset_store_manifest_prefers_requested_export_formats(tmp_path):
    store = DatasetStore(db_path=str(tmp_path / "datasets.db"))
    record = SyntheticRecord(
        record_id="rec-1",
        dataset_id="ds-1",
        topic="sepsis",
        complexity=ComplexityProfile.MODERATE,
        modalities=[Modality.CLINICAL_TEXT, Modality.LABS],
        patient=SyntheticPatient(patient_id="pat-1", age=64, sex="male"),
        encounters=[],
        provenance=Provenance(
            generator="unit-test",
            created_at="2026-05-06T10:00:00",
        ),
        metadata={
            "requested_export_formats": [
                "sft_jsonl",
                "parquet",
                "unknown_future_format",
            ]
        },
    )

    store.save_record(record)
    manifest = store.get_manifest("ds-1")

    assert manifest.export_formats == [ExportFormat.SFT_JSONL, ExportFormat.PARQUET]


def test_dataset_store_tracks_human_review_queue_and_effective_approval(tmp_path):
    store = DatasetStore(db_path=str(tmp_path / "datasets.db"))
    record = SyntheticRecord(
        record_id="rec-review",
        dataset_id="ds-1",
        topic="sepsis",
        complexity=ComplexityProfile.MODERATE,
        modalities=[Modality.CLINICAL_TEXT],
        patient=SyntheticPatient(patient_id="pat-1", age=64, sex="male"),
        encounters=[],
        provenance=Provenance(
            generator="unit-test",
            created_at="2026-05-06T10:00:00",
        ),
        validation=ValidationReport(
            schema_score=1.0,
            clinical_consistency_score=0.4,
            privacy_score=1.0,
            utility_score=0.6,
            approved=False,
            issues=[
                ValidationIssue(
                    severity="error",
                    modality=Modality.LABS,
                    field="labs",
                    message="Lactate trend contradicts note.",
                )
            ],
        ),
    )

    store.save_record(record)

    queue = store.list_review_queue(dataset_id="ds-1")
    assert queue[0].record_id == "rec-review"
    assert queue[0].blocking_issue_count == 1

    reviewed = store.save_human_review(
        "rec-review",
        HumanReviewDecision(
            status=HumanReviewStatus.APPROVED,
            reviewer="clinical-reviewer",
            notes=["Synthetic contradiction accepted for stress testing."],
        ),
    )

    assert store.effective_approved(reviewed) is True
    assert store.list_review_queue(dataset_id="ds-1") == []
    assert store.list_records(dataset_id="ds-1", approved=True)[0].record_id == "rec-review"
    assert store.get_manifest("ds-1").approved_count == 1
