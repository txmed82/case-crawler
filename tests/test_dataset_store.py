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
        metadata={
            "benchmark_reference_dataset_id": "ds-ref",
            "benchmark_passed": True,
        },
    )
    updated_manifest = store.get_manifest("ds-1")
    export_manifests = store.list_export_manifests(dataset_id="ds-1")

    assert manifest.dataset_id == "ds-1"
    assert manifest.generated_count == 1
    assert manifest.modalities == [Modality.CLINICAL_TEXT, Modality.LABS]
    assert manifest.export_formats == list(ExportFormat)
    assert export_manifest.dataset_id == "ds-1"
    assert export_manifest.record_count == 1
    assert export_manifests[0].metadata["benchmark_reference_dataset_id"] == "ds-ref"
    assert updated_manifest.metadata["latest_exports"][0]["metadata"]["benchmark_passed"] is True


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


def test_dataset_store_manifest_includes_recipe_benchmark_plan(tmp_path):
    store = DatasetStore(db_path=str(tmp_path / "datasets.db"))
    record = SyntheticRecord(
        record_id="rec-1",
        dataset_id="ds-1",
        topic="pneumonia",
        complexity=ComplexityProfile.MODERATE,
        modalities=[Modality.CLINICAL_TEXT, Modality.IMAGING],
        patient=SyntheticPatient(patient_id="pat-1", age=64, sex="female"),
        encounters=[],
        provenance=Provenance(
            generator="unit-test",
            created_at="2026-05-06T10:00:00",
        ),
        metadata={
            "generation_overrides": {
                "recipe": "radiology_cxr_report",
            }
        },
    )

    store.save_record(record)
    manifest = store.get_manifest("ds-1")

    assert manifest.metadata["primary_recipe"] == "radiology_cxr_report"
    assert manifest.metadata["generation_recipes"] == {"radiology_cxr_report": 1}
    assert "synthchex_75k" in manifest.metadata["recommended_reference_keys"]
    assert manifest.metadata["benchmark_thresholds"]["min_overall_score"] == 0.7


def test_dataset_store_manifest_derives_task_export_benchmark_plan(tmp_path):
    store = DatasetStore(db_path=str(tmp_path / "datasets.db"))
    record = SyntheticRecord(
        record_id="rec-1",
        dataset_id="ds-1",
        topic="sepsis",
        complexity=ComplexityProfile.MODERATE,
        modalities=[Modality.STRUCTURED_EHR, Modality.LABS, Modality.VITALS],
        patient=SyntheticPatient(patient_id="pat-1", age=64, sex="female"),
        encounters=[],
        provenance=Provenance(
            generator="unit-test",
            created_at="2026-05-06T10:00:00",
        ),
        metadata={
            "requested_export_formats": [
                "clinical_observation_jsonl",
                "medication_reconciliation_jsonl",
            ]
        },
    )

    store.save_record(record)
    manifest = store.get_manifest("ds-1")

    assert manifest.metadata["recommended_reference_keys"] == [
        "synthea_fhir",
        "clinical_notes_to_fhir",
        "medsynth_dialogue_note",
    ]
    assert manifest.metadata["benchmark_thresholds"] == {
        "min_overall_score": 0.75,
        "min_metric_score": 0.5,
    }


def test_dataset_store_manifest_includes_reference_keys(tmp_path):
    store = DatasetStore(db_path=str(tmp_path / "datasets.db"))
    record = SyntheticRecord(
        record_id="rec-ref",
        dataset_id="ds-ref",
        topic="clinical_note",
        complexity=ComplexityProfile.MODERATE,
        modalities=[Modality.CLINICAL_TEXT],
        patient=SyntheticPatient(patient_id="pat-1", age=64, sex="female"),
        encounters=[],
        provenance=Provenance(
            generator="huggingface-reference-import",
            created_at="2026-05-06T10:00:00",
        ),
        metadata={
            "reference_key": "asclepius",
            "reference_dataset": "starmpcc/Asclepius-Synthetic-Clinical-Notes",
        },
    )

    store.save_record(record)
    manifest = store.get_manifest("ds-ref")

    assert manifest.metadata["primary_reference_key"] == "asclepius"
    assert manifest.metadata["reference_keys"] == {"asclepius": 1}
    assert (
        manifest.metadata["primary_reference_dataset"]
        == "starmpcc/Asclepius-Synthetic-Clinical-Notes"
    )


def test_dataset_store_manifest_includes_technetium_reference_keys(tmp_path):
    store = DatasetStore(db_path=str(tmp_path / "datasets.db"))
    record = SyntheticRecord(
        record_id="rec-ref",
        dataset_id="ds-ref-technetium",
        topic="clinical_deidentification_icd_coding",
        complexity=ComplexityProfile.MODERATE,
        modalities=[Modality.STRUCTURED_EHR, Modality.CLINICAL_TEXT],
        patient=SyntheticPatient(patient_id="pat-1", age=64, sex="female"),
        encounters=[],
        provenance=Provenance(
            generator="huggingface-reference-import",
            created_at="2026-05-06T10:00:00",
        ),
        metadata={
            "reference_key": "technetium_i",
            "reference_dataset": "temlm-foundation/Technetium-I",
        },
    )

    store.save_record(record)
    manifest = store.get_manifest("ds-ref-technetium")

    assert manifest.metadata["primary_reference_key"] == "technetium_i"
    assert manifest.metadata["primary_reference_dataset"] == (
        "temlm-foundation/Technetium-I"
    )


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
    summary = store.human_review_summary(dataset_id="ds-1")
    assert queue[0].record_id == "rec-review"
    assert queue[0].blocking_issue_count == 1
    assert summary.total_records == 1
    assert summary.pending == 1
    assert summary.blocking_issue_records == 1

    reviewed = store.save_human_review(
        "rec-review",
        HumanReviewDecision(
            status=HumanReviewStatus.APPROVED,
            reviewer="clinical-reviewer",
            notes=["Synthetic contradiction accepted for stress testing."],
        ),
    )

    assert store.effective_approved(reviewed) is True
    reviewed_summary = store.human_review_summary(dataset_id="ds-1")
    assert reviewed_summary.pending == 0
    assert reviewed_summary.approved == 1
    assert store.list_review_queue(dataset_id="ds-1") == []
    assert store.list_records(dataset_id="ds-1", approved=True)[0].record_id == "rec-review"
    assert store.get_manifest("ds-1").approved_count == 1


def test_dataset_store_queues_required_human_review_even_when_validation_passes(tmp_path):
    store = DatasetStore(db_path=str(tmp_path / "datasets.db"))
    record = SyntheticRecord(
        record_id="rec-required-review",
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
            clinical_consistency_score=1.0,
            privacy_score=1.0,
            utility_score=1.0,
            approved=True,
        ),
        metadata={"require_human_review": True},
    )

    store.save_record(record)

    assert store.list_review_queue(dataset_id="ds-1")[0].record_id == (
        "rec-required-review"
    )
