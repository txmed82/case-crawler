import struct
import zlib

from casecrawler.export.release_audit import build_objective_coverage_audit
from casecrawler.models.synthetic import (
    ClinicalDocument,
    Code,
    ComplexityProfile,
    Encounter,
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
    VitalObservation,
)
from casecrawler.validation.quality import build_dataset_quality_report, export_profile_blocker


def _record(record_id: str, *, approved: bool = True, issues=None) -> SyntheticRecord:
    return SyntheticRecord(
        record_id=record_id,
        dataset_id="ds-quality",
        topic="sepsis",
        complexity=ComplexityProfile.MODERATE,
        modalities=[Modality.STRUCTURED_EHR, Modality.CLINICAL_TEXT, Modality.LABS],
        patient=SyntheticPatient(
            patient_id=f"pat-{record_id}",
            age=64,
            sex="male",
            demographics={
                "race": "synthetic_white",
                "ethnicity": "synthetic_not_hispanic_or_latino",
                "insurance": "synthetic_medicare",
            },
            social_history={"smoking_status": "former", "housing": "stable"},
        ),
        encounters=[
            Encounter(
                encounter_id=f"enc-{record_id}",
                start="2026-01-01T00:00:00",
                setting="emergency_department",
                reason="sepsis",
                diagnoses=[
                    Code(
                        system="synthetic",
                        code="sepsis",
                        display="sepsis",
                    )
                ],
            )
        ],
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
            ),
            ClinicalDocument(
                document_id=f"doc-{record_id}-lab",
                note_type="lab_report",
                author_role="laboratory",
                timestamp="2026-01-01T00:00:00",
                clean_text="Lab report.",
            ),
            ClinicalDocument(
                document_id=f"doc-{record_id}-mar",
                note_type="medication_administration_record",
                author_role="pharmacist",
                timestamp="2026-01-01T00:00:00",
                clean_text="Medication administration record.",
            ),
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
        medication_history=[
            MedicationStatement(name="Ceftriaxone", route="IV", status="active")
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
    assert report.modality_counts == {"clinical_text": 2, "labs": 2, "structured_ehr": 2}
    assert report.race_counts == {"synthetic_white": 2}
    assert report.ethnicity_counts == {"synthetic_not_hispanic_or_latino": 2}
    assert report.insurance_counts == {"synthetic_medicare": 2}
    assert report.social_history_counts == {
        "housing": {"stable": 2},
        "smoking_status": {"former": 2},
    }
    assert report.longitudinal_record_rate == 0
    assert report.mean_encounter_span_hours is None
    assert report.mean_observations_per_encounter == 1
    assert report.export_profile_readiness["sft_jsonl"]["ready"] is True
    assert report.export_profile_readiness["clinical_observation_jsonl"]["ready"] is True
    assert report.export_profile_readiness["medication_reconciliation_jsonl"]["ready"] is True
    assert report.export_profile_readiness["note_fact_sft_jsonl"]["ready"] is False
    assert report.export_profile_readiness["multimodal_jsonl"]["ready"] is False
    assert "imaging_file_assets" in report.export_profile_readiness["multimodal_jsonl"]["missing"]
    assert report.multimodal_release_ready is False
    assert "vitals" in report.multimodal_release_missing
    assert "time_series" in report.multimodal_release_missing
    assert "radiology_images" in report.multimodal_release_missing
    assert "benchmark_reference" in report.multimodal_release_missing
    assert report.recommendations == []


def test_objective_coverage_blocks_privacy_safety_failures():
    issue = ValidationIssue(
        severity="error",
        modality=Modality.CLINICAL_TEXT,
        field="privacy.memorization_risk",
        message="Generated text contains a long verbatim span from a source reference.",
    )
    report = build_dataset_quality_report(
        "ds-quality",
        [_record("rec-1", approved=False, issues=[issue])],
    )

    audit = build_objective_coverage_audit(
        quality_report=report,
        benchmark_suite={"passed": True, "reference_count": 1},
        manifest={"task_coverage": {"sft": 1}, "audit_artifacts": {}},
    )

    assert audit["criteria"]["privacy_safety"]["satisfied"] is False
    assert audit["criteria"]["privacy_safety"]["evidence"]["privacy_issue_counts"] == {
        "privacy.memorization_risk": 1
    }
    assert "privacy_safety" in audit["missing"]


def test_objective_coverage_requires_cohort_similarity_metrics():
    report = build_dataset_quality_report(
        "ds-quality",
        [_record("rec-1")],
        benchmark_plan={
            "ready": True,
            "recommended_reference_keys": ["synthea_fhir"],
            "resolved_reference_dataset_id": "ds-reference",
            "missing_reference_keys": [],
            "thresholds": {"min_overall_score": 0.1, "min_metric_score": 0.0},
            "task_export_reference_readiness": {},
        },
    )

    missing_metrics = build_objective_coverage_audit(
        quality_report=report,
        benchmark_suite={
            "passed": True,
            "reference_count": 1,
            "mean_overall_score": 0.8,
            "results": [
                {
                    "report": {
                        "metrics": [
                            {"name": "record_count"},
                            {"name": "modality_overlap"},
                        ]
                    }
                }
            ],
        },
        manifest={"task_coverage": {"sft_jsonl": 1}, "audit_artifacts": {}},
    )
    complete_metrics = build_objective_coverage_audit(
        quality_report=report,
        benchmark_suite={
            "passed": True,
            "reference_count": 1,
            "mean_overall_score": 0.8,
            "results": [
                {
                    "report": {
                        "metrics": [
                            {"name": "record_count"},
                            {"name": "mean_age"},
                            {"name": "sex_distribution"},
                            {"name": "race_distribution"},
                            {"name": "ethnicity_distribution"},
                            {"name": "insurance_distribution"},
                            {"name": "social_history_distribution:smoking_status"},
                            {"name": "modality_overlap"},
                        ]
                    }
                }
            ],
        },
        manifest={"task_coverage": {"sft_jsonl": 1}, "audit_artifacts": {}},
    )

    assert missing_metrics["criteria"]["cohort_similarity"]["satisfied"] is False
    assert "mean_age" in missing_metrics["criteria"]["cohort_similarity"]["evidence"][
        "required_metrics"
    ]
    assert "cohort_similarity" in missing_metrics["missing"]
    assert complete_metrics["criteria"]["cohort_similarity"]["satisfied"] is True


def test_objective_coverage_requires_messy_clinical_text():
    report = build_dataset_quality_report("ds-quality", [_record("rec-1")])

    audit = build_objective_coverage_audit(
        quality_report=report,
        benchmark_suite={"passed": True, "reference_count": 1},
        manifest={"task_coverage": {"sft_jsonl": 1}, "audit_artifacts": {}},
    )

    assert audit["criteria"]["messy_clinical_text"]["satisfied"] is False
    assert "messy_clinical_text" in audit["missing"]


def test_quality_report_counts_clinical_text_model_policy():
    record = _record("rec-1").model_copy(
        update={
            "metadata": {
                "clinical_text_model_policy": {
                    "backend": "llm",
                    "provider": "ollama",
                    "model_id": "medgemma-local",
                    "license": "provider_terms",
                    "gated": False,
                    "use_policy": (
                        "synthetic_clinical_text_review_outputs_before_release"
                    ),
                }
            }
        }
    )

    report = build_dataset_quality_report("ds-quality", [record])

    assert report.clinical_text_model_policy_counts == {
        (
            "backend=llm|provider=ollama|model_id=medgemma_local|"
            "gated=false|"
            "use_policy=synthetic_clinical_text_review_outputs_before_release"
        ): len(record.documents)
    }


def test_quality_report_release_coverage_requires_clinical_text_model_policy():
    report = build_dataset_quality_report("ds-quality", [_record("rec-1")])

    assert report.export_ready is True
    assert report.core_artifact_coverage["clinical_text_model_policy"] is False
    assert "clinical_text_model_policy" in report.multimodal_release_missing


def test_quality_report_marks_multimodal_release_ready_with_core_artifacts(tmp_path):
    image_path = tmp_path / "cxr.png"
    image_path.write_bytes(_png_bytes(width=96, height=96))
    base = _record("rec-1")
    record = base.model_copy(
        update={
            "modalities": [
                Modality.STRUCTURED_EHR,
                Modality.CLINICAL_TEXT,
                Modality.LABS,
                Modality.VITALS,
                Modality.TIME_SERIES,
                Modality.IMAGING,
            ],
            "documents": [
                *base.documents,
                ClinicalDocument(
                    document_id="doc-rec-1-vitals",
                    note_type="vital_signs_flowsheet",
                    author_role="nurse",
                    timestamp="2026-01-01T00:05:00",
                    clean_text="Vital signs flowsheet.",
                    messy_text="vs flow",
                ),
                ClinicalDocument(
                    document_id="doc-rec-1-radiology",
                    note_type="radiology_report",
                    author_role="radiologist",
                    timestamp="2026-01-01T00:15:00",
                    clean_text="Portable chest radiograph shows right lower lobe pneumonia.",
                    messy_text="cxr rll pna",
                ),
            ],
            "vitals": [
                VitalObservation(
                    name="HR",
                    value=112,
                    unit="/min",
                    effective_time="2026-01-01T00:00:00",
                )
            ],
            "time_series": [
                TimeSeriesChannel(
                    name="heart_rate",
                    unit="/min",
                    generation_backend="deterministic",
                    points=[
                        TimeSeriesPoint(
                            timestamp="2026-01-01T00:00:00",
                            values={"value": 112},
                        )
                    ],
                )
            ],
            "imaging": [
                ImagingAsset(
                    image_id="img-rec-1",
                    modality="XR",
                    body_region="chest",
                    prompt="portable chest x-ray pneumonia",
                    file_path=str(image_path),
                    report_text="Right lower lobe pneumonia.",
                    generation_backend="diffusers:cxr_pneumonia_dreambooth",
                )
            ],
            "metadata": {
                "clinical_text_model_policy": {
                    "backend": "deterministic",
                    "provider": "casecrawler",
                    "model_id": "casecrawler-template-clinical-documents",
                    "license": "casecrawler",
                    "gated": False,
                    "use_policy": "deterministic_synthetic_templates_validate_outputs",
                },
                "time_series_model_policy": {
                    "profile": "timediff",
                    "model_id": "MuhangTian/TimeDiff",
                    "license": "mit",
                    "gated": False,
                    "use_policy": "wrap_external_sampler_validate_outputs",
                },
                "imaging_model_policy": {
                    "profile": "cxr_pneumonia_dreambooth",
                    "model_id": "chimbiwide/cxr-pneumonia-dreambooth",
                    "license": "openrail++",
                    "gated": False,
                    "use_policy": "openrail_review_outputs_before_release",
                },
                "image_validator_policy": {
                    "profile": "biomedclip",
                    "backend": "open_clip",
                    "model_id": (
                        "hf-hub:microsoft/"
                        "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                    ),
                    "license": "mit",
                    "gated": False,
                    "use_policy": "open_model_validate_image_text_alignment",
                }
            },
            "validation": ValidationReport(
                schema_score=1.0,
                clinical_consistency_score=1.0,
                privacy_score=1.0,
                utility_score=1.0,
                modality_alignment_score=0.91,
                approved=True,
            ),
        }
    )

    report = build_dataset_quality_report(
        "ds-quality",
        [record],
        benchmark_plan={
            "recommended_reference_keys": ["synthchex_75k"],
            "ready": True,
            "resolved_reference_dataset_id": "ds-reference",
            "missing_reference_keys": [],
            "thresholds": {
                "min_overall_score": 0.75,
                "min_metric_score": 0.5,
            },
        },
    )

    assert report.export_ready is True
    assert report.multimodal_release_ready is True
    assert report.multimodal_release_missing == []
    assert all(report.core_artifact_coverage.values())
    assert report.core_artifact_coverage["lab_reports"] is True
    assert report.core_artifact_coverage["vital_signs_flowsheets"] is True
    assert report.core_artifact_coverage["medication_administration_records"] is True
    assert report.core_artifact_coverage["discharge_summaries"] is True


def test_quality_report_blocks_release_when_task_reference_coverage_is_missing(
    tmp_path,
):
    image_path = tmp_path / "cxr.png"
    image_path.write_bytes(_png_bytes(width=96, height=96))
    base = _record("rec-1")
    record = base.model_copy(
        update={
            "modalities": [
                Modality.STRUCTURED_EHR,
                Modality.CLINICAL_TEXT,
                Modality.LABS,
                Modality.VITALS,
                Modality.TIME_SERIES,
                Modality.IMAGING,
            ],
            "documents": [
                *base.documents,
                ClinicalDocument(
                    document_id="doc-rec-1-vitals",
                    note_type="vital_signs_flowsheet",
                    author_role="nurse",
                    timestamp="2026-01-01T00:05:00",
                    clean_text="Vital signs flowsheet.",
                    messy_text="vs flow",
                ),
                ClinicalDocument(
                    document_id="doc-rec-1-radiology",
                    note_type="radiology_report",
                    author_role="radiologist",
                    timestamp="2026-01-01T00:15:00",
                    clean_text="Portable chest radiograph shows pneumonia.",
                    messy_text="cxr pna",
                ),
            ],
            "vitals": [
                VitalObservation(
                    name="HR",
                    value=112,
                    unit="/min",
                    effective_time="2026-01-01T00:00:00",
                )
            ],
            "time_series": [
                TimeSeriesChannel(
                    name="heart_rate",
                    unit="/min",
                    generation_backend="deterministic",
                    points=[
                        TimeSeriesPoint(
                            timestamp="2026-01-01T00:00:00",
                            values={"value": 112},
                        )
                    ],
                )
            ],
            "imaging": [
                ImagingAsset(
                    image_id="img-rec-1",
                    modality="XR",
                    body_region="chest",
                    prompt="portable chest x-ray pneumonia",
                    file_path=str(image_path),
                    report_text="Right lower lobe pneumonia.",
                    generation_backend="diffusers:cxr_pneumonia_dreambooth",
                )
            ],
            "metadata": {
                "clinical_text_model_policy": {
                    "backend": "deterministic",
                    "provider": "casecrawler",
                    "model_id": "casecrawler-template-clinical-documents",
                    "license": "casecrawler",
                    "gated": False,
                    "use_policy": "deterministic_synthetic_templates_validate_outputs",
                },
                "time_series_model_policy": {
                    "profile": "timediff",
                    "model_id": "MuhangTian/TimeDiff",
                    "license": "mit",
                    "gated": False,
                    "use_policy": "wrap_external_sampler_validate_outputs",
                },
                "imaging_model_policy": {
                    "profile": "cxr_pneumonia_dreambooth",
                    "model_id": "chimbiwide/cxr-pneumonia-dreambooth",
                    "license": "openrail++",
                    "gated": False,
                    "use_policy": "openrail_review_outputs_before_release",
                },
                "image_validator_policy": {
                    "profile": "biomedclip",
                    "backend": "open_clip",
                    "model_id": (
                        "hf-hub:microsoft/"
                        "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                    ),
                    "license": "mit",
                    "gated": False,
                    "use_policy": "open_model_validate_image_text_alignment",
                }
            },
            "validation": ValidationReport(
                schema_score=1.0,
                clinical_consistency_score=1.0,
                privacy_score=1.0,
                utility_score=1.0,
                modality_alignment_score=0.91,
                approved=True,
            ),
        }
    )

    report = build_dataset_quality_report(
        "ds-quality",
        [record],
        benchmark_plan={
            "recommended_reference_keys": ["synthchex_75k", "clinical_notes_to_fhir"],
            "ready": True,
            "resolved_reference_dataset_id": "ds-reference",
            "missing_reference_keys": [],
            "thresholds": {
                "min_overall_score": 0.75,
                "min_metric_score": 0.5,
            },
            "task_export_reference_readiness": {
                "multimodal_jsonl": {
                    "ready": True,
                    "missing_reference_keys": [],
                },
                "note_fact_sft_jsonl": {
                    "ready": False,
                    "missing_reference_keys": ["clinical_notes_to_fhir"],
                },
            },
        },
    )

    assert report.multimodal_release_ready is False
    assert "task_reference_coverage" in report.multimodal_release_missing
    assert report.core_artifact_coverage["task_reference_coverage"] is False


def test_quality_report_blocks_release_without_expected_clinical_document_families(
    tmp_path,
):
    image_path = tmp_path / "cxr.png"
    image_path.write_bytes(_png_bytes(width=96, height=96))
    base = _record("rec-1")
    record = base.model_copy(
        update={
            "modalities": [
                Modality.STRUCTURED_EHR,
                Modality.CLINICAL_TEXT,
                Modality.LABS,
                Modality.VITALS,
                Modality.TIME_SERIES,
                Modality.IMAGING,
            ],
            "documents": [
                document
                for document in base.documents
                if document.note_type
                not in {
                    "discharge_summary",
                    "lab_report",
                    "medication_administration_record",
                }
            ]
            + [
                ClinicalDocument(
                    document_id="doc-rec-1-radiology",
                    note_type="radiology_report",
                    author_role="radiologist",
                    timestamp="2026-01-01T00:15:00",
                    clean_text="Portable chest radiograph shows pneumonia.",
                    messy_text="cxr pna",
                ),
            ],
            "vitals": [
                VitalObservation(
                    name="HR",
                    value=112,
                    unit="/min",
                    effective_time="2026-01-01T00:00:00",
                )
            ],
            "time_series": [
                TimeSeriesChannel(
                    name="heart_rate",
                    unit="/min",
                    generation_backend="deterministic",
                    points=[
                        TimeSeriesPoint(
                            timestamp="2026-01-01T00:00:00",
                            values={"value": 112},
                        )
                    ],
                )
            ],
            "imaging": [
                ImagingAsset(
                    image_id="img-rec-1",
                    modality="XR",
                    body_region="chest",
                    prompt="portable chest x-ray pneumonia",
                    file_path=str(image_path),
                    report_text="Right lower lobe pneumonia.",
                    generation_backend="diffusers:cxr_pneumonia_dreambooth",
                )
            ],
            "metadata": {
                "clinical_text_model_policy": {
                    "backend": "deterministic",
                    "provider": "casecrawler",
                    "model_id": "casecrawler-template-clinical-documents",
                    "license": "casecrawler",
                    "gated": False,
                    "use_policy": "deterministic_synthetic_templates_validate_outputs",
                },
                "time_series_model_policy": {
                    "profile": "timediff",
                    "model_id": "MuhangTian/TimeDiff",
                    "license": "mit",
                    "gated": False,
                    "use_policy": "wrap_external_sampler_validate_outputs",
                },
                "imaging_model_policy": {
                    "profile": "cxr_pneumonia_dreambooth",
                    "model_id": "chimbiwide/cxr-pneumonia-dreambooth",
                    "license": "openrail++",
                    "gated": False,
                    "use_policy": "openrail_review_outputs_before_release",
                },
                "image_validator_policy": {
                    "profile": "biomedclip",
                    "backend": "open_clip",
                    "model_id": (
                        "hf-hub:microsoft/"
                        "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                    ),
                    "license": "mit",
                    "gated": False,
                    "use_policy": "open_model_validate_image_text_alignment",
                },
            },
            "validation": ValidationReport(
                schema_score=1.0,
                clinical_consistency_score=1.0,
                privacy_score=1.0,
                utility_score=1.0,
                modality_alignment_score=0.91,
                approved=True,
            ),
        }
    )

    report = build_dataset_quality_report(
        "ds-quality",
        [record],
        benchmark_plan={
            "recommended_reference_keys": ["synthchex_75k"],
            "ready": True,
            "resolved_reference_dataset_id": "ds-reference",
            "missing_reference_keys": [],
            "thresholds": {
                "min_overall_score": 0.75,
                "min_metric_score": 0.5,
            },
        },
    )

    assert report.multimodal_release_ready is False
    assert "lab_reports" in report.multimodal_release_missing
    assert "vital_signs_flowsheets" in report.multimodal_release_missing
    assert "medication_administration_records" in report.multimodal_release_missing
    assert "discharge_summaries" in report.multimodal_release_missing
    assert report.core_artifact_coverage["nursing_notes"] is True
    assert report.core_artifact_coverage["radiology_reports"] is True


def test_quality_report_marks_task_exports_not_ready_without_artifacts():
    record = _record("rec-1").model_copy(
        update={
            "labs": [],
            "vitals": [],
            "medication_history": [],
            "time_series": [],
            "imaging": [],
        }
    )

    report = build_dataset_quality_report("ds-quality", [record])

    assert report.export_profile_readiness["clinical_observation_jsonl"]["ready"] is False
    assert report.export_profile_readiness["clinical_observation_jsonl"]["missing"] == [
        "labs_or_vitals"
    ]
    assert report.export_profile_readiness["medication_reconciliation_jsonl"]["ready"] is False
    assert report.export_profile_readiness["medication_reconciliation_jsonl"]["missing"] == [
        "medications"
    ]
    assert report.export_profile_readiness["time_series_jsonl"]["ready"] is False
    assert report.export_profile_readiness["time_series_jsonl"]["missing"] == [
        "time_series_channels",
        "time_series_points",
    ]
    assert export_profile_blocker(
        report,
        "medication_reconciliation_jsonl",
    ).startswith("Export profile medication_reconciliation_jsonl is not ready")


def test_quality_report_summarizes_longitudinal_encounter_depth():
    base = _record("rec-1")
    record = base.model_copy(
        update={
            "encounters": [
                *base.encounters,
                Encounter(
                    encounter_id="enc-rec-1-follow-up",
                    start="2026-01-02T00:00:00",
                    end="2026-01-02T04:00:00",
                    setting="inpatient",
                    reason="follow-up sepsis reassessment",
                    diagnoses=base.encounters[0].diagnoses,
                ),
            ],
            "labs": [
                *base.labs,
                LabObservation(
                    name="WBC",
                    value=9.0,
                    unit="K/uL",
                    reference_low=4.5,
                    reference_high=11.0,
                    effective_time="2026-01-02T01:00:00",
                ),
            ],
            "vitals": [
                VitalObservation(
                    name="HR",
                    value=92,
                    unit="/min",
                    effective_time="2026-01-02T00:15:00",
                )
            ],
        }
    )

    report = build_dataset_quality_report("ds-quality", [record])

    assert report.longitudinal_record_rate == 1
    assert report.mean_encounter_span_hours == 28
    assert report.mean_observations_per_encounter == 1.5


def test_quality_report_blocks_export_when_required_human_review_is_missing():
    record = _record("rec-1").model_copy(
        update={"metadata": {"require_human_review": True}}
    )

    report = build_dataset_quality_report("ds-quality", [record])

    assert report.export_ready is False
    assert report.issue_counts_by_field["human_review.missing"] == 1
    assert (
        "Complete required human review before exporting generated datasets."
        in report.recommendations
    )


def test_quality_report_accepts_required_human_review_approval():
    record = _record("rec-1").model_copy(
        update={
            "metadata": {
                "require_human_review": True,
                "human_review": {
                    "status": "approved",
                    "reviewer": "clinical-reviewer",
                    "notes": [],
                    "reviewed_at": "2026-01-01T00:00:00",
                    "metadata": {},
                },
            }
        }
    )

    report = build_dataset_quality_report("ds-quality", [record])

    assert report.export_ready is True
    assert "human_review.missing" not in report.issue_counts_by_field


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


def test_quality_report_requires_structured_ehr_artifacts():
    record = _record("rec-1").model_copy(
        update={
            "modalities": [Modality.STRUCTURED_EHR, Modality.CLINICAL_TEXT],
            "encounters": [],
            "medication_history": [],
        }
    )

    report = build_dataset_quality_report("ds-quality", [record])

    assert report.export_ready is False
    assert report.issue_counts_by_field["structured_ehr.encounters.missing"] == 1
    assert report.issue_counts_by_field["structured_ehr.diagnoses.missing"] == 1
    assert report.issue_counts_by_field["structured_ehr.medication_history.missing"] == 1
    assert any("structured EHR artifacts" in item for item in report.recommendations)


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
    assert report.issue_counts_by_field["documents.lab_report.missing"] == 1
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
                ClinicalDocument(
                    document_id="doc-lab",
                    note_type="lab_report",
                    author_role="physician",
                    timestamp="2026-01-01T00:00:00",
                    clean_text="Lab report.",
                ),
                ClinicalDocument(
                    document_id="doc-mar",
                    note_type="medication_administration_record",
                    author_role="physician",
                    timestamp="2026-01-01T00:00:00",
                    clean_text="Medication administration record.",
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
    assert report.issue_counts_by_field["documents.lab_report.author_role"] == 1
    assert (
        report.issue_counts_by_field[
            "documents.medication_administration_record.author_role"
        ]
        == 1
    )
    assert any("expected clinical document author roles" in item for item in report.recommendations)


def test_quality_report_summarizes_multimodal_training_artifacts(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(_png_bytes(width=96, height=64))
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
                    extracted_facts={
                        "lab_values": [{"name": "WBC", "value": 12.0}],
                        "medications": ["ceftriaxone"],
                    },
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
            "vitals": [
                VitalObservation(
                    name="HR",
                    value=112,
                    unit="/min",
                    effective_time="2026-01-01T00:00:00",
                )
            ],
            "medication_history": [
                MedicationStatement(
                    name="Ceftriaxone",
                    dose="1 g",
                    route="IV",
                    frequency="daily",
                    status="active",
                )
            ],
            "encounters": [
                Encounter(
                    encounter_id="enc-procedure",
                    start="2026-01-01T00:00:00",
                    setting="emergency_department",
                    reason="sepsis",
                    diagnoses=[
                        Code(system="synthetic", code="sepsis", display="sepsis")
                    ],
                    procedures=[
                        Code(
                            system="synthetic",
                            code="central_line",
                            display="Central venous catheter placement",
                        )
                    ],
                )
            ],
            "time_series": [
                TimeSeriesChannel(
                    name="heart_rate",
                    unit="/min",
                    generation_backend="deterministic",
                    points=[
                        TimeSeriesPoint(
                            timestamp="2026-01-01T00:00:00",
                            values={"value": 100},
                        ),
                        TimeSeriesPoint(
                            timestamp="2026-01-01T01:00:00",
                            values={"value": 96},
                        ),
                    ],
                ),
                TimeSeriesChannel(
                    name="ecg_lead_ii",
                    unit="mV",
                    generation_backend="external:timediff-sample",
                    sampling_rate_hz=125,
                    points=[
                        TimeSeriesPoint(
                            timestamp="2026-01-01T00:00:00",
                            values={"millivolts": 0.1, "phase": 0.1},
                        ),
                        TimeSeriesPoint(
                            timestamp="2026-01-01T00:00:01",
                            values={"millivolts": 0.2, "phase": 0.2},
                        ),
                    ],
                ),
            ],
            "imaging": [
                ImagingAsset(
                    image_id="img-1",
                    modality="XR",
                    body_region="chest",
                    prompt="portable chest x-ray pneumonia",
                    file_path=str(image_path),
                    report_text="Pneumonia.",
                    labels=[
                        Code(
                            system="synthetic",
                            code="pneumonia",
                            display="Pneumonia",
                        )
                    ],
                    generation_backend="diffusers:cxr_pneumonia_dreambooth",
                )
            ],
            "metadata": {
                "clinical_text_model_policy": {
                    "backend": "deterministic",
                    "provider": "casecrawler",
                    "model_id": "casecrawler-template-clinical-documents",
                    "license": "casecrawler",
                    "gated": False,
                    "use_policy": "deterministic_synthetic_templates_validate_outputs",
                },
                "time_series_model_policy": {
                    "profile": "timediff",
                    "model_id": "MuhangTian/TimeDiff",
                    "license": "mit",
                    "gated": False,
                    "use_policy": "wrap_external_sampler_validate_outputs",
                },
                "imaging_model_policy": {
                    "profile": "cxr_pneumonia_dreambooth",
                    "model_id": "chimbiwide/cxr-pneumonia-dreambooth",
                    "license": "openrail++",
                    "gated": False,
                    "use_policy": "openrail_review_outputs_before_release",
                },
                "image_validator_policy": {
                    "profile": "biomedclip",
                    "backend": "open_clip",
                    "model_id": (
                        "hf-hub:microsoft/"
                        "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                    ),
                    "license": "mit",
                    "gated": False,
                    "use_policy": "open_model_validate_image_text_alignment",
                },
            },
            "validation": ValidationReport(
                schema_score=1.0,
                clinical_consistency_score=1.0,
                privacy_score=1.0,
                utility_score=1.0,
                modality_alignment_score=0.82,
                approved=True,
            ),
        }
    )

    report = build_dataset_quality_report("ds-quality", [record])

    assert report.artifact_counts["documents"] == 2
    assert report.artifact_counts["messy_documents"] == 1
    assert report.artifact_counts["medications"] == 1
    assert report.artifact_counts["procedures"] == 1
    assert report.artifact_counts["time_series_waveform_channels"] == 1
    assert report.artifact_counts["imaging_assets"] == 1
    assert report.artifact_counts["imaging_file_assets"] == 1
    assert report.note_type_counts == {"ed_note": 1, "radiology_report": 1}
    assert report.extracted_fact_key_counts == {"lab_values": 1, "medications": 1}
    assert report.lab_unit_counts == {"K/uL": 1}
    assert report.lab_numeric_summaries["wbc"]["mean"] == 12.0
    assert report.vital_unit_counts == {"/min": 1}
    assert report.vital_numeric_summaries["hr"]["mean"] == 112.0
    assert report.medication_route_counts == {"IV": 1}
    assert report.medication_dose_counts == {"1 g": 1}
    assert report.medication_frequency_counts == {"daily": 1}
    assert report.medication_status_counts == {"active": 1}
    assert report.time_series_numeric_summaries["heart_rate.value"]["mean"] == 98.0
    assert report.time_series_numeric_summaries["ecg_lead_ii.millivolts"]["mean"] == 0.15
    assert report.time_series_channel_counts == {"ecg_lead_ii": 1, "heart_rate": 1}
    assert report.time_series_unit_counts == {"/min": 1, "mV": 1}
    assert report.mean_time_series_sampling_rate_hz == 125.0
    assert report.mean_time_series_points == 2.0
    assert report.mean_time_series_duration_hours == 0.5001
    assert report.time_series_backend_counts == {
        "deterministic": 1,
        "external:timediff-sample": 1,
    }
    assert report.time_series_model_policy_counts == {
        (
            "profile=timediff|license=mit|gated=false|"
            "use_policy=wrap_external_sampler_validate_outputs"
        ): 2
    }
    assert report.mean_imaging_width == 96.0
    assert report.mean_imaging_height == 64.0
    assert report.mean_imaging_prompt_chars == 30.0
    assert report.mean_imaging_report_chars == 10.0
    assert report.imaging_report_label_evidence_rate == 1.0
    assert report.imaging_backend_counts == {
        "diffusers:cxr_pneumonia_dreambooth": 1,
    }
    assert report.imaging_model_policy_counts == {
        (
            "profile=cxr_pneumonia_dreambooth|license=openrail++|"
            "gated=false|use_policy=openrail_review_outputs_before_release"
        ): 1
    }
    assert report.image_validator_policy_counts == {
        (
            "profile=biomedclip|backend=open_clip|license=mit|"
            "gated=false|use_policy=open_model_validate_image_text_alignment"
        ): 1
    }
    assert report.mean_modality_alignment_score == 0.82
    assert report.export_ready is False
    assert "vitals.missing_artifacts" not in report.issue_counts_by_field
    assert not any("local image files" in item for item in report.recommendations)


def test_quality_report_recommends_file_backed_multimodal_images():
    record = _record("rec-1").model_copy(
        update={
            "modalities": [Modality.CLINICAL_TEXT, Modality.IMAGING],
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

    assert report.artifact_counts["imaging_assets"] == 1
    assert report.artifact_counts["imaging_file_assets"] == 0
    assert any("local image files" in item for item in report.recommendations)


def test_quality_report_summarizes_phi_and_diagnosis_code_signals():
    record = _record("rec-1").model_copy(
        update={
            "documents": [
                ClinicalDocument(
                    document_id="doc-technetium",
                    note_type="discharge_summary",
                    author_role="synthetic_reference",
                    timestamp="2026-01-01T00:00:00",
                    clean_text="Synthetic de-identification note.",
                    extracted_facts={
                        "phi_annotations": [
                            {
                                "entity_type": "NAME",
                                "text": "Smith",
                                "start": 1,
                                "end": 6,
                            },
                            {
                                "entity_type": "AGE",
                                "text": "72-year-old",
                                "start": 10,
                                "end": 21,
                            },
                        ],
                    },
                )
            ],
            "encounters": [
                Encounter(
                    encounter_id="enc-icd",
                    start="2026-01-01T00:00:00",
                    setting="reference",
                    reason="clinical_deidentification_icd_coding",
                    diagnoses=[
                        Code(system="ICD-9-CM", code="428.0", display="Heart failure"),
                        Code(system="ICD-9-CM", code="401.9", display="Hypertension"),
                    ],
                )
            ],
        }
    )

    report = build_dataset_quality_report("ds-quality", [record])

    assert report.phi_entity_counts == {"AGE": 1, "NAME": 1}
    assert report.diagnosis_code_system_counts == {"ICD-9-CM": 2}
    assert report.diagnosis_code_counts == {
        "ICD-9-CM:401.9": 1,
        "ICD-9-CM:428.0": 1,
    }


def test_quality_report_requires_policy_metadata_for_diffusers_images():
    record = _record("rec-1").model_copy(
        update={
            "modalities": [Modality.CLINICAL_TEXT, Modality.IMAGING],
            "imaging": [
                ImagingAsset(
                    image_id="img-1",
                    modality="XR",
                    body_region="chest",
                    prompt="portable chest x-ray pneumonia",
                    report_text="Pneumonia.",
                    generation_backend="diffusers:cxr_pneumonia_dreambooth",
                )
            ],
        }
    )

    report = build_dataset_quality_report("ds-quality", [record])

    assert report.export_ready is False
    assert report.issue_counts_by_field["imaging.model_policy.missing"] == 1
    assert any("imaging model policy metadata" in item for item in report.recommendations)


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


def test_quality_report_surfaces_missing_benchmark_reference_plan():
    report = build_dataset_quality_report(
        "ds-quality",
        [_record("rec-1")],
        benchmark_plan={
            "recommended_reference_keys": ["synthclinicalnotes", "clinical_notes_to_fhir"],
            "ready": False,
            "missing_reference_keys": ["clinical_notes_to_fhir"],
            "thresholds": {
                "min_overall_score": 0.75,
                "min_metric_score": 0.5,
            },
        },
    )

    assert report.export_ready is True
    assert report.benchmark_ready is False
    assert report.recommended_reference_keys == [
        "synthclinicalnotes",
        "clinical_notes_to_fhir",
    ]
    assert report.missing_reference_keys == ["clinical_notes_to_fhir"]
    assert report.benchmark_thresholds == {
        "min_overall_score": 0.75,
        "min_metric_score": 0.5,
    }
    assert any("recommended reference dataset" in item for item in report.recommendations)


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
