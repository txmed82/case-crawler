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
from casecrawler.validation.quality import build_dataset_quality_report


def _record(record_id: str, *, approved: bool = True, issues=None) -> SyntheticRecord:
    return SyntheticRecord(
        record_id=record_id,
        dataset_id="ds-quality",
        topic="sepsis",
        complexity=ComplexityProfile.MODERATE,
        modalities=[Modality.STRUCTURED_EHR, Modality.CLINICAL_TEXT, Modality.LABS],
        patient=SyntheticPatient(patient_id=f"pat-{record_id}", age=64, sex="male"),
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
    assert report.longitudinal_record_rate == 0
    assert report.mean_encounter_span_hours is None
    assert report.mean_observations_per_encounter == 1
    assert report.recommendations == []


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
    image_path.write_bytes(b"synthetic image")
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
                MedicationStatement(name="Ceftriaxone", route="IV", status="active")
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
                        )
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
                    file_path=str(image_path),
                    report_text="Pneumonia.",
                    generation_backend="diffusers:cxr_pneumonia_dreambooth",
                )
            ],
            "metadata": {
                "imaging_model_policy": {
                    "profile": "cxr_pneumonia_dreambooth",
                    "model_id": "chimbiwide/cxr-pneumonia-dreambooth",
                    "license": "openrail++",
                    "gated": False,
                    "use_policy": "openrail_review_outputs_before_release",
                }
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
    assert report.lab_numeric_summaries["wbc"]["mean"] == 12.0
    assert report.vital_numeric_summaries["hr"]["mean"] == 112.0
    assert report.time_series_numeric_summaries["heart_rate.value"]["mean"] == 100.0
    assert report.time_series_numeric_summaries["ecg_lead_ii.millivolts"]["mean"] == 0.1
    assert report.time_series_backend_counts == {
        "deterministic": 1,
        "external:timediff-sample": 1,
    }
    assert report.imaging_backend_counts == {
        "diffusers:cxr_pneumonia_dreambooth": 1,
    }
    assert report.imaging_model_policy_counts == {
        (
            "profile=cxr_pneumonia_dreambooth|license=openrail++|"
            "gated=false|use_policy=openrail_review_outputs_before_release"
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
