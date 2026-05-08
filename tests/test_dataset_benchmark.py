from casecrawler.models.synthetic import (
    ClinicalDocument,
    Code,
    ComplexityProfile,
    Encounter,
    ImagingAsset,
    LabObservation,
    MedicationStatement,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
    TimeSeriesChannel,
    TimeSeriesPoint,
    ValidationReport,
    VitalObservation,
)
import pytest

from casecrawler.validation.benchmark import (
    DatasetBenchmark,
    _distribution_metric,
    profile_records,
)


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
        modalities=[
            Modality.CLINICAL_TEXT,
            Modality.STRUCTURED_EHR,
            Modality.LABS,
            Modality.VITALS,
            Modality.TIME_SERIES,
            Modality.IMAGING,
        ],
        patient=SyntheticPatient(patient_id=f"pat-{record_id}", age=age, sex=sex),
        encounters=[
            Encounter(
                encounter_id=f"enc-{record_id}",
                start="2026-01-01T00:00:00",
                setting="emergency_department",
                reason=topic,
                diagnoses=[
                    Code(system="synthetic", code=topic, display=topic),
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
            MedicationStatement(
                name="Ceftriaxone",
                route="IV",
                status="active",
            )
        ],
        time_series=[
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
                        timestamp="2026-01-01T06:00:00",
                        values={"value": 105},
                    ),
                ],
            )
        ],
        imaging=[
            ImagingAsset(
                image_id=f"img-{record_id}",
                modality="XR",
                body_region="chest",
                prompt="portable chest x-ray",
                report_text="No focal opacity.",
                labels=[
                    Code(system="synthetic", code="opacity", display="Opacity"),
                    Code(system="synthetic", code="effusion", display="Effusion"),
                ],
                generation_backend="placeholder",
            )
        ],
        documents=[
            ClinicalDocument(
                document_id=f"doc-{record_id}",
                note_type=note_type,
                author_role="physician",
                timestamp="2026-01-01T00:00:00",
                clean_text="Synthetic clinical note with labs and vitals.",
                messy_text="synthetic clinical note w/ labs + vitals",
                extracted_facts={
                    "lab_values": [{"name": "WBC", "value": 12.0, "unit": "K/uL"}],
                    "vital_values": [{"name": "HR", "value": 110, "unit": "/min"}],
                    "medications": ["Ceftriaxone"],
                    "imaging_labels": ["Opacity", "Effusion"],
                },
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
    assert profile.lab_flag_counts == {"H": 2}
    assert profile.lab_numeric_summaries["wbc"] == {
        "count": 2,
        "max": 12.0,
        "mean": 12.0,
        "min": 12.0,
    }
    assert profile.vital_numeric_summaries["hr"] == {
        "count": 2,
        "max": 110.0,
        "mean": 110.0,
        "min": 110.0,
    }
    assert profile.procedure_name_counts == {"Central venous catheter placement": 2}
    assert profile.medication_route_counts == {"IV": 2}
    assert profile.medication_status_counts == {"active": 2}
    assert profile.document_author_role_counts == {"physician": 2}
    assert profile.messy_document_rate == 1.0
    assert profile.extracted_fact_key_counts == {
        "imaging_labels": 2,
        "lab_values": 2,
        "medications": 2,
        "vital_values": 2,
    }
    assert profile.extracted_fact_density == {
        "imaging_labels_per_record": 1.0,
        "lab_values_per_record": 1.0,
        "medications_per_record": 1.0,
        "vital_values_per_record": 1.0,
    }
    assert profile.artifact_counts["documents"] == 2
    assert profile.artifact_counts["encounters"] == 2
    assert profile.artifact_counts["diagnoses"] == 2
    assert profile.artifact_counts["procedures"] == 2
    assert profile.artifact_counts["messy_documents"] == 2
    assert profile.artifact_counts["labs"] == 2
    assert profile.artifact_counts["vitals"] == 2
    assert profile.artifact_counts["medications"] == 2
    assert profile.artifact_counts["time_series_channels"] == 2
    assert profile.artifact_counts["time_series_points"] == 4
    assert profile.artifact_counts["imaging_assets"] == 2
    assert profile.artifact_density == {
        "documents_per_record": 1.0,
        "encounters_per_record": 1.0,
        "diagnoses_per_record": 1.0,
        "procedures_per_record": 1.0,
        "labs_per_record": 1.0,
        "vitals_per_record": 1.0,
        "medications_per_record": 1.0,
        "time_series_channels_per_record": 1.0,
        "imaging_assets_per_record": 1.0,
    }
    assert profile.modality_artifact_coverage == {
        "clinical_text": 1.0,
        "imaging": 1.0,
        "labs": 1.0,
        "structured_ehr": 1.0,
        "time_series": 1.0,
        "vitals": 1.0,
    }
    assert profile.time_series_channel_counts == {"heart_rate": 2}
    assert profile.time_series_backend_counts == {"deterministic": 2}
    assert profile.mean_time_series_points == 2
    assert profile.mean_time_series_duration_hours == 6
    assert profile.imaging_modality_counts == {"XR": 2}
    assert profile.imaging_body_region_counts == {"chest": 2}
    assert profile.imaging_backend_counts == {"placeholder": 2}
    assert profile.imaging_label_counts == {"effusion": 2, "opacity": 2}
    assert profile.imaging_label_pair_counts == {"effusion|opacity": 2}
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
    assert report.passed is True
    assert report.failing_metrics == []
    assert report.thresholds == {"min_overall_score": 0.75, "min_metric_score": 0.5}
    assert {metric.name for metric in report.metrics} >= {
        "modality_overlap",
        "mean_age",
        "lab_name_overlap",
        "lab_flag_distribution",
        "lab_value_mean:wbc",
        "vital_name_overlap",
        "vital_value_mean:hr",
        "procedure_name_overlap",
        "procedure_name_distribution",
        "medication_name_overlap",
        "medication_route_distribution",
        "medication_status_distribution",
        "document_author_role_overlap",
        "document_author_role_distribution",
        "messy_document_rate",
        "extracted_fact_key_overlap",
        "extracted_fact_key_distribution",
        "extracted_fact_density:imaging_labels_per_record",
        "extracted_fact_density:lab_values_per_record",
        "extracted_fact_density:medications_per_record",
        "extracted_fact_density:vital_values_per_record",
        "artifact_density:documents_per_record",
        "artifact_density:encounters_per_record",
        "artifact_density:diagnoses_per_record",
        "artifact_density:procedures_per_record",
        "artifact_density:labs_per_record",
        "artifact_density:vitals_per_record",
        "artifact_density:medications_per_record",
        "artifact_density:time_series_channels_per_record",
        "artifact_density:imaging_assets_per_record",
        "modality_artifact_coverage:clinical_text",
        "modality_artifact_coverage:structured_ehr",
        "modality_artifact_coverage:labs",
        "modality_artifact_coverage:vitals",
        "modality_artifact_coverage:time_series",
        "modality_artifact_coverage:imaging",
        "time_series_channel_overlap",
        "time_series_backend_overlap",
        "time_series_backend_distribution",
        "mean_time_series_points",
        "mean_time_series_duration_hours",
        "imaging_modality_overlap",
        "imaging_body_region_overlap",
        "imaging_backend_overlap",
        "imaging_backend_distribution",
        "imaging_label_overlap",
        "imaging_label_distribution",
        "imaging_label_pair_overlap",
    }


def test_dataset_benchmark_compares_imaging_finding_labels():
    generated = [
        _record("rec-1", "ds-gen").model_copy(
            update={
                "imaging": [
                    ImagingAsset(
                        image_id="img-gen",
                        modality="XR",
                        body_region="chest",
                        prompt="portable chest x-ray with pulmonary edema",
                        report_text="Pulmonary edema without focal pneumonia.",
                        labels=[
                            Code(
                                system="synthetic",
                                code="pulmonary_edema",
                                display="Pulmonary edema",
                            )
                        ],
                        generation_backend="placeholder",
                    )
                ]
            }
        )
    ]
    reference = [
        _record("ref-1", "ds-ref").model_copy(
            update={
                "imaging": [
                    ImagingAsset(
                        image_id="img-ref",
                        modality="XR",
                        body_region="chest",
                        prompt="portable chest x-ray with pleural effusion",
                        report_text="Small pleural effusion.",
                        labels=[
                            Code(
                                system="synthetic",
                                code="pleural_effusion",
                                display="Pleural effusion",
                            )
                        ],
                        generation_backend="reference",
                    )
                ]
            }
        )
    ]

    report = DatasetBenchmark().compare(generated, reference)
    label_metric = next(
        metric for metric in report.metrics if metric.name == "imaging_label_overlap"
    )
    distribution_metric = next(
        metric
        for metric in report.metrics
        if metric.name == "imaging_label_distribution"
    )

    assert label_metric.score == 0.0
    assert label_metric.details["generated_only"] == ["pulmonary edema"]
    assert label_metric.details["reference_only"] == ["pleural effusion"]
    assert distribution_metric.score == 0.0
    assert any("imaging_label_overlap" in warning for warning in report.warnings)


def test_dataset_benchmark_compares_imaging_generation_backends():
    generated = [_record("rec-1", "ds-gen")]
    reference = [
        _record("ref-1", "ds-ref").model_copy(
            update={
                "imaging": [
                    ImagingAsset(
                        image_id="img-ref",
                        modality="XR",
                        body_region="chest",
                        prompt="portable chest x-ray",
                        report_text="No focal opacity.",
                        labels=[
                            Code(system="synthetic", code="opacity", display="Opacity"),
                        ],
                        generation_backend="diffusers:cxr_pneumonia_dreambooth",
                    )
                ]
            }
        )
    ]

    report = DatasetBenchmark().compare(generated, reference)
    backend_metric = next(
        metric for metric in report.metrics if metric.name == "imaging_backend_overlap"
    )

    assert report.generated_profile.imaging_backend_counts == {"placeholder": 1}
    assert report.reference_profile.imaging_backend_counts == {
        "diffusers:cxr_pneumonia_dreambooth": 1
    }
    assert backend_metric.score == 0.0
    assert "imaging_backend_overlap" in report.failing_metrics


def test_dataset_benchmark_flags_numeric_lab_and_vital_drift():
    generated = [_record("rec-1", "ds-gen")]
    reference = [
        _record("ref-1", "ds-ref").model_copy(
            update={
                "labs": [
                    LabObservation(
                        name="WBC",
                        value=40.0,
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
                        value=180,
                        unit="/min",
                        effective_time="2026-01-01T00:00:00",
                    )
                ],
            }
        )
    ]

    report = DatasetBenchmark().compare(generated, reference)
    lab_metric = next(
        metric for metric in report.metrics if metric.name == "lab_value_mean:wbc"
    )
    vital_metric = next(
        metric for metric in report.metrics if metric.name == "vital_value_mean:hr"
    )

    assert lab_metric.generated_value == 12.0
    assert lab_metric.reference_value == 40.0
    assert lab_metric.score < 0.5
    assert vital_metric.generated_value == 110.0
    assert vital_metric.reference_value == 180.0
    assert vital_metric.score == 0.0
    assert any("lab_value_mean:wbc" in warning for warning in report.warnings)
    assert any("vital_value_mean:hr" in warning for warning in report.warnings)


def test_dataset_benchmark_flags_modality_mismatch():
    generated = [
        _record("rec-1", "ds-gen").model_copy(
            update={"modalities": [Modality.CLINICAL_TEXT]}
        )
    ]
    reference = [
        _record("ref-1", "ds-ref").model_copy(update={"modalities": [Modality.IMAGING]})
    ]

    report = DatasetBenchmark().compare(generated, reference)
    modality_metric = next(
        metric for metric in report.metrics if metric.name == "modality_overlap"
    )

    assert modality_metric.score == 0.0
    assert report.passed is False
    assert "modality_overlap" in report.failing_metrics
    assert any("Benchmark gate failed: modality_overlap" in warning for warning in report.warnings)
    assert any("modality_overlap" in warning for warning in report.warnings)


def test_dataset_benchmark_flags_declared_modality_without_artifacts():
    generated = [
        _record("rec-1", "ds-gen").model_copy(
            update={
                "modalities": [Modality.CLINICAL_TEXT, Modality.IMAGING],
                "imaging": [],
            }
        )
    ]
    reference = [_record("ref-1", "ds-ref")]

    report = DatasetBenchmark().compare(generated, reference)
    coverage_metric = next(
        metric
        for metric in report.metrics
        if metric.name == "modality_artifact_coverage:imaging"
    )
    density_metric = next(
        metric
        for metric in report.metrics
        if metric.name == "artifact_density:imaging_assets_per_record"
    )

    assert report.generated_profile.artifact_counts["imaging_assets"] == 0
    assert report.generated_profile.modality_artifact_coverage["imaging"] == 0.0
    assert coverage_metric.generated_value == 0.0
    assert coverage_metric.reference_value == 1.0
    assert coverage_metric.score == 0.0
    assert density_metric.generated_value == 0.0
    assert density_metric.reference_value == 1.0
    assert density_metric.score == 0.0
    assert any("modality_artifact_coverage:imaging" in warning for warning in report.warnings)


def test_dataset_benchmark_compares_time_series_generation_backends():
    generated = [_record("rec-1", "ds-gen")]
    reference = [
        _record("ref-1", "ds-ref").model_copy(
            update={
                "time_series": [
                    TimeSeriesChannel(
                        name="heart_rate",
                        unit="/min",
                        generation_backend="external:timediff-sample",
                        points=[
                            TimeSeriesPoint(
                                timestamp="2026-01-01T00:00:00",
                                values={"value": 100},
                            )
                        ],
                    )
                ]
            }
        )
    ]

    report = DatasetBenchmark().compare(generated, reference)
    backend_metric = next(
        metric for metric in report.metrics if metric.name == "time_series_backend_overlap"
    )

    assert report.generated_profile.time_series_backend_counts == {"deterministic": 1}
    assert report.reference_profile.time_series_backend_counts == {
        "external:timediff-sample": 1
    }
    assert backend_metric.score == 0.0
    assert "time_series_backend_overlap" in report.failing_metrics


def test_dataset_benchmark_supports_custom_pass_thresholds():
    generated = [_record("rec-1", "ds-gen")]
    reference = [_record("ref-1", "ds-ref")]

    report = DatasetBenchmark(min_overall_score=1.0, min_metric_score=1.0).compare(
        generated,
        reference,
    )

    assert report.passed is True
    assert report.failing_metrics == []
    assert report.thresholds == {"min_overall_score": 1.0, "min_metric_score": 1.0}


def test_dataset_benchmark_rejects_invalid_thresholds():
    with pytest.raises(ValueError, match="min_overall_score"):
        DatasetBenchmark(min_overall_score=1.1)
    with pytest.raises(ValueError, match="min_metric_score"):
        DatasetBenchmark(min_metric_score=-0.1)


def test_dataset_benchmark_flags_declared_structured_ehr_without_artifacts():
    generated = [
        _record("rec-1", "ds-gen").model_copy(
            update={
                "modalities": [Modality.STRUCTURED_EHR],
                "encounters": [],
                "medication_history": [],
            }
        )
    ]
    reference = [_record("ref-1", "ds-ref")]

    report = DatasetBenchmark().compare(generated, reference)
    coverage_metric = next(
        metric
        for metric in report.metrics
        if metric.name == "modality_artifact_coverage:structured_ehr"
    )
    encounter_density = next(
        metric
        for metric in report.metrics
        if metric.name == "artifact_density:encounters_per_record"
    )
    diagnosis_density = next(
        metric
        for metric in report.metrics
        if metric.name == "artifact_density:diagnoses_per_record"
    )

    assert report.generated_profile.artifact_counts["encounters"] == 0
    assert report.generated_profile.artifact_counts["diagnoses"] == 0
    assert report.generated_profile.modality_artifact_coverage["structured_ehr"] == 0.0
    assert coverage_metric.generated_value == 0.0
    assert coverage_metric.reference_value == 1.0
    assert coverage_metric.score == 0.0
    assert encounter_density.generated_value == 0.0
    assert diagnosis_density.generated_value == 0.0
    assert any("modality_artifact_coverage:structured_ehr" in warning for warning in report.warnings)


def test_dataset_benchmark_flags_missing_note_fact_targets():
    generated = [
        _record("rec-1", "ds-gen").model_copy(
            update={
                "documents": [
                    ClinicalDocument(
                        document_id="doc-gen",
                        note_type="progress_note",
                        author_role="physician",
                        timestamp="2026-01-01T00:00:00",
                        clean_text="Synthetic clinical note without extracted facts.",
                        messy_text="synthetic clinical note",
                        extracted_facts={},
                    )
                ]
            }
        )
    ]
    reference = [_record("ref-1", "ds-ref")]

    report = DatasetBenchmark().compare(generated, reference)
    overlap_metric = next(
        metric for metric in report.metrics if metric.name == "extracted_fact_key_overlap"
    )
    lab_density_metric = next(
        metric
        for metric in report.metrics
        if metric.name == "extracted_fact_density:lab_values_per_record"
    )

    assert report.generated_profile.extracted_fact_key_counts == {}
    assert overlap_metric.score == 0.0
    assert overlap_metric.details["reference_only"] == [
        "imaging_labels",
        "lab_values",
        "medications",
        "vital_values",
    ]
    assert lab_density_metric.generated_value is None
    assert lab_density_metric.reference_value == 1.0
    assert lab_density_metric.score == 0.0
    assert "extracted_fact_key_overlap" in report.failing_metrics
    assert any("extracted_fact_key_overlap" in warning for warning in report.warnings)


def test_profile_records_rejects_mixed_dataset_records():
    records = [_record("rec-1", "ds-one"), _record("rec-2", "ds-two")]

    with pytest.raises(ValueError, match="one dataset"):
        profile_records(records)


def test_distribution_metric_handles_empty_side_without_division_error():
    metric = _distribution_metric("sex_distribution", {}, {"female": 1})

    assert metric.score == 0.5


def test_dataset_benchmark_normalizes_mixed_timezone_time_series():
    generated = [
        _record("rec-1", "ds-gen").model_copy(
            update={
                "time_series": [
                    TimeSeriesChannel(
                        name="heart_rate",
                        unit="/min",
                        points=[
                            TimeSeriesPoint(
                                timestamp="2026-01-01T00:00:00",
                                values={"value": 100},
                            ),
                            TimeSeriesPoint(
                                timestamp="2026-01-01T06:00:00+00:00",
                                values={"value": 105},
                            ),
                        ],
                    )
                ]
            }
        )
    ]
    reference = [_record("ref-1", "ds-ref")]

    report = DatasetBenchmark().compare(generated, reference)
    duration_metric = next(
        metric for metric in report.metrics if metric.name == "mean_time_series_duration_hours"
    )

    assert duration_metric.generated_value == 6
