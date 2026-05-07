from casecrawler.models.synthetic import (
    ClinicalDocument,
    Code,
    ComplexityProfile,
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
            Modality.LABS,
            Modality.VITALS,
            Modality.TIME_SERIES,
            Modality.IMAGING,
        ],
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
    assert profile.medication_route_counts == {"IV": 2}
    assert profile.medication_status_counts == {"active": 2}
    assert profile.document_author_role_counts == {"physician": 2}
    assert profile.messy_document_rate == 1.0
    assert profile.time_series_channel_counts == {"heart_rate": 2}
    assert profile.mean_time_series_points == 2
    assert profile.mean_time_series_duration_hours == 6
    assert profile.imaging_modality_counts == {"XR": 2}
    assert profile.imaging_body_region_counts == {"chest": 2}
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
    assert {metric.name for metric in report.metrics} >= {
        "modality_overlap",
        "mean_age",
        "lab_name_overlap",
        "lab_flag_distribution",
        "lab_value_mean:wbc",
        "vital_name_overlap",
        "vital_value_mean:hr",
        "medication_name_overlap",
        "medication_route_distribution",
        "medication_status_distribution",
        "document_author_role_overlap",
        "document_author_role_distribution",
        "messy_document_rate",
        "time_series_channel_overlap",
        "mean_time_series_points",
        "mean_time_series_duration_hours",
        "imaging_modality_overlap",
        "imaging_body_region_overlap",
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
    assert any("modality_overlap" in warning for warning in report.warnings)


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
