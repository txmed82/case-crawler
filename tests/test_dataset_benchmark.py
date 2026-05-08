import struct
import zlib

import pytest

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

from casecrawler.validation.benchmark import (
    DatasetBenchmark,
    _distribution_metric,
    load_benchmark_profile_artifact,
    profile_records,
    write_benchmark_profile_artifact,
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
                dose="1 g",
                route="IV",
                frequency="daily",
                status="active",
            )
        ],
        time_series=[
            TimeSeriesChannel(
                name="heart_rate",
                unit="/min",
                generation_backend="deterministic",
                sampling_rate_hz=1.0,
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
            modality_alignment_score=0.9,
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
    assert profile.lab_unit_counts == {"K/uL": 2}
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
    assert profile.vital_unit_counts == {"/min": 2}
    assert profile.time_series_numeric_summaries["heart_rate.value"] == {
        "count": 4,
        "max": 105.0,
        "mean": 102.5,
        "min": 100.0,
    }
    assert profile.procedure_name_counts == {"Central venous catheter placement": 2}
    assert profile.medication_route_counts == {"IV": 2}
    assert profile.medication_dose_counts == {"1 g": 2}
    assert profile.medication_frequency_counts == {"daily": 2}
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
    assert profile.artifact_counts["imaging_file_assets"] == 0
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
        "imaging_file_assets_per_record": 0.0,
    }
    assert profile.longitudinal_record_rate == 0
    assert profile.mean_encounter_span_hours is None
    assert profile.mean_observations_per_encounter == 2
    assert profile.modality_artifact_coverage == {
        "clinical_text": 1.0,
        "imaging": 1.0,
        "labs": 1.0,
        "structured_ehr": 1.0,
        "time_series": 1.0,
        "vitals": 1.0,
    }
    assert profile.time_series_channel_counts == {"heart_rate": 2}
    assert profile.time_series_unit_counts == {"/min": 2}
    assert profile.time_series_backend_counts == {"deterministic": 2}
    assert profile.mean_time_series_sampling_rate_hz == 1.0
    assert profile.mean_time_series_points == 2
    assert profile.mean_time_series_duration_hours == 6
    assert profile.imaging_modality_counts == {"XR": 2}
    assert profile.imaging_body_region_counts == {"chest": 2}
    assert profile.imaging_backend_counts == {"placeholder": 2}
    assert profile.imaging_model_policy_counts == {}
    assert profile.imaging_label_counts == {"effusion": 2, "opacity": 2}
    assert profile.imaging_label_pair_counts == {"effusion|opacity": 2}
    assert profile.mean_imaging_prompt_chars == 20
    assert profile.mean_imaging_report_chars == 17
    assert profile.imaging_report_label_evidence_rate == 1.0
    assert profile.approved_rate == 1.0
    assert profile.mean_modality_alignment_score == 0.9


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


def test_dataset_benchmark_compares_portable_profile_artifacts(tmp_path):
    generated = [
        _record("rec-1", "ds-gen", age=60, sex="male"),
        _record("rec-2", "ds-gen", age=62, sex="female"),
    ]
    reference = [
        _record("ref-1", "ds-ref", age=61, sex="male"),
        _record("ref-2", "ds-ref", age=63, sex="female"),
    ]

    generated_artifact = write_benchmark_profile_artifact(
        generated,
        tmp_path / "generated-profile.json",
    )
    reference_artifact = write_benchmark_profile_artifact(
        reference,
        tmp_path / "reference-profile.json",
    )
    report = DatasetBenchmark().compare_profiles(
        load_benchmark_profile_artifact(tmp_path / "generated-profile.json"),
        load_benchmark_profile_artifact(tmp_path / "reference-profile.json"),
    )

    assert generated_artifact["artifact_type"] == "casecrawler_benchmark_profile"
    assert reference_artifact["schema_version"] == 1
    assert report.generated_dataset_id == "ds-gen"
    assert report.reference_dataset_id == "ds-ref"
    assert report.overall_score > 0.8


def test_dataset_benchmark_rejects_invalid_profile_artifact(tmp_path):
    (tmp_path / "bad-profile.json").write_text('{"artifact_type": "unknown"}')

    with pytest.raises(ValueError, match="unsupported artifact_type"):
        load_benchmark_profile_artifact(tmp_path / "bad-profile.json")


def test_dataset_benchmark_profiles_longitudinal_encounter_depth():
    base = _record("rec-long", "ds-gen")
    longitudinal = base.model_copy(
        update={
            "encounters": [
                *base.encounters,
                Encounter(
                    encounter_id="enc-rec-long-follow-up",
                    start="2026-01-03T00:00:00",
                    end="2026-01-03T06:00:00",
                    setting="inpatient",
                    reason="follow-up sepsis reassessment",
                    diagnoses=base.encounters[0].diagnoses,
                ),
            ],
            "labs": [
                *base.labs,
                LabObservation(
                    name="WBC",
                    value=10.0,
                    unit="K/uL",
                    reference_low=4.5,
                    reference_high=11.0,
                    effective_time="2026-01-03T01:00:00",
                ),
            ],
            "vitals": [
                *base.vitals,
                VitalObservation(
                    name="HR",
                    value=96,
                    unit="/min",
                    effective_time="2026-01-03T00:15:00",
                ),
            ],
        }
    )

    profile = profile_records([longitudinal])

    assert profile.longitudinal_record_rate == 1
    assert profile.mean_encounter_span_hours == 54
    assert profile.mean_observations_per_encounter == 2


def test_dataset_benchmark_compares_longitudinal_profiles():
    generated_base = _record("rec-gen", "ds-gen")
    reference_base = _record("rec-ref", "ds-ref")
    generated = generated_base.model_copy(
        update={
            "encounters": [
                *generated_base.encounters,
                Encounter(
                    encounter_id="enc-gen-follow-up",
                    start="2026-01-02T00:00:00",
                    end="2026-01-02T06:00:00",
                    setting="inpatient",
                    reason="follow-up sepsis reassessment",
                    diagnoses=generated_base.encounters[0].diagnoses,
                ),
            ]
        }
    )

    report = DatasetBenchmark(min_overall_score=0.0, min_metric_score=0.0).compare(
        [generated],
        [reference_base],
    )

    metrics = {metric.name: metric for metric in report.metrics}

    assert metrics["longitudinal_record_rate"].generated_value == 1
    assert metrics["longitudinal_record_rate"].reference_value == 0
    assert metrics["mean_encounter_span_hours"].generated_value == 30
    assert metrics["mean_encounter_span_hours"].reference_value is None
    assert report.thresholds == {"min_overall_score": 0.0, "min_metric_score": 0.0}
    assert {metric.name for metric in report.metrics} >= {
        "modality_overlap",
        "mean_age",
        "note_type_distribution",
        "clinical_text_model_policy_overlap",
        "clinical_text_model_policy_distribution",
        "lab_name_overlap",
        "lab_unit_overlap",
        "lab_unit_distribution",
        "lab_flag_distribution",
        "lab_value_mean:wbc",
        "vital_name_overlap",
        "vital_unit_overlap",
        "vital_unit_distribution",
        "vital_value_mean:hr",
        "procedure_name_overlap",
        "procedure_name_distribution",
        "medication_name_overlap",
        "medication_dose_distribution",
        "medication_frequency_distribution",
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
        "artifact_density:imaging_file_assets_per_record",
        "longitudinal_record_rate",
        "mean_encounter_span_hours",
        "mean_observations_per_encounter",
        "modality_artifact_coverage:clinical_text",
        "modality_artifact_coverage:structured_ehr",
        "modality_artifact_coverage:labs",
        "modality_artifact_coverage:vitals",
        "modality_artifact_coverage:time_series",
        "modality_artifact_coverage:imaging",
        "time_series_channel_overlap",
        "time_series_unit_overlap",
        "time_series_unit_distribution",
        "time_series_backend_overlap",
        "time_series_backend_distribution",
        "time_series_model_policy_overlap",
        "time_series_model_policy_distribution",
        "mean_time_series_sampling_rate_hz",
        "mean_time_series_points",
        "mean_time_series_duration_hours",
        "imaging_modality_overlap",
        "imaging_body_region_overlap",
        "imaging_backend_overlap",
        "imaging_backend_distribution",
        "imaging_model_policy_overlap",
        "imaging_model_policy_distribution",
        "image_validator_policy_overlap",
        "image_validator_policy_distribution",
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


def test_dataset_benchmark_compares_clinical_text_model_policies():
    generated = [
        _record("rec-1", "ds-gen").model_copy(
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
    ]
    reference = [
        _record("ref-1", "ds-ref").model_copy(
            update={
                "metadata": {
                    "clinical_text_model_policy": {
                        "backend": "llm",
                        "provider": "openrouter",
                        "model_id": "clinical-reference-model",
                        "license": "provider_terms",
                        "gated": False,
                        "use_policy": (
                            "synthetic_clinical_text_review_outputs_before_release"
                        ),
                    }
                }
            }
        )
    ]

    report = DatasetBenchmark().compare(generated, reference)
    overlap_metric = next(
        metric
        for metric in report.metrics
        if metric.name == "clinical_text_model_policy_overlap"
    )

    assert report.generated_profile.clinical_text_model_policy_counts == {
        (
            "backend=llm|provider=ollama|model_id=medgemma-local|gated=false|"
            "use_policy=synthetic clinical text review outputs before release"
        ): 1
    }
    assert overlap_metric.score == 0.0
    assert "clinical_text_model_policy_overlap" in report.failing_metrics


def test_profile_records_summarizes_imaging_file_dimensions(tmp_path):
    first_image = tmp_path / "first.png"
    second_image = tmp_path / "second.png"
    first_image.write_bytes(_png_bytes(width=64, height=48))
    second_image.write_bytes(_png_bytes(width=96, height=80))

    records = [
        _record("rec-1", "ds-gen").model_copy(
            update={
                "imaging": [
                    _record("rec-1", "ds-gen").imaging[0].model_copy(
                        update={"file_path": str(first_image)}
                    )
                ]
            }
        ),
        _record("rec-2", "ds-gen").model_copy(
            update={
                "imaging": [
                    _record("rec-2", "ds-gen").imaging[0].model_copy(
                        update={"file_path": str(second_image)}
                    )
                ]
            }
        ),
    ]

    profile = profile_records(records)

    assert profile.artifact_counts["imaging_file_assets"] == 2
    assert profile.mean_imaging_width == 80
    assert profile.mean_imaging_height == 64


def test_dataset_benchmark_fails_on_imaging_dimension_mismatch(tmp_path):
    generated_image = tmp_path / "generated.png"
    reference_image = tmp_path / "reference.png"
    generated_image.write_bytes(_png_bytes(width=64, height=64))
    reference_image.write_bytes(_png_bytes(width=256, height=256))
    generated_base = _record("rec-1", "ds-gen")
    reference_base = _record("ref-1", "ds-ref")
    generated = [
        generated_base.model_copy(
            update={
                "imaging": [
                    generated_base.imaging[0].model_copy(
                        update={"file_path": str(generated_image)}
                    )
                ]
            }
        )
    ]
    reference = [
        reference_base.model_copy(
            update={
                "imaging": [
                    reference_base.imaging[0].model_copy(
                        update={"file_path": str(reference_image)}
                    )
                ]
            }
        )
    ]

    report = DatasetBenchmark(min_metric_score=0.5).compare(generated, reference)
    metrics = {metric.name: metric for metric in report.metrics}

    assert metrics["mean_imaging_width"].score == 0.0
    assert metrics["mean_imaging_height"].score == 0.0
    assert "mean_imaging_width" in report.failing_metrics
    assert "mean_imaging_height" in report.failing_metrics


def test_dataset_benchmark_compares_modality_alignment_scores():
    generated = [
        _record("rec-1", "ds-gen").model_copy(
            update={
                "validation": ValidationReport(
                    schema_score=1.0,
                    clinical_consistency_score=1.0,
                    privacy_score=1.0,
                    utility_score=1.0,
                    modality_alignment_score=0.25,
                    approved=False,
                )
            }
        )
    ]
    reference = [
        _record("ref-1", "ds-ref").model_copy(
            update={
                "validation": ValidationReport(
                    schema_score=1.0,
                    clinical_consistency_score=1.0,
                    privacy_score=1.0,
                    utility_score=1.0,
                    modality_alignment_score=0.9,
                    approved=True,
                )
            }
        )
    ]

    report = DatasetBenchmark(min_metric_score=0.5).compare(generated, reference)
    metric = next(
        item for item in report.metrics if item.name == "mean_modality_alignment_score"
    )

    assert metric.score == 0.0
    assert metric.generated_value == 0.25
    assert metric.reference_value == 0.9
    assert "mean_modality_alignment_score" in report.failing_metrics


def test_dataset_benchmark_compares_time_series_value_summaries():
    generated = [_record("rec-1", "ds-gen")]
    reference = [
        _record("ref-1", "ds-ref").model_copy(
            update={
                "time_series": [
                    TimeSeriesChannel(
                        name="heart_rate",
                        unit="/min",
                        generation_backend="reference",
                        points=[
                            TimeSeriesPoint(
                                timestamp="2026-01-01T00:00:00",
                                values={"value": 150},
                            ),
                            TimeSeriesPoint(
                                timestamp="2026-01-01T06:00:00",
                                values={"value": 155},
                            ),
                        ],
                    )
                ]
            }
        )
    ]

    report = DatasetBenchmark(min_metric_score=0.5).compare(generated, reference)
    metric = next(
        item for item in report.metrics if item.name == "time_series_value_mean:heart_rate.value"
    )

    assert metric.score == 0.0
    assert metric.generated_value == 102.5
    assert metric.reference_value == 152.5
    assert "time_series_value_mean:heart_rate.value" in report.failing_metrics


def test_dataset_benchmark_fails_on_lab_and_vital_unit_mismatch():
    generated = [_record("rec-1", "ds-gen")]
    reference_base = _record("ref-1", "ds-ref")
    reference = [
        reference_base.model_copy(
            update={
                "labs": [
                    reference_base.labs[0].model_copy(update={"unit": "10^9/L"})
                ],
                "vitals": [
                    reference_base.vitals[0].model_copy(update={"unit": "bpm"})
                ],
            }
        )
    ]

    report = DatasetBenchmark(min_metric_score=0.5).compare(generated, reference)
    metrics = {metric.name: metric for metric in report.metrics}

    assert metrics["lab_unit_overlap"].score == 0.0
    assert metrics["lab_unit_distribution"].score == 0.0
    assert metrics["vital_unit_overlap"].score == 0.0
    assert metrics["vital_unit_distribution"].score == 0.0
    assert "lab_unit_overlap" in report.failing_metrics
    assert "vital_unit_overlap" in report.failing_metrics


def test_dataset_benchmark_fails_on_note_type_distribution_mismatch():
    generated = [_record("rec-1", "ds-gen")]
    reference_base = _record("ref-1", "ds-ref")
    reference = [
        reference_base.model_copy(
            update={
                "documents": [
                    reference_base.documents[0].model_copy(
                        update={
                            "note_type": "nursing_note",
                            "author_role": "nurse",
                        }
                    )
                ]
            }
        )
    ]

    report = DatasetBenchmark(min_metric_score=0.5).compare(generated, reference)
    metric = next(
        item for item in report.metrics if item.name == "note_type_distribution"
    )

    assert metric.score == 0.0
    assert metric.details["generated_counts"] == {"progress_note": 1}
    assert metric.details["reference_counts"] == {"nursing_note": 1}
    assert "note_type_distribution" in report.failing_metrics


def test_dataset_benchmark_fails_on_medication_regimen_mismatch():
    generated = [_record("rec-1", "ds-gen")]
    reference_base = _record("ref-1", "ds-ref")
    reference = [
        reference_base.model_copy(
            update={
                "medication_history": [
                    reference_base.medication_history[0].model_copy(
                        update={
                            "dose": "2 g",
                            "frequency": "twice daily",
                        }
                    )
                ]
            }
        )
    ]

    report = DatasetBenchmark(min_metric_score=0.5).compare(generated, reference)
    metrics = {metric.name: metric for metric in report.metrics}

    assert metrics["medication_dose_distribution"].score == 0.0
    assert metrics["medication_frequency_distribution"].score == 0.0
    assert "medication_dose_distribution" in report.failing_metrics
    assert "medication_frequency_distribution" in report.failing_metrics


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


def test_dataset_benchmark_compares_imaging_model_policies():
    generated = [
        _record("rec-1", "ds-gen").model_copy(
            update={
                "metadata": {
                    "imaging_model_policy": {
                        "profile": "medisyn",
                        "model_id": "hiesingerlab/MediSyn",
                        "license": "cc-by-nc-nd-4.0",
                        "gated": False,
                        "use_policy": "non_commercial_no_derivatives_review_before_release",
                    }
                }
            }
        )
    ]
    reference = [
        _record("ref-1", "ds-ref").model_copy(
            update={
                "metadata": {
                    "imaging_model_policy": {
                        "profile": "cxr_pneumonia_dreambooth",
                        "model_id": "chimbiwide/cxr-pneumonia-dreambooth",
                        "license": "openrail++",
                        "gated": False,
                        "use_policy": "openrail_review_outputs_before_release",
                    }
                }
            }
        )
    ]

    report = DatasetBenchmark().compare(generated, reference)
    overlap_metric = next(
        metric for metric in report.metrics if metric.name == "imaging_model_policy_overlap"
    )
    distribution_metric = next(
        metric
        for metric in report.metrics
        if metric.name == "imaging_model_policy_distribution"
    )

    assert report.generated_profile.imaging_model_policy_counts == {
        (
            "profile=medisyn|license=cc-by-nc-nd-4.0|gated=false|"
            "use_policy=non commercial no derivatives review before release"
        ): 1
    }
    assert overlap_metric.score == 0.0
    assert distribution_metric.score == 0.0
    assert "imaging_model_policy_overlap" in report.failing_metrics


def test_dataset_benchmark_compares_image_validator_policies():
    generated = [
        _record("rec-1", "ds-gen").model_copy(
            update={
                "metadata": {
                    "image_validator_policy": {
                        "profile": "lexical",
                        "backend": "lexical",
                        "model_id": None,
                        "license": "casecrawler",
                        "gated": False,
                        "use_policy": "deterministic_screening_only",
                    }
                }
            }
        )
    ]
    reference = [
        _record("ref-1", "ds-ref").model_copy(
            update={
                "metadata": {
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
                }
            }
        )
    ]

    report = DatasetBenchmark().compare(generated, reference)
    overlap_metric = next(
        metric for metric in report.metrics if metric.name == "image_validator_policy_overlap"
    )
    distribution_metric = next(
        metric
        for metric in report.metrics
        if metric.name == "image_validator_policy_distribution"
    )

    assert report.generated_profile.image_validator_policy_counts == {
        (
            "profile=lexical|backend=lexical|license=casecrawler|"
            "gated=false|use_policy=deterministic screening only"
        ): 1
    }
    assert overlap_metric.score == 0.0
    assert distribution_metric.score == 0.0
    assert "image_validator_policy_overlap" in report.failing_metrics


def test_dataset_benchmark_compares_imaging_report_label_evidence():
    generated = [
        _record("rec-1", "ds-gen").model_copy(
            update={
                "imaging": [
                    _record("rec-1", "ds-gen").imaging[0].model_copy(
                        update={
                            "prompt": "portable chest x-ray",
                            "report_text": "Portable chest radiograph reviewed.",
                        }
                    )
                ]
            }
        )
    ]
    reference = [_record("ref-1", "ds-ref")]

    report = DatasetBenchmark().compare(generated, reference)
    evidence_metric = next(
        metric
        for metric in report.metrics
        if metric.name == "imaging_report_label_evidence_rate"
    )

    assert report.generated_profile.imaging_report_label_evidence_rate == 0.0
    assert report.reference_profile.imaging_report_label_evidence_rate == 1.0
    assert evidence_metric.score == 0.0
    assert "imaging_report_label_evidence_rate" in report.failing_metrics


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


def test_dataset_benchmark_compares_time_series_model_policies():
    generated = [
        _record("rec-1", "ds-gen").model_copy(
            update={
                "metadata": {
                    "time_series_model_policy": {
                        "profile": "timediff",
                        "model_id": "MuhangTian/TimeDiff",
                        "license": "mit",
                        "gated": False,
                        "use_policy": "wrap_external_sampler_validate_outputs",
                    }
                }
            }
        )
    ]
    reference = [
        _record("ref-1", "ds-ref").model_copy(
            update={
                "metadata": {
                    "time_series_model_policy": {
                        "profile": "rawmed",
                        "model_id": "eunbyeol-cho/RawMed",
                        "license": None,
                        "gated": False,
                        "use_policy": "research_reference_validate_outputs",
                    }
                }
            }
        )
    ]

    report = DatasetBenchmark().compare(generated, reference)
    overlap_metric = next(
        metric
        for metric in report.metrics
        if metric.name == "time_series_model_policy_overlap"
    )
    distribution_metric = next(
        metric
        for metric in report.metrics
        if metric.name == "time_series_model_policy_distribution"
    )

    assert report.generated_profile.time_series_model_policy_counts == {
        (
            "profile=timediff|license=mit|gated=false|"
            "use_policy=wrap external sampler validate outputs"
        ): 1
    }
    assert overlap_metric.score == 0.0
    assert distribution_metric.score == 0.0
    assert "time_series_model_policy_overlap" in report.failing_metrics


def test_dataset_benchmark_fails_on_time_series_unit_and_sampling_rate_mismatch():
    generated = [_record("rec-1", "ds-gen")]
    reference_base = _record("ref-1", "ds-ref")
    reference = [
        reference_base.model_copy(
            update={
                "time_series": [
                    reference_base.time_series[0].model_copy(
                        update={
                            "unit": "bpm",
                            "sampling_rate_hz": 10.0,
                        }
                    )
                ]
            }
        )
    ]

    report = DatasetBenchmark(min_metric_score=0.5).compare(generated, reference)
    metrics = {metric.name: metric for metric in report.metrics}

    assert metrics["time_series_unit_overlap"].score == 0.0
    assert metrics["time_series_unit_distribution"].score == 0.0
    assert metrics["mean_time_series_sampling_rate_hz"].score == 0.0
    assert "time_series_unit_overlap" in report.failing_metrics
    assert "mean_time_series_sampling_rate_hz" in report.failing_metrics


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


def test_dataset_benchmark_compares_phi_and_diagnosis_code_reference_signals():
    generated = [_record("rec-1", "ds-gen")]
    reference = [
        _record("ref-1", "ds-ref").model_copy(
            update={
                "encounters": [
                    Encounter(
                        encounter_id="enc-ref",
                        start="2026-01-01T00:00:00",
                        setting="reference",
                        reason="clinical_deidentification_icd_coding",
                        diagnoses=[
                            Code(
                                system="ICD-9-CM",
                                code="428.0",
                                display="ICD-9-CM 428.0",
                            ),
                            Code(
                                system="ICD-9-CM",
                                code="401.9",
                                display="ICD-9-CM 401.9",
                            ),
                        ],
                    )
                ],
                "documents": [
                    ClinicalDocument(
                        document_id="doc-ref",
                        note_type="discharge_summary",
                        author_role="synthetic_reference",
                        timestamp="2026-01-01T00:00:00",
                        clean_text="Synthetic Technetium-I reference note.",
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
                                    "start": 30,
                                    "end": 41,
                                },
                            ],
                            "diagnoses": [
                                {
                                    "system": "ICD-9-CM",
                                    "code": "428.0",
                                    "display": "ICD-9-CM 428.0",
                                }
                            ],
                        },
                    )
                ],
            }
        )
    ]

    report = DatasetBenchmark(min_metric_score=0.5).compare(generated, reference)
    phi_overlap = next(
        metric for metric in report.metrics if metric.name == "phi_entity_overlap"
    )
    phi_distribution = next(
        metric for metric in report.metrics if metric.name == "phi_entity_distribution"
    )
    diagnosis_system_overlap = next(
        metric
        for metric in report.metrics
        if metric.name == "diagnosis_code_system_overlap"
    )

    assert report.reference_profile.phi_entity_counts == {"AGE": 1, "NAME": 1}
    assert report.reference_profile.diagnosis_code_system_counts == {"ICD-9-CM": 2}
    assert phi_overlap.score == 0.0
    assert phi_distribution.reference_value == 2
    assert diagnosis_system_overlap.details["reference_only"] == ["ICD-9-CM"]
    assert "phi_entity_overlap" in report.failing_metrics
    assert "diagnosis_code_system_overlap" in report.failing_metrics


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
