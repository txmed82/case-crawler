from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from casecrawler.models.evaluation import DatasetQualityReport
from casecrawler.models.synthetic import SyntheticRecord


SYNTHETIC_REFERENCE_GENERATORS = frozenset(
    {
        "casecrawler-bundled-reference-fixture",
        "synthea",
        "synthea-import",
        "synthea-run",
    }
)


def build_export_transparency_summary(
    *,
    dataset_id: str,
    export_format: str,
    quality_report: DatasetQualityReport,
    synthetic_data: bool,
    real_patient_data: bool,
    task_coverage: dict[str, int] | None = None,
    benchmark: dict[str, Any] | None = None,
    benchmark_suite: dict[str, Any] | None = None,
    objective_coverage: dict[str, Any] | None = None,
    audit_artifacts: list[str] | None = None,
    seeded_references: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a consumer-facing transparency summary for an exported dataset."""
    limitations = [
        "Clinical realism depends on configured generators and validation references.",
        "Downstream model training should be benchmarked against external references before release.",
        "Human review status and validation reports should be checked before production use.",
    ]
    if synthetic_data and not real_patient_data:
        limitations.insert(
            0,
            "Records are generated synthetic examples, not real patient records.",
        )
    elif real_patient_data:
        limitations.insert(
            0,
            "Package may include imported reference data; review provenance before downstream use.",
        )
    return {
        "schema_version": "casecrawler.transparency.v1",
        "dataset_id": dataset_id,
        "export_format": export_format,
        "synthetic_data": synthetic_data,
        "real_patient_data": real_patient_data,
        "intended_use": [
            "synthetic healthcare AI training",
            "fine-tuning experiments",
            "evaluation harnesses",
            "pipeline and format integration tests",
        ],
        "not_intended_use": [
            "clinical diagnosis or treatment",
            "substitution for real-world clinical validation",
            "re-identification or patient-level inference",
        ],
        "record_counts": {
            "total": quality_report.record_count,
            "approved": quality_report.approved_count,
            "approval_rate": quality_report.approval_rate,
        },
        "quality_gates": {
            "export_ready": quality_report.export_ready,
            "benchmark_ready": quality_report.benchmark_ready,
            "multimodal_release_ready": quality_report.multimodal_release_ready,
            "multimodal_release_missing": quality_report.multimodal_release_missing,
            "blocking_issue_count": quality_report.blocking_issue_count,
            "warning_issue_count": quality_report.warning_issue_count,
            "issue_counts_by_field": quality_report.issue_counts_by_field,
        },
        "artifact_coverage": {
            "modalities": quality_report.modality_counts,
            "core_artifact_coverage": quality_report.core_artifact_coverage,
            "artifact_counts": quality_report.artifact_counts,
            "note_type_counts": quality_report.note_type_counts,
            "task_coverage": task_coverage or {},
        },
        "population_summary": {
            "race_counts": quality_report.race_counts,
            "ethnicity_counts": quality_report.ethnicity_counts,
            "insurance_counts": quality_report.insurance_counts,
            "social_history_counts": quality_report.social_history_counts,
        },
        "generation_policy": {
            "clinical_text_model_policy_counts": (
                quality_report.clinical_text_model_policy_counts
            ),
            "time_series_model_policy_counts": (
                quality_report.time_series_model_policy_counts
            ),
            "imaging_model_policy_counts": quality_report.imaging_model_policy_counts,
            "image_validator_policy_counts": (
                quality_report.image_validator_policy_counts
            ),
        },
        "benchmark": benchmark or {},
        "benchmark_suite": _benchmark_suite_summary(benchmark_suite),
        "objective_coverage": objective_coverage or {},
        "seeded_references": seeded_references or {},
        "audit_artifacts": sorted(audit_artifacts or []),
        "limitations": limitations,
    }


def infer_dataset_origin_flags(records: Iterable[SyntheticRecord]) -> dict[str, bool]:
    """Infer high-level dataset origin flags from record provenance."""
    record_list = list(records)
    if not record_list:
        return {"synthetic_data": True, "real_patient_data": False}
    real_patient_data = any(
        _record_may_contain_real_patient_data(record) for record in record_list
    )
    synthetic_data = any(
        not _record_may_contain_real_patient_data(record) for record in record_list
    )
    return {
        "synthetic_data": synthetic_data,
        "real_patient_data": real_patient_data,
    }


def _record_may_contain_real_patient_data(record: SyntheticRecord) -> bool:
    generator = record.provenance.generator
    if generator == "huggingface-reference-import":
        return True
    if generator in SYNTHETIC_REFERENCE_GENERATORS:
        return False
    return bool(record.metadata.get("reference_dataset")) and not _metadata_marks_synthetic(
        record.metadata,
    )


def _metadata_marks_synthetic(metadata: dict[str, Any]) -> bool:
    value = metadata.get("synthetic")
    if isinstance(value, bool):
        return value
    source = metadata.get("reference_dataset")
    return isinstance(source, str) and "synthetic" in source.lower()


def _benchmark_suite_summary(benchmark_suite: dict[str, Any] | None) -> dict[str, Any]:
    if not benchmark_suite:
        return {}
    return {
        "passed": benchmark_suite.get("passed"),
        "reference_count": benchmark_suite.get("reference_count"),
        "mean_overall_score": benchmark_suite.get("mean_overall_score"),
        "recommended_reference_keys": benchmark_suite.get(
            "recommended_reference_keys",
            [],
        ),
        "task_export_results": benchmark_suite.get("task_export_results", {}),
    }
