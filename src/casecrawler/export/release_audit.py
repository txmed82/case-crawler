from __future__ import annotations

from typing import Any

from casecrawler.models.evaluation import DatasetQualityReport


OBJECTIVE_SUMMARY = (
    "Generate multimodal synthetic healthcare training data with labs, vitals, "
    "medication history, nursing notes, physician notes, radiology reports, "
    "radiology images, validation references, and fine-tuning-ready exports."
)

OBJECTIVE_COVERAGE_KEYS = frozenset(
    {
        "records",
        "cohort_similarity",
        "structured_ehr",
        "labs",
        "vitals",
        "medication_history",
        "physician_notes",
        "nursing_notes",
        "time_series",
        "radiology_reports",
        "radiology_images",
        "privacy_safety",
        "validation_references",
        "fine_tuning_exports",
        "release_audit_artifacts",
    }
)


def build_objective_coverage_audit(
    *,
    quality_report: DatasetQualityReport,
    benchmark_suite: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Map the product objective to release-package evidence."""
    coverage = quality_report.core_artifact_coverage
    criteria = {
        "records": _criterion(
            "Synthetic records are generated.",
            coverage.get("records") is True,
            ["quality_report.json", "manifest.json"],
            {"record_count": quality_report.record_count},
        ),
        "structured_ehr": _criterion(
            "Structured EHR artifacts are present.",
            coverage.get("structured_ehr") is True,
            ["quality_report.json"],
            {"modality_counts": quality_report.modality_counts},
        ),
        "cohort_similarity": _criterion(
            "Cohort demographics and distributions are compared to validation references.",
            _cohort_similarity_satisfied(benchmark_suite),
            ["benchmark_suite_report.json"],
            _cohort_similarity_evidence(benchmark_suite, quality_report),
        ),
        "labs": _criterion(
            "Lab observations and lab reports are present.",
            coverage.get("labs") is True and coverage.get("lab_reports") is True,
            ["quality_report.json"],
            {
                "lab_units": quality_report.lab_unit_counts,
                "lab_reports": coverage.get("lab_reports") is True,
            },
        ),
        "vitals": _criterion(
            "Vital observations and vital-sign flowsheets are present.",
            coverage.get("vitals") is True
            and coverage.get("vital_signs_flowsheets") is True,
            ["quality_report.json"],
            {
                "vital_units": quality_report.vital_unit_counts,
                "vital_signs_flowsheets": (
                    coverage.get("vital_signs_flowsheets") is True
                ),
            },
        ),
        "medication_history": _criterion(
            "Medication history and medication administration records are present.",
            coverage.get("medication_history") is True
            and coverage.get("medication_administration_records") is True,
            ["quality_report.json"],
            {
                "medication_routes": quality_report.medication_route_counts,
                "medication_doses": quality_report.medication_dose_counts,
            },
        ),
        "physician_notes": _criterion(
            "Physician-authored notes are present.",
            coverage.get("physician_notes") is True,
            ["quality_report.json"],
            {"note_type_counts": quality_report.note_type_counts},
        ),
        "nursing_notes": _criterion(
            "Nursing notes are present.",
            coverage.get("nursing_notes") is True,
            ["quality_report.json"],
            {"note_type_counts": quality_report.note_type_counts},
        ),
        "time_series": _criterion(
            "Time-series channels are present and exportable.",
            coverage.get("time_series") is True,
            ["quality_report.json", "manifest.json"],
            {
                "time_series_channel_counts": quality_report.time_series_channel_counts,
                "mean_time_series_points": quality_report.mean_time_series_points,
            },
        ),
        "radiology_reports": _criterion(
            "Radiology reports are present.",
            coverage.get("radiology_reports") is True,
            ["quality_report.json"],
            {
                "mean_imaging_report_chars": quality_report.mean_imaging_report_chars,
                "label_evidence_rate": (
                    quality_report.imaging_report_label_evidence_rate
                ),
            },
        ),
        "radiology_images": _criterion(
            "Radiology image artifacts are present and packaged.",
            coverage.get("radiology_images") is True
            and bool(manifest.get("image_artifacts")),
            ["quality_report.json", "manifest.json"],
            {
                "image_artifact_count": len(manifest.get("image_artifacts", {})),
                "mean_width": quality_report.mean_imaging_width,
                "mean_height": quality_report.mean_imaging_height,
            },
        ),
        "privacy_safety": _criterion(
            "Privacy and memorization-risk validation has no blocking findings.",
            _privacy_safety_satisfied(quality_report),
            ["quality_report.json"],
            {
                "blocking_issue_count": quality_report.blocking_issue_count,
                "privacy_issue_counts": _privacy_issue_counts(quality_report),
            },
        ),
        "validation_references": _criterion(
            "Generated data is compared to imported validation references.",
            quality_report.benchmark_ready is True
            and benchmark_suite.get("passed") is True
            and int(benchmark_suite.get("reference_count", 0)) > 0,
            ["benchmark_report.json", "benchmark_suite_report.json"],
            {
                "recommended_reference_keys": quality_report.recommended_reference_keys,
                "reference_count": benchmark_suite.get("reference_count"),
                "task_export_results": benchmark_suite.get("task_export_results", {}),
            },
        ),
        "fine_tuning_exports": _criterion(
            "Fine-tuning export package is ready.",
            quality_report.export_ready is True
            and bool(manifest.get("task_coverage")),
            ["manifest.json", "quality_report.json"],
            {"task_coverage": manifest.get("task_coverage", {})},
        ),
        "release_audit_artifacts": _criterion(
            "Release audit artifacts are present.",
            _has_required_audit_artifacts(manifest),
            ["manifest.json"],
            {"audit_artifacts": sorted((manifest.get("audit_artifacts") or {}).keys())},
        ),
    }
    missing = [key for key, item in criteria.items() if item["satisfied"] is not True]
    return {
        "objective": OBJECTIVE_SUMMARY,
        "criteria": criteria,
        "complete": not missing,
        "missing": missing,
    }


def _criterion(
    requirement: str,
    satisfied: bool,
    artifacts: list[str],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "requirement": requirement,
        "satisfied": bool(satisfied),
        "artifacts": artifacts,
        "evidence": evidence,
    }


def _privacy_safety_satisfied(quality_report: DatasetQualityReport) -> bool:
    return (
        quality_report.record_count > 0
        and quality_report.blocking_issue_count == 0
        and not _privacy_issue_counts(quality_report)
    )


def _privacy_issue_counts(quality_report: DatasetQualityReport) -> dict[str, int]:
    return {
        field: count
        for field, count in quality_report.issue_counts_by_field.items()
        if field.startswith("privacy")
    }


def _cohort_similarity_satisfied(benchmark_suite: dict[str, Any]) -> bool:
    if benchmark_suite.get("passed") is not True:
        return False
    metric_names = set(_benchmark_metric_names(benchmark_suite))
    required_metrics = {
        "record_count",
        "mean_age",
        "sex_distribution",
        "race_distribution",
        "ethnicity_distribution",
        "insurance_distribution",
        "social_history_distribution:smoking_status",
        "modality_overlap",
    }
    return required_metrics.issubset(metric_names)


def _cohort_similarity_evidence(
    benchmark_suite: dict[str, Any],
    quality_report: DatasetQualityReport,
) -> dict[str, Any]:
    metric_names = sorted(set(_benchmark_metric_names(benchmark_suite)))
    return {
        "reference_count": benchmark_suite.get("reference_count"),
        "mean_overall_score": benchmark_suite.get("mean_overall_score"),
        "required_metrics": [
            "record_count",
            "mean_age",
            "sex_distribution",
            "race_distribution",
            "ethnicity_distribution",
            "insurance_distribution",
            "social_history_distribution:smoking_status",
            "modality_overlap",
        ],
        "generated_race_counts": quality_report.race_counts,
        "generated_ethnicity_counts": quality_report.ethnicity_counts,
        "generated_insurance_counts": quality_report.insurance_counts,
        "generated_social_history_counts": quality_report.social_history_counts,
        "available_metrics": metric_names,
    }


def _benchmark_metric_names(benchmark_suite: dict[str, Any]) -> list[str]:
    names: list[str] = []
    results = benchmark_suite.get("results")
    if not isinstance(results, list):
        return names
    for result in results:
        if not isinstance(result, dict):
            continue
        report = result.get("report")
        if not isinstance(report, dict):
            continue
        metrics = report.get("metrics")
        if not isinstance(metrics, list):
            continue
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            name = metric.get("name")
            if isinstance(name, str) and name:
                names.append(name)
    return names


def _has_required_audit_artifacts(manifest: dict[str, Any]) -> bool:
    audit_artifacts = manifest.get("audit_artifacts")
    if not isinstance(audit_artifacts, dict):
        return False
    required = {
        "benchmark_profile.json",
        "benchmark_report.json",
        "benchmark_suite_report.json",
        "dataset_card.md",
        "model_card.md",
        "quality_report.json",
        "release_package_summary.json",
    }
    return required.issubset(audit_artifacts)
