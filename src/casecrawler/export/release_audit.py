from __future__ import annotations

from typing import Any

from casecrawler.models.evaluation import DatasetQualityReport


OBJECTIVE_SUMMARY = (
    "Generate multimodal synthetic healthcare training data with labs, vitals, "
    "medication history, nursing notes, physician notes, radiology reports, "
    "radiology images, validation references, and fine-tuning-ready exports."
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
