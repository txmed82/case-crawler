from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from casecrawler.models.evaluation import DatasetQualityReport
from casecrawler.models.synthetic import Modality, SyntheticRecord


def build_dataset_quality_report(
    dataset_id: str,
    records: list[SyntheticRecord],
    *,
    effective_approved: Callable[[SyntheticRecord], bool | None] | None = None,
    benchmark_plan: dict | None = None,
) -> DatasetQualityReport:
    approval_fn = effective_approved or _validation_approved
    modality_counts: Counter[str] = Counter()
    artifact_counts: Counter[str] = Counter()
    note_type_counts: Counter[str] = Counter()
    extracted_fact_key_counts: Counter[str] = Counter()
    time_series_backend_counts: Counter[str] = Counter()
    imaging_backend_counts: Counter[str] = Counter()
    imaging_model_policy_counts: Counter[str] = Counter()
    issue_counts_by_field: Counter[str] = Counter()
    approved_count = 0
    blocking_issue_count = 0
    warning_issue_count = 0

    for record in records:
        if approval_fn(record) is True:
            approved_count += 1
        for modality in record.modalities:
            modality_counts[modality.value] += 1
        _count_artifacts(
            record,
            artifact_counts,
            note_type_counts,
            extracted_fact_key_counts,
            time_series_backend_counts,
            imaging_backend_counts,
            imaging_model_policy_counts,
        )
        blocking_issue_count += _count_missing_declared_artifacts(
            record,
            issue_counts_by_field,
        )
        blocking_issue_count += _count_missing_structured_ehr_artifacts(
            record,
            issue_counts_by_field,
        )
        blocking_issue_count += _count_missing_expected_documents(
            record,
            issue_counts_by_field,
        )
        blocking_issue_count += _count_mismatched_document_author_roles(
            record,
            issue_counts_by_field,
        )
        blocking_issue_count += _count_missing_imaging_model_policy(
            record,
            issue_counts_by_field,
        )
        if record.validation is None:
            issue_counts_by_field["validation.missing"] += 1
            blocking_issue_count += 1
            continue
        for issue in record.validation.issues:
            issue_counts_by_field[issue.field] += 1
            if issue.severity == "error":
                blocking_issue_count += 1
            else:
                warning_issue_count += 1

    record_count = len(records)
    approval_rate = approved_count / record_count if record_count else 0.0
    recommendations = _recommendations(
        record_count=record_count,
        approved_count=approved_count,
        blocking_issue_count=blocking_issue_count,
        issue_counts_by_field=issue_counts_by_field,
        modality_counts=modality_counts,
        benchmark_plan=benchmark_plan,
    )
    benchmark_summary = _benchmark_summary(benchmark_plan)
    return DatasetQualityReport(
        dataset_id=dataset_id,
        record_count=record_count,
        approved_count=approved_count,
        approval_rate=round(approval_rate, 4),
        export_ready=record_count > 0
        and approved_count == record_count
        and blocking_issue_count == 0,
        benchmark_ready=benchmark_summary["ready"],
        recommended_reference_keys=benchmark_summary["recommended_reference_keys"],
        resolved_reference_dataset_id=benchmark_summary["resolved_reference_dataset_id"],
        missing_reference_keys=benchmark_summary["missing_reference_keys"],
        benchmark_thresholds=benchmark_summary["thresholds"],
        modality_counts=dict(sorted(modality_counts.items())),
        artifact_counts=dict(sorted(artifact_counts.items())),
        note_type_counts=dict(sorted(note_type_counts.items())),
        extracted_fact_key_counts=dict(sorted(extracted_fact_key_counts.items())),
        time_series_backend_counts=dict(sorted(time_series_backend_counts.items())),
        imaging_backend_counts=dict(sorted(imaging_backend_counts.items())),
        imaging_model_policy_counts=dict(sorted(imaging_model_policy_counts.items())),
        blocking_issue_count=blocking_issue_count,
        warning_issue_count=warning_issue_count,
        issue_counts_by_field=dict(sorted(issue_counts_by_field.items())),
        recommendations=recommendations,
    )


def _validation_approved(record: SyntheticRecord) -> bool | None:
    return None if record.validation is None else record.validation.approved


def _count_artifacts(
    record: SyntheticRecord,
    artifact_counts: Counter[str],
    note_type_counts: Counter[str],
    extracted_fact_key_counts: Counter[str],
    time_series_backend_counts: Counter[str],
    imaging_backend_counts: Counter[str],
    imaging_model_policy_counts: Counter[str],
) -> None:
    documents = len(record.documents)
    artifact_counts["documents"] += documents
    artifact_counts["messy_documents"] += sum(1 for doc in record.documents if doc.messy_text)
    artifact_counts["encounters"] += len(record.encounters)
    artifact_counts["diagnoses"] += sum(
        len(encounter.diagnoses) for encounter in record.encounters
    )
    artifact_counts["procedures"] += sum(
        len(encounter.procedures) for encounter in record.encounters
    )
    artifact_counts["labs"] += len(record.labs)
    artifact_counts["vitals"] += len(record.vitals)
    artifact_counts["medications"] += len(record.medication_history)
    artifact_counts["time_series_channels"] += len(record.time_series)
    for channel in record.time_series:
        time_series_backend_counts[channel.generation_backend or "unknown"] += 1
    artifact_counts["time_series_waveform_channels"] += sum(
        1
        for channel in record.time_series
        if _is_waveform_channel(channel.name, channel.sampling_rate_hz)
    )
    artifact_counts["time_series_points"] += sum(
        len(channel.points) for channel in record.time_series
    )
    artifact_counts["imaging_assets"] += len(record.imaging)
    for asset in record.imaging:
        imaging_backend_counts[asset.generation_backend or "unknown"] += 1
    policy_key = _imaging_model_policy_key(record)
    if policy_key:
        imaging_model_policy_counts[policy_key] += len(record.imaging)
    artifact_counts["imaging_labels"] += sum(len(asset.labels) for asset in record.imaging)
    for doc in record.documents:
        note_type_counts[doc.note_type] += 1
        for key, value in doc.extracted_facts.items():
            if _has_fact_value(value):
                extracted_fact_key_counts[_fact_key(key)] += 1


def _count_missing_declared_artifacts(
    record: SyntheticRecord,
    issue_counts_by_field: Counter[str],
) -> int:
    missing = 0
    checks = {
        Modality.CLINICAL_TEXT: ("clinical_text.missing_artifacts", bool(record.documents)),
        Modality.LABS: ("labs.missing_artifacts", bool(record.labs)),
        Modality.VITALS: ("vitals.missing_artifacts", bool(record.vitals)),
        Modality.TIME_SERIES: ("time_series.missing_artifacts", bool(record.time_series)),
        Modality.IMAGING: ("imaging.missing_artifacts", bool(record.imaging)),
    }
    for modality in record.modalities:
        check = checks.get(modality)
        if check is None:
            continue
        field, has_artifacts = check
        if not has_artifacts:
            issue_counts_by_field[field] += 1
            missing += 1
    return missing


def _count_missing_structured_ehr_artifacts(
    record: SyntheticRecord,
    issue_counts_by_field: Counter[str],
) -> int:
    if Modality.STRUCTURED_EHR not in record.modalities:
        return 0
    missing = 0
    checks = {
        "structured_ehr.encounters.missing": bool(record.encounters),
        "structured_ehr.diagnoses.missing": any(
            encounter.diagnoses for encounter in record.encounters
        ),
        "structured_ehr.medication_history.missing": bool(record.medication_history),
    }
    for field, has_artifact in checks.items():
        if not has_artifact:
            issue_counts_by_field[field] += 1
            missing += 1
    return missing


def _count_missing_expected_documents(
    record: SyntheticRecord,
    issue_counts_by_field: Counter[str],
) -> int:
    if Modality.CLINICAL_TEXT not in record.modalities:
        return 0
    expected = {
        "ed_note",
        "progress_note",
        "nursing_note",
        "discharge_summary",
    }
    if Modality.LABS in record.modalities or record.labs:
        expected.add("lab_report")
    if Modality.VITALS in record.modalities or record.vitals:
        expected.add("vital_signs_flowsheet")
    if Modality.STRUCTURED_EHR in record.modalities or record.medication_history:
        expected.add("medication_administration_record")
    if Modality.IMAGING in record.modalities:
        expected.add("radiology_report")

    present = {document.note_type for document in record.documents}
    missing = 0
    for note_type in sorted(expected - present):
        issue_counts_by_field[f"documents.{note_type}.missing"] += 1
        missing += 1
    return missing


def _count_mismatched_document_author_roles(
    record: SyntheticRecord,
    issue_counts_by_field: Counter[str],
) -> int:
    expected_roles = {
        "ed_note": {"physician"},
        "progress_note": {"physician"},
        "nursing_note": {"nurse"},
        "discharge_summary": {"physician"},
        "lab_report": {"laboratory"},
        "vital_signs_flowsheet": {"nurse"},
        "medication_administration_record": {"nurse", "pharmacist"},
        "radiology_report": {"radiologist"},
    }
    mismatches = 0
    for document in record.documents:
        expected = expected_roles.get(document.note_type)
        if expected is None:
            continue
        if document.author_role.strip().lower() not in expected:
            issue_counts_by_field[f"documents.{document.note_type}.author_role"] += 1
            mismatches += 1
    return mismatches


def _count_missing_imaging_model_policy(
    record: SyntheticRecord,
    issue_counts_by_field: Counter[str],
) -> int:
    if not any(
        asset.generation_backend.startswith("diffusers:")
        for asset in record.imaging
    ):
        return 0
    if _imaging_model_policy_key(record):
        return 0
    issue_counts_by_field["imaging.model_policy.missing"] += 1
    return 1


def _is_waveform_channel(name: str, sampling_rate_hz: float | None) -> bool:
    if sampling_rate_hz:
        return True
    normalized = name.lower()
    return normalized.startswith("ecg") or normalized in {"pleth", "arterial_waveform"}


def _has_fact_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list | tuple | set | dict):
        return bool(value)
    return True


def _fact_key(value: str) -> str:
    return "_".join(value.lower().replace("-", "_").split())


def _imaging_model_policy_key(record: SyntheticRecord) -> str | None:
    policy = record.metadata.get("imaging_model_policy")
    if not isinstance(policy, dict):
        return None
    profile = _policy_value(policy.get("profile"), "unspecified")
    license_name = _policy_value(policy.get("license"), "unspecified")
    use_policy = _policy_value(policy.get("use_policy"), "review_license_before_use")
    gated = str(bool(policy.get("gated"))).lower()
    return (
        f"profile={profile}|license={license_name}|"
        f"gated={gated}|use_policy={use_policy}"
    )


def _policy_value(value: object, fallback: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return fallback
    return "_".join(value.lower().replace("-", "_").split())


def _recommendations(
    *,
    record_count: int,
    approved_count: int,
    blocking_issue_count: int,
    issue_counts_by_field: Counter[str],
    modality_counts: Counter[str],
    benchmark_plan: dict | None = None,
) -> list[str]:
    recommendations: list[str] = []
    if record_count == 0:
        recommendations.append("Generate or import records before exporting.")
    if approved_count < record_count:
        recommendations.append("Review or regenerate unapproved records before fine-tuning export.")
    if blocking_issue_count:
        recommendations.append("Resolve blocking validation issues before marking the dataset ready.")
    if issue_counts_by_field.get("validation.missing", 0):
        recommendations.append("Run validation for records that do not have validation reports.")
    if any(field.endswith(".missing_artifacts") for field in issue_counts_by_field):
        recommendations.append("Resolve missing modality artifacts before fine-tuning export.")
    if any(field.startswith("structured_ehr.") for field in issue_counts_by_field):
        recommendations.append("Resolve missing structured EHR artifacts before fine-tuning export.")
    if any(field.startswith("documents.") and field.endswith(".missing") for field in issue_counts_by_field):
        recommendations.append("Add expected clinical document types before fine-tuning export.")
    if any(field.startswith("documents.") and field.endswith(".author_role") for field in issue_counts_by_field):
        recommendations.append("Fix expected clinical document author roles before fine-tuning export.")
    if issue_counts_by_field.get("imaging.model_policy.missing", 0):
        recommendations.append(
            "Attach imaging model policy metadata before exporting generated image datasets."
        )
    if "clinical_text" not in modality_counts:
        recommendations.append("Add clinical text records for supervised fine-tuning tasks.")
    benchmark_summary = _benchmark_summary(benchmark_plan)
    if (
        benchmark_summary["recommended_reference_keys"]
        and benchmark_summary["ready"] is False
    ):
        recommendations.append(
            "Import a recommended reference dataset before benchmark-gated release."
        )
    return recommendations


def _benchmark_summary(benchmark_plan: dict | None) -> dict:
    if not isinstance(benchmark_plan, dict):
        return {
            "ready": None,
            "recommended_reference_keys": [],
            "resolved_reference_dataset_id": None,
            "missing_reference_keys": [],
            "thresholds": {},
        }
    recommended_reference_keys = _string_list(
        benchmark_plan.get("recommended_reference_keys")
    )
    return {
        "ready": bool(benchmark_plan.get("ready"))
        if recommended_reference_keys
        else None,
        "recommended_reference_keys": recommended_reference_keys,
        "resolved_reference_dataset_id": _string_or_none(
            benchmark_plan.get("resolved_reference_dataset_id")
        ),
        "missing_reference_keys": _string_list(
            benchmark_plan.get("missing_reference_keys")
        ),
        "thresholds": _thresholds(benchmark_plan.get("thresholds")),
    }


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _string_or_none(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _thresholds(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    thresholds = {}
    for key in ("min_overall_score", "min_metric_score"):
        score = value.get(key)
        if isinstance(score, int | float):
            thresholds[key] = float(score)
    return thresholds
