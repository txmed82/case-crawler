from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from casecrawler.models.dataset import ExportFormat
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
    diagnosis_code_system_counts: Counter[str] = Counter()
    diagnosis_code_counts: Counter[str] = Counter()
    phi_entity_counts: Counter[str] = Counter()
    time_series_backend_counts: Counter[str] = Counter()
    imaging_backend_counts: Counter[str] = Counter()
    imaging_model_policy_counts: Counter[str] = Counter()
    lab_unit_counts: Counter[str] = Counter()
    vital_unit_counts: Counter[str] = Counter()
    lab_numeric_values: dict[str, list[float]] = {}
    vital_numeric_values: dict[str, list[float]] = {}
    time_series_numeric_values: dict[str, list[float]] = {}
    issue_counts_by_field: Counter[str] = Counter()
    longitudinal_values: list[int] = []
    encounter_spans: list[float] = []
    observations_per_encounter: list[float] = []
    approved_count = 0
    blocking_issue_count = 0
    warning_issue_count = 0
    modality_alignment_scores: list[float] = []

    for record in records:
        if approval_fn(record) is True:
            approved_count += 1
        longitudinal_values.append(1 if len(record.encounters) > 1 else 0)
        encounter_span = _encounter_span_hours(record)
        if encounter_span is not None:
            encounter_spans.append(encounter_span)
        if record.encounters:
            observations_per_encounter.append(
                (len(record.labs) + len(record.vitals)) / len(record.encounters)
            )
        for modality in record.modalities:
            modality_counts[modality.value] += 1
        _count_artifacts(
            record,
            artifact_counts,
            note_type_counts,
            extracted_fact_key_counts,
            diagnosis_code_system_counts,
            diagnosis_code_counts,
            phi_entity_counts,
            time_series_backend_counts,
            imaging_backend_counts,
            imaging_model_policy_counts,
            lab_unit_counts,
            vital_unit_counts,
            lab_numeric_values,
            vital_numeric_values,
            time_series_numeric_values,
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
        blocking_issue_count += _count_missing_required_human_review(
            record,
            issue_counts_by_field,
        )
        if record.validation is None:
            issue_counts_by_field["validation.missing"] += 1
            blocking_issue_count += 1
            continue
        if record.validation.modality_alignment_score is not None:
            modality_alignment_scores.append(record.validation.modality_alignment_score)
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
        artifact_counts=artifact_counts,
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
        task_export_reference_readiness=benchmark_summary[
            "task_export_reference_readiness"
        ],
        modality_counts=dict(sorted(modality_counts.items())),
        artifact_counts=dict(sorted(artifact_counts.items())),
        export_profile_readiness=_export_profile_readiness(
            record_count=record_count,
            approved_count=approved_count,
            blocking_issue_count=blocking_issue_count,
            artifact_counts=artifact_counts,
            extracted_fact_key_counts=extracted_fact_key_counts,
        ),
        longitudinal_record_rate=_mean_float(longitudinal_values),
        mean_encounter_span_hours=_mean_float(encounter_spans),
        mean_observations_per_encounter=_mean_float(observations_per_encounter),
        note_type_counts=dict(sorted(note_type_counts.items())),
        extracted_fact_key_counts=dict(sorted(extracted_fact_key_counts.items())),
        lab_unit_counts=dict(sorted(lab_unit_counts.items())),
        lab_numeric_summaries=_numeric_summaries(lab_numeric_values),
        vital_unit_counts=dict(sorted(vital_unit_counts.items())),
        vital_numeric_summaries=_numeric_summaries(vital_numeric_values),
        diagnosis_code_system_counts=dict(sorted(diagnosis_code_system_counts.items())),
        diagnosis_code_counts=dict(sorted(diagnosis_code_counts.items())),
        phi_entity_counts=dict(sorted(phi_entity_counts.items())),
        time_series_backend_counts=dict(sorted(time_series_backend_counts.items())),
        time_series_numeric_summaries=_numeric_summaries(time_series_numeric_values),
        imaging_backend_counts=dict(sorted(imaging_backend_counts.items())),
        imaging_model_policy_counts=dict(sorted(imaging_model_policy_counts.items())),
        mean_modality_alignment_score=_mean_float(modality_alignment_scores),
        blocking_issue_count=blocking_issue_count,
        warning_issue_count=warning_issue_count,
        issue_counts_by_field=dict(sorted(issue_counts_by_field.items())),
        recommendations=recommendations,
    )


def export_profile_blocker(report: DatasetQualityReport, export_format: str) -> str | None:
    readiness = report.export_profile_readiness.get(export_format)
    if readiness is None:
        return None
    if readiness.get("ready") is True:
        return None
    missing = readiness.get("missing")
    missing_items = (
        ", ".join(str(item) for item in missing)
        if isinstance(missing, list) and missing
        else "profile requirements"
    )
    reason = readiness.get("reason")
    reason_text = str(reason) if isinstance(reason, str) and reason else ""
    return (
        f"Export profile {export_format} is not ready. "
        f"Missing: {missing_items}. {reason_text}"
    ).strip()


def _validation_approved(record: SyntheticRecord) -> bool | None:
    return None if record.validation is None else record.validation.approved


def _mean_float(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _numeric_summaries(values_by_name: dict[str, list[float]]) -> dict[str, dict[str, float | int]]:
    summaries: dict[str, dict[str, float | int]] = {}
    for name, values in sorted(values_by_name.items()):
        if not values:
            continue
        summaries[name] = {
            "count": len(values),
            "max": round(max(values), 4),
            "mean": round(sum(values) / len(values), 4),
            "min": round(min(values), 4),
        }
    return summaries


def _count_artifacts(
    record: SyntheticRecord,
    artifact_counts: Counter[str],
    note_type_counts: Counter[str],
    extracted_fact_key_counts: Counter[str],
    diagnosis_code_system_counts: Counter[str],
    diagnosis_code_counts: Counter[str],
    phi_entity_counts: Counter[str],
    time_series_backend_counts: Counter[str],
    imaging_backend_counts: Counter[str],
    imaging_model_policy_counts: Counter[str],
    lab_unit_counts: Counter[str],
    vital_unit_counts: Counter[str],
    lab_numeric_values: dict[str, list[float]],
    vital_numeric_values: dict[str, list[float]],
    time_series_numeric_values: dict[str, list[float]],
) -> None:
    documents = len(record.documents)
    artifact_counts["documents"] += documents
    artifact_counts["messy_documents"] += sum(1 for doc in record.documents if doc.messy_text)
    artifact_counts["encounters"] += len(record.encounters)
    artifact_counts["diagnoses"] += sum(
        len(encounter.diagnoses) for encounter in record.encounters
    )
    for encounter in record.encounters:
        for diagnosis in encounter.diagnoses:
            if diagnosis.system:
                diagnosis_code_system_counts[diagnosis.system] += 1
            if diagnosis.code:
                diagnosis_code_counts[_diagnosis_code_key(diagnosis)] += 1
    artifact_counts["procedures"] += sum(
        len(encounter.procedures) for encounter in record.encounters
    )
    artifact_counts["labs"] += len(record.labs)
    for lab in record.labs:
        lab_unit_counts[lab.unit] += 1
        if isinstance(lab.value, (int, float)):
            lab_numeric_values.setdefault(_metric_key(lab.name), []).append(
                float(lab.value)
            )
    artifact_counts["vitals"] += len(record.vitals)
    for vital in record.vitals:
        vital_unit_counts[vital.unit] += 1
        vital_numeric_values.setdefault(_metric_key(vital.name), []).append(
            float(vital.value)
        )
    artifact_counts["medications"] += len(record.medication_history)
    artifact_counts["time_series_channels"] += len(record.time_series)
    for channel in record.time_series:
        time_series_backend_counts[channel.generation_backend or "unknown"] += 1
        _collect_time_series_numeric_values(channel, time_series_numeric_values)
    artifact_counts["time_series_waveform_channels"] += sum(
        1
        for channel in record.time_series
        if _is_waveform_channel(channel.name, channel.sampling_rate_hz)
    )
    artifact_counts["time_series_points"] += sum(
        len(channel.points) for channel in record.time_series
    )
    artifact_counts["imaging_assets"] += len(record.imaging)
    artifact_counts["imaging_file_assets"] += sum(
        1
        for asset in record.imaging
        if asset.file_path and Path(asset.file_path).is_file()
    )
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
        _count_phi_entities(doc.extracted_facts, phi_entity_counts)


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


def _count_missing_required_human_review(
    record: SyntheticRecord,
    issue_counts_by_field: Counter[str],
) -> int:
    if record.metadata.get("require_human_review") is not True:
        return 0
    review = record.metadata.get("human_review")
    if isinstance(review, dict) and review.get("status") == "approved":
        return 0
    issue_counts_by_field["human_review.missing"] += 1
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


def _count_phi_entities(
    extracted_facts: dict,
    phi_entity_counts: Counter[str],
) -> None:
    annotations = extracted_facts.get("phi_annotations")
    if isinstance(annotations, list):
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            entity_type = str(annotation.get("entity_type") or "").strip()
            if entity_type:
                phi_entity_counts[entity_type] += 1
        return
    counts = extracted_facts.get("phi_entity_counts")
    if isinstance(counts, dict):
        for entity_type, count in counts.items():
            if not str(entity_type).strip():
                continue
            try:
                phi_entity_counts[str(entity_type)] += int(count)
            except (TypeError, ValueError):
                continue


def _diagnosis_code_key(diagnosis) -> str:
    system = diagnosis.system or "unspecified"
    code = diagnosis.code or "unspecified"
    return f"{system}:{code}"


def _collect_time_series_numeric_values(
    channel,
    values_by_name: dict[str, list[float]],
) -> None:
    channel_key = _fact_key(channel.name)
    for point in channel.points:
        for value_name, value in point.values.items():
            values_by_name.setdefault(
                f"{channel_key}.{_fact_key(str(value_name))}",
                [],
            ).append(float(value))


def _encounter_span_hours(record: SyntheticRecord) -> float | None:
    if len(record.encounters) < 2:
        return None
    timestamps: list[datetime] = []
    for encounter in record.encounters:
        start = _parse_datetime(encounter.start)
        if start is not None:
            timestamps.append(start)
        if encounter.end:
            end = _parse_datetime(encounter.end)
            if end is not None:
                timestamps.append(end)
    if len(timestamps) < 2:
        return None
    return round((max(timestamps) - min(timestamps)).total_seconds() / 3600, 4)


def _parse_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _metric_key(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").split())


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
    artifact_counts: Counter[str],
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
    if artifact_counts.get("imaging_assets", 0) > artifact_counts.get(
        "imaging_file_assets", 0
    ):
        recommendations.append(
            "Attach local image files before multimodal fine-tuning export."
        )
    if issue_counts_by_field.get("human_review.missing", 0):
        recommendations.append(
            "Complete required human review before exporting generated datasets."
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


def _export_profile_readiness(
    *,
    record_count: int,
    approved_count: int,
    blocking_issue_count: int,
    artifact_counts: Counter[str],
    extracted_fact_key_counts: Counter[str],
) -> dict[str, dict[str, object]]:
    base_ready = record_count > 0 and approved_count == record_count and blocking_issue_count == 0
    checks = {
        ExportFormat.RAW_JSONL: {},
        ExportFormat.SFT_JSONL: {"documents": 1},
        ExportFormat.CHAT_JSONL: {"documents": 1},
        ExportFormat.DPO_JSONL: {"documents": 1},
        ExportFormat.RL_JSONL: {"documents": 1},
        ExportFormat.NOTE_FACT_SFT_JSONL: {"documents": 1, "extracted_facts": 1},
        ExportFormat.CLINICAL_OBSERVATION_JSONL: {"labs_or_vitals": 1},
        ExportFormat.MEDICATION_RECONCILIATION_JSONL: {"medications": 1},
        ExportFormat.TOOL_CALL_JSONL: {"documents": 1, "structured_ehr": 1},
        ExportFormat.FHIR_NDJSON: {"structured_ehr": 1},
        ExportFormat.PARQUET: {},
        ExportFormat.TIME_SERIES_JSONL: {
            "time_series_channels": 1,
            "time_series_points": 1,
        },
        ExportFormat.MULTIMODAL_JSONL: {
            "imaging_assets": 1,
            "imaging_file_assets": 1,
        },
    }
    return {
        export_format.value: _profile_readiness(
            export_format,
            required,
            base_ready=base_ready,
            artifact_counts=artifact_counts,
            extracted_fact_key_counts=extracted_fact_key_counts,
        )
        for export_format, required in checks.items()
    }


def _profile_readiness(
    export_format: ExportFormat,
    required: dict[str, int],
    *,
    base_ready: bool,
    artifact_counts: Counter[str],
    extracted_fact_key_counts: Counter[str],
) -> dict[str, object]:
    missing = [
        requirement
        for requirement, minimum in required.items()
        if _artifact_count(
            requirement,
            artifact_counts=artifact_counts,
            extracted_fact_key_counts=extracted_fact_key_counts,
        )
        < minimum
    ]
    return {
        "ready": base_ready and not missing,
        "required": dict(required),
        "available": {
            requirement: _artifact_count(
                requirement,
                artifact_counts=artifact_counts,
                extracted_fact_key_counts=extracted_fact_key_counts,
            )
            for requirement in required
        },
        "missing": missing,
        "reason": _profile_readiness_reason(export_format, base_ready, missing),
    }


def _artifact_count(
    requirement: str,
    *,
    artifact_counts: Counter[str],
    extracted_fact_key_counts: Counter[str],
) -> int:
    if requirement == "labs_or_vitals":
        return artifact_counts.get("labs", 0) + artifact_counts.get("vitals", 0)
    if requirement == "structured_ehr":
        return min(
            artifact_counts.get("encounters", 0),
            artifact_counts.get("diagnoses", 0),
        )
    if requirement == "extracted_facts":
        return sum(extracted_fact_key_counts.values())
    return artifact_counts.get(requirement, 0)


def _profile_readiness_reason(
    export_format: ExportFormat,
    base_ready: bool,
    missing: list[str],
) -> str:
    if not base_ready:
        return "Dataset must have records, approvals, and no blocking quality issues."
    if missing:
        return (
            f"{export_format.value} requires additional artifacts: "
            f"{', '.join(missing)}."
        )
    return f"{export_format.value} has the required artifacts."


def _benchmark_summary(benchmark_plan: dict | None) -> dict:
    if not isinstance(benchmark_plan, dict):
        return {
            "ready": None,
            "recommended_reference_keys": [],
            "resolved_reference_dataset_id": None,
            "missing_reference_keys": [],
            "thresholds": {},
            "task_export_reference_readiness": {},
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
        "task_export_reference_readiness": _dict_or_empty(
            benchmark_plan.get("task_export_reference_readiness")
        ),
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


def _dict_or_empty(value: object) -> dict:
    if not isinstance(value, dict):
        return {}
    return value


def _thresholds(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    thresholds = {}
    for key in ("min_overall_score", "min_metric_score"):
        score = value.get(key)
        if isinstance(score, int | float):
            thresholds[key] = float(score)
    return thresholds
