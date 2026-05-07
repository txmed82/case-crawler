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
) -> DatasetQualityReport:
    approval_fn = effective_approved or _validation_approved
    modality_counts: Counter[str] = Counter()
    artifact_counts: Counter[str] = Counter()
    note_type_counts: Counter[str] = Counter()
    issue_counts_by_field: Counter[str] = Counter()
    approved_count = 0
    blocking_issue_count = 0
    warning_issue_count = 0

    for record in records:
        if approval_fn(record) is True:
            approved_count += 1
        for modality in record.modalities:
            modality_counts[modality.value] += 1
        _count_artifacts(record, artifact_counts, note_type_counts)
        blocking_issue_count += _count_missing_declared_artifacts(
            record,
            issue_counts_by_field,
        )
        blocking_issue_count += _count_missing_expected_documents(
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
    )
    return DatasetQualityReport(
        dataset_id=dataset_id,
        record_count=record_count,
        approved_count=approved_count,
        approval_rate=round(approval_rate, 4),
        export_ready=record_count > 0
        and approved_count == record_count
        and blocking_issue_count == 0,
        modality_counts=dict(sorted(modality_counts.items())),
        artifact_counts=dict(sorted(artifact_counts.items())),
        note_type_counts=dict(sorted(note_type_counts.items())),
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
) -> None:
    documents = len(record.documents)
    artifact_counts["documents"] += documents
    artifact_counts["messy_documents"] += sum(1 for doc in record.documents if doc.messy_text)
    artifact_counts["labs"] += len(record.labs)
    artifact_counts["vitals"] += len(record.vitals)
    artifact_counts["medications"] += len(record.medication_history)
    artifact_counts["time_series_channels"] += len(record.time_series)
    artifact_counts["time_series_waveform_channels"] += sum(
        1
        for channel in record.time_series
        if _is_waveform_channel(channel.name, channel.sampling_rate_hz)
    )
    artifact_counts["time_series_points"] += sum(
        len(channel.points) for channel in record.time_series
    )
    artifact_counts["imaging_assets"] += len(record.imaging)
    artifact_counts["imaging_labels"] += sum(len(asset.labels) for asset in record.imaging)
    for doc in record.documents:
        note_type_counts[doc.note_type] += 1


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
    if Modality.IMAGING in record.modalities:
        expected.add("radiology_report")

    present = {document.note_type for document in record.documents}
    missing = 0
    for note_type in sorted(expected - present):
        issue_counts_by_field[f"documents.{note_type}.missing"] += 1
        missing += 1
    return missing


def _is_waveform_channel(name: str, sampling_rate_hz: float | None) -> bool:
    if sampling_rate_hz:
        return True
    normalized = name.lower()
    return normalized.startswith("ecg") or normalized in {"pleth", "arterial_waveform"}


def _recommendations(
    *,
    record_count: int,
    approved_count: int,
    blocking_issue_count: int,
    issue_counts_by_field: Counter[str],
    modality_counts: Counter[str],
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
    if any(field.startswith("documents.") and field.endswith(".missing") for field in issue_counts_by_field):
        recommendations.append("Add expected clinical document types before fine-tuning export.")
    if "clinical_text" not in modality_counts:
        recommendations.append("Add clinical text records for supervised fine-tuning tasks.")
    return recommendations
