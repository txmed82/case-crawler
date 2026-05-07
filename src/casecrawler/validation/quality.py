from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from casecrawler.models.evaluation import DatasetQualityReport
from casecrawler.models.synthetic import SyntheticRecord


def build_dataset_quality_report(
    dataset_id: str,
    records: list[SyntheticRecord],
    *,
    effective_approved: Callable[[SyntheticRecord], bool | None] | None = None,
) -> DatasetQualityReport:
    approval_fn = effective_approved or _validation_approved
    modality_counts: Counter[str] = Counter()
    issue_counts_by_field: Counter[str] = Counter()
    approved_count = 0
    blocking_issue_count = 0
    warning_issue_count = 0

    for record in records:
        if approval_fn(record) is True:
            approved_count += 1
        for modality in record.modalities:
            modality_counts[modality.value] += 1
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
        blocking_issue_count=blocking_issue_count,
        warning_issue_count=warning_issue_count,
        issue_counts_by_field=dict(sorted(issue_counts_by_field.items())),
        recommendations=recommendations,
    )


def _validation_approved(record: SyntheticRecord) -> bool | None:
    return None if record.validation is None else record.validation.approved


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
    if "clinical_text" not in modality_counts:
        recommendations.append("Add clinical text records for supervised fine-tuning tasks.")
    return recommendations
