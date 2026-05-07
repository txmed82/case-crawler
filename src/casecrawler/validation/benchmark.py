from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from casecrawler.models.evaluation import BenchmarkMetric, BenchmarkReport, CohortProfile
from casecrawler.models.synthetic import SyntheticRecord


class DatasetBenchmark:
    def compare(
        self,
        generated_records: list[SyntheticRecord],
        reference_records: list[SyntheticRecord],
    ) -> BenchmarkReport:
        if not generated_records:
            raise ValueError("generated_records must not be empty.")
        if not reference_records:
            raise ValueError("reference_records must not be empty.")

        generated_profile = profile_records(generated_records)
        reference_profile = profile_records(reference_records)
        metrics = [
            _ratio_metric(
                "record_count",
                generated_profile.record_count,
                reference_profile.record_count,
            ),
            _jaccard_metric(
                "modality_overlap",
                set(generated_profile.modality_counts),
                set(reference_profile.modality_counts),
            ),
            _closeness_metric(
                "mean_age",
                generated_profile.mean_age,
                reference_profile.mean_age,
                tolerance=25.0,
            ),
            _distribution_metric(
                "sex_distribution",
                generated_profile.sex_counts,
                reference_profile.sex_counts,
            ),
            _closeness_metric(
                "mean_document_chars",
                generated_profile.mean_document_chars,
                reference_profile.mean_document_chars,
                tolerance=2500.0,
            ),
            _jaccard_metric(
                "note_type_overlap",
                set(generated_profile.note_type_counts),
                set(reference_profile.note_type_counts),
            ),
            _jaccard_metric(
                "document_author_role_overlap",
                set(generated_profile.document_author_role_counts),
                set(reference_profile.document_author_role_counts),
            ),
            _distribution_metric(
                "document_author_role_distribution",
                generated_profile.document_author_role_counts,
                reference_profile.document_author_role_counts,
            ),
            _closeness_metric(
                "messy_document_rate",
                generated_profile.messy_document_rate,
                reference_profile.messy_document_rate,
                tolerance=0.5,
            ),
            _jaccard_metric(
                "lab_name_overlap",
                set(generated_profile.lab_name_counts),
                set(reference_profile.lab_name_counts),
            ),
            _distribution_metric(
                "lab_flag_distribution",
                generated_profile.lab_flag_counts,
                reference_profile.lab_flag_counts,
            ),
            *_numeric_summary_metrics(
                prefix="lab_value_mean",
                generated_summaries=generated_profile.lab_numeric_summaries,
                reference_summaries=reference_profile.lab_numeric_summaries,
                tolerance=50.0,
            ),
            _jaccard_metric(
                "vital_name_overlap",
                set(generated_profile.vital_name_counts),
                set(reference_profile.vital_name_counts),
            ),
            *_numeric_summary_metrics(
                prefix="vital_value_mean",
                generated_summaries=generated_profile.vital_numeric_summaries,
                reference_summaries=reference_profile.vital_numeric_summaries,
                tolerance=25.0,
            ),
            _jaccard_metric(
                "medication_name_overlap",
                set(generated_profile.medication_name_counts),
                set(reference_profile.medication_name_counts),
            ),
            _distribution_metric(
                "medication_route_distribution",
                generated_profile.medication_route_counts,
                reference_profile.medication_route_counts,
            ),
            _distribution_metric(
                "medication_status_distribution",
                generated_profile.medication_status_counts,
                reference_profile.medication_status_counts,
            ),
            _jaccard_metric(
                "time_series_channel_overlap",
                set(generated_profile.time_series_channel_counts),
                set(reference_profile.time_series_channel_counts),
            ),
            _closeness_metric(
                "mean_time_series_points",
                generated_profile.mean_time_series_points,
                reference_profile.mean_time_series_points,
                tolerance=12.0,
            ),
            _closeness_metric(
                "mean_time_series_duration_hours",
                generated_profile.mean_time_series_duration_hours,
                reference_profile.mean_time_series_duration_hours,
                tolerance=48.0,
            ),
            _jaccard_metric(
                "imaging_modality_overlap",
                set(generated_profile.imaging_modality_counts),
                set(reference_profile.imaging_modality_counts),
            ),
            _jaccard_metric(
                "imaging_body_region_overlap",
                set(generated_profile.imaging_body_region_counts),
                set(reference_profile.imaging_body_region_counts),
            ),
            _jaccard_metric(
                "imaging_label_overlap",
                set(generated_profile.imaging_label_counts),
                set(reference_profile.imaging_label_counts),
            ),
            _distribution_metric(
                "imaging_label_distribution",
                generated_profile.imaging_label_counts,
                reference_profile.imaging_label_counts,
            ),
            _jaccard_metric(
                "imaging_label_pair_overlap",
                set(generated_profile.imaging_label_pair_counts),
                set(reference_profile.imaging_label_pair_counts),
            ),
            _closeness_metric(
                "approved_rate",
                generated_profile.approved_rate,
                reference_profile.approved_rate,
                tolerance=0.5,
            ),
        ]
        overall = sum(metric.score for metric in metrics) / len(metrics)
        warnings = _warnings(generated_profile, reference_profile, metrics)
        return BenchmarkReport(
            generated_dataset_id=generated_profile.dataset_id,
            reference_dataset_id=reference_profile.dataset_id,
            overall_score=round(overall, 4),
            generated_profile=generated_profile,
            reference_profile=reference_profile,
            metrics=metrics,
            warnings=warnings,
        )


def profile_records(records: list[SyntheticRecord]) -> CohortProfile:
    if not records:
        raise ValueError("records must not be empty.")
    dataset_ids = {record.dataset_id for record in records}
    if len(dataset_ids) != 1:
        raise ValueError(
            "records must all belong to one dataset; got "
            f"{', '.join(sorted(dataset_ids))}."
        )
    modality_counts: Counter[str] = Counter()
    sex_counts: Counter[str] = Counter()
    note_type_counts: Counter[str] = Counter()
    document_author_role_counts: Counter[str] = Counter()
    lab_name_counts: Counter[str] = Counter()
    lab_flag_counts: Counter[str] = Counter()
    vital_name_counts: Counter[str] = Counter()
    medication_name_counts: Counter[str] = Counter()
    medication_route_counts: Counter[str] = Counter()
    medication_status_counts: Counter[str] = Counter()
    time_series_channel_counts: Counter[str] = Counter()
    imaging_modality_counts: Counter[str] = Counter()
    imaging_body_region_counts: Counter[str] = Counter()
    imaging_label_counts: Counter[str] = Counter()
    imaging_label_pair_counts: Counter[str] = Counter()
    ages: list[int] = []
    document_lengths: list[int] = []
    messy_document_values: list[int] = []
    lab_numeric_values: dict[str, list[float]] = {}
    vital_numeric_values: dict[str, list[float]] = {}
    time_series_point_counts: list[int] = []
    time_series_durations: list[float] = []
    approved_values: list[bool] = []

    for record in records:
        ages.append(record.patient.age)
        sex_counts[record.patient.sex or "unknown"] += 1
        for modality in record.modalities:
            modality_counts[modality.value] += 1
        for document in record.documents:
            note_type_counts[document.note_type] += 1
            document_author_role_counts[document.author_role or "unknown"] += 1
            document_lengths.append(len(document.clean_text))
            messy_document_values.append(1 if document.messy_text else 0)
        for lab in record.labs:
            lab_name_counts[lab.name] += 1
            if lab.flag:
                lab_flag_counts[lab.flag] += 1
            if isinstance(lab.value, (int, float)):
                lab_numeric_values.setdefault(_metric_key(lab.name), []).append(
                    float(lab.value)
                )
        for vital in record.vitals:
            vital_name_counts[vital.name] += 1
            vital_numeric_values.setdefault(_metric_key(vital.name), []).append(
                float(vital.value)
            )
        for medication in record.medication_history:
            medication_name_counts[medication.name] += 1
            if medication.route:
                medication_route_counts[medication.route] += 1
            medication_status_counts[medication.status or "unknown"] += 1
        for channel in record.time_series:
            time_series_channel_counts[channel.name] += 1
            time_series_point_counts.append(len(channel.points))
            duration = _channel_duration_hours(channel)
            if duration is not None:
                time_series_durations.append(duration)
        for asset in record.imaging:
            imaging_modality_counts[asset.modality] += 1
            imaging_body_region_counts[asset.body_region] += 1
            asset_labels = sorted(
                {
                    _imaging_label_key(label.display, label.code)
                    for label in asset.labels
                    if label.display or label.code
                }
            )
            for label in asset_labels:
                imaging_label_counts[label] += 1
            for index, left in enumerate(asset_labels):
                for right in asset_labels[index + 1 :]:
                    imaging_label_pair_counts[f"{left}|{right}"] += 1
        if record.validation is not None:
            approved_values.append(record.validation.approved)

    return CohortProfile(
        dataset_id=records[0].dataset_id,
        record_count=len(records),
        modality_counts=dict(sorted(modality_counts.items())),
        mean_age=_mean(ages),
        sex_counts=dict(sorted(sex_counts.items())),
        mean_document_chars=_mean(document_lengths),
        note_type_counts=dict(sorted(note_type_counts.items())),
        document_author_role_counts=dict(sorted(document_author_role_counts.items())),
        messy_document_rate=_mean(messy_document_values),
        lab_name_counts=dict(sorted(lab_name_counts.items())),
        lab_flag_counts=dict(sorted(lab_flag_counts.items())),
        lab_numeric_summaries=_numeric_summaries(lab_numeric_values),
        vital_name_counts=dict(sorted(vital_name_counts.items())),
        vital_numeric_summaries=_numeric_summaries(vital_numeric_values),
        medication_name_counts=dict(sorted(medication_name_counts.items())),
        medication_route_counts=dict(sorted(medication_route_counts.items())),
        medication_status_counts=dict(sorted(medication_status_counts.items())),
        time_series_channel_counts=dict(sorted(time_series_channel_counts.items())),
        mean_time_series_points=_mean(time_series_point_counts),
        mean_time_series_duration_hours=_mean_float(time_series_durations),
        imaging_modality_counts=dict(sorted(imaging_modality_counts.items())),
        imaging_body_region_counts=dict(sorted(imaging_body_region_counts.items())),
        imaging_label_counts=dict(sorted(imaging_label_counts.items())),
        imaging_label_pair_counts=dict(sorted(imaging_label_pair_counts.items())),
        approved_rate=_mean([int(value) for value in approved_values])
        if approved_values
        else None,
    )


def _ratio_metric(name: str, generated: int, reference: int) -> BenchmarkMetric:
    if generated == reference:
        score = 1.0
    else:
        score = min(generated, reference) / max(generated, reference)
    return BenchmarkMetric(
        name=name,
        score=round(score, 4),
        generated_value=generated,
        reference_value=reference,
    )


def _closeness_metric(
    name: str,
    generated: float | None,
    reference: float | None,
    *,
    tolerance: float,
) -> BenchmarkMetric:
    if generated is None and reference is None:
        score = 1.0
    elif generated is None or reference is None:
        score = 0.0
    else:
        score = max(0.0, 1.0 - abs(generated - reference) / tolerance)
    return BenchmarkMetric(
        name=name,
        score=round(score, 4),
        generated_value=_rounded(generated),
        reference_value=_rounded(reference),
        details={"tolerance": tolerance},
    )


def _jaccard_metric(name: str, generated: set[str], reference: set[str]) -> BenchmarkMetric:
    if not generated and not reference:
        score = 1.0
    elif not generated or not reference:
        score = 0.0
    else:
        score = len(generated & reference) / len(generated | reference)
    return BenchmarkMetric(
        name=name,
        score=round(score, 4),
        generated_value=len(generated),
        reference_value=len(reference),
        details={
            "generated_only": sorted(generated - reference),
            "reference_only": sorted(reference - generated),
            "overlap": sorted(generated & reference),
        },
    )


def _distribution_metric(
    name: str,
    generated_counts: dict[str, int],
    reference_counts: dict[str, int],
) -> BenchmarkMetric:
    keys = set(generated_counts) | set(reference_counts)
    if not keys:
        score = 1.0
    else:
        generated_total = sum(generated_counts.values())
        reference_total = sum(reference_counts.values())
        distance = 0.0
        for key in keys:
            generated_share = (
                generated_counts.get(key, 0) / generated_total
                if generated_total > 0
                else 0.0
            )
            reference_share = (
                reference_counts.get(key, 0) / reference_total
                if reference_total > 0
                else 0.0
            )
            distance += abs(generated_share - reference_share)
        score = max(0.0, 1.0 - distance / 2)
    return BenchmarkMetric(
        name=name,
        score=round(score, 4),
        generated_value=sum(generated_counts.values()),
        reference_value=sum(reference_counts.values()),
        details={
            "generated_counts": generated_counts,
            "reference_counts": reference_counts,
        },
    )


def _numeric_summary_metrics(
    *,
    prefix: str,
    generated_summaries: dict[str, dict[str, float | int]],
    reference_summaries: dict[str, dict[str, float | int]],
    tolerance: float,
) -> list[BenchmarkMetric]:
    metrics: list[BenchmarkMetric] = []
    for name in sorted(set(generated_summaries) | set(reference_summaries)):
        generated = generated_summaries.get(name, {}).get("mean")
        reference = reference_summaries.get(name, {}).get("mean")
        metrics.append(
            _closeness_metric(
                f"{prefix}:{name}",
                float(generated) if generated is not None else None,
                float(reference) if reference is not None else None,
                tolerance=tolerance,
            )
        )
    return metrics


def _warnings(
    generated_profile: CohortProfile,
    reference_profile: CohortProfile,
    metrics: list[BenchmarkMetric],
) -> list[str]:
    warnings: list[str] = []
    if generated_profile.record_count < 10 or reference_profile.record_count < 10:
        warnings.append("Small cohort sizes make distribution metrics unstable.")
    for metric in metrics:
        if metric.score < 0.5:
            warnings.append(f"Low benchmark metric: {metric.name}={metric.score:.2f}.")
    return warnings


def _mean(values: list[int]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _mean_float(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


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


def _rounded(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 4)


def _channel_duration_hours(channel) -> float | None:
    if len(channel.points) < 2:
        return None
    timestamps = [_parse_datetime(point.timestamp) for point in channel.points]
    valid_timestamps = [timestamp for timestamp in timestamps if timestamp is not None]
    if len(valid_timestamps) < 2:
        return None
    return (max(valid_timestamps) - min(valid_timestamps)).total_seconds() / 3600


def _parse_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _imaging_label_key(display: str, code: str) -> str:
    value = display or code.replace("_", " ")
    return " ".join(value.lower().replace("_", " ").split())


def _metric_key(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").split())
