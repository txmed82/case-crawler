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
                "lab_name_overlap",
                set(generated_profile.lab_name_counts),
                set(reference_profile.lab_name_counts),
            ),
            _jaccard_metric(
                "vital_name_overlap",
                set(generated_profile.vital_name_counts),
                set(reference_profile.vital_name_counts),
            ),
            _jaccard_metric(
                "medication_name_overlap",
                set(generated_profile.medication_name_counts),
                set(reference_profile.medication_name_counts),
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
    lab_name_counts: Counter[str] = Counter()
    vital_name_counts: Counter[str] = Counter()
    medication_name_counts: Counter[str] = Counter()
    time_series_channel_counts: Counter[str] = Counter()
    imaging_modality_counts: Counter[str] = Counter()
    imaging_body_region_counts: Counter[str] = Counter()
    ages: list[int] = []
    document_lengths: list[int] = []
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
            document_lengths.append(len(document.clean_text))
        for lab in record.labs:
            lab_name_counts[lab.name] += 1
        for vital in record.vitals:
            vital_name_counts[vital.name] += 1
        for medication in record.medication_history:
            medication_name_counts[medication.name] += 1
        for channel in record.time_series:
            time_series_channel_counts[channel.name] += 1
            time_series_point_counts.append(len(channel.points))
            duration = _channel_duration_hours(channel)
            if duration is not None:
                time_series_durations.append(duration)
        for asset in record.imaging:
            imaging_modality_counts[asset.modality] += 1
            imaging_body_region_counts[asset.body_region] += 1
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
        lab_name_counts=dict(sorted(lab_name_counts.items())),
        vital_name_counts=dict(sorted(vital_name_counts.items())),
        medication_name_counts=dict(sorted(medication_name_counts.items())),
        time_series_channel_counts=dict(sorted(time_series_channel_counts.items())),
        mean_time_series_points=_mean(time_series_point_counts),
        mean_time_series_duration_hours=_mean_float(time_series_durations),
        imaging_modality_counts=dict(sorted(imaging_modality_counts.items())),
        imaging_body_region_counts=dict(sorted(imaging_body_region_counts.items())),
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
