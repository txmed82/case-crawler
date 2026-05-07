from __future__ import annotations

from collections import Counter

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
    modality_counts: Counter[str] = Counter()
    sex_counts: Counter[str] = Counter()
    note_type_counts: Counter[str] = Counter()
    lab_name_counts: Counter[str] = Counter()
    vital_name_counts: Counter[str] = Counter()
    medication_name_counts: Counter[str] = Counter()
    ages: list[int] = []
    document_lengths: list[int] = []
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
            generated_share = generated_counts.get(key, 0) / generated_total
            reference_share = reference_counts.get(key, 0) / reference_total
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


def _rounded(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 4)

