from __future__ import annotations

from collections import Counter
from statistics import mean

from casecrawler.models.dataset import DatasetManifest
from casecrawler.models.synthetic import SyntheticRecord


def build_dataset_card(
    manifest: DatasetManifest,
    records: list[SyntheticRecord],
) -> str:
    validation_scores = _validation_score_summary(records)
    modality_counts = Counter(
        modality.value for record in records for modality in record.modalities
    )
    provenance_counts = Counter(record.provenance.generator for record in records)
    review_counts = Counter(
        record.metadata.get("human_review", {}).get("status", "unreviewed")
        for record in records
    )
    return "\n".join(
        [
            f"# Dataset Card: {manifest.name}",
            "",
            "## Summary",
            "",
            f"- Dataset ID: `{manifest.dataset_id}`",
            f"- Topic: {manifest.topic}",
            f"- Generated records: {manifest.generated_count}",
            f"- Approved records: {manifest.approved_count}",
            f"- Created at: {manifest.created_at}",
            "- Synthetic data: yes",
            "",
            "## Modalities",
            "",
            *_counter_lines(modality_counts),
            "",
            "## Validation",
            "",
            f"- Approved fraction: {_fraction(manifest.approved_count, manifest.generated_count)}",
            *_score_lines(validation_scores),
            "",
            "## Human Review",
            "",
            *_counter_lines(review_counts),
            "",
            "## Provenance",
            "",
            *_counter_lines(provenance_counts),
            "",
            "## Intended Use",
            "",
            (
                "This dataset is intended for synthetic healthcare AI training, "
                "fine-tuning experiments, evaluation harnesses, pipeline testing, "
                "and data-format integration work."
            ),
            "",
            "## Limitations",
            "",
            (
                "Records are synthetic and must not be treated as real patient data. "
                "Clinical realism depends on configured generators, source grounding, "
                "validation thresholds, and human review. Downstream model training "
                "should benchmark against external reference datasets before release."
            ),
            "",
            "## Export Formats",
            "",
            *_list_lines([export_format.value for export_format in manifest.export_formats]),
            "",
        ]
    )


def build_model_card(
    manifest: DatasetManifest,
    records: list[SyntheticRecord],
) -> str:
    generators = Counter(record.provenance.generator for record in records)
    models = Counter(
        record.provenance.model or "unspecified" for record in records
    )
    backends = Counter(
        asset.generation_backend
        for record in records
        for asset in record.imaging
    )
    return "\n".join(
        [
            f"# Model Card: {manifest.name} synthetic generation pipeline",
            "",
            "## Overview",
            "",
            (
                "This card documents the synthetic generation pipeline used to "
                f"create dataset `{manifest.dataset_id}`. It describes generator "
                "components rather than a trained downstream model."
            ),
            "",
            "## Generator Components",
            "",
            *_counter_lines(generators),
            "",
            "## Model Backends",
            "",
            *_counter_lines(models),
            "",
            "## Imaging Backends",
            "",
            *_counter_lines(backends or Counter({"none": 1})),
            "",
            "## Validation Gates",
            "",
            "- Schema validation through Pydantic models",
            "- Deterministic clinical rules for labs, vitals, temporal order, and text contradictions",
            "- PHI-like privacy scanning",
            "- Optional image-text alignment validation",
            "- Optional human review decisions before export",
            "",
            "## Responsible Use",
            "",
            (
                "Generated records are synthetic training artifacts. They should be "
                "versioned with this card, benchmark reports, export manifests, and "
                "the configuration used to generate them."
            ),
            "",
        ]
    )


def _validation_score_summary(records: list[SyntheticRecord]) -> dict[str, float]:
    scored = [record.validation for record in records if record.validation is not None]
    if not scored:
        return {}
    fields = [
        "schema_score",
        "clinical_consistency_score",
        "privacy_score",
        "utility_score",
    ]
    return {
        field: mean(getattr(report, field) for report in scored)
        for field in fields
    }


def _score_lines(scores: dict[str, float]) -> list[str]:
    if not scores:
        return ["- Validation scores: not available"]
    return [f"- Mean {name.replace('_', ' ')}: {score:.3f}" for name, score in scores.items()]


def _counter_lines(counter: Counter[str]) -> list[str]:
    if not counter:
        return ["- None"]
    return [f"- {name}: {count}" for name, count in sorted(counter.items())]


def _list_lines(values: list[str]) -> list[str]:
    return [f"- {value}" for value in values] or ["- None"]


def _fraction(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.000"
    return f"{numerator / denominator:.3f}"
