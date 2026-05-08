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
    generation_overrides = _generation_override_counts(records)
    extracted_fact_counts = _extracted_fact_counts(records)
    procedure_counts = _procedure_counts(records)
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
            *_benchmark_plan_lines(manifest),
            "",
            "## Extracted Fact Targets",
            "",
            *_counter_lines(extracted_fact_counts or Counter({"none": 1})),
            "",
            "## Procedures",
            "",
            *_counter_lines(procedure_counts or Counter({"none": 1})),
            "",
            "## Human Review",
            "",
            *_counter_lines(review_counts),
            "",
            "## Provenance",
            "",
            *_counter_lines(provenance_counts),
            "",
            "## Generation Overrides",
            "",
            *_counter_lines(generation_overrides or Counter({"none": 1})),
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
            "## Export Audit Trail",
            "",
            *_export_manifest_lines(manifest),
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
    time_series_backends = Counter(
        channel.generation_backend
        for record in records
        for channel in record.time_series
    )
    procedure_counts = _procedure_counts(records)
    generation_overrides = _generation_override_counts(records)
    imaging_model_policies = _imaging_model_policy_counts(records)
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
            "## Imaging Model Policies",
            "",
            *_counter_lines(imaging_model_policies or Counter({"none": 1})),
            "",
            "## Time-Series Backends",
            "",
            *_counter_lines(time_series_backends or Counter({"none": 1})),
            "",
            "## Procedure Coverage",
            "",
            *_counter_lines(procedure_counts or Counter({"none": 1})),
            "",
            "## Request-Scoped Overrides",
            "",
            *_counter_lines(generation_overrides or Counter({"none": 1})),
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


def _generation_override_counts(records: list[SyntheticRecord]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        overrides = record.metadata.get("generation_overrides", {})
        if not isinstance(overrides, dict):
            continue
        for key, value in overrides.items():
            if isinstance(value, list):
                rendered = " ".join(str(item) for item in value)
            else:
                rendered = str(value)
            counter[f"{key}={rendered}"] += 1
    return counter


def _imaging_model_policy_counts(records: list[SyntheticRecord]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        policy = record.metadata.get("imaging_model_policy", {})
        if not isinstance(policy, dict):
            continue
        profile = policy.get("profile") or "unspecified"
        use_policy = policy.get("use_policy") or "review_license_before_use"
        license_name = policy.get("license") or "unspecified"
        gated = policy.get("gated")
        counter[
            f"profile={profile} license={license_name} gated={gated} use_policy={use_policy}"
        ] += 1
    return counter


def _extracted_fact_counts(records: list[SyntheticRecord]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        for document in record.documents:
            for key, value in document.extracted_facts.items():
                if _has_fact_value(value):
                    counter[key] += 1
    return counter


def _procedure_counts(records: list[SyntheticRecord]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        for encounter in record.encounters:
            for procedure in encounter.procedures:
                counter[procedure.display] += 1
    return counter


def _has_fact_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list | tuple | set | dict):
        return bool(value)
    return True


def _list_lines(values: list[str]) -> list[str]:
    return [f"- {value}" for value in values] or ["- None"]


def _export_manifest_lines(manifest: DatasetManifest) -> list[str]:
    exports = manifest.metadata.get("latest_exports", [])
    if not isinstance(exports, list) or not exports:
        return ["- No exports recorded"]
    lines = []
    for export in exports:
        if not isinstance(export, dict):
            continue
        metadata = export.get("metadata", {})
        gate = ""
        if isinstance(metadata, dict) and "benchmark_passed" in metadata:
            gate = (
                f", benchmark_passed={metadata.get('benchmark_passed')}, "
                f"reference={metadata.get('benchmark_reference_dataset_id')}"
            )
        lines.append(
            "- "
            f"{export.get('export_format')} to {export.get('file_path')} "
            f"records={export.get('record_count')}{gate}"
        )
    return lines or ["- No exports recorded"]


def _benchmark_plan_lines(manifest: DatasetManifest) -> list[str]:
    references = manifest.metadata.get("recommended_reference_keys", [])
    thresholds = manifest.metadata.get("benchmark_thresholds", {})
    if not isinstance(references, list) or not references:
        return []
    lines = ["", "## Recommended Benchmark Plan", ""]
    recipe = manifest.metadata.get("primary_recipe")
    if isinstance(recipe, str):
        lines.append(f"- Recipe: {recipe}")
    lines.append(f"- Reference datasets: {', '.join(str(item) for item in references)}")
    if isinstance(thresholds, dict):
        min_overall = thresholds.get("min_overall_score")
        min_metric = thresholds.get("min_metric_score")
        if min_overall is not None and min_metric is not None:
            lines.append(
                "- Thresholds: "
                f"overall >= {min_overall}, metric >= {min_metric}"
            )
    return lines


def _fraction(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.000"
    return f"{numerator / denominator:.3f}"
