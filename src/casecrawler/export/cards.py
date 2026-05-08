from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean

from casecrawler.imaging.file_metadata import raster_dimensions
from casecrawler.models.dataset import DatasetManifest
from casecrawler.models.evaluation import DatasetQualityReport
from casecrawler.models.synthetic import SyntheticRecord
from casecrawler.validation.quality import build_dataset_quality_report


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
    clinical_unit_counts = _clinical_unit_counts(records)
    medication_regimen_counts = _medication_regimen_counts(records)
    diagnosis_code_system_counts = _diagnosis_code_system_counts(records)
    diagnosis_code_counts = _diagnosis_code_counts(records)
    phi_entity_counts = _phi_entity_counts(records)
    quality_report = _quality_report_for_card(manifest, records)
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
            *_multimodal_release_lines(quality_report),
            *_benchmark_plan_lines(manifest),
            *_task_export_reference_lines(manifest),
            "",
            "## Extracted Fact Targets",
            "",
            *_counter_lines(extracted_fact_counts or Counter({"none": 1})),
            "",
            "## Procedures",
            "",
            *_counter_lines(procedure_counts or Counter({"none": 1})),
            "",
            "## Clinical Units",
            "",
            *_counter_lines(clinical_unit_counts or Counter({"none": 1})),
            "",
            "## Medication Regimens",
            "",
            *_counter_lines(medication_regimen_counts or Counter({"none": 1})),
            "",
            "## Diagnosis Coding Signals",
            "",
            "### Code Systems",
            "",
            *_counter_lines(diagnosis_code_system_counts or Counter({"none": 1})),
            "",
            "### Diagnosis Codes",
            "",
            *_counter_lines(diagnosis_code_counts or Counter({"none": 1})),
            "",
            "## PHI Annotation Signals",
            "",
            *_counter_lines(phi_entity_counts or Counter({"none": 1})),
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
    time_series_model_policies = _time_series_model_policy_counts(records)
    time_series_units = Counter(
        channel.unit for record in records for channel in record.time_series
    )
    clinical_text_model_policies = _clinical_text_model_policy_counts(records)
    imaging_dimension_summary = _imaging_dimension_summary(records)
    procedure_counts = _procedure_counts(records)
    generation_overrides = _generation_override_counts(records)
    imaging_model_policies = _imaging_model_policy_counts(records)
    image_validator_policies = _image_validator_policy_counts(records)
    diagnosis_code_system_counts = _diagnosis_code_system_counts(records)
    diagnosis_code_counts = _diagnosis_code_counts(records)
    phi_entity_counts = _phi_entity_counts(records)
    quality_report = _quality_report_for_card(manifest, records)
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
            "## Clinical Text Model Policies",
            "",
            *_counter_lines(clinical_text_model_policies or Counter({"none": 1})),
            "",
            "## Imaging Backends",
            "",
            *_counter_lines(backends or Counter({"none": 1})),
            "",
            "## Imaging Model Policies",
            "",
            *_counter_lines(imaging_model_policies or Counter({"none": 1})),
            "",
            "## Image Validator Policies",
            "",
            *_counter_lines(image_validator_policies or Counter({"none": 1})),
            "",
            "## Time-Series Backends",
            "",
            *_counter_lines(time_series_backends or Counter({"none": 1})),
            "",
            "## Time-Series Model Policies",
            "",
            *_counter_lines(time_series_model_policies or Counter({"none": 1})),
            "",
            "## Time-Series Units",
            "",
            *_counter_lines(time_series_units or Counter({"none": 1})),
            "",
            "## Imaging Dimensions",
            "",
            *_imaging_dimension_lines(imaging_dimension_summary),
            "",
            "## Procedure Coverage",
            "",
            *_counter_lines(procedure_counts or Counter({"none": 1})),
            "",
            "## Diagnosis Coding Signals",
            "",
            "### Code Systems",
            "",
            *_counter_lines(diagnosis_code_system_counts or Counter({"none": 1})),
            "",
            "### Diagnosis Codes",
            "",
            *_counter_lines(diagnosis_code_counts or Counter({"none": 1})),
            "",
            "## PHI Annotation Signals",
            "",
            *_counter_lines(phi_entity_counts or Counter({"none": 1})),
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
            "- Multimodal release readiness across clinical text, structured EHR, labs, vitals, medications, time series, imaging, model policy, and benchmark references",
            "",
            "## Multimodal Release Readiness",
            "",
            *_multimodal_release_status_lines(quality_report),
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
        "modality_alignment_score",
    ]
    return {
        field: mean(getattr(report, field) for report in scored)
        for field in fields
        if all(getattr(report, field) is not None for report in scored)
    }


def _score_lines(scores: dict[str, float]) -> list[str]:
    if not scores:
        return ["- Validation scores: not available"]
    return [f"- Mean {name.replace('_', ' ')}: {score:.3f}" for name, score in scores.items()]


def _quality_report_for_card(
    manifest: DatasetManifest,
    records: list[SyntheticRecord],
) -> DatasetQualityReport:
    return build_dataset_quality_report(
        manifest.dataset_id,
        records,
        benchmark_plan=_benchmark_plan_for_quality(manifest),
    )


def _benchmark_plan_for_quality(manifest: DatasetManifest) -> dict | None:
    reference_keys = manifest.metadata.get("recommended_reference_keys", [])
    if not isinstance(reference_keys, list) or not reference_keys:
        return None
    resolved_reference_dataset_id = _resolved_benchmark_reference_dataset_id(manifest)
    missing_reference_keys = []
    if resolved_reference_dataset_id is None:
        missing_reference_keys = [str(key) for key in reference_keys if str(key).strip()]
    return {
        "recommended_reference_keys": reference_keys,
        "ready": resolved_reference_dataset_id is not None,
        "resolved_reference_dataset_id": resolved_reference_dataset_id,
        "missing_reference_keys": missing_reference_keys,
        "thresholds": manifest.metadata.get("benchmark_thresholds", {}),
        "task_export_reference_readiness": {},
    }


def _resolved_benchmark_reference_dataset_id(manifest: DatasetManifest) -> str | None:
    direct = manifest.metadata.get("resolved_reference_dataset_id")
    if isinstance(direct, str) and direct.strip():
        return direct
    exports = manifest.metadata.get("latest_exports", [])
    if not isinstance(exports, list):
        return None
    for export in exports:
        if not isinstance(export, dict):
            continue
        metadata = export.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        if metadata.get("benchmark_passed") is not True:
            continue
        reference_id = metadata.get("benchmark_reference_dataset_id")
        if isinstance(reference_id, str) and reference_id.strip():
            return reference_id
    return None


def _multimodal_release_lines(report: DatasetQualityReport) -> list[str]:
    return [
        "",
        "## Multimodal Release Readiness",
        "",
        *_multimodal_release_status_lines(report),
        "",
        "### Core Artifact Coverage",
        "",
        *_coverage_lines(report.core_artifact_coverage),
    ]


def _multimodal_release_status_lines(report: DatasetQualityReport) -> list[str]:
    missing = (
        ", ".join(report.multimodal_release_missing)
        if report.multimodal_release_missing
        else "none"
    )
    return [
        f"- Ready: {report.multimodal_release_ready}",
        f"- Missing: {missing}",
    ]


def _coverage_lines(coverage: dict[str, bool]) -> list[str]:
    if not coverage:
        return ["- none: False"]
    return [f"- {key}: {value}" for key, value in sorted(coverage.items())]


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


def _image_validator_policy_counts(records: list[SyntheticRecord]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        policy = record.metadata.get("image_validator_policy", {})
        if not isinstance(policy, dict):
            continue
        profile = policy.get("profile") or "unspecified"
        backend = policy.get("backend") or "unspecified"
        use_policy = policy.get("use_policy") or "review_license_before_use"
        license_name = policy.get("license") or "unspecified"
        gated = policy.get("gated")
        counter[
            f"profile={profile} backend={backend} license={license_name} "
            f"gated={gated} use_policy={use_policy}"
        ] += 1
    return counter


def _time_series_model_policy_counts(records: list[SyntheticRecord]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        policy = record.metadata.get("time_series_model_policy", {})
        if not isinstance(policy, dict):
            continue
        profile = policy.get("profile") or "unspecified"
        use_policy = policy.get("use_policy") or "review_license_before_use"
        license_name = policy.get("license") or "unspecified"
        gated = policy.get("gated")
        counter[
            f"profile={profile} license={license_name} "
            f"gated={gated} use_policy={use_policy}"
        ] += 1
    return counter


def _clinical_text_model_policy_counts(records: list[SyntheticRecord]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        policy = record.metadata.get("clinical_text_model_policy", {})
        if not isinstance(policy, dict):
            continue
        backend = policy.get("backend") or "unspecified"
        provider = policy.get("provider") or "unspecified"
        model_id = policy.get("model_id") or "unspecified"
        use_policy = policy.get("use_policy") or "review_outputs_before_release"
        gated = policy.get("gated")
        counter[
            f"backend={backend} provider={provider} model_id={model_id} "
            f"gated={gated} use_policy={use_policy}"
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


def _clinical_unit_counts(records: list[SyntheticRecord]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        for lab in record.labs:
            counter[f"lab:{lab.unit}"] += 1
        for vital in record.vitals:
            counter[f"vital:{vital.unit}"] += 1
    return counter


def _medication_regimen_counts(records: list[SyntheticRecord]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        for medication in record.medication_history:
            if medication.dose:
                counter[f"dose={medication.dose}"] += 1
            if medication.frequency:
                counter[f"frequency={medication.frequency}"] += 1
            if medication.route:
                counter[f"route={medication.route}"] += 1
            counter[f"status={medication.status or 'unknown'}"] += 1
    return counter


def _diagnosis_code_system_counts(records: list[SyntheticRecord]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        for encounter in record.encounters:
            for diagnosis in encounter.diagnoses:
                if diagnosis.system:
                    counter[diagnosis.system] += 1
    return counter


def _diagnosis_code_counts(records: list[SyntheticRecord]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        for encounter in record.encounters:
            for diagnosis in encounter.diagnoses:
                if diagnosis.code:
                    counter[_diagnosis_code_key(diagnosis)] += 1
    return counter


def _imaging_dimension_summary(records: list[SyntheticRecord]) -> dict[str, float]:
    widths: list[int] = []
    heights: list[int] = []
    for record in records:
        for asset in record.imaging:
            if not asset.file_path or not Path(asset.file_path).is_file():
                continue
            width, height = raster_dimensions(asset.file_path)
            if width is not None and height is not None:
                widths.append(width)
                heights.append(height)
    summary: dict[str, float] = {}
    if widths:
        summary["mean_width"] = round(mean(widths), 1)
    if heights:
        summary["mean_height"] = round(mean(heights), 1)
    return summary


def _imaging_dimension_lines(summary: dict[str, float]) -> list[str]:
    if not summary:
        return ["- No readable image dimensions"]
    return [
        f"- Mean width: {summary['mean_width']:.1f} px",
        f"- Mean height: {summary['mean_height']:.1f} px",
    ]


def _phi_entity_counts(records: list[SyntheticRecord]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        for document in record.documents:
            _count_phi_entities(document.extracted_facts, counter)
    return counter


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


def _task_export_reference_lines(manifest: DatasetManifest) -> list[str]:
    references = manifest.metadata.get("task_export_reference_keys", {})
    if not isinstance(references, dict) or not references:
        return []
    lines = ["", "## Task-Specific Export References", ""]
    for export_format, reference_keys in sorted(references.items()):
        if not isinstance(reference_keys, list):
            continue
        rendered_keys = [
            str(reference_key)
            for reference_key in reference_keys
            if str(reference_key).strip()
        ]
        if rendered_keys:
            lines.append(f"- {export_format}: {', '.join(rendered_keys)}")
    return lines if len(lines) > 3 else []


def _fraction(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.000"
    return f"{numerator / denominator:.3f}"
