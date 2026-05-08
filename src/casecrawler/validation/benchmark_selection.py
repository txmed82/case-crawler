from __future__ import annotations

from dataclasses import dataclass

from casecrawler.storage.dataset_store import DatasetStore


@dataclass(frozen=True)
class BenchmarkGateSelection:
    reference_dataset_id: str
    min_overall_score: float
    min_metric_score: float
    auto_selected: bool = False
    reference_key: str | None = None


def resolve_benchmark_gate(
    store: DatasetStore,
    *,
    dataset_id: str,
    reference_dataset_id: str | None = None,
    auto_benchmark: bool = False,
    min_overall_score: float = 0.75,
    min_metric_score: float = 0.5,
) -> BenchmarkGateSelection | None:
    if reference_dataset_id:
        reference_manifest = store.get_manifest(reference_dataset_id)
        return BenchmarkGateSelection(
            reference_dataset_id=reference_dataset_id,
            min_overall_score=min_overall_score,
            min_metric_score=min_metric_score,
            reference_key=_metadata_string(reference_manifest.metadata.get("primary_reference_key")),
        )
    if not auto_benchmark:
        return None

    manifest = store.get_manifest(dataset_id)
    reference_keys = _metadata_string_list(manifest.metadata.get("recommended_reference_keys"))
    selected_reference_id = store.find_reference_dataset_id(
        reference_keys,
        exclude_dataset_id=dataset_id,
    )
    if not selected_reference_id:
        raise LookupError(
            "No imported reference dataset matches this dataset's recommended benchmark keys."
        )
    reference_manifest = store.get_manifest(selected_reference_id)
    thresholds = _metadata_thresholds(manifest.metadata.get("benchmark_thresholds"))
    return BenchmarkGateSelection(
        reference_dataset_id=selected_reference_id,
        min_overall_score=thresholds[0] if thresholds else min_overall_score,
        min_metric_score=thresholds[1] if thresholds else min_metric_score,
        auto_selected=True,
        reference_key=_metadata_string(reference_manifest.metadata.get("primary_reference_key")),
    )


def build_benchmark_plan_summary(store: DatasetStore, dataset_id: str) -> dict:
    manifest = store.get_manifest(dataset_id)
    reference_keys = _metadata_string_list(manifest.metadata.get("recommended_reference_keys"))
    thresholds = _metadata_thresholds(manifest.metadata.get("benchmark_thresholds"))
    selected_reference_id = store.find_reference_dataset_id(
        reference_keys,
        exclude_dataset_id=dataset_id,
    )
    resolved_reference_key = None
    if selected_reference_id:
        reference_manifest = store.get_manifest(selected_reference_id)
        resolved_reference_key = _metadata_string(
            reference_manifest.metadata.get("primary_reference_key")
        )
    available_reference_keys = [
        reference_key
        for reference_key in reference_keys
        if store.find_reference_dataset_id([reference_key], exclude_dataset_id=dataset_id)
    ]
    missing_reference_keys = [
        reference_key
        for reference_key in reference_keys
        if reference_key not in available_reference_keys
    ]
    return {
        "dataset_id": dataset_id,
        "primary_recipe": _metadata_string(manifest.metadata.get("primary_recipe")),
        "recommended_reference_keys": reference_keys,
        "resolved_reference_dataset_id": selected_reference_id,
        "resolved_reference_key": resolved_reference_key,
        "ready": selected_reference_id is not None,
        "missing_reference_keys": missing_reference_keys,
        "thresholds": {
            "min_overall_score": thresholds[0] if thresholds else 0.75,
            "min_metric_score": thresholds[1] if thresholds else 0.5,
        },
    }


def _metadata_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _metadata_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _metadata_thresholds(value: object) -> tuple[float, float] | None:
    if not isinstance(value, dict):
        return None
    min_overall_score = value.get("min_overall_score")
    min_metric_score = value.get("min_metric_score")
    if not isinstance(min_overall_score, int | float):
        return None
    if not isinstance(min_metric_score, int | float):
        return None
    return (
        _clamp_score(float(min_overall_score)),
        _clamp_score(float(min_metric_score)),
    )


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))
