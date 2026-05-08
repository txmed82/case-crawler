from __future__ import annotations

from dataclasses import dataclass

from casecrawler.models.evaluation import BenchmarkReport
from casecrawler.storage.dataset_store import DatasetStore
from casecrawler.validation.benchmark import DatasetBenchmark


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
    task_reference_keys = _metadata_string_list_map(
        manifest.metadata.get("task_export_reference_keys")
    )
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
    task_reference_readiness = {
        export_format: _task_reference_readiness(
            store,
            dataset_id=dataset_id,
            reference_keys=export_reference_keys,
        )
        for export_format, export_reference_keys in task_reference_keys.items()
    }
    return {
        "dataset_id": dataset_id,
        "primary_recipe": _metadata_string(manifest.metadata.get("primary_recipe")),
        "recommended_reference_keys": reference_keys,
        "task_export_reference_readiness": task_reference_readiness,
        "resolved_reference_dataset_id": selected_reference_id,
        "resolved_reference_key": resolved_reference_key,
        "ready": selected_reference_id is not None,
        "missing_reference_keys": missing_reference_keys,
        "thresholds": {
            "min_overall_score": thresholds[0] if thresholds else 0.75,
            "min_metric_score": thresholds[1] if thresholds else 0.5,
        },
    }


def run_recommended_benchmark_suite(
    store: DatasetStore,
    dataset_id: str,
    *,
    min_overall_score: float | None = None,
    min_metric_score: float | None = None,
) -> dict:
    manifest = store.get_manifest(dataset_id)
    reference_keys = _metadata_string_list(manifest.metadata.get("recommended_reference_keys"))
    manifest_thresholds = _metadata_thresholds(manifest.metadata.get("benchmark_thresholds"))
    thresholds = (
        min_overall_score if min_overall_score is not None else None,
        min_metric_score if min_metric_score is not None else None,
    )
    thresholds = (
        thresholds
        if thresholds[0] is not None and thresholds[1] is not None
        else manifest_thresholds or (0.75, 0.5)
    )
    generated_records = list(store.iter_records(dataset_id=dataset_id))
    results = []
    seen_reference_ids: set[str] = set()
    for reference_key in reference_keys:
        for reference_manifest in store.list_manifests():
            if reference_manifest.dataset_id == dataset_id:
                continue
            if reference_manifest.dataset_id in seen_reference_ids:
                continue
            if reference_manifest.metadata.get("primary_reference_key") != reference_key:
                continue
            reference_records = list(store.iter_records(dataset_id=reference_manifest.dataset_id))
            report = DatasetBenchmark(
                min_overall_score=thresholds[0],
                min_metric_score=thresholds[1],
            ).compare(generated_records, reference_records)
            results.append(
                {
                    "reference_key": reference_key,
                    "reference_dataset_id": reference_manifest.dataset_id,
                    "passed": report.passed,
                    "overall_score": report.overall_score,
                    "failing_metrics": report.failing_metrics,
                    "report": report.model_dump(),
                }
            )
            seen_reference_ids.add(reference_manifest.dataset_id)
    if not results:
        raise LookupError(
            "No imported reference dataset matches this dataset's recommended benchmark keys."
        )
    task_reference_keys = _metadata_string_list_map(
        manifest.metadata.get("task_export_reference_keys")
    )
    task_export_results = _task_export_results(task_reference_keys, results)
    return {
        "dataset_id": dataset_id,
        "primary_recipe": _metadata_string(manifest.metadata.get("primary_recipe")),
        "recommended_reference_keys": reference_keys,
        "task_export_results": task_export_results,
        "reference_count": len(results),
        "passed": all(result["passed"] for result in results),
        "mean_overall_score": round(
            sum(float(result["overall_score"]) for result in results) / len(results),
            4,
        ),
        "thresholds": {
            "min_overall_score": thresholds[0],
            "min_metric_score": thresholds[1],
        },
        "results": results,
    }


def benchmark_suite_from_report(
    *,
    dataset_id: str,
    benchmark_report: BenchmarkReport,
    reference_dataset_id: str,
    reference_key: str | None = None,
    primary_recipe: str | None = None,
    recommended_reference_keys: list[str] | None = None,
    task_export_results: dict | None = None,
) -> dict:
    """Build a one-reference benchmark suite artifact from an explicit benchmark gate."""
    resolved_reference_key = reference_key or reference_dataset_id
    return {
        "dataset_id": dataset_id,
        "primary_recipe": primary_recipe,
        "recommended_reference_keys": recommended_reference_keys or [resolved_reference_key],
        "task_export_results": task_export_results or {},
        "reference_count": 1,
        "passed": benchmark_report.passed,
        "mean_overall_score": benchmark_report.overall_score,
        "thresholds": benchmark_report.thresholds,
        "results": [
            {
                "reference_key": resolved_reference_key,
                "reference_dataset_id": reference_dataset_id,
                "passed": benchmark_report.passed,
                "overall_score": benchmark_report.overall_score,
                "failing_metrics": benchmark_report.failing_metrics,
                "report": benchmark_report.model_dump(mode="json"),
            }
        ],
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


def _metadata_string_list_map(value: object) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    mapping: dict[str, list[str]] = {}
    for key, items in value.items():
        if not isinstance(key, str):
            continue
        cleaned_key = key.strip()
        if not cleaned_key:
            continue
        reference_keys = _metadata_string_list(items)
        if reference_keys:
            mapping[cleaned_key] = reference_keys
    return mapping


def _task_reference_readiness(
    store: DatasetStore,
    *,
    dataset_id: str,
    reference_keys: list[str],
) -> dict:
    available_reference_keys: list[str] = []
    resolved_reference_dataset_id = None
    resolved_reference_key = None
    for reference_key in reference_keys:
        reference_dataset_id = store.find_reference_dataset_id(
            [reference_key],
            exclude_dataset_id=dataset_id,
        )
        if not reference_dataset_id:
            continue
        available_reference_keys.append(reference_key)
        if resolved_reference_dataset_id is None:
            reference_manifest = store.get_manifest(reference_dataset_id)
            resolved_reference_dataset_id = reference_dataset_id
            resolved_reference_key = _metadata_string(
                reference_manifest.metadata.get("primary_reference_key")
            )
    missing_reference_keys = [
        reference_key
        for reference_key in reference_keys
        if reference_key not in available_reference_keys
    ]
    return {
        "recommended_reference_keys": reference_keys,
        "available_reference_keys": available_reference_keys,
        "missing_reference_keys": missing_reference_keys,
        "resolved_reference_dataset_id": resolved_reference_dataset_id,
        "resolved_reference_key": resolved_reference_key,
        "ready": not missing_reference_keys,
    }


def _task_export_results(
    task_reference_keys: dict[str, list[str]],
    results: list[dict],
) -> dict[str, dict]:
    grouped_results: dict[str, dict] = {}
    for export_format, reference_keys in task_reference_keys.items():
        profile_results = [
            result
            for result in results
            if result.get("reference_key") in reference_keys
        ]
        matched_reference_keys = {
            result.get("reference_key")
            for result in profile_results
            if isinstance(result.get("reference_key"), str)
        }
        missing_reference_keys = [
            reference_key
            for reference_key in reference_keys
            if reference_key not in matched_reference_keys
        ]
        grouped_results[export_format] = {
            "recommended_reference_keys": reference_keys,
            "reference_count": len(profile_results),
            "missing_reference_keys": missing_reference_keys,
            "passed": bool(profile_results)
            and not missing_reference_keys
            and all(result["passed"] for result in profile_results),
            "mean_overall_score": round(
                sum(float(result["overall_score"]) for result in profile_results)
                / len(profile_results),
                4,
            )
            if profile_results
            else None,
            "results": profile_results,
        }
    return grouped_results


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
