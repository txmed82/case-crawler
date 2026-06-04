from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from casecrawler.models.blueprint import ReleaseReadinessTier
from casecrawler.storage.dataset_store import DatasetStore


def build_blueprint_release_summary(
    store: DatasetStore,
    dataset_id: str,
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    blueprints = store.list_blueprints(dataset_id=dataset_id, limit=100_000)
    records = store.list_records(dataset_id=dataset_id, limit=None)
    judge_reports = store.list_judge_reports(dataset_id=dataset_id, limit=100_000)
    attempts = store.list_generation_attempts(dataset_id=dataset_id, limit=100_000)

    validation_reports = {
        blueprint.blueprint_id: store.get_blueprint_validation_report(
            blueprint.blueprint_id
        )
        for blueprint in blueprints
    }
    materialized_blueprint_ids = sorted(
        {
            str(record.metadata["blueprint_id"])
            for record in records
            if record.metadata.get("blueprint_id")
        }
    )
    ready_blueprint_ids = sorted(
        blueprint_id
        for blueprint_id, report in validation_reports.items()
        if report is not None and report.research_release_ready
    )
    missing_materialized = sorted(
        set(ready_blueprint_ids) - set(materialized_blueprint_ids)
    )

    return {
        "dataset_id": dataset_id,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "blueprint_count": len(blueprints),
        "validation_report_count": sum(
            1 for report in validation_reports.values() if report is not None
        ),
        "research_release_ready_count": len(ready_blueprint_ids),
        "materialized_record_count": sum(
            1 for record in records if record.metadata.get("blueprint_id")
        ),
        "materialized_blueprint_ids": materialized_blueprint_ids,
        "missing_materialized_blueprint_ids": missing_materialized,
        "judge_report_count": len(judge_reports),
        "passing_judge_report_count": sum(1 for report in judge_reports if report.passed),
        "attempt_counts": dict(Counter(attempt.status.value for attempt in attempts)),
        "role_attempt_counts": dict(Counter(attempt.role.value for attempt in attempts)),
        "tier_counts": _tier_counts(validation_reports),
        "organ_system_counts": dict(
            Counter(blueprint.organ_system for blueprint in blueprints)
        ),
        "setting_counts": dict(Counter(blueprint.setting for blueprint in blueprints)),
        "non_ready_blueprints": _non_ready_blueprints(validation_reports),
    }


def write_blueprint_release_summary(
    store: DatasetStore,
    dataset_id: str,
    output: str | Path,
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    summary = build_blueprint_release_summary(
        store,
        dataset_id,
        generated_at=generated_at,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _tier_counts(validation_reports: dict) -> dict[str, int]:
    counts = Counter()
    for report in validation_reports.values():
        if report is None:
            counts["missing"] += 1
        else:
            counts[report.tier.value] += 1
    for tier in ReleaseReadinessTier:
        counts.setdefault(tier.value, 0)
    return dict(counts)


def _non_ready_blueprints(validation_reports: dict) -> list[dict[str, Any]]:
    non_ready = []
    for blueprint_id, report in sorted(validation_reports.items()):
        if report is not None and report.research_release_ready:
            continue
        if report is None:
            non_ready.append(
                {
                    "blueprint_id": blueprint_id,
                    "tier": "missing",
                    "issues": [
                        {
                            "field": "blueprint_validation_report",
                            "message": "No validation report is persisted.",
                        }
                    ],
                }
            )
            continue
        non_ready.append(
            {
                "blueprint_id": blueprint_id,
                "tier": report.tier.value,
                "issues": report.issues,
                "schema_valid": report.schema_valid,
                "clinically_plausible": report.clinically_plausible,
                "grounded": report.grounded,
                "judge_validated": report.judge_validated,
            }
        )
    return non_ready
