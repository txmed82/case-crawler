from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from casecrawler.models.blueprint import ClinicalBlueprint, CohortPlan


def export_blueprint_payload(
    blueprint: ClinicalBlueprint,
    *,
    plan: CohortPlan | None = None,
) -> dict[str, Any]:
    payload = {
        "artifact_type": "casecrawler_clinical_blueprint",
        "blueprint": blueprint.model_dump(),
    }
    if plan is not None:
        payload["cohort_plan"] = plan.model_dump()
    return payload


def export_blueprints_jsonl(
    blueprints: Iterable[ClinicalBlueprint],
    output: str | Path,
    *,
    plan_lookup=None,
) -> int:
    count = 0
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for blueprint in blueprints:
            plan = plan_lookup(blueprint.cohort_plan_id) if plan_lookup else None
            payload = export_blueprint_payload(blueprint, plan=plan)
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            count += 1
    return count
