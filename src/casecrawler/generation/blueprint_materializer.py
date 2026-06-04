from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5

from casecrawler.models.blueprint import (
    BlueprintValidationReport,
    ClinicalBlueprint,
)
from casecrawler.models.synthetic import (
    ClinicalDocument,
    ClinicalOrder,
    Code,
    ComplexityProfile,
    Encounter,
    LabObservation,
    MedicationStatement,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
    VitalObservation,
)
from casecrawler.storage.dataset_store import DatasetStore


class BlueprintMaterializer:
    def __init__(self, *, created_at: str | None = None) -> None:
        self._created_at = created_at

    def materialize(
        self,
        blueprint: ClinicalBlueprint,
        *,
        validation_report: BlueprintValidationReport | None = None,
        store: DatasetStore | None = None,
        require_release_ready: bool = False,
    ) -> SyntheticRecord:
        if (
            require_release_ready
            and validation_report is not None
            and not validation_report.research_release_ready
        ):
            raise ValueError(
                "Blueprint must be research release ready before materialization."
            )
        if require_release_ready and validation_report is None:
            raise ValueError(
                "Blueprint must be research release ready before materialization."
            )

        created_at = self._created_at or datetime.now(timezone.utc).isoformat()
        record = SyntheticRecord(
            record_id=_record_id(blueprint),
            dataset_id=blueprint.dataset_id,
            topic=blueprint.organ_system,
            complexity=_complexity(blueprint),
            modalities=_modalities(blueprint),
            patient=_patient(blueprint),
            encounters=[_encounter(blueprint, created_at=created_at)],
            labs=_labs(blueprint, created_at=created_at),
            vitals=_vitals(blueprint, created_at=created_at),
            medication_history=_medications(blueprint),
            orders=_orders(blueprint, created_at=created_at),
            documents=[_summary_document(blueprint, created_at=created_at)],
            provenance=Provenance(
                generator="blueprint-materializer",
                source_refs=[
                    {
                        "blueprint_id": blueprint.blueprint_id,
                        "cohort_plan_id": blueprint.cohort_plan_id,
                        "evidence_citations": blueprint.evidence.citations,
                    }
                ],
                prompt_hash=_blueprint_hash(blueprint),
                created_at=created_at,
            ),
            metadata=_metadata(blueprint, validation_report),
        )
        if store is not None:
            store.save_record(record)
        return record


def _record_id(blueprint: ClinicalBlueprint) -> str:
    return f"rec-{uuid5(NAMESPACE_URL, f'blueprint:{blueprint.blueprint_id}')}"


def _blueprint_hash(blueprint: ClinicalBlueprint) -> str:
    payload = json.dumps(
        blueprint.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _complexity(blueprint: ClinicalBlueprint) -> ComplexityProfile:
    raw = str(
        blueprint.metadata.get("difficulty")
        or blueprint.metadata.get("complexity")
        or ""
    ).lower()
    for complexity in ComplexityProfile:
        if raw == complexity.value:
            return complexity
    if blueprint.uncertainty_points or len(blueprint.differential) > 1:
        return ComplexityProfile.COMPLEX
    return ComplexityProfile.MODERATE


def _modalities(blueprint: ClinicalBlueprint) -> list[Modality]:
    modalities = list(blueprint.required_modalities)
    if blueprint.expected_labs and Modality.LABS not in modalities:
        modalities.append(Modality.LABS)
    if blueprint.expected_vitals and Modality.VITALS not in modalities:
        modalities.append(Modality.VITALS)
    return modalities


def _patient(blueprint: ClinicalBlueprint) -> SyntheticPatient:
    demographics = dict(blueprint.patient)
    age = int(demographics.pop("age", 0))
    sex = str(demographics.pop("sex", "unknown"))
    social_history = demographics.pop("social_history", {})
    return SyntheticPatient(
        patient_id=f"pat-{uuid5(NAMESPACE_URL, f'patient:{blueprint.blueprint_id}')}",
        age=age,
        sex=sex,
        demographics=demographics,
        social_history=social_history if isinstance(social_history, dict) else {},
    )


def _encounter(blueprint: ClinicalBlueprint, *, created_at: str) -> Encounter:
    start = _first_timeline_time(blueprint) or created_at
    return Encounter(
        encounter_id=f"enc-{uuid5(NAMESPACE_URL, f'encounter:{blueprint.blueprint_id}')}",
        start=start,
        setting=blueprint.setting,
        reason=blueprint.chief_concern,
        diagnoses=[
            Code(system="blueprint", code=_code_value(item.name), display=item.name)
            for item in blueprint.diagnoses
        ],
    )


def _first_timeline_time(blueprint: ClinicalBlueprint) -> str | None:
    for event in blueprint.timeline:
        value = event.get("time") or event.get("timestamp") or event.get("date")
        if value:
            return str(value)
    return None


def _labs(
    blueprint: ClinicalBlueprint,
    *,
    created_at: str,
) -> list[LabObservation]:
    labs: list[LabObservation] = []
    for item in blueprint.expected_labs:
        labs.append(
            LabObservation(
                name=str(item.get("name", "Unspecified lab")),
                loinc=_optional_str(item.get("loinc")),
                value=item.get("value", "expected"),
                unit=str(item.get("unit", "")),
                reference_low=_optional_float(item.get("reference_low")),
                reference_high=_optional_float(item.get("reference_high")),
                flag=_optional_str(item.get("flag")),
                effective_time=str(item.get("effective_time") or created_at),
                specimen=_optional_str(item.get("specimen")),
            )
        )
    return labs


def _vitals(
    blueprint: ClinicalBlueprint,
    *,
    created_at: str,
) -> list[VitalObservation]:
    vitals: list[VitalObservation] = []
    for item in blueprint.expected_vitals:
        vitals.append(
            VitalObservation(
                name=str(item.get("name", "Unspecified vital")),
                value=_float_value(item.get("value")),
                unit=str(item.get("unit", "")),
                effective_time=str(item.get("effective_time") or created_at),
            )
        )
    return vitals


def _medications(blueprint: ClinicalBlueprint) -> list[MedicationStatement]:
    medications: list[MedicationStatement] = []
    for item in blueprint.medications:
        medications.append(
            MedicationStatement(
                name=str(item.get("name", "Unspecified medication")),
                rxnorm=_optional_str(item.get("rxnorm")),
                dose=_optional_str(item.get("dose")),
                route=_optional_str(item.get("route")),
                frequency=_optional_str(item.get("frequency")),
                status=str(item.get("status", "unknown")),
                start=_optional_str(item.get("start")),
                end=_optional_str(item.get("end")),
            )
        )
    return medications


def _orders(
    blueprint: ClinicalBlueprint,
    *,
    created_at: str,
) -> list[ClinicalOrder]:
    orders: list[ClinicalOrder] = []
    for index, item in enumerate(blueprint.orders):
        orders.append(
            ClinicalOrder(
                order_id=str(
                    item.get("order_id")
                    or uuid5(
                        NAMESPACE_URL,
                        f"order:{blueprint.blueprint_id}:{index}",
                    )
                ),
                order_type=str(item.get("order_type", "order")),
                display=str(item.get("display", "Unspecified order")),
                code=_optional_str(item.get("code")),
                system=_optional_str(item.get("system")),
                status=str(item.get("status", "active")),
                intent=str(item.get("intent", "order")),
                priority=_optional_str(item.get("priority")),
                ordered_at=str(item.get("ordered_at") or created_at),
            )
        )
    return orders


def _summary_document(
    blueprint: ClinicalBlueprint,
    *,
    created_at: str,
) -> ClinicalDocument:
    diagnoses = ", ".join(item.name for item in blueprint.diagnoses)
    differential = ", ".join(item.name for item in blueprint.differential) or "none"
    targets = "; ".join(blueprint.clinical_reasoning_targets) or "none"
    safety = "; ".join(blueprint.safety_constraints) or "none"
    uncertainty = "; ".join(blueprint.uncertainty_points) or "none"
    evidence = "; ".join(blueprint.evidence.supported_claims) or "none"
    clean_text = "\n".join(
        [
            f"Chief concern: {blueprint.chief_concern}",
            f"Primary diagnoses: {diagnoses}",
            f"Differential: {differential}",
            f"Clinical reasoning targets: {targets}",
            f"Safety constraints: {safety}",
            f"Uncertainty points: {uncertainty}",
            f"Supported evidence: {evidence}",
        ]
    )
    return ClinicalDocument(
        document_id=f"doc-{uuid5(NAMESPACE_URL, f'document:{blueprint.blueprint_id}')}",
        note_type="blueprint_summary",
        author_role="synthetic_blueprint_materializer",
        timestamp=created_at,
        clean_text=clean_text,
        extracted_facts={
            "blueprint_id": blueprint.blueprint_id,
            "intended_tasks": blueprint.intended_tasks,
            "archetype_name": blueprint.archetype_name,
        },
    )


def _metadata(
    blueprint: ClinicalBlueprint,
    validation_report: BlueprintValidationReport | None,
) -> dict:
    metadata = {
        "blueprint_id": blueprint.blueprint_id,
        "cohort_plan_id": blueprint.cohort_plan_id,
        "archetype_name": blueprint.archetype_name,
        "intended_tasks": blueprint.intended_tasks,
        "source_blueprint_metadata": blueprint.metadata,
    }
    if validation_report is not None:
        metadata.update(
            {
                "release_readiness_tier": validation_report.tier.value,
                "schema_valid": validation_report.schema_valid,
                "clinically_plausible": validation_report.clinically_plausible,
                "grounded": validation_report.grounded,
                "judge_validated": validation_report.judge_validated,
                "judge_report_ids": [
                    report.report_id for report in validation_report.judge_reports
                ],
            }
        )
    return metadata


def _code_value(value: str) -> str:
    return value.strip().lower().replace(" ", "-") or "unspecified"


def _optional_str(value) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return _float_value(value)


def _float_value(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
