from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from casecrawler.models.synthetic import (
    ComplexityProfile,
    Encounter,
    LabObservation,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
)


class SyntheaAdapter:
    def import_fhir_bundle(self, path: str, dataset_id: str) -> SyntheticRecord:
        bundle = json.loads(Path(path).read_text())
        resources = [entry.get("resource", {}) for entry in bundle.get("entry", [])]
        patient_resource = _first_resource(resources, "Patient")
        encounter_resources = _resources(resources, "Encounter")
        observation_resources = _resources(resources, "Observation")

        patient_id = patient_resource.get("id", "synthea-patient")
        topic = _topic(encounter_resources) or "synthea import"
        created_at = _encounter_start(encounter_resources) or "2026-01-01T00:00:00"

        patient = SyntheticPatient(
            patient_id=patient_id,
            age=_age(patient_resource.get("birthDate"), created_at),
            sex=patient_resource.get("gender", "unknown"),
            demographics={"source": "synthea_fhir"},
        )
        encounters = [
            Encounter(
                encounter_id=resource.get("id", f"enc-{index}"),
                start=resource.get("period", {}).get("start", created_at),
                end=resource.get("period", {}).get("end"),
                setting="synthea",
                reason=_reason(resource) or topic,
            )
            for index, resource in enumerate(encounter_resources)
        ]
        labs = [_observation_to_lab(resource, created_at) for resource in observation_resources]

        return SyntheticRecord(
            record_id=f"synthea-{patient_id}",
            dataset_id=dataset_id,
            topic=topic,
            complexity=ComplexityProfile.MODERATE,
            modalities=[Modality.STRUCTURED_EHR, Modality.LABS],
            patient=patient,
            encounters=encounters,
            labs=labs,
            provenance=Provenance(
                generator="synthea-fhir-import",
                created_at=created_at,
                source_refs=[{"path": path}],
            ),
            metadata={"source": "synthea"},
        )


def _resources(resources: list[dict], resource_type: str) -> list[dict]:
    return [resource for resource in resources if resource.get("resourceType") == resource_type]


def _first_resource(resources: list[dict], resource_type: str) -> dict:
    matches = _resources(resources, resource_type)
    return matches[0] if matches else {}


def _topic(encounters: list[dict]) -> str | None:
    for encounter in encounters:
        reason = _reason(encounter)
        if reason:
            return reason
    return None


def _reason(encounter: dict) -> str | None:
    reasons = encounter.get("reasonCode") or []
    if reasons:
        return reasons[0].get("text") or reasons[0].get("coding", [{}])[0].get("display")
    return None


def _encounter_start(encounters: list[dict]) -> str | None:
    for encounter in encounters:
        start = encounter.get("period", {}).get("start")
        if start:
            return start
    return None


def _age(birth_date: str | None, created_at: str) -> int:
    if not birth_date:
        return 0
    try:
        birth = _partial_fhir_date(birth_date)
        current = _partial_fhir_date(created_at)
    except (TypeError, ValueError):
        return 0
    return current.year - birth.year - ((current.month, current.day) < (birth.month, birth.day))


def _partial_fhir_date(value: str) -> date:
    if not isinstance(value, str):
        raise TypeError("FHIR date must be a string.")
    parts = value[:10].split("-")
    year = int(parts[0])
    month = int(parts[1]) if len(parts) > 1 and parts[1] else 1
    day = int(parts[2]) if len(parts) > 2 and parts[2] else 1
    return date(year, month, day)


def _observation_to_lab(resource: dict, created_at: str) -> LabObservation:
    quantity = resource.get("valueQuantity") or {}
    code = resource.get("code") or {}
    coding = code.get("coding") or [{}]
    primary_code = coding[0] if coding else {}
    quantity_value = quantity.get("value")
    value = quantity_value if quantity_value is not None else (resource.get("valueString") or "")
    return LabObservation(
        name=code.get("text") or primary_code.get("display", "Observation"),
        loinc=primary_code.get("code"),
        value=value,
        unit=quantity.get("unit", ""),
        effective_time=resource.get("effectiveDateTime", created_at),
    )
