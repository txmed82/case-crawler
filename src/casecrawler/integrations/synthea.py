from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Protocol

from casecrawler.models.synthetic import (
    ClinicalDocument,
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


SYNTHEA_TIMEOUT_SECONDS = 600.0


class SyntheaRunner(Protocol):
    def __call__(self, command: list[str]) -> None: ...


class SyntheaAdapter:
    def __init__(self, runner: SyntheaRunner | None = None) -> None:
        self._runner = runner or _run_synthea_command

    def run_and_import(
        self,
        *,
        executable: str,
        output_dir: str,
        dataset_id: str,
        population: int = 1,
    ) -> list[SyntheticRecord]:
        if population < 1:
            raise ValueError("population must be at least 1.")
        command = [executable, "-p", str(population)]
        self._runner(command)
        return self.import_fhir_path(output_dir, dataset_id=dataset_id)

    def import_fhir_path(self, path: str, dataset_id: str) -> list[SyntheticRecord]:
        source = Path(path)
        if source.is_dir():
            bundle_paths = sorted(
                item
                for item in source.iterdir()
                if item.is_file() and item.suffix.lower() == ".json"
            )
            return [
                self.import_fhir_bundle(str(bundle_path), dataset_id=dataset_id)
                for bundle_path in bundle_paths
            ]
        return [self.import_fhir_bundle(str(source), dataset_id=dataset_id)]

    def import_fhir_bundle(self, path: str, dataset_id: str) -> SyntheticRecord:
        bundle = json.loads(Path(path).read_text())
        resources = []
        for entry in bundle.get("entry", []):
            if not isinstance(entry, Mapping):
                continue
            resource = entry.get("resource", {})
            if isinstance(resource, Mapping):
                resources.append(resource)
        patient_resource = _first_resource(resources, "Patient")
        encounter_resources = _resources(resources, "Encounter")
        condition_resources = _resources(resources, "Condition")
        procedure_resources = _resources(resources, "Procedure")
        diagnostic_report_resources = _resources(resources, "DiagnosticReport")
        observation_resources = _resources(resources, "Observation")
        medication_resources = _resources(resources, "MedicationStatement")

        patient_id = patient_resource.get("id", "synthea-patient")
        topic = _topic(encounter_resources) or "synthea import"
        created_at = _encounter_start(encounter_resources) or "2026-01-01T00:00:00"

        patient = SyntheticPatient(
            patient_id=patient_id,
            age=_age(patient_resource.get("birthDate"), created_at),
            sex=patient_resource.get("gender", "unknown"),
            demographics={"source": "synthea_fhir"},
        )
        diagnoses = [
            diagnosis
            for diagnosis in (_condition_to_code(resource) for resource in condition_resources)
            if diagnosis is not None
        ]
        procedures = [
            procedure
            for procedure in (_procedure_to_code(resource) for resource in procedure_resources)
            if procedure is not None
        ]
        encounters = []
        for index, resource in enumerate(encounter_resources):
            raw_period = resource.get("period") or {}
            period = raw_period if isinstance(raw_period, Mapping) else {}
            encounters.append(
                Encounter(
                    encounter_id=resource.get("id", f"enc-{index}"),
                    start=period.get("start", created_at),
                    end=period.get("end"),
                    setting="synthea",
                    reason=_reason(resource) or topic,
                    diagnoses=diagnoses if index == 0 else [],
                    procedures=procedures if index == 0 else [],
                )
            )
        labs: list[LabObservation] = []
        vitals: list[VitalObservation] = []
        for resource in observation_resources:
            if _is_vital_observation(resource):
                vital = _observation_to_vital(resource, created_at)
                if vital is not None:
                    vitals.append(vital)
            else:
                labs.append(_observation_to_lab(resource, created_at))
        medications = [
            medication
            for medication in (
                _medication_statement(resource) for resource in medication_resources
            )
            if medication is not None
        ]
        documents = [
            document
            for document in (
                _diagnostic_report_to_document(
                    resource,
                    created_at=created_at,
                    diagnoses=diagnoses,
                )
                for resource in diagnostic_report_resources
            )
            if document is not None
        ]

        return SyntheticRecord(
            record_id=f"synthea-{patient_id}",
            dataset_id=dataset_id,
            topic=topic,
            complexity=ComplexityProfile.MODERATE,
            modalities=_modalities(labs=labs, vitals=vitals, documents=documents),
            patient=patient,
            encounters=encounters,
            labs=labs,
            vitals=vitals,
            documents=documents,
            medication_history=medications,
            provenance=Provenance(
                generator="synthea-fhir-import",
                created_at=created_at,
                source_refs=[{"path": path}],
            ),
            metadata={
                "source": "synthea",
                "reference_key": "synthea_fhir",
                "reference_dataset": "synthea_fhir",
            },
        )


def _run_synthea_command(command: list[str]) -> None:
    subprocess.run(
        command,
        check=True,
        timeout=SYNTHEA_TIMEOUT_SECONDS,
    )


def _condition_to_code(resource: dict) -> Code | None:
    codeable = resource.get("code")
    if not isinstance(codeable, Mapping):
        return None
    return _codeable_concept_to_code(codeable, fallback_code=resource.get("id"))


def _procedure_to_code(resource: dict) -> Code | None:
    codeable = resource.get("code")
    if not isinstance(codeable, Mapping):
        return None
    return _codeable_concept_to_code(codeable, fallback_code=resource.get("id"))


def _codeable_concept_to_code(
    codeable: Mapping,
    *,
    fallback_code: object = None,
) -> Code | None:
    codings = codeable.get("coding") or []
    primary = next((item for item in codings if isinstance(item, Mapping)), {})
    display = codeable.get("text") or primary.get("display")
    code = primary.get("code") or fallback_code
    if not display and not code:
        return None
    return Code(
        system=str(primary.get("system") or "synthea-fhir"),
        code=str(code or display),
        display=str(display or code),
    )


def _diagnostic_report_to_document(
    resource: dict,
    *,
    created_at: str,
    diagnoses: list[Code],
) -> ClinicalDocument | None:
    text = _diagnostic_report_text(resource)
    if not text:
        return None
    note_type, author_role = _diagnostic_report_document_type(resource)
    code = resource.get("code") if isinstance(resource.get("code"), Mapping) else {}
    report_name = code.get("text") or _codeable_concept_to_code(code or {}) or "Diagnostic report"
    if isinstance(report_name, Code):
        report_name = report_name.display
    clean_text = f"{report_name}: {text}"
    return ClinicalDocument(
        document_id=f"synthea-{resource.get('id', 'diagnostic-report')}",
        note_type=note_type,
        author_role=author_role,
        timestamp=resource.get("effectiveDateTime") or resource.get("issued") or created_at,
        clean_text=clean_text,
        messy_text=_synthea_messy_text(clean_text),
        extracted_facts={
            "diagnoses": [diagnosis.model_dump() for diagnosis in diagnoses],
            "report_code": code.get("text") if isinstance(code, Mapping) else None,
        },
    )


def _diagnostic_report_text(resource: dict) -> str:
    conclusion = resource.get("conclusion")
    if isinstance(conclusion, str) and conclusion.strip():
        return conclusion.strip()
    presented_forms = resource.get("presentedForm") or []
    for form in presented_forms:
        if not isinstance(form, Mapping):
            continue
        title = form.get("title")
        data = form.get("data")
        if isinstance(title, str) and title.strip():
            return title.strip()
        if isinstance(data, str) and data.strip():
            return data.strip()
    return ""


def _diagnostic_report_document_type(resource: dict) -> tuple[str, str]:
    for category in resource.get("category") or []:
        if not isinstance(category, Mapping):
            continue
        for coding in category.get("coding") or []:
            if not isinstance(coding, Mapping):
                continue
            code = str(coding.get("code") or "").lower()
            display = str(coding.get("display") or "").lower()
            if code in {"rad", "radiology"} or "radiology" in display:
                return "radiology_report", "radiologist"
    return "diagnostic_report", "synthea"


def _synthea_messy_text(clean_text: str) -> str:
    return (
        clean_text.replace("Patient", "Pt")
        .replace("patient", "pt")
        .replace("with", "w/")
        .replace("Right", "R")
        .replace("left", "L")
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
        raw_period = encounter.get("period") or {}
        period = raw_period if isinstance(raw_period, Mapping) else {}
        start = period.get("start")
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


def _modalities(
    *,
    labs: list[LabObservation],
    vitals: list[VitalObservation],
    documents: list[ClinicalDocument] | None = None,
) -> list[Modality]:
    modalities = [Modality.STRUCTURED_EHR]
    if documents:
        modalities.append(Modality.CLINICAL_TEXT)
    if labs:
        modalities.append(Modality.LABS)
    if vitals:
        modalities.append(Modality.VITALS)
    return modalities


def _is_vital_observation(resource: dict) -> bool:
    for category in resource.get("category") or []:
        if not isinstance(category, Mapping):
            continue
        for coding in category.get("coding") or []:
            if isinstance(coding, Mapping) and coding.get("code") == "vital-signs":
                return True
    return False


def _observation_to_vital(resource: dict, created_at: str) -> VitalObservation | None:
    quantity = resource.get("valueQuantity") or {}
    value = quantity.get("value")
    if not isinstance(value, (int, float)):
        return None
    code = resource.get("code") or {}
    coding = code.get("coding") or [{}]
    primary_code = coding[0] if coding else {}
    return VitalObservation(
        name=code.get("text") or primary_code.get("display", "Vital sign"),
        value=float(value),
        unit=quantity.get("unit", ""),
        effective_time=resource.get("effectiveDateTime", created_at),
    )


def _medication_statement(resource: dict) -> MedicationStatement | None:
    concept = resource.get("medicationCodeableConcept") or {}
    name = concept.get("text") if isinstance(concept, Mapping) else None
    if not name:
        return None
    dosage = next(
        (item for item in resource.get("dosage") or [] if isinstance(item, Mapping)),
        {},
    )
    route = dosage.get("route") or {}
    return MedicationStatement(
        name=name,
        dose=dosage.get("text"),
        route=route.get("text") if isinstance(route, Mapping) else None,
        status=resource.get("status", "unknown"),
    )
