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
SYNTHEA_REFERENCE_KEY = "synthea_fhir"
SYNTHEA_REFERENCE_DESCRIPTION = (
    "Local Synthea FHIR imports for standards-shaped synthetic patient history "
    "benchmarking. Use import-synthea-fhir or run-synthea to create this reference."
)


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
            if bundle_paths:
                return [
                    self.import_fhir_bundle(str(bundle_path), dataset_id=dataset_id)
                    for bundle_path in bundle_paths
                ]
            ndjson_paths = sorted(source.glob("*.ndjson"))
            if ndjson_paths:
                return self.import_fhir_ndjson_path(path, dataset_id=dataset_id)
            return []
        if source.suffix.lower() == ".ndjson":
            return self.import_fhir_ndjson_path(path, dataset_id=dataset_id)
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
        return self.import_fhir_resources(
            resources,
            dataset_id=dataset_id,
            source_ref={"path": path, "format": "fhir_bundle"},
        )

    def import_fhir_ndjson_path(self, path: str, dataset_id: str) -> list[SyntheticRecord]:
        source = Path(path)
        ndjson_paths = sorted(source.glob("*.ndjson")) if source.is_dir() else [source]
        resources = [
            resource
            for ndjson_path in ndjson_paths
            for resource in _read_ndjson_resources(ndjson_path)
        ]
        grouped = _group_resources_by_patient(resources)
        records = []
        for patient_id in sorted(grouped):
            source_ref = {
                "path": str(source),
                "format": "fhir_ndjson",
                "patient_id": patient_id,
            }
            record = self.import_fhir_resources(
                grouped[patient_id],
                dataset_id=dataset_id,
                source_ref=source_ref,
            )
            records.append(
                record.model_copy(
                    update={
                        "metadata": {
                            **record.metadata,
                            "source_format": "fhir_ndjson",
                        }
                    }
                )
            )
        return records

    def import_fhir_resources(
        self,
        resources: list[Mapping],
        *,
        dataset_id: str,
        source_ref: dict,
    ) -> SyntheticRecord:
        resources = [
            dict(resource)
            for resource in resources
            if isinstance(resource, Mapping)
        ]
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
            demographics=_patient_demographics(patient_resource),
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
                source_refs=[source_ref],
            ),
            metadata={
                "source": "synthea",
                "reference_key": SYNTHEA_REFERENCE_KEY,
                "reference_dataset": SYNTHEA_REFERENCE_KEY,
            },
        )


def _read_ndjson_resources(path: Path) -> list[dict]:
    resources = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, Mapping):
            resources.append(dict(value))
    return resources


def _group_resources_by_patient(resources: list[dict]) -> dict[str, list[dict]]:
    patients = {
        str(resource.get("id")): [resource]
        for resource in resources
        if resource.get("resourceType") == "Patient" and resource.get("id")
    }
    if not patients:
        return {"synthea-patient": resources}

    encounter_patient_ids = {
        str(resource.get("id")): patient_id
        for resource in resources
        if resource.get("resourceType") == "Encounter"
        and resource.get("id")
        and (patient_id := _resource_patient_id(resource))
    }

    grouped = {patient_id: list(patient_resources) for patient_id, patient_resources in patients.items()}
    for resource in resources:
        if resource.get("resourceType") == "Patient":
            continue
        patient_id = _resource_patient_id(resource)
        if patient_id is None:
            patient_id = _resource_encounter_patient_id(resource, encounter_patient_ids)
        if patient_id is None:
            continue
        grouped.setdefault(patient_id, []).append(resource)
    return grouped


def _resource_patient_id(resource: Mapping) -> str | None:
    subject = resource.get("subject")
    if isinstance(subject, Mapping):
        return _patient_id_from_reference(subject.get("reference"))
    patient = resource.get("patient")
    if isinstance(patient, Mapping):
        return _patient_id_from_reference(patient.get("reference"))
    return None


def _resource_encounter_patient_id(
    resource: Mapping,
    encounter_patient_ids: dict[str, str],
) -> str | None:
    encounter = resource.get("encounter")
    if not isinstance(encounter, Mapping):
        return None
    reference = encounter.get("reference")
    if not isinstance(reference, str):
        return None
    encounter_id = reference.rsplit("/", 1)[-1]
    return encounter_patient_ids.get(encounter_id)


def _patient_id_from_reference(reference: object) -> str | None:
    if not isinstance(reference, str) or not reference:
        return None
    return reference.rsplit("/", 1)[-1] if "Patient/" in reference else None


def _run_synthea_command(command: list[str]) -> None:
    subprocess.run(
        command,
        check=True,
        timeout=SYNTHEA_TIMEOUT_SECONDS,
    )


def _patient_demographics(resource: dict) -> dict:
    demographics = {"source": "synthea_fhir"}
    if resource.get("birthDate"):
        demographics["birth_date"] = resource["birthDate"]
    if resource.get("maritalStatus"):
        demographics["marital_status"] = _codeable_text(resource["maritalStatus"])
    if resource.get("communication"):
        languages = []
        for item in resource.get("communication") or []:
            if not isinstance(item, Mapping):
                continue
            language = item.get("language")
            if isinstance(language, Mapping):
                languages.append(_codeable_text(language))
        if languages:
            demographics["languages"] = [language for language in languages if language]
    race = _patient_us_core_extension(resource, "us-core-race")
    ethnicity = _patient_us_core_extension(resource, "us-core-ethnicity")
    if race:
        demographics["race"] = race
    if ethnicity:
        demographics["ethnicity"] = ethnicity
    address = next(
        (item for item in resource.get("address") or [] if isinstance(item, Mapping)),
        {},
    )
    if address:
        demographics["address"] = {
            key: address[key]
            for key in ("city", "state", "postalCode", "country")
            if address.get(key)
        }
    return demographics


def _patient_us_core_extension(resource: dict, suffix: str) -> str | None:
    for extension in resource.get("extension") or []:
        if not isinstance(extension, Mapping):
            continue
        url = str(extension.get("url") or "")
        if not url.endswith(suffix):
            continue
        for nested in extension.get("extension") or []:
            if not isinstance(nested, Mapping):
                continue
            value = nested.get("valueCoding")
            if isinstance(value, Mapping):
                return str(value.get("display") or value.get("code") or "")
            value_string = nested.get("valueString")
            if isinstance(value_string, str) and value_string:
                return value_string
    return None


def _codeable_text(value) -> str | None:
    if not isinstance(value, Mapping):
        return None
    if value.get("text"):
        return str(value["text"])
    coding = next(
        (item for item in value.get("coding") or [] if isinstance(item, Mapping)),
        {},
    )
    display = coding.get("display") or coding.get("code")
    return str(display) if display else None


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
