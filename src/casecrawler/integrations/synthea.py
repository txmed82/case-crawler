from __future__ import annotations

import csv
import json
import subprocess
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Protocol

from casecrawler.models.synthetic import (
    AllergyIntolerance,
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


SYNTHEA_TIMEOUT_SECONDS = 600.0
SYNTHEA_REFERENCE_KEY = "synthea_fhir"
SYNTHEA_REFERENCE_DESCRIPTION = (
    "Local Synthea FHIR or CSV imports for standards-shaped synthetic patient "
    "history benchmarking. Use import-synthea or run-synthea to create this reference."
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
            if _has_synthea_csv_files(source):
                return self.import_csv_path(path, dataset_id=dataset_id)
            return []
        if source.suffix.lower() == ".ndjson":
            return self.import_fhir_ndjson_path(path, dataset_id=dataset_id)
        return [self.import_fhir_bundle(str(source), dataset_id=dataset_id)]

    def import_csv_path(self, path: str, dataset_id: str) -> list[SyntheticRecord]:
        source = Path(path)
        if not source.is_dir():
            raise ValueError("Synthea CSV import requires a directory.")
        patients = _read_csv_table(source / "patients.csv")
        if not patients:
            return []
        encounters_by_patient = _csv_rows_by_patient(
            _read_csv_table(source / "encounters.csv")
        )
        conditions_by_patient = _csv_rows_by_patient(
            _read_csv_table(source / "conditions.csv")
        )
        procedures_by_patient = _csv_rows_by_patient(
            _read_csv_table(source / "procedures.csv")
        )
        observations_by_patient = _csv_rows_by_patient(
            _read_csv_table(source / "observations.csv")
        )
        medications_by_patient = _csv_rows_by_patient(
            _read_csv_table(source / "medications.csv")
        )
        records = []
        for patient_row in sorted(patients, key=lambda row: _csv_value(row, "Id")):
            patient_id = _csv_value(patient_row, "Id") or "synthea-patient"
            patient_encounters = encounters_by_patient.get(patient_id, [])
            created_at = (
                _csv_value(patient_encounters[0], "START")
                if patient_encounters
                else "2026-01-01T00:00:00"
            )
            diagnoses = [
                code
                for code in (
                    _csv_code(row, default_system="synthea-csv-condition")
                    for row in conditions_by_patient.get(patient_id, [])
                )
                if code is not None
            ]
            procedures = [
                code
                for code in (
                    _csv_code(row, default_system="synthea-csv-procedure")
                    for row in procedures_by_patient.get(patient_id, [])
                )
                if code is not None
            ]
            encounters = _csv_encounters(
                patient_encounters,
                diagnoses=diagnoses,
                procedures=procedures,
                created_at=created_at,
            )
            labs, vitals = _csv_observations(
                observations_by_patient.get(patient_id, []),
                created_at=created_at,
            )
            medications = [
                medication
                for medication in (
                    _csv_medication(row)
                    for row in medications_by_patient.get(patient_id, [])
                )
                if medication is not None
            ]
            topic = _csv_topic(patient_encounters, conditions_by_patient.get(patient_id, []))
            records.append(
                SyntheticRecord(
                    record_id=f"synthea-{patient_id}",
                    dataset_id=dataset_id,
                    topic=topic,
                    complexity=ComplexityProfile.MODERATE,
                    modalities=_modalities(labs=labs, vitals=vitals, documents=[]),
                    patient=_csv_patient(patient_row, created_at),
                    encounters=encounters,
                    labs=labs,
                    vitals=vitals,
                    medication_history=medications,
                    provenance=Provenance(
                        generator="synthea-csv-import",
                        created_at=created_at,
                        source_refs=[
                            {
                                "path": str(source),
                                "format": "synthea_csv",
                                "patient_id": patient_id,
                            }
                        ],
                    ),
                    metadata={
                        "source": "synthea",
                        "source_format": "synthea_csv",
                        "reference_key": SYNTHEA_REFERENCE_KEY,
                        "reference_dataset": SYNTHEA_REFERENCE_KEY,
                    },
                )
            )
        return records

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
        medication_request_resources = _resources(resources, "MedicationRequest")
        service_request_resources = _resources(resources, "ServiceRequest")
        allergy_resources = _resources(resources, "AllergyIntolerance")

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
            components = _observation_components(resource)
            if components:
                if _is_vital_observation(resource):
                    vitals.extend(
                        vital
                        for vital in (
                            _component_to_vital(resource, component, created_at)
                            for component in components
                        )
                        if vital is not None
                    )
                else:
                    labs.extend(
                        _component_to_lab(resource, component, created_at)
                        for component in components
                    )
                continue
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
        allergies = [
            allergy
            for allergy in (_allergy_intolerance(resource) for resource in allergy_resources)
            if allergy is not None
        ]
        orders = [
            order
            for order in (
                [
                    *(
                        _medication_request_order(resource, created_at)
                        for resource in medication_request_resources
                    ),
                    *(
                        _service_request_order(resource, created_at)
                        for resource in service_request_resources
                    ),
                ]
            )
            if order is not None
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
            allergies=allergies,
            orders=orders,
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


def _has_synthea_csv_files(path: Path) -> bool:
    return (path / "patients.csv").is_file()


def _read_csv_table(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _csv_rows_by_patient(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        patient_id = _csv_value(row, "PATIENT") or _csv_value(row, "Id")
        if not patient_id:
            continue
        grouped.setdefault(patient_id, []).append(row)
    return grouped


def _csv_patient(row: dict[str, str], created_at: str) -> SyntheticPatient:
    patient_id = _csv_value(row, "Id") or "synthea-patient"
    gender = _csv_value(row, "GENDER").lower()
    sex_map = {"f": "female", "m": "male"}
    demographics = {"source": "synthea_csv"}
    birth_date = _csv_value(row, "BIRTHDATE")
    if birth_date:
        demographics["birth_date"] = birth_date
    for key, csv_key in (
        ("race", "RACE"),
        ("ethnicity", "ETHNICITY"),
        ("marital_status", "MARITAL"),
        ("birthplace", "BIRTHPLACE"),
    ):
        value = _csv_value(row, csv_key)
        if value:
            demographics[key] = value
    address = {
        target: value
        for target, value in (
            ("city", _csv_value(row, "CITY")),
            ("state", _csv_value(row, "STATE")),
            ("postalCode", _csv_value(row, "ZIP")),
            ("county", _csv_value(row, "COUNTY")),
        )
        if value
    }
    if address:
        demographics["address"] = address
    return SyntheticPatient(
        patient_id=patient_id,
        age=_age(birth_date, created_at),
        sex=sex_map.get(gender, gender or "unknown"),
        demographics=demographics,
    )


def _csv_encounters(
    rows: list[dict[str, str]],
    *,
    diagnoses: list[Code],
    procedures: list[Code],
    created_at: str,
) -> list[Encounter]:
    if not rows:
        return [
            Encounter(
                encounter_id="synthea-csv-encounter",
                start=created_at,
                setting="synthea_csv",
                reason="synthea csv import",
                diagnoses=diagnoses,
                procedures=procedures,
            )
        ]
    encounters = []
    for index, row in enumerate(rows):
        reason = (
            _csv_value(row, "REASONDESCRIPTION")
            or _csv_value(row, "DESCRIPTION")
            or "synthea csv import"
        )
        encounter_diagnoses = diagnoses if index == 0 else []
        encounter_procedures = procedures if index == 0 else []
        encounters.append(
            Encounter(
                encounter_id=_csv_value(row, "Id") or f"synthea-csv-enc-{index}",
                start=_csv_value(row, "START") or created_at,
                end=_csv_value(row, "STOP") or None,
                setting=_csv_value(row, "ENCOUNTERCLASS") or "synthea_csv",
                reason=reason,
                diagnoses=encounter_diagnoses,
                procedures=encounter_procedures,
            )
        )
    return encounters


def _csv_observations(
    rows: list[dict[str, str]],
    *,
    created_at: str,
) -> tuple[list[LabObservation], list[VitalObservation]]:
    labs: list[LabObservation] = []
    vitals: list[VitalObservation] = []
    for row in rows:
        name = _csv_value(row, "DESCRIPTION") or "Observation"
        value = _csv_number_or_text(_csv_value(row, "VALUE"))
        unit = _csv_value(row, "UNITS")
        effective_time = _csv_value(row, "DATE") or created_at
        code = _csv_value(row, "CODE") or None
        if _csv_observation_is_vital(row):
            numeric_value = _csv_float(value)
            if numeric_value is None:
                continue
            vitals.append(
                VitalObservation(
                    name=name,
                    value=numeric_value,
                    unit=unit,
                    effective_time=effective_time,
                )
            )
            continue
        labs.append(
            LabObservation(
                name=name,
                loinc=code,
                value=value,
                unit=unit,
                effective_time=effective_time,
            )
        )
    return labs, vitals


def _csv_medication(row: dict[str, str]) -> MedicationStatement | None:
    name = _csv_value(row, "DESCRIPTION")
    if not name:
        return None
    start = _csv_value(row, "START")
    stop = _csv_value(row, "STOP")
    status = "stopped" if stop else "active"
    return MedicationStatement(
        name=name,
        dose=None,
        route=None,
        frequency=None,
        start=start or None,
        end=stop or None,
        status=status,
    )


def _csv_code(row: dict[str, str], *, default_system: str) -> Code | None:
    code = _csv_value(row, "CODE")
    display = _csv_value(row, "DESCRIPTION")
    if not code and not display:
        return None
    return Code(
        system=default_system,
        code=code or display,
        display=display or code,
    )


def _csv_topic(
    encounter_rows: list[dict[str, str]],
    condition_rows: list[dict[str, str]],
) -> str:
    for row in encounter_rows:
        topic = _csv_value(row, "REASONDESCRIPTION") or _csv_value(row, "DESCRIPTION")
        if topic:
            return topic
    for row in condition_rows:
        topic = _csv_value(row, "DESCRIPTION")
        if topic:
            return topic
    return "synthea csv import"


def _csv_observation_is_vital(row: dict[str, str]) -> bool:
    code = _csv_value(row, "CODE")
    description = _csv_value(row, "DESCRIPTION").lower()
    vital_codes = {
        "8867-4",
        "9279-1",
        "8310-5",
        "8302-2",
        "29463-7",
        "8480-6",
        "8462-4",
        "2708-6",
    }
    vital_terms = {
        "heart rate",
        "respiratory rate",
        "body temperature",
        "body height",
        "body weight",
        "systolic blood pressure",
        "diastolic blood pressure",
        "oxygen saturation",
    }
    return code in vital_codes or description in vital_terms


def _csv_number_or_text(value: str) -> float | str:
    number = _csv_float(value)
    return number if number is not None else value


def _csv_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _csv_value(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if value is None:
        return ""
    return str(value).strip()


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


def _observation_components(resource: dict) -> list[Mapping]:
    return [
        component
        for component in resource.get("component") or []
        if isinstance(component, Mapping)
        and isinstance(component.get("valueQuantity"), Mapping)
    ]


def _component_to_lab(
    resource: dict,
    component: Mapping,
    created_at: str,
) -> LabObservation:
    quantity = component.get("valueQuantity") or {}
    code = component.get("code") or {}
    primary_code = _primary_coding(code)
    quantity_value = quantity.get("value")
    value = quantity_value if quantity_value is not None else ""
    return LabObservation(
        name=_component_name(code, primary_code, fallback="Observation component"),
        loinc=primary_code.get("code"),
        value=value,
        unit=quantity.get("unit", ""),
        effective_time=resource.get("effectiveDateTime", created_at),
    )


def _component_to_vital(
    resource: dict,
    component: Mapping,
    created_at: str,
) -> VitalObservation | None:
    quantity = component.get("valueQuantity") or {}
    value = quantity.get("value")
    if not isinstance(value, (int, float)):
        return None
    code = component.get("code") or {}
    primary_code = _primary_coding(code)
    return VitalObservation(
        name=_component_name(code, primary_code, fallback="Vital sign component"),
        value=float(value),
        unit=quantity.get("unit", ""),
        effective_time=resource.get("effectiveDateTime", created_at),
    )


def _primary_coding(codeable: object) -> Mapping:
    if not isinstance(codeable, Mapping):
        return {}
    coding = codeable.get("coding") or []
    return next((item for item in coding if isinstance(item, Mapping)), {})


def _component_name(
    codeable: object,
    primary_code: Mapping,
    *,
    fallback: str,
) -> str:
    if isinstance(codeable, Mapping) and codeable.get("text"):
        return str(codeable["text"])
    return str(primary_code.get("display") or primary_code.get("code") or fallback)


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


def _allergy_intolerance(resource: dict) -> AllergyIntolerance | None:
    codeable = resource.get("code") or {}
    if not isinstance(codeable, Mapping):
        return None
    substance = _codeable_text(codeable)
    if not substance:
        return None
    code = _codeable_concept_to_code(codeable, fallback_code=resource.get("id"))
    reaction = next(
        (
            item
            for item in resource.get("reaction") or []
            if isinstance(item, Mapping)
        ),
        {},
    )
    manifestation = next(
        (
            item
            for item in reaction.get("manifestation") or []
            if isinstance(item, Mapping)
        ),
        {},
    )
    return AllergyIntolerance(
        substance=substance,
        code=code.code if code else None,
        system=code.system if code else None,
        reaction=_codeable_text(manifestation),
        severity=reaction.get("severity"),
        status=_codeable_text(resource.get("clinicalStatus")) or "active",
        recorded_at=resource.get("recordedDate"),
    )


def _medication_request_order(resource: dict, created_at: str) -> ClinicalOrder | None:
    concept = resource.get("medicationCodeableConcept") or {}
    if not isinstance(concept, Mapping):
        return None
    display = _codeable_text(concept)
    if not display:
        return None
    code = _codeable_concept_to_code(concept, fallback_code=resource.get("id"))
    return ClinicalOrder(
        order_id=resource.get("id", f"medication-request-{display}"),
        order_type="medication",
        display=display,
        code=code.code if code else None,
        system=code.system if code else None,
        status=resource.get("status", "unknown"),
        intent=resource.get("intent", "order"),
        priority=resource.get("priority"),
        ordered_at=resource.get("authoredOn") or created_at,
        encounter_id=_resource_reference_id(resource.get("encounter")),
    )


def _service_request_order(resource: dict, created_at: str) -> ClinicalOrder | None:
    codeable = resource.get("code") or {}
    if not isinstance(codeable, Mapping):
        return None
    display = _codeable_text(codeable)
    if not display:
        return None
    code = _codeable_concept_to_code(codeable, fallback_code=resource.get("id"))
    return ClinicalOrder(
        order_id=resource.get("id", f"service-request-{display}"),
        order_type=_service_request_order_type(resource),
        display=display,
        code=code.code if code else None,
        system=code.system if code else None,
        status=resource.get("status", "unknown"),
        intent=resource.get("intent", "order"),
        priority=resource.get("priority"),
        ordered_at=resource.get("authoredOn") or created_at,
        encounter_id=_resource_reference_id(resource.get("encounter")),
    )


def _service_request_order_type(resource: dict) -> str:
    categories = resource.get("category") or []
    for category in categories:
        text = _codeable_text(category)
        if text:
            normalized = text.lower()
            if "lab" in normalized:
                return "laboratory"
            if "image" in normalized or "radiology" in normalized:
                return "imaging"
            if "procedure" in normalized:
                return "procedure"
            if "nursing" in normalized:
                return "nursing"
            return normalized.replace(" ", "_")
    return "procedure"


def _resource_reference_id(value) -> str | None:
    if not isinstance(value, Mapping):
        return None
    reference = value.get("reference")
    if not isinstance(reference, str) or not reference:
        return None
    return reference.rsplit("/", 1)[-1]
