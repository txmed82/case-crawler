import json

from casecrawler.integrations.synthea import SyntheaAdapter
from casecrawler.models.synthetic import Modality


def test_synthea_adapter_imports_minimal_fhir_patient_bundle(tmp_path):
    bundle = {
        "resourceType": "Bundle",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "pat-1",
                    "gender": "female",
                    "birthDate": "1970-01-01",
                }
            },
            {
                "resource": {
                    "resourceType": "Encounter",
                    "id": "enc-1",
                    "period": {"start": "2026-01-01T00:00:00"},
                    "reasonCode": [{"text": "sepsis"}],
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"text": "Lactate"},
                    "valueQuantity": {"value": 3.4, "unit": "mmol/L"},
                    "effectiveDateTime": "2026-01-01T01:00:00",
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "category": [{"coding": [{"code": "vital-signs"}]}],
                    "code": {"text": "Heart rate"},
                    "valueQuantity": {"value": 118, "unit": "/min"},
                    "effectiveDateTime": "2026-01-01T01:05:00",
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationStatement",
                    "medicationCodeableConcept": {"text": "Ceftriaxone"},
                    "status": "active",
                    "dosage": [
                        {"text": "1 g daily", "route": {"text": "IV"}}
                    ],
                }
            },
        ],
    }
    path = tmp_path / "patient.json"
    path.write_text(json.dumps(bundle))

    record = SyntheaAdapter().import_fhir_bundle(str(path), dataset_id="ds-1")

    assert record.patient.patient_id == "pat-1"
    assert record.encounters[0].encounter_id == "enc-1"
    assert record.labs[0].name == "Lactate"
    assert record.vitals[0].name == "Heart rate"
    assert record.vitals[0].value == 118
    assert record.medication_history[0].name == "Ceftriaxone"
    assert record.medication_history[0].route == "IV"
    assert Modality.STRUCTURED_EHR in record.modalities
    assert Modality.LABS in record.modalities
    assert Modality.VITALS in record.modalities


def test_synthea_adapter_handles_partial_fhir_dates(tmp_path):
    bundle = {
        "resourceType": "Bundle",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "pat-1",
                    "gender": "female",
                    "birthDate": "1970",
                }
            },
            {
                "resource": {
                    "resourceType": "Encounter",
                    "id": "enc-1",
                    "period": {"start": "2026"},
                    "reasonCode": [{"text": "wellness"}],
                }
            },
        ],
    }
    path = tmp_path / "patient-partial.json"
    path.write_text(json.dumps(bundle))

    record = SyntheaAdapter().import_fhir_bundle(str(path), dataset_id="ds-1")

    assert record.patient.age == 56


def test_synthea_adapter_handles_null_observation_fields(tmp_path):
    bundle = {
        "resourceType": "Bundle",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "pat-1",
                    "gender": "female",
                    "birthDate": 1970,
                }
            },
            {
                "resource": {
                    "resourceType": "Encounter",
                    "id": "enc-1",
                    "period": {"start": "2026-01-01T00:00:00"},
                    "reasonCode": [{"text": "screening"}],
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": None,
                    "valueQuantity": None,
                    "valueString": "positive",
                    "effectiveDateTime": "2026-01-01T01:00:00",
                }
            },
        ],
    }
    path = tmp_path / "patient-null-observation.json"
    path.write_text(json.dumps(bundle))

    record = SyntheaAdapter().import_fhir_bundle(str(path), dataset_id="ds-1")

    assert record.patient.age == 0
    assert record.labs[0].name == "Observation"
    assert record.labs[0].value == "positive"


def test_synthea_adapter_skips_malformed_entries_and_null_periods(tmp_path):
    bundle = {
        "resourceType": "Bundle",
        "entry": [
            None,
            {"resource": None},
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "pat-1",
                    "gender": "unknown",
                    "birthDate": "1980-01-01",
                }
            },
            {
                "resource": {
                    "resourceType": "Encounter",
                    "id": "enc-1",
                    "period": None,
                    "reasonCode": [{"text": "screening"}],
                }
            },
        ],
    }
    path = tmp_path / "patient-malformed-entries.json"
    path.write_text(json.dumps(bundle))

    record = SyntheaAdapter().import_fhir_bundle(str(path), dataset_id="ds-1")

    assert record.patient.patient_id == "pat-1"
    assert record.encounters[0].start == "2026-01-01T00:00:00"
