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
        ],
    }
    path = tmp_path / "patient.json"
    path.write_text(json.dumps(bundle))

    record = SyntheaAdapter().import_fhir_bundle(str(path), dataset_id="ds-1")

    assert record.patient.patient_id == "pat-1"
    assert record.encounters[0].encounter_id == "enc-1"
    assert record.labs[0].name == "Lactate"
    assert Modality.STRUCTURED_EHR in record.modalities


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
