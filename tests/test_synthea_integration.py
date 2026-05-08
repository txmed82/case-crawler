import json

import pytest

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
                    "maritalStatus": {"text": "Married"},
                    "communication": [
                        {"language": {"coding": [{"display": "English"}]}}
                    ],
                    "address": [
                        {
                            "city": "Austin",
                            "state": "TX",
                            "postalCode": "78701",
                            "country": "US",
                        }
                    ],
                    "extension": [
                        {
                            "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race",
                            "extension": [
                                {
                                    "url": "ombCategory",
                                    "valueCoding": {"display": "White"},
                                }
                            ],
                        },
                        {
                            "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity",
                            "extension": [
                                {
                                    "url": "ombCategory",
                                    "valueCoding": {"display": "Not Hispanic or Latino"},
                                }
                            ],
                        },
                    ],
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
    assert record.patient.demographics["birth_date"] == "1970-01-01"
    assert record.patient.demographics["marital_status"] == "Married"
    assert record.patient.demographics["languages"] == ["English"]
    assert record.patient.demographics["race"] == "White"
    assert record.patient.demographics["ethnicity"] == "Not Hispanic or Latino"
    assert record.patient.demographics["address"] == {
        "city": "Austin",
        "state": "TX",
        "postalCode": "78701",
        "country": "US",
    }
    assert record.encounters[0].encounter_id == "enc-1"
    assert record.labs[0].name == "Lactate"
    assert record.vitals[0].name == "Heart rate"
    assert record.vitals[0].value == 118
    assert record.medication_history[0].name == "Ceftriaxone"
    assert record.medication_history[0].route == "IV"
    assert record.metadata["reference_key"] == "synthea_fhir"
    assert record.metadata["reference_dataset"] == "synthea_fhir"
    assert Modality.STRUCTURED_EHR in record.modalities
    assert Modality.LABS in record.modalities
    assert Modality.VITALS in record.modalities


def test_synthea_adapter_imports_conditions_and_diagnostic_reports(tmp_path):
    bundle = {
        "resourceType": "Bundle",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "pat-report",
                    "gender": "male",
                    "birthDate": "1965-01-01",
                }
            },
            {
                "resource": {
                    "resourceType": "Encounter",
                    "id": "enc-report",
                    "period": {"start": "2026-01-01T00:00:00"},
                    "reasonCode": [{"text": "dyspnea"}],
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "cond-1",
                    "code": {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": "233604007",
                                "display": "Pneumonia",
                            }
                        ],
                        "text": "Community acquired pneumonia",
                    },
                }
            },
            {
                "resource": {
                    "resourceType": "DiagnosticReport",
                    "id": "dr-1",
                    "category": [
                        {"coding": [{"code": "RAD", "display": "Radiology"}]}
                    ],
                    "code": {"text": "Chest radiograph"},
                    "effectiveDateTime": "2026-01-01T01:30:00",
                    "conclusion": "Right lower lobe opacity suspicious for pneumonia.",
                }
            },
        ],
    }
    path = tmp_path / "patient-report.json"
    path.write_text(json.dumps(bundle))

    record = SyntheaAdapter().import_fhir_bundle(str(path), dataset_id="ds-1")

    assert record.encounters[0].diagnoses[0].display == "Community acquired pneumonia"
    assert record.encounters[0].diagnoses[0].code == "233604007"
    assert record.documents[0].document_id == "synthea-dr-1"
    assert record.documents[0].note_type == "radiology_report"
    assert record.documents[0].author_role == "radiologist"
    assert "Right lower lobe opacity" in record.documents[0].clean_text
    assert record.documents[0].extracted_facts["diagnoses"][0]["display"] == (
        "Community acquired pneumonia"
    )
    assert Modality.CLINICAL_TEXT in record.modalities


def test_synthea_adapter_imports_procedures(tmp_path):
    bundle = {
        "resourceType": "Bundle",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "pat-proc",
                    "gender": "female",
                    "birthDate": "1980-01-01",
                }
            },
            {
                "resource": {
                    "resourceType": "Encounter",
                    "id": "enc-proc",
                    "period": {"start": "2026-01-01T00:00:00"},
                    "reasonCode": [{"text": "acute coronary syndrome"}],
                }
            },
            {
                "resource": {
                    "resourceType": "Procedure",
                    "id": "proc-1",
                    "code": {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": "415070008",
                                "display": "Percutaneous coronary intervention",
                            }
                        ],
                    },
                }
            },
        ],
    }
    path = tmp_path / "patient-procedure.json"
    path.write_text(json.dumps(bundle))

    record = SyntheaAdapter().import_fhir_bundle(str(path), dataset_id="ds-1")

    assert record.encounters[0].procedures[0].code == "415070008"
    assert record.encounters[0].procedures[0].display == (
        "Percutaneous coronary intervention"
    )


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


def test_synthea_adapter_imports_bundle_directory_in_stable_order(tmp_path):
    first = {
        "resourceType": "Bundle",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "pat-b"}},
        ],
    }
    second = {
        "resourceType": "Bundle",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "pat-a"}},
        ],
    }
    (tmp_path / "b.json").write_text(json.dumps(first))
    (tmp_path / "a.json").write_text(json.dumps(second))
    (tmp_path / "ignored.txt").write_text("not json")

    records = SyntheaAdapter().import_fhir_path(str(tmp_path), dataset_id="ds-1")

    assert [record.patient.patient_id for record in records] == ["pat-a", "pat-b"]


def test_synthea_adapter_imports_fhir_ndjson_directory_grouped_by_patient(tmp_path):
    (tmp_path / "Patient.ndjson").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "resourceType": "Patient",
                        "id": "pat-a",
                        "gender": "female",
                        "birthDate": "1970-01-01",
                    }
                ),
                json.dumps(
                    {
                        "resourceType": "Patient",
                        "id": "pat-b",
                        "gender": "male",
                        "birthDate": "1980-01-01",
                    }
                ),
            ]
        )
        + "\n"
    )
    (tmp_path / "Encounter.ndjson").write_text(
        json.dumps(
            {
                "resourceType": "Encounter",
                "id": "enc-a",
                "subject": {"reference": "Patient/pat-a"},
                "period": {"start": "2026-01-01T00:00:00"},
                "reasonCode": [{"text": "sepsis"}],
            }
        )
        + "\n"
    )
    (tmp_path / "Observation.ndjson").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "resourceType": "Observation",
                        "id": "obs-a",
                        "subject": {"reference": "Patient/pat-a"},
                        "code": {"text": "Lactate"},
                        "valueQuantity": {"value": 3.2, "unit": "mmol/L"},
                        "effectiveDateTime": "2026-01-01T01:00:00",
                    }
                ),
                json.dumps(
                    {
                        "resourceType": "Observation",
                        "id": "obs-b",
                        "subject": {"reference": "Patient/pat-b"},
                        "category": [{"coding": [{"code": "vital-signs"}]}],
                        "code": {"text": "Heart rate"},
                        "valueQuantity": {"value": 88, "unit": "/min"},
                        "effectiveDateTime": "2026-01-01T01:00:00",
                    }
                ),
            ]
        )
        + "\n"
    )

    records = SyntheaAdapter().import_fhir_ndjson_path(
        str(tmp_path),
        dataset_id="ds-ndjson",
    )

    assert [record.patient.patient_id for record in records] == ["pat-a", "pat-b"]
    pat_a = records[0]
    pat_b = records[1]
    assert pat_a.topic == "sepsis"
    assert pat_a.labs[0].name == "Lactate"
    assert pat_b.vitals[0].name == "Heart rate"
    assert pat_a.provenance.source_refs[0]["format"] == "fhir_ndjson"
    assert pat_a.metadata["source_format"] == "fhir_ndjson"


def test_synthea_adapter_auto_detects_ndjson_output_directory(tmp_path):
    (tmp_path / "Patient.ndjson").write_text(
        json.dumps({"resourceType": "Patient", "id": "pat-ndjson"}) + "\n"
    )

    records = SyntheaAdapter().import_fhir_path(str(tmp_path), dataset_id="ds-ndjson")

    assert len(records) == 1
    assert records[0].patient.patient_id == "pat-ndjson"
    assert records[0].metadata["source_format"] == "fhir_ndjson"


def test_synthea_adapter_runs_command_and_imports_output_directory(tmp_path):
    output_dir = tmp_path / "fhir"
    output_dir.mkdir()

    def fake_runner(command: list[str]) -> None:
        assert command == ["/opt/synthea/run_synthea", "-p", "2"]
        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {"resource": {"resourceType": "Patient", "id": "pat-run"}},
            ],
        }
        (output_dir / "patient.json").write_text(json.dumps(bundle))

    records = SyntheaAdapter(runner=fake_runner).run_and_import(
        executable="/opt/synthea/run_synthea",
        output_dir=str(output_dir),
        dataset_id="ds-synthea",
        population=2,
    )

    assert len(records) == 1
    assert records[0].patient.patient_id == "pat-run"


def test_synthea_adapter_rejects_invalid_population(tmp_path):
    with pytest.raises(ValueError, match="population"):
        SyntheaAdapter().run_and_import(
            executable="/opt/synthea/run_synthea",
            output_dir=str(tmp_path),
            dataset_id="ds-synthea",
            population=0,
        )
