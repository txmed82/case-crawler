import json
from pathlib import Path

import pytest

from casecrawler.integrations.huggingface import (
    REFERENCE_DATASETS,
    import_reference_rows,
    list_reference_datasets,
    reference_dataset_spec,
    load_reference_dataset,
    reference_row_to_record,
)
from casecrawler.models.synthetic import Modality


def test_reference_dataset_catalog_includes_asclepius_license():
    catalog = list_reference_datasets()
    asclepius = next(
        item
        for item in catalog
        if item.repo_id == "starmpcc/Asclepius-Synthetic-Clinical-Notes"
    )

    assert asclepius.license == "cc-by-nc-sa-4.0"


def test_reference_dataset_catalog_includes_clinical_note_fhir_and_radiology_benchmarks():
    catalog = {key: spec for key, spec in REFERENCE_DATASETS.items()}

    assert catalog["augmented_clinical_notes"].repo_id == "AGBonnet/augmented-clinical-notes"
    assert catalog["augmented_clinical_notes"].note_field == "full_note"
    assert catalog["augmented_clinical_notes"].answer_field == "summary"
    assert catalog["augmented_clinical_notes"].license == "mit"
    assert catalog["clinical_notes_to_fhir"].repo_id == "ai-galileo/clinical-notes-to-fhir"
    assert catalog["clinical_notes_to_fhir"].note_field == "note"
    assert catalog["clinical_notes_to_fhir"].answer_field == "fhir_bundle"
    assert catalog["clinical_notes_to_fhir"].license == "apache-2.0"
    assert (
        catalog["radiology_report_consistency"].repo_id
        == "ClarusC64/image-report-consistency-radiology-v01"
    )
    assert catalog["radiology_report_consistency"].note_field == "report_excerpt"
    assert catalog["radiology_report_consistency"].question_field == "imaging_findings"
    assert catalog["radiology_report_consistency"].answer_field == "expected_decision"
    assert catalog["radiology_report_consistency"].license == "mit"
    assert catalog["synthchex_75k"].repo_id == "raman07/SynthCheX-75K-v2"
    assert catalog["synthchex_75k"].image_field == "image"
    assert catalog["rexgradient_160k"].repo_id == "rajpurkarlab/ReXGradient-160K"
    assert catalog["rexgradient_160k"].license == "rexgradient-non-commercial-gated"
    assert catalog["rexgradient_160k"].image_field == "image"
    assert catalog["rexgradient_160k"].gated is True
    assert catalog["rexgradient_160k"].use_policy == "non_commercial_research_only"
    assert "gated" in catalog["rexgradient_160k"].description.lower()
    assert catalog["synthetic_chest_xray_pneumonia"].repo_id == (
        "chimbiwide/synthetic-chest-xray-pneumonia"
    )
    assert catalog["synthetic_chest_xray_pneumonia"].image_label_map == {
        "0": "normal",
        "1": "pneumonia",
    }
    assert catalog["medsynth_dialogue_note"].repo_id == "Ahmad0067/MedSynth"
    assert catalog["medsynth_dialogue_note"].note_field == "Note"
    assert catalog["medsynth_dialogue_note"].question_field == "Dialogue"
    assert catalog["medsynth_dialogue_note"].task_field == "ICD10_desc"
    assert catalog["medsynth_dialogue_note"].license == "unspecified"
    assert catalog["technetium_i"].repo_id == "temlm-foundation/Technetium-I"
    assert catalog["technetium_i"].license == "eupl-1.2"
    assert catalog["technetium_i"].note_field == "text"
    assert catalog["technetium_i"].patient_id_field == "note_id"
    assert catalog["technetium_i"].note_type_field == "note_type"
    assert catalog["technetium_i"].phi_annotations_field == "phi_annotations"
    assert catalog["technetium_i"].diagnosis_codes_field == "icd_codes"
    assert catalog["technetium_i"].quality_score_field == "quality_score"


def test_asclepius_row_maps_to_synthetic_record():
    row = {
        "patient_id": 42,
        "note": "Discharge Summary: Patient: 60-year-old male with pneumonia.",
        "question": "Summarize the record.",
        "answer": "The patient has pneumonia.",
        "task": "Summarization",
    }

    record = reference_row_to_record(
        row,
        dataset_id="ds-hf",
        spec=REFERENCE_DATASETS["asclepius"],
        split="validation",
    )

    assert record.dataset_id == "ds-hf"
    assert record.patient.age == 60
    assert record.patient.sex == "male"
    assert record.documents[0].note_type == "discharge_summary"
    assert record.documents[0].extracted_facts["instruction"] == "Summarize the record."
    assert record.metadata["reference_license"] == "cc-by-nc-sa-4.0"
    assert record.metadata["reference_split"] == "validation"
    assert record.provenance.source_refs[0]["split"] == "validation"
    assert record.modalities == [Modality.CLINICAL_TEXT]


def test_medsynth_dialogue_note_row_maps_dialogue_and_icd_metadata():
    row = {
        "Note": "SOAP Note: 52-year-old male with left knee pain.",
        "Dialogue": "[doctor] How can I help? [patient] My left knee hurts.",
        "ICD10": "M25562",
        "ICD10_desc": "PAIN IN LEFT KNEE",
    }

    record = reference_row_to_record(
        row,
        dataset_id="ds-medsynth",
        spec=REFERENCE_DATASETS["medsynth_dialogue_note"],
        reference_key="medsynth_dialogue_note",
    )

    assert record.dataset_id == "ds-medsynth"
    assert record.patient.age == 52
    assert record.patient.sex == "male"
    assert record.topic == "PAIN IN LEFT KNEE"
    assert record.documents[0].clean_text == row["Note"]
    assert record.documents[0].extracted_facts["instruction"] == row["Dialogue"]
    assert record.documents[0].extracted_facts["source_fields"] == {
        "ICD10": "M25562",
        "ICD10_desc": "PAIN IN LEFT KNEE",
    }
    assert record.metadata["reference_dataset"] == "Ahmad0067/MedSynth"
    assert record.metadata["reference_key"] == "medsynth_dialogue_note"
    assert record.modalities == [Modality.CLINICAL_TEXT]


def test_technetium_i_row_maps_phi_annotations_and_icd_codes():
    row = {
        "note_id": "TEMLM_000001",
        "source": "temlm_generated",
        "note_type": "discharge_summary",
        "admission_date": "2015-03-15T00:00:00",
        "discharge_date": "2015-03-22T00:00:00",
        "text": (
            "DISCHARGE SUMMARY\n\n"
            "Patient Name: Smith, John\n"
            "Patient is a 72-year-old male with congestive heart failure."
        ),
        "phi_annotations": [
            {"entity_type": "NAME", "text": "Smith", "start": 32, "end": 37},
            {"entity_type": "AGE", "text": "72-year-old", "start": 57, "end": 68},
        ],
        "icd_codes": ["428.0", "401.9"],
        "quality_score": 0.95,
    }

    record = reference_row_to_record(
        row,
        dataset_id="ds-technetium",
        spec=REFERENCE_DATASETS["technetium_i"],
        reference_key="technetium_i",
        split="validation",
    )

    assert record.dataset_id == "ds-technetium"
    assert record.patient.age == 72
    assert record.patient.sex == "male"
    assert record.topic == "clinical_deidentification_icd_coding"
    assert record.documents[0].note_type == "discharge_summary"
    assert record.documents[0].extracted_facts["phi_annotations"] == row["phi_annotations"]
    assert record.documents[0].extracted_facts["phi_entity_counts"] == {
        "AGE": 1,
        "NAME": 1,
    }
    assert record.documents[0].extracted_facts["diagnoses"] == [
        {"system": "ICD-9-CM", "code": "428.0", "display": "ICD-9-CM 428.0"},
        {"system": "ICD-9-CM", "code": "401.9", "display": "ICD-9-CM 401.9"},
    ]
    assert record.documents[0].extracted_facts["source_quality_score"] == 0.95
    assert record.encounters[0].diagnoses[0].code == "428.0"
    assert record.metadata["reference_key"] == "technetium_i"
    assert record.metadata["reference_license"] == "eupl-1.2"
    assert record.modalities == [Modality.STRUCTURED_EHR, Modality.CLINICAL_TEXT]


def test_fhir_reference_row_preserves_bundle_and_validation_fields():
    row = {
        "exampleId": "10004",
        "difficulty": "easy",
        "scenario": "Annual check-up with diabetes family history.",
        "note": "Patient: Jane Doe, 48-year-old female. HbA1c ordered.",
        "fhir_bundle": (
            '{"resourceType":"Bundle","type":"collection","entry":['
            '{"resource":{"resourceType":"Observation","id":"obs-hba1c",'
            '"code":{"coding":[{"system":"http://loinc.org","code":"4548-4",'
            '"display":"Hemoglobin A1c/Hemoglobin.total in Blood"}],"text":"HbA1c"},'
            '"valueQuantity":{"value":7.4,"unit":"%"},"effectiveDateTime":"2026-01-01T00:00:00",'
            '"referenceRange":[{"low":{"value":4.0,"unit":"%"},"high":{"value":5.6,"unit":"%"}}]}},'
            '{"resource":{"resourceType":"Observation","id":"obs-hr",'
            '"category":[{"coding":[{"code":"vital-signs"}]}],'
            '"code":{"text":"Heart rate"},"valueQuantity":{"value":88,"unit":"/min"},'
            '"effectiveDateTime":"2026-01-01T00:05:00"}},'
            '{"resource":{"resourceType":"MedicationStatement","id":"med-metformin",'
            '"medicationCodeableConcept":{"text":"Metformin"},'
            '"status":"active","dosage":[{"route":{"text":"PO"},"text":"500 mg twice daily"}]}},'
            '{"resource":{"resourceType":"Condition","id":"cond-diabetes",'
            '"code":{"coding":[{"system":"http://snomed.info/sct","code":"44054006",'
            '"display":"Diabetes mellitus type 2"}],"text":"Type 2 diabetes mellitus"}}},'
            '{"resource":{"resourceType":"Procedure","id":"proc-foot",'
            '"code":{"coding":[{"system":"http://snomed.info/sct","code":"225358003",'
            '"display":"Foot examination"}],"text":"Diabetic foot examination"}}},'
            '{"resource":{"resourceType":"DiagnosticReport","id":"dr-hba1c",'
            '"code":{"coding":[{"system":"http://loinc.org","code":"58410-2",'
            '"display":"Complete blood count report"}],"text":"Lab report"},'
            '"effectiveDateTime":"2026-01-01T00:10:00",'
            '"conclusion":"HbA1c is elevated and consistent with diabetes."}}'
            ']}'
        ),
        "valid": True,
        "validation_errors": None,
    }

    record = reference_row_to_record(
        row,
        dataset_id="ds-fhir",
        spec=REFERENCE_DATASETS["clinical_notes_to_fhir"],
    )

    assert record.patient.age == 48
    assert record.patient.sex == "female"
    assert record.documents[0].extracted_facts["instruction"] == (
        "Annual check-up with diabetes family history."
    )
    assert record.documents[0].extracted_facts["answer"] == row["fhir_bundle"]
    assert record.documents[0].extracted_facts["source_fields"] == {
        "difficulty": "easy",
        "exampleId": "10004",
        "valid": True,
        "validation_errors": None,
    }
    assert record.documents[0].extracted_facts["lab_values"][0]["name"] == "HbA1c"
    assert record.documents[0].extracted_facts["lab_values"][0]["value"] == 7.4
    assert record.documents[0].extracted_facts["vital_values"][0]["name"] == "Heart rate"
    assert record.documents[0].extracted_facts["vital_values"][0]["value"] == 88.0
    assert record.documents[0].extracted_facts["medications"] == ["Metformin"]
    assert record.documents[0].extracted_facts["medication_details"][0]["route"] == "PO"
    assert record.documents[0].extracted_facts["diagnoses"][0] == {
        "system": "http://snomed.info/sct",
        "code": "44054006",
        "display": "Type 2 diabetes mellitus",
    }
    assert record.documents[0].extracted_facts["procedures"] == [
        "Diabetic foot examination"
    ]
    assert record.documents[0].extracted_facts["procedure_details"][0] == {
        "system": "http://snomed.info/sct",
        "code": "225358003",
        "display": "Diabetic foot examination",
    }
    assert record.modalities == [
        Modality.STRUCTURED_EHR,
        Modality.CLINICAL_TEXT,
        Modality.LABS,
        Modality.VITALS,
    ]
    assert record.labs[0].name == "HbA1c"
    assert record.labs[0].loinc == "4548-4"
    assert record.labs[0].value == 7.4
    assert record.labs[0].reference_low == 4.0
    assert record.labs[0].reference_high == 5.6
    assert record.vitals[0].name == "Heart rate"
    assert record.vitals[0].value == 88
    assert record.medication_history[0].name == "Metformin"
    assert record.medication_history[0].route == "PO"
    assert record.encounters[0].diagnoses[0].code == "44054006"
    assert record.encounters[0].procedures[0].display == "Diabetic foot examination"
    assert record.documents[1].note_type == "diagnostic_report"
    assert record.documents[1].clean_text == (
        "HbA1c is elevated and consistent with diabetes."
    )
    assert record.documents[1].extracted_facts["diagnostic_report_code"] == {
        "system": "http://loinc.org",
        "code": "58410-2",
        "display": "Lab report",
    }
    assert record.topic == "easy"
    assert record.metadata["reference_dataset"] == "ai-galileo/clinical-notes-to-fhir"


def test_structured_reference_fields_map_labs_vitals_meds_and_time_series():
    spec = reference_dataset_spec(
        repo_id="example/structured-icu-reference",
        split="validation",
        license="mit",
        note_field="note",
        task_field="cohort",
        patient_id_field="encounter_id",
        lab_values_field="labs",
        vital_values_field="vitals",
        medications_field="medications",
        time_series_field="time_series",
        description="Structured ICU reference rows for benchmark imports.",
    )
    row = {
        "encounter_id": "icu-1",
        "cohort": "sepsis_icu",
        "note": "Progress note: 67-year-old female treated for septic shock.",
        "labs": [
            {
                "name": "Lactate",
                "loinc": "2524-7",
                "value": 3.8,
                "unit": "mmol/L",
                "reference_low": 0.5,
                "reference_high": 2.0,
                "flag": "H",
                "effective_time": "2026-01-01T01:00:00",
                "specimen": "plasma",
            }
        ],
        "vitals": [
            {
                "name": "MAP",
                "value": 62,
                "unit": "mmHg",
                "effective_time": "2026-01-01T01:05:00",
            }
        ],
        "medications": [
            {
                "name": "Norepinephrine",
                "dose": "0.08 mcg/kg/min",
                "route": "IV",
                "frequency": "continuous",
                "status": "active",
                "start": "2026-01-01T01:10:00",
            }
        ],
        "time_series": [
            {
                "name": "arterial_pressure",
                "unit": "mmHg",
                "generation_backend": "reference:waveform",
                "sampling_rate_hz": 1.0,
                "points": [
                    {
                        "timestamp": "2026-01-01T01:05:00",
                        "values": {"systolic": 82, "diastolic": 48, "mean": 62},
                    }
                ],
            }
        ],
    }

    record = reference_row_to_record(
        row,
        dataset_id="ds-structured",
        spec=spec,
        reference_key="structured_icu",
    )

    assert record.patient.age == 67
    assert record.patient.sex == "female"
    assert record.topic == "sepsis_icu"
    assert record.labs[0].name == "Lactate"
    assert record.labs[0].loinc == "2524-7"
    assert record.vitals[0].name == "MAP"
    assert record.medication_history[0].name == "Norepinephrine"
    assert record.time_series[0].name == "arterial_pressure"
    assert record.time_series[0].points[0].values["mean"] == 62.0
    assert record.documents[0].extracted_facts["lab_values"][0]["specimen"] == "plasma"
    assert record.documents[0].extracted_facts["vital_values"][0]["name"] == "MAP"
    assert record.documents[0].extracted_facts["medication_details"][0]["route"] == "IV"
    assert record.documents[0].extracted_facts["time_series_channels"][0]["name"] == (
        "arterial_pressure"
    )
    assert record.modalities == [
        Modality.STRUCTURED_EHR,
        Modality.CLINICAL_TEXT,
        Modality.LABS,
        Modality.VITALS,
        Modality.TIME_SERIES,
    ]


def test_structured_reference_fields_accept_json_encoded_columns():
    spec = reference_dataset_spec(
        repo_id="example/json-structured-reference",
        split="validation",
        license="mit",
        note_field="note",
        lab_values_field="labs_json",
        vital_values_field="vitals_json",
        medications_field="medications_json",
        time_series_field="signals_json",
    )
    row = {
        "note": "Progress note: 67-year-old female with septic shock.",
        "labs_json": json.dumps(
            [
                {
                    "name": "Creatinine",
                    "value": "2.1",
                    "unit": "mg/dL",
                    "effective_time": "2026-01-01T01:00:00",
                }
            ]
        ),
        "vitals_json": json.dumps(
            [
                {
                    "name": "MAP",
                    "value": "61",
                    "unit": "mmHg",
                    "effective_time": "2026-01-01T01:05:00",
                }
            ]
        ),
        "medications_json": json.dumps(
            [{"name": "Norepinephrine", "status": "active"}]
        ),
        "signals_json": json.dumps(
            [
                {
                    "name": "arterial_pressure",
                    "unit": "mmHg",
                    "points": [
                        {
                            "timestamp": "2026-01-01T01:05:00",
                            "values": {"mean": "61"},
                        }
                    ],
                }
            ]
        ),
    }

    record = reference_row_to_record(
        row,
        dataset_id="ds-json-structured",
        spec=spec,
    )

    assert record.labs[0].name == "Creatinine"
    assert record.labs[0].value == 2.1
    assert record.vitals[0].value == 61
    assert record.medication_history[0].name == "Norepinephrine"
    assert record.time_series[0].points[0].values == {"mean": 61.0}


def test_radiology_consistency_reference_row_maps_image_evidence_to_instruction():
    row = {
        "case_id": "rad-1",
        "modality": "XR",
        "study": "chest radiograph",
        "imaging_findings": "Small left pleural effusion without pneumothorax.",
        "report_excerpt": "The report says no pleural effusion.",
        "consistency_issue": "contradiction",
        "expected_decision": "INCONSISTENT",
        "expected_rationale_bullets": "effusion present|report denies effusion",
    }

    record = reference_row_to_record(
        row,
        dataset_id="ds-rad",
        spec=REFERENCE_DATASETS["radiology_report_consistency"],
    )

    assert record.documents[0].clean_text == "The report says no pleural effusion."
    assert record.documents[0].extracted_facts["instruction"] == (
        "Small left pleural effusion without pneumothorax."
    )
    assert record.documents[0].extracted_facts["answer"] == "INCONSISTENT"
    assert record.documents[0].extracted_facts["source_fields"] == {
        "case_id": "rad-1",
        "consistency_issue": "contradiction",
        "expected_rationale_bullets": (
            "effusion present|report denies effusion"
        ),
        "modality": "XR",
        "study": "chest radiograph",
    }
    assert record.documents[0].note_type == "radiology_report"
    assert record.imaging[0].image_id in (
        record.documents[0].extracted_facts["imaging_asset_ids"]
    )
    assert record.documents[0].extracted_facts["imaging_modalities"] == ["XR"]
    assert record.documents[0].extracted_facts["imaging_body_regions"] == ["chest"]
    assert record.documents[0].extracted_facts["imaging_labels"] == [
        "Pleural effusion",
        "Pneumothorax",
    ]
    assert record.modalities == [Modality.CLINICAL_TEXT, Modality.IMAGING]
    assert record.imaging[0].image_id.startswith("img-")
    assert record.imaging[0].modality == "XR"
    assert record.imaging[0].body_region == "chest"
    assert record.imaging[0].report_text == "The report says no pleural effusion."
    assert {label.display for label in record.imaging[0].labels} == {
        "Pleural effusion",
        "Pneumothorax",
    }
    assert record.topic == "contradiction"
    assert record.metadata["reference_dataset"] == (
        "ClarusC64/image-report-consistency-radiology-v01"
    )


def test_image_reference_row_persists_image_asset(tmp_path):
    class FakeImage:
        def save(self, path):
            path.write_bytes(b"fake-image")

    row = {"image": FakeImage(), "label": 1}

    record = reference_row_to_record(
        row,
        dataset_id="ds-image",
        spec=REFERENCE_DATASETS["synthetic_chest_xray_pneumonia"],
        image_output_dir=tmp_path,
    )

    assert record.modalities == [Modality.CLINICAL_TEXT, Modality.IMAGING]
    assert record.documents[0].clean_text.endswith("labeled pneumonia.")
    assert record.imaging[0].file_path is not None
    assert record.imaging[0].generation_backend == (
        "huggingface-reference:chimbiwide/synthetic-chest-xray-pneumonia"
    )
    assert record.imaging[0].modality == "XR"
    assert record.imaging[0].body_region == "chest"
    assert {label.display for label in record.imaging[0].labels} == {"Pneumonia"}
    assert record.documents[0].extracted_facts["imaging_labels"] == ["Pneumonia"]
    assert Path(record.imaging[0].file_path).read_bytes() == b"fake-image"


def test_image_reference_row_preserves_meaningful_note_text(tmp_path):
    class FakeImage:
        def save(self, path):
            path.write_bytes(b"fake-image")

    spec = reference_dataset_spec(
        repo_id="org/image-caption-reference",
        split="train",
        license="cc-by-4.0",
        note_field="caption",
        image_field="image",
        image_label_field="label",
    )
    row = {
        "image": FakeImage(),
        "label": "pneumonia",
        "caption": "Portable chest radiograph shows right lower lobe pneumonia.",
    }

    record = reference_row_to_record(
        row,
        dataset_id="ds-image",
        spec=spec,
        image_output_dir=tmp_path,
    )

    assert record.documents[0].clean_text == row["caption"]
    assert record.imaging[0].report_text == row["caption"]


def test_import_reference_rows_honors_limit_and_stable_ids():
    rows = [
        {
            "patient_id": "a",
            "note": "Progress Note: 72-year-old female with heart failure.",
            "question": "Extract problems.",
            "answer": "Heart failure.",
            "task": "Extraction",
        },
        {
            "patient_id": "b",
            "note": "Progress Note: 40-year-old male with sepsis.",
            "question": "Extract problems.",
            "answer": "Sepsis.",
            "task": "Extraction",
        },
    ]

    first = import_reference_rows(rows, dataset_id="ds-hf", limit=1)
    second = import_reference_rows(rows, dataset_id="ds-hf", limit=1)

    assert len(first) == 1
    assert first[0].record_id == second[0].record_id
    assert first[0].documents[0].note_type == "progress_note"


def test_reference_row_ids_do_not_depend_on_row_order():
    first_row = {
        "patient_id": "a",
        "note": "Progress Note: 72-year-old female with heart failure.",
        "question": "Extract problems.",
        "answer": "Heart failure.",
        "task": "Extraction",
    }
    second_row = {
        "patient_id": "b",
        "note": "Progress Note: 40-year-old male with sepsis.",
        "question": "Extract problems.",
        "answer": "Sepsis.",
        "task": "Extraction",
    }

    original = import_reference_rows([first_row, second_row], dataset_id="ds-hf")
    reordered = import_reference_rows([second_row, first_row], dataset_id="ds-hf")

    assert {record.record_id for record in original} == {
        record.record_id for record in reordered
    }


def test_reference_row_ids_are_scoped_to_dataset_id():
    row = {
        "patient_id": "a",
        "note": "Progress Note: 72-year-old female with heart failure.",
        "question": "Extract problems.",
        "answer": "Heart failure.",
        "task": "Extraction",
    }

    first = import_reference_rows([row], dataset_id="ds-one")
    second = import_reference_rows([row], dataset_id="ds-two")

    assert first[0].record_id != second[0].record_id
    assert first[0].patient.patient_id != second[0].patient.patient_id


def test_import_reference_rows_accepts_custom_dataset_spec():
    custom_spec = reference_dataset_spec(
        repo_id="org/custom-synthetic-notes",
        split="eval",
        license="cc-by-4.0",
        note_field="clinical_note",
        question_field="prompt",
        answer_field="completion",
        task_field="task_name",
        patient_id_field="subject_id",
        description="Custom synthetic notes fixture.",
    )
    rows = [
        {
            "subject_id": "abc",
            "clinical_note": "Progress Note: 57-year-old female with COPD.",
            "prompt": "Extract diagnosis.",
            "completion": "COPD.",
            "task_name": "extraction",
        }
    ]

    records = import_reference_rows(
        rows,
        dataset_id="ds-custom",
        reference_key="org/custom-synthetic-notes",
        split="eval",
        limit=1,
        spec=custom_spec,
    )

    assert len(records) == 1
    assert records[0].metadata["reference_key"] == "org/custom-synthetic-notes"
    assert records[0].metadata["reference_dataset"] == "org/custom-synthetic-notes"
    assert records[0].metadata["reference_license"] == "cc-by-4.0"
    assert records[0].documents[0].extracted_facts["instruction"] == "Extract diagnosis."


def test_load_reference_dataset_requires_hf_extra_when_datasets_missing(monkeypatch):
    def fake_require_package(import_name: str, extra: str):
        raise RuntimeError(f"Install casecrawler[{extra}] to use this backend.")

    monkeypatch.setattr("casecrawler.integrations.huggingface.require_package", fake_require_package)

    with pytest.raises(RuntimeError, match=r"casecrawler\[hf\]"):
        load_reference_dataset("asclepius")
