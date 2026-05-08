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
            '"status":"active","dosage":[{"route":{"text":"PO"},"text":"500 mg twice daily"}]}}'
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
    assert record.topic == "easy"
    assert record.metadata["reference_dataset"] == "ai-galileo/clinical-notes-to-fhir"


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
