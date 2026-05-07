import pytest

from casecrawler.integrations.huggingface import (
    REFERENCE_DATASETS,
    import_reference_rows,
    list_reference_datasets,
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


def test_load_reference_dataset_requires_hf_extra_when_datasets_missing(monkeypatch):
    def fake_require_package(import_name: str, extra: str):
        raise RuntimeError(f"Install casecrawler[{extra}] to use this backend.")

    monkeypatch.setattr("casecrawler.integrations.huggingface.require_package", fake_require_package)

    with pytest.raises(RuntimeError, match=r"casecrawler\[hf\]"):
        load_reference_dataset("asclepius")
