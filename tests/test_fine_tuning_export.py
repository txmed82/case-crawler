from casecrawler.export.fine_tuning import (
    export_chat_record,
    export_multimodal_record,
    export_record,
    export_sft_record,
)
from casecrawler.models.synthetic import (
    ClinicalDocument,
    ComplexityProfile,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
)


def test_export_sft_record_contains_messages():
    record = SyntheticRecord(
        record_id="rec-1",
        dataset_id="ds-1",
        topic="sepsis",
        complexity=ComplexityProfile.MODERATE,
        modalities=[Modality.CLINICAL_TEXT],
        patient=SyntheticPatient(patient_id="pat-1", age=64, sex="male"),
        encounters=[],
        documents=[
            ClinicalDocument(
                document_id="doc-1",
                note_type="ed_note",
                author_role="physician",
                timestamp="2026-05-06T10:00:00",
                clean_text="Patient has fever, hypotension, elevated lactate.",
            )
        ],
        provenance=Provenance(
            generator="unit-test",
            created_at="2026-05-06T10:00:00",
        ),
    )

    exported = export_sft_record(record, task="summarize")

    assert exported["record_id"] == "rec-1"
    assert exported["messages"][0]["role"] == "system"
    assert exported["messages"][1]["role"] == "user"
    assert exported["messages"][2]["role"] == "assistant"


def test_export_record_dispatches_chat_and_multimodal():
    record = SyntheticRecord(
        record_id="rec-1",
        dataset_id="ds-1",
        topic="sepsis",
        complexity=ComplexityProfile.MODERATE,
        modalities=[Modality.CLINICAL_TEXT],
        patient=SyntheticPatient(patient_id="pat-1", age=64, sex="male"),
        encounters=[],
        provenance=Provenance(
            generator="unit-test",
            created_at="2026-05-06T10:00:00",
        ),
    )

    chat = export_chat_record(record)
    multimodal = export_multimodal_record(record)

    assert export_record(record, "chat_jsonl") == chat
    assert export_record(record, "multimodal_jsonl") == multimodal
    assert chat["messages"][0]["role"] == "system"
    assert multimodal["images"] == []
    assert multimodal["clinical_context"]["record_id"] == "rec-1"
