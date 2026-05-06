from casecrawler.generation.structured_generator import StructuredGenerator
from casecrawler.generation.text_generator import TextGenerator
from casecrawler.models.dataset import GenerationRequest
from casecrawler.models.synthetic import Modality


def test_text_generator_adds_multiple_clinical_note_types():
    req = GenerationRequest(
        topic="pneumonia",
        modalities=[Modality.CLINICAL_TEXT, Modality.IMAGING],
        cohort_constraints={"base_time": "2026-01-01T00:00:00"},
    )
    record = StructuredGenerator().generate("ds-1", req, 0)

    updated = TextGenerator().add_documents(record)

    note_types = {document.note_type for document in updated.documents}
    assert note_types >= {
        "ed_note",
        "progress_note",
        "nursing_note",
        "discharge_summary",
        "radiology_report",
    }
    assert any("Medication history" in document.clean_text for document in updated.documents)
