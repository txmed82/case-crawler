import pytest

from casecrawler.llm.base import StructuredGenerationResult
from casecrawler.generation.structured_generator import StructuredGenerator
from casecrawler.generation.text_generator import TextGenerator
from casecrawler.models.dataset import GenerationRequest
from casecrawler.models.synthetic import ClinicalDocument, Modality


def test_text_generator_adds_multiple_clinical_note_types():
    req = GenerationRequest(
        topic="pneumonia",
        modalities=[Modality.CLINICAL_TEXT, Modality.IMAGING],
        cohort_constraints={"base_time": "2026-01-01T00:00:00"},
    )
    record = StructuredGenerator().generate("ds-1", req, 0)

    updated = TextGenerator().add_documents(record)

    note_types = {document.note_type for document in updated.documents}
    assert note_types == {
        "ed_note",
        "progress_note",
        "nursing_note",
        "discharge_summary",
        "radiology_report",
    }
    assert len(updated.documents) == 5
    assert any("Medication history" in document.clean_text for document in updated.documents)


@pytest.mark.asyncio
async def test_text_generator_can_use_llm_provider_for_documents():
    req = GenerationRequest(
        topic="pneumonia",
        modalities=[Modality.CLINICAL_TEXT],
        cohort_constraints={"base_time": "2026-01-01T00:00:00"},
    )
    record = StructuredGenerator().generate("ds-1", req, 0)
    provider = FakeTextProvider(
        [
            ClinicalDocument(
                document_id="llm-ed-note",
                note_type="ed_note",
                author_role="physician",
                timestamp="2026-01-01T00:00:00",
                clean_text="LLM drafted ED note with fever, cough, and elevated WBC.",
                messy_text="llm ed note fever cough wbc hi",
                extracted_facts={"source": "fake-provider"},
            )
        ]
    )

    updated = await TextGenerator(provider=provider).add_documents_async(record)

    assert updated.documents[0].document_id == "llm-ed-note"
    assert updated.documents[0].extracted_facts["source"] == "fake-provider"
    assert updated.provenance.model == "fake-clinical-note-model"
    assert provider.prompt
    assert "pneumonia" in provider.prompt


class FakeTextProvider:
    def __init__(self, documents: list[ClinicalDocument]) -> None:
        self.documents = documents
        self.prompt = ""

    async def generate(self, prompt: str, system: str = "", **kwargs):
        raise NotImplementedError

    async def generate_structured(self, prompt: str, schema, system: str = "", **kwargs):
        self.prompt = prompt
        return StructuredGenerationResult(
            data=schema(documents=self.documents),
            input_tokens=10,
            output_tokens=20,
            model="fake-clinical-note-model",
        )
