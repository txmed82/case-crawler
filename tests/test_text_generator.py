import pytest

from casecrawler.llm.base import StructuredGenerationResult
from casecrawler.generation.structured_generator import StructuredGenerator
from casecrawler.generation.text_generator import TextGenerator
from casecrawler.models.dataset import GenerationRequest
from casecrawler.models.synthetic import ClinicalDocument, Code, ImagingAsset, Modality


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


def test_text_generator_adds_messy_variants_and_extracted_facts():
    req = GenerationRequest(
        topic="pneumonia",
        modalities=[Modality.CLINICAL_TEXT],
        cohort_constraints={"base_time": "2026-01-01T00:00:00"},
    )
    record = StructuredGenerator().generate("ds-1", req, 0)

    updated = TextGenerator().add_documents(record)
    documents_by_type = {document.note_type: document for document in updated.documents}

    assert "pt msg:" in documents_by_type["ed_note"].messy_text
    assert "OCR:" in documents_by_type["radiology_report"].messy_text
    assert "MAR:" in documents_by_type["nursing_note"].messy_text
    assert documents_by_type["ed_note"].extracted_facts["topic"] == "pneumonia"
    assert "WBC" in documents_by_type["ed_note"].extracted_facts["lab_names"]
    assert "Ceftriaxone" in documents_by_type["ed_note"].extracted_facts["medications"]
    assert documents_by_type["ed_note"].extracted_facts["lab_values"][0]["name"] == "WBC"
    assert documents_by_type["ed_note"].extracted_facts["vital_values"][0]["name"] == "HR"
    assert any(
        vital["name"] == "SpO2"
        for vital in documents_by_type["ed_note"].extracted_facts["abnormal_vitals"]
    )
    assert documents_by_type["ed_note"].extracted_facts["medication_details"][0][
        "route"
    ] == "IV"


def test_text_generator_radiology_report_reflects_imaging_assets():
    req = GenerationRequest(
        topic="appendicitis",
        modalities=[Modality.CLINICAL_TEXT, Modality.IMAGING],
        cohort_constraints={"base_time": "2026-01-01T00:00:00"},
    )
    record = StructuredGenerator().generate("ds-1", req, 0).model_copy(
        update={
            "imaging": [
                ImagingAsset(
                    image_id="img-appendicitis",
                    modality="CT",
                    body_region="abdomen",
                    prompt="CT abdomen dilated appendix fat stranding",
                    report_text="Synthetic CT abdomen report. Impression: Appendicitis.",
                    labels=[
                        Code(
                            system="synthetic",
                            code="appendicitis",
                            display="Appendicitis",
                        )
                    ],
                    generation_backend="placeholder",
                )
            ]
        }
    )

    updated = TextGenerator().add_documents(record)
    radiology_report = next(
        document for document in updated.documents if document.note_type == "radiology_report"
    )

    assert "CT abdomen" in radiology_report.clean_text
    assert "Appendicitis" in radiology_report.clean_text
    assert "img-appendicitis" in radiology_report.extracted_facts["imaging_asset_ids"]
    assert "Appendicitis" in radiology_report.extracted_facts["imaging_labels"]


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
