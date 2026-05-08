import pytest

from casecrawler.llm.base import StructuredGenerationResult
from casecrawler.generation.structured_generator import StructuredGenerator
from casecrawler.generation.text_generator import TextGenerator
from casecrawler.models.dataset import GenerationRequest
from casecrawler.models.synthetic import (
    AllergyIntolerance,
    ClinicalDocument,
    Code,
    ComplexityProfile,
    ImagingAsset,
    Modality,
)


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
        complexity=ComplexityProfile.COMPLEX,
        modalities=[
            Modality.STRUCTURED_EHR,
            Modality.CLINICAL_TEXT,
            Modality.LABS,
            Modality.VITALS,
        ],
        cohort_constraints={"base_time": "2026-01-01T00:00:00"},
    )
    record = StructuredGenerator().generate("ds-1", req, 0)

    updated = TextGenerator().add_documents(record)
    documents_by_type = {document.note_type: document for document in updated.documents}

    assert "pt msg:" in documents_by_type["ed_note"].messy_text
    assert "MAR:" in documents_by_type["nursing_note"].messy_text
    assert "radiology_report" not in documents_by_type
    assert "Encounter diagnoses:" in documents_by_type["ed_note"].clean_text
    assert "pneumonia" in documents_by_type["ed_note"].clean_text
    assert "Procedures performed or planned:" in documents_by_type["ed_note"].clean_text
    assert "specialty consultation" in documents_by_type["ed_note"].clean_text
    assert "Allergies:" in documents_by_type["ed_note"].clean_text
    assert "Relevant diagnoses:" in documents_by_type[
        "medication_administration_record"
    ].clean_text
    assert "Related procedures:" in documents_by_type[
        "medication_administration_record"
    ].clean_text
    assert documents_by_type["ed_note"].extracted_facts["topic"] == "pneumonia"
    assert "WBC" in documents_by_type["ed_note"].extracted_facts["lab_names"]
    assert "Ceftriaxone" in documents_by_type["ed_note"].extracted_facts["medications"]
    assert "specialty consultation" in documents_by_type["ed_note"].extracted_facts[
        "procedures"
    ]
    assert documents_by_type["ed_note"].extracted_facts["procedure_details"][0] == {
        "encounter_id": record.encounters[0].encounter_id,
        "system": "synthetic",
        "code": "specialty_consultation",
        "display": "specialty consultation",
    }
    assert documents_by_type["ed_note"].extracted_facts["lab_values"][0]["name"] == "WBC"
    assert documents_by_type["ed_note"].extracted_facts["vital_values"][0]["name"] == "HR"
    assert any(
        vital["name"] == "SpO2"
        for vital in documents_by_type["ed_note"].extracted_facts["abnormal_vitals"]
    )
    assert documents_by_type["ed_note"].extracted_facts["medication_details"][0][
        "route"
    ] == "IV"
    assert documents_by_type["ed_note"].extracted_facts["allergies"] == []
    assert documents_by_type["ed_note"].extracted_facts["messy_text_profile"] == "standard"


def test_text_generator_includes_allergy_details_in_notes_and_facts():
    req = GenerationRequest(
        topic="sepsis",
        modalities=[Modality.STRUCTURED_EHR, Modality.CLINICAL_TEXT],
        cohort_constraints={"base_time": "2026-01-01T00:00:00"},
    )
    record = StructuredGenerator().generate("ds-1", req, 0).model_copy(
        update={
            "allergies": [
                AllergyIntolerance(
                    substance="Penicillin",
                    code="7980",
                    system="RxNorm",
                    reaction="hives",
                    severity="moderate",
                    recorded_at="2026-01-01",
                )
            ]
        }
    )

    updated = TextGenerator().add_documents(record)
    ed_note = next(document for document in updated.documents if document.note_type == "ed_note")

    assert "Allergies: Penicillin (hives)." in ed_note.clean_text
    assert ed_note.extracted_facts["allergies"] == ["Penicillin"]
    assert ed_note.extracted_facts["allergy_details"][0]["severity"] == "moderate"


def test_text_generator_supports_message_ocr_and_heavy_noise_profiles():
    req = GenerationRequest(
        topic="pneumonia",
        modalities=[
            Modality.STRUCTURED_EHR,
            Modality.CLINICAL_TEXT,
            Modality.LABS,
            Modality.VITALS,
            Modality.IMAGING,
        ],
        cohort_constraints={"base_time": "2026-01-01T00:00:00"},
    )
    record = StructuredGenerator().generate("ds-1", req, 0).model_copy(
        update={
            "imaging": [
                ImagingAsset(
                    image_id="img-pneumonia",
                    modality="XR",
                    body_region="chest",
                    prompt="Chest radiograph right lower lobe opacity",
                    report_text="Synthetic chest radiology report with opacity.",
                    labels=[
                        Code(
                            system="synthetic",
                            code="opacity",
                            display="Opacity",
                        )
                    ],
                    generation_backend="placeholder",
                )
            ]
        }
    )

    message_docs = {
        document.note_type: document
        for document in TextGenerator(noise_profile="message").add_documents(record).documents
    }
    ocr_docs = {
        document.note_type: document
        for document in TextGenerator(noise_profile="ocr").add_documents(record).documents
    }
    heavy_docs = {
        document.note_type: document
        for document in TextGenerator(noise_profile="heavy").add_documents(record).documents
    }

    assert message_docs["nursing_note"].messy_text.startswith("rn handoff:")
    assert "pt presents w/" in message_docs["ed_note"].messy_text
    assert ocr_docs["ed_note"].messy_text.startswith("OCR ED_NOTE:")
    assert "5ynthetic" in ocr_docs["radiology_report"].messy_text
    assert heavy_docs["lab_report"].messy_text.startswith("OCR LAB_REPORT:")
    assert "synth" in heavy_docs["ed_note"].messy_text
    assert heavy_docs["ed_note"].extracted_facts["messy_text_profile"] == "heavy"


def test_text_generator_rejects_unknown_noise_profile():
    with pytest.raises(ValueError, match="noise_profile must be one of"):
        TextGenerator(noise_profile="chaos")


def test_text_generator_adds_follow_up_notes_for_longitudinal_encounters():
    req = GenerationRequest(
        topic="sepsis",
        complexity=ComplexityProfile.COMPLEX,
        modalities=[
            Modality.STRUCTURED_EHR,
            Modality.CLINICAL_TEXT,
            Modality.LABS,
            Modality.VITALS,
        ],
        cohort_constraints={
            "base_time": "2026-01-01T00:00:00",
            "encounter_count": 3,
        },
    )
    record = StructuredGenerator().generate("ds-1", req, 0)

    updated = TextGenerator().add_documents(record)
    follow_ups = [
        document
        for document in updated.documents
        if document.extracted_facts.get("encounter_index") in {2, 3}
    ]

    assert len(follow_ups) == 4
    assert {document.note_type for document in follow_ups} == {
        "progress_note",
        "nursing_note",
    }
    assert len({document.document_id for document in updated.documents}) == len(
        updated.documents
    )
    assert {
        document.extracted_facts["encounter_id"] for document in follow_ups
    } == {record.encounters[1].encounter_id, record.encounters[2].encounter_id}
    assert any("Follow-up progress note for encounter 2" in doc.clean_text for doc in follow_ups)
    assert any("Follow-up nursing note for encounter 3" in doc.clean_text for doc in follow_ups)


def test_text_generator_adds_radiology_report_when_imaging_modality_is_requested():
    req = GenerationRequest(
        topic="pneumonia",
        modalities=[Modality.CLINICAL_TEXT, Modality.IMAGING],
        cohort_constraints={"base_time": "2026-01-01T00:00:00"},
    )
    record = StructuredGenerator().generate("ds-1", req, 0)

    updated = TextGenerator().add_documents(record)
    documents_by_type = {document.note_type: document for document in updated.documents}

    assert "OCR:" in documents_by_type["radiology_report"].messy_text
    assert "Radiology review for pneumonia" in documents_by_type["radiology_report"].clean_text


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


def test_text_generator_can_use_external_backend_for_documents():
    req = GenerationRequest(
        topic="pneumonia",
        modalities=[Modality.CLINICAL_TEXT],
        cohort_constraints={"base_time": "2026-01-01T00:00:00"},
    )
    record = StructuredGenerator().generate("ds-1", req, 0)
    calls = []

    def fake_runner(command, payload):
        calls.append((command, payload))
        return (
            '{"documents":[{"note_type":"ed_note","author_role":"external",'
            '"clean_text":"External synthetic ED note for pneumonia.",'
            '"messy_text":"ext ed note pna",'
            '"extracted_facts":{"source":"fake-external"}}]}'
        )

    updated = TextGenerator(
        external_command=["hf-note-sample"],
        external_runner=fake_runner,
    ).add_documents(record)

    assert calls[0][0] == ["hf-note-sample"]
    assert '"topic": "pneumonia"' in calls[0][1]
    assert updated.documents[0].note_type == "ed_note"
    assert updated.documents[0].timestamp == "2026-01-01T00:00:00"
    assert updated.documents[0].extracted_facts["source"] == "fake-external"
    assert (
        updated.documents[0].extracted_facts["generation_backend"]
        == "external:hf-note-sample"
    )


def test_text_generator_rejects_empty_external_command():
    with pytest.raises(ValueError, match="external_command must not be empty"):
        TextGenerator(external_command=[])


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
    assert "Medication history:" in provider.prompt
    assert "Time series:" in provider.prompt
    assert "Imaging:" in provider.prompt


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
