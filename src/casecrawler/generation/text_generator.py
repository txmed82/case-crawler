from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, Field

from casecrawler.llm.base import BaseLLMProvider
from casecrawler.models.synthetic import ClinicalDocument, SyntheticRecord


class ClinicalDocumentBatch(BaseModel):
    documents: list[ClinicalDocument] = Field(default_factory=list)


class TextGenerator:
    def __init__(self, provider: BaseLLMProvider | None = None) -> None:
        self._provider = provider

    def add_documents(self, record: SyntheticRecord) -> SyntheticRecord:
        if self._provider is not None:
            raise RuntimeError(
                "TextGenerator with an LLM provider must be used via "
                "add_documents_async."
            )
        return self._add_deterministic_documents(record)

    async def add_documents_async(self, record: SyntheticRecord) -> SyntheticRecord:
        if self._provider is None:
            return self._add_deterministic_documents(record)

        result = await self._provider.generate_structured(
            prompt=_clinical_document_prompt(record),
            schema=ClinicalDocumentBatch,
            system=(
                "You generate synthetic clinical documentation for healthcare AI "
                "training datasets. Use only the supplied synthetic facts. Return "
                "clean text, messy text variants, and extracted facts. Do not add "
                "real patient identifiers."
            ),
        )
        documents = _normalize_llm_documents(record, result.data.documents)
        provenance = record.provenance.model_copy(update={"model": result.model})
        return record.model_copy(
            update={
                "documents": [*record.documents, *documents],
                "provenance": provenance,
            }
        )

    def _add_deterministic_documents(self, record: SyntheticRecord) -> SyntheticRecord:
        timestamp = record.provenance.created_at
        labs = ", ".join(f"{lab.name} {lab.value} {lab.unit}" for lab in record.labs)
        vitals = ", ".join(
            f"{vital.name} {vital.value}{vital.unit}" for vital in record.vitals
        )
        medications = ", ".join(
            f"{med.name} {med.dose or ''} {med.route or ''} {med.frequency or ''}".strip()
            for med in record.medication_history
        )
        ed_note = (
            f"{record.patient.age}-year-old {record.patient.sex} patient presents "
            f"with {record.topic}. Initial vitals: {vitals}. Initial labs: {labs}. "
            f"Medication history: {medications or 'none documented'}. "
            "Assessment and plan document a synthetic but clinically plausible "
            "presentation."
        )
        documents = [
            _document(record, "ed_note", "physician", timestamp, ed_note),
            _document(
                record,
                "progress_note",
                "physician",
                timestamp,
                (
                    f"Synthetic progress note for {record.topic}. Vitals trend is "
                    f"reviewed with current values: {vitals}. Abnormal labs include {labs}."
                ),
            ),
            _document(
                record,
                "nursing_note",
                "nurse",
                timestamp,
                (
                    f"Patient monitored for {record.topic}. Nursing assessment notes "
                    "fall risk screening, intake/output review, medication administration, "
                    f"and response to active medications: {medications or 'none documented'}."
                ),
            ),
            _document(
                record,
                "discharge_summary",
                "physician",
                timestamp,
                (
                    f"Discharge summary for synthetic admission related to {record.topic}. "
                    "Hospital course summarizes presenting symptoms, diagnostic results, "
                    "treatments, medication reconciliation, and follow-up needs."
                ),
            ),
            _document(
                record,
                "radiology_report",
                "radiologist",
                timestamp,
                (
                    f"Radiology review for {record.topic}. Portable chest imaging is "
                    "synthetic; findings should be cross-checked against structured labels."
                ),
            ),
        ]
        return record.model_copy(update={"documents": [*record.documents, *documents]})


def _document(
    record: SyntheticRecord,
    note_type: str,
    author_role: str,
    timestamp: str,
    clean_text: str,
) -> ClinicalDocument:
    messy = clean_text.replace("patient", "pt").replace("with", "w/").replace(
        "Initial",
        "Init",
    )
    return ClinicalDocument(
        document_id=f"doc-{uuid5(NAMESPACE_URL, f'{record.record_id}:{note_type}')}",
        note_type=note_type,
        author_role=author_role,
        timestamp=timestamp,
        clean_text=clean_text,
        messy_text=messy,
    )


def _clinical_document_prompt(record: SyntheticRecord) -> str:
    return (
        f"Create synthetic clinical documents for topic: {record.topic}\n"
        f"Record id: {record.record_id}\n"
        f"Patient: {record.patient.model_dump()}\n"
        f"Encounters: {[encounter.model_dump() for encounter in record.encounters]}\n"
        f"Labs: {[lab.model_dump() for lab in record.labs]}\n"
        f"Vitals: {[vital.model_dump() for vital in record.vitals]}\n"
        "Required note types: ed_note, progress_note, nursing_note, "
        "discharge_summary, radiology_report when relevant.\n"
        "Each document must be synthetic, internally consistent with the structured "
        "facts, and include a messy_text variant with common clinical shorthand or "
        "message/OCR-style noise."
    )


def _normalize_llm_documents(
    record: SyntheticRecord,
    documents: list[ClinicalDocument],
) -> list[ClinicalDocument]:
    normalized = []
    for index, document in enumerate(documents):
        document_id = document.document_id or (
            f"doc-{uuid5(NAMESPACE_URL, f'{record.record_id}:llm:{index}')}"
        )
        timestamp = document.timestamp or record.provenance.created_at
        normalized.append(
            document.model_copy(
                update={
                    "document_id": document_id,
                    "timestamp": timestamp,
                }
            )
        )
    return normalized
