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
                _radiology_review_text(record),
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
    messy = _messy_text(note_type, clean_text)
    return ClinicalDocument(
        document_id=f"doc-{uuid5(NAMESPACE_URL, f'{record.record_id}:{note_type}')}",
        note_type=note_type,
        author_role=author_role,
        timestamp=timestamp,
        clean_text=clean_text,
        messy_text=messy,
        extracted_facts=_extracted_facts(record, note_type),
    )


def _messy_text(note_type: str, clean_text: str) -> str:
    shorthand = (
        clean_text.replace("patient", "pt")
        .replace("Patient", "Pt")
        .replace("with", "w/")
        .replace("Initial", "Init")
        .replace("Medication", "Med")
        .replace("medication", "med")
        .replace("Assessment and plan", "A/P")
        .replace("shortness of breath", "SOB")
    )
    if note_type == "ed_note":
        return f"pt msg: {shorthand}"
    if note_type == "progress_note":
        return f"prog note - {shorthand}"
    if note_type == "nursing_note":
        return f"MAR: {shorthand}; I/O ck, fall scrn, meds given per synthetic MAR"
    if note_type == "discharge_summary":
        return f"d/c summ: {shorthand}"
    if note_type == "radiology_report":
        ocr_like = shorthand.replace("Synthetic", "5ynthetic").replace("labels", "1abels")
        return f"OCR: {ocr_like}"
    return shorthand


def _extracted_facts(record: SyntheticRecord, note_type: str) -> dict:
    lab_values = [
        {
            "name": lab.name,
            "value": lab.value,
            "unit": lab.unit,
            "reference_low": lab.reference_low,
            "reference_high": lab.reference_high,
            "flag": lab.flag,
            "effective_time": lab.effective_time,
        }
        for lab in record.labs
    ]
    abnormal_labs = [
        {
            "name": lab.name,
            "value": lab.value,
            "unit": lab.unit,
            "flag": lab.flag,
        }
        for lab in record.labs
        if lab.flag
    ]
    vital_values = [
        {
            "name": vital.name,
            "value": vital.value,
            "unit": vital.unit,
            "effective_time": vital.effective_time,
        }
        for vital in record.vitals
    ]
    abnormal_vitals = [
        {
            "name": vital.name,
            "value": vital.value,
            "unit": vital.unit,
            "direction": direction,
        }
        for vital in record.vitals
        if (direction := _vital_abnormality(vital.name, vital.value)) is not None
    ]
    medication_details = [
        {
            "name": med.name,
            "rxnorm": med.rxnorm,
            "dose": med.dose,
            "route": med.route,
            "frequency": med.frequency,
            "status": med.status,
            "start": med.start,
            "end": med.end,
        }
        for med in record.medication_history
    ]
    facts = {
        "topic": record.topic,
        "note_type": note_type,
        "patient_age": record.patient.age,
        "patient_sex": record.patient.sex,
        "diagnoses": [
            diagnosis.display
            for encounter in record.encounters
            for diagnosis in encounter.diagnoses
        ],
        "lab_names": [lab.name for lab in record.labs],
        "lab_values": lab_values,
        "abnormal_labs": abnormal_labs,
        "vital_names": [vital.name for vital in record.vitals],
        "vital_values": vital_values,
        "abnormal_vitals": abnormal_vitals,
        "medications": [med.name for med in record.medication_history],
        "medication_details": medication_details,
        "time_series_channels": [channel.name for channel in record.time_series],
    }
    if note_type == "radiology_report":
        facts.update(
            {
                "imaging_asset_ids": [asset.image_id for asset in record.imaging],
                "imaging_modalities": [asset.modality for asset in record.imaging],
                "imaging_body_regions": [asset.body_region for asset in record.imaging],
                "imaging_labels": [
                    label.display
                    for asset in record.imaging
                    for label in asset.labels
                ],
            }
        )
    return facts


def _vital_abnormality(name: str, value: float) -> str | None:
    normalized = name.lower().replace("_", " ")
    if normalized in {"hr", "heart rate"}:
        if value > 100:
            return "high"
        if value < 50:
            return "low"
    if normalized in {"sbp", "systolic bp", "systolic blood pressure"}:
        if value < 90:
            return "low"
        if value > 180:
            return "high"
    if normalized in {"spo2", "oxygen saturation"} and value < 94:
        return "low"
    if normalized in {"temperature", "temp"} and value >= 38:
        return "high"
    if normalized in {"respiratory rate", "rr"} and value > 22:
        return "high"
    return None


def _radiology_review_text(record: SyntheticRecord) -> str:
    if not record.imaging:
        return (
            f"Radiology review for {record.topic}. Imaging is synthetic; findings "
            "should be cross-checked against structured labels when present."
        )
    asset_summaries = []
    for asset in record.imaging:
        label_text = ", ".join(label.display for label in asset.labels) or "No labels"
        asset_summaries.append(
            f"{asset.image_id}: {asset.modality} {asset.body_region} with {label_text}. "
            f"Report excerpt: {asset.report_text}"
        )
    return (
        f"Radiology review for {record.topic}. "
        + " ".join(asset_summaries)
        + " Synthetic image-text alignment should be validated before training use."
    )


def _clinical_document_prompt(record: SyntheticRecord) -> str:
    return (
        f"Create synthetic clinical documents for topic: {record.topic}\n"
        f"Record id: {record.record_id}\n"
        f"Patient: {record.patient.model_dump()}\n"
        f"Encounters: {[encounter.model_dump() for encounter in record.encounters]}\n"
        f"Labs: {[lab.model_dump() for lab in record.labs]}\n"
        f"Vitals: {[vital.model_dump() for vital in record.vitals]}\n"
        f"Medication history: {[med.model_dump() for med in record.medication_history]}\n"
        f"Time series: {[channel.model_dump() for channel in record.time_series]}\n"
        f"Imaging: {[asset.model_dump() for asset in record.imaging]}\n"
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
