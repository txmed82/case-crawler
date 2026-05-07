from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from casecrawler.models.synthetic import ClinicalDocument, SyntheticRecord


class TextGenerator:
    def add_documents(self, record: SyntheticRecord) -> SyntheticRecord:
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
