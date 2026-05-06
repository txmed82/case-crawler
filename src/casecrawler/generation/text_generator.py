from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from casecrawler.models.synthetic import ClinicalDocument, SyntheticRecord


class TextGenerator:
    def add_documents(self, record: SyntheticRecord) -> SyntheticRecord:
        timestamp = datetime.now().isoformat()
        labs = ", ".join(f"{lab.name} {lab.value} {lab.unit}" for lab in record.labs)
        vitals = ", ".join(
            f"{vital.name} {vital.value}{vital.unit}" for vital in record.vitals
        )
        clean = (
            f"{record.patient.age}-year-old {record.patient.sex} patient presents "
            f"with {record.topic}. Initial vitals: {vitals}. Initial labs: {labs}. "
            "Assessment and plan document a synthetic but clinically plausible "
            "presentation."
        )
        messy = clean.replace("patient", "pt").replace("with", "w/").replace(
            "Initial",
            "Init",
        )
        document = ClinicalDocument(
            document_id=f"doc-{uuid4()}",
            note_type="ed_note",
            author_role="physician",
            timestamp=timestamp,
            clean_text=clean,
            messy_text=messy,
        )
        return record.model_copy(update={"documents": [*record.documents, document]})

