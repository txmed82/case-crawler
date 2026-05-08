from __future__ import annotations

import json
import subprocess
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, Field

from casecrawler.llm.base import BaseLLMProvider
from casecrawler.models.synthetic import ClinicalDocument, Modality, SyntheticRecord


EXTERNAL_CLINICAL_TEXT_TIMEOUT_SECONDS = 120.0


class ExternalClinicalTextRunner(Protocol):
    def __call__(self, command: list[str], payload: str) -> str: ...


class ClinicalDocumentBatch(BaseModel):
    documents: list[ClinicalDocument] = Field(default_factory=list)


class TextGenerator:
    def __init__(
        self,
        provider: BaseLLMProvider | None = None,
        external_command: list[str] | None = None,
        external_runner: ExternalClinicalTextRunner | None = None,
        noise_profile: str = "standard",
    ) -> None:
        if provider is not None and external_command is not None:
            raise ValueError("TextGenerator cannot use both provider and external_command.")
        if external_command is not None and not external_command:
            raise ValueError("external_command must not be empty when provided.")
        if noise_profile not in {"standard", "message", "ocr", "heavy"}:
            raise ValueError(
                "noise_profile must be one of: standard, message, ocr, heavy."
            )
        self._provider = provider
        self._external_command = external_command
        self._external_runner = external_runner or _run_external_command
        self._noise_profile = noise_profile

    def add_documents(self, record: SyntheticRecord) -> SyntheticRecord:
        if self._provider is not None:
            raise RuntimeError(
                "TextGenerator with an LLM provider must be used via "
                "add_documents_async."
            )
        if self._external_command is not None:
            return self._add_external_documents(record)
        return self._add_deterministic_documents(record)

    async def add_documents_async(self, record: SyntheticRecord) -> SyntheticRecord:
        if self._external_command is not None:
            return self._add_external_documents(record)
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

    def _add_external_documents(self, record: SyntheticRecord) -> SyntheticRecord:
        assert self._external_command is not None
        payload = json.dumps({"record": record.model_dump()}, sort_keys=True)
        output = self._external_runner(self._external_command, payload)
        try:
            raw_documents = json.loads(output)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "External clinical text backend returned invalid JSON."
            ) from exc
        if isinstance(raw_documents, dict):
            raw_documents = raw_documents.get("documents")
        if not isinstance(raw_documents, list):
            raise RuntimeError(
                "External clinical text backend must return a JSON list or "
                "an object with a documents list."
            )
        backend = f"external:{' '.join(self._external_command)}"
        documents = [
            _external_document(record, document, index=index, backend=backend)
            for index, document in enumerate(raw_documents)
        ]
        return record.model_copy(
            update={"documents": [*record.documents, *documents]}
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
        diagnoses = ", ".join(
            diagnosis.display
            for encounter in record.encounters
            for diagnosis in encounter.diagnoses
        )
        procedures = ", ".join(
            procedure.display
            for encounter in record.encounters
            for procedure in encounter.procedures
        )
        ed_note = (
            f"{record.patient.age}-year-old {record.patient.sex} patient presents "
            f"with {record.topic}. Initial vitals: {vitals}. Initial labs: {labs}. "
            f"Encounter diagnoses: {diagnoses or 'none documented'}. "
            f"Procedures performed or planned: {procedures or 'none documented'}. "
            f"Medication history: {medications or 'none documented'}. "
            "Assessment and plan document a synthetic but clinically plausible "
            "presentation."
        )
        documents = [
            _document(
                record,
                "ed_note",
                "physician",
                timestamp,
                ed_note,
                noise_profile=self._noise_profile,
            ),
            _document(
                record,
                "progress_note",
                "physician",
                timestamp,
                (
                    f"Synthetic progress note for {record.topic}. Vitals trend is "
                    f"reviewed with current values: {vitals}. Abnormal labs include {labs}. "
                    f"Active diagnoses include {diagnoses or 'none documented'}, with "
                    f"procedures tracked as {procedures or 'none documented'}."
                ),
                noise_profile=self._noise_profile,
            ),
            _document(
                record,
                "nursing_note",
                "nurse",
                timestamp,
                (
                    f"Patient monitored for {record.topic}. Nursing assessment notes "
                    "fall risk screening, intake/output review, medication administration, "
                    f"diagnosis awareness ({diagnoses or 'none documented'}), "
                    f"and response to active medications: {medications or 'none documented'}."
                ),
                noise_profile=self._noise_profile,
            ),
            *(
                [
                    _document(
                        record,
                        "lab_report",
                        "laboratory",
                        timestamp,
                        (
                            f"Laboratory report for {record.topic}. Reported results: "
                            f"{labs}. Encounter diagnoses: {diagnoses or 'none documented'}. "
                            "Abnormal flags and reference ranges are preserved "
                            "for synthetic training and validation workflows."
                        ),
                        noise_profile=self._noise_profile,
                    )
                ]
                if record.labs
                else []
            ),
            *(
                [
                    _document(
                        record,
                        "vital_signs_flowsheet",
                        "nurse",
                        timestamp,
                        (
                            f"Vital signs flowsheet for {record.topic}. Recorded values: "
                            f"{vitals}. Encounter diagnoses: {diagnoses or 'none documented'}. "
                            "Values are time-stamped synthetic observations "
                            "for trend review."
                        ),
                        noise_profile=self._noise_profile,
                    )
                ]
                if record.vitals
                else []
            ),
            *(
                [
                    _document(
                        record,
                        "medication_administration_record",
                        "pharmacist",
                        timestamp,
                        (
                            f"Medication administration record for {record.topic}. "
                            f"Active and historical medications: {medications}. "
                            f"Relevant diagnoses: {diagnoses or 'none documented'}. "
                            f"Related procedures: {procedures or 'none documented'}. "
                            "Doses, routes, frequencies, and statuses are synthetic "
                            "and reconciled to the structured medication history."
                        ),
                        noise_profile=self._noise_profile,
                    )
                ]
                if record.medication_history
                else []
            ),
            _document(
                record,
                "discharge_summary",
                "physician",
                timestamp,
                (
                    f"Discharge summary for synthetic admission related to {record.topic}. "
                    "Hospital course summarizes presenting symptoms, diagnostic results, "
                    f"encounter diagnoses ({diagnoses or 'none documented'}), "
                    f"procedures ({procedures or 'none documented'}), treatments, "
                    "medication reconciliation, and follow-up needs."
                ),
                noise_profile=self._noise_profile,
            ),
            *_longitudinal_follow_up_documents(record, noise_profile=self._noise_profile),
            *(
                [
                    _document(
                        record,
                        "radiology_report",
                        "radiologist",
                        timestamp,
                        _radiology_review_text(record),
                        noise_profile=self._noise_profile,
                    )
                ]
                if _needs_radiology_report(record)
                else []
            ),
        ]
        return record.model_copy(update={"documents": [*record.documents, *documents]})


def _document(
    record: SyntheticRecord,
    note_type: str,
    author_role: str,
    timestamp: str,
    clean_text: str,
    *,
    document_key: str | None = None,
    extracted_fact_updates: dict | None = None,
    noise_profile: str = "standard",
) -> ClinicalDocument:
    messy = _messy_text(note_type, clean_text, profile=noise_profile)
    extracted_facts = _extracted_facts(record, note_type)
    extracted_facts["messy_text_profile"] = noise_profile
    if extracted_fact_updates:
        extracted_facts.update(extracted_fact_updates)
    return ClinicalDocument(
        document_id=(
            f"doc-{uuid5(NAMESPACE_URL, f'{record.record_id}:{document_key or note_type}')}"
        ),
        note_type=note_type,
        author_role=author_role,
        timestamp=timestamp,
        clean_text=clean_text,
        messy_text=messy,
        extracted_facts=extracted_facts,
    )


def _longitudinal_follow_up_documents(
    record: SyntheticRecord,
    *,
    noise_profile: str = "standard",
) -> list[ClinicalDocument]:
    if len(record.encounters) < 2:
        return []
    documents = []
    for index, encounter in enumerate(record.encounters[1:], start=2):
        diagnoses = ", ".join(diagnosis.display for diagnosis in encounter.diagnoses)
        procedures = ", ".join(procedure.display for procedure in encounter.procedures)
        facts = {
            "encounter_id": encounter.encounter_id,
            "encounter_index": index,
            "encounter_start": encounter.start,
            "encounter_end": encounter.end,
            "encounter_setting": encounter.setting,
            "encounter_reason": encounter.reason,
            "encounter_diagnoses": [
                diagnosis.display for diagnosis in encounter.diagnoses
            ],
            "encounter_procedures": [
                procedure.display for procedure in encounter.procedures
            ],
        }
        documents.append(
            _document(
                record,
                "progress_note",
                "physician",
                encounter.start,
                (
                    f"Follow-up progress note for encounter {index} in "
                    f"{encounter.setting}. Reason: {encounter.reason}. "
                    f"Diagnoses: {diagnoses or 'none documented'}. "
                    f"Procedures: {procedures or 'none documented'}. "
                    "Synthetic longitudinal reassessment compares current "
                    "structured observations with the prior encounter timeline."
                ),
                document_key=f"progress_note:{encounter.encounter_id}",
                extracted_fact_updates=facts,
                noise_profile=noise_profile,
            )
        )
        documents.append(
            _document(
                record,
                "nursing_note",
                "nurse",
                encounter.start,
                (
                    f"Follow-up nursing note for encounter {index} in "
                    f"{encounter.setting}. Reason: {encounter.reason}. "
                    "Nursing reassessment documents safety checks, symptom "
                    "trajectory, intake/output review, and medication response "
                    "for the synthetic longitudinal record."
                ),
                document_key=f"nursing_note:{encounter.encounter_id}",
                extracted_fact_updates=facts,
                noise_profile=noise_profile,
            )
        )
    return documents


def _messy_text(note_type: str, clean_text: str, *, profile: str = "standard") -> str:
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
    if profile == "message":
        return _message_style_text(note_type, shorthand)
    if profile == "ocr":
        return _ocr_style_text(note_type, shorthand)
    if profile == "heavy":
        return _heavy_messy_text(note_type, shorthand)
    if note_type == "ed_note":
        return f"pt msg: {shorthand}"
    if note_type == "progress_note":
        return f"prog note - {shorthand}"
    if note_type == "nursing_note":
        return f"MAR: {shorthand}; I/O ck, fall scrn, meds given per synthetic MAR"
    if note_type == "lab_report":
        return f"lab rpt: {shorthand}"
    if note_type == "vital_signs_flowsheet":
        return f"VS flowsheet: {shorthand}"
    if note_type == "medication_administration_record":
        return f"MAR: {shorthand}"
    if note_type == "discharge_summary":
        return f"d/c summ: {shorthand}"
    if note_type == "radiology_report":
        ocr_like = shorthand.replace("Synthetic", "5ynthetic").replace("labels", "1abels")
        return f"OCR: {ocr_like}"
    return shorthand


def _message_style_text(note_type: str, shorthand: str) -> str:
    text = (
        shorthand.replace("diagnoses", "dx")
        .replace("diagnosis", "dx")
        .replace("Procedures", "Proc")
        .replace("procedures", "proc")
        .replace("review", "rvw")
        .replace("document", "doc")
        .replace("current", "cur")
    )
    text = " ".join(text.split())
    return f"{_message_prefix(note_type)} {text[:420]}"


def _ocr_style_text(note_type: str, shorthand: str) -> str:
    text = (
        shorthand.replace("Synthetic", "5ynthetic")
        .replace("synthetic", "5ynthetic")
        .replace("labels", "1abels")
        .replace("labs", "1abs")
        .replace("Initial", "Initia1")
        .replace("Medication", "Medicat1on")
        .replace("history", "h1story")
    )
    chunks = [text[index : index + 72] for index in range(0, len(text), 72)]
    return f"OCR {note_type.upper()}:\n" + "\n".join(chunks)


def _heavy_messy_text(note_type: str, shorthand: str) -> str:
    text = _message_style_text(note_type, shorthand)
    text = text.replace(" and ", " + ").replace(" or ", " / ")
    text = text.replace("documented", "doc'd").replace("follow-up", "f/u")
    text = text.replace("synthetic", "synth").replace("Synthetic", "Synth")
    if note_type in {"radiology_report", "lab_report"}:
        text = _ocr_style_text(note_type, text)
    return text


def _message_prefix(note_type: str) -> str:
    prefixes = {
        "ed_note": "triage msg:",
        "progress_note": "prog:",
        "nursing_note": "rn handoff:",
        "lab_report": "lab inbox:",
        "vital_signs_flowsheet": "vs flowsheet:",
        "medication_administration_record": "mar:",
        "discharge_summary": "dc msg:",
        "radiology_report": "rad prelim:",
    }
    return prefixes.get(note_type, "clinical msg:")


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
    procedure_details = [
        {
            "encounter_id": encounter.encounter_id,
            "system": procedure.system,
            "code": procedure.code,
            "display": procedure.display,
        }
        for encounter in record.encounters
        for procedure in encounter.procedures
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
        "procedures": [procedure["display"] for procedure in procedure_details],
        "procedure_details": procedure_details,
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


def _needs_radiology_report(record: SyntheticRecord) -> bool:
    return bool(record.imaging) or Modality.IMAGING in record.modalities


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
        "lab_report when labs are present, vital_signs_flowsheet when vitals "
        "are present, medication_administration_record when medication history "
        "is present, discharge_summary, and radiology_report when relevant.\n"
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


def _external_document(
    record: SyntheticRecord,
    document: object,
    *,
    index: int,
    backend: str,
) -> ClinicalDocument:
    if not isinstance(document, dict):
        document = ClinicalDocument.model_validate(document).model_dump()
    payload = {
        "document_id": f"doc-{uuid5(NAMESPACE_URL, f'{record.record_id}:external:{index}')}",
        "note_type": "clinical_note",
        "author_role": "external_generator",
        "timestamp": record.provenance.created_at,
        "clean_text": "",
        "messy_text": None,
        "extracted_facts": {},
        **document,
    }
    extracted_facts = payload.get("extracted_facts") or {}
    payload["extracted_facts"] = {
        **extracted_facts,
        "generation_backend": backend,
    }
    return ClinicalDocument.model_validate(payload)


def _run_external_command(command: list[str], payload: str) -> str:
    try:
        result = subprocess.run(
            command,
            input=payload,
            capture_output=True,
            check=True,
            text=True,
            timeout=EXTERNAL_CLINICAL_TEXT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "External clinical text backend timed out after "
            f"{EXTERNAL_CLINICAL_TEXT_TIMEOUT_SECONDS:.0f}s: {command!r}."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "External clinical text backend failed with exit code "
            f"{exc.returncode}: {command!r}. stdout={exc.stdout!r} stderr={exc.stderr!r}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"External clinical text backend could not be executed: {command!r}."
        ) from exc
    return result.stdout
