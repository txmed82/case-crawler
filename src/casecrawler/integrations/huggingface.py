from __future__ import annotations

import importlib
import re
from hashlib import sha256
from dataclasses import dataclass
from typing import Iterable
from uuid import NAMESPACE_URL, uuid5

from casecrawler.models.synthetic import (
    ClinicalDocument,
    ComplexityProfile,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
)


def require_package(import_name: str, extra: str):
    try:
        return importlib.import_module(import_name)
    except ModuleNotFoundError as exc:
        if exc.name != import_name:
            raise
        raise RuntimeError(f"Install casecrawler[{extra}] to use this backend.") from exc


@dataclass(frozen=True)
class HuggingFaceReferenceDataset:
    repo_id: str
    split: str
    license: str
    note_field: str
    question_field: str | None = None
    answer_field: str | None = None
    task_field: str | None = None
    patient_id_field: str | None = None
    description: str = ""


REFERENCE_DATASETS: dict[str, HuggingFaceReferenceDataset] = {
    "asclepius": HuggingFaceReferenceDataset(
        repo_id="starmpcc/Asclepius-Synthetic-Clinical-Notes",
        split="train",
        license="cc-by-nc-sa-4.0",
        note_field="note",
        question_field="question",
        answer_field="answer",
        task_field="task",
        patient_id_field="patient_id",
        description="Synthetic discharge summaries with instruction-answer pairs.",
    ),
    "synthclinicalnotes": HuggingFaceReferenceDataset(
        repo_id="IntelLabs/SynthClinicalNotes",
        split="test",
        license="Intel OBL Internal R&D Use License Agreement",
        note_field="ground_truth",
        question_field="model_input",
        task_field=None,
        patient_id_field=None,
        description="Fully synthetic inpatient trajectories for progress note generation benchmarking.",
    ),
}


def list_reference_datasets() -> list[HuggingFaceReferenceDataset]:
    return list(REFERENCE_DATASETS.values())


def load_reference_dataset(
    key: str,
    *,
    split: str | None = None,
    streaming: bool = True,
):
    datasets = require_package("datasets", "hf")
    spec = REFERENCE_DATASETS[key]
    return datasets.load_dataset(
        spec.repo_id,
        split=split or spec.split,
        streaming=streaming,
    )


def load_huggingface_dataset(
    repo_id: str,
    *,
    split: str,
    streaming: bool = True,
):
    datasets = require_package("datasets", "hf")
    return datasets.load_dataset(repo_id, split=split, streaming=streaming)


def reference_dataset_spec(
    *,
    repo_id: str,
    split: str,
    license: str,
    note_field: str,
    question_field: str | None = None,
    answer_field: str | None = None,
    task_field: str | None = None,
    patient_id_field: str | None = None,
    description: str = "",
) -> HuggingFaceReferenceDataset:
    return HuggingFaceReferenceDataset(
        repo_id=repo_id,
        split=split,
        license=license,
        note_field=note_field,
        question_field=question_field,
        answer_field=answer_field,
        task_field=task_field,
        patient_id_field=patient_id_field,
        description=description,
    )


def import_reference_rows(
    rows: Iterable[dict],
    *,
    dataset_id: str,
    reference_key: str = "asclepius",
    split: str | None = None,
    limit: int | None = None,
    spec: HuggingFaceReferenceDataset | None = None,
) -> list[SyntheticRecord]:
    resolved_spec = spec or REFERENCE_DATASETS[reference_key]
    effective_split = split or resolved_spec.split
    records: list[SyntheticRecord] = []
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            break
        records.append(
            reference_row_to_record(
                row,
                dataset_id=dataset_id,
                spec=resolved_spec,
                split=effective_split,
            )
        )
    return records


def reference_row_to_record(
    row: dict,
    *,
    dataset_id: str,
    spec: HuggingFaceReferenceDataset,
    split: str | None = None,
) -> SyntheticRecord:
    effective_split = split or spec.split
    note = _coerce_text(row.get(spec.note_field))
    question = _coerce_text(row.get(spec.question_field)) if spec.question_field else ""
    answer = _coerce_text(row.get(spec.answer_field)) if spec.answer_field else ""
    task = _coerce_text(row.get(spec.task_field)) if spec.task_field else "clinical_note"
    patient_source_id = (
        _coerce_text(row.get(spec.patient_id_field))
        if spec.patient_id_field
        else _stable_note_hash(note)
    )
    stable_seed = (
        f"{dataset_id}:{spec.repo_id}:{patient_source_id}:{_stable_note_hash(note)}"
    )
    patient_age = _extract_age(note)
    patient_sex = _extract_sex(note)
    record_id = f"hf-{uuid5(NAMESPACE_URL, stable_seed)}"
    patient_id = f"pat-{uuid5(NAMESPACE_URL, stable_seed + ':patient')}"
    document = ClinicalDocument(
        document_id=f"doc-{uuid5(NAMESPACE_URL, stable_seed + ':document')}",
        note_type=_note_type(note),
        author_role="synthetic_reference",
        timestamp="2026-01-01T00:00:00",
        clean_text=note,
        messy_text=None,
        extracted_facts={
            "source_task": task,
            "instruction": question,
            "answer": answer,
        },
    )
    return SyntheticRecord(
        record_id=record_id,
        dataset_id=dataset_id,
        topic=task or "synthetic clinical note",
        complexity=ComplexityProfile.MODERATE,
        modalities=[Modality.CLINICAL_TEXT],
        patient=SyntheticPatient(
            patient_id=patient_id,
            age=patient_age,
            sex=patient_sex,
            demographics={"source_patient_id": patient_source_id},
        ),
        encounters=[],
        documents=[document],
        provenance=Provenance(
            generator="huggingface-reference-import",
            model=None,
            source_refs=[
                {
                    "repo_id": spec.repo_id,
                    "split": effective_split,
                    "license": spec.license,
                }
            ],
            created_at="2026-01-01T00:00:00",
        ),
        metadata={
            "reference_dataset": spec.repo_id,
            "reference_license": spec.license,
            "reference_split": effective_split,
            "use_policy": "reference_import_not_relicensed",
        },
    )


def _coerce_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _stable_note_hash(note: str) -> str:
    return sha256(note.encode("utf-8")).hexdigest()[:16]


def _extract_age(note: str) -> int:
    match = re.search(r"\b(\d{1,3})\s*[- ]year[- ]old\b", note, flags=re.IGNORECASE)
    if not match:
        return 0
    age = int(match.group(1))
    return age if 0 <= age <= 120 else 0


def _extract_sex(note: str) -> str:
    lowered = note.lower()
    if re.search(r"\b(female|woman|girl)\b", lowered):
        return "female"
    if re.search(r"\b(male|man|boy)\b", lowered):
        return "male"
    return "unknown"


def _note_type(note: str) -> str:
    prefix = note[:80].lower()
    if "discharge summary" in prefix:
        return "discharge_summary"
    if "progress note" in prefix:
        return "progress_note"
    return "clinical_note"
