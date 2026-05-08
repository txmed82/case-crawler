from __future__ import annotations

import json
import importlib
import re
from hashlib import sha256
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from uuid import NAMESPACE_URL, uuid5

from casecrawler.models.synthetic import (
    ClinicalDocument,
    Code,
    ComplexityProfile,
    Encounter,
    ImagingAsset,
    LabObservation,
    MedicationStatement,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
    TimeSeriesChannel,
    TimeSeriesPoint,
    VitalObservation,
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
    default_task: str = "clinical_note"
    patient_id_field: str | None = None
    image_field: str | None = None
    image_label_field: str | None = None
    image_label_map: dict[str, str] | None = None
    image_modality: str = "XR"
    image_body_region: str = "chest"
    note_type_field: str | None = None
    phi_annotations_field: str | None = None
    diagnosis_codes_field: str | None = None
    diagnosis_code_system: str = "ICD-9-CM"
    lab_values_field: str | None = None
    vital_values_field: str | None = None
    medications_field: str | None = None
    time_series_field: str | None = None
    quality_score_field: str | None = None
    gated: bool = False
    use_policy: str = "review_license_before_use"
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
    "augmented_clinical_notes": HuggingFaceReferenceDataset(
        repo_id="AGBonnet/augmented-clinical-notes",
        split="train",
        license="mit",
        note_field="full_note",
        question_field="conversation",
        answer_field="summary",
        task_field=None,
        patient_id_field="idx",
        description=(
            "Clinical note, synthetic dialogue, and structured patient-summary "
            "triplets for clinical note generation and extraction benchmarking."
        ),
    ),
    "medsynth_dialogue_note": HuggingFaceReferenceDataset(
        repo_id="Ahmad0067/MedSynth",
        split="train",
        license="unspecified",
        note_field="Note",
        question_field="Dialogue",
        task_field="ICD10_desc",
        description=(
            "Synthetic medical dialogue-note pairs for dialogue-to-note and "
            "note-to-dialogue clinical documentation benchmarking."
        ),
    ),
    "clinical_notes_to_fhir": HuggingFaceReferenceDataset(
        repo_id="ai-galileo/clinical-notes-to-fhir",
        split="train",
        license="apache-2.0",
        note_field="note",
        question_field="scenario",
        answer_field="fhir_bundle",
        task_field="difficulty",
        patient_id_field="exampleId",
        description=(
            "Synthetic clinical-note-to-FHIR preference corpus with validation "
            "signals for extraction and structured-output benchmarking."
        ),
    ),
    "radiology_report_consistency": HuggingFaceReferenceDataset(
        repo_id="ClarusC64/image-report-consistency-radiology-v01",
        split="train",
        license="mit",
        note_field="report_excerpt",
        question_field="imaging_findings",
        answer_field="expected_decision",
        task_field="consistency_issue",
        patient_id_field="case_id",
        description=(
            "Synthetic radiology report-consistency evaluation rows for "
            "image-evidence and report-language alignment benchmarking."
        ),
    ),
    "synthchex_75k": HuggingFaceReferenceDataset(
        repo_id="raman07/SynthCheX-75K-v2",
        split="train",
        license="apache-2.0",
        note_field="label",
        patient_id_field=None,
        image_field="image",
        image_label_field="label",
        image_modality="XR",
        image_body_region="chest",
        description=(
            "Synthetic chest radiograph image-text reference set from "
            "CheXGenBench, with pathological annotations."
        ),
    ),
    "rexgradient_160k": HuggingFaceReferenceDataset(
        repo_id="rajpurkarlab/ReXGradient-160K",
        split="validation",
        license="rexgradient-non-commercial-gated",
        note_field="findings",
        answer_field="impression",
        patient_id_field="patient_id",
        image_field="image",
        image_modality="XR",
        image_body_region="chest",
        gated=True,
        use_policy="non_commercial_research_only",
        description=(
            "Gated non-commercial chest radiograph/report reference dataset "
            "for radiology generation benchmarking. Access requires accepting "
            "the Hugging Face dataset terms."
        ),
    ),
    "synthetic_chest_xray_pneumonia": HuggingFaceReferenceDataset(
        repo_id="chimbiwide/synthetic-chest-xray-pneumonia",
        split="train",
        license="cc-by-4.0",
        note_field="label",
        patient_id_field=None,
        image_field="image",
        image_label_field="label",
        image_label_map={"0": "normal", "1": "pneumonia"},
        image_modality="XR",
        image_body_region="chest",
        description=(
            "Synthetic chest X-ray classification reference dataset with "
            "normal and pneumonia labels."
        ),
    ),
    "technetium_i": HuggingFaceReferenceDataset(
        repo_id="temlm-foundation/Technetium-I",
        split="validation",
        license="eupl-1.2",
        note_field="text",
        task_field=None,
        default_task="clinical_deidentification_icd_coding",
        patient_id_field="note_id",
        note_type_field="note_type",
        phi_annotations_field="phi_annotations",
        diagnosis_codes_field="icd_codes",
        diagnosis_code_system="ICD-9-CM",
        quality_score_field="quality_score",
        description=(
            "Large synthetic clinical NLP reference set with PHI annotations "
            "and ICD-9-CM labels for de-identification and coding validation."
        ),
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
    default_task: str = "clinical_note",
    patient_id_field: str | None = None,
    image_field: str | None = None,
    image_label_field: str | None = None,
    image_label_map: dict[str, str] | None = None,
    image_modality: str = "XR",
    image_body_region: str = "chest",
    note_type_field: str | None = None,
    phi_annotations_field: str | None = None,
    diagnosis_codes_field: str | None = None,
    diagnosis_code_system: str = "ICD-9-CM",
    lab_values_field: str | None = None,
    vital_values_field: str | None = None,
    medications_field: str | None = None,
    time_series_field: str | None = None,
    quality_score_field: str | None = None,
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
        default_task=default_task,
        patient_id_field=patient_id_field,
        image_field=image_field,
        image_label_field=image_label_field,
        image_label_map=image_label_map,
        image_modality=image_modality,
        image_body_region=image_body_region,
        note_type_field=note_type_field,
        phi_annotations_field=phi_annotations_field,
        diagnosis_codes_field=diagnosis_codes_field,
        diagnosis_code_system=diagnosis_code_system,
        lab_values_field=lab_values_field,
        vital_values_field=vital_values_field,
        medications_field=medications_field,
        time_series_field=time_series_field,
        quality_score_field=quality_score_field,
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
    image_output_dir: str | Path = "./data/reference_images",
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
                reference_key=reference_key,
                split=effective_split,
                image_output_dir=image_output_dir,
            )
        )
    return records


def reference_row_to_record(
    row: dict,
    *,
    dataset_id: str,
    spec: HuggingFaceReferenceDataset,
    reference_key: str = "unspecified",
    split: str | None = None,
    image_output_dir: str | Path = "./data/reference_images",
) -> SyntheticRecord:
    effective_split = split or spec.split
    image_label = _image_label(row, spec)
    raw_note = _coerce_text(row.get(spec.note_field))
    raw_image_label = (
        _coerce_text(row.get(spec.image_label_field))
        if spec.image_label_field
        else ""
    )
    note = raw_note
    if spec.image_field and image_label and (not note or note == raw_image_label):
        note = f"Synthetic reference {spec.image_body_region} {spec.image_modality} labeled {image_label}."
    question = _coerce_text(row.get(spec.question_field)) if spec.question_field else ""
    answer = _coerce_text(row.get(spec.answer_field)) if spec.answer_field else ""
    task = _coerce_text(row.get(spec.task_field)) if spec.task_field else spec.default_task
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
    labs, vitals, medications, diagnoses, procedures, diagnostic_documents = _fhir_artifacts(
        answer,
        stable_seed=stable_seed,
    )
    labs = [*labs, *_structured_labs(row.get(spec.lab_values_field))]
    vitals = [*vitals, *_structured_vitals(row.get(spec.vital_values_field))]
    medications = [
        *medications,
        *_structured_medications(row.get(spec.medications_field)),
    ]
    time_series = _structured_time_series(row.get(spec.time_series_field))
    if spec.diagnosis_codes_field:
        diagnoses = [
            *diagnoses,
            *_diagnosis_codes(row.get(spec.diagnosis_codes_field), spec),
        ]
    imaging = [
        *_radiology_artifacts(row, spec, stable_seed, report_text=note),
        *_image_reference_artifacts(
            row,
            spec,
            stable_seed,
            report_text=note,
            image_output_dir=image_output_dir,
        ),
    ]
    modalities = _reference_modalities(
        labs=labs,
        vitals=vitals,
        medications=medications,
        diagnoses=diagnoses,
        procedures=procedures,
        imaging=imaging,
        time_series=time_series,
    )
    document = ClinicalDocument(
        document_id=f"doc-{uuid5(NAMESPACE_URL, stable_seed + ':document')}",
        note_type=_note_type(note, spec=spec, row=row),
        author_role="synthetic_reference",
        timestamp="2026-01-01T00:00:00",
        clean_text=note,
        messy_text=None,
        extracted_facts=_reference_extracted_facts(
            row,
            spec,
            source_task=task,
            instruction=question,
            answer=answer,
            labs=labs,
            vitals=vitals,
            medications=medications,
            diagnoses=diagnoses,
            procedures=procedures,
            imaging=imaging,
            time_series=time_series,
        ),
    )
    encounters = []
    if diagnoses or procedures:
        encounters = [
            Encounter(
                encounter_id=f"enc-{uuid5(NAMESPACE_URL, stable_seed + ':encounter')}",
                start="2026-01-01T00:00:00",
                setting="reference",
                reason=task or "synthetic clinical note",
                diagnoses=diagnoses,
                procedures=procedures,
            )
        ]
    return SyntheticRecord(
        record_id=record_id,
        dataset_id=dataset_id,
        topic=task or "synthetic clinical note",
        complexity=ComplexityProfile.MODERATE,
        modalities=modalities,
        patient=SyntheticPatient(
            patient_id=patient_id,
            age=patient_age,
            sex=patient_sex,
            demographics={"source_patient_id": patient_source_id},
        ),
        encounters=encounters,
        labs=labs,
        vitals=vitals,
        medication_history=medications,
        time_series=time_series,
        documents=[document, *diagnostic_documents],
        imaging=imaging,
        provenance=Provenance(
            generator="huggingface-reference-import",
            model=None,
            source_refs=[
                {
                    "reference_key": reference_key,
                    "repo_id": spec.repo_id,
                    "split": effective_split,
                    "license": spec.license,
                }
            ],
            created_at="2026-01-01T00:00:00",
        ),
        metadata={
            "reference_key": reference_key,
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


def _reference_modalities(
    *,
    labs: list[LabObservation],
    vitals: list[VitalObservation],
    medications: list[MedicationStatement],
    diagnoses: list[Code],
    procedures: list[Code],
    imaging: list[ImagingAsset],
    time_series: list[TimeSeriesChannel],
) -> list[Modality]:
    modalities = [Modality.CLINICAL_TEXT]
    if labs or vitals or medications or diagnoses or procedures:
        modalities.insert(0, Modality.STRUCTURED_EHR)
    if labs:
        modalities.append(Modality.LABS)
    if vitals:
        modalities.append(Modality.VITALS)
    if time_series:
        modalities.append(Modality.TIME_SERIES)
    if imaging:
        modalities.append(Modality.IMAGING)
    return modalities


def _note_type(
    note: str,
    *,
    spec: HuggingFaceReferenceDataset | None = None,
    row: dict | None = None,
) -> str:
    if spec and row is not None and spec.note_type_field:
        note_type = _coerce_text(row.get(spec.note_type_field))
        if note_type:
            return _normalize_note_type(note_type)
    if spec and spec.repo_id == "ClarusC64/image-report-consistency-radiology-v01":
        return "radiology_report"
    prefix = note[:80].lower()
    if "discharge summary" in prefix:
        return "discharge_summary"
    if "progress note" in prefix:
        return "progress_note"
    return "clinical_note"


def _normalize_note_type(value: str) -> str:
    return "_".join(value.lower().replace("-", "_").split())


def _source_fields(row: dict, spec: HuggingFaceReferenceDataset) -> dict:
    mapped_fields = {
        spec.note_field,
        spec.question_field,
        spec.answer_field,
    }
    return {
        str(key): value
        for key, value in sorted(row.items())
        if key not in mapped_fields and _is_source_field_value(value)
    }


def _is_source_field_value(value) -> bool:
    return value is None or isinstance(value, (str, int, float, bool, list, dict))


def _reference_extracted_facts(
    row: dict,
    spec: HuggingFaceReferenceDataset,
    *,
    source_task: str,
    instruction: str,
    answer: str,
    labs: list[LabObservation],
    vitals: list[VitalObservation],
    medications: list[MedicationStatement],
    diagnoses: list[Code],
    procedures: list[Code],
    imaging: list[ImagingAsset],
    time_series: list[TimeSeriesChannel],
) -> dict:
    facts = {
        "source_task": source_task,
        "instruction": instruction,
        "answer": answer,
        "source_fields": _source_fields(row, spec),
    }
    phi_annotations = (
        _list_of_dicts(row.get(spec.phi_annotations_field))
        if spec.phi_annotations_field
        else []
    )
    if phi_annotations:
        facts["phi_annotations"] = phi_annotations
        facts["phi_entity_counts"] = _phi_entity_counts(phi_annotations)
    if spec.quality_score_field:
        quality_score = _numeric_or_none(row.get(spec.quality_score_field))
        if quality_score is not None:
            facts["source_quality_score"] = quality_score
    if labs:
        facts["lab_values"] = [
            {
                "name": lab.name,
                "value": lab.value,
                "unit": lab.unit,
                "reference_low": lab.reference_low,
                "reference_high": lab.reference_high,
                "flag": lab.flag,
                "effective_time": lab.effective_time,
                "specimen": lab.specimen,
            }
            for lab in labs
        ]
    if vitals:
        facts["vital_values"] = [
            {
                "name": vital.name,
                "value": vital.value,
                "unit": vital.unit,
                "effective_time": vital.effective_time,
            }
            for vital in vitals
        ]
    if medications:
        facts["medications"] = [medication.name for medication in medications]
        facts["medication_details"] = [
            {
                "name": medication.name,
                "rxnorm": medication.rxnorm,
                "dose": medication.dose,
                "route": medication.route,
                "frequency": medication.frequency,
                "status": medication.status,
                "start": medication.start,
                "end": medication.end,
            }
            for medication in medications
        ]
    if time_series:
        facts["time_series_channels"] = [
            {
                "name": channel.name,
                "unit": channel.unit,
                "generation_backend": channel.generation_backend,
                "sampling_rate_hz": channel.sampling_rate_hz,
                "points": [
                    {
                        "timestamp": point.timestamp,
                        "values": point.values,
                    }
                    for point in channel.points
                ],
            }
            for channel in time_series
        ]
    if diagnoses:
        facts["diagnoses"] = [
            {
                "system": diagnosis.system,
                "code": diagnosis.code,
                "display": diagnosis.display,
            }
            for diagnosis in diagnoses
        ]
    if procedures:
        facts["procedures"] = [procedure.display for procedure in procedures]
        facts["procedure_details"] = [
            {
                "system": procedure.system,
                "code": procedure.code,
                "display": procedure.display,
            }
            for procedure in procedures
        ]
    if imaging:
        facts["imaging_asset_ids"] = [asset.image_id for asset in imaging]
        facts["imaging_modalities"] = [asset.modality for asset in imaging]
        facts["imaging_body_regions"] = [asset.body_region for asset in imaging]
        facts["imaging_labels"] = [
            label.display
            for asset in imaging
            for label in asset.labels
            if label.display
        ]
    return facts


def _diagnosis_codes(value: object, spec: HuggingFaceReferenceDataset) -> list[Code]:
    codes = _string_list(value)
    return [
        Code(
            system=spec.diagnosis_code_system,
            code=code,
            display=f"{spec.diagnosis_code_system} {code}",
        )
        for code in codes
    ]


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list | tuple | set):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]
    return [str(value).strip()] if str(value).strip() else []


def _list_of_dicts(value: object) -> list[dict]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _phi_entity_counts(annotations: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for annotation in annotations:
        entity_type = _coerce_text(annotation.get("entity_type"))
        if not entity_type:
            continue
        counts[entity_type] = counts.get(entity_type, 0) + 1
    return dict(sorted(counts.items()))


def _numeric_or_none(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _structured_labs(value: object) -> list[LabObservation]:
    labs: list[LabObservation] = []
    for item in _list_of_dicts(value):
        name = _coerce_text(item.get("name") or item.get("test") or item.get("label"))
        unit = _coerce_text(item.get("unit"))
        effective_time = _coerce_text(
            item.get("effective_time") or item.get("timestamp") or item.get("time")
        )
        if not name or not unit:
            continue
        raw_value = item.get("value")
        numeric_value = _numeric_or_none(raw_value)
        labs.append(
            LabObservation(
                name=name,
                loinc=_coerce_optional_text(item.get("loinc")),
                value=numeric_value if numeric_value is not None else _coerce_text(raw_value),
                unit=unit,
                reference_low=_numeric_or_none(item.get("reference_low")),
                reference_high=_numeric_or_none(item.get("reference_high")),
                flag=_coerce_optional_text(item.get("flag")),
                effective_time=effective_time or "2026-01-01T00:00:00",
                specimen=_coerce_optional_text(item.get("specimen")),
            )
        )
    return labs


def _structured_vitals(value: object) -> list[VitalObservation]:
    vitals: list[VitalObservation] = []
    for item in _list_of_dicts(value):
        name = _coerce_text(item.get("name") or item.get("vital") or item.get("label"))
        unit = _coerce_text(item.get("unit"))
        numeric_value = _numeric_or_none(item.get("value"))
        effective_time = _coerce_text(
            item.get("effective_time") or item.get("timestamp") or item.get("time")
        )
        if not name or not unit or numeric_value is None:
            continue
        vitals.append(
            VitalObservation(
                name=name,
                value=numeric_value,
                unit=unit,
                effective_time=effective_time or "2026-01-01T00:00:00",
            )
        )
    return vitals


def _structured_medications(value: object) -> list[MedicationStatement]:
    medications: list[MedicationStatement] = []
    for item in _list_of_dicts(value):
        name = _coerce_text(item.get("name") or item.get("medication") or item.get("drug"))
        if not name:
            continue
        medications.append(
            MedicationStatement(
                name=name,
                rxnorm=_coerce_optional_text(item.get("rxnorm")),
                dose=_coerce_optional_text(item.get("dose")),
                route=_coerce_optional_text(item.get("route")),
                frequency=_coerce_optional_text(item.get("frequency")),
                status=_coerce_text(item.get("status")) or "unknown",
                start=_coerce_optional_text(item.get("start")),
                end=_coerce_optional_text(item.get("end")),
            )
        )
    return medications


def _structured_time_series(value: object) -> list[TimeSeriesChannel]:
    channels: list[TimeSeriesChannel] = []
    for item in _list_of_dicts(value):
        name = _coerce_text(item.get("name") or item.get("channel") or item.get("label"))
        unit = _coerce_text(item.get("unit"))
        if not name or not unit:
            continue
        points = _structured_time_series_points(item.get("points"))
        if not points:
            continue
        channels.append(
            TimeSeriesChannel(
                name=name,
                unit=unit,
                generation_backend=(
                    _coerce_text(item.get("generation_backend")) or "reference"
                ),
                sampling_rate_hz=_time_series_sampling_rate(item),
                points=points,
            )
        )
    return channels


def _time_series_sampling_rate(item: dict) -> float | None:
    for key in (
        "sampling_rate_hz",
        "sample_rate_hz",
        "sampling_frequency_hz",
        "frequency_hz",
        "hz",
    ):
        value = _numeric_or_none(item.get(key))
        if value is not None:
            return value
    return None


def _structured_time_series_points(value: object) -> list[TimeSeriesPoint]:
    points: list[TimeSeriesPoint] = []
    for item in _list_of_dicts(value):
        timestamp = _coerce_text(
            item.get("timestamp") or item.get("effective_time") or item.get("time")
        )
        values = item.get("values")
        if not isinstance(values, dict):
            numeric_value = _numeric_or_none(item.get("value"))
            values = {"value": numeric_value} if numeric_value is not None else {}
        numeric_values = {
            str(key): numeric
            for key, raw in values.items()
            if (numeric := _numeric_or_none(raw)) is not None
        }
        if not timestamp or not numeric_values:
            continue
        points.append(TimeSeriesPoint(timestamp=timestamp, values=numeric_values))
    return points


def _coerce_optional_text(value: object) -> str | None:
    text = _coerce_text(value)
    return text or None


def _fhir_artifacts(
    fhir_text: str,
    *,
    stable_seed: str,
) -> tuple[
    list[LabObservation],
    list[VitalObservation],
    list[MedicationStatement],
    list[Code],
    list[Code],
    list[ClinicalDocument],
]:
    if not fhir_text:
        return [], [], [], [], [], []
    try:
        parsed = json.loads(fhir_text)
    except json.JSONDecodeError:
        return [], [], [], [], [], []
    resources = _fhir_resources(parsed)
    labs: list[LabObservation] = []
    vitals: list[VitalObservation] = []
    medications: list[MedicationStatement] = []
    diagnoses: list[Code] = []
    procedures: list[Code] = []
    diagnostic_documents: list[ClinicalDocument] = []
    for resource in resources:
        resource_type = resource.get("resourceType")
        if resource_type == "Observation":
            components = _fhir_observation_components(resource)
            if components:
                if _is_fhir_vital_observation(resource):
                    vitals.extend(
                        vital
                        for vital in (
                            _fhir_component_to_vital(resource, component)
                            for component in components
                        )
                        if vital is not None
                    )
                else:
                    labs.extend(
                        lab
                        for lab in (
                            _fhir_component_to_lab(resource, component)
                            for component in components
                        )
                        if lab is not None
                    )
                continue
            if isinstance(resource.get("valueQuantity"), dict):
                if _is_fhir_vital_observation(resource):
                    vital = _fhir_observation_to_vital(resource)
                    if vital is not None:
                        vitals.append(vital)
                else:
                    lab = _fhir_observation_to_lab(resource)
                    if lab is not None:
                        labs.append(lab)
        elif resource_type == "MedicationStatement":
            medication = _fhir_medication_statement(resource)
            if medication is not None:
                medications.append(medication)
        elif resource_type == "Condition":
            diagnosis = _fhir_codeable_concept_to_code(resource.get("code"), fallback="Condition")
            if diagnosis is not None:
                diagnoses.append(diagnosis)
        elif resource_type == "Procedure":
            procedure = _fhir_codeable_concept_to_code(resource.get("code"), fallback="Procedure")
            if procedure is not None:
                procedures.append(procedure)
        elif resource_type == "DiagnosticReport":
            document = _fhir_diagnostic_report_document(resource, stable_seed=stable_seed)
            if document is not None:
                diagnostic_documents.append(document)
    return labs, vitals, medications, diagnoses, procedures, diagnostic_documents


def _fhir_resources(parsed) -> list[dict]:
    if not isinstance(parsed, dict):
        return []
    if parsed.get("resourceType") == "Bundle":
        resources: list[dict] = []
        for entry in parsed.get("entry", []):
            if isinstance(entry, dict) and isinstance(entry.get("resource"), dict):
                resources.append(entry["resource"])
        return resources
    return [parsed]


def _is_fhir_vital_observation(resource: dict) -> bool:
    for category in resource.get("category", []):
        if not isinstance(category, dict):
            continue
        for coding in category.get("coding", []):
            if isinstance(coding, dict) and coding.get("code") == "vital-signs":
                return True
    return False


def _fhir_observation_to_lab(resource: dict) -> LabObservation | None:
    quantity = resource.get("valueQuantity")
    if not isinstance(quantity, dict) or "value" not in quantity:
        return None
    low, high = _fhir_reference_range(resource, quantity.get("unit", ""))
    return LabObservation(
        name=_fhir_code_text(resource) or resource.get("id") or "Observation",
        loinc=_fhir_loinc(resource),
        value=quantity["value"],
        unit=_coerce_text(quantity.get("unit")),
        reference_low=low,
        reference_high=high,
        flag=_lab_flag(quantity["value"], low, high),
        effective_time=_coerce_text(resource.get("effectiveDateTime")) or "2026-01-01T00:00:00",
    )


def _fhir_observation_to_vital(resource: dict) -> VitalObservation | None:
    quantity = resource.get("valueQuantity")
    if not isinstance(quantity, dict) or not isinstance(quantity.get("value"), (int, float)):
        return None
    return VitalObservation(
        name=_fhir_code_text(resource) or resource.get("id") or "Vital sign",
        value=float(quantity["value"]),
        unit=_coerce_text(quantity.get("unit")),
        effective_time=_coerce_text(resource.get("effectiveDateTime")) or "2026-01-01T00:00:00",
    )


def _fhir_observation_components(resource: dict) -> list[dict]:
    return [
        component
        for component in resource.get("component", [])
        if isinstance(component, dict)
        and isinstance(component.get("valueQuantity"), dict)
    ]


def _fhir_component_to_lab(
    resource: dict,
    component: dict,
) -> LabObservation | None:
    quantity = component.get("valueQuantity")
    if not isinstance(quantity, dict) or "value" not in quantity:
        return None
    low, high = _fhir_reference_range(component, quantity.get("unit", ""))
    return LabObservation(
        name=_fhir_code_text(component) or component.get("id") or "Observation component",
        loinc=_fhir_loinc(component),
        value=quantity["value"],
        unit=_coerce_text(quantity.get("unit")),
        reference_low=low,
        reference_high=high,
        flag=_lab_flag(quantity["value"], low, high),
        effective_time=_coerce_text(resource.get("effectiveDateTime"))
        or "2026-01-01T00:00:00",
    )


def _fhir_component_to_vital(
    resource: dict,
    component: dict,
) -> VitalObservation | None:
    quantity = component.get("valueQuantity")
    if not isinstance(quantity, dict) or not isinstance(quantity.get("value"), (int, float)):
        return None
    return VitalObservation(
        name=_fhir_code_text(component) or component.get("id") or "Vital sign component",
        value=float(quantity["value"]),
        unit=_coerce_text(quantity.get("unit")),
        effective_time=_coerce_text(resource.get("effectiveDateTime"))
        or "2026-01-01T00:00:00",
    )


def _fhir_medication_statement(resource: dict) -> MedicationStatement | None:
    name = _coerce_text(
        resource.get("medicationCodeableConcept", {}).get("text")
        if isinstance(resource.get("medicationCodeableConcept"), dict)
        else resource.get("medication")
    )
    if not name:
        return None
    dosage = next(
        (item for item in resource.get("dosage", []) if isinstance(item, dict)),
        {},
    )
    route = ""
    if isinstance(dosage.get("route"), dict):
        route = _coerce_text(dosage["route"].get("text"))
    return MedicationStatement(
        name=name,
        dose=_coerce_text(dosage.get("text")) or None,
        route=route or None,
        status=_coerce_text(resource.get("status")) or "unknown",
    )


def _fhir_diagnostic_report_document(
    resource: dict,
    *,
    stable_seed: str,
) -> ClinicalDocument | None:
    text = _coerce_text(resource.get("conclusion"))
    if not text:
        presented_forms = resource.get("presentedForm")
        first_presented_form = (
            presented_forms[0]
            if (
                isinstance(presented_forms, list)
                and presented_forms
                and isinstance(presented_forms[0], dict)
            )
            else {}
        )
        text = _coerce_text(first_presented_form.get("data"))
    code = _fhir_codeable_concept_to_code(resource.get("code"), fallback="Diagnostic report")
    if not text and code is None:
        return None
    report_id = _coerce_text(resource.get("id")) or _stable_note_hash(json.dumps(resource, sort_keys=True))
    timestamp = _coerce_text(resource.get("effectiveDateTime")) or "2026-01-01T00:00:00"
    clean_text = text or f"Diagnostic report: {code.display}"
    extracted_facts = {
        "source_fhir_resource_type": "DiagnosticReport",
        "source_fhir_resource_id": report_id,
    }
    if code is not None:
        extracted_facts["diagnostic_report_code"] = {
            "system": code.system,
            "code": code.code,
            "display": code.display,
        }
    return ClinicalDocument(
        document_id=f"doc-{uuid5(NAMESPACE_URL, stable_seed + ':diagnostic-report:' + report_id)}",
        note_type="diagnostic_report",
        author_role="synthetic_reference",
        timestamp=timestamp,
        clean_text=clean_text,
        messy_text=None,
        extracted_facts=extracted_facts,
    )


def _fhir_codeable_concept_to_code(value, *, fallback: str) -> Code | None:
    if not isinstance(value, dict):
        return None
    coding = next(
        (item for item in value.get("coding", []) if isinstance(item, dict)),
        {},
    )
    system = _coerce_text(coding.get("system")) or "fhir"
    code = _coerce_text(coding.get("code"))
    display = (
        _coerce_text(value.get("text"))
        or _coerce_text(coding.get("display"))
        or code
        or fallback
    )
    if not display:
        return None
    return Code(
        system=system,
        code=code or re.sub(r"\W+", "_", display.lower()).strip("_"),
        display=display,
    )


def _fhir_code_text(resource: dict) -> str:
    code = resource.get("code")
    if not isinstance(code, dict):
        return ""
    return _coerce_text(code.get("text")) or _coerce_text(
        next(
            (
                coding.get("display")
                for coding in code.get("coding", [])
                if isinstance(coding, dict) and coding.get("display")
            ),
            "",
        )
    )


def _fhir_loinc(resource: dict) -> str | None:
    code = resource.get("code")
    if not isinstance(code, dict):
        return None
    for coding in code.get("coding", []):
        if not isinstance(coding, dict):
            continue
        if "loinc.org" in _coerce_text(coding.get("system")).lower():
            return _coerce_text(coding.get("code")) or None
    return None


def _fhir_reference_range(
    resource: dict,
    unit: str,
) -> tuple[float | None, float | None]:
    ranges = resource.get("referenceRange")
    if not isinstance(ranges, list) or not ranges:
        return None, None
    first = ranges[0] if isinstance(ranges[0], dict) else {}
    low = first.get("low", {})
    high = first.get("high", {})
    return _quantity_value(low, unit), _quantity_value(high, unit)


def _quantity_value(quantity, unit: str) -> float | None:
    if not isinstance(quantity, dict):
        return None
    value = quantity.get("value")
    if not isinstance(value, (int, float)):
        return None
    quantity_unit = _coerce_text(quantity.get("unit"))
    if quantity_unit and unit and quantity_unit != unit:
        return None
    return float(value)


def _lab_flag(value, low: float | None, high: float | None) -> str | None:
    if not isinstance(value, (int, float)):
        return None
    if low is not None and value < low:
        return "L"
    if high is not None and value > high:
        return "H"
    return None


def _radiology_artifacts(
    row: dict,
    spec: HuggingFaceReferenceDataset,
    stable_seed: str,
    *,
    report_text: str,
) -> list[ImagingAsset]:
    if spec.repo_id != "ClarusC64/image-report-consistency-radiology-v01":
        return []
    findings = _coerce_text(row.get("imaging_findings"))
    study = _coerce_text(row.get("study"))
    labels = _radiology_labels(findings)
    return [
        ImagingAsset(
            image_id=f"img-{uuid5(NAMESPACE_URL, stable_seed + ':image')}",
            modality=_coerce_text(row.get("modality")) or "unknown",
            body_region=_body_region(study or findings),
            prompt=findings or study or report_text,
            report_text=report_text,
            labels=labels,
            generation_backend="huggingface-reference",
        )
    ]


def _image_reference_artifacts(
    row: dict,
    spec: HuggingFaceReferenceDataset,
    stable_seed: str,
    *,
    report_text: str,
    image_output_dir: str | Path,
) -> list[ImagingAsset]:
    if not spec.image_field:
        return []
    image_value = row.get(spec.image_field)
    image_path = _persist_reference_image(
        image_value,
        stable_seed=stable_seed,
        image_output_dir=image_output_dir,
    )
    label_text = _image_label(row, spec)
    labels = _radiology_labels(label_text)
    if label_text and not labels and label_text != "normal":
        labels = [
            Code(
                system="huggingface-reference",
                code=re.sub(r"\W+", "_", label_text.lower()).strip("_"),
                display=label_text,
            )
        ]
    prompt = (
        f"Synthetic reference {spec.image_body_region} {spec.image_modality} "
        f"labeled {label_text or 'unknown'}."
    )
    return [
        ImagingAsset(
            image_id=f"img-{uuid5(NAMESPACE_URL, stable_seed + ':image-reference')}",
            modality=spec.image_modality,
            body_region=spec.image_body_region,
            prompt=prompt,
            file_path=image_path,
            report_text=report_text or prompt,
            labels=labels,
            generation_backend=f"huggingface-reference:{spec.repo_id}",
        )
    ]


def _persist_reference_image(
    image_value,
    *,
    stable_seed: str,
    image_output_dir: str | Path,
) -> str | None:
    if image_value is None:
        return None
    if isinstance(image_value, str):
        return image_value
    if isinstance(image_value, dict):
        path = image_value.get("path")
        if isinstance(path, str) and path:
            return path
        bytes_value = image_value.get("bytes")
        if isinstance(bytes_value, bytes):
            output_dir = Path(image_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"hf-{uuid5(NAMESPACE_URL, stable_seed + ':image')}.png"
            output_path.write_bytes(bytes_value)
            return str(output_path)
    if hasattr(image_value, "save"):
        output_dir = Path(image_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"hf-{uuid5(NAMESPACE_URL, stable_seed + ':image')}.png"
        image_value.save(output_path)
        return str(output_path)
    return None


def _image_label(row: dict, spec: HuggingFaceReferenceDataset) -> str:
    if not spec.image_label_field:
        return ""
    raw_label = row.get(spec.image_label_field)
    label = _coerce_text(raw_label)
    if spec.image_label_map:
        return spec.image_label_map.get(label, label)
    return label


def _radiology_labels(findings: str) -> list[Code]:
    lowered = findings.lower()
    candidates = {
        "pleural_effusion": "Pleural effusion",
        "pneumothorax": "Pneumothorax",
        "pneumonia": "Pneumonia",
        "opacity": "Opacity",
        "edema": "Pulmonary edema",
        "atelectasis": "Atelectasis",
        "fracture": "Fracture",
        "appendicitis": "Appendicitis",
    }
    labels: list[Code] = []
    for code, display in candidates.items():
        terms = {code.replace("_", " "), display.lower()}
        if code == "edema":
            terms.add("pulmonary edema")
        if any(term in lowered for term in terms):
            labels.append(Code(system="huggingface-reference", code=code, display=display))
    return labels


def _body_region(text: str) -> str:
    lowered = text.lower()
    for region in ["chest", "abdomen", "pelvis", "head", "brain", "spine"]:
        if region in lowered:
            return region
    return "unknown"
