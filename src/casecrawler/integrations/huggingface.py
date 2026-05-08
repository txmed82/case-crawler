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
    ImagingAsset,
    LabObservation,
    MedicationStatement,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
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
    patient_id_field: str | None = None
    image_field: str | None = None
    image_label_field: str | None = None
    image_label_map: dict[str, str] | None = None
    image_modality: str = "XR"
    image_body_region: str = "chest"
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
    image_field: str | None = None,
    image_label_field: str | None = None,
    image_label_map: dict[str, str] | None = None,
    image_modality: str = "XR",
    image_body_region: str = "chest",
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
        image_field=image_field,
        image_label_field=image_label_field,
        image_label_map=image_label_map,
        image_modality=image_modality,
        image_body_region=image_body_region,
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
    labs, vitals, medications = _fhir_artifacts(answer)
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
        imaging=imaging,
    )
    document = ClinicalDocument(
        document_id=f"doc-{uuid5(NAMESPACE_URL, stable_seed + ':document')}",
        note_type=_note_type(note, spec=spec),
        author_role="synthetic_reference",
        timestamp="2026-01-01T00:00:00",
        clean_text=note,
        messy_text=None,
        extracted_facts={
            "source_task": task,
            "instruction": question,
            "answer": answer,
            "source_fields": _source_fields(row, spec),
        },
    )
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
        encounters=[],
        labs=labs,
        vitals=vitals,
        medication_history=medications,
        documents=[document],
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
    imaging: list[ImagingAsset],
) -> list[Modality]:
    modalities = [Modality.CLINICAL_TEXT]
    if labs or vitals or medications:
        modalities.insert(0, Modality.STRUCTURED_EHR)
    if labs:
        modalities.append(Modality.LABS)
    if vitals:
        modalities.append(Modality.VITALS)
    if imaging:
        modalities.append(Modality.IMAGING)
    return modalities


def _note_type(note: str, *, spec: HuggingFaceReferenceDataset | None = None) -> str:
    if spec and spec.repo_id == "ClarusC64/image-report-consistency-radiology-v01":
        return "radiology_report"
    prefix = note[:80].lower()
    if "discharge summary" in prefix:
        return "discharge_summary"
    if "progress note" in prefix:
        return "progress_note"
    return "clinical_note"


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


def _fhir_artifacts(
    fhir_text: str,
) -> tuple[list[LabObservation], list[VitalObservation], list[MedicationStatement]]:
    if not fhir_text:
        return [], [], []
    try:
        parsed = json.loads(fhir_text)
    except json.JSONDecodeError:
        return [], [], []
    resources = _fhir_resources(parsed)
    labs: list[LabObservation] = []
    vitals: list[VitalObservation] = []
    medications: list[MedicationStatement] = []
    for resource in resources:
        resource_type = resource.get("resourceType")
        if resource_type == "Observation" and isinstance(resource.get("valueQuantity"), dict):
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
    return labs, vitals, medications


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
