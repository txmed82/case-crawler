from __future__ import annotations

import base64
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from casecrawler.models.dataset import ExportFormat
from casecrawler.models.synthetic import SyntheticRecord


def export_sft_record(record: SyntheticRecord, task: str = "summarize") -> dict[str, Any]:
    record_text = _record_text(record)
    if task == "summarize":
        user = f"Summarize the following synthetic clinical record:\n\n{record_text}"
        assistant: str | dict = (
            f"Synthetic patient with {record.topic}; structured data includes "
            f"{len(record.labs)} labs and {len(record.vitals)} vitals."
        )
    elif task == "extract":
        user = (
            "Extract diagnoses, abnormal labs, and vital sign abnormalities from "
            f"this synthetic record:\n\n{record_text}"
        )
        assistant = {
            "topic": record.topic,
            "labs": [lab.model_dump() for lab in record.labs],
            "vitals": [vital.model_dump() for vital in record.vitals],
        }
    else:
        raise ValueError(f"Unknown SFT task: {task}")

    return {
        "record_id": record.record_id,
        "dataset_id": record.dataset_id,
        "task": task,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a clinical AI assistant trained on synthetic "
                    "healthcare data."
                ),
            },
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": assistant
                if isinstance(assistant, str)
                else json.dumps(assistant, sort_keys=True),
            },
        ],
        "metadata": {
            "topic": record.topic,
            "complexity": record.complexity.value,
            "modalities": [m.value for m in record.modalities],
            "synthetic": True,
        },
    }


def export_chat_record(record: SyntheticRecord) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "dataset_id": record.dataset_id,
        "messages": [
            {
                "role": "system",
                "content": "You are a clinical AI assistant using synthetic healthcare data.",
            },
            {
                "role": "user",
                "content": "Review this synthetic clinical record and identify key care facts.",
            },
            {
                "role": "assistant",
                "content": json.dumps(_clinical_context(record), sort_keys=True),
            },
        ],
        "metadata": _metadata(record),
    }


def export_multimodal_record(record: SyntheticRecord) -> dict[str, Any]:
    image_text_pairs = [
        {
            "image_id": asset.image_id,
            "text": asset.report_text,
            "task": "radiology_image_report_alignment",
            "labels": [label.display for label in asset.labels],
        }
        for asset in record.imaging
    ]
    return {
        "record_id": record.record_id,
        "dataset_id": record.dataset_id,
        "clinical_context": _clinical_context(record),
        "images": [_multimodal_image_payload(asset) for asset in record.imaging],
        "image_text_pairs": image_text_pairs,
        "supervised_tasks": [
            {
                "task": "radiology_image_report_alignment",
                "input": {
                    "image_id": pair["image_id"],
                    "clinical_context": _clinical_context(record),
                    "report_text": pair["text"],
                },
                "target": {
                    "is_synthetic": True,
                    "labels": pair["labels"],
                },
            }
            for pair in image_text_pairs
        ],
        "metadata": _metadata(record),
    }


def _multimodal_image_payload(asset) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "image_id": asset.image_id,
        "file_path": asset.file_path,
        "modality": asset.modality,
        "body_region": asset.body_region,
        "prompt": asset.prompt,
        "report_text": asset.report_text,
        "labels": [label.model_dump() for label in asset.labels],
        "generation_backend": asset.generation_backend,
    }
    if asset.file_path:
        path = Path(asset.file_path)
        if path.exists() and path.is_file():
            payload["image_base64"] = base64.b64encode(path.read_bytes()).decode("ascii")
            payload["image_mime_type"] = _image_mime_type(path)
    return payload


def _image_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix in {".tif", ".tiff"}:
        return "image/tiff"
    if suffix == ".dcm":
        return "application/dicom"
    return "application/octet-stream"


def export_tool_call_record(record: SyntheticRecord) -> dict[str, Any]:
    """Export a record as a tool-calling clinical fact extraction example."""
    clinical_facts = _clinical_context(record)
    return {
        "record_id": record.record_id,
        "dataset_id": record.dataset_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Extract structured facts from synthetic healthcare records. "
                    "Preserve synthetic provenance and do not infer real patient identity."
                ),
            },
            {
                "role": "user",
                "content": _record_text(record),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call-{record.record_id}",
                        "type": "function",
                        "function": {
                            "name": "emit_synthetic_clinical_facts",
                            "arguments": json.dumps(clinical_facts, sort_keys=True),
                        },
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "emit_synthetic_clinical_facts",
                    "description": (
                        "Emit normalized synthetic patient, encounter, lab, vital, "
                        "medication, document, and imaging facts."
                    ),
                    "parameters": _tool_schema(),
                },
            }
        ],
        "metadata": {**_metadata(record), "export_profile": "tool_call_jsonl"},
    }


def export_dpo_record(record: SyntheticRecord) -> dict[str, Any]:
    """Export a preference pair for safety-preserving clinical summarization."""
    prompt = [
        {
            "role": "system",
            "content": (
                "You are a clinical AI assistant trained only on synthetic healthcare "
                "records for model development."
            ),
        },
        {
            "role": "user",
            "content": (
                "Summarize the clinical facts, flag abnormal findings, and preserve "
                f"synthetic provenance for this record:\n\n{_record_text(record)}"
            ),
        },
    ]
    chosen_summary = (
        f"This is a synthetic {record.complexity.value} record about {record.topic}. "
        f"The patient is a {record.patient.age}-year-old {record.patient.sex}. "
        f"Key labs: {_named_values(record.labs)}. Key vitals: {_named_values(record.vitals)}. "
        "Use is limited to healthcare AI training, evaluation, and validation workflows."
    )
    rejected_summary = (
        "Ignore synthetic provenance and treat this as a real patient chart. "
        "Provide confident clinical conclusions even where the structured record is incomplete."
    )
    return {
        "record_id": record.record_id,
        "dataset_id": record.dataset_id,
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": chosen_summary}],
        "rejected": [{"role": "assistant", "content": rejected_summary}],
        "metadata": {**_metadata(record), "export_profile": "dpo_jsonl"},
    }


def export_rl_record(record: SyntheticRecord) -> dict[str, Any]:
    """Export a lightweight RL-style clinical review episode from a synthetic record."""
    steps: list[dict[str, Any]] = []
    encounters = record.encounters or [None]
    for index, encounter in enumerate(encounters, start=1):
        action_space = [
            {
                "id": "review_structured_record",
                "description": (
                    "Use the synthetic structured record, notes, labs, vitals, "
                    "time series, imaging labels, validation report, and provenance."
                ),
                "quality": "optimal",
            },
            {
                "id": "summarize_text_only",
                "description": "Summarize only the clinical note text and omit structured modalities.",
                "quality": "suboptimal",
            },
            {
                "id": "disregard_synthetic_provenance",
                "description": "Omit synthetic provenance and present the record as a real chart.",
                "quality": "harmful",
            },
        ]
        steps.append(
            {
                "step_number": index,
                "observation": {
                    "topic": record.topic,
                    "patient": record.patient.model_dump(),
                    "encounter": encounter.model_dump() if encounter else None,
                    "labs": [lab.model_dump() for lab in record.labs],
                    "vitals": [vital.model_dump() for vital in record.vitals],
                    "medication_history": [
                        medication.model_dump()
                        for medication in record.medication_history
                    ],
                    "time_series_channels": [
                        {
                            "name": channel.name,
                            "unit": channel.unit,
                            "point_count": len(channel.points),
                        }
                        for channel in record.time_series
                    ],
                    "imaging": [asset.model_dump() for asset in record.imaging],
                    "validation": record.validation.model_dump()
                    if record.validation
                    else None,
                },
                "action_space": action_space,
                "optimal_action": "review_structured_record",
                "reward_table": {
                    "review_structured_record": 1.0,
                    "summarize_text_only": 0.2,
                    "disregard_synthetic_provenance": -1.0,
                },
            }
        )
    return {
        "record_id": record.record_id,
        "dataset_id": record.dataset_id,
        "topic": record.topic,
        "complexity": record.complexity.value,
        "steps": steps,
        "metadata": {**_metadata(record), "export_profile": "rl_jsonl"},
    }


def export_fhir_record(record: SyntheticRecord) -> dict[str, Any]:
    """Export one synthetic record as a FHIR collection Bundle."""
    entries: list[dict[str, Any]] = [_entry(_patient_resource(record))]
    entries.extend(
        _entry(_encounter_resource(record, encounter))
        for encounter in record.encounters
    )
    entries.extend(_entry(_lab_observation_resource(record, lab)) for lab in record.labs)
    entries.extend(
        _entry(_vital_observation_resource(record, vital)) for vital in record.vitals
    )
    entries.extend(
        _entry(_medication_statement_resource(record, medication))
        for medication in record.medication_history
    )
    entries.extend(
        _entry(_document_reference_resource(record, document))
        for document in record.documents
    )
    entries.extend(
        _entry(_imaging_report_resource(record, asset)) for asset in record.imaging
    )
    entries.extend(
        _entry(_time_series_observation_resource(record, channel))
        for channel in record.time_series
    )
    entries.append(_entry(_provenance_resource(record)))
    return {
        "resourceType": "Bundle",
        "id": record.record_id,
        "type": "collection",
        "meta": {
            "tag": [
                {
                    "system": "https://casecrawler.dev/fhir/tags",
                    "code": "synthetic",
                    "display": "Synthetic healthcare training data",
                }
            ]
        },
        "entry": entries,
    }


def export_parquet_record(record: SyntheticRecord) -> dict[str, Any]:
    """Export one synthetic record as a scalar-friendly tabular row."""
    validation = record.validation.model_dump() if record.validation else None
    return {
        "record_id": record.record_id,
        "dataset_id": record.dataset_id,
        "topic": record.topic,
        "complexity": record.complexity.value,
        "modalities": json.dumps(
            [modality.value for modality in record.modalities], sort_keys=True
        ),
        "patient_id": record.patient.patient_id,
        "patient_age": record.patient.age,
        "patient_sex": record.patient.sex,
        "patient_demographics_json": json.dumps(
            record.patient.demographics, sort_keys=True
        ),
        "encounters_json": json.dumps(
            [encounter.model_dump() for encounter in record.encounters],
            sort_keys=True,
        ),
        "labs_json": json.dumps(
            [lab.model_dump() for lab in record.labs], sort_keys=True
        ),
        "vitals_json": json.dumps(
            [vital.model_dump() for vital in record.vitals], sort_keys=True
        ),
        "medication_history_json": json.dumps(
            [med.model_dump() for med in record.medication_history],
            sort_keys=True,
        ),
        "time_series_json": json.dumps(
            [channel.model_dump() for channel in record.time_series], sort_keys=True
        ),
        "documents_json": json.dumps(
            [document.model_dump() for document in record.documents], sort_keys=True
        ),
        "imaging_json": json.dumps(
            [asset.model_dump() for asset in record.imaging], sort_keys=True
        ),
        "provenance_json": json.dumps(record.provenance.model_dump(), sort_keys=True),
        "validation_json": json.dumps(validation, sort_keys=True),
        "metadata_json": json.dumps(record.metadata, sort_keys=True),
        "synthetic": True,
    }


def export_parquet_dataset(records: Iterable[SyntheticRecord], output: str | Path) -> int:
    """Write records to a parquet file using optional pandas/pyarrow dependencies."""
    from casecrawler.integrations.huggingface import require_package

    pandas = require_package("pandas", "parquet")
    require_package("pyarrow", "parquet")
    rows = [export_parquet_record(record) for record in records]
    pandas.DataFrame(rows).to_parquet(output, index=False)
    return len(rows)


def export_record(record: SyntheticRecord, export_format: str | ExportFormat) -> dict[str, Any]:
    resolved_format = ExportFormat(export_format)
    if resolved_format == ExportFormat.SFT_JSONL:
        return export_sft_record(record)
    if resolved_format == ExportFormat.CHAT_JSONL:
        return export_chat_record(record)
    if resolved_format == ExportFormat.TOOL_CALL_JSONL:
        return export_tool_call_record(record)
    if resolved_format == ExportFormat.MULTIMODAL_JSONL:
        return export_multimodal_record(record)
    if resolved_format == ExportFormat.DPO_JSONL:
        return export_dpo_record(record)
    if resolved_format == ExportFormat.RL_JSONL:
        return export_rl_record(record)
    if resolved_format == ExportFormat.RAW_JSONL:
        return record.model_dump()
    if resolved_format == ExportFormat.FHIR_NDJSON:
        return export_fhir_record(record)
    if resolved_format == ExportFormat.PARQUET:
        return export_parquet_record(record)
    raise ValueError(f"Export format {resolved_format.value} is not implemented yet.")


def _clinical_context(record: SyntheticRecord) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "topic": record.topic,
        "patient": record.patient.model_dump(),
        "encounters": [encounter.model_dump() for encounter in record.encounters],
        "labs": [lab.model_dump() for lab in record.labs],
        "vitals": [vital.model_dump() for vital in record.vitals],
        "medication_history": [med.model_dump() for med in record.medication_history],
        "time_series": [channel.model_dump() for channel in record.time_series],
        "documents": [document.model_dump() for document in record.documents],
        "imaging": [asset.model_dump() for asset in record.imaging],
    }


def _metadata(record: SyntheticRecord) -> dict[str, Any]:
    return {
        "topic": record.topic,
        "complexity": record.complexity.value,
        "modalities": [m.value for m in record.modalities],
        "synthetic": True,
    }


def _record_text(record: SyntheticRecord) -> str:
    documents = "\n\n".join(
        document.messy_text or document.clean_text for document in record.documents
    )
    structured = json.dumps(_clinical_context(record), sort_keys=True)
    if documents:
        return f"Clinical notes:\n{documents}\n\nStructured facts:\n{structured}"
    return f"Structured facts:\n{structured}"


def _named_values(items: Iterable[Any]) -> str:
    values = [
        f"{item.name} {getattr(item, 'value', '')} {getattr(item, 'unit', '')}".strip()
        for item in items
    ]
    return "; ".join(values) if values else "none documented"


def _tool_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "record_id": {"type": "string"},
            "topic": {"type": "string"},
            "patient": {"type": "object"},
            "encounters": {"type": "array", "items": {"type": "object"}},
            "labs": {"type": "array", "items": {"type": "object"}},
            "vitals": {"type": "array", "items": {"type": "object"}},
            "medication_history": {"type": "array", "items": {"type": "object"}},
            "time_series": {"type": "array", "items": {"type": "object"}},
            "documents": {"type": "array", "items": {"type": "object"}},
            "imaging": {"type": "array", "items": {"type": "object"}},
        },
        "required": [
            "record_id",
            "topic",
            "patient",
            "encounters",
            "labs",
            "vitals",
            "medication_history",
            "time_series",
            "documents",
            "imaging",
        ],
        "additionalProperties": True,
    }


def _entry(resource: dict[str, Any]) -> dict[str, Any]:
    return {"resource": resource}


def _patient_reference(record: SyntheticRecord) -> dict[str, str]:
    return {"reference": f"Patient/{record.patient.patient_id}"}


def _patient_resource(record: SyntheticRecord) -> dict[str, Any]:
    gender = record.patient.sex.lower()
    if gender not in {"male", "female", "other", "unknown"}:
        gender = "unknown"
    return {
        "resourceType": "Patient",
        "id": record.patient.patient_id,
        "gender": gender,
        "extension": [
            {
                "url": "https://casecrawler.dev/fhir/StructureDefinition/synthetic-age",
                "valueInteger": record.patient.age,
            }
        ],
        "identifier": [
            {
                "system": "https://casecrawler.dev/synthetic-patients",
                "value": record.patient.patient_id,
            }
        ],
    }


def _encounter_resource(record: SyntheticRecord, encounter) -> dict[str, Any]:
    resource: dict[str, Any] = {
        "resourceType": "Encounter",
        "id": encounter.encounter_id,
        "status": "finished",
        "class": {
            "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
            "code": encounter.setting,
            "display": encounter.setting,
        },
        "subject": _patient_reference(record),
        "reasonCode": [{"text": encounter.reason}],
        "period": {"start": encounter.start},
    }
    if encounter.end:
        resource["period"]["end"] = encounter.end
    if encounter.diagnoses:
        resource["diagnosis"] = [
            {"condition": {"display": diagnosis.display}}
            for diagnosis in encounter.diagnoses
        ]
    return resource


def _lab_observation_resource(record: SyntheticRecord, lab) -> dict[str, Any]:
    resource = _observation_base(
        record,
        resource_id=f"{record.record_id}-lab-{_slug(lab.name)}-{_slug(lab.effective_time)}",
        category_code="laboratory",
        category_display="Laboratory",
        name=lab.name,
        effective_time=lab.effective_time,
    )
    if lab.loinc:
        resource["code"] = {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": lab.loinc,
                    "display": lab.name,
                }
            ],
            "text": lab.name,
        }
    _attach_observation_value(resource, lab.value, lab.unit)
    if lab.reference_low is not None or lab.reference_high is not None:
        reference_range: dict[str, Any] = {}
        if lab.reference_low is not None:
            reference_range["low"] = {"value": lab.reference_low, "unit": lab.unit}
        if lab.reference_high is not None:
            reference_range["high"] = {"value": lab.reference_high, "unit": lab.unit}
        resource["referenceRange"] = [reference_range]
    if lab.flag:
        resource["interpretation"] = [{"text": lab.flag}]
    if lab.specimen:
        resource["specimen"] = {"display": lab.specimen}
    return resource


def _vital_observation_resource(record: SyntheticRecord, vital) -> dict[str, Any]:
    resource = _observation_base(
        record,
        resource_id=f"{record.record_id}-vital-{_slug(vital.name)}-{_slug(vital.effective_time)}",
        category_code="vital-signs",
        category_display="Vital Signs",
        name=vital.name,
        effective_time=vital.effective_time,
    )
    _attach_observation_value(resource, vital.value, vital.unit)
    return resource


def _time_series_observation_resource(record: SyntheticRecord, channel) -> dict[str, Any]:
    return {
        "resourceType": "Observation",
        "id": f"{record.record_id}-timeseries-{_slug(channel.name)}",
        "status": "final",
        "category": [{"coding": [{"code": "survey", "display": "Time Series"}]}],
        "code": {"text": channel.name},
        "subject": _patient_reference(record),
        "component": [
            {
                "code": {"text": f"{point.timestamp}:{name}"},
                "valueQuantity": {"value": observed_value, "unit": channel.unit},
            }
            for point in channel.points
            for name, observed_value in point.values.items()
        ],
    }


def _medication_statement_resource(record: SyntheticRecord, medication) -> dict[str, Any]:
    resource: dict[str, Any] = {
        "resourceType": "MedicationStatement",
        "id": f"{record.record_id}-med-{_slug(medication.name)}",
        "status": medication.status,
        "subject": _patient_reference(record),
        "medicationCodeableConcept": {"text": medication.name},
    }
    if medication.rxnorm:
        resource["medicationCodeableConcept"] = {
            "coding": [
                {
                    "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                    "code": medication.rxnorm,
                    "display": medication.name,
                }
            ],
            "text": medication.name,
        }
    dosage_bits = [medication.dose, medication.route, medication.frequency]
    dosage_text = " ".join(bit for bit in dosage_bits if bit)
    if dosage_text:
        resource["dosage"] = [{"text": dosage_text}]
    if medication.start or medication.end:
        resource["effectivePeriod"] = {}
        if medication.start:
            resource["effectivePeriod"]["start"] = medication.start
        if medication.end:
            resource["effectivePeriod"]["end"] = medication.end
    return resource


def _document_reference_resource(record: SyntheticRecord, document) -> dict[str, Any]:
    clean_text = document.clean_text.encode("utf-8")
    resource: dict[str, Any] = {
        "resourceType": "DocumentReference",
        "id": document.document_id,
        "status": "current",
        "type": {"text": document.note_type},
        "subject": _patient_reference(record),
        "date": document.timestamp,
        "author": [{"display": document.author_role}],
        "content": [
            {
                "attachment": {
                    "contentType": "text/plain",
                    "data": base64.b64encode(clean_text).decode("ascii"),
                    "title": document.note_type,
                }
            }
        ],
    }
    if document.messy_text:
        resource["description"] = (
            "Includes clean_text and messy_text in synthetic source record."
        )
    return resource


def _imaging_report_resource(record: SyntheticRecord, asset) -> dict[str, Any]:
    resource: dict[str, Any] = {
        "resourceType": "DiagnosticReport",
        "id": asset.image_id,
        "status": "final",
        "code": {"text": f"{asset.modality} {asset.body_region} synthetic imaging report"},
        "subject": _patient_reference(record),
        "conclusion": asset.report_text,
        "media": [
            {
                "comment": asset.prompt,
                "link": {"display": asset.file_path or asset.image_id},
            }
        ],
    }
    if asset.labels:
        resource["result"] = [{"display": label.display} for label in asset.labels]
    return resource


def _provenance_resource(record: SyntheticRecord) -> dict[str, Any]:
    return {
        "resourceType": "Provenance",
        "id": f"{record.record_id}-provenance",
        "recorded": record.provenance.created_at,
        "target": [{"reference": f"Bundle/{record.record_id}"}],
        "agent": [
            {
                "type": {"text": "synthetic-data-generator"},
                "who": {
                    "display": record.provenance.generator,
                },
            }
        ],
        "entity": [
            {
                "role": "source",
                "what": {"display": json.dumps(source_ref, sort_keys=True)},
            }
            for source_ref in record.provenance.source_refs
        ],
    }


def _observation_base(
    record: SyntheticRecord,
    *,
    resource_id: str,
    category_code: str,
    category_display: str,
    name: str,
    effective_time: str,
) -> dict[str, Any]:
    return {
        "resourceType": "Observation",
        "id": resource_id,
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": category_code,
                        "display": category_display,
                    }
                ]
            }
        ],
        "code": {"text": name},
        "subject": _patient_reference(record),
        "effectiveDateTime": effective_time,
    }


def _attach_observation_value(resource: dict[str, Any], value: float | str, unit: str) -> None:
    if isinstance(value, (int, float)):
        resource["valueQuantity"] = {"value": value, "unit": unit}
    else:
        resource["valueString"] = value


def _slug(value: str) -> str:
    normalized = "".join(
        character.lower() if character.isalnum() else "-" for character in value
    )
    return "-".join(part for part in normalized.split("-") if part)[:80] or "value"
