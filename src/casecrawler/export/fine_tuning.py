from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import zipfile
from collections.abc import Iterable
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from casecrawler.imaging.file_metadata import image_file_metadata
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
            "Extract the complete structured synthetic clinical record, including "
            "patient, encounters, diagnoses, procedures, abnormal labs, vital sign "
            "abnormalities, medications, time series, documents, imaging, and "
            f"provenance:\n\n{record_text}"
        )
        assistant = {
            **_clinical_context(record),
            "provenance": record.provenance.model_dump(),
            "synthetic": True,
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


def export_note_fact_sft_records(record: SyntheticRecord) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    record_context = {
        "topic": record.topic,
        "patient": record.patient.model_dump(),
        "encounters": [encounter.model_dump() for encounter in record.encounters],
        "diagnoses": _diagnoses(record),
        "procedures": _procedures(record),
        "labs": [lab.model_dump() for lab in record.labs],
        "vitals": [vital.model_dump() for vital in record.vitals],
        "medication_history": [med.model_dump() for med in record.medication_history],
        "imaging_labels": [
            {
                "image_id": asset.image_id,
                "modality": asset.modality,
                "body_region": asset.body_region,
                "labels": [label.model_dump() for label in asset.labels],
                "report_text": asset.report_text,
            }
            for asset in record.imaging
        ],
        "provenance": record.provenance.model_dump(),
        "synthetic": True,
    }
    for document in record.documents:
        target = {
            "document": {
                "document_id": document.document_id,
                "note_type": document.note_type,
                "author_role": document.author_role,
                "timestamp": document.timestamp,
                "extracted_facts": document.extracted_facts,
            },
            "record_context": record_context,
        }
        examples.append(
            {
                "record_id": record.record_id,
                "dataset_id": record.dataset_id,
                "document_id": document.document_id,
                "task": "extract_clinical_facts_from_note",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Extract structured clinical facts from synthetic "
                            "healthcare documentation. Preserve uncertainty and "
                            "synthetic provenance."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Note type: {document.note_type}\n"
                            f"Author role: {document.author_role}\n"
                            f"Timestamp: {document.timestamp}\n\n"
                            f"{document.messy_text or document.clean_text}"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(target, sort_keys=True),
                    },
                ],
                "metadata": {
                    **_metadata(record),
                    "document_id": document.document_id,
                    "note_type": document.note_type,
                    "author_role": document.author_role,
                    "export_profile": "note_fact_sft_jsonl",
                },
            }
        )
    return examples


def export_clinical_observation_records(record: SyntheticRecord) -> list[dict[str, Any]]:
    """Export observation-level lab and vital interpretation examples."""
    examples: list[dict[str, Any]] = []
    clinical_context = _clinical_context(record)
    note_context = [
        {
            "document_id": document.document_id,
            "note_type": document.note_type,
            "author_role": document.author_role,
            "timestamp": document.timestamp,
            "extracted_facts": document.extracted_facts,
        }
        for document in record.documents
    ]
    for lab in record.labs:
        lab_flag = lab.flag or _numeric_reference_flag(
            lab.value,
            lab.reference_low,
            lab.reference_high,
        )
        examples.append(
            {
                "record_id": record.record_id,
                "dataset_id": record.dataset_id,
                "task": "clinical_lab_observation_interpretation",
                "input": {
                    "patient": record.patient.model_dump(),
                    "encounters": [
                        encounter.model_dump() for encounter in record.encounters
                    ],
                    "observation": lab.model_dump(),
                    "observation_kind": "lab",
                    "notes": note_context,
                    "medication_history": [
                        medication.model_dump()
                        for medication in record.medication_history
                    ],
                },
                "target": {
                    "name": lab.name,
                    "loinc": lab.loinc,
                    "value": lab.value,
                    "unit": lab.unit,
                    "reference_low": lab.reference_low,
                    "reference_high": lab.reference_high,
                    "flag": lab_flag,
                    "effective_time": lab.effective_time,
                    "specimen": lab.specimen,
                    "abnormal": _is_abnormal_lab(lab.value, lab_flag),
                },
                "clinical_context": clinical_context,
                "metadata": {
                    **_metadata(record),
                    "export_profile": "clinical_observation_jsonl",
                    "observation_kind": "lab",
                    "observation_name": lab.name,
                },
            }
        )
    for vital in record.vitals:
        direction = _vital_abnormality(vital.name, vital.value)
        examples.append(
            {
                "record_id": record.record_id,
                "dataset_id": record.dataset_id,
                "task": "clinical_vital_observation_interpretation",
                "input": {
                    "patient": record.patient.model_dump(),
                    "encounters": [
                        encounter.model_dump() for encounter in record.encounters
                    ],
                    "observation": vital.model_dump(),
                    "observation_kind": "vital",
                    "notes": note_context,
                    "medication_history": [
                        medication.model_dump()
                        for medication in record.medication_history
                    ],
                },
                "target": {
                    "name": vital.name,
                    "value": vital.value,
                    "unit": vital.unit,
                    "effective_time": vital.effective_time,
                    "abnormal": direction is not None,
                    "direction": direction,
                },
                "clinical_context": clinical_context,
                "metadata": {
                    **_metadata(record),
                    "export_profile": "clinical_observation_jsonl",
                    "observation_kind": "vital",
                    "observation_name": vital.name,
                },
            }
        )
    return examples


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
    clinical_context = _clinical_context(record)
    images = [_multimodal_image_payload(asset) for asset in record.imaging]
    image_payloads = {image["image_id"]: image for image in images}
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
        "clinical_context": clinical_context,
        "images": images,
        "image_text_pairs": image_text_pairs,
        "supervised_tasks": _multimodal_supervised_tasks(
            image_text_pairs,
            image_payloads,
            clinical_context,
        ),
        "metadata": _metadata(record),
    }


def _multimodal_supervised_tasks(
    image_text_pairs: list[dict[str, Any]],
    image_payloads: dict[str, dict[str, Any]],
    clinical_context: dict[str, Any],
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for pair in image_text_pairs:
        image_payload = image_payloads.get(pair["image_id"], {})
        image_input = {
            "image_id": pair["image_id"],
            "clinical_context": clinical_context,
            "image_metadata": image_payload.get("image_metadata"),
        }
        tasks.extend(
            [
                {
                    "task": "radiology_image_report_alignment",
                    "input": {
                        **image_input,
                        "report_text": pair["text"],
                    },
                    "target": {
                        "is_synthetic": True,
                        "labels": pair["labels"],
                    },
                },
                {
                    "task": "radiology_report_generation",
                    "input": {
                        **image_input,
                        "labels": pair["labels"],
                    },
                    "target": {
                        "report_text": pair["text"],
                        "is_synthetic": True,
                    },
                },
                {
                    "task": "radiology_label_extraction",
                    "input": {
                        **image_input,
                        "report_text": pair["text"],
                    },
                    "target": {
                        "labels": pair["labels"],
                    },
                },
            ]
        )
    return tasks


def export_time_series_records(record: SyntheticRecord) -> list[dict[str, Any]]:
    """Export channel-level clinical time-series forecasting examples."""
    examples: list[dict[str, Any]] = []
    for channel in record.time_series:
        if len(channel.points) > 1:
            input_points = channel.points[:-1]
            target_points = channel.points[-1:]
        else:
            input_points = []
            target_points = channel.points
        examples.append(
            {
                "record_id": record.record_id,
                "dataset_id": record.dataset_id,
                "task": "clinical_time_series_forecasting",
                "channel": {
                    "name": channel.name,
                    "unit": channel.unit,
                    "sampling_rate_hz": channel.sampling_rate_hz,
                    "generation_backend": channel.generation_backend,
                    "point_count": len(channel.points),
                },
                "input": {
                    "patient": record.patient.model_dump(),
                    "encounters": [
                        encounter.model_dump() for encounter in record.encounters
                    ],
                    "points": [point.model_dump() for point in input_points],
                },
                "target": {
                    "points": [point.model_dump() for point in target_points],
                },
                "clinical_context": {
                    "topic": record.topic,
                    "complexity": record.complexity.value,
                    "diagnoses": _diagnoses(record),
                    "procedures": _procedures(record),
                    "labs": [lab.model_dump() for lab in record.labs],
                    "vitals": [vital.model_dump() for vital in record.vitals],
                    "medication_history": [
                        medication.model_dump()
                        for medication in record.medication_history
                    ],
                    "documents": [
                        {
                            "document_id": document.document_id,
                            "note_type": document.note_type,
                            "author_role": document.author_role,
                            "timestamp": document.timestamp,
                            "extracted_facts": document.extracted_facts,
                        }
                        for document in record.documents
                    ],
                },
                "provenance": record.provenance.model_dump(),
                "metadata": {
                    **_metadata(record),
                    "export_profile": "time_series_jsonl",
                    "channel_name": channel.name,
                    "generation_backend": channel.generation_backend,
                },
            }
        )
    return examples


def export_medication_reconciliation_records(
    record: SyntheticRecord,
) -> list[dict[str, Any]]:
    """Export medication-level reconciliation examples."""
    examples: list[dict[str, Any]] = []
    clinical_context = _clinical_context(record)
    note_context = [
        {
            "document_id": document.document_id,
            "note_type": document.note_type,
            "author_role": document.author_role,
            "timestamp": document.timestamp,
            "text": document.messy_text or document.clean_text,
            "extracted_medications": document.extracted_facts.get("medications", []),
            "extracted_medication_details": document.extracted_facts.get(
                "medication_details", []
            ),
        }
        for document in record.documents
    ]
    for medication in record.medication_history:
        examples.append(
            {
                "record_id": record.record_id,
                "dataset_id": record.dataset_id,
                "task": "medication_reconciliation",
                "input": {
                    "patient": record.patient.model_dump(),
                    "encounters": [
                        encounter.model_dump() for encounter in record.encounters
                    ],
                    "labs": [lab.model_dump() for lab in record.labs],
                    "vitals": [vital.model_dump() for vital in record.vitals],
                    "notes": note_context,
                    "candidate_medication": medication.name,
                },
                "target": {
                    "medication": medication.model_dump(),
                    "normalized_name": medication.name,
                    "rxnorm": medication.rxnorm,
                    "dose": medication.dose,
                    "route": medication.route,
                    "frequency": medication.frequency,
                    "status": medication.status,
                    "active": medication.status.lower()
                    not in {"stopped", "inactive", "discontinued", "held"},
                    "period": {
                        "start": medication.start,
                        "end": medication.end,
                    },
                },
                "clinical_context": clinical_context,
                "metadata": {
                    **_metadata(record),
                    "export_profile": "medication_reconciliation_jsonl",
                    "medication_name": medication.name,
                    "rxnorm": medication.rxnorm,
                },
            }
        )
    return examples


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
            image_bytes = path.read_bytes()
            payload["image_base64"] = base64.b64encode(image_bytes).decode("ascii")
            payload["image_metadata"] = image_file_metadata(path)
    return payload


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
    condition_refs = _condition_references(record)
    entries.extend(_entry(condition["resource"]) for condition in condition_refs.values())
    entries.extend(
        _entry(_encounter_resource(record, encounter, condition_refs))
        for encounter in record.encounters
    )
    entries.extend(
        _entry(_procedure_resource(record, encounter, procedure))
        for encounter in record.encounters
        for procedure in encounter.procedures
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


def verify_fhir_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    if bundle.get("resourceType") != "Bundle":
        return {
            "valid": False,
            "issues": [
                {
                    "field": "resourceType",
                    "message": "FHIR export payload is not a Bundle.",
                }
            ],
            "resource_counts": {},
        }
    entries = bundle.get("entry")
    if not isinstance(entries, list) or not entries:
        return {
            "valid": False,
            "issues": [
                {
                    "field": "entry",
                    "message": "FHIR Bundle must contain at least one entry.",
                }
            ],
            "resource_counts": {},
        }
    resources = [
        entry.get("resource")
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("resource"), dict)
    ]
    if len(resources) != len(entries):
        issues.append(
            {
                "field": "entry.resource",
                "message": "Every FHIR Bundle entry must contain a resource object.",
            }
        )
    resource_counts: dict[str, int] = {}
    resource_ids: set[tuple[str, str]] = set()
    duplicate_ids: set[str] = set()
    patient_ids: set[str] = set()
    for resource in resources:
        resource_type = str(resource.get("resourceType", ""))
        resource_counts[resource_type] = resource_counts.get(resource_type, 0) + 1
        resource_id = resource.get("id")
        if not isinstance(resource_id, str) or not resource_id:
            issues.append(
                {
                    "field": f"{resource_type or '<unknown>'}.id",
                    "message": "FHIR resource is missing a non-empty id.",
                }
            )
        else:
            identity = (resource_type, resource_id)
            if identity in resource_ids:
                duplicate_ids.add(f"{resource_type}/{resource_id}")
            resource_ids.add(identity)
        if resource_type == "Patient" and isinstance(resource_id, str):
            patient_ids.add(resource_id)
    if resource_counts.get("Patient", 0) != 1:
        issues.append(
            {
                "field": "Patient",
                "message": "FHIR Bundle must contain exactly one Patient resource.",
            }
        )
    if resource_counts.get("Provenance", 0) < 1:
        issues.append(
            {
                "field": "Provenance",
                "message": "FHIR Bundle must contain synthetic provenance.",
            }
        )
    if duplicate_ids:
        issues.append(
            {
                "field": "entry.resource.id",
                "message": (
                    "FHIR Bundle contains duplicate resource IDs: "
                    f"{', '.join(sorted(duplicate_ids))}."
                ),
            }
        )
    expected_patient_reference = (
        f"Patient/{next(iter(patient_ids))}" if len(patient_ids) == 1 else None
    )
    for resource in resources:
        _verify_fhir_resource(resource, expected_patient_reference, issues)
    return {
        "valid": not issues,
        "issues": issues,
        "resource_counts": dict(sorted(resource_counts.items())),
    }


def verify_fhir_ndjson_export(path: str | Path) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    bundle_count = 0
    resource_counts: dict[str, int] = {}
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(
                {
                    "field": f"line.{line_number}",
                    "message": f"Invalid JSON: {exc}.",
                }
            )
            continue
        if not isinstance(payload, dict):
            issues.append(
                {
                    "field": f"line.{line_number}",
                    "message": "FHIR export line must be a JSON object.",
                }
            )
            continue
        bundle_report = verify_fhir_bundle(payload)
        bundle_count += 1
        for resource_type, count in bundle_report["resource_counts"].items():
            resource_counts[resource_type] = resource_counts.get(resource_type, 0) + count
        for issue in bundle_report["issues"]:
            issues.append(
                {
                    "field": f"line.{line_number}.{issue['field']}",
                    "message": issue["message"],
                }
            )
    if bundle_count == 0:
        issues.append(
            {
                "field": "file",
                "message": "FHIR NDJSON export contains no bundle records.",
            }
        )
    return {
        "valid": not issues,
        "bundle_count": bundle_count,
        "issues": issues,
        "resource_counts": dict(sorted(resource_counts.items())),
    }


def _verify_fhir_resource(
    resource: dict[str, Any],
    expected_patient_reference: str | None,
    issues: list[dict[str, str]],
) -> None:
    resource_type = str(resource.get("resourceType", ""))
    if expected_patient_reference and resource_type not in {"Patient", "Provenance"}:
        subject = resource.get("subject")
        if not isinstance(subject, dict) or subject.get("reference") != expected_patient_reference:
            issues.append(
                {
                    "field": f"{resource_type}.subject",
                    "message": (
                        f"{resource_type} must reference {expected_patient_reference}."
                    ),
                }
            )
    if resource_type == "Observation":
        if resource.get("status") != "final":
            issues.append(
                {
                    "field": "Observation.status",
                    "message": "Observation resources must have final status.",
                }
            )
        if "valueQuantity" not in resource and "valueString" not in resource and "component" not in resource:
            issues.append(
                {
                    "field": "Observation.value",
                    "message": "Observation must include a value or components.",
                }
            )
    if resource_type == "MedicationStatement":
        if not resource.get("medicationCodeableConcept"):
            issues.append(
                {
                    "field": "MedicationStatement.medicationCodeableConcept",
                    "message": "MedicationStatement must include medication coding or text.",
                }
            )
    if resource_type == "DocumentReference":
        content = resource.get("content")
        if not isinstance(content, list) or not content:
            issues.append(
                {
                    "field": "DocumentReference.content",
                    "message": "DocumentReference must include document content.",
                }
            )


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
        "diagnoses_json": json.dumps(_diagnoses(record), sort_keys=True),
        "diagnosis_names_json": json.dumps(
            [diagnosis["display"] for diagnosis in _diagnoses(record)],
            sort_keys=True,
        ),
        "procedures_json": json.dumps(_procedures(record), sort_keys=True),
        "procedure_names_json": json.dumps(
            [procedure["display"] for procedure in _procedures(record)],
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


def export_parquet_bytes(records: Iterable[SyntheticRecord]) -> tuple[bytes, int]:
    """Write records to an in-memory parquet payload for API responses."""
    from casecrawler.integrations.huggingface import require_package

    pandas = require_package("pandas", "parquet")
    require_package("pyarrow", "parquet")
    rows = [export_parquet_record(record) for record in records]
    buffer = BytesIO()
    pandas.DataFrame(rows).to_parquet(buffer, index=False)
    return buffer.getvalue(), len(rows)


def export_record(record: SyntheticRecord, export_format: str | ExportFormat) -> dict[str, Any]:
    resolved_format = ExportFormat(export_format)
    if resolved_format == ExportFormat.SFT_JSONL:
        return export_sft_record(record)
    if resolved_format == ExportFormat.NOTE_FACT_SFT_JSONL:
        return {
            "record_id": record.record_id,
            "dataset_id": record.dataset_id,
            "examples": export_note_fact_sft_records(record),
            "metadata": {**_metadata(record), "export_profile": "note_fact_sft_jsonl"},
        }
    if resolved_format == ExportFormat.CLINICAL_OBSERVATION_JSONL:
        return {
            "record_id": record.record_id,
            "dataset_id": record.dataset_id,
            "examples": export_clinical_observation_records(record),
            "metadata": {
                **_metadata(record),
                "export_profile": "clinical_observation_jsonl",
            },
        }
    if resolved_format == ExportFormat.MEDICATION_RECONCILIATION_JSONL:
        return {
            "record_id": record.record_id,
            "dataset_id": record.dataset_id,
            "examples": export_medication_reconciliation_records(record),
            "metadata": {
                **_metadata(record),
                "export_profile": "medication_reconciliation_jsonl",
            },
        }
    if resolved_format == ExportFormat.CHAT_JSONL:
        return export_chat_record(record)
    if resolved_format == ExportFormat.TOOL_CALL_JSONL:
        return export_tool_call_record(record)
    if resolved_format == ExportFormat.MULTIMODAL_JSONL:
        return export_multimodal_record(record)
    if resolved_format == ExportFormat.TIME_SERIES_JSONL:
        return {
            "record_id": record.record_id,
            "dataset_id": record.dataset_id,
            "examples": export_time_series_records(record),
            "metadata": {**_metadata(record), "export_profile": "time_series_jsonl"},
        }
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


def export_record_payloads(
    record: SyntheticRecord,
    export_format: str | ExportFormat,
) -> list[dict[str, Any]]:
    resolved_format = ExportFormat(export_format)
    if resolved_format == ExportFormat.NOTE_FACT_SFT_JSONL:
        return export_note_fact_sft_records(record)
    if resolved_format == ExportFormat.CLINICAL_OBSERVATION_JSONL:
        return export_clinical_observation_records(record)
    if resolved_format == ExportFormat.MEDICATION_RECONCILIATION_JSONL:
        return export_medication_reconciliation_records(record)
    if resolved_format == ExportFormat.TIME_SERIES_JSONL:
        return export_time_series_records(record)
    return [export_record(record, resolved_format)]


def export_jsonl_split_package(
    records: Iterable[SyntheticRecord],
    output_dir: str | Path,
    export_format: str | ExportFormat,
    *,
    dataset_id: str | None = None,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: str = "casecrawler",
    audit_artifacts: dict[str, str | dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write deterministic train/validation/test JSONL files plus a manifest."""
    resolved_format = ExportFormat(export_format)
    if resolved_format == ExportFormat.PARQUET:
        raise ValueError("Split package export writes JSONL profiles, not parquet.")
    record_list = list(records)
    ratios = _normalized_split_ratios(train_ratio, validation_ratio, test_ratio)
    split_records = _split_records(record_list, ratios=ratios, seed=seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    split_entries = {}
    total_examples = 0
    for split_name, split_items in split_records.items():
        file_path = output_path / f"{split_name}.jsonl"
        example_count = 0
        with file_path.open("w") as f:
            for record in split_items:
                for payload in export_record_payloads(record, resolved_format):
                    f.write(json.dumps(payload, sort_keys=True) + "\n")
                    example_count += 1
        total_examples += example_count
        split_entries[split_name] = {
            "file_path": str(file_path),
            "record_count": len(split_items),
            "example_count": example_count,
            "record_ids": [record.record_id for record in split_items],
        }
    artifact_entries = _write_audit_artifacts(output_path, audit_artifacts or {})
    files = _package_file_metadata(
        {
            **{
                f"{split_name}.jsonl": entry["file_path"]
                for split_name, entry in split_entries.items()
            },
            **artifact_entries,
        }
    )

    manifest = {
        "dataset_id": dataset_id or _dataset_id(record_list),
        "export_format": resolved_format.value,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "ratios": {
            "train": ratios[0],
            "validation": ratios[1],
            "test": ratios[2],
        },
        "record_count": len(record_list),
        "example_count": total_examples,
        "splits": split_entries,
        "audit_artifacts": artifact_entries,
        "files": files,
        "synthetic": True,
    }
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def verify_jsonl_split_package(package_dir: str | Path) -> dict[str, Any]:
    package_path = Path(package_dir)
    if package_path.is_file():
        return _verify_jsonl_split_package_archive(package_path)
    return _verify_jsonl_split_package_dir(package_path)


def _verify_jsonl_split_package_archive(archive_path: Path) -> dict[str, Any]:
    if archive_path.suffix.lower() != ".zip":
        return {
            "package_dir": str(archive_path),
            "valid": False,
            "issues": [
                {
                    "field": "package",
                    "message": "Split package file verification only supports .zip archives.",
                }
            ],
            "checked_files": {},
            "splits": {},
        }
    try:
        with zipfile.ZipFile(archive_path) as archive:
            unsafe_names = [
                name
                for name in archive.namelist()
                if Path(name).is_absolute() or ".." in Path(name).parts
            ]
            if unsafe_names:
                return {
                    "package_dir": str(archive_path),
                    "valid": False,
                    "issues": [
                        {
                            "field": "zip",
                            "message": (
                                "Split package zip contains unsafe paths: "
                                f"{', '.join(sorted(unsafe_names))}."
                            ),
                        }
                    ],
                    "checked_files": {},
                    "splits": {},
                }
            with tempfile.TemporaryDirectory() as temp_dir:
                archive.extractall(temp_dir)
                report = _verify_jsonl_split_package_dir(Path(temp_dir))
    except zipfile.BadZipFile as exc:
        return {
            "package_dir": str(archive_path),
            "valid": False,
            "issues": [
                {
                    "field": "zip",
                    "message": f"Split package zip is invalid: {exc}.",
                }
            ],
            "checked_files": {},
            "splits": {},
        }
    report["package_dir"] = str(archive_path)
    report["archive"] = True
    return report


def _verify_jsonl_split_package_dir(package_path: Path) -> dict[str, Any]:
    manifest_path = package_path / "manifest.json"
    issues: list[dict[str, str]] = []
    if not manifest_path.exists():
        return {
            "package_dir": str(package_path),
            "valid": False,
            "issues": [
                {
                    "field": "manifest.json",
                    "message": "Split package manifest.json is missing.",
                }
            ],
            "checked_files": {},
            "splits": {},
        }
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        return {
            "package_dir": str(package_path),
            "valid": False,
            "issues": [
                {
                    "field": "manifest.json",
                    "message": f"Split package manifest.json is invalid JSON: {exc}.",
                }
            ],
            "checked_files": {},
            "splits": {},
        }

    checked_files = _verify_package_files(package_path, manifest, issues)
    split_summaries = _verify_package_splits(package_path, manifest, issues)
    _verify_package_audit_artifacts(package_path, manifest, issues)
    return {
        "package_dir": str(package_path),
        "dataset_id": manifest.get("dataset_id"),
        "export_format": manifest.get("export_format"),
        "valid": not issues,
        "issues": issues,
        "checked_files": checked_files,
        "splits": split_summaries,
    }


def _verify_package_files(
    package_path: Path,
    manifest: dict[str, Any],
    issues: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    checked_files: dict[str, dict[str, Any]] = {}
    files = manifest.get("files")
    if not isinstance(files, dict):
        issues.append(
            {
                "field": "files",
                "message": "Split package manifest is missing file checksum metadata.",
            }
        )
        return checked_files
    for file_name, metadata in sorted(files.items()):
        if not isinstance(file_name, str) or Path(file_name).name != file_name:
            issues.append(
                {
                    "field": "files",
                    "message": f"Invalid package file name in manifest: {file_name!r}.",
                }
            )
            continue
        if not isinstance(metadata, dict):
            issues.append(
                {
                    "field": f"files.{file_name}",
                    "message": "File metadata must be an object.",
                }
            )
            continue
        file_path = package_path / file_name
        if not file_path.exists():
            issues.append(
                {
                    "field": f"files.{file_name}",
                    "message": f"Package file {file_name} is missing.",
                }
            )
            checked_files[file_name] = {"exists": False}
            continue
        byte_size = file_path.stat().st_size
        sha256 = hashlib.sha256(file_path.read_bytes()).hexdigest()
        expected_byte_size = metadata.get("byte_size")
        expected_sha256 = metadata.get("sha256")
        checked_files[file_name] = {
            "exists": True,
            "byte_size": byte_size,
            "sha256": sha256,
        }
        if isinstance(expected_byte_size, int) and byte_size != expected_byte_size:
            issues.append(
                {
                    "field": f"files.{file_name}.byte_size",
                    "message": (
                        f"Package file {file_name} byte size {byte_size} "
                        f"does not match manifest value {expected_byte_size}."
                    ),
                }
            )
        elif not isinstance(expected_byte_size, int):
            issues.append(
                {
                    "field": f"files.{file_name}.byte_size",
                    "message": f"Package file {file_name} has no integer byte_size.",
                }
            )
        if isinstance(expected_sha256, str) and sha256 != expected_sha256:
            issues.append(
                {
                    "field": f"files.{file_name}.sha256",
                    "message": f"Package file {file_name} checksum does not match manifest.",
                }
            )
        elif not isinstance(expected_sha256, str):
            issues.append(
                {
                    "field": f"files.{file_name}.sha256",
                    "message": f"Package file {file_name} has no sha256 checksum.",
                }
            )
    return checked_files


def _verify_package_splits(
    package_path: Path,
    manifest: dict[str, Any],
    issues: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        issues.append(
            {
                "field": "splits",
                "message": "Split package manifest is missing split metadata.",
            }
        )
        return {}

    summaries: dict[str, dict[str, Any]] = {}
    total_examples = 0
    all_record_ids: set[str] = set()
    duplicate_record_ids: set[str] = set()
    export_format = manifest.get("export_format")
    for split_name in ("train", "validation", "test"):
        split_metadata = splits.get(split_name)
        if not isinstance(split_metadata, dict):
            issues.append(
                {
                    "field": f"splits.{split_name}",
                    "message": f"Split {split_name} metadata is missing.",
                }
            )
            continue
        jsonl_path = package_path / f"{split_name}.jsonl"
        examples, parse_issues = _read_jsonl_examples(jsonl_path)
        for message in parse_issues:
            issues.append({"field": f"{split_name}.jsonl", "message": message})
        if export_format == ExportFormat.FHIR_NDJSON.value:
            _verify_fhir_split_examples(split_name, examples, issues)
        example_count = len(examples)
        record_ids = [
            record_id
            for example in examples
            if isinstance((record_id := example.get("record_id")), str)
        ]
        observed_record_ids = sorted(set(record_ids))
        for record_id in observed_record_ids:
            if record_id in all_record_ids:
                duplicate_record_ids.add(record_id)
            all_record_ids.add(record_id)
        total_examples += example_count
        summaries[split_name] = {
            "example_count": example_count,
            "record_ids": observed_record_ids,
        }
        expected_example_count = split_metadata.get("example_count")
        if (
            isinstance(expected_example_count, int)
            and example_count != expected_example_count
        ):
            issues.append(
                {
                    "field": f"splits.{split_name}.example_count",
                    "message": (
                        f"Split {split_name} has {example_count} examples but "
                        f"manifest declares {expected_example_count}."
                    ),
                }
            )
        elif not isinstance(expected_example_count, int):
            issues.append(
                {
                    "field": f"splits.{split_name}.example_count",
                    "message": f"Split {split_name} has no integer example_count.",
                }
            )
        expected_record_ids = split_metadata.get("record_ids")
        if isinstance(expected_record_ids, list):
            expected = sorted(
                item for item in expected_record_ids if isinstance(item, str)
            )
            if observed_record_ids != expected:
                issues.append(
                    {
                        "field": f"splits.{split_name}.record_ids",
                        "message": (
                            f"Split {split_name} record IDs do not match manifest."
                        ),
                    }
                )
        else:
            issues.append(
                {
                    "field": f"splits.{split_name}.record_ids",
                    "message": f"Split {split_name} has no record_ids list.",
                }
            )

    expected_total_examples = manifest.get("example_count")
    if isinstance(expected_total_examples, int) and total_examples != expected_total_examples:
        issues.append(
            {
                "field": "example_count",
                "message": (
                    f"Package has {total_examples} examples but manifest declares "
                    f"{expected_total_examples}."
                ),
            }
        )
    elif not isinstance(expected_total_examples, int):
        issues.append(
            {
                "field": "example_count",
                "message": "Split package manifest has no integer example_count.",
            }
        )
    if duplicate_record_ids:
        issues.append(
            {
                "field": "splits.record_ids",
                "message": (
                    "Record IDs appear in multiple splits: "
                    f"{', '.join(sorted(duplicate_record_ids))}."
                ),
            }
        )
    return summaries


def _verify_fhir_split_examples(
    split_name: str,
    examples: list[dict[str, Any]],
    issues: list[dict[str, str]],
) -> None:
    for index, example in enumerate(examples, start=1):
        report = verify_fhir_bundle(example)
        for issue in report["issues"]:
            issues.append(
                {
                    "field": f"{split_name}.jsonl.line.{index}.{issue['field']}",
                    "message": issue["message"],
                }
            )


def _read_jsonl_examples(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], [f"{path.name} is missing."]
    examples: list[dict[str, Any]] = []
    issues: list[str] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(f"Line {line_number} is invalid JSON: {exc}.")
            continue
        if not isinstance(payload, dict):
            issues.append(f"Line {line_number} is not a JSON object.")
            continue
        examples.append(payload)
    return examples, issues


def _verify_package_audit_artifacts(
    package_path: Path,
    manifest: dict[str, Any],
    issues: list[dict[str, str]],
) -> None:
    audit_artifacts = manifest.get("audit_artifacts", {})
    if not isinstance(audit_artifacts, dict):
        issues.append(
            {
                "field": "audit_artifacts",
                "message": "Split package audit_artifacts metadata must be an object.",
            }
        )
        return
    for file_name in sorted(audit_artifacts):
        if not isinstance(file_name, str) or Path(file_name).name != file_name:
            issues.append(
                {
                    "field": "audit_artifacts",
                    "message": f"Invalid audit artifact name: {file_name!r}.",
                }
            )
            continue
        if not (package_path / file_name).exists():
            issues.append(
                {
                    "field": f"audit_artifacts.{file_name}",
                    "message": f"Audit artifact {file_name} is missing.",
                }
            )
            continue
        if file_name == "benchmark_profile.json":
            _verify_benchmark_profile_artifact(
                package_path / file_name,
                manifest,
                issues,
            )


def _verify_benchmark_profile_artifact(
    path: Path,
    manifest: dict[str, Any],
    issues: list[dict[str, str]],
) -> None:
    from casecrawler.validation.benchmark import parse_benchmark_profile_artifact

    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        issues.append(
            {
                "field": "audit_artifacts.benchmark_profile.json",
                "message": f"Benchmark profile artifact is invalid JSON: {exc}.",
            }
        )
        return
    try:
        profile = parse_benchmark_profile_artifact(payload)
    except ValueError as exc:
        issues.append(
            {
                "field": "audit_artifacts.benchmark_profile.json",
                "message": str(exc),
            }
        )
        return
    manifest_dataset_id = manifest.get("dataset_id")
    if isinstance(manifest_dataset_id, str) and profile.dataset_id != manifest_dataset_id:
        issues.append(
            {
                "field": "audit_artifacts.benchmark_profile.json.profile.dataset_id",
                "message": (
                    "Benchmark profile dataset_id "
                    f"{profile.dataset_id!r} does not match package dataset_id "
                    f"{manifest_dataset_id!r}."
                ),
            }
        )


def _package_file_metadata(file_paths: dict[str, str]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for file_name, file_path in sorted(file_paths.items()):
        path = Path(file_path)
        metadata[file_name] = {
            "path": str(path),
            "byte_size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return metadata


def _write_audit_artifacts(
    output_path: Path,
    audit_artifacts: dict[str, str | dict[str, Any]],
) -> dict[str, str]:
    entries: dict[str, str] = {}
    for file_name, content in sorted(audit_artifacts.items()):
        if Path(file_name).name != file_name:
            raise ValueError("Audit artifact names must be plain file names.")
        artifact_path = output_path / file_name
        if isinstance(content, str):
            artifact_path.write_text(content)
        else:
            artifact_path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")
        entries[file_name] = str(artifact_path)
    return entries


def _split_records(
    records: list[SyntheticRecord],
    *,
    ratios: tuple[float, float, float],
    seed: str,
) -> dict[str, list[SyntheticRecord]]:
    ordered = sorted(
        records,
        key=lambda record: _split_sort_key(record.record_id, seed),
    )
    record_count = len(ordered)
    train_count = int(record_count * ratios[0])
    validation_count = int(record_count * ratios[1])
    if record_count > 0 and train_count == 0:
        train_count = 1
    if record_count >= 3 and validation_count == 0 and ratios[1] > 0:
        validation_count = 1
    if train_count + validation_count > record_count:
        validation_count = max(0, record_count - train_count)
    return {
        "train": ordered[:train_count],
        "validation": ordered[train_count : train_count + validation_count],
        "test": ordered[train_count + validation_count :],
    }


def _split_sort_key(record_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).hexdigest()


def _normalized_split_ratios(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[float, float, float]:
    ratios = (float(train_ratio), float(validation_ratio), float(test_ratio))
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative.")
    total = sum(ratios)
    if total <= 0:
        raise ValueError("At least one split ratio must be greater than zero.")
    return tuple(ratio / total for ratio in ratios)


def _dataset_id(records: list[SyntheticRecord]) -> str | None:
    dataset_ids = {record.dataset_id for record in records}
    if len(dataset_ids) == 1:
        return next(iter(dataset_ids))
    return None


def _clinical_context(record: SyntheticRecord) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "topic": record.topic,
        "patient": record.patient.model_dump(),
        "encounters": [encounter.model_dump() for encounter in record.encounters],
        "diagnoses": _diagnoses(record),
        "procedures": _procedures(record),
        "labs": [lab.model_dump() for lab in record.labs],
        "vitals": [vital.model_dump() for vital in record.vitals],
        "medication_history": [med.model_dump() for med in record.medication_history],
        "time_series": [channel.model_dump() for channel in record.time_series],
        "documents": [document.model_dump() for document in record.documents],
        "imaging": [asset.model_dump() for asset in record.imaging],
    }


def _diagnoses(record: SyntheticRecord) -> list[dict[str, Any]]:
    return [
        {
            "encounter_id": encounter.encounter_id,
            **diagnosis.model_dump(),
        }
        for encounter in record.encounters
        for diagnosis in encounter.diagnoses
    ]


def _procedures(record: SyntheticRecord) -> list[dict[str, Any]]:
    return [
        {
            "encounter_id": encounter.encounter_id,
            **procedure.model_dump(),
        }
        for encounter in record.encounters
        for procedure in encounter.procedures
    ]


def _metadata(record: SyntheticRecord) -> dict[str, Any]:
    return {
        "topic": record.topic,
        "complexity": record.complexity.value,
        "modalities": [m.value for m in record.modalities],
        "synthetic": True,
    }


def _numeric_reference_flag(
    value: float | str,
    reference_low: float | None,
    reference_high: float | None,
) -> str | None:
    if not isinstance(value, (int, float)):
        return None
    if reference_low is not None and value < reference_low:
        return "L"
    if reference_high is not None and value > reference_high:
        return "H"
    return None


def _is_abnormal_lab(value: float | str, flag: str | None) -> bool:
    if flag:
        return flag.upper() not in {"N", "NORMAL"}
    return False


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
            "diagnoses": {"type": "array", "items": {"type": "object"}},
            "procedures": {"type": "array", "items": {"type": "object"}},
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
            "diagnoses",
            "procedures",
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


def _condition_references(record: SyntheticRecord) -> dict[str, dict[str, Any]]:
    condition_refs: dict[str, dict[str, Any]] = {}
    for encounter in record.encounters:
        for diagnosis in encounter.diagnoses:
            key = _condition_key(diagnosis)
            if key in condition_refs:
                continue
            condition_id = f"{record.record_id}-condition-{_slug(key)}"
            condition_refs[key] = {
                "id": condition_id,
                "resource": _condition_resource(record, diagnosis, condition_id),
            }
    return condition_refs


def _condition_key(diagnosis) -> str:
    return f"{diagnosis.system}:{diagnosis.code}:{diagnosis.display}"


def _condition_resource(
    record: SyntheticRecord,
    diagnosis,
    condition_id: str,
) -> dict[str, Any]:
    return {
        "resourceType": "Condition",
        "id": condition_id,
        "clinicalStatus": {
            "coding": [
                {
                    "system": (
                        "http://terminology.hl7.org/CodeSystem/"
                        "condition-clinical"
                    ),
                    "code": "active",
                    "display": "Active",
                }
            ]
        },
        "code": {
            "coding": [
                {
                    "system": diagnosis.system,
                    "code": diagnosis.code,
                    "display": diagnosis.display,
                }
            ],
            "text": diagnosis.display,
        },
        "subject": _patient_reference(record),
    }


def _encounter_resource(
    record: SyntheticRecord,
    encounter,
    condition_refs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
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
            {
                "condition": {
                    "reference": (
                        f"Condition/{condition_refs[_condition_key(diagnosis)]['id']}"
                    ),
                    "display": diagnosis.display,
                }
            }
            for diagnosis in encounter.diagnoses
        ]
    return resource


def _procedure_resource(record: SyntheticRecord, encounter, procedure) -> dict[str, Any]:
    return {
        "resourceType": "Procedure",
        "id": f"{record.record_id}-procedure-{_slug(procedure.code)}",
        "status": "completed",
        "code": {
            "coding": [
                {
                    "system": procedure.system,
                    "code": procedure.code,
                    "display": procedure.display,
                }
            ],
            "text": procedure.display,
        },
        "subject": _patient_reference(record),
        "encounter": {"reference": f"Encounter/{encounter.encounter_id}"},
        "performedDateTime": encounter.start,
    }


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
    _attach_observation_encounter(record, resource, lab.effective_time)
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
    _attach_observation_encounter(record, resource, vital.effective_time)
    return resource


def _time_series_observation_resource(record: SyntheticRecord, channel) -> dict[str, Any]:
    components = []
    encounter_refs: set[str] = set()
    for point in channel.points:
        encounter_ref = _encounter_reference_for_timestamp(record, point.timestamp)
        if encounter_ref:
            encounter_refs.add(encounter_ref)
        for name, observed_value in point.values.items():
            extensions = [
                {
                    "url": "https://casecrawler.dev/fhir/StructureDefinition/sample-timestamp",
                    "valueDateTime": point.timestamp,
                }
            ]
            if encounter_ref:
                extensions.append(
                    {
                        "url": "https://casecrawler.dev/fhir/StructureDefinition/sample-encounter",
                        "valueReference": {"reference": encounter_ref},
                    }
                )
            components.append(
                {
                    "code": {"text": name},
                    "extension": extensions,
                    "valueQuantity": {"value": observed_value, "unit": channel.unit},
                }
            )
    resource = {
        "resourceType": "Observation",
        "id": f"{record.record_id}-timeseries-{_slug(channel.name)}",
        "status": "final",
        "category": [{"coding": [{"code": "survey", "display": "Time Series"}]}],
        "code": {"text": channel.name},
        "subject": _patient_reference(record),
        "component": components,
    }
    if len(encounter_refs) == 1:
        resource["encounter"] = {"reference": next(iter(encounter_refs))}
    if channel.points:
        resource["effectivePeriod"] = {
            "start": channel.points[0].timestamp,
            "end": channel.points[-1].timestamp,
        }
    if channel.sampling_rate_hz is not None:
        resource["extension"] = [
            {
                "url": "https://casecrawler.dev/fhir/StructureDefinition/sampling-rate-hz",
                "valueDecimal": channel.sampling_rate_hz,
            }
        ]
    return resource


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


def _attach_observation_encounter(
    record: SyntheticRecord,
    resource: dict[str, Any],
    effective_time: str,
) -> None:
    encounter_ref = _encounter_reference_for_timestamp(record, effective_time)
    if encounter_ref:
        resource["encounter"] = {"reference": encounter_ref}


def _encounter_reference_for_timestamp(
    record: SyntheticRecord,
    timestamp: str,
) -> str | None:
    observed_at = _parse_datetime(timestamp)
    if observed_at is None:
        return None
    starts = [_parse_datetime(encounter.start) for encounter in record.encounters]
    intervals = []
    for index, encounter in enumerate(record.encounters):
        start = starts[index]
        if start is None:
            continue
        end = _parse_datetime(encounter.end) if encounter.end else None
        if end is None:
            later_starts = [value for value in starts[index + 1 :] if value is not None]
            end = min(later_starts) if later_starts else None
        intervals.append((encounter.encounter_id, start, end))
    for encounter_id, start, end in intervals:
        if end is None:
            if observed_at >= start:
                return f"Encounter/{encounter_id}"
            continue
        if start <= observed_at <= end:
            return f"Encounter/{encounter_id}"
    return None


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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
