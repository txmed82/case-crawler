from __future__ import annotations

import base64
import hashlib
import json
import shutil
import tempfile
import zipfile
from collections.abc import Iterable
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from casecrawler.imaging.file_metadata import image_file_metadata
from casecrawler.export.release_audit import OBJECTIVE_COVERAGE_KEYS
from casecrawler.models.dataset import ExportFormat
from casecrawler.models.synthetic import SyntheticRecord


REQUIRED_RELEASE_COVERAGE_KEYS = frozenset(
    {
        "approved_records",
        "benchmark_reference",
        "clinical_text_model_policy",
        "discharge_summaries",
        "imaging_model_policy",
        "lab_reports",
        "labs",
        "medication_administration_records",
        "medication_history",
        "messy_clinical_text",
        "modality_alignment_scores",
        "no_blocking_quality_issues",
        "nursing_notes",
        "physician_notes",
        "radiology_images",
        "radiology_reports",
        "records",
        "structured_ehr",
        "task_reference_coverage",
        "time_series",
        "validation_reports",
        "vital_signs_flowsheets",
        "vitals",
    }
)


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
        "metadata": _metadata(record),
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


def export_multimodal_record(
    record: SyntheticRecord,
    *,
    image_package_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    clinical_context = _clinical_context(record)
    image_package_paths = image_package_paths or {}
    images = [
        _multimodal_image_payload(
            asset,
            package_path=image_package_paths.get(asset.image_id),
        )
        for asset in record.imaging
    ]
    image_payloads = {image["image_id"]: image for image in images}
    image_text_pairs = [
        {
            "image_id": asset.image_id,
            **(
                {"package_path": image_package_paths[asset.image_id]}
                if asset.image_id in image_package_paths
                else {}
            ),
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
            "package_path": image_payload.get("package_path"),
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


def export_time_series_records(
    record: SyntheticRecord,
    *,
    time_series_package_paths: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
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
                    **(
                        {"package_path": time_series_package_paths[channel.name]}
                        if time_series_package_paths
                        and channel.name in time_series_package_paths
                        else {}
                    ),
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


def _multimodal_image_payload(
    asset,
    *,
    package_path: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "image_id": asset.image_id,
        "file_path": asset.file_path,
        "package_path": package_path,
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


def export_record(
    record: SyntheticRecord,
    export_format: str | ExportFormat,
    *,
    image_package_paths: dict[str, str] | None = None,
    time_series_package_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
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
        return export_multimodal_record(
            record,
            image_package_paths=image_package_paths,
        )
    if resolved_format == ExportFormat.TIME_SERIES_JSONL:
        return {
            "record_id": record.record_id,
            "dataset_id": record.dataset_id,
            "examples": export_time_series_records(
                record,
                time_series_package_paths=time_series_package_paths,
            ),
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
    *,
    image_package_paths: dict[str, str] | None = None,
    time_series_package_paths: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    resolved_format = ExportFormat(export_format)
    if resolved_format == ExportFormat.NOTE_FACT_SFT_JSONL:
        return export_note_fact_sft_records(record)
    if resolved_format == ExportFormat.CLINICAL_OBSERVATION_JSONL:
        return export_clinical_observation_records(record)
    if resolved_format == ExportFormat.MEDICATION_RECONCILIATION_JSONL:
        return export_medication_reconciliation_records(record)
    if resolved_format == ExportFormat.TIME_SERIES_JSONL:
        return export_time_series_records(
            record,
            time_series_package_paths=time_series_package_paths,
        )
    return [
        export_record(
            record,
            resolved_format,
            image_package_paths=image_package_paths,
            time_series_package_paths=time_series_package_paths,
        )
    ]


def summarize_export_task_coverage(
    records: Iterable[SyntheticRecord],
    export_format: str | ExportFormat,
) -> dict[str, int]:
    """Summarize fine-tuning task coverage for records and export format."""
    coverage: dict[str, int] = {}
    resolved_format = ExportFormat(export_format)
    for record in records:
        for payload in export_record_payloads(record, resolved_format):
            _merge_task_coverage(coverage, _payload_task_coverage(payload))
    return dict(sorted(coverage.items()))


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
    image_artifact_entries, image_artifacts = _copy_image_artifacts(record_list, output_path)
    if resolved_format == ExportFormat.TIME_SERIES_JSONL or _audit_requires_time_series_artifacts(
        audit_artifacts or {}
    ):
        time_series_artifact_entries, time_series_artifacts = (
            _write_time_series_artifacts(record_list, output_path)
        )
    else:
        time_series_artifact_entries, time_series_artifacts = {}, {}

    split_entries = {}
    total_examples = 0
    total_task_coverage: dict[str, int] = {}
    for split_name, split_items in split_records.items():
        file_path = output_path / f"{split_name}.jsonl"
        example_count = 0
        split_task_coverage: dict[str, int] = {}
        with file_path.open("w") as f:
            for record in split_items:
                for payload in export_record_payloads(
                    record,
                    resolved_format,
                    image_package_paths=_record_image_package_paths(
                        record,
                        image_artifacts,
                    ),
                    time_series_package_paths=_record_time_series_package_paths(
                        record,
                        time_series_artifacts,
                    ),
                ):
                    f.write(json.dumps(payload, sort_keys=True) + "\n")
                    example_count += 1
                    _merge_task_coverage(
                        split_task_coverage,
                        _payload_task_coverage(payload),
                    )
        total_examples += example_count
        _merge_task_coverage(total_task_coverage, split_task_coverage)
        split_entries[split_name] = {
            "file_path": str(file_path),
            "record_count": len(split_items),
            "example_count": example_count,
            "task_coverage": split_task_coverage,
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
            **image_artifact_entries,
            **time_series_artifact_entries,
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
        "task_coverage": total_task_coverage,
        "splits": split_entries,
        "audit_artifacts": artifact_entries,
        "image_artifacts": image_artifacts,
        "time_series_artifacts": time_series_artifacts,
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
    _verify_multimodal_image_package_paths(manifest, split_summaries, issues)
    _verify_time_series_package_paths(manifest, split_summaries, issues)
    _verify_package_audit_artifacts(package_path, manifest, issues)
    quality_report = _split_package_quality_report_summary(package_path)
    _verify_release_training_row_metadata(
        manifest,
        split_summaries,
        quality_report,
        issues,
    )
    _verify_package_task_coverage(manifest, split_summaries, quality_report, issues)
    _verify_package_image_artifacts(manifest, quality_report, issues)
    _verify_package_time_series_artifacts(manifest, quality_report, issues)
    return {
        "package_dir": str(package_path),
        "dataset_id": manifest.get("dataset_id"),
        "export_format": manifest.get("export_format"),
        "valid": not issues,
        "issues": issues,
        "checked_files": checked_files,
        "splits": split_summaries,
        "quality_report": quality_report,
    }


def _verify_package_time_series_artifacts(
    manifest: dict[str, Any],
    quality_report: dict[str, Any] | None,
    issues: list[dict[str, str]],
) -> None:
    artifacts = manifest.get("time_series_artifacts")
    files = manifest.get("files")
    release_ready = (
        isinstance(quality_report, dict)
        and quality_report.get("multimodal_release_ready") is True
    )
    coverage = (
        quality_report.get("core_artifact_coverage")
        if isinstance(quality_report, dict)
        else None
    )
    release_requires_time_series = (
        release_ready
        and isinstance(coverage, dict)
        and coverage.get("time_series") is True
    )
    if artifacts is None:
        if release_requires_time_series:
            issues.append(
                {
                    "field": "time_series_artifacts",
                    "message": (
                        "Release-ready multimodal package has no time-series "
                        "artifacts."
                    ),
                }
            )
        return
    if not isinstance(artifacts, dict):
        issues.append(
            {
                "field": "time_series_artifacts",
                "message": "Split package time_series_artifacts must be an object.",
            }
        )
        return
    if release_requires_time_series and not artifacts:
        issues.append(
            {
                "field": "time_series_artifacts",
                "message": (
                    "Release-ready multimodal package has no time-series artifacts."
                ),
            }
        )
    for key, artifact in sorted(artifacts.items()):
        if not isinstance(key, str) or not isinstance(artifact, dict):
            issues.append(
                {
                    "field": "time_series_artifacts",
                    "message": (
                        "Each time-series artifact entry must be an object keyed "
                        "by string."
                    ),
                }
            )
            continue
        package_path = artifact.get("package_path")
        if not isinstance(package_path, str) or not _is_safe_package_path(package_path):
            issues.append(
                {
                    "field": f"time_series_artifacts.{key}.package_path",
                    "message": (
                        "Time-series artifact package_path must be a safe relative "
                        "path."
                    ),
                }
            )
            continue
        if not package_path.startswith("time_series/"):
            issues.append(
                {
                    "field": f"time_series_artifacts.{key}.package_path",
                    "message": (
                        "Time-series artifact package_path must be under "
                        "time_series/."
                    ),
                }
            )
        if not isinstance(files, dict) or package_path not in files:
            issues.append(
                {
                    "field": f"time_series_artifacts.{key}.package_path",
                    "message": (
                        "Time-series artifact package_path is missing from "
                        "manifest files."
                    ),
                }
            )
        if release_ready:
            _verify_release_time_series_artifact_metadata(key, artifact, issues)


def _verify_release_time_series_artifact_metadata(
    key: str,
    artifact: dict[str, Any],
    issues: list[dict[str, str]],
) -> None:
    for field in ("record_id", "channel_name", "unit", "generation_backend"):
        if not isinstance(artifact.get(field), str) or not artifact[field].strip():
            issues.append(
                {
                    "field": f"time_series_artifacts.{key}.{field}",
                    "message": (
                        "Release-ready time-series artifact is missing required "
                        f"metadata field {field}."
                    ),
                }
            )
    expected_record_id, expected_channel_name = _time_series_artifact_key_parts(key)
    if expected_record_id is not None and artifact.get("record_id") != expected_record_id:
        issues.append(
            {
                "field": f"time_series_artifacts.{key}.record_id",
                "message": (
                    "Release-ready time-series artifact record_id does not match "
                    "its manifest key."
                ),
            }
        )
    if (
        expected_channel_name is not None
        and artifact.get("channel_name") != expected_channel_name
    ):
        issues.append(
            {
                "field": f"time_series_artifacts.{key}.channel_name",
                "message": (
                    "Release-ready time-series artifact channel_name does not match "
                    "its manifest key."
                ),
            }
        )
    if not isinstance(artifact.get("point_count"), int) or artifact["point_count"] < 1:
        issues.append(
            {
                "field": f"time_series_artifacts.{key}.point_count",
                "message": (
                    "Release-ready time-series artifact point_count must be a "
                    "positive integer."
                ),
            }
        )
    sampling_rate = artifact.get("sampling_rate_hz")
    if sampling_rate is not None and not isinstance(sampling_rate, (int, float)):
        issues.append(
            {
                "field": f"time_series_artifacts.{key}.sampling_rate_hz",
                "message": (
                    "Release-ready time-series artifact sampling_rate_hz must be "
                    "numeric when present."
                ),
            }
        )
    policy = artifact.get("time_series_model_policy")
    backend = artifact.get("generation_backend")
    if policy is None:
        if isinstance(backend, str) and backend.startswith("external:"):
            issues.append(
                {
                    "field": f"time_series_artifacts.{key}.time_series_model_policy",
                    "message": (
                        "Release-ready external time-series artifact is missing "
                        "time-series model policy metadata."
                    ),
                }
            )
        return
    if not isinstance(policy, dict):
        issues.append(
            {
                "field": f"time_series_artifacts.{key}.time_series_model_policy",
                "message": "Time-series model policy metadata must be an object.",
            }
        )
        return
    for field in ("profile", "license", "use_policy"):
        if not isinstance(policy.get(field), str) or not policy[field].strip():
            issues.append(
                {
                    "field": (
                        f"time_series_artifacts.{key}."
                        f"time_series_model_policy.{field}"
                    ),
                    "message": (
                        "Release-ready time-series artifact time-series model "
                        f"policy is missing {field}."
                    ),
                }
            )


def _time_series_artifact_key_parts(key: str) -> tuple[str | None, str | None]:
    if ":" not in key:
        return None, None
    record_id, channel_name = key.split(":", 1)
    if not record_id or not channel_name:
        return None, None
    return record_id, channel_name


def _verify_package_image_artifacts(
    manifest: dict[str, Any],
    quality_report: dict[str, Any] | None,
    issues: list[dict[str, str]],
) -> None:
    image_artifacts = manifest.get("image_artifacts")
    files = manifest.get("files")
    release_ready = (
        isinstance(quality_report, dict)
        and quality_report.get("multimodal_release_ready") is True
    )
    coverage = (
        quality_report.get("core_artifact_coverage")
        if isinstance(quality_report, dict)
        else None
    )
    release_requires_images = (
        release_ready
        and isinstance(coverage, dict)
        and coverage.get("radiology_images") is True
    )
    if image_artifacts is None:
        if release_requires_images:
            issues.append(
                {
                    "field": "image_artifacts",
                    "message": "Release-ready multimodal package has no image artifacts.",
                }
            )
        return
    if not isinstance(image_artifacts, dict):
        issues.append(
            {
                "field": "image_artifacts",
                "message": "Split package image_artifacts must be an object.",
            }
        )
        return
    if release_requires_images and not image_artifacts:
        issues.append(
            {
                "field": "image_artifacts",
                "message": "Release-ready multimodal package has no image artifacts.",
            }
        )
    for key, artifact in sorted(image_artifacts.items()):
        if not isinstance(key, str) or not isinstance(artifact, dict):
            issues.append(
                {
                    "field": "image_artifacts",
                    "message": "Each image artifact entry must be an object keyed by string.",
                }
            )
            continue
        package_path = artifact.get("package_path")
        if not isinstance(package_path, str) or not _is_safe_package_path(package_path):
            issues.append(
                {
                    "field": f"image_artifacts.{key}.package_path",
                    "message": "Image artifact package_path must be a safe relative path.",
                }
            )
            continue
        if not package_path.startswith("images/"):
            issues.append(
                {
                    "field": f"image_artifacts.{key}.package_path",
                    "message": "Image artifact package_path must be under images/.",
                }
            )
        if not isinstance(files, dict) or package_path not in files:
            issues.append(
                {
                    "field": f"image_artifacts.{key}.package_path",
                    "message": "Image artifact package_path is missing from manifest files.",
                }
            )
        if release_ready:
            _verify_release_image_artifact_metadata(key, artifact, issues)


def _verify_release_image_artifact_metadata(
    key: str,
    artifact: dict[str, Any],
    issues: list[dict[str, str]],
) -> None:
    required_strings = (
        "record_id",
        "image_id",
        "modality",
        "body_region",
        "generation_backend",
        "prompt",
        "report_text",
    )
    for field in required_strings:
        if not isinstance(artifact.get(field), str) or not artifact[field].strip():
            issues.append(
                {
                    "field": f"image_artifacts.{key}.{field}",
                    "message": (
                        "Release-ready image artifact is missing required "
                        f"metadata field {field}."
                    ),
                }
            )
    expected_record_id, expected_image_id = _image_artifact_key_parts(key)
    if expected_record_id is not None and artifact.get("record_id") != expected_record_id:
        issues.append(
            {
                "field": f"image_artifacts.{key}.record_id",
                "message": (
                    "Release-ready image artifact record_id does not match "
                    "its manifest key."
                ),
            }
        )
    if expected_image_id is not None and artifact.get("image_id") != expected_image_id:
        issues.append(
            {
                "field": f"image_artifacts.{key}.image_id",
                "message": (
                    "Release-ready image artifact image_id does not match "
                    "its manifest key."
                ),
            }
        )
    labels = artifact.get("labels")
    if not isinstance(labels, list) or not all(
        isinstance(label, dict) and _coded_label_complete(label)
        for label in labels
    ):
        issues.append(
            {
                "field": f"image_artifacts.{key}.labels",
                "message": (
                    "Release-ready image artifact labels must be a list of "
                    "coded label objects."
                ),
            }
        )
    policy = artifact.get("imaging_model_policy")
    if not isinstance(policy, dict):
        issues.append(
            {
                "field": f"image_artifacts.{key}.imaging_model_policy",
                "message": (
                    "Release-ready image artifact is missing imaging model "
                    "policy metadata."
                ),
            }
        )
    else:
        for field in ("profile", "model_id", "license", "use_policy"):
            if not isinstance(policy.get(field), str) or not policy[field].strip():
                issues.append(
                    {
                        "field": f"image_artifacts.{key}.imaging_model_policy.{field}",
                        "message": (
                            "Release-ready image artifact imaging model policy "
                            f"is missing {field}."
                        ),
                    }
                )
    validator_policy = artifact.get("image_validator_policy")
    if not isinstance(validator_policy, dict):
        issues.append(
            {
                "field": f"image_artifacts.{key}.image_validator_policy",
                "message": (
                    "Release-ready image artifact is missing image validator "
                    "policy metadata."
                ),
            }
        )
    else:
        for field in ("profile", "backend", "license", "use_policy"):
            if (
                not isinstance(validator_policy.get(field), str)
                or not validator_policy[field].strip()
            ):
                issues.append(
                    {
                        "field": f"image_artifacts.{key}.image_validator_policy.{field}",
                        "message": (
                            "Release-ready image artifact image validator "
                            f"policy is missing {field}."
                        ),
                    }
                )


def _image_artifact_key_parts(key: str) -> tuple[str | None, str | None]:
    if ":" not in key:
        return None, None
    record_id, image_id = key.split(":", 1)
    if not record_id or not image_id:
        return None, None
    return record_id, image_id


def _coded_label_complete(label: dict[str, Any]) -> bool:
    return all(
        isinstance(label.get(field), str) and label[field].strip()
        for field in ("system", "code", "display")
    )


def _split_package_quality_report_summary(package_path: Path) -> dict[str, Any] | None:
    path = package_path / "quality_report.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return {
        "export_ready": payload.get("export_ready"),
        "multimodal_release_ready": payload.get("multimodal_release_ready"),
        "multimodal_release_missing": payload.get("multimodal_release_missing"),
        "core_artifact_coverage": payload.get("core_artifact_coverage"),
        "clinical_text_model_policy_counts": payload.get(
            "clinical_text_model_policy_counts"
        ),
        "imaging_model_policy_counts": payload.get("imaging_model_policy_counts"),
        "image_validator_policy_counts": payload.get("image_validator_policy_counts"),
        "time_series_model_policy_counts": payload.get(
            "time_series_model_policy_counts"
        ),
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
        if not isinstance(file_name, str) or not _is_safe_package_path(file_name):
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
        task_coverage = _examples_task_coverage(examples)
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
            "examples": examples,
            "example_count": example_count,
            "task_coverage": task_coverage,
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
        expected_task_coverage = split_metadata.get("task_coverage")
        if expected_task_coverage != task_coverage:
            issues.append(
                {
                    "field": f"splits.{split_name}.task_coverage",
                    "message": (
                        f"Split {split_name} task coverage does not match JSONL "
                        "payloads."
                    ),
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


def _verify_release_training_row_metadata(
    manifest: dict[str, Any],
    split_summaries: dict[str, dict[str, Any]],
    quality_report: dict[str, Any] | None,
    issues: list[dict[str, str]],
) -> None:
    if not isinstance(quality_report, dict) or quality_report.get("export_ready") is not True:
        return
    if manifest.get("export_format") in {
        ExportFormat.FHIR_NDJSON.value,
        ExportFormat.RAW_JSONL.value,
    }:
        return
    required_policies = _required_training_row_policy_fields(quality_report)
    for split_name, summary in split_summaries.items():
        examples = summary.get("examples")
        if not isinstance(examples, list):
            continue
        for line_index, example in enumerate(examples, start=1):
            for path, payload in _training_payloads_with_metadata(
                example,
                f"{split_name}.jsonl.line.{line_index}",
            ):
                _verify_training_payload_metadata(path, payload, required_policies, issues)


def _required_training_row_policy_fields(
    quality_report: dict[str, Any],
) -> tuple[str, ...]:
    required: list[str] = []
    policy_count_fields = {
        "clinical_text_model_policy_counts": "clinical_text_model_policy",
        "imaging_model_policy_counts": "imaging_model_policy",
        "image_validator_policy_counts": "image_validator_policy",
        "time_series_model_policy_counts": "time_series_model_policy",
    }
    for counts_field, metadata_field in policy_count_fields.items():
        counts = quality_report.get(counts_field)
        if isinstance(counts, dict) and counts:
            required.append(metadata_field)
    return tuple(required)


def _training_payloads_with_metadata(
    payload: dict[str, Any],
    path: str,
) -> list[tuple[str, dict[str, Any]]]:
    payloads = [(path, payload)]
    for index, nested in enumerate(_dict_items(payload.get("examples"))):
        payloads.extend(
            _training_payloads_with_metadata(nested, f"{path}.examples.{index}")
        )
    return payloads


def _verify_training_payload_metadata(
    path: str,
    payload: dict[str, Any],
    required_policies: tuple[str, ...],
    issues: list[dict[str, str]],
) -> None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        issues.append(
            {
                "field": f"{path}.metadata",
                "message": (
                    "Release-ready training row is missing metadata needed for "
                    "fine-tuning auditability."
                ),
            }
        )
        return
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        issues.append(
            {
                "field": f"{path}.metadata.provenance",
                "message": "Release-ready training row is missing provenance metadata.",
            }
        )
    else:
        for field in ("generator", "created_at"):
            if not isinstance(provenance.get(field), str) or not provenance[field].strip():
                issues.append(
                    {
                        "field": f"{path}.metadata.provenance.{field}",
                        "message": (
                            "Release-ready training row provenance is missing "
                            f"{field}."
                        ),
                    }
                )
    for policy_field in required_policies:
        policy = metadata.get(policy_field)
        if not isinstance(policy, dict):
            issues.append(
                {
                    "field": f"{path}.metadata.{policy_field}",
                    "message": (
                        "Release-ready training row is missing model policy "
                        f"metadata field {policy_field}."
                    ),
                }
            )
            continue
        for field in ("use_policy",):
            if not isinstance(policy.get(field), str) or not policy[field].strip():
                issues.append(
                    {
                        "field": f"{path}.metadata.{policy_field}.{field}",
                        "message": (
                            "Release-ready training row model policy metadata "
                            f"is missing {field}."
                        ),
                    }
                )


def _verify_package_task_coverage(
    manifest: dict[str, Any],
    split_summaries: dict[str, dict[str, Any]],
    quality_report: dict[str, Any] | None,
    issues: list[dict[str, str]],
) -> None:
    observed: dict[str, int] = {}
    for summary in split_summaries.values():
        coverage = summary.get("task_coverage")
        if isinstance(coverage, dict):
            _merge_task_coverage(observed, coverage)
    if manifest.get("task_coverage") != observed:
        issues.append(
            {
                "field": "task_coverage",
                "message": "Package task coverage does not match split JSONL payloads.",
            }
        )
    coverage = (
        quality_report.get("core_artifact_coverage")
        if isinstance(quality_report, dict)
        else None
    )
    release_ready = (
        isinstance(quality_report, dict)
        and quality_report.get("multimodal_release_ready") is True
    )
    if (
        release_ready
        and isinstance(coverage, dict)
        and coverage.get("task_reference_coverage") is True
        and not observed
    ):
        issues.append(
            {
                "field": "task_coverage",
                "message": "Release-ready package has no fine-tuning task coverage.",
            }
        )


def _verify_multimodal_image_package_paths(
    manifest: dict[str, Any],
    split_summaries: dict[str, dict[str, Any]],
    issues: list[dict[str, str]],
) -> None:
    if manifest.get("export_format") != ExportFormat.MULTIMODAL_JSONL.value:
        return
    image_artifacts = manifest.get("image_artifacts")
    files = manifest.get("files")
    declared_paths = {
        artifact.get("package_path")
        for artifact in image_artifacts.values()
        if isinstance(artifact, dict) and isinstance(artifact.get("package_path"), str)
    } if isinstance(image_artifacts, dict) else set()
    file_paths = set(files) if isinstance(files, dict) else set()
    for split_name, summary in split_summaries.items():
        examples = summary.get("examples")
        if not isinstance(examples, list):
            continue
        for index, example in enumerate(examples, start=1):
            for field, path in _multimodal_payload_package_paths(example):
                if not _is_safe_package_path(path):
                    issues.append(
                        {
                            "field": f"{split_name}.jsonl.line.{index}.{field}",
                            "message": "Multimodal image package_path is not safe.",
                        }
                    )
                    continue
                if path not in declared_paths:
                    issues.append(
                        {
                            "field": f"{split_name}.jsonl.line.{index}.{field}",
                            "message": (
                                "Multimodal image package_path is not declared in "
                                "manifest image_artifacts."
                            ),
                        }
                    )
                if path not in file_paths:
                    issues.append(
                        {
                            "field": f"{split_name}.jsonl.line.{index}.{field}",
                            "message": (
                                "Multimodal image package_path is missing from "
                                "manifest files."
                            ),
                        }
                    )


def _verify_time_series_package_paths(
    manifest: dict[str, Any],
    split_summaries: dict[str, dict[str, Any]],
    issues: list[dict[str, str]],
) -> None:
    if manifest.get("export_format") != ExportFormat.TIME_SERIES_JSONL.value:
        return
    artifacts = manifest.get("time_series_artifacts")
    files = manifest.get("files")
    declared_paths = {
        artifact.get("package_path")
        for artifact in artifacts.values()
        if isinstance(artifact, dict) and isinstance(artifact.get("package_path"), str)
    } if isinstance(artifacts, dict) else set()
    file_paths = set(files) if isinstance(files, dict) else set()
    for split_name, summary in split_summaries.items():
        examples = summary.get("examples")
        if not isinstance(examples, list):
            continue
        for line_index, example in enumerate(examples, start=1):
            for field, path in _time_series_payload_package_paths(example):
                if not _is_safe_package_path(path):
                    issues.append(
                        {
                            "field": f"{split_name}.jsonl.line.{line_index}.{field}",
                            "message": "Time-series package_path is not safe.",
                        }
                    )
                    continue
                if path not in declared_paths:
                    issues.append(
                        {
                            "field": f"{split_name}.jsonl.line.{line_index}.{field}",
                            "message": (
                                "Time-series package_path is not declared in "
                                "manifest time_series_artifacts."
                            ),
                        }
                    )
                if path not in file_paths:
                    issues.append(
                        {
                            "field": f"{split_name}.jsonl.line.{line_index}.{field}",
                            "message": (
                                "Time-series package_path is missing from manifest "
                                "files."
                            ),
                        }
                    )


def _time_series_payload_package_paths(example: dict[str, Any]) -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    channel = example.get("channel")
    if isinstance(channel, dict):
        path = channel.get("package_path")
        if isinstance(path, str) and path:
            paths.append(("channel.package_path", path))
    for example_index, time_series_example in enumerate(_dict_items(example.get("examples"))):
        channel = time_series_example.get("channel")
        if not isinstance(channel, dict):
            continue
        path = channel.get("package_path")
        if isinstance(path, str) and path:
            paths.append((f"examples.{example_index}.channel.package_path", path))
    return paths


def _multimodal_payload_package_paths(example: dict[str, Any]) -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    for index, image in enumerate(_dict_items(example.get("images"))):
        path = image.get("package_path")
        if isinstance(path, str) and path:
            paths.append((f"images.{index}.package_path", path))
    for index, pair in enumerate(_dict_items(example.get("image_text_pairs"))):
        path = pair.get("package_path")
        if isinstance(path, str) and path:
            paths.append((f"image_text_pairs.{index}.package_path", path))
    for task_index, task in enumerate(_dict_items(example.get("supervised_tasks"))):
        input_payload = task.get("input")
        if not isinstance(input_payload, dict):
            continue
        path = input_payload.get("package_path")
        if isinstance(path, str) and path:
            paths.append((f"supervised_tasks.{task_index}.input.package_path", path))
    return paths


def _dict_items(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _examples_task_coverage(examples: list[dict[str, Any]]) -> dict[str, int]:
    coverage: dict[str, int] = {}
    for example in examples:
        _merge_task_coverage(coverage, _payload_task_coverage(example))
    return coverage


def _payload_task_coverage(payload: dict[str, Any]) -> dict[str, int]:
    coverage: dict[str, int] = {}
    task = payload.get("task")
    if isinstance(task, str) and task.strip():
        coverage[task.strip()] = coverage.get(task.strip(), 0) + 1
    for pair in _dict_items(payload.get("image_text_pairs")):
        task = pair.get("task")
        if isinstance(task, str) and task.strip():
            coverage[task.strip()] = coverage.get(task.strip(), 0) + 1
    for task_payload in _dict_items(payload.get("supervised_tasks")):
        task = task_payload.get("task")
        if isinstance(task, str) and task.strip():
            coverage[task.strip()] = coverage.get(task.strip(), 0) + 1
    for example in _dict_items(payload.get("examples")):
        _merge_task_coverage(coverage, _payload_task_coverage(example))
    return dict(sorted(coverage.items()))


def _merge_task_coverage(target: dict[str, int], source: dict[str, int]) -> None:
    for task, count in source.items():
        if isinstance(task, str) and isinstance(count, int):
            target[task] = target.get(task, 0) + count


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
        elif file_name == "benchmark_report.json":
            _verify_benchmark_report_artifact(
                package_path / file_name,
                manifest,
                issues,
            )
        elif file_name == "benchmark_suite_report.json":
            _verify_benchmark_suite_report_artifact(
                package_path / file_name,
                manifest,
                issues,
            )
        elif file_name == "quality_report.json":
            _verify_quality_report_artifact(package_path / file_name, manifest, issues)
        elif file_name == "dataset_card.md":
            _verify_card_artifact(
                package_path / file_name,
                manifest,
                issues,
                artifact_name=file_name,
                title_prefix="# Dataset Card:",
            )
        elif file_name == "model_card.md":
            _verify_card_artifact(
                package_path / file_name,
                manifest,
                issues,
                artifact_name=file_name,
                title_prefix="# Model Card:",
            )
        elif file_name == "release_package_summary.json":
            _verify_release_package_summary_artifact(
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


def _verify_benchmark_report_artifact(
    path: Path,
    manifest: dict[str, Any],
    issues: list[dict[str, str]],
) -> None:
    from casecrawler.models.evaluation import BenchmarkReport

    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        issues.append(
            {
                "field": "audit_artifacts.benchmark_report.json",
                "message": f"Benchmark report artifact is invalid JSON: {exc}.",
            }
        )
        return
    try:
        report = BenchmarkReport.model_validate(payload)
    except ValueError as exc:
        issues.append(
            {
                "field": "audit_artifacts.benchmark_report.json",
                "message": f"Benchmark report artifact is invalid: {exc}.",
            }
        )
        return
    manifest_dataset_id = manifest.get("dataset_id")
    if (
        isinstance(manifest_dataset_id, str)
        and report.generated_dataset_id != manifest_dataset_id
    ):
        issues.append(
            {
                "field": "audit_artifacts.benchmark_report.json.generated_dataset_id",
                "message": (
                    "Benchmark report generated_dataset_id "
                    f"{report.generated_dataset_id!r} does not match package "
                    f"dataset_id {manifest_dataset_id!r}."
                ),
            }
        )
    if (
        isinstance(manifest_dataset_id, str)
        and report.generated_profile.dataset_id != manifest_dataset_id
    ):
        issues.append(
            {
                "field": (
                    "audit_artifacts.benchmark_report.json."
                    "generated_profile.dataset_id"
                ),
                "message": (
                    "Benchmark report generated_profile.dataset_id "
                    f"{report.generated_profile.dataset_id!r} does not match "
                    f"package dataset_id {manifest_dataset_id!r}."
                ),
            }
        )


def _verify_benchmark_suite_report_artifact(
    path: Path,
    manifest: dict[str, Any],
    issues: list[dict[str, str]],
) -> None:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        issues.append(
            {
                "field": "audit_artifacts.benchmark_suite_report.json",
                "message": f"Benchmark suite artifact is invalid JSON: {exc}.",
            }
        )
        return
    if not isinstance(payload, dict):
        issues.append(
            {
                "field": "audit_artifacts.benchmark_suite_report.json",
                "message": "Benchmark suite artifact must be a JSON object.",
            }
        )
        return
    manifest_dataset_id = manifest.get("dataset_id")
    suite_dataset_id = payload.get("dataset_id")
    if isinstance(manifest_dataset_id, str) and suite_dataset_id != manifest_dataset_id:
        issues.append(
            {
                "field": "audit_artifacts.benchmark_suite_report.json.dataset_id",
                "message": (
                    "Benchmark suite dataset_id "
                    f"{suite_dataset_id!r} does not match package dataset_id "
                    f"{manifest_dataset_id!r}."
                ),
            }
        )
    if not isinstance(payload.get("passed"), bool):
        issues.append(
            {
                "field": "audit_artifacts.benchmark_suite_report.json.passed",
                "message": "Benchmark suite passed must be a boolean.",
            }
        )
    if not isinstance(payload.get("reference_count"), int):
        issues.append(
            {
                "field": "audit_artifacts.benchmark_suite_report.json.reference_count",
                "message": "Benchmark suite reference_count must be an integer.",
            }
        )
    results = payload.get("results")
    if not isinstance(results, list):
        issues.append(
            {
                "field": "audit_artifacts.benchmark_suite_report.json.results",
                "message": "Benchmark suite results must be a list.",
            }
        )
        return
    reference_count = payload.get("reference_count")
    if isinstance(reference_count, int) and reference_count != len(results):
        issues.append(
            {
                "field": (
                    "audit_artifacts.benchmark_suite_report.json.reference_count"
                ),
                "message": (
                    "Benchmark suite reference_count does not match results length."
                ),
            }
        )
    suite_passed = payload.get("passed")
    recommended_reference_keys = _string_list_payload(
        payload.get("recommended_reference_keys")
    )
    result_reference_keys: set[str] = set()
    failed_result_indexes: list[int] = []
    for index, item in enumerate(results):
        if not isinstance(item, dict):
            issues.append(
                {
                    "field": f"audit_artifacts.benchmark_suite_report.json.results.{index}",
                    "message": "Benchmark suite result must be an object.",
                }
            )
            continue
        reference_key = item.get("reference_key")
        if not isinstance(reference_key, str):
            issues.append(
                {
                    "field": (
                        "audit_artifacts.benchmark_suite_report.json."
                        f"results.{index}.reference_key"
                    ),
                    "message": "Benchmark suite result reference_key must be a string.",
                }
            )
        elif reference_key.strip():
            result_reference_keys.add(reference_key.strip())
        if not isinstance(item.get("reference_dataset_id"), str):
            issues.append(
                {
                    "field": (
                        "audit_artifacts.benchmark_suite_report.json."
                        f"results.{index}.reference_dataset_id"
                    ),
                    "message": (
                        "Benchmark suite result reference_dataset_id must be a string."
                    ),
                }
            )
        result_passed = item.get("passed")
        if not isinstance(result_passed, bool):
            issues.append(
                {
                    "field": (
                        "audit_artifacts.benchmark_suite_report.json."
                        f"results.{index}.passed"
                    ),
                    "message": "Benchmark suite result passed must be a boolean.",
                }
            )
        elif result_passed is False:
            failed_result_indexes.append(index)
        failing_metrics = item.get("failing_metrics")
        if isinstance(failing_metrics, list) and failing_metrics:
            failed_result_indexes.append(index)
        report_payload = item.get("report")
        if not isinstance(report_payload, dict):
            issues.append(
                {
                    "field": (
                        "audit_artifacts.benchmark_suite_report.json."
                        f"results.{index}.report"
                    ),
                    "message": "Benchmark suite result report must be an object.",
                }
            )
            continue
        _verify_benchmark_report_payload(
            report_payload,
            manifest=manifest,
            field_prefix=(
                "audit_artifacts.benchmark_suite_report.json."
                f"results.{index}.report"
            ),
            issues=issues,
        )
    missing_recommended_keys = sorted(recommended_reference_keys - result_reference_keys)
    if missing_recommended_keys:
        issues.append(
            {
                "field": (
                    "audit_artifacts.benchmark_suite_report.json."
                    "recommended_reference_keys"
                ),
                "message": (
                    "Benchmark suite recommended_reference_keys are missing "
                    f"matching results: {missing_recommended_keys}."
                ),
            }
        )
    if suite_passed is True and failed_result_indexes:
        issues.append(
            {
                "field": "audit_artifacts.benchmark_suite_report.json.passed",
                "message": (
                    "Benchmark suite marks passed true but includes failed "
                    f"result entries: {sorted(set(failed_result_indexes))}."
                ),
            }
        )
    _verify_benchmark_suite_task_results(payload, issues)


def _verify_benchmark_suite_task_results(
    payload: dict[str, Any],
    issues: list[dict[str, str]],
    *,
    field_prefix: str = "audit_artifacts.benchmark_suite_report.json",
) -> None:
    task_results = payload.get("task_export_results")
    if task_results is None:
        return
    if not isinstance(task_results, dict):
        issues.append(
            {
                "field": f"{field_prefix}.task_export_results",
                "message": "Benchmark suite task_export_results must be an object.",
            }
        )
        return
    suite_passed = payload.get("passed") is True
    for export_format, result in sorted(task_results.items()):
        task_field_prefix = f"{field_prefix}.task_export_results.{export_format}"
        if not isinstance(export_format, str) or not export_format.strip():
            issues.append(
                {
                    "field": f"{field_prefix}.task_export_results",
                    "message": "Benchmark suite task export key must be a string.",
                }
            )
            continue
        if not isinstance(result, dict):
            issues.append(
                {
                    "field": task_field_prefix,
                    "message": "Benchmark suite task export result must be an object.",
                }
            )
            continue
        reference_count = result.get("reference_count")
        if not isinstance(reference_count, int):
            issues.append(
                {
                    "field": f"{task_field_prefix}.reference_count",
                    "message": (
                        "Benchmark suite task export result reference_count "
                        "must be an integer."
                    ),
                }
            )
        elif suite_passed and reference_count < 1:
            issues.append(
                {
                    "field": f"{task_field_prefix}.reference_count",
                    "message": (
                        "Benchmark suite marks passed true but task export "
                        "has no reference results."
                    ),
                }
            )
        task_passed = result.get("passed")
        if not isinstance(task_passed, bool):
            issues.append(
                {
                    "field": f"{task_field_prefix}.passed",
                    "message": (
                        "Benchmark suite task export result passed must be a boolean."
                    ),
                }
            )
        elif suite_passed and task_passed is not True:
            issues.append(
                {
                    "field": f"{task_field_prefix}.passed",
                    "message": (
                        "Benchmark suite marks passed true but task export "
                        "result is not passing."
                    ),
                }
            )
        missing = result.get("missing_reference_keys")
        if not isinstance(missing, list) or not all(
            isinstance(item, str) for item in missing
        ):
            issues.append(
                {
                    "field": f"{task_field_prefix}.missing_reference_keys",
                    "message": (
                        "Benchmark suite task export result missing_reference_keys "
                        "must be a string list."
                    ),
                }
            )
        elif suite_passed and missing:
            issues.append(
                {
                    "field": f"{task_field_prefix}.missing_reference_keys",
                    "message": (
                        "Benchmark suite marks passed true but task export "
                        f"is missing reference keys: {missing}."
                    ),
                }
            )


def _string_list_payload(value: object) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item.strip() for item in value if isinstance(item, str) and item.strip()}


def _string_int_map(value: object) -> bool:
    return isinstance(value, dict) and all(
        isinstance(key, str)
        and bool(key.strip())
        and isinstance(item, int)
        and not isinstance(item, bool)
        and item >= 0
        for key, item in value.items()
    )


def _verify_benchmark_report_payload(
    payload: dict[str, Any],
    *,
    manifest: dict[str, Any],
    field_prefix: str,
    issues: list[dict[str, str]],
) -> None:
    from casecrawler.models.evaluation import BenchmarkReport

    try:
        report = BenchmarkReport.model_validate(payload)
    except ValueError as exc:
        issues.append(
            {
                "field": field_prefix,
                "message": f"Benchmark report artifact is invalid: {exc}.",
            }
        )
        return
    manifest_dataset_id = manifest.get("dataset_id")
    if (
        isinstance(manifest_dataset_id, str)
        and report.generated_dataset_id != manifest_dataset_id
    ):
        issues.append(
            {
                "field": f"{field_prefix}.generated_dataset_id",
                "message": (
                    "Benchmark report generated_dataset_id "
                    f"{report.generated_dataset_id!r} does not match package "
                    f"dataset_id {manifest_dataset_id!r}."
                ),
            }
        )


def _verify_quality_report_artifact(
    path: Path,
    manifest: dict[str, Any],
    issues: list[dict[str, str]],
) -> None:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        issues.append(
            {
                "field": "audit_artifacts.quality_report.json",
                "message": f"Quality report artifact is invalid JSON: {exc}.",
            }
        )
        return
    if not isinstance(payload, dict):
        issues.append(
            {
                "field": "audit_artifacts.quality_report.json",
                "message": "Quality report artifact must be a JSON object.",
            }
        )
        return
    manifest_dataset_id = manifest.get("dataset_id")
    quality_dataset_id = payload.get("dataset_id")
    if (
        isinstance(manifest_dataset_id, str)
        and isinstance(quality_dataset_id, str)
        and quality_dataset_id != manifest_dataset_id
    ):
        issues.append(
            {
                "field": "audit_artifacts.quality_report.json.dataset_id",
                "message": (
                    "Quality report dataset_id "
                    f"{quality_dataset_id!r} does not match package dataset_id "
                    f"{manifest_dataset_id!r}."
                ),
            }
        )
    elif not isinstance(quality_dataset_id, str):
        issues.append(
            {
                "field": "audit_artifacts.quality_report.json.dataset_id",
                "message": "Quality report artifact is missing dataset_id.",
            }
        )
    for key in ("record_count", "approved_count"):
        if not isinstance(payload.get(key), int):
            issues.append(
                {
                    "field": f"audit_artifacts.quality_report.json.{key}",
                    "message": f"Quality report artifact has no integer {key}.",
                }
            )
    if not isinstance(payload.get("export_ready"), bool):
        issues.append(
            {
                "field": "audit_artifacts.quality_report.json.export_ready",
                "message": "Quality report artifact has no boolean export_ready.",
            }
        )
    _verify_optional_quality_numeric_fields(payload, issues)
    if not isinstance(payload.get("multimodal_release_ready"), bool):
        issues.append(
            {
                "field": "audit_artifacts.quality_report.json.multimodal_release_ready",
                "message": (
                    "Quality report artifact has no boolean "
                    "multimodal_release_ready."
                ),
            }
        )
    coverage = payload.get("core_artifact_coverage")
    if not isinstance(coverage, dict) or not all(
        isinstance(key, str) and isinstance(value, bool)
        for key, value in (coverage or {}).items()
    ):
        issues.append(
            {
                "field": "audit_artifacts.quality_report.json.core_artifact_coverage",
                "message": (
                    "Quality report artifact has no boolean core artifact "
                    "coverage map."
                ),
            }
        )
    elif payload.get("multimodal_release_ready") is True:
        if missing_keys := sorted(REQUIRED_RELEASE_COVERAGE_KEYS - set(coverage)):
            issues.append(
                {
                    "field": "audit_artifacts.quality_report.json.core_artifact_coverage",
                    "message": (
                        "Quality report artifact is missing required release coverage "
                        f"keys: {missing_keys}."
                    ),
                }
            )
        else:
            failed_keys = sorted(
                key
                for key in REQUIRED_RELEASE_COVERAGE_KEYS
                if coverage.get(key) is not True
            )
            if failed_keys:
                issues.append(
                    {
                        "field": (
                            "audit_artifacts.quality_report.json."
                            "core_artifact_coverage"
                        ),
                        "message": (
                            "Quality report artifact marks multimodal_release_ready "
                            f"but has false release coverage keys: {failed_keys}."
                        ),
                    }
                )
    missing = payload.get("multimodal_release_missing")
    if not isinstance(missing, list) or not all(
        isinstance(item, str) for item in missing
    ):
        issues.append(
            {
                "field": "audit_artifacts.quality_report.json.multimodal_release_missing",
                "message": (
                    "Quality report artifact has no string list "
                    "multimodal_release_missing."
                ),
            }
        )


def _verify_optional_quality_numeric_fields(
    payload: dict[str, Any],
    issues: list[dict[str, str]],
) -> None:
    for key in (
        "mean_time_series_points",
        "mean_time_series_duration_hours",
        "mean_imaging_prompt_chars",
        "mean_imaging_report_chars",
        "mean_imaging_width",
        "mean_imaging_height",
        "mean_modality_alignment_score",
    ):
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            issues.append(
                {
                    "field": f"audit_artifacts.quality_report.json.{key}",
                    "message": f"Quality report artifact {key} must be numeric.",
                }
            )
    _verify_optional_quality_rate(
        payload,
        issues,
        key="imaging_report_label_evidence_rate",
    )
    time_series_channel_counts = payload.get("time_series_channel_counts")
    if time_series_channel_counts is not None and not _string_int_map(
        time_series_channel_counts
    ):
        issues.append(
            {
                "field": "audit_artifacts.quality_report.json.time_series_channel_counts",
                "message": (
                    "Quality report artifact time_series_channel_counts must be "
                    "a string-to-integer map."
                ),
            }
        )
    time_series_model_policy_counts = payload.get("time_series_model_policy_counts")
    if time_series_model_policy_counts is not None and not _string_int_map(
        time_series_model_policy_counts
    ):
        issues.append(
            {
                "field": "audit_artifacts.quality_report.json.time_series_model_policy_counts",
                "message": (
                    "Quality report artifact time_series_model_policy_counts "
                    "must be a string-to-integer map."
                ),
            }
        )
    clinical_text_model_policy_counts = payload.get("clinical_text_model_policy_counts")
    if clinical_text_model_policy_counts is not None and not _string_int_map(
        clinical_text_model_policy_counts
    ):
        issues.append(
            {
                "field": "audit_artifacts.quality_report.json.clinical_text_model_policy_counts",
                "message": (
                    "Quality report artifact clinical_text_model_policy_counts "
                    "must be a string-to-integer map."
                ),
            }
        )
    imaging_model_policy_counts = payload.get("imaging_model_policy_counts")
    if imaging_model_policy_counts is not None and not _string_int_map(
        imaging_model_policy_counts
    ):
        issues.append(
            {
                "field": "audit_artifacts.quality_report.json.imaging_model_policy_counts",
                "message": (
                    "Quality report artifact imaging_model_policy_counts "
                    "must be a string-to-integer map."
                ),
            }
        )
    image_validator_policy_counts = payload.get("image_validator_policy_counts")
    if image_validator_policy_counts is not None and not _string_int_map(
        image_validator_policy_counts
    ):
        issues.append(
            {
                "field": "audit_artifacts.quality_report.json.image_validator_policy_counts",
                "message": (
                    "Quality report artifact image_validator_policy_counts must "
                    "be a string-to-integer map."
                ),
            }
        )


def _verify_optional_quality_rate(
    payload: dict[str, Any],
    issues: list[dict[str, str]],
    *,
    key: str,
) -> None:
    value = payload.get(key)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | float):
        issues.append(
            {
                "field": f"audit_artifacts.quality_report.json.{key}",
                "message": f"Quality report artifact {key} must be numeric.",
            }
        )
        return
    if not 0 <= float(value) <= 1:
        issues.append(
            {
                "field": f"audit_artifacts.quality_report.json.{key}",
                "message": f"Quality report artifact {key} must be between 0 and 1.",
            }
        )


def _verify_card_artifact(
    path: Path,
    manifest: dict[str, Any],
    issues: list[dict[str, str]],
    *,
    artifact_name: str,
    title_prefix: str,
) -> None:
    text = path.read_text()
    title = next((line.strip() for line in text.splitlines() if line.strip()), "")
    field = f"audit_artifacts.{artifact_name}.title"
    if not title.startswith(title_prefix):
        issues.append(
            {
                "field": field,
                "message": f"{artifact_name} is missing a {title_prefix!r} heading.",
            }
        )
        return
    package_name = manifest.get("name")
    package_dataset_id = manifest.get("dataset_id")
    expected_tokens = [
        token
        for token in (package_name, package_dataset_id)
        if isinstance(token, str) and token.strip()
    ]
    if expected_tokens and not any(token in text for token in expected_tokens):
        issues.append(
            {
                "field": field,
                "message": (
                    f"{artifact_name} does not reference package name or "
                    "dataset_id."
                ),
            }
        )


def _verify_release_package_summary_artifact(
    path: Path,
    manifest: dict[str, Any],
    issues: list[dict[str, str]],
) -> None:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        issues.append(
            {
                "field": "audit_artifacts.release_package_summary.json",
                "message": f"Release package summary is invalid JSON: {exc}.",
            }
        )
        return
    if not isinstance(payload, dict):
        issues.append(
            {
                "field": "audit_artifacts.release_package_summary.json",
                "message": "Release package summary must be a JSON object.",
            }
        )
        return
    manifest_dataset_id = manifest.get("dataset_id")
    summary_dataset_id = payload.get("dataset_id")
    if isinstance(manifest_dataset_id, str) and summary_dataset_id != manifest_dataset_id:
        issues.append(
            {
                "field": "audit_artifacts.release_package_summary.json.dataset_id",
                "message": (
                    "Release package summary dataset_id "
                    f"{summary_dataset_id!r} does not match package dataset_id "
                    f"{manifest_dataset_id!r}."
                ),
            }
        )
    _verify_release_summary_task_coverage(payload, manifest, issues)
    quality = payload.get("quality_report")
    if not isinstance(quality, dict):
        issues.append(
            {
                "field": "audit_artifacts.release_package_summary.json.quality_report",
                "message": "Release package summary quality_report must be an object.",
            }
        )
    else:
        _verify_release_summary_quality(quality, issues)
    benchmark = payload.get("benchmark")
    if not isinstance(benchmark, dict):
        issues.append(
            {
                "field": "audit_artifacts.release_package_summary.json.benchmark",
                "message": "Release package summary benchmark must be an object.",
            }
        )
    else:
        _verify_release_summary_benchmark(benchmark, issues)
    benchmark_suite = payload.get("benchmark_suite")
    if not isinstance(benchmark_suite, dict):
        issues.append(
            {
                "field": (
                    "audit_artifacts.release_package_summary.json.benchmark_suite"
                ),
                "message": (
                    "Release package summary benchmark_suite must be an object."
                ),
            }
        )
    else:
        _verify_release_summary_benchmark_suite(benchmark_suite, issues)
    _verify_release_summary_objective_coverage(payload, quality, issues)


def _verify_release_summary_quality(
    quality: dict[str, Any],
    issues: list[dict[str, str]],
) -> None:
    field_prefix = "audit_artifacts.release_package_summary.json.quality_report"
    for key in ("export_ready", "multimodal_release_ready"):
        if not isinstance(quality.get(key), bool):
            issues.append(
                {
                    "field": f"{field_prefix}.{key}",
                    "message": (
                        f"Release package summary quality_report.{key} "
                        "must be a boolean."
                    ),
                }
            )
    _verify_release_summary_quality_numeric_fields(quality, issues)
    for key in ("race_counts", "ethnicity_counts", "insurance_counts"):
        if not _string_int_map(quality.get(key)):
            issues.append(
                {
                    "field": f"{field_prefix}.{key}",
                    "message": (
                        "Release package summary quality_report."
                        f"{key} must be a string-to-integer map."
                    ),
                }
            )
    social_history_counts = quality.get("social_history_counts")
    if not isinstance(social_history_counts, dict) or not all(
        isinstance(key, str) and _string_int_map(value)
        for key, value in social_history_counts.items()
    ):
        issues.append(
            {
                "field": f"{field_prefix}.social_history_counts",
                "message": (
                    "Release package summary quality_report.social_history_counts "
                    "must be a map of string-to-integer maps."
                ),
            }
        )
    coverage = quality.get("core_artifact_coverage")
    if not isinstance(coverage, dict) or not all(
        isinstance(key, str) and isinstance(value, bool)
        for key, value in (coverage or {}).items()
    ):
        issues.append(
            {
                "field": f"{field_prefix}.core_artifact_coverage",
                "message": (
                    "Release package summary quality_report.core_artifact_coverage "
                    "must be a boolean map."
                ),
            }
        )
    elif quality.get("multimodal_release_ready") is True:
        missing_keys = sorted(REQUIRED_RELEASE_COVERAGE_KEYS - set(coverage))
        failed_keys = sorted(
            key
            for key in REQUIRED_RELEASE_COVERAGE_KEYS
            if coverage.get(key) is not True
        )
        if missing_keys or failed_keys:
            issues.append(
                {
                    "field": f"{field_prefix}.core_artifact_coverage",
                    "message": (
                        "Release package summary marks multimodal_release_ready "
                        "but release coverage is incomplete."
                    ),
                }
            )
    missing = quality.get("multimodal_release_missing")
    if not isinstance(missing, list) or not all(isinstance(item, str) for item in missing):
        issues.append(
            {
                "field": f"{field_prefix}.multimodal_release_missing",
                "message": (
                    "Release package summary quality_report.multimodal_release_missing "
                    "must be a string list."
                ),
            }
        )
    elif quality.get("multimodal_release_ready") is True and missing:
        issues.append(
            {
                "field": f"{field_prefix}.multimodal_release_missing",
                "message": (
                    "Release package summary marks multimodal_release_ready "
                    f"but lists missing requirements: {missing}."
                ),
            }
        )


def _verify_release_summary_objective_coverage(
    payload: dict[str, Any],
    quality: Any,
    issues: list[dict[str, str]],
) -> None:
    field_prefix = "audit_artifacts.release_package_summary.json.objective_coverage"
    release_ready = (
        isinstance(quality, dict)
        and quality.get("multimodal_release_ready") is True
    )
    objective_coverage = payload.get("objective_coverage")
    if objective_coverage is None:
        if release_ready:
            issues.append(
                {
                    "field": field_prefix,
                    "message": (
                        "Release package summary is missing objective_coverage."
                    ),
                }
            )
        return
    if not isinstance(objective_coverage, dict):
        issues.append(
            {
                "field": field_prefix,
                "message": "Release package summary objective_coverage must be an object.",
            }
        )
        return
    if not isinstance(objective_coverage.get("objective"), str):
        issues.append(
            {
                "field": f"{field_prefix}.objective",
                "message": (
                    "Release package summary objective_coverage.objective "
                    "must be a string."
                ),
            }
        )
    complete = objective_coverage.get("complete")
    if not isinstance(complete, bool):
        issues.append(
            {
                "field": f"{field_prefix}.complete",
                "message": (
                    "Release package summary objective_coverage.complete "
                    "must be a boolean."
                ),
            }
        )
    elif release_ready and complete is not True:
        issues.append(
            {
                "field": f"{field_prefix}.complete",
                "message": (
                    "Release package summary marks multimodal_release_ready "
                    "but objective_coverage is incomplete."
                ),
            }
        )
    missing = objective_coverage.get("missing")
    if not isinstance(missing, list) or not all(
        isinstance(item, str) for item in missing
    ):
        issues.append(
            {
                "field": f"{field_prefix}.missing",
                "message": (
                    "Release package summary objective_coverage.missing "
                    "must be a string list."
                ),
            }
        )
    elif complete is True and missing:
        issues.append(
            {
                "field": f"{field_prefix}.missing",
                "message": (
                    "Release package summary marks objective_coverage.complete "
                    f"but lists missing criteria: {missing}."
                ),
            }
        )
    criteria = objective_coverage.get("criteria")
    if not isinstance(criteria, dict):
        issues.append(
            {
                "field": f"{field_prefix}.criteria",
                "message": (
                    "Release package summary objective_coverage.criteria "
                    "must be an object."
                ),
            }
        )
        return
    missing_keys = sorted(OBJECTIVE_COVERAGE_KEYS - set(criteria))
    if complete is True and missing_keys:
        issues.append(
            {
                "field": f"{field_prefix}.criteria",
                "message": (
                    "Release package summary marks objective_coverage.complete "
                    f"but is missing criteria: {missing_keys}."
                ),
            }
        )
    for key, criterion in criteria.items():
        if not isinstance(key, str) or not isinstance(criterion, dict):
            issues.append(
                {
                    "field": f"{field_prefix}.criteria",
                    "message": (
                        "Release package summary objective_coverage.criteria "
                        "must be a string-to-object map."
                    ),
                }
            )
            return
        _verify_release_summary_objective_criterion(
            key,
            criterion,
            complete is True and key in OBJECTIVE_COVERAGE_KEYS,
            issues,
        )


def _verify_release_summary_objective_criterion(
    key: str,
    criterion: dict[str, Any],
    require_satisfied: bool,
    issues: list[dict[str, str]],
) -> None:
    field_prefix = (
        "audit_artifacts.release_package_summary.json.objective_coverage"
        f".criteria.{key}"
    )
    if not isinstance(criterion.get("requirement"), str):
        issues.append(
            {
                "field": f"{field_prefix}.requirement",
                "message": (
                    "Release package summary objective criterion requirement "
                    "must be a string."
                ),
            }
        )
    satisfied = criterion.get("satisfied")
    if not isinstance(satisfied, bool):
        issues.append(
            {
                "field": f"{field_prefix}.satisfied",
                "message": (
                    "Release package summary objective criterion satisfied "
                    "must be a boolean."
                ),
            }
        )
    elif require_satisfied and satisfied is not True:
        issues.append(
            {
                "field": f"{field_prefix}.satisfied",
                "message": (
                    "Release package summary marks objective_coverage.complete "
                    f"but criterion {key!r} is not satisfied."
                ),
            }
        )
    artifacts = criterion.get("artifacts")
    if not isinstance(artifacts, list) or not all(
        isinstance(item, str) for item in artifacts
    ):
        issues.append(
            {
                "field": f"{field_prefix}.artifacts",
                "message": (
                    "Release package summary objective criterion artifacts "
                    "must be a string list."
                ),
            }
        )
    if not isinstance(criterion.get("evidence"), dict):
        issues.append(
            {
                "field": f"{field_prefix}.evidence",
                "message": (
                    "Release package summary objective criterion evidence "
                    "must be an object."
                ),
            }
        )


def _verify_release_summary_task_coverage(
    payload: dict[str, Any],
    manifest: dict[str, Any],
    issues: list[dict[str, str]],
) -> None:
    field = "audit_artifacts.release_package_summary.json.task_coverage"
    task_coverage = payload.get("task_coverage")
    quality = payload.get("quality_report")
    release_ready = (
        isinstance(quality, dict)
        and quality.get("multimodal_release_ready") is True
    )
    if task_coverage is None:
        if release_ready:
            issues.append(
                {
                    "field": field,
                    "message": "Release package summary is missing task_coverage.",
                }
            )
        return
    if not _string_int_map(task_coverage):
        issues.append(
            {
                "field": field,
                "message": (
                    "Release package summary task_coverage must be a "
                    "string-to-integer map."
                ),
            }
        )
        return
    manifest_task_coverage = manifest.get("task_coverage")
    if isinstance(manifest_task_coverage, dict) and task_coverage != manifest_task_coverage:
        issues.append(
            {
                "field": field,
                "message": (
                    "Release package summary task_coverage does not match "
                    "manifest task_coverage."
                ),
            }
        )


def _verify_release_summary_quality_numeric_fields(
    quality: dict[str, Any],
    issues: list[dict[str, str]],
) -> None:
    field_prefix = "audit_artifacts.release_package_summary.json.quality_report"
    for key in (
        "mean_time_series_points",
        "mean_time_series_duration_hours",
        "mean_imaging_prompt_chars",
        "mean_imaging_report_chars",
        "mean_imaging_width",
        "mean_imaging_height",
        "mean_modality_alignment_score",
    ):
        value = quality.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            issues.append(
                {
                    "field": f"{field_prefix}.{key}",
                    "message": (
                        "Release package summary quality_report."
                        f"{key} must be numeric."
                    ),
                }
            )
    time_series_channel_counts = quality.get("time_series_channel_counts")
    if time_series_channel_counts is not None and not _string_int_map(
        time_series_channel_counts
    ):
        issues.append(
            {
                "field": f"{field_prefix}.time_series_channel_counts",
                "message": (
                    "Release package summary quality_report."
                    "time_series_channel_counts must be a string-to-integer map."
                ),
            }
        )
    time_series_model_policy_counts = quality.get("time_series_model_policy_counts")
    if time_series_model_policy_counts is not None and not _string_int_map(
        time_series_model_policy_counts
    ):
        issues.append(
            {
                "field": f"{field_prefix}.time_series_model_policy_counts",
                "message": (
                    "Release package summary quality_report."
                    "time_series_model_policy_counts must be a string-to-integer map."
                ),
            }
        )
    clinical_text_model_policy_counts = quality.get("clinical_text_model_policy_counts")
    if clinical_text_model_policy_counts is not None and not _string_int_map(
        clinical_text_model_policy_counts
    ):
        issues.append(
            {
                "field": f"{field_prefix}.clinical_text_model_policy_counts",
                "message": (
                    "Release package summary quality_report."
                    "clinical_text_model_policy_counts must be a string-to-integer map."
                ),
            }
        )
    imaging_model_policy_counts = quality.get("imaging_model_policy_counts")
    if imaging_model_policy_counts is not None and not _string_int_map(
        imaging_model_policy_counts
    ):
        issues.append(
            {
                "field": f"{field_prefix}.imaging_model_policy_counts",
                "message": (
                    "Release package summary quality_report."
                    "imaging_model_policy_counts must be a string-to-integer map."
                ),
            }
        )
    image_validator_policy_counts = quality.get("image_validator_policy_counts")
    if image_validator_policy_counts is not None and not _string_int_map(
        image_validator_policy_counts
    ):
        issues.append(
            {
                "field": f"{field_prefix}.image_validator_policy_counts",
                "message": (
                    "Release package summary quality_report."
                    "image_validator_policy_counts must be a string-to-integer map."
                ),
            }
        )
    value = quality.get("imaging_report_label_evidence_rate")
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | float):
        issues.append(
            {
                "field": f"{field_prefix}.imaging_report_label_evidence_rate",
                "message": (
                    "Release package summary quality_report."
                    "imaging_report_label_evidence_rate must be numeric."
                ),
            }
        )
        return
    if not 0 <= float(value) <= 1:
        issues.append(
            {
                "field": f"{field_prefix}.imaging_report_label_evidence_rate",
                "message": (
                    "Release package summary quality_report."
                    "imaging_report_label_evidence_rate must be between 0 and 1."
                ),
            }
        )


def _verify_release_summary_benchmark(
    benchmark: dict[str, Any],
    issues: list[dict[str, str]],
) -> None:
    field_prefix = "audit_artifacts.release_package_summary.json.benchmark"
    if not isinstance(benchmark.get("reference_dataset_id"), str):
        issues.append(
            {
                "field": f"{field_prefix}.reference_dataset_id",
                "message": (
                    "Release package summary benchmark.reference_dataset_id "
                    "must be a string."
                ),
            }
        )
    if not isinstance(benchmark.get("passed"), bool):
        issues.append(
            {
                "field": f"{field_prefix}.passed",
                "message": "Release package summary benchmark.passed must be a boolean.",
            }
        )
    if not isinstance(benchmark.get("overall_score"), int | float):
        issues.append(
            {
                "field": f"{field_prefix}.overall_score",
                "message": (
                    "Release package summary benchmark.overall_score must be numeric."
                ),
            }
        )
    failing_metrics = benchmark.get("failing_metrics")
    if not isinstance(failing_metrics, list) or not all(
        isinstance(item, str) for item in failing_metrics
    ):
        issues.append(
            {
                "field": f"{field_prefix}.failing_metrics",
                "message": (
                    "Release package summary benchmark.failing_metrics "
                    "must be a string list."
                ),
            }
        )
    elif benchmark.get("passed") is True and failing_metrics:
        issues.append(
            {
                "field": f"{field_prefix}.failing_metrics",
                "message": (
                    "Release package summary benchmark marks passed true "
                    f"but lists failing metrics: {failing_metrics}."
                ),
            }
        )


def _verify_release_summary_benchmark_suite(
    benchmark_suite: dict[str, Any],
    issues: list[dict[str, str]],
) -> None:
    field_prefix = "audit_artifacts.release_package_summary.json.benchmark_suite"
    if not isinstance(benchmark_suite.get("passed"), bool):
        issues.append(
            {
                "field": f"{field_prefix}.passed",
                "message": (
                    "Release package summary benchmark_suite.passed must be a boolean."
                ),
            }
        )
    reference_count = benchmark_suite.get("reference_count")
    if not isinstance(reference_count, int):
        issues.append(
            {
                "field": f"{field_prefix}.reference_count",
                "message": (
                    "Release package summary benchmark_suite.reference_count "
                    "must be an integer."
                ),
            }
        )
    elif benchmark_suite.get("passed") is True and reference_count < 1:
        issues.append(
            {
                "field": f"{field_prefix}.reference_count",
                "message": (
                    "Release package summary benchmark_suite marks passed true "
                    "but has no reference datasets."
                ),
            }
        )
    if not isinstance(benchmark_suite.get("mean_overall_score"), int | float):
        issues.append(
            {
                "field": f"{field_prefix}.mean_overall_score",
                "message": (
                    "Release package summary benchmark_suite.mean_overall_score "
                    "must be numeric."
                ),
            }
        )
    if not isinstance(benchmark_suite.get("task_export_results"), dict):
        issues.append(
            {
                "field": f"{field_prefix}.task_export_results",
                "message": (
                    "Release package summary benchmark_suite.task_export_results "
                    "must be an object."
                ),
            }
        )
    else:
        _verify_benchmark_suite_task_results(
            benchmark_suite,
            issues,
            field_prefix=field_prefix,
        )
    _verify_release_summary_benchmark_suite_results(
        benchmark_suite,
        issues,
        field_prefix=field_prefix,
    )


def _verify_release_summary_benchmark_suite_results(
    benchmark_suite: dict[str, Any],
    issues: list[dict[str, str]],
    *,
    field_prefix: str,
) -> None:
    results = benchmark_suite.get("results")
    if results is None:
        if benchmark_suite.get("passed") is True:
            issues.append(
                {
                    "field": f"{field_prefix}.results",
                    "message": (
                        "Release package summary benchmark_suite marks passed "
                        "true but is missing results."
                    ),
                }
            )
        return
    if not isinstance(results, list):
        issues.append(
            {
                "field": f"{field_prefix}.results",
                "message": (
                    "Release package summary benchmark_suite.results "
                    "must be a list."
                ),
            }
        )
        return
    reference_count = benchmark_suite.get("reference_count")
    if isinstance(reference_count, int) and reference_count != len(results):
        issues.append(
            {
                "field": f"{field_prefix}.reference_count",
                "message": (
                    "Release package summary benchmark_suite.reference_count "
                    "does not match results length."
                ),
            }
        )
    result_reference_keys: set[str] = set()
    failed_result_indexes: list[int] = []
    for index, item in enumerate(results):
        if not isinstance(item, dict):
            issues.append(
                {
                    "field": f"{field_prefix}.results.{index}",
                    "message": (
                        "Release package summary benchmark_suite result "
                        "must be an object."
                    ),
                }
            )
            continue
        reference_key = item.get("reference_key")
        if isinstance(reference_key, str) and reference_key.strip():
            result_reference_keys.add(reference_key.strip())
        result_passed = item.get("passed")
        if result_passed is False:
            failed_result_indexes.append(index)
        failing_metrics = item.get("failing_metrics")
        if isinstance(failing_metrics, list) and failing_metrics:
            failed_result_indexes.append(index)
    recommended_reference_keys = _string_list_payload(
        benchmark_suite.get("recommended_reference_keys")
    )
    missing_recommended_keys = sorted(
        recommended_reference_keys - result_reference_keys
    )
    if missing_recommended_keys:
        issues.append(
            {
                "field": f"{field_prefix}.recommended_reference_keys",
                "message": (
                    "Release package summary benchmark_suite recommended "
                    "reference keys are missing matching results: "
                    f"{missing_recommended_keys}."
                ),
            }
        )
    if benchmark_suite.get("passed") is True and failed_result_indexes:
        issues.append(
            {
                "field": f"{field_prefix}.passed",
                "message": (
                    "Release package summary benchmark_suite marks passed true "
                    "but includes failed result entries: "
                    f"{sorted(set(failed_result_indexes))}."
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


def _copy_image_artifacts(
    records: list[SyntheticRecord],
    output_path: Path,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    entries: dict[str, str] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    image_dir = output_path / "images"
    seen_package_paths: set[str] = set()
    for record in records:
        for asset in record.imaging:
            if not asset.file_path:
                continue
            source_path = Path(asset.file_path)
            if not source_path.is_file():
                continue
            image_dir.mkdir(parents=True, exist_ok=True)
            package_path = _image_package_path(
                record_id=record.record_id,
                image_id=asset.image_id,
                source_path=source_path,
                seen_package_paths=seen_package_paths,
            )
            target_path = output_path / package_path
            shutil.copyfile(source_path, target_path)
            key = f"{record.record_id}:{asset.image_id}"
            entries[package_path] = str(target_path)
            artifacts[key] = {
                "record_id": record.record_id,
                "image_id": asset.image_id,
                "package_path": package_path,
                "source_path": str(source_path),
                "modality": asset.modality,
                "body_region": asset.body_region,
                "generation_backend": asset.generation_backend,
                "prompt": asset.prompt,
                "report_text": asset.report_text,
                "labels": [label.model_dump(mode="json") for label in asset.labels],
                "imaging_model_policy": _json_object_or_none(
                    record.metadata.get("imaging_model_policy")
                ),
                "image_validator_policy": _json_object_or_none(
                    record.metadata.get("image_validator_policy")
                ),
            }
    return entries, artifacts


def _write_time_series_artifacts(
    records: list[SyntheticRecord],
    output_path: Path,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    entries: dict[str, str] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    time_series_dir = output_path / "time_series"
    seen_package_paths: set[str] = set()
    for record in records:
        for channel in record.time_series:
            time_series_dir.mkdir(parents=True, exist_ok=True)
            package_path = _time_series_package_path(
                record_id=record.record_id,
                channel_name=channel.name,
                seen_package_paths=seen_package_paths,
            )
            target_path = output_path / package_path
            target_path.write_text(
                json.dumps(
                    {
                        "record_id": record.record_id,
                        "dataset_id": record.dataset_id,
                        "channel": channel.model_dump(mode="json"),
                        "synthetic": True,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            key = f"{record.record_id}:{channel.name}"
            entries[package_path] = str(target_path)
            artifacts[key] = {
                "record_id": record.record_id,
                "channel_name": channel.name,
                "package_path": package_path,
                "unit": channel.unit,
                "generation_backend": channel.generation_backend,
                "sampling_rate_hz": channel.sampling_rate_hz,
                "point_count": len(channel.points),
                "time_series_model_policy": _json_object_or_none(
                    record.metadata.get("time_series_model_policy")
                ),
            }
    return entries, artifacts


def _record_image_package_paths(
    record: SyntheticRecord,
    image_artifacts: dict[str, dict[str, Any]],
) -> dict[str, str]:
    package_paths: dict[str, str] = {}
    for asset in record.imaging:
        artifact = image_artifacts.get(f"{record.record_id}:{asset.image_id}")
        if artifact and artifact.get("package_path"):
            package_paths[asset.image_id] = artifact["package_path"]
    return package_paths


def _record_time_series_package_paths(
    record: SyntheticRecord,
    time_series_artifacts: dict[str, dict[str, Any]],
) -> dict[str, str]:
    package_paths: dict[str, str] = {}
    for channel in record.time_series:
        artifact = time_series_artifacts.get(f"{record.record_id}:{channel.name}")
        if artifact and artifact.get("package_path"):
            package_paths[channel.name] = artifact["package_path"]
    return package_paths


def _json_object_or_none(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _image_package_path(
    *,
    record_id: str,
    image_id: str,
    source_path: Path,
    seen_package_paths: set[str],
) -> str:
    suffix = source_path.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        suffix = ".png"
    stem = _package_slug(f"{record_id}-{image_id}") or "image"
    candidate = f"images/{stem}{suffix}"
    counter = 2
    while candidate in seen_package_paths:
        candidate = f"images/{stem}-{counter}{suffix}"
        counter += 1
    seen_package_paths.add(candidate)
    return candidate


def _time_series_package_path(
    *,
    record_id: str,
    channel_name: str,
    seen_package_paths: set[str],
) -> str:
    stem = _package_slug(f"{record_id}-{channel_name}") or "time-series"
    candidate = f"time_series/{stem}.json"
    counter = 2
    while candidate in seen_package_paths:
        candidate = f"time_series/{stem}-{counter}.json"
        counter += 1
    seen_package_paths.add(candidate)
    return candidate


def _package_slug(value: str) -> str:
    slug = []
    for char in value.lower():
        if char.isalnum():
            slug.append(char)
        elif char in {"-", "_"}:
            slug.append("-")
    return "-".join("".join(slug).split("-"))


def _is_safe_package_path(value: str) -> bool:
    path = Path(value)
    return (
        bool(value)
        and not path.is_absolute()
        and ".." not in path.parts
        and all(part not in {"", "."} for part in path.parts)
    )


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


def _audit_requires_time_series_artifacts(
    audit_artifacts: dict[str, str | dict[str, Any]],
) -> bool:
    quality_report = audit_artifacts.get("quality_report.json")
    if isinstance(quality_report, str):
        try:
            parsed = json.loads(quality_report)
        except json.JSONDecodeError:
            return False
        quality_report = parsed
    if not isinstance(quality_report, dict):
        return False
    coverage = quality_report.get("core_artifact_coverage")
    return (
        quality_report.get("multimodal_release_ready") is True
        and isinstance(coverage, dict)
        and coverage.get("time_series") is True
    )


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
    metadata: dict[str, Any] = {
        "topic": record.topic,
        "complexity": record.complexity.value,
        "modalities": [m.value for m in record.modalities],
        "synthetic": True,
        "provenance": record.provenance.model_dump(),
    }
    for key in (
        "clinical_text_model_policy",
        "imaging_model_policy",
        "image_validator_policy",
        "time_series_model_policy",
        "generation_overrides",
        "cohort_constraints",
    ):
        value = _json_object_or_none(record.metadata.get(key))
        if value is not None:
            metadata[key] = value
    return metadata


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
