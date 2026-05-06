from __future__ import annotations

import json
from typing import Any

from casecrawler.models.dataset import ExportFormat
from casecrawler.models.synthetic import SyntheticRecord


def export_sft_record(record: SyntheticRecord, task: str = "summarize") -> dict[str, Any]:
    note_text = "\n\n".join(document.clean_text for document in record.documents)
    if task == "summarize":
        user = f"Summarize the following synthetic clinical record:\n\n{note_text}"
        assistant: str | dict = (
            f"Synthetic patient with {record.topic}; structured data includes "
            f"{len(record.labs)} labs and {len(record.vitals)} vitals."
        )
    elif task == "extract":
        user = (
            "Extract diagnoses, abnormal labs, and vital sign abnormalities from "
            f"this synthetic note:\n\n{note_text}"
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
    return {
        "record_id": record.record_id,
        "dataset_id": record.dataset_id,
        "clinical_context": _clinical_context(record),
        "images": [
            {
                "image_id": asset.image_id,
                "file_path": asset.file_path,
                "modality": asset.modality,
                "body_region": asset.body_region,
                "prompt": asset.prompt,
                "report_text": asset.report_text,
            }
            for asset in record.imaging
        ],
        "metadata": _metadata(record),
    }


def export_record(record: SyntheticRecord, export_format: str | ExportFormat) -> dict[str, Any]:
    resolved_format = ExportFormat(export_format)
    if resolved_format == ExportFormat.SFT_JSONL:
        return export_sft_record(record)
    if resolved_format == ExportFormat.CHAT_JSONL:
        return export_chat_record(record)
    if resolved_format == ExportFormat.MULTIMODAL_JSONL:
        return export_multimodal_record(record)
    if resolved_format == ExportFormat.RAW_JSONL:
        return record.model_dump()
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
    }


def _metadata(record: SyntheticRecord) -> dict[str, Any]:
    return {
        "topic": record.topic,
        "complexity": record.complexity.value,
        "modalities": [m.value for m in record.modalities],
        "synthetic": True,
    }
