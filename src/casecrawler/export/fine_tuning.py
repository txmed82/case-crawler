from __future__ import annotations

import json
from typing import Any

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
