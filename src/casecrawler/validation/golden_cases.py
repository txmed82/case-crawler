from __future__ import annotations

import json
from pathlib import Path

from casecrawler.models.synthetic import SyntheticRecord


def summarize_golden_case(record: SyntheticRecord) -> dict[str, object]:
    diagnoses = [
        diagnosis.display
        for encounter in record.encounters
        for diagnosis in encounter.diagnoses
    ]
    procedures = [
        procedure.display
        for encounter in record.encounters
        for procedure in encounter.procedures
    ]
    return {
        "topic": record.topic,
        "complexity": record.complexity.value,
        "approved": None if record.validation is None else record.validation.approved,
        "issue_count": 0 if record.validation is None else len(record.validation.issues),
        "diagnoses": sorted(set(diagnoses)),
        "procedures": sorted(set(procedures)),
        "lab_names": [lab.name for lab in record.labs],
        "vital_names": [vital.name for vital in record.vitals],
        "medication_names": [medication.name for medication in record.medication_history],
        "order_types": [order.order_type for order in record.orders],
        "note_types": sorted({document.note_type for document in record.documents}),
        "time_series_channels": [channel.name for channel in record.time_series],
        "artifact_counts": {
            "encounters": len(record.encounters),
            "labs": len(record.labs),
            "vitals": len(record.vitals),
            "medications": len(record.medication_history),
            "allergies": len(record.allergies),
            "orders": len(record.orders),
            "documents": len(record.documents),
            "time_series_channels": len(record.time_series),
            "imaging_assets": len(record.imaging),
        },
    }


def load_golden_summary(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())
