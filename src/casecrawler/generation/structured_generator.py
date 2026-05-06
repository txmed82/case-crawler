from __future__ import annotations

from datetime import datetime
from uuid import NAMESPACE_URL, uuid5

from casecrawler.models.dataset import GenerationRequest
from casecrawler.models.synthetic import (
    Code,
    Encounter,
    LabObservation,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
    VitalObservation,
)


class StructuredGenerator:
    def generate(
        self,
        dataset_id: str,
        req: GenerationRequest,
        index: int,
    ) -> SyntheticRecord:
        now = _normalize_base_time(req.cohort_constraints.get("base_time"))
        stable_prefix = f"{dataset_id}:{req.topic}:{index}"
        patient = SyntheticPatient(
            patient_id=f"pat-{uuid5(NAMESPACE_URL, f'{stable_prefix}:patient')}",
            age=45 + (index % 35),
            sex="female" if index % 2 else "male",
        )
        encounter = Encounter(
            encounter_id=f"enc-{uuid5(NAMESPACE_URL, f'{stable_prefix}:encounter')}",
            start=now,
            setting="emergency_department",
            reason=req.topic,
            diagnoses=[
                Code(
                    system="synthetic",
                    code=req.topic.replace(" ", "_"),
                    display=req.topic,
                )
            ],
        )
        return SyntheticRecord(
            record_id=f"rec-{uuid5(NAMESPACE_URL, f'{stable_prefix}:record')}",
            dataset_id=dataset_id,
            topic=req.topic,
            complexity=req.complexity,
            modalities=req.modalities,
            patient=patient,
            encounters=[encounter],
            labs=[
                LabObservation(
                    name="WBC",
                    value=15.2,
                    unit="K/uL",
                    reference_low=4.5,
                    reference_high=11.0,
                    flag="H",
                    effective_time=now,
                ),
                LabObservation(
                    name="Lactate",
                    value=3.4,
                    unit="mmol/L",
                    reference_low=0.5,
                    reference_high=2.0,
                    flag="H",
                    effective_time=now,
                ),
            ],
            vitals=[
                VitalObservation(name="HR", value=112, unit="/min", effective_time=now),
                VitalObservation(name="SBP", value=92, unit="mmHg", effective_time=now),
                VitalObservation(name="SpO2", value=94, unit="%", effective_time=now),
            ],
            provenance=Provenance(generator="structured-generator", created_at=now),
        )


def _normalize_base_time(value) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
        except ValueError:
            pass
    return "2026-01-01T00:00:00"
