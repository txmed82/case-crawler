from __future__ import annotations

from datetime import datetime
from uuid import uuid4

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
        now = datetime.now().isoformat()
        patient = SyntheticPatient(
            patient_id=f"pat-{uuid4()}",
            age=45 + (index % 35),
            sex="female" if index % 2 else "male",
        )
        encounter = Encounter(
            encounter_id=f"enc-{uuid4()}",
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
            record_id=f"rec-{uuid4()}",
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

