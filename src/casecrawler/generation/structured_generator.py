from __future__ import annotations

import json
import re
from datetime import datetime
from uuid import NAMESPACE_URL, uuid5

from casecrawler.models.dataset import GenerationRequest
from casecrawler.models.synthetic import (
    Code,
    Encounter,
    LabObservation,
    MedicationStatement,
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
        stable_prefix = _stable_record_seed(req, index)
        age = _age_for_index(req.cohort_constraints, index)
        sex = _sex_for_index(req.cohort_constraints, index)
        patient = SyntheticPatient(
            patient_id=f"pat-{uuid5(NAMESPACE_URL, f'{stable_prefix}:patient')}",
            age=age,
            sex=sex,
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
            medication_history=_medications_for_topic(req.topic, now[:10]),
            provenance=Provenance(generator="structured-generator", created_at=now),
            metadata={
                "cohort_constraints": _metadata_cohort_constraints(
                    req.cohort_constraints
                )
            },
        )


def _normalize_base_time(value) -> str:
    if value is None:
        return "2026-01-01T00:00:00"
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
        except ValueError as exc:
            raise ValueError(
                f"cohort_constraints.base_time must be ISO-8601, got {value!r}"
            ) from exc
    raise ValueError(
        "cohort_constraints.base_time must be a datetime or ISO-8601 string, "
        f"got {value!r}"
    )


def _stable_record_seed(req: GenerationRequest, index: int) -> str:
    canonical_constraints = dict(req.cohort_constraints)
    if "base_time" in canonical_constraints:
        canonical_constraints["base_time"] = _normalize_base_time(
            canonical_constraints["base_time"]
        )
    constraints = json.dumps(canonical_constraints, sort_keys=True, default=str)
    modalities = ",".join(sorted(modality.value for modality in req.modalities))
    return f"{req.topic}:{req.complexity.value}:{modalities}:{constraints}:{index}"


def _age_for_index(cohort_constraints: dict, index: int) -> int:
    age_min = _coerce_int(
        cohort_constraints.get("age_min", cohort_constraints.get("min_age", 45)),
        "age_min",
    )
    age_max = _coerce_int(
        cohort_constraints.get("age_max", cohort_constraints.get("max_age", 79)),
        "age_max",
    )
    if age_min > age_max:
        raise ValueError("cohort_constraints.age_min must be <= age_max.")
    span = age_max - age_min + 1
    return age_min + (index % span)


def _sex_for_index(cohort_constraints: dict, index: int) -> str:
    configured = cohort_constraints.get("sexes", cohort_constraints.get("sex_cycle"))
    if configured is None:
        sexes = ["male", "female"]
    elif isinstance(configured, str):
        sexes = [part.strip() for part in configured.split(",") if part.strip()]
    elif isinstance(configured, list):
        sexes = [str(part).strip() for part in configured if str(part).strip()]
    else:
        raise ValueError(
            "cohort_constraints.sexes must be a list or comma-separated string."
        )
    if not sexes:
        raise ValueError("cohort_constraints.sexes must contain at least one value.")
    return sexes[index % len(sexes)]


def _coerce_int(value, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"cohort_constraints.{field_name} must be an integer.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"cohort_constraints.{field_name} must be an integer."
        ) from exc


def _metadata_cohort_constraints(cohort_constraints: dict) -> dict:
    preserved_keys = [
        "age_min",
        "age_max",
        "min_age",
        "max_age",
        "sexes",
        "sex_cycle",
        "base_time",
    ]
    metadata = {
        key: cohort_constraints[key]
        for key in preserved_keys
        if key in cohort_constraints
    }
    if "base_time" in metadata:
        metadata["base_time"] = _normalize_base_time(metadata["base_time"])
    return metadata


def _medications_for_topic(topic: str, start: str) -> list[MedicationStatement]:
    normalized = re.sub(r"\s+", " ", topic.lower().replace("-", " ").replace("_", " "))
    medications = []
    if any(
        term in normalized
        for term in ["pain", "fever", "headache", "sepsis", "pneumonia", "infection"]
    ):
        medications.append(
            MedicationStatement(
                name="Acetaminophen",
                rxnorm="161",
                dose="650 mg",
                route="oral",
                frequency="every 6 hours as needed",
                status="active",
                start=start,
            )
        )
    if any(term in normalized for term in ["sepsis", "pneumonia", "infection"]):
        medications.append(
            MedicationStatement(
                name="Ceftriaxone",
                rxnorm="2193",
                dose="1 g",
                route="IV",
                frequency="daily",
                status="active",
                start=start,
            )
        )
    if any(term in normalized for term in ["heart failure", "edema", "pulmonary edema"]):
        medications.append(
            MedicationStatement(
                name="Furosemide",
                rxnorm="4603",
                dose="40 mg",
                route="IV",
                frequency="once",
                status="active",
                start=start,
            )
        )
    return medications
