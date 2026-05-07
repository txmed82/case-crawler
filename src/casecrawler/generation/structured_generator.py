from __future__ import annotations

import json
import re
from datetime import datetime
from typing import NamedTuple
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


class ClinicalProfile(NamedTuple):
    diagnosis_display: str
    diagnosis_code: str
    labs: list[dict]
    vitals: list[dict]
    medications: list[dict]


class StructuredGenerator:
    def generate(
        self,
        dataset_id: str,
        req: GenerationRequest,
        index: int,
    ) -> SyntheticRecord:
        now = _normalize_base_time(req.cohort_constraints.get("base_time"))
        stable_prefix = _stable_record_seed(dataset_id, req, index)
        age = _age_for_index(req.cohort_constraints, index)
        sex = _sex_for_index(req.cohort_constraints, index)
        profile = _profile_for_topic(req.topic)
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
                    code=profile.diagnosis_code,
                    display=profile.diagnosis_display,
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
            labs=[_lab_observation(lab, now, index) for lab in profile.labs],
            vitals=[_vital_observation(vital, now, index) for vital in profile.vitals],
            medication_history=[
                _medication_statement(medication, now[:10])
                for medication in profile.medications
            ],
            provenance=Provenance(generator="structured-generator", created_at=now),
            metadata={
                "cohort_constraints": _metadata_cohort_constraints(
                    req.cohort_constraints
                ),
                "clinical_profile": profile.diagnosis_code,
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


def _stable_record_seed(dataset_id: str, req: GenerationRequest, index: int) -> str:
    canonical_constraints = dict(req.cohort_constraints)
    if "base_time" in canonical_constraints:
        canonical_constraints["base_time"] = _normalize_base_time(
            canonical_constraints["base_time"]
        )
    constraints = json.dumps(canonical_constraints, sort_keys=True, default=str)
    modalities = ",".join(sorted(modality.value for modality in req.modalities))
    return (
        f"{dataset_id}:{req.topic}:{req.complexity.value}:"
        f"{modalities}:{constraints}:{index}"
    )


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


def _profile_for_topic(topic: str) -> ClinicalProfile:
    normalized = re.sub(r"\s+", " ", topic.lower().replace("-", " ").replace("_", " "))
    for keywords, profile in _TOPIC_PROFILES:
        if any(keyword in normalized for keyword in keywords):
            return profile
    return ClinicalProfile(
        diagnosis_display=topic,
        diagnosis_code=re.sub(r"\W+", "_", topic.lower()).strip("_") or "general",
        labs=[],
        vitals=[
            _vital("HR", 82, "/min"),
            _vital("SBP", 124, "mmHg"),
            _vital("SpO2", 98, "%"),
        ],
        medications=[],
    )


def _lab_observation(template: dict, effective_time: str, index: int) -> LabObservation:
    value = _indexed_value(template["value"], index, template.get("step", 0.0))
    return LabObservation(
        name=template["name"],
        loinc=template.get("loinc"),
        value=value,
        unit=template["unit"],
        reference_low=template.get("reference_low"),
        reference_high=template.get("reference_high"),
        flag=template.get("flag"),
        effective_time=effective_time,
        specimen=template.get("specimen"),
    )


def _vital_observation(template: dict, effective_time: str, index: int) -> VitalObservation:
    value = _indexed_value(template["value"], index, template.get("step", 0.0))
    return VitalObservation(
        name=template["name"],
        value=value,
        unit=template["unit"],
        effective_time=effective_time,
    )


def _medication_statement(template: dict, start: str) -> MedicationStatement:
    return MedicationStatement(
        name=template["name"],
        rxnorm=template.get("rxnorm"),
        dose=template.get("dose"),
        route=template.get("route"),
        frequency=template.get("frequency"),
        status=template.get("status", "active"),
        start=start,
        end=template.get("end"),
    )


def _indexed_value(value: float, index: int, step: float) -> float:
    adjusted = value + (index % 3) * step
    return round(adjusted, 2)


def _lab(
    name: str,
    value: float,
    unit: str,
    *,
    reference_low: float | None = None,
    reference_high: float | None = None,
    flag: str | None = None,
    loinc: str | None = None,
    step: float = 0.0,
) -> dict:
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "reference_low": reference_low,
        "reference_high": reference_high,
        "flag": flag,
        "loinc": loinc,
        "step": step,
    }


def _vital(name: str, value: float, unit: str, *, step: float = 0.0) -> dict:
    return {"name": name, "value": value, "unit": unit, "step": step}


def _med(
    name: str,
    *,
    rxnorm: str | None = None,
    dose: str | None = None,
    route: str | None = None,
    frequency: str | None = None,
) -> dict:
    return {
        "name": name,
        "rxnorm": rxnorm,
        "dose": dose,
        "route": route,
        "frequency": frequency,
    }


_TOPIC_PROFILES: list[tuple[tuple[str, ...], ClinicalProfile]] = [
    (
        ("pneumonia",),
        ClinicalProfile(
            diagnosis_display="pneumonia",
            diagnosis_code="pneumonia",
            labs=[
                _lab("WBC", 13.8, "K/uL", reference_low=4.5, reference_high=11.0, flag="H", step=0.4),
                _lab("Procalcitonin", 1.4, "ng/mL", reference_low=0, reference_high=0.1, flag="H", step=0.2),
                _lab("Sodium", 134, "mmol/L", reference_low=135, reference_high=145, flag="L", step=-1),
            ],
            vitals=[
                _vital("HR", 108, "/min", step=2),
                _vital("SBP", 116, "mmHg", step=-1),
                _vital("SpO2", 91, "%", step=-1),
                _vital("Temperature", 38.4, "C", step=0.1),
                _vital("Respiratory rate", 24, "/min", step=1),
            ],
            medications=[
                _med("Ceftriaxone", rxnorm="2193", dose="1 g", route="IV", frequency="daily"),
                _med("Azithromycin", rxnorm="18631", dose="500 mg", route="IV", frequency="daily"),
                _med("Acetaminophen", rxnorm="161", dose="650 mg", route="oral", frequency="every 6 hours as needed"),
            ],
        ),
    ),
    (
        ("sepsis", "infection"),
        ClinicalProfile(
            diagnosis_display="sepsis",
            diagnosis_code="sepsis",
            labs=[
                _lab("WBC", 15.2, "K/uL", reference_low=4.5, reference_high=11.0, flag="H", step=0.4),
                _lab("Lactate", 3.4, "mmol/L", reference_low=0.5, reference_high=2.0, flag="H", loinc="2524-7", step=0.2),
                _lab("Creatinine", 1.5, "mg/dL", reference_low=0.6, reference_high=1.3, flag="H", step=0.1),
            ],
            vitals=[
                _vital("HR", 112, "/min", step=3),
                _vital("SBP", 92, "mmHg", step=-2),
                _vital("SpO2", 94, "%", step=-1),
                _vital("Temperature", 38.8, "C", step=0.1),
            ],
            medications=[
                _med("Acetaminophen", rxnorm="161", dose="650 mg", route="oral", frequency="every 6 hours as needed"),
                _med("Ceftriaxone", rxnorm="2193", dose="1 g", route="IV", frequency="daily"),
            ],
        ),
    ),
    (
        ("heart failure", "pulmonary edema", "volume overload", "edema"),
        ClinicalProfile(
            diagnosis_display="heart failure exacerbation",
            diagnosis_code="heart_failure_exacerbation",
            labs=[
                _lab("BNP", 1240, "pg/mL", reference_low=0, reference_high=100, flag="H", step=75),
                _lab("Creatinine", 1.4, "mg/dL", reference_low=0.6, reference_high=1.3, flag="H", step=0.1),
                _lab("Sodium", 132, "mmol/L", reference_low=135, reference_high=145, flag="L", step=-1),
            ],
            vitals=[
                _vital("HR", 104, "/min", step=2),
                _vital("SBP", 156, "mmHg", step=4),
                _vital("SpO2", 90, "%", step=-1),
                _vital("Respiratory rate", 26, "/min", step=1),
            ],
            medications=[
                _med("Furosemide", rxnorm="4603", dose="40 mg", route="IV", frequency="once"),
                _med("Nitroglycerin", rxnorm="4917", dose="0.4 mg", route="sublingual", frequency="as needed"),
            ],
        ),
    ),
    (
        ("diabetic ketoacidosis", "dka", "hyperglycemia"),
        ClinicalProfile(
            diagnosis_display="diabetic ketoacidosis",
            diagnosis_code="diabetic_ketoacidosis",
            labs=[
                _lab("Glucose", 486, "mg/dL", reference_low=70, reference_high=110, flag="H", step=20),
                _lab("Bicarbonate", 12, "mmol/L", reference_low=22, reference_high=29, flag="L", step=-1),
                _lab("Anion gap", 24, "mmol/L", reference_low=8, reference_high=16, flag="H", step=1),
                _lab("Beta-hydroxybutyrate", 5.1, "mmol/L", reference_low=0, reference_high=0.6, flag="H", step=0.2),
            ],
            vitals=[
                _vital("HR", 118, "/min", step=3),
                _vital("SBP", 98, "mmHg", step=-2),
                _vital("Respiratory rate", 30, "/min", step=1),
                _vital("Temperature", 37.1, "C", step=0.0),
            ],
            medications=[
                _med("Regular insulin", rxnorm="253182", dose="0.1 units/kg/hr", route="IV", frequency="continuous"),
                _med("Normal saline", dose="1 L", route="IV", frequency="bolus"),
            ],
        ),
    ),
    (
        ("stroke", "cva", "aphasia", "hemiparesis"),
        ClinicalProfile(
            diagnosis_display="ischemic stroke",
            diagnosis_code="ischemic_stroke",
            labs=[
                _lab("Glucose", 126, "mg/dL", reference_low=70, reference_high=110, flag="H", step=5),
                _lab("Platelets", 238, "K/uL", reference_low=150, reference_high=450, step=6),
                _lab("INR", 1.0, "", reference_low=0.8, reference_high=1.2, step=0.0),
            ],
            vitals=[
                _vital("HR", 88, "/min", step=2),
                _vital("SBP", 184, "mmHg", step=3),
                _vital("SpO2", 97, "%", step=0),
            ],
            medications=[
                _med("Aspirin", rxnorm="1191", dose="325 mg", route="oral", frequency="once"),
                _med("Atorvastatin", rxnorm="83367", dose="80 mg", route="oral", frequency="daily"),
            ],
        ),
    ),
    (
        ("pulmonary embolism", "pulmonary embolus", "pe ", "pleuritic chest pain"),
        ClinicalProfile(
            diagnosis_display="pulmonary embolism",
            diagnosis_code="pulmonary_embolism",
            labs=[
                _lab("D-dimer", 2.8, "mcg/mL FEU", reference_low=0, reference_high=0.5, flag="H", step=0.3),
                _lab("Troponin I", 0.03, "ng/mL", reference_low=0, reference_high=0.04, step=0.01),
                _lab("BNP", 180, "pg/mL", reference_low=0, reference_high=100, flag="H", step=20),
            ],
            vitals=[
                _vital("HR", 122, "/min", step=3),
                _vital("SBP", 108, "mmHg", step=-2),
                _vital("SpO2", 90, "%", step=-1),
                _vital("Respiratory rate", 28, "/min", step=1),
            ],
            medications=[
                _med("Heparin", rxnorm="5224", dose="80 units/kg", route="IV", frequency="bolus then infusion"),
                _med("Acetaminophen", rxnorm="161", dose="650 mg", route="oral", frequency="every 6 hours as needed"),
            ],
        ),
    ),
    (
        ("acute coronary syndrome", "myocardial infarction", "stemi", "nstemi", "chest pain"),
        ClinicalProfile(
            diagnosis_display="acute coronary syndrome",
            diagnosis_code="acute_coronary_syndrome",
            labs=[
                _lab("Troponin I", 1.8, "ng/mL", reference_low=0, reference_high=0.04, flag="H", step=0.4),
                _lab("Creatinine", 1.1, "mg/dL", reference_low=0.6, reference_high=1.3, step=0.1),
                _lab("Potassium", 4.1, "mmol/L", reference_low=3.5, reference_high=5.1, step=0.1),
            ],
            vitals=[
                _vital("HR", 96, "/min", step=2),
                _vital("SBP", 148, "mmHg", step=3),
                _vital("SpO2", 96, "%", step=0),
            ],
            medications=[
                _med("Aspirin", rxnorm="1191", dose="325 mg", route="oral", frequency="once"),
                _med("Heparin", rxnorm="5224", dose="60 units/kg", route="IV", frequency="bolus then infusion"),
                _med("Atorvastatin", rxnorm="83367", dose="80 mg", route="oral", frequency="daily"),
                _med("Nitroglycerin", rxnorm="4917", dose="0.4 mg", route="sublingual", frequency="as needed"),
            ],
        ),
    ),
    (
        ("copd", "chronic obstructive", "wheezing"),
        ClinicalProfile(
            diagnosis_display="COPD exacerbation",
            diagnosis_code="copd_exacerbation",
            labs=[
                _lab("WBC", 12.4, "K/uL", reference_low=4.5, reference_high=11.0, flag="H", step=0.3),
                _lab("pCO2", 58, "mmHg", reference_low=35, reference_high=45, flag="H", step=2),
                _lab("Bicarbonate", 31, "mmol/L", reference_low=22, reference_high=29, flag="H", step=1),
            ],
            vitals=[
                _vital("HR", 110, "/min", step=2),
                _vital("SBP", 138, "mmHg", step=2),
                _vital("SpO2", 88, "%", step=-1),
                _vital("Respiratory rate", 30, "/min", step=1),
            ],
            medications=[
                _med("Albuterol", rxnorm="435", dose="2.5 mg", route="nebulized", frequency="every 4 hours"),
                _med("Ipratropium", rxnorm="7213", dose="0.5 mg", route="nebulized", frequency="every 6 hours"),
                _med("Methylprednisolone", rxnorm="6902", dose="125 mg", route="IV", frequency="once"),
            ],
        ),
    ),
    (
        ("gi bleed", "gastrointestinal bleed", "melena", "hematemesis"),
        ClinicalProfile(
            diagnosis_display="upper gastrointestinal bleeding",
            diagnosis_code="upper_gastrointestinal_bleeding",
            labs=[
                _lab("Hemoglobin", 8.4, "g/dL", reference_low=12, reference_high=16, flag="L", step=-0.3),
                _lab("BUN", 42, "mg/dL", reference_low=7, reference_high=20, flag="H", step=3),
                _lab("INR", 1.3, "", reference_low=0.8, reference_high=1.2, flag="H", step=0.1),
            ],
            vitals=[
                _vital("HR", 118, "/min", step=3),
                _vital("SBP", 94, "mmHg", step=-2),
                _vital("SpO2", 97, "%", step=0),
            ],
            medications=[
                _med("Pantoprazole", rxnorm="40790", dose="80 mg", route="IV", frequency="bolus then infusion"),
                _med("Normal saline", dose="1 L", route="IV", frequency="bolus"),
            ],
        ),
    ),
    (
        ("acute kidney injury", "aki", "renal failure"),
        ClinicalProfile(
            diagnosis_display="acute kidney injury",
            diagnosis_code="acute_kidney_injury",
            labs=[
                _lab("Creatinine", 3.2, "mg/dL", reference_low=0.6, reference_high=1.3, flag="H", step=0.2),
                _lab("BUN", 58, "mg/dL", reference_low=7, reference_high=20, flag="H", step=4),
                _lab("Potassium", 5.6, "mmol/L", reference_low=3.5, reference_high=5.1, flag="H", step=0.1),
            ],
            vitals=[
                _vital("HR", 102, "/min", step=2),
                _vital("SBP", 104, "mmHg", step=-2),
                _vital("SpO2", 96, "%", step=0),
            ],
            medications=[
                _med("Normal saline", dose="1 L", route="IV", frequency="bolus"),
                _med("Calcium gluconate", rxnorm="1895", dose="1 g", route="IV", frequency="once"),
            ],
        ),
    ),
]
