from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import NamedTuple
from uuid import NAMESPACE_URL, uuid5

from casecrawler.models.dataset import GenerationRequest
from casecrawler.models.synthetic import (
    AllergyIntolerance,
    ClinicalOrder,
    Code,
    ComplexityProfile,
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


@dataclass(frozen=True)
class ClinicalProfileCatalogItem:
    key: str
    keywords: tuple[str, ...]
    diagnosis_display: str
    diagnosis_code: str
    lab_names: list[str]
    vital_names: list[str]
    medication_names: list[str]


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
            demographics=_patient_demographics(req.cohort_constraints, age, sex, index),
            social_history=_social_history_for_index(req.cohort_constraints, index),
        )
        encounters = _encounter_timeline(
            stable_prefix=stable_prefix,
            base_time=now,
            topic=req.topic,
            primary_diagnosis=Code(
                system="synthetic",
                code=profile.diagnosis_code,
                display=profile.diagnosis_display,
            ),
            complexity_diagnoses=_complexity_diagnoses(req.complexity, req.topic),
            procedures=_complexity_procedures(req.complexity, req.topic),
            encounter_count=_encounter_count(req.cohort_constraints),
        )
        include_labs = _includes_modality(req, "labs")
        include_vitals = _includes_modality(req, "vitals")
        include_medications = _includes_modality(req, "structured_ehr")
        lab_templates = [
            *profile.labs,
            *_complexity_labs(req.complexity, req.topic),
        ]
        vital_templates = [
            *profile.vitals,
            *_complexity_vitals(req.complexity, req.topic),
        ]
        labs = _timeline_labs(lab_templates, encounters, index) if include_labs else []
        vitals = (
            _timeline_vitals(vital_templates, encounters, index) if include_vitals else []
        )
        medication_history = (
            [
                _medication_statement(medication, now[:10])
                for medication in [
                    *profile.medications,
                    *_complexity_medications(req.complexity, req.topic),
                ]
            ]
            if include_medications
            else []
        )
        allergies = (
            _allergies_for_index(req.cohort_constraints, index, now[:10])
            if include_medications
            else []
        )
        return SyntheticRecord(
            record_id=f"rec-{uuid5(NAMESPACE_URL, f'{stable_prefix}:record')}",
            dataset_id=dataset_id,
            topic=req.topic,
            complexity=req.complexity,
            modalities=req.modalities,
            patient=patient,
            encounters=encounters,
            labs=labs,
            vitals=vitals,
            medication_history=medication_history,
            allergies=allergies,
            orders=_orders_for_record(
                encounters=encounters,
                labs=labs,
                medications=medication_history,
                procedures=encounters[0].procedures if encounters else [],
                topic=req.topic,
            )
            if include_medications
            else [],
            provenance=Provenance(generator="structured-generator", created_at=now),
            metadata={
                "cohort_constraints": _metadata_cohort_constraints(
                    req.cohort_constraints
                ),
                "clinical_profile": profile.diagnosis_code,
                "requested_export_formats": [
                    export_format.value for export_format in req.export_formats
                ],
                "require_human_review": req.require_human_review,
                "generation_overrides": _metadata_generation_overrides(req),
            },
        )


def _includes_modality(req: GenerationRequest, modality_value: str) -> bool:
    return any(modality.value == modality_value for modality in req.modalities)


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


def _encounter_count(cohort_constraints: dict) -> int:
    count = _coerce_int(cohort_constraints.get("encounter_count", 1), "encounter_count")
    if count < 1:
        raise ValueError("cohort_constraints.encounter_count must be at least 1.")
    if count > 30:
        raise ValueError("cohort_constraints.encounter_count must be <= 30.")
    return count


def _encounter_timeline(
    *,
    stable_prefix: str,
    base_time: str,
    topic: str,
    primary_diagnosis: Code,
    complexity_diagnoses: list[Code],
    procedures: list[Code],
    encounter_count: int,
) -> list[Encounter]:
    start = datetime.fromisoformat(base_time)
    encounters = []
    diagnoses = [primary_diagnosis, *complexity_diagnoses]
    for index in range(encounter_count):
        encounter_start = start + timedelta(days=index)
        encounter_end = encounter_start + timedelta(hours=6)
        encounters.append(
            Encounter(
                encounter_id=f"enc-{uuid5(NAMESPACE_URL, f'{stable_prefix}:encounter:{index}')}",
                start=encounter_start.isoformat(),
                end=encounter_end.isoformat(),
                setting="emergency_department" if index == 0 else "outpatient_follow_up",
                reason=topic if index == 0 else f"{topic} follow-up {index + 1}",
                diagnoses=diagnoses,
                procedures=procedures if index == 0 else [],
            )
        )
    return encounters


def _timeline_labs(
    templates: list[dict],
    encounters: list[Encounter],
    record_index: int,
) -> list[LabObservation]:
    return [
        _lab_observation(
            template,
            _offset_time(encounter.start, hours=1),
            record_index + encounter_index,
        )
        for encounter_index, encounter in enumerate(encounters)
        for template in templates
    ]


def _timeline_vitals(
    templates: list[dict],
    encounters: list[Encounter],
    record_index: int,
) -> list[VitalObservation]:
    return [
        _vital_observation(
            template,
            _offset_time(encounter.start, minutes=15),
            record_index + encounter_index,
        )
        for encounter_index, encounter in enumerate(encounters)
        for template in templates
    ]


def _offset_time(value: str, *, hours: int = 0, minutes: int = 0) -> str:
    return (datetime.fromisoformat(value) + timedelta(hours=hours, minutes=minutes)).isoformat()


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


def _patient_demographics(
    cohort_constraints: dict,
    age: int,
    sex: str,
    index: int,
) -> dict:
    return {
        "age_group": _age_group(age),
        "sex_at_generation": sex,
        "race": _cycle_constraint(
            cohort_constraints,
            "races",
            ["synthetic_white", "synthetic_black", "synthetic_asian", "synthetic_other"],
            index,
        ),
        "ethnicity": _cycle_constraint(
            cohort_constraints,
            "ethnicities",
            ["synthetic_not_hispanic_or_latino", "synthetic_hispanic_or_latino"],
            index,
        ),
        "insurance": _cycle_constraint(
            cohort_constraints,
            "insurance",
            ["synthetic_medicare", "synthetic_private", "synthetic_medicaid"],
            index,
        ),
    }


def _social_history_for_index(cohort_constraints: dict, index: int) -> dict:
    return {
        "smoking_status": _cycle_constraint(
            cohort_constraints,
            "smoking_statuses",
            ["never", "former", "current"],
            index,
        ),
        "alcohol_use": _cycle_constraint(
            cohort_constraints,
            "alcohol_use",
            ["none", "occasional", "moderate"],
            index,
        ),
        "housing": _cycle_constraint(
            cohort_constraints,
            "housing",
            ["stable", "unstable"],
            index,
        ),
    }


def _cycle_constraint(
    cohort_constraints: dict,
    key: str,
    fallback: list[str],
    index: int,
) -> str:
    configured = cohort_constraints.get(key)
    if configured is None:
        values = fallback
    elif isinstance(configured, str):
        values = [part.strip() for part in configured.split(",") if part.strip()]
    elif isinstance(configured, list):
        values = [str(part).strip() for part in configured if str(part).strip()]
    else:
        raise ValueError(f"cohort_constraints.{key} must be a list or comma-separated string.")
    if not values:
        raise ValueError(f"cohort_constraints.{key} must contain at least one value.")
    return values[index % len(values)]


def _age_group(age: int) -> str:
    if age < 18:
        return "pediatric"
    if age < 45:
        return "adult"
    if age < 65:
        return "middle_aged"
    return "older_adult"


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
        "races",
        "ethnicities",
        "insurance",
        "smoking_statuses",
        "alcohol_use",
        "housing",
        "allergies",
        "orders",
        "base_time",
        "topic_mix",
        "topic_mix_weights",
        "encounter_count",
    ]
    metadata = {
        key: cohort_constraints[key]
        for key in preserved_keys
        if key in cohort_constraints
    }
    if "base_time" in metadata:
        metadata["base_time"] = _normalize_base_time(metadata["base_time"])
    return metadata


def _metadata_generation_overrides(req: GenerationRequest) -> dict:
    optional_fields = [
        "clinical_text_backend",
        "clinical_text_noise_profile",
        "recipe",
        "llm_provider",
        "llm_model",
        "ollama_base_url",
        "imaging_backend",
        "imaging_model_profile",
        "diffusers_model_id",
        "time_series_backend",
        "time_series_model_profile",
        "time_series_command",
        "require_human_review",
        "validation_threshold",
    ]
    metadata = {}
    for field in optional_fields:
        value = getattr(req, field)
        if value is None:
            continue
        if field == "require_human_review" and value is False:
            continue
        metadata[field] = value
    return metadata


def _complexity_diagnoses(complexity: ComplexityProfile, topic: str) -> list[Code]:
    if complexity in {ComplexityProfile.SIMPLE, ComplexityProfile.MODERATE}:
        return []
    diagnoses = [
        Code(
            system="synthetic",
            code="type_2_diabetes_mellitus",
            display="type 2 diabetes mellitus",
        ),
        Code(
            system="synthetic",
            code="chronic_kidney_disease",
            display="chronic kidney disease",
        ),
    ]
    if complexity == ComplexityProfile.RARE:
        diagnoses.append(
            Code(
                system="synthetic",
                code=_rare_complication_code(topic),
                display=_rare_complication_display(topic),
            )
        )
    return diagnoses


def _complexity_procedures(complexity: ComplexityProfile, topic: str) -> list[Code]:
    if complexity == ComplexityProfile.SIMPLE:
        return []
    if complexity == ComplexityProfile.RARE:
        return [
            Code(
                system="synthetic",
                code=_rare_procedure_code(topic),
                display=_rare_procedure_display(topic),
            )
        ]
    if complexity == ComplexityProfile.COMPLEX:
        return [
            Code(
                system="synthetic",
                code="specialty_consultation",
                display="specialty consultation",
            )
        ]
    return []


def _complexity_labs(complexity: ComplexityProfile, topic: str) -> list[dict]:
    if complexity in {ComplexityProfile.SIMPLE, ComplexityProfile.MODERATE}:
        return []
    labs = [
        _lab("Albumin", 2.8, "g/dL", reference_low=3.5, reference_high=5.0, flag="L", step=-0.1),
        _lab("Magnesium", 1.5, "mg/dL", reference_low=1.7, reference_high=2.4, flag="L", step=-0.1),
    ]
    if complexity == ComplexityProfile.RARE:
        labs.append(_rare_lab(topic))
    return labs


def _complexity_vitals(complexity: ComplexityProfile, topic: str) -> list[dict]:
    if complexity in {ComplexityProfile.SIMPLE, ComplexityProfile.MODERATE}:
        return []
    vitals = [_vital("Pain score", 7, "0-10", step=1)]
    if complexity == ComplexityProfile.RARE:
        vitals.append(_vital("Mean arterial pressure", 62, "mmHg", step=-2))
    return vitals


def _complexity_medications(complexity: ComplexityProfile, topic: str) -> list[dict]:
    if complexity in {ComplexityProfile.SIMPLE, ComplexityProfile.MODERATE}:
        return []
    medications = [
        _med("Insulin glargine", rxnorm="274783", dose="20 units", route="subcutaneous", frequency="nightly"),
        _med("Heparin prophylaxis", rxnorm="5224", dose="5000 units", route="subcutaneous", frequency="every 8 hours"),
    ]
    if complexity == ComplexityProfile.RARE:
        medications.append(_rare_medication(topic))
    return medications


def _rare_complication_code(topic: str) -> str:
    return re.sub(r"\W+", "_", _rare_complication_display(topic).lower()).strip("_")


def _rare_complication_display(topic: str) -> str:
    normalized = topic.lower()
    if "pneumonia" in normalized:
        return "necrotizing pneumonia"
    if "heart failure" in normalized:
        return "cardiorenal syndrome"
    if "diabetic ketoacidosis" in normalized or "dka" in normalized:
        return "cerebral edema"
    if "stroke" in normalized:
        return "basilar artery occlusion"
    if "sepsis" in normalized:
        return "distributive shock with disseminated intravascular coagulation"
    return "rare disease complication"


def _rare_procedure_code(topic: str) -> str:
    return re.sub(r"\W+", "_", _rare_procedure_display(topic).lower()).strip("_")


def _rare_procedure_display(topic: str) -> str:
    normalized = topic.lower()
    if "stroke" in normalized:
        return "mechanical thrombectomy evaluation"
    if "heart failure" in normalized:
        return "right heart catheterization"
    if "sepsis" in normalized:
        return "central venous catheter placement"
    return "advanced diagnostic consultation"


def _rare_lab(topic: str) -> dict:
    normalized = topic.lower()
    if "sepsis" in normalized:
        return _lab("Fibrinogen", 95, "mg/dL", reference_low=200, reference_high=400, flag="L", step=-10)
    if "heart failure" in normalized:
        return _lab("Mixed venous oxygen saturation", 52, "%", reference_low=60, reference_high=80, flag="L", step=-2)
    if "stroke" in normalized:
        return _lab("Anti-Xa level", 0.1, "IU/mL", reference_low=0.3, reference_high=0.7, flag="L", step=0.0)
    return _lab("Ferritin", 1850, "ng/mL", reference_low=20, reference_high=300, flag="H", step=120)


def _rare_medication(topic: str) -> dict:
    normalized = topic.lower()
    if "sepsis" in normalized:
        return _med("Norepinephrine", rxnorm="7512", dose="0.08 mcg/kg/min", route="IV", frequency="continuous")
    if "heart failure" in normalized:
        return _med("Dobutamine", rxnorm="3616", dose="2.5 mcg/kg/min", route="IV", frequency="continuous")
    if "stroke" in normalized:
        return _med("Alteplase", rxnorm="8410", dose="0.9 mg/kg", route="IV", frequency="once")
    return _med("Specialty-directed rescue therapy", route="IV", frequency="per protocol")


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


def list_clinical_profile_catalog() -> list[ClinicalProfileCatalogItem]:
    return [
        ClinicalProfileCatalogItem(
            key=profile.diagnosis_code,
            keywords=keywords,
            diagnosis_display=profile.diagnosis_display,
            diagnosis_code=profile.diagnosis_code,
            lab_names=[lab["name"] for lab in profile.labs],
            vital_names=[vital["name"] for vital in profile.vitals],
            medication_names=[medication["name"] for medication in profile.medications],
        )
        for keywords, profile in _TOPIC_PROFILES
    ]


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


def _allergies_for_index(
    cohort_constraints: dict,
    index: int,
    recorded_at: str,
) -> list[AllergyIntolerance]:
    configured = cohort_constraints.get("allergies")
    if configured is None:
        templates = _DEFAULT_ALLERGY_CYCLE
    elif isinstance(configured, str):
        values = [part.strip() for part in configured.split(",") if part.strip()]
        templates = [{"substance": value} for value in values]
    elif isinstance(configured, list):
        templates = [
            item if isinstance(item, dict) else {"substance": str(item).strip()}
            for item in configured
            if (isinstance(item, dict) or str(item).strip())
        ]
    else:
        raise ValueError(
            "cohort_constraints.allergies must be a list or comma-separated string."
        )
    if not templates:
        return []
    template = templates[index % len(templates)]
    substance = str(template.get("substance") or template.get("name") or "").strip()
    if not substance or substance.lower() in {
        "none",
        "nkda",
        "no known allergies",
        "no known drug allergies",
    }:
        return []
    return [
        AllergyIntolerance(
            substance=substance,
            code=template.get("code"),
            system=template.get("system") or "synthetic",
            reaction=template.get("reaction") or "rash",
            severity=template.get("severity") or "mild",
            status=template.get("status") or "active",
            recorded_at=template.get("recorded_at") or recorded_at,
        )
    ]


def _orders_for_record(
    *,
    encounters: list[Encounter],
    labs: list[LabObservation],
    medications: list[MedicationStatement],
    procedures: list[Code],
    topic: str,
) -> list[ClinicalOrder]:
    if not encounters:
        return []
    encounter = encounters[0]
    ordered_at = encounter.start
    orders: list[ClinicalOrder] = []
    orders.extend(
        ClinicalOrder(
            order_id=f"{encounter.encounter_id}-order-lab-{_slug(lab.name)}",
            order_type="laboratory",
            display=lab.name,
            code=lab.loinc,
            system="LOINC" if lab.loinc else "synthetic",
            status="completed",
            priority="routine",
            ordered_at=ordered_at,
            encounter_id=encounter.encounter_id,
        )
        for lab in labs[:4]
    )
    orders.extend(
        ClinicalOrder(
            order_id=f"{encounter.encounter_id}-order-med-{_slug(medication.name)}",
            order_type="medication",
            display=medication.name,
            code=medication.rxnorm,
            system="RxNorm" if medication.rxnorm else "synthetic",
            status="active",
            priority="stat" if medication.route and medication.route.lower() == "iv" else "routine",
            ordered_at=ordered_at,
            encounter_id=encounter.encounter_id,
        )
        for medication in medications[:4]
    )
    orders.extend(
        ClinicalOrder(
            order_id=f"{encounter.encounter_id}-order-procedure-{_slug(procedure.code)}",
            order_type="procedure",
            display=procedure.display,
            code=procedure.code,
            system=procedure.system,
            status="completed",
            priority="routine",
            ordered_at=ordered_at,
            encounter_id=encounter.encounter_id,
        )
        for procedure in procedures
    )
    if _topic_needs_imaging_order(topic):
        orders.append(
            ClinicalOrder(
                order_id=f"{encounter.encounter_id}-order-imaging-{_slug(topic)}",
                order_type="imaging",
                display=f"{topic} imaging study",
                code="synthetic_imaging_study",
                system="synthetic",
                status="completed",
                priority="urgent",
                ordered_at=ordered_at,
                encounter_id=encounter.encounter_id,
            )
        )
    orders.append(
        ClinicalOrder(
            order_id=f"{encounter.encounter_id}-order-nursing-monitoring",
            order_type="nursing",
            display="vital signs and intake/output monitoring",
            code="nursing_monitoring",
            system="synthetic",
            status="active",
            priority="routine",
            ordered_at=ordered_at,
            encounter_id=encounter.encounter_id,
        )
    )
    return orders


def _topic_needs_imaging_order(topic: str) -> bool:
    normalized = topic.lower()
    return any(
        keyword in normalized
        for keyword in (
            "pneumonia",
            "pulmonary",
            "stroke",
            "appendicitis",
            "pancreatitis",
            "trauma",
            "chest pain",
        )
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


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "order"


_DEFAULT_ALLERGY_CYCLE = [
    {
        "substance": "Sulfonamide antibiotics",
        "code": "91939003",
        "system": "SNOMED-CT",
        "reaction": "rash",
        "severity": "moderate",
    },
    {
        "substance": "Morphine",
        "code": "7052",
        "system": "RxNorm",
        "reaction": "nausea",
        "severity": "mild",
    },
    {"substance": "No known drug allergies", "status": "inactive"},
]


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
        ("type 2 diabetes", "diabetes mellitus", "a1c", "diabetic follow-up"),
        ClinicalProfile(
            diagnosis_display="type 2 diabetes mellitus",
            diagnosis_code="type_2_diabetes_mellitus",
            labs=[
                _lab("Hemoglobin A1c", 8.4, "%", reference_low=4.0, reference_high=5.6, flag="H", step=-0.1),
                _lab("Glucose", 178, "mg/dL", reference_low=70, reference_high=110, flag="H", step=-4),
                _lab("eGFR", 82, "mL/min/1.73m2", reference_low=60, reference_high=120, step=-1),
                _lab("Urine albumin-creatinine ratio", 86, "mg/g", reference_low=0, reference_high=30, flag="H", step=5),
            ],
            vitals=[
                _vital("HR", 84, "/min", step=1),
                _vital("SBP", 142, "mmHg", step=1),
                _vital("SpO2", 98, "%", step=0),
                _vital("Temperature", 36.9, "C", step=0.0),
            ],
            medications=[
                _med("Metformin", rxnorm="6809", dose="1000 mg", route="oral", frequency="twice daily"),
                _med("Empagliflozin", rxnorm="1545653", dose="10 mg", route="oral", frequency="daily"),
                _med("Insulin glargine", rxnorm="274783", dose="18 units", route="subcutaneous", frequency="nightly"),
            ],
        ),
    ),
    (
        ("chronic kidney disease", "ckd", "proteinuria"),
        ClinicalProfile(
            diagnosis_display="chronic kidney disease",
            diagnosis_code="chronic_kidney_disease",
            labs=[
                _lab("Creatinine", 2.0, "mg/dL", reference_low=0.6, reference_high=1.3, flag="H", step=0.05),
                _lab("eGFR", 34, "mL/min/1.73m2", reference_low=60, reference_high=120, flag="L", step=-1),
                _lab("Potassium", 4.8, "mmol/L", reference_low=3.5, reference_high=5.1, step=0.05),
                _lab("Urine albumin-creatinine ratio", 420, "mg/g", reference_low=0, reference_high=30, flag="H", step=20),
            ],
            vitals=[
                _vital("HR", 78, "/min", step=1),
                _vital("SBP", 154, "mmHg", step=2),
                _vital("SpO2", 97, "%", step=0),
                _vital("Temperature", 36.8, "C", step=0.0),
            ],
            medications=[
                _med("Losartan", rxnorm="52175", dose="50 mg", route="oral", frequency="daily"),
                _med("Furosemide", rxnorm="4603", dose="20 mg", route="oral", frequency="daily"),
                _med("Sodium bicarbonate", rxnorm="36676", dose="650 mg", route="oral", frequency="twice daily"),
            ],
        ),
    ),
    (
        ("hypertension", "high blood pressure", "bp follow-up"),
        ClinicalProfile(
            diagnosis_display="essential hypertension",
            diagnosis_code="essential_hypertension",
            labs=[
                _lab("Creatinine", 1.0, "mg/dL", reference_low=0.6, reference_high=1.3, step=0.0),
                _lab("Potassium", 4.0, "mmol/L", reference_low=3.5, reference_high=5.1, step=0.0),
                _lab("Sodium", 139, "mmol/L", reference_low=135, reference_high=145, step=0.0),
                _lab("LDL cholesterol", 132, "mg/dL", reference_low=0, reference_high=100, flag="H", step=-3),
            ],
            vitals=[
                _vital("HR", 76, "/min", step=1),
                _vital("SBP", 166, "mmHg", step=2),
                _vital("SpO2", 98, "%", step=0),
                _vital("Temperature", 36.7, "C", step=0.0),
            ],
            medications=[
                _med("Amlodipine", rxnorm="17767", dose="10 mg", route="oral", frequency="daily"),
                _med("Hydrochlorothiazide", rxnorm="5487", dose="25 mg", route="oral", frequency="daily"),
                _med("Atorvastatin", rxnorm="83367", dose="40 mg", route="oral", frequency="daily"),
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
    (
        ("asthma", "status asthmaticus"),
        ClinicalProfile(
            diagnosis_display="asthma exacerbation",
            diagnosis_code="asthma_exacerbation",
            labs=[
                _lab("pCO2", 48, "mmHg", reference_low=35, reference_high=45, flag="H", step=1),
                _lab("Eosinophils", 0.7, "K/uL", reference_low=0, reference_high=0.5, flag="H", step=0.1),
                _lab("Potassium", 3.4, "mmol/L", reference_low=3.5, reference_high=5.1, flag="L", step=-0.1),
            ],
            vitals=[
                _vital("HR", 124, "/min", step=3),
                _vital("SBP", 132, "mmHg", step=2),
                _vital("SpO2", 89, "%", step=-1),
                _vital("Respiratory rate", 34, "/min", step=1),
            ],
            medications=[
                _med("Albuterol", rxnorm="435", dose="5 mg", route="nebulized", frequency="continuous"),
                _med("Ipratropium", rxnorm="7213", dose="0.5 mg", route="nebulized", frequency="every 20 minutes"),
                _med("Magnesium sulfate", rxnorm="6585", dose="2 g", route="IV", frequency="once"),
                _med("Methylprednisolone", rxnorm="6902", dose="125 mg", route="IV", frequency="once"),
            ],
        ),
    ),
    (
        ("pancreatitis", "epigastric pain"),
        ClinicalProfile(
            diagnosis_display="acute pancreatitis",
            diagnosis_code="acute_pancreatitis",
            labs=[
                _lab("Lipase", 1240, "U/L", reference_low=0, reference_high=160, flag="H", step=80),
                _lab("WBC", 14.1, "K/uL", reference_low=4.5, reference_high=11.0, flag="H", step=0.4),
                _lab("Calcium", 8.1, "mg/dL", reference_low=8.5, reference_high=10.5, flag="L", step=-0.1),
                _lab("BUN", 27, "mg/dL", reference_low=7, reference_high=20, flag="H", step=2),
            ],
            vitals=[
                _vital("HR", 112, "/min", step=3),
                _vital("SBP", 106, "mmHg", step=-2),
                _vital("Temperature", 37.9, "C", step=0.1),
                _vital("SpO2", 96, "%", step=0),
            ],
            medications=[
                _med("Lactated Ringer's", dose="200 mL/hr", route="IV", frequency="continuous"),
                _med("Hydromorphone", rxnorm="3423", dose="0.5 mg", route="IV", frequency="every 3 hours as needed"),
                _med("Ondansetron", rxnorm="26225", dose="4 mg", route="IV", frequency="every 6 hours as needed"),
            ],
        ),
    ),
    (
        ("appendicitis", "right lower quadrant", "rlq pain"),
        ClinicalProfile(
            diagnosis_display="acute appendicitis",
            diagnosis_code="acute_appendicitis",
            labs=[
                _lab("WBC", 16.4, "K/uL", reference_low=4.5, reference_high=11.0, flag="H", step=0.4),
                _lab("CRP", 72, "mg/L", reference_low=0, reference_high=10, flag="H", step=5),
                _lab("Creatinine", 0.9, "mg/dL", reference_low=0.6, reference_high=1.3, step=0.0),
            ],
            vitals=[
                _vital("HR", 106, "/min", step=2),
                _vital("SBP", 122, "mmHg", step=1),
                _vital("Temperature", 38.1, "C", step=0.1),
                _vital("SpO2", 98, "%", step=0),
            ],
            medications=[
                _med("Ceftriaxone", rxnorm="2193", dose="2 g", route="IV", frequency="once"),
                _med("Metronidazole", rxnorm="6922", dose="500 mg", route="IV", frequency="every 8 hours"),
                _med("Morphine", rxnorm="7052", dose="4 mg", route="IV", frequency="as needed"),
            ],
        ),
    ),
    (
        ("pyelonephritis", "uti", "urinary tract infection", "flank pain"),
        ClinicalProfile(
            diagnosis_display="acute pyelonephritis",
            diagnosis_code="acute_pyelonephritis",
            labs=[
                _lab("WBC", 14.7, "K/uL", reference_low=4.5, reference_high=11.0, flag="H", step=0.4),
                _lab("Urine WBC", 80, "/HPF", reference_low=0, reference_high=5, flag="H", step=5),
                _lab("Nitrite", 1, "positive", reference_low=0, reference_high=0, flag="H", step=0),
                _lab("Creatinine", 1.2, "mg/dL", reference_low=0.6, reference_high=1.3, step=0.1),
            ],
            vitals=[
                _vital("HR", 114, "/min", step=2),
                _vital("SBP", 104, "mmHg", step=-2),
                _vital("Temperature", 39.0, "C", step=0.1),
                _vital("SpO2", 97, "%", step=0),
            ],
            medications=[
                _med("Ceftriaxone", rxnorm="2193", dose="1 g", route="IV", frequency="daily"),
                _med("Ketorolac", rxnorm="35827", dose="15 mg", route="IV", frequency="once"),
                _med("Normal saline", dose="1 L", route="IV", frequency="bolus"),
            ],
        ),
    ),
    (
        ("meningitis", "neck stiffness", "photophobia"),
        ClinicalProfile(
            diagnosis_display="bacterial meningitis",
            diagnosis_code="bacterial_meningitis",
            labs=[
                _lab("WBC", 18.6, "K/uL", reference_low=4.5, reference_high=11.0, flag="H", step=0.5),
                _lab("CSF WBC", 820, "cells/uL", reference_low=0, reference_high=5, flag="H", step=40),
                _lab("CSF glucose", 32, "mg/dL", reference_low=40, reference_high=70, flag="L", step=-2),
                _lab("CSF protein", 180, "mg/dL", reference_low=15, reference_high=45, flag="H", step=10),
            ],
            vitals=[
                _vital("HR", 118, "/min", step=3),
                _vital("SBP", 128, "mmHg", step=2),
                _vital("Temperature", 39.4, "C", step=0.1),
                _vital("SpO2", 96, "%", step=0),
            ],
            medications=[
                _med("Ceftriaxone", rxnorm="2193", dose="2 g", route="IV", frequency="every 12 hours"),
                _med("Vancomycin", rxnorm="11124", dose="15 mg/kg", route="IV", frequency="every 12 hours"),
                _med("Dexamethasone", rxnorm="3264", dose="10 mg", route="IV", frequency="every 6 hours"),
            ],
        ),
    ),
    (
        ("seizure", "status epilepticus", "postictal"),
        ClinicalProfile(
            diagnosis_display="generalized seizure",
            diagnosis_code="generalized_seizure",
            labs=[
                _lab("Sodium", 128, "mmol/L", reference_low=135, reference_high=145, flag="L", step=-1),
                _lab("Glucose", 68, "mg/dL", reference_low=70, reference_high=110, flag="L", step=-2),
                _lab("Lactate", 5.6, "mmol/L", reference_low=0.5, reference_high=2.0, flag="H", step=0.3),
                _lab("Creatine kinase", 680, "U/L", reference_low=30, reference_high=200, flag="H", step=60),
            ],
            vitals=[
                _vital("HR", 128, "/min", step=3),
                _vital("SBP", 146, "mmHg", step=2),
                _vital("SpO2", 93, "%", step=-1),
                _vital("Temperature", 37.6, "C", step=0.0),
            ],
            medications=[
                _med("Lorazepam", rxnorm="6470", dose="2 mg", route="IV", frequency="once"),
                _med("Levetiracetam", rxnorm="114477", dose="1500 mg", route="IV", frequency="once"),
                _med("Dextrose 50%", dose="25 g", route="IV", frequency="as needed"),
            ],
        ),
    ),
]
