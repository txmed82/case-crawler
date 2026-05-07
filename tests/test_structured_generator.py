import pytest

from casecrawler.generation.structured_generator import (
    StructuredGenerator,
    list_clinical_profile_catalog,
)
from casecrawler.models.dataset import GenerationRequest
from casecrawler.models.synthetic import Modality


def test_structured_generator_ids_are_scoped_to_dataset_id():
    req = GenerationRequest(topic="sepsis", cohort_constraints={"base_time": "2026-02-03T04:05:06"})
    generator = StructuredGenerator()

    first = generator.generate("ds-one", req, 0)
    second = generator.generate("ds-two", req, 0)
    repeat = generator.generate("ds-one", req, 0)

    assert first.record_id != second.record_id
    assert first.patient.patient_id != second.patient.patient_id
    assert first.encounters[0].encounter_id != second.encounters[0].encounter_id
    assert first.dataset_id != second.dataset_id
    assert first.record_id == repeat.record_id


def test_structured_generator_rejects_invalid_base_time():
    req = GenerationRequest(topic="sepsis", cohort_constraints={"base_time": "not-a-date"})

    with pytest.raises(ValueError, match="base_time must be ISO-8601"):
        StructuredGenerator().generate("ds-one", req, 0)


def test_structured_generator_canonicalizes_base_time_for_seed():
    first_req = GenerationRequest(
        topic="sepsis",
        cohort_constraints={"base_time": "2026-02-03T04:05:06Z"},
    )
    second_req = GenerationRequest(
        topic="sepsis",
        cohort_constraints={"base_time": "2026-02-03T04:05:06+00:00"},
    )
    generator = StructuredGenerator()

    first = generator.generate("ds-one", first_req, 0)
    second = generator.generate("ds-one", second_req, 0)

    assert first.record_id == second.record_id
    assert first.provenance.created_at == second.provenance.created_at


def test_structured_generator_seed_sorts_modalities():
    first_req = GenerationRequest(
        topic="sepsis",
        modalities=[Modality.CLINICAL_TEXT, Modality.TIME_SERIES],
    )
    second_req = GenerationRequest(
        topic="sepsis",
        modalities=[Modality.TIME_SERIES, Modality.CLINICAL_TEXT],
    )
    generator = StructuredGenerator()

    first = generator.generate("ds-one", first_req, 0)
    second = generator.generate("ds-one", second_req, 0)

    assert first.record_id == second.record_id


def test_structured_generator_omits_medications_for_unrelated_topic():
    req = GenerationRequest(topic="annual wellness")

    record = StructuredGenerator().generate("ds-one", req, 0)

    assert record.medication_history == []


def test_structured_generator_uses_topic_specific_profiles():
    generator = StructuredGenerator()

    heart_failure = generator.generate("ds-one", GenerationRequest(topic="heart failure"), 0)
    pneumonia = generator.generate("ds-one", GenerationRequest(topic="pneumonia"), 0)
    dka = generator.generate("ds-one", GenerationRequest(topic="diabetic ketoacidosis"), 0)
    stroke = generator.generate("ds-one", GenerationRequest(topic="ischemic stroke"), 0)

    assert _lab_value(heart_failure, "BNP") > 500
    assert _vital_value(heart_failure, "SpO2") < 94
    assert any(medication.name == "Furosemide" for medication in heart_failure.medication_history)
    assert pneumonia.encounters[0].diagnoses[0].display == "pneumonia"
    assert any(medication.name == "Azithromycin" for medication in pneumonia.medication_history)
    assert _lab_value(dka, "Glucose") > 250
    assert _lab_value(dka, "Bicarbonate") < 18
    assert any(medication.name == "Regular insulin" for medication in dka.medication_history)
    assert _lab_value(stroke, "Glucose") >= 70
    assert any(medication.name == "Aspirin" for medication in stroke.medication_history)
    assert stroke.encounters[0].diagnoses[0].display == "ischemic stroke"


def test_structured_generator_lists_clinical_profile_catalog():
    catalog = {profile.key: profile for profile in list_clinical_profile_catalog()}

    assert "sepsis" in catalog
    assert catalog["sepsis"].keywords == ("sepsis", "infection")
    assert "Lactate" in catalog["sepsis"].lab_names
    assert "Ceftriaxone" in catalog["sepsis"].medication_names


def test_structured_generator_uses_additional_common_clinical_profiles():
    generator = StructuredGenerator()

    pe = generator.generate("ds-one", GenerationRequest(topic="pulmonary embolism"), 0)
    acs = generator.generate("ds-one", GenerationRequest(topic="acute coronary syndrome"), 0)
    copd = generator.generate("ds-one", GenerationRequest(topic="COPD exacerbation"), 0)
    gi_bleed = generator.generate("ds-one", GenerationRequest(topic="upper GI bleed"), 0)
    aki = generator.generate("ds-one", GenerationRequest(topic="acute kidney injury"), 0)

    assert _lab_value(pe, "D-dimer") > 0.5
    assert _vital_value(pe, "SpO2") < 94
    assert any(medication.name == "Heparin" for medication in pe.medication_history)
    assert _lab_value(acs, "Troponin I") > 0.04
    assert any(medication.name == "Aspirin" for medication in acs.medication_history)
    assert _lab_value(copd, "pCO2") > 45
    assert any(medication.name == "Albuterol" for medication in copd.medication_history)
    assert _lab_value(gi_bleed, "Hemoglobin") < 12
    assert any(medication.name == "Pantoprazole" for medication in gi_bleed.medication_history)
    assert _lab_value(aki, "Creatinine") > 1.5
    assert any(medication.name == "Normal saline" for medication in aki.medication_history)


def test_structured_generator_adds_furosemide_for_heart_failure_topic_variant():
    req = GenerationRequest(topic="heart-failure")

    record = StructuredGenerator().generate("ds-one", req, 0)

    assert any(medication.name == "Furosemide" for medication in record.medication_history)


def test_structured_generator_applies_age_and_sex_cohort_constraints():
    req = GenerationRequest(
        topic="sepsis",
        cohort_constraints={
            "age_min": 70,
            "age_max": 72,
            "sexes": ["female", "male", "other"],
        },
    )
    generator = StructuredGenerator()

    records = [generator.generate("ds-one", req, index) for index in range(4)]

    assert [record.patient.age for record in records] == [70, 71, 72, 70]
    assert [record.patient.sex for record in records] == [
        "female",
        "male",
        "other",
        "female",
    ]
    assert records[0].metadata["cohort_constraints"]["age_min"] == 70


def test_structured_generator_rejects_invalid_age_constraints():
    req = GenerationRequest(
        topic="sepsis",
        cohort_constraints={"age_min": 90, "age_max": 70},
    )

    with pytest.raises(ValueError, match="age_min must be <= age_max"):
        StructuredGenerator().generate("ds-one", req, 0)


def test_structured_generator_rejects_empty_sex_constraint():
    req = GenerationRequest(
        topic="sepsis",
        cohort_constraints={"sexes": []},
    )

    with pytest.raises(ValueError, match="sexes must contain at least one value"):
        StructuredGenerator().generate("ds-one", req, 0)


def _lab_value(record, name: str):
    return next(lab.value for lab in record.labs if lab.name == name)


def _vital_value(record, name: str):
    return next(vital.value for vital in record.vitals if vital.name == name)
