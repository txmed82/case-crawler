import pytest

from casecrawler.generation.structured_generator import (
    StructuredGenerator,
    list_clinical_profile_catalog,
)
from casecrawler.models.dataset import ExportFormat, GenerationRequest
from casecrawler.models.synthetic import ComplexityProfile, Modality


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


def test_structured_generator_can_emit_longitudinal_encounter_timeline():
    req = GenerationRequest(
        topic="heart failure",
        cohort_constraints={
            "base_time": "2026-02-03T04:05:06",
            "encounter_count": 3,
        },
    )

    record = StructuredGenerator().generate("ds-one", req, 0)

    assert [encounter.start for encounter in record.encounters] == [
        "2026-02-03T04:05:06",
        "2026-02-04T04:05:06",
        "2026-02-05T04:05:06",
    ]
    assert [encounter.end for encounter in record.encounters] == [
        "2026-02-03T10:05:06",
        "2026-02-04T10:05:06",
        "2026-02-05T10:05:06",
    ]
    assert record.encounters[0].diagnoses[0].display == "heart failure exacerbation"
    assert record.encounters[1].diagnoses[0].display == "heart failure exacerbation"
    assert record.encounters[1].reason == "heart failure follow-up 2"
    assert record.encounters[2].reason == "heart failure follow-up 3"
    assert record.metadata["cohort_constraints"]["encounter_count"] == 3


def test_structured_generator_repeats_observations_across_longitudinal_encounters():
    req = GenerationRequest(
        topic="sepsis",
        cohort_constraints={
            "base_time": "2026-02-03T04:05:06",
            "encounter_count": 2,
        },
    )

    record = StructuredGenerator().generate("ds-one", req, 0)

    lactates = [lab for lab in record.labs if lab.name == "Lactate"]
    heart_rates = [vital for vital in record.vitals if vital.name == "HR"]
    assert [lab.effective_time for lab in lactates] == [
        "2026-02-03T05:05:06",
        "2026-02-04T05:05:06",
    ]
    assert [lab.value for lab in lactates] == [3.4, 3.6]
    assert [vital.effective_time for vital in heart_rates] == [
        "2026-02-03T04:20:06",
        "2026-02-04T04:20:06",
    ]
    assert [vital.value for vital in heart_rates] == [112, 115]


def test_structured_generator_rejects_invalid_encounter_count():
    req = GenerationRequest(
        topic="sepsis",
        cohort_constraints={"encounter_count": 0},
    )

    with pytest.raises(ValueError, match="encounter_count"):
        StructuredGenerator().generate("ds-one", req, 0)


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


def test_structured_generator_emits_only_requested_observation_modalities():
    generator = StructuredGenerator()

    imaging_only = generator.generate(
        "ds-one",
        GenerationRequest(topic="sepsis", modalities=[Modality.IMAGING]),
        0,
    )
    labs_only = generator.generate(
        "ds-one",
        GenerationRequest(topic="sepsis", modalities=[Modality.LABS]),
        0,
    )
    vitals_only = generator.generate(
        "ds-one",
        GenerationRequest(topic="sepsis", modalities=[Modality.VITALS]),
        0,
    )
    structured_only = generator.generate(
        "ds-one",
        GenerationRequest(topic="sepsis", modalities=[Modality.STRUCTURED_EHR]),
        0,
    )

    assert imaging_only.labs == []
    assert imaging_only.vitals == []
    assert imaging_only.medication_history == []
    assert labs_only.labs
    assert labs_only.vitals == []
    assert vitals_only.labs == []
    assert vitals_only.vitals
    assert structured_only.medication_history
    assert structured_only.labs == []
    assert structured_only.vitals == []


def test_structured_generator_persists_requested_export_formats():
    req = GenerationRequest(
        topic="sepsis",
        export_formats=[ExportFormat.CHAT_JSONL, ExportFormat.PARQUET],
    )

    record = StructuredGenerator().generate("ds-one", req, 0)

    assert record.metadata["requested_export_formats"] == ["chat_jsonl", "parquet"]


def test_structured_generator_persists_required_human_review_flag():
    generator = StructuredGenerator()
    record = generator.generate(
        dataset_id="ds-1",
        req=GenerationRequest(topic="sepsis", require_human_review=True),
        index=0,
    )

    assert record.metadata["require_human_review"] is True
    assert record.metadata["generation_overrides"]["require_human_review"] is True


def test_structured_generator_persists_generation_overrides():
    req = GenerationRequest(
        topic="pneumonia",
        modalities=[Modality.CLINICAL_TEXT, Modality.IMAGING, Modality.TIME_SERIES],
        clinical_text_backend="llm",
        llm_provider="ollama",
        llm_model="medgemma-local",
        imaging_backend="diffusers",
        imaging_model_profile="cxr_pneumonia_dreambooth",
        time_series_backend="external",
        time_series_model_profile="timediff",
        time_series_command=["timediff-sample"],
        validation_threshold=0.9,
    )

    record = StructuredGenerator().generate("ds-one", req, 0)

    assert record.metadata["generation_overrides"] == {
        "clinical_text_backend": "llm",
        "llm_provider": "ollama",
        "llm_model": "medgemma-local",
        "imaging_backend": "diffusers",
        "imaging_model_profile": "cxr_pneumonia_dreambooth",
        "time_series_backend": "external",
        "time_series_model_profile": "timediff",
        "time_series_command": ["timediff-sample"],
        "validation_threshold": 0.9,
    }


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
    assert "bacterial_meningitis" in catalog
    assert "generalized_seizure" in catalog
    assert catalog["sepsis"].keywords == ("sepsis", "infection")
    assert "Lactate" in catalog["sepsis"].lab_names
    assert "Ceftriaxone" in catalog["sepsis"].medication_names
    assert "CSF WBC" in catalog["bacterial_meningitis"].lab_names


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


def test_structured_generator_uses_expanded_common_clinical_profiles():
    generator = StructuredGenerator()

    asthma = generator.generate("ds-one", GenerationRequest(topic="status asthmaticus"), 0)
    pancreatitis = generator.generate("ds-one", GenerationRequest(topic="acute pancreatitis"), 0)
    appendicitis = generator.generate("ds-one", GenerationRequest(topic="appendicitis"), 0)
    pyelo = generator.generate("ds-one", GenerationRequest(topic="pyelonephritis"), 0)
    meningitis = generator.generate("ds-one", GenerationRequest(topic="bacterial meningitis"), 0)
    seizure = generator.generate("ds-one", GenerationRequest(topic="status epilepticus"), 0)

    assert _vital_value(asthma, "Respiratory rate") >= 30
    assert any(medication.name == "Magnesium sulfate" for medication in asthma.medication_history)
    assert _lab_value(pancreatitis, "Lipase") > 500
    assert any(medication.name == "Lactated Ringer's" for medication in pancreatitis.medication_history)
    assert _lab_value(appendicitis, "CRP") > 10
    assert any(medication.name == "Metronidazole" for medication in appendicitis.medication_history)
    assert _lab_value(pyelo, "Urine WBC") > 5
    assert any(medication.name == "Ceftriaxone" for medication in pyelo.medication_history)
    assert _lab_value(meningitis, "CSF WBC") > 100
    assert any(medication.name == "Vancomycin" for medication in meningitis.medication_history)
    assert _lab_value(seizure, "Lactate") > 2
    assert any(medication.name == "Lorazepam" for medication in seizure.medication_history)


def test_structured_generator_adds_furosemide_for_heart_failure_topic_variant():
    req = GenerationRequest(topic="heart-failure")

    record = StructuredGenerator().generate("ds-one", req, 0)

    assert any(medication.name == "Furosemide" for medication in record.medication_history)


def test_structured_generator_complexity_changes_artifact_density():
    generator = StructuredGenerator()

    moderate = generator.generate(
        "ds-one",
        GenerationRequest(topic="sepsis", complexity=ComplexityProfile.MODERATE),
        0,
    )
    complex_record = generator.generate(
        "ds-one",
        GenerationRequest(topic="sepsis", complexity=ComplexityProfile.COMPLEX),
        0,
    )
    rare = generator.generate(
        "ds-one",
        GenerationRequest(topic="sepsis", complexity=ComplexityProfile.RARE),
        0,
    )

    assert len(complex_record.encounters[0].diagnoses) > len(moderate.encounters[0].diagnoses)
    assert len(complex_record.labs) > len(moderate.labs)
    assert len(complex_record.medication_history) > len(moderate.medication_history)
    assert any(
        diagnosis.display == "distributive shock with disseminated intravascular coagulation"
        for diagnosis in rare.encounters[0].diagnoses
    )
    assert any(lab.name == "Fibrinogen" for lab in rare.labs)
    assert any(medication.name == "Norepinephrine" for medication in rare.medication_history)


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


def test_structured_generator_populates_demographics_and_social_history_context():
    req = GenerationRequest(
        topic="sepsis",
        cohort_constraints={
            "age_min": 40,
            "age_max": 40,
            "sexes": ["female"],
            "races": ["synthetic_race_a", "synthetic_race_b"],
            "ethnicities": ["synthetic_ethnicity"],
            "insurance": ["synthetic_plan"],
            "smoking_statuses": ["never", "former"],
            "alcohol_use": ["none"],
            "housing": ["stable"],
        },
    )
    generator = StructuredGenerator()

    first = generator.generate("ds-one", req, 0)
    second = generator.generate("ds-one", req, 1)

    assert first.patient.demographics == {
        "age_group": "adult",
        "sex_at_generation": "female",
        "race": "synthetic_race_a",
        "ethnicity": "synthetic_ethnicity",
        "insurance": "synthetic_plan",
    }
    assert first.patient.social_history == {
        "smoking_status": "never",
        "alcohol_use": "none",
        "housing": "stable",
    }
    assert first.metadata["cohort_constraints"]["races"] == [
        "synthetic_race_a",
        "synthetic_race_b",
    ]
    assert first.metadata["cohort_constraints"]["ethnicities"] == [
        "synthetic_ethnicity"
    ]
    assert first.metadata["cohort_constraints"]["insurance"] == ["synthetic_plan"]
    assert first.metadata["cohort_constraints"]["smoking_statuses"] == [
        "never",
        "former",
    ]
    assert first.metadata["cohort_constraints"]["alcohol_use"] == ["none"]
    assert first.metadata["cohort_constraints"]["housing"] == ["stable"]
    assert second.patient.demographics["race"] == "synthetic_race_b"
    assert second.patient.social_history["smoking_status"] == "former"


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


def test_structured_generator_rejects_empty_demographic_constraint():
    req = GenerationRequest(topic="sepsis", cohort_constraints={"races": []})

    with pytest.raises(ValueError, match="races must contain at least one value"):
        StructuredGenerator().generate("ds-one", req, 0)


def _lab_value(record, name: str):
    return next(lab.value for lab in record.labs if lab.name == name)


def _vital_value(record, name: str):
    return next(vital.value for vital in record.vitals if vital.name == name)
