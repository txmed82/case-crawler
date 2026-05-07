import pytest

from casecrawler.generation.structured_generator import StructuredGenerator
from casecrawler.models.dataset import GenerationRequest
from casecrawler.models.synthetic import Modality


def test_structured_generator_ids_do_not_depend_on_dataset_id():
    req = GenerationRequest(topic="sepsis", cohort_constraints={"base_time": "2026-02-03T04:05:06"})
    generator = StructuredGenerator()

    first = generator.generate("ds-one", req, 0)
    second = generator.generate("ds-two", req, 0)

    assert first.record_id == second.record_id
    assert first.patient.patient_id == second.patient.patient_id
    assert first.encounters[0].encounter_id == second.encounters[0].encounter_id
    assert first.dataset_id != second.dataset_id


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


def test_structured_generator_adds_furosemide_for_heart_failure_topic_variant():
    req = GenerationRequest(topic="heart-failure")

    record = StructuredGenerator().generate("ds-one", req, 0)

    assert any(medication.name == "Furosemide" for medication in record.medication_history)
