import pytest

from casecrawler.generation.structured_generator import StructuredGenerator
from casecrawler.models.dataset import GenerationRequest


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
