from casecrawler.generation.modality_plan import ModalityPlanner
from casecrawler.models.dataset import GenerationRequest
from casecrawler.models.synthetic import Modality


def test_modality_planner_builds_requested_modalities():
    req = GenerationRequest(
        topic="heart failure",
        modalities=[Modality.CLINICAL_TEXT, Modality.TIME_SERIES, Modality.IMAGING],
        cohort_constraints={"age_min": 60, "age_max": 90},
    )

    plan = ModalityPlanner().build(req)

    assert plan.topic == "heart failure"
    assert plan.cohort_size == 1
    assert plan.modalities == req.modalities
    assert "heart_rate" in plan.time_series_channels
    assert "ecg_lead_ii" in plan.time_series_channels
    assert "pleth" in plan.time_series_channels
    assert plan.imaging_views == ["portable_chest_xray"]


def test_modality_planner_uses_count_for_cohort_size():
    plan = ModalityPlanner().build(GenerationRequest(topic="sepsis", count=3))

    assert plan.cohort_size == 3
