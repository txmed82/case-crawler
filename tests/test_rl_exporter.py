from casecrawler.export.rl_exporter import DEFAULT_REWARD_MAP, export_rl_episode
from casecrawler.models.blueprint import CaseBlueprint, PhaseBlueprint
from casecrawler.models.case import DifficultyLevel, GeneratedCase, GroundTruth, Patient, ReviewResult
from casecrawler.models.diagnostics import LabResult, LabValue, VitalSigns
from casecrawler.models.phase import CasePhase, PhaseDecision, PhaseOutcome

def _make_multi_step_case():
    blueprint = CaseBlueprint(diagnosis="SAH", clinical_arc="SAH via CT", phase_count=2, phases=[
        PhaseBlueprint(phase_number=1, time_offset="T+0", clinical_context="ED", available_diagnostics=[], pending_diagnostics=[], decision_type="order_workup", correct_action="CT head", key_reasoning="Most sensitive"),
        PhaseBlueprint(phase_number=2, time_offset="T+45min", clinical_context="CT back", available_diagnostics=["CT"], pending_diagnostics=[], decision_type="start_treatment", correct_action="Consult neuro", key_reasoning="Confirmed SAH"),
    ], branching_points=[], expected_complications=[])
    phases = [
        CasePhase(phase_number=1, time_offset="T+0", narrative="Woman with thunderclap headache.", vitals=VitalSigns(hr=92, bp_systolic=168, bp_diastolic=95, rr=20, spo2=98.0, temp_c=37.4), lab_results=[], imaging_results=[], decisions=[
            PhaseDecision(action="Order CT head", is_optimal=True, quality="optimal", reasoning="Sensitive", clinical_outcome="SAH confirmed"),
            PhaseDecision(action="Order MRI", is_optimal=False, quality="suboptimal", reasoning="Slower", clinical_outcome="Delayed diagnosis", time_cost="2h delay"),
            PhaseDecision(action="Discharge", is_optimal=False, quality="catastrophic", reasoning="Misdiagnosis", clinical_outcome="Rebleed"),
        ], phase_outcome=PhaseOutcome(optimal_next_phase=2, patient_status="stable", narrative_transition="CT ordered")),
        CasePhase(phase_number=2, time_offset="T+45min", narrative="CT shows SAH.", vitals=VitalSigns(hr=88, bp_systolic=155, bp_diastolic=90, rr=18, spo2=98.0, temp_c=37.2), lab_results=[LabResult(panel="CBC", values=[LabValue(name="WBC", value=9.8, unit="K/uL", reference_low=4.5, reference_high=11.0, flag=None)], timestamp="T+40min")], imaging_results=[], decisions=[
            PhaseDecision(action="Consult neurosurgery", is_optimal=True, quality="optimal", reasoning="Confirmed SAH", clinical_outcome="Surgical eval"),
        ], phase_outcome=PhaseOutcome(optimal_next_phase=None, patient_status="stable", narrative_transition="Admitted")),
    ]
    return GeneratedCase(case_id="rl-test-1", topic="SAH", difficulty=DifficultyLevel.RESIDENT, specialty=["neurosurgery"], patient=Patient(age=42, sex="female", demographics="Healthy"), blueprint=blueprint, phases=phases, vignette="Woman with thunderclap headache.\n\nCT shows SAH.", decision_prompt="What would you do next?", ground_truth=GroundTruth(diagnosis="SAH", optimal_next_step="CT head", rationale="Most sensitive", key_findings=[]), decision_tree=[], complications=[], review=ReviewResult(accuracy_score=0.95, pedagogy_score=0.9, bias_score=0.92, approved=True, notes=[]), sources=[], metadata={"multi_step": True})

def test_export_rl_episode_basic():
    case = _make_multi_step_case()
    episode = export_rl_episode(case)
    assert episode.case_id == "rl-test-1"
    assert episode.difficulty == "resident"
    assert len(episode.steps) == 2

def test_export_rl_episode_rewards():
    case = _make_multi_step_case()
    episode = export_rl_episode(case)
    step1 = episode.steps[0]
    assert len(step1.action_space) == 3
    optimal = next(a for a in step1.action_space if a.quality == "optimal")
    assert step1.reward_table[optimal.id] == DEFAULT_REWARD_MAP["optimal"]
    catastrophic = next(a for a in step1.action_space if a.quality == "catastrophic")
    assert step1.reward_table[catastrophic.id] == DEFAULT_REWARD_MAP["catastrophic"]

def test_export_rl_episode_observations_include_diagnostics():
    case = _make_multi_step_case()
    episode = export_rl_episode(case)
    step2 = episode.steps[1]
    assert step2.observation.vitals is not None
    assert len(step2.observation.new_lab_results) == 1

def test_export_rl_episode_custom_rewards():
    case = _make_multi_step_case()
    custom = {"optimal": 10.0, "acceptable": 5.0, "suboptimal": 0.0, "harmful": -5.0, "catastrophic": -10.0}
    episode = export_rl_episode(case, reward_map=custom)
    step1 = episode.steps[0]
    optimal = next(a for a in step1.action_space if a.quality == "optimal")
    assert step1.reward_table[optimal.id] == 10.0
