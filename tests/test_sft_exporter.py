from casecrawler.export.sft_exporter import export_sft_conversation
from casecrawler.models.blueprint import CaseBlueprint, PhaseBlueprint
from casecrawler.models.case import DifficultyLevel, GeneratedCase, GroundTruth, Patient, ReviewResult
from casecrawler.models.diagnostics import VitalSigns
from casecrawler.models.phase import CasePhase, PhaseDecision, PhaseOutcome

def _make_case():
    blueprint = CaseBlueprint(diagnosis="SAH", clinical_arc="SAH via CT", phase_count=2, phases=[
        PhaseBlueprint(phase_number=1, time_offset="T+0", clinical_context="ED", available_diagnostics=[], pending_diagnostics=[], decision_type="order_workup", correct_action="CT head", key_reasoning="Most sensitive initial test for SAH"),
        PhaseBlueprint(phase_number=2, time_offset="T+45min", clinical_context="CT back", available_diagnostics=["CT"], pending_diagnostics=[], decision_type="start_treatment", correct_action="Consult neurosurgery", key_reasoning="CT confirms SAH needs surgical evaluation"),
    ], branching_points=[], expected_complications=[])
    phases = [
        CasePhase(phase_number=1, time_offset="T+0", narrative="A 42-year-old woman with thunderclap headache.", vitals=VitalSigns(hr=92, bp_systolic=168, bp_diastolic=95, rr=20, spo2=98.0, temp_c=37.4), lab_results=[], imaging_results=[], decisions=[
            PhaseDecision(action="Order CT head", is_optimal=True, quality="optimal", reasoning="Most sensitive", clinical_outcome="SAH confirmed"),
            PhaseDecision(action="Discharge", is_optimal=False, quality="catastrophic", reasoning="Misdiagnosis", clinical_outcome="Rebleed"),
        ], phase_outcome=PhaseOutcome(optimal_next_phase=2, patient_status="stable", narrative_transition="CT ordered")),
        CasePhase(phase_number=2, time_offset="T+45min", narrative="CT shows diffuse SAH.", vitals=VitalSigns(hr=88, bp_systolic=155, bp_diastolic=90, rr=18, spo2=98.0, temp_c=37.2), lab_results=[], imaging_results=[], decisions=[
            PhaseDecision(action="Consult neurosurgery", is_optimal=True, quality="optimal", reasoning="Confirmed SAH", clinical_outcome="Surgical eval"),
        ], phase_outcome=PhaseOutcome(optimal_next_phase=None, patient_status="stable", narrative_transition="Admitted")),
    ]
    return GeneratedCase(case_id="sft-test-1", topic="SAH", difficulty=DifficultyLevel.RESIDENT, specialty=["neurosurgery"], patient=Patient(age=42, sex="female", demographics="Healthy"), blueprint=blueprint, phases=phases, vignette="...", decision_prompt="What would you do next?", ground_truth=GroundTruth(diagnosis="SAH", optimal_next_step="CT head", rationale="", key_findings=[]), decision_tree=[], complications=[], review=ReviewResult(accuracy_score=0.95, pedagogy_score=0.9, bias_score=0.92, approved=True, notes=[]), sources=[], metadata={})

def test_sft_conversation_structure():
    case = _make_case()
    conv = export_sft_conversation(case)
    assert conv.case_id == "sft-test-1"
    assert "physician" in conv.system_prompt.lower() or "clinician" in conv.system_prompt.lower()
    assert len(conv.turns) == 4
    assert conv.turns[0].role == "system"
    assert conv.turns[1].role == "assistant"
    assert conv.turns[2].role == "system"
    assert conv.turns[3].role == "assistant"

def test_sft_conversation_system_turns_contain_narrative():
    case = _make_case()
    conv = export_sft_conversation(case)
    assert "thunderclap headache" in conv.turns[0].content
    assert "CT shows" in conv.turns[2].content

def test_sft_conversation_assistant_turns_contain_reasoning():
    case = _make_case()
    conv = export_sft_conversation(case)
    assert "CT head" in conv.turns[1].content or "CT" in conv.turns[1].content
    assert "neurosurgery" in conv.turns[3].content.lower()

def test_sft_wrong_path_variant():
    case = _make_case()
    conv = export_sft_conversation(case, include_wrong_path=True)
    assert conv.case_id == "sft-test-1"
    assert "Discharge" in conv.turns[1].content or "discharge" in conv.turns[1].content.lower()
