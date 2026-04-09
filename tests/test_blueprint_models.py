from casecrawler.models.blueprint import BranchPoint, CaseBlueprint, PhaseBlueprint

def test_phase_blueprint():
    pb = PhaseBlueprint(phase_number=1, time_offset="T+0", clinical_context="Patient presents to ED with thunderclap headache", available_diagnostics=[], pending_diagnostics=[], decision_type="order_workup", correct_action="Order non-contrast CT head", key_reasoning="CT is 95% sensitive for SAH within 6 hours")
    assert pb.phase_number == 1
    assert pb.decision_type == "order_workup"

def test_branch_point():
    bp = BranchPoint(phase_number=2, branch_type="redirect", trigger_action_quality="suboptimal", description="Ordering MRI instead of CT delays diagnosis by ~2 hours")
    assert bp.branch_type == "redirect"

def test_case_blueprint_full():
    phases = [
        PhaseBlueprint(phase_number=1, time_offset="T+0", clinical_context="ED presentation", available_diagnostics=[], pending_diagnostics=[], decision_type="order_workup", correct_action="Order CT head", key_reasoning="Most sensitive initial test"),
        PhaseBlueprint(phase_number=2, time_offset="T+45min", clinical_context="CT results available", available_diagnostics=["CT head"], pending_diagnostics=["CBC", "BMP"], decision_type="interpret_results", correct_action="Consult neurosurgery", key_reasoning="CT confirms SAH, needs surgical evaluation"),
        PhaseBlueprint(phase_number=3, time_offset="T+2h", clinical_context="Neurosurgery consulted, labs back", available_diagnostics=["CT head", "CBC", "BMP"], pending_diagnostics=[], decision_type="start_treatment", correct_action="Nimodipine and ICU admission", key_reasoning="Prevent vasospasm, close monitoring"),
    ]
    bp = CaseBlueprint(diagnosis="Aneurysmal subarachnoid hemorrhage", clinical_arc="SAH presenting as thunderclap headache, diagnosed via CT, surgical management", phase_count=3, phases=phases, branching_points=[BranchPoint(phase_number=1, branch_type="terminal", trigger_action_quality="catastrophic", description="Discharge leads to rebleed and death")], expected_complications=["vasospasm", "rebleeding", "hydrocephalus"])
    assert bp.phase_count == 3
    assert len(bp.phases) == 3
    assert len(bp.branching_points) == 1
    assert len(bp.expected_complications) == 3
