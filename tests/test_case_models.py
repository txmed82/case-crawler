from casecrawler.models.blueprint import BranchPoint, CaseBlueprint, PhaseBlueprint
from casecrawler.models.case import (
    Complication,
    DecisionChoice,
    DifficultyLevel,
    GeneratedCase,
    GroundTruth,
    Patient,
    ReviewResult,
)
from casecrawler.models.diagnostics import LabResult, LabValue, VitalSigns
from casecrawler.models.phase import CasePhase, PhaseDecision, PhaseOutcome


def test_difficulty_levels():
    assert DifficultyLevel.MEDICAL_STUDENT == "medical_student"
    assert DifficultyLevel.RESIDENT == "resident"
    assert DifficultyLevel.ATTENDING == "attending"


def test_patient_creation():
    p = Patient(age=42, sex="female", demographics="No significant PMH")
    assert p.age == 42
    assert p.sex == "female"


def test_ground_truth():
    gt = GroundTruth(
        diagnosis="aneurysmal subarachnoid hemorrhage",
        optimal_next_step="Non-contrast CT head",
        rationale="Most sensitive initial test for SAH",
        key_findings=["thunderclap headache", "neck stiffness"],
    )
    assert len(gt.key_findings) == 2


def test_decision_choice_correct():
    dc = DecisionChoice(
        choice="Order non-contrast CT head",
        is_correct=True,
        error_type=None,
        reasoning="Most sensitive for acute SAH",
        outcome="CT shows hyperdense material in basal cisterns",
        consequence=None,
        next_decision="Consult neurosurgery",
    )
    assert dc.is_correct is True
    assert dc.error_type is None


def test_decision_choice_wrong():
    dc = DecisionChoice(
        choice="Discharge with migraine diagnosis",
        is_correct=False,
        error_type="catastrophic",
        reasoning="Misdiagnosis",
        outcome="Patient rebleeds at home",
        consequence="Mortality",
        next_decision=None,
    )
    assert dc.error_type == "catastrophic"


def test_complication():
    c = Complication(
        trigger="delayed_diagnosis",
        detail="6 hour delay",
        event="Rebleeding",
        outcome="50% mortality",
    )
    assert c.trigger == "delayed_diagnosis"


def test_review_result_approved():
    r = ReviewResult(
        accuracy_score=0.95,
        pedagogy_score=0.88,
        bias_score=0.92,
        approved=True,
        notes=[],
    )
    assert r.approved is True


def test_review_result_rejected():
    r = ReviewResult(
        accuracy_score=0.5,
        pedagogy_score=0.88,
        bias_score=0.92,
        approved=False,
        notes=["Diagnosis is incorrect for the presented findings"],
    )
    assert r.approved is False
    assert len(r.notes) == 1


def test_generated_case_full():
    case = GeneratedCase(
        case_id="test-123",
        topic="subarachnoid hemorrhage",
        difficulty=DifficultyLevel.RESIDENT,
        specialty=["neurosurgery", "emergency_medicine"],
        patient=Patient(age=42, sex="female", demographics="Healthy"),
        vignette="A 42-year-old woman presents...",
        decision_prompt="What would you do next?",
        ground_truth=GroundTruth(
            diagnosis="SAH",
            optimal_next_step="CT head",
            rationale="Most sensitive",
            key_findings=["thunderclap headache"],
        ),
        decision_tree=[
            DecisionChoice(
                choice="CT head",
                is_correct=True,
                error_type=None,
                reasoning="Correct",
                outcome="SAH confirmed",
                consequence=None,
                next_decision=None,
            ),
        ],
        complications=[
            Complication(
                trigger="delayed_diagnosis",
                detail="6h delay",
                event="Rebleed",
                outcome="Death",
            ),
        ],
        review=ReviewResult(
            accuracy_score=0.95,
            pedagogy_score=0.9,
            bias_score=0.95,
            approved=True,
            notes=[],
        ),
        sources=[{"type": "pubmed", "reference": "PMID:123", "chunk_ids": ["abc"]}],
        metadata={"generated_at": "2026-04-08", "model": "claude-sonnet-4-6", "retry_count": 0},
    )
    assert case.case_id == "test-123"
    assert case.difficulty == DifficultyLevel.RESIDENT
    assert len(case.decision_tree) == 1
    assert case.review.approved is True


def test_generated_case_legacy_compat():
    case = GeneratedCase(case_id="legacy-1", topic="SAH", difficulty=DifficultyLevel.RESIDENT, specialty=["neurosurgery"], patient=Patient(age=42, sex="female", demographics="Healthy"), vignette="A 42-year-old woman presents...", decision_prompt="What would you do next?", ground_truth=GroundTruth(diagnosis="SAH", optimal_next_step="CT head", rationale="Most sensitive", key_findings=["thunderclap headache"]), decision_tree=[DecisionChoice(choice="CT head", is_correct=True, error_type=None, reasoning="Correct", outcome="Confirmed", consequence=None, next_decision=None)], complications=[], sources=[], metadata={})
    assert case.blueprint is None
    assert case.phases == []
    assert case.is_multi_step() is False


def test_generated_case_multi_step():
    blueprint = CaseBlueprint(diagnosis="SAH", clinical_arc="SAH via CT, surgical management", phase_count=2, phases=[PhaseBlueprint(phase_number=1, time_offset="T+0", clinical_context="ED presentation", available_diagnostics=[], pending_diagnostics=[], decision_type="order_workup", correct_action="Order CT head", key_reasoning="Most sensitive"), PhaseBlueprint(phase_number=2, time_offset="T+45min", clinical_context="CT results available", available_diagnostics=["CT head"], pending_diagnostics=[], decision_type="start_treatment", correct_action="Consult neurosurgery", key_reasoning="CT confirms SAH")], branching_points=[], expected_complications=["vasospasm"])
    phases = [
        CasePhase(phase_number=1, time_offset="T+0", narrative="A 42-year-old woman presents with thunderclap headache.", vitals=VitalSigns(hr=92, bp_systolic=168, bp_diastolic=95, rr=20, spo2=98.0, temp_c=37.4), lab_results=[], imaging_results=[], decisions=[PhaseDecision(action="Order CT head", is_optimal=True, quality="optimal", reasoning="Most sensitive", clinical_outcome="SAH confirmed")], phase_outcome=PhaseOutcome(optimal_next_phase=2, patient_status="stable", narrative_transition="CT ordered.")),
        CasePhase(phase_number=2, time_offset="T+45min", narrative="CT results show hyperdense material in basal cisterns.", vitals=VitalSigns(hr=88, bp_systolic=155, bp_diastolic=90, rr=18, spo2=98.0, temp_c=37.2), lab_results=[], imaging_results=[], decisions=[PhaseDecision(action="Consult neurosurgery", is_optimal=True, quality="optimal", reasoning="CT confirms SAH", clinical_outcome="Neurosurgery evaluates")], phase_outcome=PhaseOutcome(optimal_next_phase=None, patient_status="stable", narrative_transition="Admitted to ICU.")),
    ]
    case = GeneratedCase(case_id="multi-1", topic="SAH", difficulty=DifficultyLevel.RESIDENT, specialty=["neurosurgery"], patient=Patient(age=42, sex="female", demographics="Healthy"), vignette="A 42-year-old woman presents with thunderclap headache.", decision_prompt="What would you do next?", ground_truth=GroundTruth(diagnosis="SAH", optimal_next_step="CT head", rationale="Most sensitive", key_findings=["thunderclap headache"]), decision_tree=[], complications=[], blueprint=blueprint, phases=phases, sources=[], metadata={})
    assert case.is_multi_step() is True
    assert case.blueprint is not None
    assert len(case.phases) == 2
