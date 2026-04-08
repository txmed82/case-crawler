from casecrawler.models.case import (
    Complication,
    DecisionChoice,
    DifficultyLevel,
    GeneratedCase,
    GroundTruth,
    Patient,
    ReviewResult,
)


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
