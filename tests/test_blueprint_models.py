import pytest
from pydantic import ValidationError

from casecrawler.models.blueprint import (
    BlueprintEvidence,
    BlueprintGenerationRequest,
    BlueprintValidationReport,
    ClinicalBlueprint,
    CohortArchetype,
    CohortPlan,
    GenerationAttempt,
    GenerationAttemptStatus,
    GenerationRole,
    GenerationRolePolicy,
    JudgeReport,
    ReleaseReadinessTier,
)
from casecrawler.models.synthetic import Modality


def _archetype(
    name: str,
    count: int,
    *,
    organ_system: str = "cardiovascular",
    setting: str = "outpatient",
) -> CohortArchetype:
    return CohortArchetype(
        name=name,
        organ_system=organ_system,
        setting=setting,
        target_count=count,
        acuity_mix={"routine": 0.7, "urgent": 0.3},
        difficulty_mix={"moderate": 1.0},
        required_modalities=[Modality.STRUCTURED_EHR, Modality.CLINICAL_TEXT],
        task_targets=["note_fact_extraction"],
        safety_constraints=["avoid contraindicated anticoagulation"],
    )


def _blueprint() -> ClinicalBlueprint:
    return ClinicalBlueprint(
        blueprint_id="bp-1",
        dataset_id="ds-1",
        cohort_plan_id="plan-1",
        archetype_name="anticoagulation decision",
        organ_system="cardiovascular",
        setting="outpatient",
        patient={
            "age": 72,
            "sex": "female",
            "demographics": {"race": "synthetic_white"},
        },
        chief_concern="Follow-up for atrial fibrillation anticoagulation decision.",
        diagnoses=[
            {
                "name": "atrial fibrillation",
                "supporting_findings": ["irregularly irregular rhythm", "ECG confirms AF"],
            }
        ],
        differential=[
            {
                "name": "sinus rhythm with frequent PACs",
                "rationale": "palpitations can mimic AF before ECG confirmation",
            }
        ],
        comorbidities=["hypertension", "chronic kidney disease"],
        timeline=[
            {
                "day": 0,
                "event": "Clinic visit with ECG review and shared decision making.",
            }
        ],
        expected_labs=[
            {
                "name": "Creatinine",
                "value": 1.6,
                "unit": "mg/dL",
                "clinical_reason": "renal dosing safety",
            }
        ],
        expected_vitals=[
            {
                "name": "HR",
                "value": 96,
                "unit": "/min",
                "clinical_reason": "rate control assessment",
            }
        ],
        medications=[
            {
                "name": "Apixaban",
                "indication": "stroke prevention in atrial fibrillation",
                "safety_considerations": ["renal dose review"],
            }
        ],
        orders=[
            {
                "order_type": "laboratory",
                "display": "Basic metabolic panel",
                "clinical_reason": "renal function monitoring",
            }
        ],
        clinical_reasoning_targets=[
            "Assess stroke risk, bleeding risk, and renal-dose appropriateness.",
        ],
        safety_constraints=["Do not recommend anticoagulation without bleeding-risk review."],
        uncertainty_points=["Patient reports intermittent dark stools last month."],
        intended_tasks=["medication_reconciliation", "note_fact_extraction"],
        evidence=BlueprintEvidence(
            supported_claims=["AF anticoagulation requires renal-dose review."],
            inferred_claims=["Patient may need GI evaluation before dose escalation."],
            unsupported_claims=[],
            citations=[
                {
                    "source": "dailymed",
                    "url": "https://example.test/apixaban",
                    "claim": "renal-dose review",
                }
            ],
        ),
    )


def test_cohort_plan_requires_archetype_counts_to_match_target_count():
    with pytest.raises(ValidationError, match="target_count"):
        CohortPlan(
            plan_id="plan-1",
            request="Generate cardiology cases.",
            target_count=10,
            domains=["cardiology"],
            settings=["outpatient"],
            archetypes=[
                _archetype("heart failure medication optimization", 4),
                _archetype("anticoagulation decision", 3),
            ],
            created_by=GenerationRole.PLANNER,
        )


def test_cohort_plan_preserves_broad_generation_constraints():
    plan = CohortPlan(
        plan_id="plan-1",
        request="Generate cardiology cases.",
        target_count=7,
        domains=["cardiology"],
        settings=["outpatient"],
        archetypes=[
            _archetype("heart failure medication optimization", 4),
            _archetype("anticoagulation decision", 3),
        ],
        required_grounding=True,
        diversity_targets={"min_archetype_count": 2},
        created_by=GenerationRole.PLANNER,
    )

    assert plan.target_count == 7
    assert plan.required_grounding is True
    assert plan.archetype_counts == {
        "heart failure medication optimization": 4,
        "anticoagulation decision": 3,
    }


def test_clinical_blueprint_carries_model_planned_case_source_of_truth():
    blueprint = _blueprint()

    assert blueprint.primary_diagnosis == "atrial fibrillation"
    assert blueprint.has_unsupported_claims is False
    assert blueprint.required_modalities == [
        Modality.STRUCTURED_EHR,
        Modality.CLINICAL_TEXT,
    ]
    assert blueprint.evidence.citations[0]["source"] == "dailymed"


def test_clinical_blueprint_rejects_diagnoses_without_supporting_findings():
    payload = _blueprint().model_dump()
    payload["diagnoses"] = [{"name": "atrial fibrillation", "supporting_findings": []}]

    with pytest.raises(ValidationError, match="supporting_findings"):
        ClinicalBlueprint(**payload)


def test_generation_role_policy_separates_byok_model_roles():
    policy = GenerationRolePolicy(
        role=GenerationRole.JUDGE,
        provider="openai",
        model="gpt-4.1-mini",
        temperature=0.0,
        metadata={"purpose": "independent clinical validation"},
    )

    assert policy.role == GenerationRole.JUDGE
    assert policy.provider == "openai"
    assert policy.model == "gpt-4.1-mini"


def test_blueprint_generation_request_is_model_driven_not_recipe_driven():
    req = BlueprintGenerationRequest(
        request="Generate outpatient anticoagulation decision cases with uncertainty.",
        target_count=25,
        domains=["cardiology"],
        settings=["outpatient"],
        role_policies=[
            GenerationRolePolicy(
                role=GenerationRole.PLANNER,
                provider="openrouter",
                model="anthropic/claude-sonnet-4-6",
                temperature=0.2,
            ),
            GenerationRolePolicy(
                role=GenerationRole.JUDGE,
                provider="openai",
                model="gpt-4.1-mini",
                temperature=0.0,
            ),
        ],
        required_grounding=True,
        diversity_targets={"min_archetype_count": 5},
        max_repair_rounds=2,
    )

    assert req.target_count == 25
    assert req.policy_for(GenerationRole.PLANNER).model == (
        "anthropic/claude-sonnet-4-6"
    )
    assert req.policy_for(GenerationRole.BLUEPRINT_GENERATOR) is None
    assert req.required_grounding is True
    assert req.diversity_targets["min_archetype_count"] == 5


def test_blueprint_generation_request_rejects_topic_pack_recipe_fields():
    with pytest.raises(ValidationError, match="Extra inputs"):
        BlueprintGenerationRequest(
            request="Generate kidney transplant medication cases.",
            target_count=5,
            topic="renal",
            recipe="transplant_pack",
        )


def test_generation_attempt_tracks_failed_and_repaired_model_outputs():
    attempt = GenerationAttempt(
        attempt_id="attempt-1",
        dataset_id="ds-1",
        role=GenerationRole.BLUEPRINT_GENERATOR,
        status=GenerationAttemptStatus.REPAIR_REQUESTED,
        provider="openrouter",
        model="anthropic/claude-sonnet-4-6",
        prompt_hash="abc123",
        input_tokens=1200,
        output_tokens=900,
        errors=["diagnosis lacked supporting findings"],
        artifact_id="bp-1",
        metadata={"repair_round": 1},
    )

    assert attempt.status == GenerationAttemptStatus.REPAIR_REQUESTED
    assert attempt.errors == ["diagnosis lacked supporting findings"]
    assert attempt.total_tokens == 2100


def test_judge_report_and_blueprint_validation_report_support_release_tiers():
    judge = JudgeReport(
        report_id="judge-1",
        dataset_id="ds-1",
        artifact_id="bp-1",
        role=GenerationRole.JUDGE,
        score=0.91,
        passed=True,
        rubric="blueprint_plausibility",
        findings=[
            {
                "criterion": "diagnostic_support",
                "passed": True,
                "rationale": "diagnosis has ECG and exam support",
            }
        ],
    )
    validation = BlueprintValidationReport(
        blueprint_id="bp-1",
        tier=ReleaseReadinessTier.CLINICALLY_PLAUSIBLE,
        schema_valid=True,
        clinically_plausible=True,
        grounded=False,
        judge_validated=False,
        issues=[],
        judge_reports=[judge],
    )

    assert validation.tier == ReleaseReadinessTier.CLINICALLY_PLAUSIBLE
    assert validation.research_release_ready is False
    assert validation.judge_reports[0].score == 0.91
