from casecrawler.models.blueprint import (
    BlueprintEvidence,
    ClinicalBlueprint,
    JudgeReport,
    GenerationRole,
    ReleaseReadinessTier,
)


def _blueprint(**overrides) -> ClinicalBlueprint:
    payload = {
        "blueprint_id": "bp-1",
        "dataset_id": "ds-1",
        "cohort_plan_id": "plan-1",
        "archetype_name": "anticoagulation decision",
        "organ_system": "cardiovascular",
        "setting": "outpatient",
        "patient": {"age": 72, "sex": "female"},
        "chief_concern": "Atrial fibrillation anticoagulation follow-up.",
        "diagnoses": [
            {
                "name": "atrial fibrillation",
                "supporting_findings": ["ECG confirms AF"],
            }
        ],
        "clinical_reasoning_targets": ["Review renal dosing and bleeding risk."],
        "safety_constraints": ["Review bleeding risk before anticoagulation."],
        "evidence": BlueprintEvidence(
            supported_claims=["AF anticoagulation requires renal-dose review."],
            citations=[{"source": "dailymed", "claim": "renal-dose review"}],
        ),
    }
    payload.update(overrides)
    return ClinicalBlueprint(**payload)


def test_blueprint_validator_marks_grounded_clinically_plausible_blueprint():
    from casecrawler.validation.blueprints import BlueprintValidator

    report = BlueprintValidator(require_grounding=True).validate(_blueprint())

    assert report.blueprint_id == "bp-1"
    assert report.schema_valid is True
    assert report.clinically_plausible is True
    assert report.grounded is True
    assert report.judge_validated is False
    assert report.tier == ReleaseReadinessTier.CLINICALLY_PLAUSIBLE
    assert report.issues == []


def test_blueprint_validator_blocks_unsupported_or_ungrounded_claims():
    from casecrawler.validation.blueprints import BlueprintValidator

    report = BlueprintValidator(require_grounding=True).validate(
        _blueprint(
            evidence=BlueprintEvidence(
                supported_claims=[],
                inferred_claims=["May need GI workup."],
                unsupported_claims=["Start high-dose anticoagulation immediately."],
                citations=[],
            )
        )
    )

    assert report.clinically_plausible is False
    assert report.grounded is False
    assert report.tier == ReleaseReadinessTier.SCHEMA_VALID
    issue_fields = {issue["field"] for issue in report.issues}
    assert "evidence.unsupported_claims" in issue_fields
    assert "evidence.citations" in issue_fields


def test_blueprint_validator_incorporates_passing_judge_report():
    from casecrawler.validation.blueprints import BlueprintValidator

    judge = JudgeReport(
        report_id="judge-1",
        dataset_id="ds-1",
        artifact_id="bp-1",
        role=GenerationRole.JUDGE,
        score=0.91,
        passed=True,
        rubric="blueprint_plausibility",
        findings=[{"criterion": "diagnostic_support", "passed": True}],
    )

    report = BlueprintValidator(require_grounding=True).validate(
        _blueprint(),
        judge_reports=[judge],
    )

    assert report.judge_validated is True
    assert report.tier == ReleaseReadinessTier.RESEARCH_RELEASE_READY
    assert report.research_release_ready is True
    assert report.judge_reports == [judge]
