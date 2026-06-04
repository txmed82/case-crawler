import json

from casecrawler.export.blueprints import (
    export_blueprint_payload,
    export_blueprints_jsonl,
)
from casecrawler.models.blueprint import (
    BlueprintEvidence,
    ClinicalBlueprint,
    CohortArchetype,
    CohortPlan,
    GenerationRole,
)
from casecrawler.models.synthetic import Modality


def _plan() -> CohortPlan:
    return CohortPlan(
        plan_id="plan-1",
        request="Generate anticoagulation decision cases.",
        target_count=1,
        archetypes=[
            CohortArchetype(
                name="anticoagulation decision",
                organ_system="cardiovascular",
                setting="outpatient",
                target_count=1,
                required_modalities=[Modality.STRUCTURED_EHR],
            )
        ],
        created_by=GenerationRole.PLANNER,
    )


def _blueprint() -> ClinicalBlueprint:
    return ClinicalBlueprint(
        blueprint_id="bp-1",
        dataset_id="ds-1",
        cohort_plan_id="plan-1",
        archetype_name="anticoagulation decision",
        organ_system="cardiovascular",
        setting="outpatient",
        patient={"age": 72, "sex": "female"},
        chief_concern="Atrial fibrillation anticoagulation follow-up.",
        diagnoses=[
            {
                "name": "atrial fibrillation",
                "supporting_findings": ["ECG confirms AF"],
            }
        ],
        evidence=BlueprintEvidence(
            supported_claims=["AF anticoagulation requires renal-dose review."],
        ),
    )


def test_export_blueprint_payload_includes_plan_context():
    payload = export_blueprint_payload(_blueprint(), plan=_plan())

    assert payload["artifact_type"] == "casecrawler_clinical_blueprint"
    assert payload["blueprint"]["blueprint_id"] == "bp-1"
    assert payload["cohort_plan"]["plan_id"] == "plan-1"


def test_export_blueprints_jsonl_writes_one_payload_per_blueprint(tmp_path):
    output = tmp_path / "blueprints.jsonl"

    count = export_blueprints_jsonl(
        [_blueprint()],
        output,
        plan_lookup=lambda plan_id: _plan(),
    )

    lines = output.read_text().splitlines()
    assert count == 1
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["blueprint"]["dataset_id"] == "ds-1"
    assert payload["cohort_plan"]["request"] == (
        "Generate anticoagulation decision cases."
    )
