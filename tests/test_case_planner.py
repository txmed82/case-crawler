from unittest.mock import AsyncMock
import pytest
from casecrawler.generation.case_planner import CasePlannerAgent, CasePlannerOutput
from casecrawler.llm.base import StructuredGenerationResult
from casecrawler.models.blueprint import BranchPoint, CaseBlueprint, PhaseBlueprint
from casecrawler.models.case import Patient

def _mock_planner_result():
    return StructuredGenerationResult(
        data=CasePlannerOutput(
            patient=Patient(age=42, sex="female", demographics="No significant PMH"),
            specialty=["neurosurgery", "emergency_medicine"],
            blueprint=CaseBlueprint(diagnosis="SAH", clinical_arc="SAH via CT, surgical management", phase_count=3, phases=[
                PhaseBlueprint(phase_number=1, time_offset="T+0", clinical_context="ED presentation", available_diagnostics=[], pending_diagnostics=[], decision_type="order_workup", correct_action="Order CT head", key_reasoning="Most sensitive"),
                PhaseBlueprint(phase_number=2, time_offset="T+45min", clinical_context="CT results", available_diagnostics=["CT head"], pending_diagnostics=[], decision_type="interpret_results", correct_action="Consult neurosurgery", key_reasoning="CT confirms SAH"),
                PhaseBlueprint(phase_number=3, time_offset="T+2h", clinical_context="Neurosurgery consulted", available_diagnostics=["CT head", "CBC"], pending_diagnostics=[], decision_type="start_treatment", correct_action="Nimodipine and ICU admission", key_reasoning="Prevent vasospasm"),
            ], branching_points=[BranchPoint(phase_number=1, branch_type="terminal", trigger_action_quality="catastrophic", description="Discharge leads to rebleed")], expected_complications=["vasospasm"]),
        ),
        input_tokens=500, output_tokens=400, model="test-model",
    )

@pytest.mark.asyncio
async def test_case_planner_generates_blueprint():
    provider = AsyncMock()
    provider.generate_structured.return_value = _mock_planner_result()
    planner = CasePlannerAgent(provider=provider)
    result = await planner.plan(topic="SAH", difficulty="resident", context="SAH content")
    assert result.data.blueprint.diagnosis == "SAH"
    assert result.data.blueprint.phase_count == 3
    assert len(result.data.blueprint.phases) == 3
    provider.generate_structured.assert_called_once()

@pytest.mark.asyncio
async def test_case_planner_passes_retry_notes():
    provider = AsyncMock()
    provider.generate_structured.return_value = _mock_planner_result()
    planner = CasePlannerAgent(provider=provider)
    await planner.plan(topic="SAH", difficulty="resident", context="SAH content", retry_notes=["Phase count too low"])
    call_args = provider.generate_structured.call_args
    prompt = call_args.args[0] if call_args.args else call_args.kwargs.get("prompt", "")
    assert "Phase count too low" in prompt
