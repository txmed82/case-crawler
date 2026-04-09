from unittest.mock import AsyncMock
import pytest
from casecrawler.generation.phase_renderer import PhaseRendererAgent, PhaseRendererOutput
from casecrawler.llm.base import StructuredGenerationResult
from casecrawler.models.diagnostics import VitalSigns
from casecrawler.models.phase import CasePhase, PhaseDecision, PhaseOutcome

def _mock_render_result():
    return StructuredGenerationResult(
        data=PhaseRendererOutput(phase=CasePhase(phase_number=1, time_offset="T+0", narrative="A 42-year-old woman presents with thunderclap headache.", vitals=VitalSigns(hr=92, bp_systolic=168, bp_diastolic=95, rr=20, spo2=98.0, temp_c=37.4, gcs=14), lab_results=[], imaging_results=[], clinical_update=None, decisions=[PhaseDecision(action="Order CT head", is_optimal=True, quality="optimal", reasoning="Most sensitive", clinical_outcome="SAH confirmed"), PhaseDecision(action="Discharge", is_optimal=False, quality="catastrophic", reasoning="Misdiagnosis", clinical_outcome="Rebleed at home")], phase_outcome=PhaseOutcome(optimal_next_phase=2, patient_status="stable", narrative_transition="CT ordered stat."))),
        input_tokens=800, output_tokens=600, model="test-model",
    )

@pytest.mark.asyncio
async def test_phase_renderer_renders_phase():
    provider = AsyncMock()
    provider.generate_structured.return_value = _mock_render_result()
    renderer = PhaseRendererAgent(provider=provider)
    result = await renderer.render(blueprint_json='{"diagnosis":"SAH"}', phase_json='{"phase_number":1}', difficulty="resident", context="SAH content", lab_panel_context="CBC reference ranges")
    assert result.data.phase.phase_number == 1
    assert len(result.data.phase.decisions) == 2
    assert result.data.phase.vitals.hr == 92
    provider.generate_structured.assert_called_once()
