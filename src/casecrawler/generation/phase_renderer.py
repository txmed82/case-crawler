from __future__ import annotations
from pydantic import BaseModel
from casecrawler.generation.prompts import PHASE_RENDERER_SYSTEM, build_phase_renderer_prompt
from casecrawler.llm.base import BaseLLMProvider, StructuredGenerationResult
from casecrawler.models.phase import CasePhase

class PhaseRendererOutput(BaseModel):
    phase: CasePhase

class PhaseRendererAgent:
    def __init__(self, provider: BaseLLMProvider) -> None:
        self._provider = provider

    async def render(self, blueprint_json: str, phase_json: str, difficulty: str, context: str, lab_panel_context: str) -> StructuredGenerationResult:
        prompt = build_phase_renderer_prompt(blueprint_json=blueprint_json, phase_json=phase_json, difficulty=difficulty, context=context, lab_panel_context=lab_panel_context)
        return await self._provider.generate_structured(prompt, PhaseRendererOutput, system=PHASE_RENDERER_SYSTEM)
