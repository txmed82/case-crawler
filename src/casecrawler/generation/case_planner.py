from __future__ import annotations
from pydantic import BaseModel
from casecrawler.generation.prompts import CASE_PLANNER_SYSTEM, build_case_planner_prompt, build_retry_prompt
from casecrawler.llm.base import BaseLLMProvider, StructuredGenerationResult
from casecrawler.models.blueprint import CaseBlueprint
from casecrawler.models.case import Patient

class CasePlannerOutput(BaseModel):
    blueprint: CaseBlueprint
    patient: Patient
    specialty: list[str]

class CasePlannerAgent:
    def __init__(self, provider: BaseLLMProvider) -> None:
        self._provider = provider

    async def plan(self, topic: str, difficulty: str, context: str, retry_notes: list[str] | None = None) -> StructuredGenerationResult:
        prompt = build_case_planner_prompt(topic, difficulty, context)
        if retry_notes:
            prompt = build_retry_prompt(prompt, retry_notes)
        return await self._provider.generate_structured(prompt, CasePlannerOutput, system=CASE_PLANNER_SYSTEM)
