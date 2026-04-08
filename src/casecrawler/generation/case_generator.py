from __future__ import annotations

from pydantic import BaseModel

from casecrawler.generation.prompts import CASE_GENERATOR_SYSTEM, build_case_generator_prompt
from casecrawler.llm.base import BaseLLMProvider, StructuredGenerationResult
from casecrawler.models.case import GroundTruth, Patient


class CaseGeneratorOutput(BaseModel):
    patient: Patient
    vignette: str
    decision_prompt: str
    ground_truth: GroundTruth
    specialty: list[str]


class CaseGeneratorAgent:
    def __init__(self, provider: BaseLLMProvider) -> None:
        self._provider = provider

    async def generate(
        self,
        topic: str,
        difficulty: str,
        context: str,
        retry_notes: list[str] | None = None,
    ) -> StructuredGenerationResult:
        prompt = build_case_generator_prompt(topic, difficulty, context)

        if retry_notes:
            from casecrawler.generation.prompts import build_retry_prompt
            prompt = build_retry_prompt(prompt, retry_notes)

        return await self._provider.generate_structured(
            prompt=prompt,
            schema=CaseGeneratorOutput,
            system=CASE_GENERATOR_SYSTEM,
        )
