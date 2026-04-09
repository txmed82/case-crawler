from __future__ import annotations
from pydantic import BaseModel
from casecrawler.generation.prompts import BLUEPRINT_REVIEWER_SYSTEM, build_blueprint_reviewer_prompt
from casecrawler.llm.base import BaseLLMProvider, StructuredGenerationResult

class BlueprintReviewResult(BaseModel):
    approved: bool
    notes: list[str]

class BlueprintReviewerAgent:
    def __init__(self, provider: BaseLLMProvider) -> None:
        self._provider = provider

    async def review(self, blueprint_json: str, context: str) -> StructuredGenerationResult:
        prompt = build_blueprint_reviewer_prompt(blueprint_json, context)
        return await self._provider.generate_structured(prompt, BlueprintReviewResult, system=BLUEPRINT_REVIEWER_SYSTEM)
