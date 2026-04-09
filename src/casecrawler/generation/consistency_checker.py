from __future__ import annotations
from pydantic import BaseModel
from casecrawler.generation.prompts import CONSISTENCY_CHECKER_SYSTEM, build_consistency_checker_prompt
from casecrawler.llm.base import BaseLLMProvider, StructuredGenerationResult

class ConsistencyIssue(BaseModel):
    phase_number: int
    field: str
    issue: str
    suggested_fix: str

class ConsistencyCheckerOutput(BaseModel):
    issues: list[ConsistencyIssue]

class ConsistencyCheckerAgent:
    def __init__(self, provider: BaseLLMProvider) -> None:
        self._provider = provider

    async def check(self, phases_json: str) -> StructuredGenerationResult:
        prompt = build_consistency_checker_prompt(phases_json)
        return await self._provider.generate_structured(prompt, ConsistencyCheckerOutput, system=CONSISTENCY_CHECKER_SYSTEM)
