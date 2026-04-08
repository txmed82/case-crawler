from __future__ import annotations

from casecrawler.generation.prompts import CLINICAL_REVIEWER_SYSTEM, build_reviewer_prompt
from casecrawler.llm.base import BaseLLMProvider, StructuredGenerationResult
from casecrawler.models.case import ReviewResult


class ClinicalReviewerAgent:
    def __init__(self, provider: BaseLLMProvider, threshold: float = 0.7) -> None:
        self._provider = provider
        self._threshold = threshold

    async def review(self, case_json: str, context: str) -> StructuredGenerationResult:
        prompt = build_reviewer_prompt(case_json, context, self._threshold)
        return await self._provider.generate_structured(
            prompt=prompt,
            schema=ReviewResult,
            system=CLINICAL_REVIEWER_SYSTEM,
        )
