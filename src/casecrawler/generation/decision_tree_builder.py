from __future__ import annotations

from pydantic import BaseModel

from casecrawler.generation.prompts import DECISION_TREE_SYSTEM, build_decision_tree_prompt
from casecrawler.llm.base import BaseLLMProvider, StructuredGenerationResult
from casecrawler.models.case import Complication, DecisionChoice


class DecisionTreeOutput(BaseModel):
    decision_tree: list[DecisionChoice]
    complications: list[Complication]


class DecisionTreeBuilderAgent:
    def __init__(self, provider: BaseLLMProvider) -> None:
        self._provider = provider

    async def build(
        self,
        vignette: str,
        ground_truth_json: str,
        difficulty: str,
        context: str,
        retry_notes: list[str] | None = None,
    ) -> StructuredGenerationResult:
        prompt = build_decision_tree_prompt(vignette, ground_truth_json, difficulty, context)

        if retry_notes:
            from casecrawler.generation.prompts import build_retry_prompt
            prompt = build_retry_prompt(prompt, retry_notes)

        return await self._provider.generate_structured(
            prompt=prompt,
            schema=DecisionTreeOutput,
            system=DECISION_TREE_SYSTEM,
        )
