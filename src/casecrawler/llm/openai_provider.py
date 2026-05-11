from __future__ import annotations

import openai
from pydantic import BaseModel

from casecrawler.llm._openai_compat import chat_complete, chat_complete_structured
from casecrawler.llm.base import BaseLLMProvider, GenerationResult, StructuredGenerationResult


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str) -> None:
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model

    async def generate(self, prompt: str, system: str = "", **kwargs) -> GenerationResult:
        return await chat_complete(
            self._client,
            model=self._model,
            prompt=prompt,
            system=system,
            **kwargs,
        )

    async def generate_structured(
        self, prompt: str, schema: type[BaseModel], system: str = "", **kwargs
    ) -> StructuredGenerationResult:
        return await chat_complete_structured(
            self._client,
            model=self._model,
            prompt=prompt,
            schema=schema,
            system=system,
            provider_label="OpenAI",
            **kwargs,
        )
