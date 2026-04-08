from __future__ import annotations

import json

import openai
from pydantic import BaseModel

from casecrawler.llm.base import BaseLLMProvider, GenerationResult, StructuredGenerationResult


class OpenRouterProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str) -> None:
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self._model = model

    async def generate(self, prompt: str, system: str = "", **kwargs) -> GenerationResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=4096,
        )
        return GenerationResult(
            text=response.choices[0].message.content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=response.model,
        )

    async def generate_structured(
        self, prompt: str, schema: type[BaseModel], system: str = "", **kwargs
    ) -> StructuredGenerationResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": f"{prompt}\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema.model_json_schema(), indent=2)}",
        })

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        raw = json.loads(response.choices[0].message.content)
        data = schema(**raw)
        return StructuredGenerationResult(
            data=data,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=response.model,
        )
