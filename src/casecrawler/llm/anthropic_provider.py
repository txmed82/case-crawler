from __future__ import annotations

import json

import anthropic
from pydantic import BaseModel

from casecrawler.llm.base import BaseLLMProvider, GenerationResult, StructuredGenerationResult


class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    async def generate(self, prompt: str, system: str = "", **kwargs) -> GenerationResult:
        messages = [{"role": "user", "content": prompt}]
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system or "You are a helpful assistant.",
            messages=messages,
        )
        text = response.content[0].text
        return GenerationResult(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
        )

    async def generate_structured(
        self, prompt: str, schema: type[BaseModel], system: str = "", **kwargs
    ) -> StructuredGenerationResult:
        # Convert Pydantic schema to Anthropic tool format
        json_schema = schema.model_json_schema()
        tool = {
            "name": "structured_output",
            "description": f"Output structured data as {schema.__name__}",
            "input_schema": json_schema,
        }

        messages = [{"role": "user", "content": prompt}]
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system or "You are a helpful assistant.",
            messages=messages,
            tools=[tool],
            tool_choice={"type": "tool", "name": "structured_output"},
        )

        # Extract tool use result
        tool_result = None
        for block in response.content:
            if block.type == "tool_use":
                tool_result = block.input
                break

        if tool_result is None:
            raise ValueError("No structured output returned from Anthropic")

        data = schema(**tool_result)
        return StructuredGenerationResult(
            data=data,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
        )
