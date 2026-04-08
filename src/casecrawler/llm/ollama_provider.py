from __future__ import annotations

import json

import httpx
from pydantic import BaseModel

from casecrawler.llm.base import BaseLLMProvider, GenerationResult, StructuredGenerationResult


class OllamaProvider(BaseLLMProvider):
    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")

    async def generate(self, prompt: str, system: str = "", **kwargs) -> GenerationResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/api/chat",
                json={"model": self._model, "messages": messages, "stream": False},
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()

        return GenerationResult(
            text=data["message"]["content"],
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            model=data.get("model", self._model),
        )

    async def generate_structured(
        self, prompt: str, schema: type[BaseModel], system: str = "", **kwargs
    ) -> StructuredGenerationResult:
        full_prompt = f"{prompt}\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema.model_json_schema(), indent=2)}"

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": full_prompt})

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": messages,
                    "stream": False,
                    "format": "json",
                },
                timeout=120.0,
            )
            response.raise_for_status()
            resp_data = response.json()

        raw = json.loads(resp_data["message"]["content"])
        data = schema(**raw)
        return StructuredGenerationResult(
            data=data,
            input_tokens=resp_data.get("prompt_eval_count", 0),
            output_tokens=resp_data.get("eval_count", 0),
            model=resp_data.get("model", self._model),
        )
