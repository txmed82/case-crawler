"""Shared OpenAI-compatible chat completions logic.

Both ``OpenAIProvider`` and ``OpenRouterProvider`` talk to the same
SDK (``openai.AsyncOpenAI``), so the retry / parse / structured-output
machinery now lives here. Each concrete provider only needs to supply
its base URL and any provider-specific tag for error messages.
"""

from __future__ import annotations

import json
from typing import Any

import openai
from pydantic import BaseModel, ValidationError

from casecrawler.llm.base import GenerationResult, StructuredGenerationResult


def build_messages(prompt: str, system: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


async def chat_complete(
    client: openai.AsyncOpenAI,
    *,
    model: str,
    prompt: str,
    system: str = "",
    max_tokens: int = 4096,
) -> GenerationResult:
    response = await client.chat.completions.create(
        model=model,
        messages=build_messages(prompt, system),
        max_tokens=max_tokens,
    )
    return GenerationResult(
        text=response.choices[0].message.content,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        model=response.model,
    )


async def chat_complete_structured(
    client: openai.AsyncOpenAI,
    *,
    model: str,
    prompt: str,
    schema: type[BaseModel],
    system: str = "",
    max_tokens: int = 4096,
    provider_label: str = "OpenAI-compatible",
) -> StructuredGenerationResult:
    """Generate a structured response using native ``json_schema`` mode.

    Modern OpenAI-compatible APIs accept

        response_format={
            "type": "json_schema",
            "json_schema": {"name": ..., "schema": ..., "strict": False},
        }

    which both shrinks the prompt (no need to embed the schema in the
    user message) and gives the model the schema in a more reliable
    channel. We use ``strict=False`` so that Pydantic schemas with
    optional fields, defaults, or unions still work; the post-decode
    Pydantic validation catches the rare drift.

    Older / non-compliant servers can fall back to plain ``json_object``
    mode by raising ``openai.BadRequestError`` -- we catch that and
    retry once with the legacy shape.
    """

    json_schema = schema.model_json_schema()
    response_format: dict[str, Any] = {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": json_schema,
            "strict": False,
        },
    }
    messages = build_messages(prompt, system)
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            response_format=response_format,
        )
    except openai.BadRequestError:
        # Fallback for endpoints that don't yet support json_schema mode.
        # We append the schema to the prompt as a last-resort hint so the
        # model still has structural guidance.
        messages = build_messages(
            f"{prompt}\n\nRespond with valid JSON matching this schema:\n"
            f"{json.dumps(json_schema, indent=2)}",
            system,
        )
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

    content = response.choices[0].message.content
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{provider_label} structured response was not valid JSON: {exc}"
        ) from exc
    try:
        data = schema(**raw)
    except ValidationError as exc:
        raise ValueError(
            f"{provider_label} structured response did not match schema "
            f"{schema.__name__}: {exc}"
        ) from exc
    return StructuredGenerationResult(
        data=data,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        model=response.model,
    )
