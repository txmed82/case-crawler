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


# A 400 only counts as "this endpoint can't do json_schema mode" if it
# specifically names the feature. Generic tokens like "unsupported" on
# their own are too noisy — a quota or model-tier 400 also says
# "unsupported" and shouldn't silently downgrade to json_object.
def _is_json_schema_unsupported(exc: openai.BadRequestError) -> bool:
    message = str(exc).lower()
    if "response_format" not in message and "json_schema" not in message:
        return False
    return (
        "unsupported" in message
        or "not supported" in message
        or "invalid type" in message
        or "unrecognized" in message
    )


def _usage_tokens(response: Any) -> tuple[int, int]:
    """Return (prompt_tokens, completion_tokens), defaulting to 0.

    Some OpenAI-compatible endpoints (notably older proxy routers) omit
    the ``usage`` block on streaming-tail responses. Treat missing usage
    as zero rather than crashing the caller mid-pipeline.
    """

    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return (
        getattr(usage, "prompt_tokens", 0) or 0,
        getattr(usage, "completion_tokens", 0) or 0,
    )


def _content_or_empty(response: Any) -> str:
    """Return ``choices[0].message.content`` or "" if the server omitted it."""

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError):
        return ""
    return content or ""


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
    **kwargs: Any,
) -> GenerationResult:
    response = await client.chat.completions.create(
        model=model,
        messages=build_messages(prompt, system),
        max_tokens=max_tokens,
        **kwargs,
    )
    prompt_tokens, completion_tokens = _usage_tokens(response)
    return GenerationResult(
        text=_content_or_empty(response),
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
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
    **kwargs: Any,
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
    retry once with the legacy shape, but ONLY when the error message
    looks like an unsupported-response_format complaint. Other 400s
    (auth, malformed payload, model-not-found) re-raise unchanged so
    callers see the real failure.
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
            **kwargs,
        )
    except openai.BadRequestError as exc:
        if not _is_json_schema_unsupported(exc):
            raise
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
            **kwargs,
        )

    content = _content_or_empty(response)
    if not content:
        raise ValueError(
            f"{provider_label} structured response was empty (no content "
            "in choices[0].message)."
        )
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{provider_label} structured response was not valid JSON: {exc}"
        ) from exc
    try:
        data = schema.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(
            f"{provider_label} structured response did not match schema "
            f"{schema.__name__}: {exc}"
        ) from exc
    prompt_tokens, completion_tokens = _usage_tokens(response)
    return StructuredGenerationResult(
        data=data,
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        model=response.model,
    )
