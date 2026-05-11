"""Regression tests for Phase 5 (refactor + CodeRabbit fixes).

These pin the behavioral changes the CR review pushed back on:

- Native ``json_schema`` is preferred; the legacy ``json_object``
  fallback only fires on 400s that look like response_format
  complaints, not for every BadRequestError.
- Missing ``response.usage`` no longer crashes the wrapper; it returns
  zero tokens instead.
- Provider-level ``**kwargs`` (e.g. ``temperature``) flow through to
  the underlying ``client.chat.completions.create`` call.
- Oversized backend stderr/stdout is truncated in the formatted
  RuntimeError so a chatty failing process can't produce a multi-MB
  exception string.
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from pydantic import BaseModel

from casecrawler.generation._external_subprocess import (
    _MAX_OUTPUT_CHARS,
    run_external_command,
)
from casecrawler.llm.openai_provider import OpenAIProvider


class _Schema(BaseModel):
    age: int
    sex: str


def _mock_chat_response(content: str, *, with_usage: bool = True) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    if with_usage:
        response.usage.prompt_tokens = 100
        response.usage.completion_tokens = 50
    else:
        response.usage = None
    response.model = "test-model"
    return response


def _bad_request(message: str) -> openai.BadRequestError:
    """Build a BadRequestError that behaves like the real SDK one.

    The real exception requires ``response`` and ``body`` kwargs, but for
    test purposes we just need ``str(exc)`` to contain the message so
    the substring match logic in ``_is_json_schema_unsupported`` can run.
    """

    return openai.BadRequestError(
        message=message,
        response=MagicMock(status_code=400),
        body={"error": {"message": message}},
    )


# ---- Narrow BadRequestError fallback ----------------------------------------


@pytest.mark.asyncio
async def test_structured_falls_back_when_server_rejects_json_schema():
    """A 400 mentioning ``response_format`` should trigger the legacy
    ``json_object`` fallback, not propagate."""

    provider = OpenAIProvider(api_key="test", model="gpt-4")
    happy = _mock_chat_response(json.dumps({"age": 42, "sex": "f"}))
    create_mock = AsyncMock(
        side_effect=[
            _bad_request("response_format is not supported on this model"),
            happy,
        ]
    )
    with patch.object(
        provider._client.chat.completions, "create", new=create_mock
    ):
        result = await provider.generate_structured("p", _Schema)
    assert result.data.age == 42
    # Two calls: first the json_schema attempt, then the json_object retry.
    assert create_mock.await_count == 2
    second_call_kwargs = create_mock.await_args_list[1].kwargs
    assert second_call_kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_structured_does_not_fall_back_on_unrelated_400():
    """A 400 about, e.g., the model not existing should NOT trigger the
    legacy fallback — the caller deserves the real error."""

    provider = OpenAIProvider(api_key="test", model="gpt-4")
    create_mock = AsyncMock(
        side_effect=_bad_request("The model `gpt-99` does not exist")
    )
    with patch.object(
        provider._client.chat.completions, "create", new=create_mock
    ):
        with pytest.raises(openai.BadRequestError, match="does not exist"):
            await provider.generate_structured("p", _Schema)
    # Only one attempt — no silent retry that would mask the real failure.
    assert create_mock.await_count == 1


# ---- Missing usage guard ----------------------------------------------------


@pytest.mark.asyncio
async def test_structured_handles_response_without_usage():
    """Some OpenAI-compatible proxies omit ``response.usage``. We should
    return zero tokens, not AttributeError."""

    provider = OpenAIProvider(api_key="test", model="gpt-4")
    response = _mock_chat_response(
        json.dumps({"age": 42, "sex": "f"}), with_usage=False
    )
    with patch.object(
        provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=response,
    ):
        result = await provider.generate_structured("p", _Schema)
    assert result.input_tokens == 0
    assert result.output_tokens == 0


# ---- kwargs passthrough -----------------------------------------------------


@pytest.mark.asyncio
async def test_provider_forwards_kwargs_to_underlying_create():
    """``temperature`` and other generation params passed to the provider
    must flow through to ``client.chat.completions.create`` rather than
    being silently dropped."""

    provider = OpenAIProvider(api_key="test", model="gpt-4")
    response = _mock_chat_response(json.dumps({"age": 42, "sex": "f"}))
    create_mock = AsyncMock(return_value=response)
    with patch.object(
        provider._client.chat.completions, "create", new=create_mock
    ):
        await provider.generate_structured(
            "p", _Schema, temperature=0.2, top_p=0.9
        )
    kwargs = create_mock.await_args.kwargs
    assert kwargs["temperature"] == 0.2
    assert kwargs["top_p"] == 0.9


# ---- Truncated subprocess output --------------------------------------------


def test_run_external_command_truncates_oversized_stderr(monkeypatch):
    huge = "X" * (_MAX_OUTPUT_CHARS * 4)

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=["fake"],
            output="",
            stderr=huge,
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as exc_info:
        run_external_command(
            ["fake"], "{}", backend_label="imaging", timeout_seconds=5.0
        )
    message = str(exc_info.value)
    # Real stderr was 8000 chars; the formatted exception must be much smaller.
    assert len(message) < _MAX_OUTPUT_CHARS * 2
    assert "(truncated)" in message
