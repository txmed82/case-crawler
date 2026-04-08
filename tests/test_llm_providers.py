import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from casecrawler.llm.openai_provider import OpenAIProvider
from casecrawler.llm.openrouter_provider import OpenRouterProvider
from casecrawler.llm.ollama_provider import OllamaProvider
from casecrawler.models.case import Patient


# --- OpenAI ---


@pytest.fixture
def openai_provider():
    return OpenAIProvider(api_key="test-key", model="gpt-4o")


@pytest.mark.asyncio
async def test_openai_generate(openai_provider):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Generated text"))]
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_response.model = "gpt-4o"

    with patch.object(
        openai_provider._client.chat.completions, "create",
        new_callable=AsyncMock, return_value=mock_response,
    ):
        result = await openai_provider.generate("test prompt", system="system")
        assert result.text == "Generated text"
        assert result.input_tokens == 100


@pytest.mark.asyncio
async def test_openai_generate_structured(openai_provider):
    patient_json = json.dumps({"age": 42, "sex": "female", "demographics": "Healthy"})
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=patient_json))]
    mock_response.usage.prompt_tokens = 150
    mock_response.usage.completion_tokens = 30
    mock_response.model = "gpt-4o"

    with patch.object(
        openai_provider._client.chat.completions, "create",
        new_callable=AsyncMock, return_value=mock_response,
    ):
        result = await openai_provider.generate_structured("create patient", Patient)
        assert result.data.age == 42


# --- OpenRouter ---


def test_openrouter_uses_correct_base_url():
    provider = OpenRouterProvider(api_key="test-key", model="anthropic/claude-3.5-sonnet")
    assert provider._client.base_url.host == "openrouter.ai"


# --- Ollama ---


@pytest.fixture
def ollama_provider():
    return OllamaProvider(model="llama3", base_url="http://localhost:11434")


@pytest.mark.asyncio
async def test_ollama_generate(ollama_provider, httpx_mock):
    import re
    httpx_mock.add_response(
        url=re.compile(r"http://localhost:11434/api/chat"),
        json={
            "message": {"content": "Generated text"},
            "prompt_eval_count": 100,
            "eval_count": 50,
            "model": "llama3",
        },
    )
    result = await ollama_provider.generate("test prompt", system="system")
    assert result.text == "Generated text"
    assert result.input_tokens == 100


@pytest.mark.asyncio
async def test_ollama_generate_structured(ollama_provider, httpx_mock):
    import re
    patient_json = json.dumps({"age": 42, "sex": "female", "demographics": "Healthy"})
    httpx_mock.add_response(
        url=re.compile(r"http://localhost:11434/api/chat"),
        json={
            "message": {"content": patient_json},
            "prompt_eval_count": 150,
            "eval_count": 30,
            "model": "llama3",
        },
    )
    result = await ollama_provider.generate_structured("create patient", Patient)
    assert result.data.age == 42
