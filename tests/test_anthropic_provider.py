import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from casecrawler.llm.anthropic_provider import AnthropicProvider
from casecrawler.models.case import Patient


@pytest.fixture
def provider():
    return AnthropicProvider(api_key="test-key", model="claude-sonnet-4-6")


@pytest.mark.asyncio
async def test_generate(provider):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Generated text")]
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 50
    mock_response.model = "claude-sonnet-4-6"

    with patch.object(provider._client.messages, "create", new_callable=AsyncMock, return_value=mock_response):
        result = await provider.generate("test prompt", system="system prompt")
        assert result.text == "Generated text"
        assert result.input_tokens == 100
        assert result.output_tokens == 50


@pytest.mark.asyncio
async def test_generate_structured(provider):
    patient_json = json.dumps({"age": 42, "sex": "female", "demographics": "Healthy"})
    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(type="tool_use", input={"age": 42, "sex": "female", "demographics": "Healthy"}),
    ]
    mock_response.usage.input_tokens = 150
    mock_response.usage.output_tokens = 30
    mock_response.model = "claude-sonnet-4-6"

    with patch.object(provider._client.messages, "create", new_callable=AsyncMock, return_value=mock_response):
        result = await provider.generate_structured("create a patient", Patient, system="system")
        assert result.data.age == 42
        assert result.data.sex == "female"
        assert result.input_tokens == 150
