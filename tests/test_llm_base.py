import os
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from casecrawler.llm.base import GenerationResult, StructuredGenerationResult
from casecrawler.llm.factory import get_provider


class StructuredPatient(BaseModel):
    age: int
    sex: str
    demographics: str


def test_generation_result():
    r = GenerationResult(text="Hello", input_tokens=10, output_tokens=5, model="test")
    assert r.text == "Hello"
    assert r.input_tokens == 10


def test_structured_generation_result():
    p = StructuredPatient(age=42, sex="female", demographics="Healthy")
    r = StructuredGenerationResult(data=p, input_tokens=10, output_tokens=20, model="test")
    assert r.data.age == 42


def test_factory_raises_without_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="API key"):
            get_provider("anthropic", "claude-sonnet-4-6")


def test_factory_raises_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("unknown_provider", "some-model")
