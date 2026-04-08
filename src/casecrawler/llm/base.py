from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class GenerationResult(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class StructuredGenerationResult(BaseModel):
    data: Any
    input_tokens: int
    output_tokens: int
    model: str


class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, system: str = "", **kwargs) -> GenerationResult:
        """Generate a text completion."""

    @abstractmethod
    async def generate_structured(
        self, prompt: str, schema: type[BaseModel], system: str = "", **kwargs
    ) -> StructuredGenerationResult:
        """Generate a response conforming to a Pydantic schema."""
