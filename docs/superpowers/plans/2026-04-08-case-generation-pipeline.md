# Case Generation Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-agent LLM pipeline that generates realistic clinical cases from ingested medical knowledge, with clinical review, SQLite storage, and an interactive case player UI.

**Architecture:** Four-stage pipeline (Retriever → Case Generator → Decision Tree Builder → Clinical Reviewer) with targeted retry on rejection. Cases stored in SQLite. Extends existing CLI, API, and React UI.

**Tech Stack:** Existing stack + anthropic SDK, openai SDK, sqlite3, new Pydantic models for cases

**Spec:** `docs/superpowers/specs/2026-04-08-case-generation-pipeline-design.md`

---

## File Map

### New Files

| File | Responsibility |
|------|---------------|
| `src/casecrawler/models/case.py` | DifficultyLevel, Patient, GroundTruth, DecisionChoice, Complication, ReviewResult, GeneratedCase |
| `src/casecrawler/llm/base.py` | BaseLLMProvider, GenerationResult, StructuredGenerationResult |
| `src/casecrawler/llm/factory.py` | get_provider() from config |
| `src/casecrawler/llm/anthropic_provider.py` | Anthropic Claude implementation |
| `src/casecrawler/llm/openai_provider.py` | OpenAI implementation |
| `src/casecrawler/llm/openrouter_provider.py` | OpenRouter implementation (wraps openai SDK) |
| `src/casecrawler/llm/ollama_provider.py` | Ollama local model implementation |
| `src/casecrawler/storage/case_store.py` | SQLite CRUD for generated cases |
| `src/casecrawler/generation/retriever.py` | ChromaDB query + credibility ranking |
| `src/casecrawler/generation/prompts.py` | System prompts for all LLM stages |
| `src/casecrawler/generation/case_generator.py` | Stage 2: vignette + ground truth |
| `src/casecrawler/generation/decision_tree_builder.py` | Stage 3: choices + complications |
| `src/casecrawler/generation/clinical_reviewer.py` | Stage 4: scoring + feedback |
| `src/casecrawler/generation/pipeline.py` | Orchestrates stages + retry logic |
| `src/casecrawler/api/routes/generate.py` | POST /generate, GET /generate/{job_id} |
| `src/casecrawler/api/routes/cases.py` | GET /cases, GET /cases/{id}, GET /cases/export |
| `ui/src/pages/GeneratePage.tsx` | Case generation form + progress |
| `ui/src/pages/CasesPage.tsx` | Browse/filter cases |
| `ui/src/pages/PlayCasePage.tsx` | Interactive case player |
| `ui/src/components/VignetteCard.tsx` | Case vignette display |
| `ui/src/components/DecisionCards.tsx` | Choice selection cards |
| `ui/src/components/OutcomeReveal.tsx` | Choice outcome display |
| `ui/src/components/CaseDebrief.tsx` | Full case breakdown |
| `ui/src/components/DecisionTreeViz.tsx` | Visual decision tree |

### Modified Files

| File | Change |
|------|--------|
| `src/casecrawler/models/config.py` | Add LlmConfig, GenerationConfig |
| `src/casecrawler/cli.py` | Add generate, cases commands |
| `src/casecrawler/api/app.py` | Add generate, cases routers |
| `ui/src/App.tsx` | Add routes + nav links |
| `ui/src/api/client.ts` | Add generate/cases API functions |
| `pyproject.toml` | Add optional anthropic/openai deps |

---

## Task 1: Case Data Models + Config Updates

**Files:**
- Create: `src/casecrawler/models/case.py`
- Modify: `src/casecrawler/models/config.py`
- Test: `tests/test_case_models.py`

- [ ] **Step 1: Write tests for case models**

```python
# tests/test_case_models.py
from casecrawler.models.case import (
    Complication,
    DecisionChoice,
    DifficultyLevel,
    GeneratedCase,
    GroundTruth,
    Patient,
    ReviewResult,
)


def test_difficulty_levels():
    assert DifficultyLevel.MEDICAL_STUDENT == "medical_student"
    assert DifficultyLevel.RESIDENT == "resident"
    assert DifficultyLevel.ATTENDING == "attending"


def test_patient_creation():
    p = Patient(age=42, sex="female", demographics="No significant PMH")
    assert p.age == 42
    assert p.sex == "female"


def test_ground_truth():
    gt = GroundTruth(
        diagnosis="aneurysmal subarachnoid hemorrhage",
        optimal_next_step="Non-contrast CT head",
        rationale="Most sensitive initial test for SAH",
        key_findings=["thunderclap headache", "neck stiffness"],
    )
    assert len(gt.key_findings) == 2


def test_decision_choice_correct():
    dc = DecisionChoice(
        choice="Order non-contrast CT head",
        is_correct=True,
        error_type=None,
        reasoning="Most sensitive for acute SAH",
        outcome="CT shows hyperdense material in basal cisterns",
        consequence=None,
        next_decision="Consult neurosurgery",
    )
    assert dc.is_correct is True
    assert dc.error_type is None


def test_decision_choice_wrong():
    dc = DecisionChoice(
        choice="Discharge with migraine diagnosis",
        is_correct=False,
        error_type="catastrophic",
        reasoning="Misdiagnosis",
        outcome="Patient rebleeds at home",
        consequence="Mortality",
        next_decision=None,
    )
    assert dc.error_type == "catastrophic"


def test_complication():
    c = Complication(
        trigger="delayed_diagnosis",
        detail="6 hour delay",
        event="Rebleeding",
        outcome="50% mortality",
    )
    assert c.trigger == "delayed_diagnosis"


def test_review_result_approved():
    r = ReviewResult(
        accuracy_score=0.95,
        pedagogy_score=0.88,
        bias_score=0.92,
        approved=True,
        notes=[],
    )
    assert r.approved is True


def test_review_result_rejected():
    r = ReviewResult(
        accuracy_score=0.5,
        pedagogy_score=0.88,
        bias_score=0.92,
        approved=False,
        notes=["Diagnosis is incorrect for the presented findings"],
    )
    assert r.approved is False
    assert len(r.notes) == 1


def test_generated_case_full():
    case = GeneratedCase(
        case_id="test-123",
        topic="subarachnoid hemorrhage",
        difficulty=DifficultyLevel.RESIDENT,
        specialty=["neurosurgery", "emergency_medicine"],
        patient=Patient(age=42, sex="female", demographics="Healthy"),
        vignette="A 42-year-old woman presents...",
        decision_prompt="What would you do next?",
        ground_truth=GroundTruth(
            diagnosis="SAH",
            optimal_next_step="CT head",
            rationale="Most sensitive",
            key_findings=["thunderclap headache"],
        ),
        decision_tree=[
            DecisionChoice(
                choice="CT head",
                is_correct=True,
                error_type=None,
                reasoning="Correct",
                outcome="SAH confirmed",
                consequence=None,
                next_decision=None,
            ),
        ],
        complications=[
            Complication(
                trigger="delayed_diagnosis",
                detail="6h delay",
                event="Rebleed",
                outcome="Death",
            ),
        ],
        review=ReviewResult(
            accuracy_score=0.95,
            pedagogy_score=0.9,
            bias_score=0.95,
            approved=True,
            notes=[],
        ),
        sources=[{"type": "pubmed", "reference": "PMID:123", "chunk_ids": ["abc"]}],
        metadata={"generated_at": "2026-04-08", "model": "claude-sonnet-4-6", "retry_count": 0},
    )
    assert case.case_id == "test-123"
    assert case.difficulty == DifficultyLevel.RESIDENT
    assert len(case.decision_tree) == 1
    assert case.review.approved is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_case_models.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement case models**

```python
# src/casecrawler/models/case.py
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class DifficultyLevel(str, Enum):
    MEDICAL_STUDENT = "medical_student"
    RESIDENT = "resident"
    ATTENDING = "attending"


class Patient(BaseModel):
    age: int
    sex: str
    demographics: str


class GroundTruth(BaseModel):
    diagnosis: str
    optimal_next_step: str
    rationale: str
    key_findings: list[str]


class DecisionChoice(BaseModel):
    choice: str
    is_correct: bool
    error_type: str | None = None
    reasoning: str
    outcome: str
    consequence: str | None = None
    next_decision: str | None = None


class Complication(BaseModel):
    trigger: str
    detail: str
    event: str
    outcome: str


class ReviewResult(BaseModel):
    accuracy_score: float
    pedagogy_score: float
    bias_score: float
    approved: bool
    notes: list[str]


class GeneratedCase(BaseModel):
    case_id: str
    topic: str
    difficulty: DifficultyLevel
    specialty: list[str]
    patient: Patient
    vignette: str
    decision_prompt: str
    ground_truth: GroundTruth
    decision_tree: list[DecisionChoice]
    complications: list[Complication]
    review: ReviewResult | None = None
    sources: list[dict]
    metadata: dict
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_case_models.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Update config models**

Add to the end of `src/casecrawler/models/config.py`:

```python
class LlmConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    ollama_base_url: str = "http://localhost:11434"


class GenerationConfig(BaseModel):
    max_retries: int = 3
    review_threshold: float = 0.7
    default_difficulty: str = "resident"
    retriever_chunk_count: int = 25
```

And add them to `AppConfig`:

```python
class AppConfig(BaseModel):
    ingestion: IngestionConfig = IngestionConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    storage: StorageConfig = StorageConfig()
    api: ApiConfig = ApiConfig()
    llm: LlmConfig = LlmConfig()
    generation: GenerationConfig = GenerationConfig()
```

- [ ] **Step 6: Update pyproject.toml optional dependencies**

Add to `[project.optional-dependencies]`:

```toml
anthropic = ["anthropic>=0.40"]
openai = ["openai>=1.50"]
all-llm = ["anthropic>=0.40", "openai>=1.50"]
```

- [ ] **Step 7: Commit**

```bash
git add src/casecrawler/models/case.py src/casecrawler/models/config.py tests/test_case_models.py pyproject.toml
git commit -m "feat: case data models, LLM/generation config, optional LLM dependencies"
```

---

## Task 2: LLM Provider Base + Factory

**Files:**
- Create: `src/casecrawler/llm/base.py`
- Create: `src/casecrawler/llm/factory.py`
- Create: `src/casecrawler/llm/__init__.py`
- Test: `tests/test_llm_base.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_llm_base.py
import os
from unittest.mock import patch

import pytest

from casecrawler.llm.base import BaseLLMProvider, GenerationResult, StructuredGenerationResult
from casecrawler.llm.factory import get_provider


def test_generation_result():
    r = GenerationResult(text="Hello", input_tokens=10, output_tokens=5, model="test")
    assert r.text == "Hello"
    assert r.input_tokens == 10


def test_structured_generation_result():
    from casecrawler.models.case import Patient

    p = Patient(age=42, sex="female", demographics="Healthy")
    r = StructuredGenerationResult(data=p, input_tokens=10, output_tokens=20, model="test")
    assert r.data.age == 42


def test_factory_raises_without_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="API key"):
            get_provider("anthropic", "claude-sonnet-4-6")


def test_factory_raises_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("unknown_provider", "some-model")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_base.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement base classes**

```python
# src/casecrawler/llm/__init__.py
```

```python
# src/casecrawler/llm/base.py
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
```

- [ ] **Step 4: Implement factory**

```python
# src/casecrawler/llm/factory.py
from __future__ import annotations

from casecrawler.config import get_env
from casecrawler.llm.base import BaseLLMProvider

_PROVIDER_KEY_MAP = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "ollama": None,
}


def get_provider(provider: str, model: str, **kwargs) -> BaseLLMProvider:
    """Create an LLM provider instance from config."""
    if provider not in _PROVIDER_KEY_MAP:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(_PROVIDER_KEY_MAP.keys())}")

    required_key = _PROVIDER_KEY_MAP[provider]
    if required_key:
        api_key = get_env(required_key)
        if not api_key:
            raise ValueError(
                f"API key {required_key} not set. Add it to your .env file for provider '{provider}'."
            )
    else:
        api_key = None

    if provider == "anthropic":
        from casecrawler.llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider(api_key=api_key, model=model)
    elif provider == "openai":
        from casecrawler.llm.openai_provider import OpenAIProvider
        return OpenAIProvider(api_key=api_key, model=model)
    elif provider == "openrouter":
        from casecrawler.llm.openrouter_provider import OpenRouterProvider
        return OpenRouterProvider(api_key=api_key, model=model)
    elif provider == "ollama":
        from casecrawler.llm.ollama_provider import OllamaProvider
        base_url = kwargs.get("base_url", "http://localhost:11434")
        return OllamaProvider(model=model, base_url=base_url)

    raise ValueError(f"Unknown provider: {provider}")
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_llm_base.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/casecrawler/llm/ tests/test_llm_base.py
git commit -m "feat: LLM provider base classes and factory"
```

---

## Task 3: Anthropic Provider

**Files:**
- Create: `src/casecrawler/llm/anthropic_provider.py`
- Test: `tests/test_anthropic_provider.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_anthropic_provider.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_anthropic_provider.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Install anthropic SDK**

Run: `source .venv/bin/activate && pip install "anthropic>=0.40"`

- [ ] **Step 4: Implement Anthropic provider**

```python
# src/casecrawler/llm/anthropic_provider.py
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
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_anthropic_provider.py -v`
Expected: All 2 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/casecrawler/llm/anthropic_provider.py tests/test_anthropic_provider.py
git commit -m "feat: Anthropic LLM provider with structured output via tool use"
```

---

## Task 4: OpenAI + OpenRouter + Ollama Providers

**Files:**
- Create: `src/casecrawler/llm/openai_provider.py`
- Create: `src/casecrawler/llm/openrouter_provider.py`
- Create: `src/casecrawler/llm/ollama_provider.py`
- Test: `tests/test_llm_providers.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_llm_providers.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_providers.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Install openai SDK**

Run: `source .venv/bin/activate && pip install "openai>=1.50"`

- [ ] **Step 4: Implement OpenAI provider**

```python
# src/casecrawler/llm/openai_provider.py
from __future__ import annotations

import json

import openai
from pydantic import BaseModel

from casecrawler.llm.base import BaseLLMProvider, GenerationResult, StructuredGenerationResult


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str) -> None:
        self._client = openai.AsyncOpenAI(api_key=api_key)
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
```

- [ ] **Step 5: Implement OpenRouter provider**

```python
# src/casecrawler/llm/openrouter_provider.py
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
```

- [ ] **Step 6: Implement Ollama provider**

```python
# src/casecrawler/llm/ollama_provider.py
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
```

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_llm_providers.py -v`
Expected: All 5 tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/casecrawler/llm/openai_provider.py src/casecrawler/llm/openrouter_provider.py src/casecrawler/llm/ollama_provider.py tests/test_llm_providers.py
git commit -m "feat: OpenAI, OpenRouter, and Ollama LLM providers"
```

---

## Task 5: SQLite Case Store

**Files:**
- Create: `src/casecrawler/storage/__init__.py`
- Create: `src/casecrawler/storage/case_store.py`
- Test: `tests/test_case_store.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_case_store.py
import tempfile

import pytest

from casecrawler.models.case import (
    Complication,
    DecisionChoice,
    DifficultyLevel,
    GeneratedCase,
    GroundTruth,
    Patient,
    ReviewResult,
)
from casecrawler.storage.case_store import CaseStore


def _make_case(case_id: str = "test-1", topic: str = "SAH", difficulty: str = "resident") -> GeneratedCase:
    return GeneratedCase(
        case_id=case_id,
        topic=topic,
        difficulty=DifficultyLevel(difficulty),
        specialty=["neurosurgery", "emergency_medicine"],
        patient=Patient(age=42, sex="female", demographics="Healthy"),
        vignette="A 42-year-old woman presents with thunderclap headache.",
        decision_prompt="What would you do next?",
        ground_truth=GroundTruth(
            diagnosis="SAH", optimal_next_step="CT head",
            rationale="Most sensitive", key_findings=["thunderclap headache"],
        ),
        decision_tree=[
            DecisionChoice(
                choice="CT head", is_correct=True, error_type=None,
                reasoning="Correct", outcome="SAH confirmed",
                consequence=None, next_decision=None,
            ),
        ],
        complications=[
            Complication(trigger="delayed_diagnosis", detail="6h", event="Rebleed", outcome="Death"),
        ],
        review=ReviewResult(
            accuracy_score=0.95, pedagogy_score=0.9, bias_score=0.95, approved=True, notes=[],
        ),
        sources=[{"type": "pubmed", "reference": "PMID:123", "chunk_ids": ["abc"]}],
        metadata={"generated_at": "2026-04-08", "model": "claude-sonnet-4-6", "retry_count": 0},
    )


def test_save_and_get():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = CaseStore(db_path=f"{tmpdir}/cases.db")
        case = _make_case()
        store.save(case)
        retrieved = store.get("test-1")
        assert retrieved is not None
        assert retrieved.case_id == "test-1"
        assert retrieved.topic == "SAH"
        assert retrieved.difficulty == DifficultyLevel.RESIDENT


def test_get_nonexistent():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = CaseStore(db_path=f"{tmpdir}/cases.db")
        assert store.get("nonexistent") is None


def test_list_all():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = CaseStore(db_path=f"{tmpdir}/cases.db")
        store.save(_make_case("c1", "SAH", "resident"))
        store.save(_make_case("c2", "MI", "attending"))
        store.save(_make_case("c3", "SAH", "medical_student"))
        cases = store.list_cases()
        assert len(cases) == 3


def test_list_filter_topic():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = CaseStore(db_path=f"{tmpdir}/cases.db")
        store.save(_make_case("c1", "SAH", "resident"))
        store.save(_make_case("c2", "MI", "attending"))
        cases = store.list_cases(topic="SAH")
        assert len(cases) == 1
        assert cases[0].topic == "SAH"


def test_list_filter_difficulty():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = CaseStore(db_path=f"{tmpdir}/cases.db")
        store.save(_make_case("c1", "SAH", "resident"))
        store.save(_make_case("c2", "SAH", "attending"))
        cases = store.list_cases(difficulty="attending")
        assert len(cases) == 1
        assert cases[0].difficulty == DifficultyLevel.ATTENDING


def test_list_filter_min_accuracy():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = CaseStore(db_path=f"{tmpdir}/cases.db")
        store.save(_make_case("c1"))
        low_acc = _make_case("c2")
        low_acc = low_acc.model_copy(
            update={"review": low_acc.review.model_copy(update={"accuracy_score": 0.5})}
        )
        store.save(low_acc)
        cases = store.list_cases(min_accuracy=0.8)
        assert len(cases) == 1
        assert cases[0].case_id == "c1"


def test_count():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = CaseStore(db_path=f"{tmpdir}/cases.db")
        assert store.count() == 0
        store.save(_make_case())
        assert store.count() == 1


def test_export_jsonl():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = CaseStore(db_path=f"{tmpdir}/cases.db")
        store.save(_make_case("c1", "SAH"))
        store.save(_make_case("c2", "MI"))
        lines = store.export_jsonl()
        assert len(lines) == 2
        import json
        parsed = json.loads(lines[0])
        assert "case_id" in parsed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_case_store.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement case store**

```python
# src/casecrawler/storage/__init__.py
```

```python
# src/casecrawler/storage/case_store.py
from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from casecrawler.models.case import GeneratedCase


class CaseStore:
    def __init__(self, db_path: str = "./data/cases.db") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                specialty TEXT NOT NULL,
                accuracy_score REAL,
                pedagogy_score REAL,
                bias_score REAL,
                model TEXT,
                generated_at TIMESTAMP,
                case_json TEXT NOT NULL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_topic ON cases(topic)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_difficulty ON cases(difficulty)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_specialty ON cases(specialty)")
        self._conn.commit()

    def save(self, case: GeneratedCase) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO cases
            (case_id, topic, difficulty, specialty, accuracy_score, pedagogy_score, bias_score, model, generated_at, case_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                case.case_id,
                case.topic,
                case.difficulty.value,
                ",".join(case.specialty),
                case.review.accuracy_score,
                case.review.pedagogy_score,
                case.review.bias_score,
                case.metadata.get("model", ""),
                case.metadata.get("generated_at", datetime.now().isoformat()),
                case.model_dump_json(),
            ),
        )
        self._conn.commit()

    def get(self, case_id: str) -> GeneratedCase | None:
        row = self._conn.execute(
            "SELECT case_json FROM cases WHERE case_id = ?", (case_id,)
        ).fetchone()
        if row is None:
            return None
        return GeneratedCase.model_validate_json(row["case_json"])

    def list_cases(
        self,
        topic: str | None = None,
        difficulty: str | None = None,
        min_accuracy: float | None = None,
        limit: int = 100,
    ) -> list[GeneratedCase]:
        query = "SELECT case_json FROM cases WHERE 1=1"
        params: list = []

        if topic:
            query += " AND topic = ?"
            params.append(topic)
        if difficulty:
            query += " AND difficulty = ?"
            params.append(difficulty)
        if min_accuracy is not None:
            query += " AND accuracy_score >= ?"
            params.append(min_accuracy)

        query += " ORDER BY generated_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [GeneratedCase.model_validate_json(row["case_json"]) for row in rows]

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM cases").fetchone()
        return row["cnt"]

    def export_jsonl(
        self,
        topic: str | None = None,
        difficulty: str | None = None,
    ) -> list[str]:
        """Export cases as JSONL lines."""
        cases = self.list_cases(topic=topic, difficulty=difficulty, limit=10000)
        return [case.model_dump_json() for case in cases]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_case_store.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/casecrawler/storage/ tests/test_case_store.py
git commit -m "feat: SQLite case store with filtering and JSONL export"
```

---

## Task 6: Retriever + Generation Prompts

**Files:**
- Create: `src/casecrawler/generation/__init__.py`
- Create: `src/casecrawler/generation/retriever.py`
- Create: `src/casecrawler/generation/prompts.py`
- Test: `tests/test_retriever.py`

- [ ] **Step 1: Write retriever tests**

```python
# tests/test_retriever.py
import tempfile

from casecrawler.generation.retriever import Retriever
from casecrawler.models.document import Chunk, CredibilityLevel, DocumentMetadata
from casecrawler.pipeline.embedder import Embedder
from casecrawler.pipeline.store import Store


def _store_test_chunks(store: Store, embedder: Embedder) -> None:
    chunks = [
        Chunk(
            chunk_id="c1", source_document_id="pubmed:1", text="SAH is a neurosurgical emergency requiring immediate CT.",
            position=0, metadata=DocumentMetadata(credibility=CredibilityLevel.GUIDELINE, specialty=["neurosurgery"]),
        ),
        Chunk(
            chunk_id="c2", source_document_id="pubmed:2", text="Thunderclap headache is the hallmark of SAH.",
            position=0, metadata=DocumentMetadata(credibility=CredibilityLevel.PEER_REVIEWED),
        ),
        Chunk(
            chunk_id="c3", source_document_id="medrxiv:3", text="Novel biomarkers for SAH prognosis.",
            position=0, metadata=DocumentMetadata(credibility=CredibilityLevel.PREPRINT),
        ),
    ]
    embedded = embedder.embed(chunks)
    store.store(embedded)


def test_retriever_returns_chunks():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = Store(chroma_dir=tmpdir)
        embedder = Embedder()
        _store_test_chunks(store, embedder)
        retriever = Retriever(store=store)
        results = retriever.retrieve("subarachnoid hemorrhage", limit=10)
        assert len(results) >= 1
        assert "text" in results[0]
        assert "credibility" in results[0]
        assert "chunk_id" in results[0]


def test_retriever_orders_by_credibility():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = Store(chroma_dir=tmpdir)
        embedder = Embedder()
        _store_test_chunks(store, embedder)
        retriever = Retriever(store=store)
        results = retriever.retrieve("SAH", limit=10)
        # Guidelines should come before preprints
        credibilities = [r["credibility"] for r in results]
        if "guideline" in credibilities and "preprint" in credibilities:
            assert credibilities.index("guideline") < credibilities.index("preprint")


def test_retriever_includes_source_info():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = Store(chroma_dir=tmpdir)
        embedder = Embedder()
        _store_test_chunks(store, embedder)
        retriever = Retriever(store=store)
        results = retriever.retrieve("SAH", limit=10)
        for r in results:
            assert "source_document_id" in r
            assert "chunk_id" in r
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_retriever.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement retriever**

```python
# src/casecrawler/generation/__init__.py
```

```python
# src/casecrawler/generation/retriever.py
from __future__ import annotations

from casecrawler.pipeline.store import Store

CREDIBILITY_ORDER = {
    "guideline": 0,
    "fda_label": 1,
    "peer_reviewed": 2,
    "curated": 3,
    "preprint": 4,
}


class Retriever:
    def __init__(self, store: Store) -> None:
        self._store = store

    def retrieve(self, topic: str, limit: int = 25) -> list[dict]:
        """Query ChromaDB and return chunks ranked by relevance then credibility."""
        results = self._store.search(topic, n_results=limit)

        # Enrich and sort: credibility first, then relevance score
        enriched = []
        for r in results:
            credibility = r["metadata"].get("credibility", "preprint")
            enriched.append({
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "score": r["score"],
                "credibility": credibility,
                "credibility_rank": CREDIBILITY_ORDER.get(credibility, 99),
                "source_document_id": r["metadata"].get("source_document_id", ""),
                "source": r["metadata"].get("source", ""),
                "specialty": r["metadata"].get("specialty", ""),
                "doi": r["metadata"].get("doi", ""),
                "url": r["metadata"].get("url", ""),
            })

        enriched.sort(key=lambda x: (x["credibility_rank"], -x["score"]))
        return enriched

    def format_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into a context string for LLM prompts."""
        sections = []
        for i, chunk in enumerate(chunks, 1):
            sections.append(
                f"[Source {i}] ({chunk['credibility']}, {chunk['source']})\n{chunk['text']}"
            )
        return "\n\n---\n\n".join(sections)
```

- [ ] **Step 4: Implement prompts**

```python
# src/casecrawler/generation/prompts.py
from __future__ import annotations

DIFFICULTY_RULES = {
    "medical_student": {
        "vignette": "Include most key findings with fewer distractors. Classic textbook presentation.",
        "decision_tree": "Provide 2-3 choices. One clearly correct, one common mistake, one possible catastrophic error.",
        "complications": "Simple cause-effect relationships.",
        "knowledge": "Foundational pathophysiology. Basic diagnostic and treatment knowledge.",
    },
    "resident": {
        "vignette": "Some findings missing, moderate distractors. Atypical or overlapping presentations.",
        "decision_tree": "Provide 3-4 choices with nuanced distinctions. Include management algorithm decisions.",
        "complications": "Multi-step cascades where one error leads to another.",
        "knowledge": "Management algorithms, workup prioritization, time-sensitive decisions.",
    },
    "attending": {
        "vignette": "Incomplete data, many distractors, red herrings. Rare variants or multiple concurrent pathologies.",
        "decision_tree": "Provide 4-5 choices with subtle distinctions between reasonable options.",
        "complications": "System-level failures (e.g., missed consult leads to delayed OR leads to herniation).",
        "knowledge": "Nuanced judgment calls, resource constraints, ambiguity tolerance.",
    },
}

CASE_GENERATOR_SYSTEM = """You are a clinical case author creating realistic, decision-forcing medical scenarios.

Your cases must:
- Be messy and incomplete, like real medicine
- Force the clinician to make decisions with incomplete information
- Include distractors that could lead to wrong diagnoses
- Be grounded in real medical knowledge from the provided sources
- Have diverse patient demographics (vary age, sex, background)
- Avoid demographic stereotypes (do not default to stereotypical presentations)

You will receive:
1. A medical topic
2. A difficulty level with specific rules
3. Retrieved medical knowledge from real sources

Generate a realistic clinical vignette with patient demographics and ground truth."""

DECISION_TREE_SYSTEM = """You are a clinical decision tree architect.

Given a clinical vignette and its ground truth, build a decision tree that:
- Has exactly ONE correct path
- Has plausible wrong paths that a clinician might actually choose
- Labels each wrong path as "common_mistake" or "catastrophic"
- Provides realistic consequences for each wrong choice
- Includes a complications layer showing what happens with delayed or incorrect decisions

Each choice must include:
- The action the clinician would take
- Whether it is correct
- Clinical reasoning for why someone might choose it
- The realistic outcome of that choice
- For wrong choices: the consequence and what error type it represents

Also generate complications that show temporal consequences:
- What happens if diagnosis is delayed
- What happens if incorrect treatment is given"""

CLINICAL_REVIEWER_SYSTEM = """You are a senior clinical reviewer evaluating AI-generated medical cases for quality.

Score each case on three dimensions (0.0 to 1.0):

**Accuracy (0.0-1.0):**
- Is the diagnosis correct for the presented findings?
- Are the treatment options real and appropriate?
- Are lab values, vital signs, and timelines physiologically possible?
- Do the decision tree outcomes match known clinical evidence?

**Pedagogy (0.0-1.0):**
- Is the case appropriately challenging for the stated difficulty level?
- Are distractors plausible but distinguishable with proper knowledge?
- Does the decision tree cover the most important learning points?
- Would a learner gain meaningful clinical reasoning from this case?

**Bias (0.0-1.0):**
- Does the case avoid demographic stereotyping?
- Is the patient presentation free from cultural assumptions?
- Would the case work equally well with a different patient demographic?
- Are gendered language patterns avoided?

If ANY score is below the threshold, you MUST reject and provide specific, actionable feedback.
Your feedback should tell the generator exactly what to fix."""


def build_case_generator_prompt(topic: str, difficulty: str, context: str) -> str:
    rules = DIFFICULTY_RULES.get(difficulty, DIFFICULTY_RULES["resident"])
    return f"""Generate a clinical case for the following topic.

## Topic
{topic}

## Difficulty Level: {difficulty}
- Vignette: {rules['vignette']}
- Knowledge level: {rules['knowledge']}

## Medical Knowledge (from real sources)
{context}

## Instructions
Create a realistic clinical vignette with:
1. Patient demographics (age, sex, relevant background)
2. The clinical presentation (history, exam findings, initial labs if relevant)
3. A decision prompt ("What would you do next?")
4. Ground truth: the correct diagnosis, optimal next step, rationale, and key findings

Make the vignette realistic and messy — like a real patient encounter, not a textbook question."""


def build_decision_tree_prompt(vignette: str, ground_truth_json: str, difficulty: str, context: str) -> str:
    rules = DIFFICULTY_RULES.get(difficulty, DIFFICULTY_RULES["resident"])
    return f"""Build a decision tree for this clinical case.

## Vignette
{vignette}

## Ground Truth
{ground_truth_json}

## Difficulty Level: {difficulty}
- Decision tree: {rules['decision_tree']}
- Complications: {rules['complications']}

## Medical Knowledge (from real sources)
{context}

## Instructions
Create:
1. Decision choices — one correct path and plausible wrong paths
2. Complications — what happens with delayed diagnosis or incorrect treatment

Each wrong choice must have an error_type of "common_mistake" or "catastrophic"."""


def build_reviewer_prompt(case_json: str, context: str, threshold: float) -> str:
    return f"""Review this AI-generated clinical case for quality.

## Generated Case
{case_json}

## Source Material (what the case was built from)
{context}

## Scoring Threshold
All scores must be >= {threshold} for approval.

## Instructions
Score accuracy, pedagogy, and bias (0.0-1.0 each).
Set approved=true only if ALL scores meet the threshold.
If rejecting, provide specific, actionable notes explaining what to fix."""


def build_retry_prompt(original_prompt: str, reviewer_notes: list[str]) -> str:
    notes_text = "\n".join(f"- {note}" for note in reviewer_notes)
    return f"""{original_prompt}

## IMPORTANT: Previous Attempt Was Rejected
The clinical reviewer provided this feedback. You MUST address each issue:

{notes_text}

Fix these specific issues while preserving what was already correct."""
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_retriever.py -v`
Expected: All 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/casecrawler/generation/ tests/test_retriever.py
git commit -m "feat: retriever with credibility ranking and generation prompts"
```

---

## Task 7: Case Generator + Decision Tree Builder + Clinical Reviewer

**Files:**
- Create: `src/casecrawler/generation/case_generator.py`
- Create: `src/casecrawler/generation/decision_tree_builder.py`
- Create: `src/casecrawler/generation/clinical_reviewer.py`
- Test: `tests/test_generation_agents.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_generation_agents.py
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from casecrawler.generation.case_generator import CaseGeneratorAgent
from casecrawler.generation.clinical_reviewer import ClinicalReviewerAgent
from casecrawler.generation.decision_tree_builder import DecisionTreeBuilderAgent
from casecrawler.llm.base import StructuredGenerationResult
from casecrawler.models.case import (
    Complication,
    DecisionChoice,
    DifficultyLevel,
    GroundTruth,
    Patient,
    ReviewResult,
)


@pytest.fixture
def mock_provider():
    return AsyncMock()


# --- Case Generator ---


@pytest.mark.asyncio
async def test_case_generator(mock_provider):
    mock_provider.generate_structured.return_value = StructuredGenerationResult(
        data=MagicMock(
            patient=Patient(age=42, sex="female", demographics="Healthy"),
            vignette="A 42-year-old woman...",
            decision_prompt="What would you do next?",
            ground_truth=GroundTruth(
                diagnosis="SAH", optimal_next_step="CT head",
                rationale="Most sensitive", key_findings=["thunderclap headache"],
            ),
            specialty=["neurosurgery"],
        ),
        input_tokens=500,
        output_tokens=300,
        model="test-model",
    )

    agent = CaseGeneratorAgent(provider=mock_provider)
    result = await agent.generate(
        topic="subarachnoid hemorrhage",
        difficulty="resident",
        context="SAH requires immediate CT...",
    )
    assert result.data.vignette == "A 42-year-old woman..."
    assert mock_provider.generate_structured.called


# --- Decision Tree Builder ---


@pytest.mark.asyncio
async def test_decision_tree_builder(mock_provider):
    mock_provider.generate_structured.return_value = StructuredGenerationResult(
        data=MagicMock(
            decision_tree=[
                DecisionChoice(
                    choice="CT head", is_correct=True, error_type=None,
                    reasoning="Correct", outcome="SAH confirmed",
                    consequence=None, next_decision=None,
                ),
                DecisionChoice(
                    choice="MRI", is_correct=False, error_type="common_mistake",
                    reasoning="Takes too long", outcome="Delay",
                    consequence="Risk of rebleed", next_decision=None,
                ),
            ],
            complications=[
                Complication(trigger="delayed_diagnosis", detail="6h", event="Rebleed", outcome="Death"),
            ],
        ),
        input_tokens=600,
        output_tokens=400,
        model="test-model",
    )

    agent = DecisionTreeBuilderAgent(provider=mock_provider)
    result = await agent.build(
        vignette="A 42-year-old woman...",
        ground_truth_json='{"diagnosis": "SAH"}',
        difficulty="resident",
        context="SAH management guidelines...",
    )
    assert len(result.data.decision_tree) == 2
    assert result.data.decision_tree[0].is_correct is True


# --- Clinical Reviewer ---


@pytest.mark.asyncio
async def test_clinical_reviewer_approves(mock_provider):
    mock_provider.generate_structured.return_value = StructuredGenerationResult(
        data=ReviewResult(
            accuracy_score=0.95, pedagogy_score=0.9, bias_score=0.92,
            approved=True, notes=[],
        ),
        input_tokens=800,
        output_tokens=100,
        model="test-model",
    )

    agent = ClinicalReviewerAgent(provider=mock_provider, threshold=0.7)
    result = await agent.review(case_json="{}", context="source material")
    assert result.data.approved is True


@pytest.mark.asyncio
async def test_clinical_reviewer_rejects(mock_provider):
    mock_provider.generate_structured.return_value = StructuredGenerationResult(
        data=ReviewResult(
            accuracy_score=0.5, pedagogy_score=0.9, bias_score=0.92,
            approved=False, notes=["Diagnosis is incorrect"],
        ),
        input_tokens=800,
        output_tokens=100,
        model="test-model",
    )

    agent = ClinicalReviewerAgent(provider=mock_provider, threshold=0.7)
    result = await agent.review(case_json="{}", context="source material")
    assert result.data.approved is False
    assert "Diagnosis is incorrect" in result.data.notes
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_generation_agents.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement case generator agent**

```python
# src/casecrawler/generation/case_generator.py
from __future__ import annotations

from pydantic import BaseModel

from casecrawler.generation.prompts import CASE_GENERATOR_SYSTEM, build_case_generator_prompt
from casecrawler.llm.base import BaseLLMProvider, StructuredGenerationResult
from casecrawler.models.case import GroundTruth, Patient


class CaseGeneratorOutput(BaseModel):
    patient: Patient
    vignette: str
    decision_prompt: str
    ground_truth: GroundTruth
    specialty: list[str]


class CaseGeneratorAgent:
    def __init__(self, provider: BaseLLMProvider) -> None:
        self._provider = provider

    async def generate(
        self,
        topic: str,
        difficulty: str,
        context: str,
        retry_notes: list[str] | None = None,
    ) -> StructuredGenerationResult:
        prompt = build_case_generator_prompt(topic, difficulty, context)

        if retry_notes:
            from casecrawler.generation.prompts import build_retry_prompt
            prompt = build_retry_prompt(prompt, retry_notes)

        return await self._provider.generate_structured(
            prompt=prompt,
            schema=CaseGeneratorOutput,
            system=CASE_GENERATOR_SYSTEM,
        )
```

- [ ] **Step 4: Implement decision tree builder agent**

```python
# src/casecrawler/generation/decision_tree_builder.py
from __future__ import annotations

from pydantic import BaseModel

from casecrawler.generation.prompts import DECISION_TREE_SYSTEM, build_decision_tree_prompt
from casecrawler.llm.base import BaseLLMProvider, StructuredGenerationResult
from casecrawler.models.case import Complication, DecisionChoice


class DecisionTreeOutput(BaseModel):
    decision_tree: list[DecisionChoice]
    complications: list[Complication]


class DecisionTreeBuilderAgent:
    def __init__(self, provider: BaseLLMProvider) -> None:
        self._provider = provider

    async def build(
        self,
        vignette: str,
        ground_truth_json: str,
        difficulty: str,
        context: str,
        retry_notes: list[str] | None = None,
    ) -> StructuredGenerationResult:
        prompt = build_decision_tree_prompt(vignette, ground_truth_json, difficulty, context)

        if retry_notes:
            from casecrawler.generation.prompts import build_retry_prompt
            prompt = build_retry_prompt(prompt, retry_notes)

        return await self._provider.generate_structured(
            prompt=prompt,
            schema=DecisionTreeOutput,
            system=DECISION_TREE_SYSTEM,
        )
```

- [ ] **Step 5: Implement clinical reviewer agent**

```python
# src/casecrawler/generation/clinical_reviewer.py
from __future__ import annotations

from casecrawler.generation.prompts import CLINICAL_REVIEWER_SYSTEM, build_reviewer_prompt
from casecrawler.llm.base import BaseLLMProvider, StructuredGenerationResult
from casecrawler.models.case import ReviewResult


class ClinicalReviewerAgent:
    def __init__(self, provider: BaseLLMProvider, threshold: float = 0.7) -> None:
        self._provider = provider
        self._threshold = threshold

    async def review(self, case_json: str, context: str) -> StructuredGenerationResult:
        prompt = build_reviewer_prompt(case_json, context, self._threshold)
        return await self._provider.generate_structured(
            prompt=prompt,
            schema=ReviewResult,
            system=CLINICAL_REVIEWER_SYSTEM,
        )
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_generation_agents.py -v`
Expected: All 4 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/casecrawler/generation/case_generator.py src/casecrawler/generation/decision_tree_builder.py src/casecrawler/generation/clinical_reviewer.py tests/test_generation_agents.py
git commit -m "feat: case generator, decision tree builder, and clinical reviewer agents"
```

---

## Task 8: Generation Pipeline Orchestrator

**Files:**
- Create: `src/casecrawler/generation/pipeline.py`
- Test: `tests/test_generation_pipeline.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_generation_pipeline.py
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from casecrawler.generation.pipeline import GenerationPipeline
from casecrawler.llm.base import StructuredGenerationResult
from casecrawler.models.case import (
    Complication,
    DecisionChoice,
    DifficultyLevel,
    GeneratedCase,
    GroundTruth,
    Patient,
    ReviewResult,
)


def _mock_case_gen_result():
    return StructuredGenerationResult(
        data=MagicMock(
            patient=Patient(age=42, sex="female", demographics="Healthy"),
            vignette="A 42-year-old woman presents with thunderclap headache.",
            decision_prompt="What would you do next?",
            ground_truth=GroundTruth(
                diagnosis="SAH", optimal_next_step="CT head",
                rationale="Most sensitive", key_findings=["thunderclap headache"],
            ),
            specialty=["neurosurgery"],
        ),
        input_tokens=500, output_tokens=300, model="test-model",
    )


def _mock_tree_result():
    return StructuredGenerationResult(
        data=MagicMock(
            decision_tree=[
                DecisionChoice(
                    choice="CT head", is_correct=True, error_type=None,
                    reasoning="Correct", outcome="Confirmed",
                    consequence=None, next_decision=None,
                ),
            ],
            complications=[
                Complication(trigger="delayed_diagnosis", detail="6h", event="Rebleed", outcome="Death"),
            ],
        ),
        input_tokens=600, output_tokens=400, model="test-model",
    )


def _mock_review_approved():
    return StructuredGenerationResult(
        data=ReviewResult(
            accuracy_score=0.95, pedagogy_score=0.9, bias_score=0.92,
            approved=True, notes=[],
        ),
        input_tokens=800, output_tokens=100, model="test-model",
    )


def _mock_review_rejected():
    return StructuredGenerationResult(
        data=ReviewResult(
            accuracy_score=0.5, pedagogy_score=0.9, bias_score=0.92,
            approved=False, notes=["Diagnosis incorrect"],
        ),
        input_tokens=800, output_tokens=100, model="test-model",
    )


@pytest.fixture
def mock_retriever():
    r = MagicMock()
    r.retrieve.return_value = [
        {"chunk_id": "c1", "text": "SAH content", "credibility": "guideline",
         "credibility_rank": 0, "source_document_id": "pubmed:1", "source": "pubmed",
         "score": 0.9, "specialty": "", "doi": "", "url": ""},
    ]
    r.format_context.return_value = "[Source 1] (guideline, pubmed)\nSAH content"
    return r


@pytest.fixture
def mock_provider():
    return AsyncMock()


@pytest.mark.asyncio
async def test_pipeline_generates_case(mock_retriever, mock_provider):
    mock_provider.generate_structured.side_effect = [
        _mock_case_gen_result(),
        _mock_tree_result(),
        _mock_review_approved(),
    ]

    pipeline = GenerationPipeline(
        provider=mock_provider,
        retriever=mock_retriever,
        max_retries=3,
        review_threshold=0.7,
    )

    result = await pipeline.generate_one(topic="SAH", difficulty="resident")
    assert result is not None
    assert isinstance(result, GeneratedCase)
    assert result.topic == "SAH"
    assert result.review.approved is True
    assert mock_provider.generate_structured.call_count == 3


@pytest.mark.asyncio
async def test_pipeline_retries_on_rejection(mock_retriever, mock_provider):
    mock_provider.generate_structured.side_effect = [
        _mock_case_gen_result(),
        _mock_tree_result(),
        _mock_review_rejected(),    # first review rejects
        _mock_case_gen_result(),     # retry case gen
        _mock_tree_result(),         # retry tree
        _mock_review_approved(),     # second review approves
    ]

    pipeline = GenerationPipeline(
        provider=mock_provider,
        retriever=mock_retriever,
        max_retries=3,
        review_threshold=0.7,
    )

    result = await pipeline.generate_one(topic="SAH", difficulty="resident")
    assert result is not None
    assert result.review.approved is True
    assert result.metadata["retry_count"] == 1


@pytest.mark.asyncio
async def test_pipeline_returns_none_after_max_retries(mock_retriever, mock_provider):
    mock_provider.generate_structured.side_effect = [
        _mock_case_gen_result(), _mock_tree_result(), _mock_review_rejected(),
        _mock_case_gen_result(), _mock_tree_result(), _mock_review_rejected(),
        _mock_case_gen_result(), _mock_tree_result(), _mock_review_rejected(),
    ]

    pipeline = GenerationPipeline(
        provider=mock_provider,
        retriever=mock_retriever,
        max_retries=3,
        review_threshold=0.7,
    )

    result = await pipeline.generate_one(topic="SAH", difficulty="resident")
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_generation_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement generation pipeline**

```python
# src/casecrawler/generation/pipeline.py
from __future__ import annotations

import uuid
from datetime import datetime

from casecrawler.generation.case_generator import CaseGeneratorAgent
from casecrawler.generation.clinical_reviewer import ClinicalReviewerAgent
from casecrawler.generation.decision_tree_builder import DecisionTreeBuilderAgent
from casecrawler.generation.retriever import Retriever
from casecrawler.llm.base import BaseLLMProvider
from casecrawler.models.case import DifficultyLevel, GeneratedCase


class GenerationPipeline:
    def __init__(
        self,
        provider: BaseLLMProvider,
        retriever: Retriever,
        max_retries: int = 3,
        review_threshold: float = 0.7,
    ) -> None:
        self._case_gen = CaseGeneratorAgent(provider=provider)
        self._tree_builder = DecisionTreeBuilderAgent(provider=provider)
        self._reviewer = ClinicalReviewerAgent(provider=provider, threshold=review_threshold)
        self._retriever = retriever
        self._max_retries = max_retries
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    async def generate_one(
        self, topic: str, difficulty: str = "resident",
    ) -> GeneratedCase | None:
        """Generate a single case with retry logic. Returns None if all retries fail."""
        # Stage 1: Retrieve
        chunks = self._retriever.retrieve(topic)
        context = self._retriever.format_context(chunks)
        source_refs = [
            {"type": c["source"], "reference": c["source_document_id"], "chunk_ids": [c["chunk_id"]]}
            for c in chunks
        ]

        retry_notes: list[str] | None = None
        for attempt in range(self._max_retries):
            # Stage 2: Generate case
            gen_result = await self._case_gen.generate(
                topic=topic, difficulty=difficulty, context=context, retry_notes=retry_notes,
            )
            self._total_input_tokens += gen_result.input_tokens
            self._total_output_tokens += gen_result.output_tokens
            gen_data = gen_result.data

            # Stage 3: Build decision tree
            gt_json = gen_data.ground_truth.model_dump_json()
            tree_result = await self._tree_builder.build(
                vignette=gen_data.vignette,
                ground_truth_json=gt_json,
                difficulty=difficulty,
                context=context,
                retry_notes=retry_notes,
            )
            self._total_input_tokens += tree_result.input_tokens
            self._total_output_tokens += tree_result.output_tokens
            tree_data = tree_result.data

            # Assemble case for review
            case = GeneratedCase(
                case_id=str(uuid.uuid4()),
                topic=topic,
                difficulty=DifficultyLevel(difficulty),
                specialty=gen_data.specialty,
                patient=gen_data.patient,
                vignette=gen_data.vignette,
                decision_prompt=gen_data.decision_prompt,
                ground_truth=gen_data.ground_truth,
                decision_tree=tree_data.decision_tree,
                complications=tree_data.complications,
                review=None,  # placeholder
                sources=source_refs,
                metadata={
                    "generated_at": datetime.now().isoformat(),
                    "model": gen_result.model,
                    "retry_count": attempt,
                    "total_input_tokens": self._total_input_tokens,
                    "total_output_tokens": self._total_output_tokens,
                },
            )

            # Stage 4: Clinical review
            review_result = await self._reviewer.review(
                case_json=case.model_dump_json(),
                context=context,
            )
            self._total_input_tokens += review_result.input_tokens
            self._total_output_tokens += review_result.output_tokens

            case = case.model_copy(update={
                "review": review_result.data,
                "metadata": {
                    **case.metadata,
                    "total_input_tokens": self._total_input_tokens,
                    "total_output_tokens": self._total_output_tokens,
                },
            })

            if review_result.data.approved:
                return case

            # Rejected — set retry notes for next attempt
            retry_notes = review_result.data.notes

        # All retries exhausted
        return None

    async def generate_batch(
        self, topic: str, count: int, difficulty: str = "resident",
    ) -> dict:
        """Generate multiple cases sequentially. Returns summary."""
        generated = []
        failed = 0

        for _ in range(count):
            case = await self.generate_one(topic=topic, difficulty=difficulty)
            if case:
                generated.append(case)
            else:
                failed += 1

        return {
            "cases": generated,
            "generated": len(generated),
            "failed": failed,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
        }
```

- [ ] **Step 4: Fix the GeneratedCase model to allow None review temporarily**

The pipeline assembles a case with `review=None` before the reviewer runs. Update `src/casecrawler/models/case.py` — change the `review` field:

```python
class GeneratedCase(BaseModel):
    # ... all fields same as before ...
    review: ReviewResult | None = None  # None during assembly, set after review
    # ... rest same ...
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_generation_pipeline.py -v`
Expected: All 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/casecrawler/generation/pipeline.py src/casecrawler/models/case.py tests/test_generation_pipeline.py
git commit -m "feat: generation pipeline with 4-stage orchestration and retry logic"
```

---

## Task 9: CLI Extensions

**Files:**
- Modify: `src/casecrawler/cli.py`
- Test: `tests/test_cli_generate.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_cli_generate.py
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from casecrawler.cli import cli
from casecrawler.models.case import (
    Complication,
    DecisionChoice,
    DifficultyLevel,
    GeneratedCase,
    GroundTruth,
    Patient,
    ReviewResult,
)


def _fake_case() -> GeneratedCase:
    return GeneratedCase(
        case_id="test-1",
        topic="SAH",
        difficulty=DifficultyLevel.RESIDENT,
        specialty=["neurosurgery"],
        patient=Patient(age=42, sex="female", demographics="Healthy"),
        vignette="A 42-year-old woman...",
        decision_prompt="What would you do next?",
        ground_truth=GroundTruth(
            diagnosis="SAH", optimal_next_step="CT",
            rationale="Sensitive", key_findings=["headache"],
        ),
        decision_tree=[
            DecisionChoice(
                choice="CT", is_correct=True, error_type=None,
                reasoning="Right", outcome="Confirmed",
                consequence=None, next_decision=None,
            ),
        ],
        complications=[Complication(trigger="delay", detail="6h", event="Rebleed", outcome="Death")],
        review=ReviewResult(accuracy_score=0.95, pedagogy_score=0.9, bias_score=0.92, approved=True, notes=[]),
        sources=[],
        metadata={"generated_at": "2026-04-08", "model": "test", "retry_count": 0},
    )


def test_generate_command():
    runner = CliRunner()
    with patch("casecrawler.cli.get_provider") as mock_get_provider, \
         patch("casecrawler.cli.GenerationPipeline") as MockPipeline, \
         patch("casecrawler.cli.Retriever"), \
         patch("casecrawler.cli.Store"), \
         patch("casecrawler.cli.CaseStore") as MockCaseStore:
        mock_pipeline = MockPipeline.return_value
        mock_pipeline.generate_batch = AsyncMock(return_value={
            "cases": [_fake_case()],
            "generated": 1,
            "failed": 0,
            "total_input_tokens": 1000,
            "total_output_tokens": 500,
        })
        mock_case_store = MockCaseStore.return_value
        mock_case_store.save = MagicMock()

        result = runner.invoke(cli, ["generate", "SAH", "--count", "1"])
        assert result.exit_code == 0
        assert "1" in result.output


def test_cases_command():
    runner = CliRunner()
    with patch("casecrawler.cli.CaseStore") as MockCaseStore:
        mock_store = MockCaseStore.return_value
        mock_store.list_cases.return_value = [_fake_case()]

        result = runner.invoke(cli, ["cases"])
        assert result.exit_code == 0
        assert "SAH" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli_generate.py -v`
Expected: FAIL

- [ ] **Step 3: Add generate and cases commands to CLI**

Add these imports near the top of `src/casecrawler/cli.py`:

```python
from casecrawler.generation.pipeline import GenerationPipeline
from casecrawler.generation.retriever import Retriever
from casecrawler.llm.factory import get_provider
from casecrawler.pipeline.store import Store
from casecrawler.storage.case_store import CaseStore
```

Add these commands after the existing `serve` command:

```python
@cli.command()
@click.argument("topic")
@click.option("--difficulty", default=None, help="medical_student, resident, or attending")
@click.option("--count", default=1, type=int, help="Number of cases to generate")
@click.option("--ingest", "ingest_first", is_flag=True, help="Ingest topic first")
@click.option("--output", default=None, help="Output JSONL file path")
def generate(topic: str, difficulty: str | None, count: int, ingest_first: bool, output: str | None) -> None:
    """Generate clinical cases for a medical topic."""
    config = get_config()
    difficulty = difficulty or config.generation.default_difficulty

    if ingest_first:
        click.echo(f"Ingesting '{topic}' first...")
        # Reuse existing ingest logic
        from casecrawler.sources.registry import SourceRegistry
        registry = SourceRegistry()
        registry.discover()
        active_sources = registry.get_sources()
        if active_sources:
            all_docs = asyncio.run(_search_all(active_sources, topic, config.ingestion.default_limit_per_source))
            pipeline_orch = PipelineOrchestrator()
            for source_name, docs in all_docs.items():
                if docs:
                    pipeline_orch.process(docs)

    # Check ChromaDB has content
    store = Store(chroma_dir=config.storage.chroma_persist_dir)
    if store.count == 0:
        click.echo(f"No content found for '{topic}'. Run 'casecrawler ingest \"{topic}\"' first.")
        return

    try:
        provider = get_provider(config.llm.provider, config.llm.model, base_url=config.llm.ollama_base_url)
    except ValueError as e:
        click.echo(f"Error: {e}")
        return

    retriever = Retriever(store=store)
    gen_pipeline = GenerationPipeline(
        provider=provider,
        retriever=retriever,
        max_retries=config.generation.max_retries,
        review_threshold=config.generation.review_threshold,
    )

    click.echo(f"Generating {count} case(s) for '{topic}' at {difficulty} difficulty...")
    start = time.time()

    result = asyncio.run(gen_pipeline.generate_batch(topic=topic, count=count, difficulty=difficulty))
    elapsed = time.time() - start

    # Save to SQLite
    case_store = CaseStore()
    for case in result["cases"]:
        case_store.save(case)

    click.echo(f"\n--- Generation Summary ---")
    click.echo(f"  Generated: {result['generated']}")
    click.echo(f"  Failed: {result['failed']}")
    click.echo(f"  Tokens: {result['total_input_tokens']} in / {result['total_output_tokens']} out")
    click.echo(f"  Time: {elapsed:.1f}s")

    if output and result["cases"]:
        with open(output, "w") as f:
            for case in result["cases"]:
                f.write(case.model_dump_json() + "\n")
        click.echo(f"  Exported to: {output}")


@cli.group()
def cases() -> None:
    """Manage generated cases."""
    pass


@cases.command("list")
@click.option("--topic", default=None, help="Filter by topic")
@click.option("--difficulty", default=None, help="Filter by difficulty")
@click.option("--limit", default=20, type=int, help="Max results")
def cases_list(topic: str | None, difficulty: str | None, limit: int) -> None:
    """List generated cases."""
    case_store = CaseStore()
    results = case_store.list_cases(topic=topic, difficulty=difficulty, limit=limit)

    if not results:
        click.echo("No cases found.")
        return

    click.echo(f"Found {len(results)} case(s):\n")
    for case in results:
        acc = case.review.accuracy_score if case.review else 0
        click.echo(f"  [{case.case_id[:8]}] {case.topic} ({case.difficulty.value}) — accuracy: {acc:.2f}")


@cases.command("show")
@click.argument("case_id")
def cases_show(case_id: str) -> None:
    """Show a single case."""
    case_store = CaseStore()
    case = case_store.get(case_id)
    if case is None:
        click.echo(f"Case {case_id} not found.")
        return
    click.echo(case.model_dump_json(indent=2))


@cases.command("export")
@click.option("--output", required=True, help="Output JSONL file path")
@click.option("--topic", default=None, help="Filter by topic")
@click.option("--difficulty", default=None, help="Filter by difficulty")
def cases_export(output: str, topic: str | None, difficulty: str | None) -> None:
    """Export cases to JSONL."""
    case_store = CaseStore()
    lines = case_store.export_jsonl(topic=topic, difficulty=difficulty)
    with open(output, "w") as f:
        for line in lines:
            f.write(line + "\n")
    click.echo(f"Exported {len(lines)} case(s) to {output}")
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_cli_generate.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/casecrawler/cli.py tests/test_cli_generate.py
git commit -m "feat: CLI generate and cases commands"
```

---

## Task 10: API Extensions

**Files:**
- Create: `src/casecrawler/api/routes/generate.py`
- Create: `src/casecrawler/api/routes/cases.py`
- Modify: `src/casecrawler/api/app.py`
- Test: `tests/test_api_generate.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_api_generate.py
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from casecrawler.api.app import create_app
from casecrawler.models.case import (
    Complication, DecisionChoice, DifficultyLevel,
    GeneratedCase, GroundTruth, Patient, ReviewResult,
)


def _fake_case(case_id="test-1"):
    return GeneratedCase(
        case_id=case_id, topic="SAH", difficulty=DifficultyLevel.RESIDENT,
        specialty=["neurosurgery"],
        patient=Patient(age=42, sex="female", demographics="Healthy"),
        vignette="A 42-year-old woman...", decision_prompt="What next?",
        ground_truth=GroundTruth(diagnosis="SAH", optimal_next_step="CT",
                                  rationale="Sensitive", key_findings=["headache"]),
        decision_tree=[DecisionChoice(choice="CT", is_correct=True, error_type=None,
                                       reasoning="Right", outcome="Confirmed",
                                       consequence=None, next_decision=None)],
        complications=[Complication(trigger="delay", detail="6h", event="Rebleed", outcome="Death")],
        review=ReviewResult(accuracy_score=0.95, pedagogy_score=0.9, bias_score=0.92, approved=True, notes=[]),
        sources=[], metadata={"generated_at": "2026-04-08", "model": "test", "retry_count": 0},
    )


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_generate_endpoint(client):
    with patch("casecrawler.api.routes.generate.run_generation"):
        resp = client.post("/api/generate", json={"topic": "SAH"})
        assert resp.status_code == 202
        assert "job_id" in resp.json()


def test_cases_list(client):
    with patch("casecrawler.api.routes.cases.get_case_store") as mock:
        mock.return_value.list_cases.return_value = [_fake_case()]
        resp = client.get("/api/cases")
        assert resp.status_code == 200
        assert len(resp.json()["cases"]) == 1


def test_cases_detail(client):
    with patch("casecrawler.api.routes.cases.get_case_store") as mock:
        mock.return_value.get.return_value = _fake_case()
        resp = client.get("/api/cases/test-1")
        assert resp.status_code == 200
        assert resp.json()["case_id"] == "test-1"


def test_cases_not_found(client):
    with patch("casecrawler.api.routes.cases.get_case_store") as mock:
        mock.return_value.get.return_value = None
        resp = client.get("/api/cases/nonexistent")
        assert resp.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api_generate.py -v`
Expected: FAIL

- [ ] **Step 3: Implement generate route**

```python
# src/casecrawler/api/routes/generate.py
from __future__ import annotations

import asyncio
import time
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from casecrawler.config import get_config
from casecrawler.generation.pipeline import GenerationPipeline
from casecrawler.generation.retriever import Retriever
from casecrawler.llm.factory import get_provider
from casecrawler.pipeline.store import Store
from casecrawler.storage.case_store import CaseStore

router = APIRouter()

_jobs: dict[str, dict] = {}


class GenerateRequest(BaseModel):
    topic: str
    difficulty: str | None = None
    count: int = 1
    ingest_first: bool = False


@router.post("/generate", status_code=202)
async def start_generation(req: GenerateRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running"}
    background_tasks.add_task(run_generation, job_id, req.topic, req.difficulty, req.count)
    return {"job_id": job_id, "status": "running"}


@router.get("/generate/{job_id}")
async def get_generation_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **_jobs[job_id]}


async def run_generation(job_id: str, topic: str, difficulty: str | None, count: int) -> None:
    config = get_config()
    difficulty = difficulty or config.generation.default_difficulty
    start = time.time()

    try:
        provider = get_provider(config.llm.provider, config.llm.model, base_url=config.llm.ollama_base_url)
        store = Store(chroma_dir=config.storage.chroma_persist_dir)
        retriever = Retriever(store=store)
        pipeline = GenerationPipeline(
            provider=provider, retriever=retriever,
            max_retries=config.generation.max_retries,
            review_threshold=config.generation.review_threshold,
        )

        result = await pipeline.generate_batch(topic=topic, count=count, difficulty=difficulty)

        case_store = CaseStore()
        for case in result["cases"]:
            case_store.save(case)

        _jobs[job_id] = {
            "status": "completed",
            "cases_generated": result["generated"],
            "cases_failed": result["failed"],
            "elapsed_seconds": round(time.time() - start, 1),
        }
    except Exception as e:
        _jobs[job_id] = {
            "status": "failed",
            "error": str(e),
            "elapsed_seconds": round(time.time() - start, 1),
        }
```

- [ ] **Step 4: Implement cases route**

```python
# src/casecrawler/api/routes/cases.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from casecrawler.storage.case_store import CaseStore

router = APIRouter()


def get_case_store() -> CaseStore:
    return CaseStore()


@router.get("/cases")
async def list_cases(
    topic: str | None = Query(None),
    difficulty: str | None = Query(None),
    min_accuracy: float | None = Query(None),
    limit: int = Query(20, le=100),
):
    store = get_case_store()
    cases = store.list_cases(topic=topic, difficulty=difficulty, min_accuracy=min_accuracy, limit=limit)
    return {
        "cases": [case.model_dump() for case in cases],
        "total": len(cases),
    }


@router.get("/cases/export")
async def export_cases(
    topic: str | None = Query(None),
    difficulty: str | None = Query(None),
):
    store = get_case_store()
    lines = store.export_jsonl(topic=topic, difficulty=difficulty)

    def generate():
        for line in lines:
            yield line + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.get("/cases/{case_id}")
async def get_case(case_id: str):
    store = get_case_store()
    case = store.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return case.model_dump()
```

- [ ] **Step 5: Update app.py to include new routers**

Add to `src/casecrawler/api/app.py`:

Import the new routes:
```python
from casecrawler.api.routes import ingest, search, sources, generate, cases
```

Add routers in `create_app()`:
```python
    app.include_router(generate.router, prefix="/api")
    app.include_router(cases.router, prefix="/api")
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_api_generate.py -v`
Expected: All 4 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/casecrawler/api/ tests/test_api_generate.py
git commit -m "feat: REST API endpoints for case generation and case management"
```

---

## Task 11: Web UI — Generate + Cases Pages

**Files:**
- Modify: `ui/src/api/client.ts`
- Create: `ui/src/pages/GeneratePage.tsx`
- Create: `ui/src/pages/CasesPage.tsx`
- Modify: `ui/src/App.tsx`

- [ ] **Step 1: Add API functions to client.ts**

Append to `ui/src/api/client.ts`:

```typescript
export interface GenerateRequest {
  topic: string;
  difficulty?: string;
  count?: number;
  ingest_first?: boolean;
}

export interface GenerateJobResponse {
  job_id: string;
  status: string;
  cases_generated?: number;
  cases_failed?: number;
  elapsed_seconds?: number;
  error?: string;
}

export interface CaseListResponse {
  cases: GeneratedCase[];
  total: number;
}

export interface GeneratedCase {
  case_id: string;
  topic: string;
  difficulty: string;
  specialty: string[];
  patient: { age: number; sex: string; demographics: string };
  vignette: string;
  decision_prompt: string;
  ground_truth: {
    diagnosis: string;
    optimal_next_step: string;
    rationale: string;
    key_findings: string[];
  };
  decision_tree: {
    choice: string;
    is_correct: boolean;
    error_type: string | null;
    reasoning: string;
    outcome: string;
    consequence: string | null;
    next_decision: string | null;
  }[];
  complications: {
    trigger: string;
    detail: string;
    event: string;
    outcome: string;
  }[];
  review: {
    accuracy_score: number;
    pedagogy_score: number;
    bias_score: number;
    approved: boolean;
    notes: string[];
  } | null;
  sources: Record<string, unknown>[];
  metadata: Record<string, unknown>;
}

export async function startGenerate(req: GenerateRequest): Promise<GenerateJobResponse> {
  const resp = await fetch(`${BASE}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  return resp.json();
}

export async function getGenerateStatus(jobId: string): Promise<GenerateJobResponse> {
  const resp = await fetch(`${BASE}/generate/${jobId}`);
  return resp.json();
}

export async function fetchCases(params?: {
  topic?: string;
  difficulty?: string;
  limit?: number;
}): Promise<CaseListResponse> {
  const qs = new URLSearchParams();
  if (params?.topic) qs.set("topic", params.topic);
  if (params?.difficulty) qs.set("difficulty", params.difficulty);
  if (params?.limit) qs.set("limit", String(params.limit));
  const resp = await fetch(`${BASE}/cases?${qs}`);
  return resp.json();
}

export async function fetchCase(caseId: string): Promise<GeneratedCase> {
  const resp = await fetch(`${BASE}/cases/${caseId}`);
  return resp.json();
}
```

- [ ] **Step 2: Create GeneratePage**

```tsx
// ui/src/pages/GeneratePage.tsx
import { useState, useCallback } from "react";
import { startGenerate, getGenerateStatus } from "../api/client";
import type { GenerateJobResponse } from "../api/client";
import { useQuery } from "@tanstack/react-query";

export default function GeneratePage() {
  const [topic, setTopic] = useState("");
  const [difficulty, setDifficulty] = useState("resident");
  const [count, setCount] = useState(1);
  const [jobId, setJobId] = useState<string | null>(null);
  const [result, setResult] = useState<GenerateJobResponse | null>(null);

  const { data: jobStatus } = useQuery({
    queryKey: ["generate-status", jobId],
    queryFn: () => getGenerateStatus(jobId!),
    enabled: !!jobId,
    refetchInterval: (query) => query.state.data?.status === "running" ? 2000 : false,
  });

  if (jobStatus && jobStatus.status !== "running" && jobId) {
    if (!result || result.job_id !== jobStatus.job_id) {
      setResult(jobStatus);
      setJobId(null);
    }
  }

  const handleGenerate = async () => {
    if (!topic.trim()) return;
    setResult(null);
    const resp = await startGenerate({ topic: topic.trim(), difficulty, count });
    setJobId(resp.job_id);
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Generate Cases</h1>

      <div className="space-y-4">
        <input
          type="text" value={topic} onChange={(e) => setTopic(e.target.value)}
          placeholder="e.g. subarachnoid hemorrhage"
          className="w-full rounded-lg border border-gray-300 px-4 py-2"
          onKeyDown={(e) => e.key === "Enter" && handleGenerate()}
        />
        <div className="flex gap-4">
          <select value={difficulty} onChange={(e) => setDifficulty(e.target.value)}
            className="rounded-lg border border-gray-300 px-3 py-2">
            <option value="medical_student">Medical Student</option>
            <option value="resident">Resident</option>
            <option value="attending">Attending</option>
          </select>
          <input type="number" value={count} onChange={(e) => setCount(Number(e.target.value))}
            min={1} max={100} className="w-24 rounded-lg border border-gray-300 px-3 py-2" />
          <button onClick={handleGenerate} disabled={!topic.trim() || !!jobId}
            className="rounded-lg bg-blue-600 px-6 py-2 text-white hover:bg-blue-700 disabled:opacity-50">
            Generate
          </button>
        </div>
      </div>

      {jobId && <div className="text-sm text-gray-600">Generating cases... (this may take a minute)</div>}

      {result && result.status === "completed" && (
        <div className="rounded-lg bg-green-50 border border-green-200 p-4">
          <p className="font-medium text-green-800">Generation complete ({result.elapsed_seconds}s)</p>
          <p className="text-sm text-green-700">{result.cases_generated} generated, {result.cases_failed} failed</p>
        </div>
      )}

      {result && result.status === "failed" && (
        <div className="rounded-lg bg-red-50 border border-red-200 p-4">
          <p className="font-medium text-red-800">Generation failed</p>
          <p className="text-sm text-red-700">{result.error}</p>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Create CasesPage**

```tsx
// ui/src/pages/CasesPage.tsx
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchCases } from "../api/client";
import { useNavigate } from "react-router-dom";

export default function CasesPage() {
  const [topicFilter, setTopicFilter] = useState("");
  const [difficultyFilter, setDifficultyFilter] = useState("");
  const navigate = useNavigate();

  const { data, isLoading } = useQuery({
    queryKey: ["cases", topicFilter, difficultyFilter],
    queryFn: () => fetchCases({
      topic: topicFilter || undefined,
      difficulty: difficultyFilter || undefined,
      limit: 50,
    }),
  });

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Cases</h1>

      <div className="flex gap-3">
        <input type="text" value={topicFilter} onChange={(e) => setTopicFilter(e.target.value)}
          placeholder="Filter by topic" className="rounded-lg border border-gray-300 px-3 py-2 text-sm" />
        <select value={difficultyFilter} onChange={(e) => setDifficultyFilter(e.target.value)}
          className="rounded-lg border border-gray-300 px-3 py-2 text-sm">
          <option value="">All difficulties</option>
          <option value="medical_student">Medical Student</option>
          <option value="resident">Resident</option>
          <option value="attending">Attending</option>
        </select>
      </div>

      {isLoading && <p className="text-sm text-gray-500">Loading...</p>}

      {data && (
        <div className="space-y-2">
          <p className="text-sm text-gray-500">{data.total} case(s)</p>
          {data.cases.map((c) => (
            <div key={c.case_id}
              onClick={() => navigate(`/play/${c.case_id}`)}
              className="flex items-center justify-between rounded-lg border border-gray-200 p-4 hover:border-blue-300 cursor-pointer">
              <div>
                <p className="font-medium">{c.topic}</p>
                <p className="text-sm text-gray-500">{c.difficulty} | {c.specialty.join(", ")}</p>
              </div>
              <div className="text-right text-sm">
                <p className="text-green-600">Acc: {(c.review?.accuracy_score ?? 0).toFixed(2)}</p>
                <p className="text-gray-400">{c.case_id.slice(0, 8)}</p>
              </div>
            </div>
          ))}
          {data.cases.length === 0 && <p className="text-gray-500">No cases yet. Generate some first.</p>}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Update App.tsx with new routes and nav**

Replace the full content of `ui/src/App.tsx`:

```tsx
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import IngestPage from './pages/IngestPage';
import SearchPage from './pages/SearchPage';
import SourcesPage from './pages/SourcesPage';
import GeneratePage from './pages/GeneratePage';
import CasesPage from './pages/CasesPage';
import PlayCasePage from './pages/PlayCasePage';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 30_000,
    },
  },
});

const navLinkClass = ({ isActive }: { isActive: boolean }) =>
  `px-3 py-2 rounded-md text-sm font-medium transition-colors ${
    isActive
      ? 'bg-blue-100 text-blue-700'
      : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
  }`;

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="min-h-screen bg-gray-50">
          <nav className="border-b border-gray-200 bg-white shadow-sm">
            <div className="mx-auto max-w-4xl px-4 sm:px-6">
              <div className="flex h-14 items-center justify-between">
                <div className="flex items-center gap-1">
                  <span className="mr-4 text-base font-bold text-gray-900 tracking-tight">
                    CaseCrawler
                  </span>
                  <NavLink to="/" end className={navLinkClass}>Ingest</NavLink>
                  <NavLink to="/search" className={navLinkClass}>Search</NavLink>
                  <NavLink to="/sources" className={navLinkClass}>Sources</NavLink>
                  <NavLink to="/generate" className={navLinkClass}>Generate</NavLink>
                  <NavLink to="/cases" className={navLinkClass}>Cases</NavLink>
                </div>
              </div>
            </div>
          </nav>

          <main className="mx-auto max-w-4xl px-4 py-8 sm:px-6">
            <Routes>
              <Route path="/" element={<IngestPage />} />
              <Route path="/search" element={<SearchPage />} />
              <Route path="/sources" element={<SourcesPage />} />
              <Route path="/generate" element={<GeneratePage />} />
              <Route path="/cases" element={<CasesPage />} />
              <Route path="/play/:caseId" element={<PlayCasePage />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
```

- [ ] **Step 5: Create placeholder PlayCasePage** (implemented fully in Task 12)

```tsx
// ui/src/pages/PlayCasePage.tsx
import { useParams } from "react-router-dom";

export default function PlayCasePage() {
  const { caseId } = useParams();
  return <div>Case Player for {caseId} — coming in next task</div>;
}
```

- [ ] **Step 6: Build UI**

Run: `cd ui && npm run build`
Expected: Build succeeds

- [ ] **Step 7: Commit**

```bash
git add ui/
git commit -m "feat: Generate and Cases pages with API integration"
```

---

## Task 12: Case Player UI

**Files:**
- Replace: `ui/src/pages/PlayCasePage.tsx`
- Create: `ui/src/components/VignetteCard.tsx`
- Create: `ui/src/components/DecisionCards.tsx`
- Create: `ui/src/components/OutcomeReveal.tsx`
- Create: `ui/src/components/CaseDebrief.tsx`
- Create: `ui/src/components/DecisionTreeViz.tsx`

- [ ] **Step 1: Create VignetteCard**

```tsx
// ui/src/components/VignetteCard.tsx
import type { GeneratedCase } from "../api/client";

export default function VignetteCard({ caseData }: { caseData: GeneratedCase }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-6 space-y-4">
      <div className="flex items-center gap-2 text-sm text-gray-500">
        <span className="rounded bg-blue-100 px-2 py-0.5 text-blue-700">{caseData.difficulty}</span>
        {caseData.specialty.map((s) => (
          <span key={s} className="rounded bg-gray-100 px-2 py-0.5">{s}</span>
        ))}
      </div>
      <div className="text-sm text-gray-500">
        Patient: {caseData.patient.age}yo {caseData.patient.sex} | {caseData.patient.demographics}
      </div>
      <p className="text-gray-800 leading-relaxed whitespace-pre-wrap">{caseData.vignette}</p>
      <p className="text-lg font-semibold text-gray-900">{caseData.decision_prompt}</p>
    </div>
  );
}
```

- [ ] **Step 2: Create DecisionCards**

```tsx
// ui/src/components/DecisionCards.tsx
import type { GeneratedCase } from "../api/client";

interface Props {
  caseData: GeneratedCase;
  onSelect: (index: number) => void;
  disabled: boolean;
}

export default function DecisionCards({ caseData, onSelect, disabled }: Props) {
  return (
    <div className="space-y-3">
      {caseData.decision_tree.map((choice, i) => (
        <button key={i} onClick={() => onSelect(i)} disabled={disabled}
          className="w-full text-left rounded-lg border border-gray-200 p-4 hover:border-blue-400 hover:bg-blue-50 transition-colors disabled:opacity-60 disabled:hover:border-gray-200 disabled:hover:bg-white">
          <p className="font-medium text-gray-800">{choice.choice}</p>
        </button>
      ))}
    </div>
  );
}
```

- [ ] **Step 3: Create OutcomeReveal**

```tsx
// ui/src/components/OutcomeReveal.tsx
import type { GeneratedCase } from "../api/client";

interface Props {
  caseData: GeneratedCase;
  selectedIndex: number;
}

export default function OutcomeReveal({ caseData, selectedIndex }: Props) {
  const choice = caseData.decision_tree[selectedIndex];
  const isCorrect = choice.is_correct;

  return (
    <div className={`rounded-lg border-2 p-6 space-y-3 ${
      isCorrect ? "border-green-300 bg-green-50" : "border-red-300 bg-red-50"
    }`}>
      <div className="flex items-center gap-2">
        <span className={`text-lg font-bold ${isCorrect ? "text-green-700" : "text-red-700"}`}>
          {isCorrect ? "Correct!" : choice.error_type === "catastrophic" ? "Catastrophic Error" : "Incorrect"}
        </span>
        {choice.error_type && (
          <span className={`text-xs rounded px-2 py-0.5 ${
            choice.error_type === "catastrophic" ? "bg-red-200 text-red-800" : "bg-yellow-200 text-yellow-800"
          }`}>{choice.error_type}</span>
        )}
      </div>
      <p className="text-gray-800"><strong>Your choice:</strong> {choice.choice}</p>
      <p className="text-gray-700"><strong>Outcome:</strong> {choice.outcome}</p>
      <p className="text-gray-600"><strong>Reasoning:</strong> {choice.reasoning}</p>
      {choice.consequence && (
        <p className="text-red-700"><strong>Consequence:</strong> {choice.consequence}</p>
      )}
      {!isCorrect && (
        <div className="mt-3 pt-3 border-t border-gray-200">
          <p className="text-green-700"><strong>Correct answer:</strong>{" "}
            {caseData.decision_tree.find((d) => d.is_correct)?.choice}
          </p>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Create DecisionTreeViz**

```tsx
// ui/src/components/DecisionTreeViz.tsx
import type { GeneratedCase } from "../api/client";

export default function DecisionTreeViz({ caseData }: { caseData: GeneratedCase }) {
  return (
    <div className="space-y-2">
      <h3 className="font-semibold text-gray-700">Decision Tree</h3>
      {caseData.decision_tree.map((choice, i) => {
        const color = choice.is_correct
          ? "border-green-300 bg-green-50"
          : choice.error_type === "catastrophic"
            ? "border-red-300 bg-red-50"
            : "border-yellow-300 bg-yellow-50";
        return (
          <div key={i} className={`rounded-lg border-2 p-3 ${color}`}>
            <div className="flex items-center justify-between">
              <span className="font-medium text-sm">{choice.choice}</span>
              <span className="text-xs">
                {choice.is_correct ? "Correct" : choice.error_type}
              </span>
            </div>
            <p className="text-xs text-gray-600 mt-1">{choice.outcome}</p>
          </div>
        );
      })}
    </div>
  );
}
```

- [ ] **Step 5: Create CaseDebrief**

```tsx
// ui/src/components/CaseDebrief.tsx
import type { GeneratedCase } from "../api/client";
import DecisionTreeViz from "./DecisionTreeViz";

export default function CaseDebrief({ caseData }: { caseData: GeneratedCase }) {
  return (
    <div className="space-y-6">
      <div className="rounded-lg border border-gray-200 bg-white p-6 space-y-3">
        <h3 className="font-semibold text-gray-700">Ground Truth</h3>
        <p><strong>Diagnosis:</strong> {caseData.ground_truth.diagnosis}</p>
        <p><strong>Optimal Next Step:</strong> {caseData.ground_truth.optimal_next_step}</p>
        <p><strong>Rationale:</strong> {caseData.ground_truth.rationale}</p>
        <div>
          <strong>Key Findings:</strong>
          <ul className="list-disc list-inside ml-2 mt-1">
            {caseData.ground_truth.key_findings.map((f, i) => (
              <li key={i} className="text-sm text-gray-600">{f}</li>
            ))}
          </ul>
        </div>
      </div>

      <DecisionTreeViz caseData={caseData} />

      {caseData.complications.length > 0 && (
        <div className="rounded-lg border border-gray-200 bg-white p-6 space-y-3">
          <h3 className="font-semibold text-gray-700">Complications</h3>
          {caseData.complications.map((c, i) => (
            <div key={i} className="text-sm">
              <strong>{c.trigger}:</strong> {c.detail} — {c.event} ({c.outcome})
            </div>
          ))}
        </div>
      )}

      {caseData.sources.length > 0 && (
        <div className="rounded-lg border border-gray-200 bg-white p-6 space-y-2">
          <h3 className="font-semibold text-gray-700">Sources</h3>
          {caseData.sources.map((s, i) => (
            <p key={i} className="text-sm text-gray-600">{String(s.reference || s.type || "")}</p>
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 6: Implement PlayCasePage**

```tsx
// ui/src/pages/PlayCasePage.tsx
import { useState } from "react";
import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { fetchCase } from "../api/client";
import VignetteCard from "../components/VignetteCard";
import DecisionCards from "../components/DecisionCards";
import OutcomeReveal from "../components/OutcomeReveal";
import CaseDebrief from "../components/CaseDebrief";

type Phase = "presentation" | "decision" | "reveal" | "debrief";

export default function PlayCasePage() {
  const { caseId } = useParams();
  const [phase, setPhase] = useState<Phase>("presentation");
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

  const { data: caseData, isLoading } = useQuery({
    queryKey: ["case", caseId],
    queryFn: () => fetchCase(caseId!),
    enabled: !!caseId,
  });

  if (isLoading) return <p>Loading case...</p>;
  if (!caseData) return <p>Case not found.</p>;

  const handleSelect = (index: number) => {
    setSelectedIndex(index);
    setPhase("reveal");
  };

  return (
    <div className="space-y-6">
      <VignetteCard caseData={caseData} />

      {phase === "presentation" && (
        <button onClick={() => setPhase("decision")}
          className="rounded-lg bg-blue-600 px-6 py-2 text-white hover:bg-blue-700">
          Show Choices
        </button>
      )}

      {(phase === "decision" || phase === "reveal" || phase === "debrief") && (
        <DecisionCards caseData={caseData} onSelect={handleSelect} disabled={phase !== "decision"} />
      )}

      {phase === "reveal" && selectedIndex !== null && (
        <>
          <OutcomeReveal caseData={caseData} selectedIndex={selectedIndex} />
          <button onClick={() => setPhase("debrief")}
            className="rounded-lg bg-gray-600 px-6 py-2 text-white hover:bg-gray-700">
            View Full Debrief
          </button>
        </>
      )}

      {phase === "debrief" && (
        <>
          {selectedIndex !== null && <OutcomeReveal caseData={caseData} selectedIndex={selectedIndex} />}
          <CaseDebrief caseData={caseData} />
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 7: Build UI**

Run: `cd ui && npm run build`
Expected: Build succeeds

- [ ] **Step 8: Commit**

```bash
git add ui/
git commit -m "feat: interactive case player with vignette, decisions, outcome reveal, and debrief"
```

---

## Summary

After completing all 12 tasks:

- **4 LLM providers** (Anthropic, OpenAI, OpenRouter, Ollama) with structured output
- **4-stage generation pipeline** (Retriever → Case Generator → Decision Tree Builder → Clinical Reviewer)
- **Retry with feedback** — critic notes fed back to generator on rejection
- **SQLite case store** with filtering, listing, and JSONL export
- **CLI extensions** — `generate` and `cases` commands
- **API extensions** — generation jobs, case CRUD, JSONL export endpoint
- **Case player UI** — interactive 4-phase flow (presentation → decision → reveal → debrief)
- **Full test coverage** for all new components
