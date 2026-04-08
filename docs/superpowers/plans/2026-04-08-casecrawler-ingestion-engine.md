# CaseCrawler Ingestion Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular medical knowledge ingestion engine that crawls, normalizes, chunks, embeds, and stores content from multiple medical data sources into ChromaDB.

**Architecture:** Hybrid plugin-source + processing-pipeline. Source plugins handle per-source API logic. A shared pipeline handles chunking, tagging, embedding, and storage. Graceful degradation — free sources work with zero config, paid sources unlock with API keys.

**Tech Stack:** Python 3.12+, FastAPI, Click, ChromaDB, sentence-transformers, httpx, Pydantic v2, React + TypeScript + Vite + Tailwind CSS + TanStack Query

**Spec:** `docs/superpowers/specs/2026-04-08-casecrawler-ingestion-engine-design.md`

---

## File Map

### Python Backend (`src/casecrawler/`)

| File | Responsibility |
|------|---------------|
| `__init__.py` | Package version |
| `config.py` | Load `.env` + `config.yaml`, expose typed config |
| `models/document.py` | `Document`, `DocumentMetadata`, `Chunk`, `CredibilityLevel` |
| `models/config.py` | Pydantic config schema |
| `sources/base.py` | `BaseSource` ABC |
| `sources/registry.py` | `SourceRegistry` — discover + manage sources |
| `sources/pubmed.py` | PubMed E-utilities source |
| `sources/openfda.py` | OpenFDA adverse events + drug labeling |
| `sources/dailymed.py` | DailyMed drug labels |
| `sources/rxnorm.py` | RxNorm drug data |
| `sources/medrxiv.py` | medRxiv preprints |
| `sources/clinicaltrials.py` | ClinicalTrials.gov v2 |
| `sources/glass.py` | Glass Health (paid) |
| `sources/annas_archive.py` | Anna's Archive (paid) |
| `sources/firecrawl.py` | Firecrawl scraping (paid) |
| `pipeline/chunker.py` | Content-aware text splitting |
| `pipeline/tagger.py` | Rule-based metadata enrichment |
| `pipeline/embedder.py` | Embedding generation (sentence-transformers) |
| `pipeline/store.py` | ChromaDB storage + search |
| `pipeline/orchestrator.py` | Ties pipeline stages together |
| `cli.py` | Click CLI commands |
| `api/app.py` | FastAPI app + CORS |
| `api/routes/ingest.py` | Ingest endpoint + background jobs |
| `api/routes/search.py` | Search endpoint |
| `api/routes/sources.py` | Sources status endpoint |

### React Frontend (`ui/`)

| File | Responsibility |
|------|---------------|
| `src/App.tsx` | Router + layout |
| `src/api/client.ts` | API client functions |
| `src/pages/IngestPage.tsx` | Topic input + source selection + progress |
| `src/pages/SearchPage.tsx` | Search bar + results + filters |
| `src/pages/SourcesPage.tsx` | Source status + config display |
| `src/components/SourceBadge.tsx` | Credibility-colored source badge |
| `src/components/ChunkCard.tsx` | Search result card |
| `src/components/JobProgress.tsx` | Ingest job progress bar |

### Root

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Python packaging + dependencies |
| `config.example.yaml` | Example config |
| `.env.example` | Example env vars |
| `docker-compose.yml` | Multi-service compose |
| `Dockerfile` | Backend container |
| `ui/Dockerfile` | Frontend container |
| `.gitignore` | Ignore data/, .env, node_modules, etc. |
| `LICENSE` | Apache 2.0 |

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/casecrawler/__init__.py`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `config.example.yaml`
- Create: `LICENSE`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "casecrawler"
version = "0.1.0"
description = "Medical knowledge ingestion engine for clinical case generation"
license = "Apache-2.0"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.27",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "pyyaml>=6.0",
    "click>=8.0",
    "chromadb>=0.5",
    "sentence-transformers>=3.0",
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "pytest-httpx>=0.30",
    "httpx[http2]",
    "ruff>=0.6",
]

[project.scripts]
casecrawler = "casecrawler.cli:cli"

[tool.hatch.build.targets.wheel]
packages = ["src/casecrawler"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
target-version = "py312"
```

- [ ] **Step 2: Create package init**

```python
# src/casecrawler/__init__.py
__version__ = "0.1.0"
```

- [ ] **Step 3: Create .gitignore**

```
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.env
data/
.venv/
node_modules/
ui/dist/
.ruff_cache/
.pytest_cache/
*.db
```

- [ ] **Step 4: Create .env.example**

```
# LLM providers (not required for ingestion)
# ANTHROPIC_API_KEY=
# OPENAI_API_KEY=
# OPENROUTER_API_KEY=

# Paid data sources (all optional — system works without these)
# GLASS_API_KEY=
# ANNAS_ARCHIVE_API_KEY=
# FIRECRAWL_API_KEY=

# Free sources (optional — increases rate limits)
# NCBI_API_KEY=
# OPENFDA_API_KEY=
```

- [ ] **Step 5: Create config.example.yaml**

```yaml
ingestion:
  default_limit_per_source: 20
  sources:
    priority:
      - pubmed
      - glass
      - openfda
      - annas_archive
      - dailymed
      - rxnorm
      - medrxiv
      - clinicaltrials
      - firecrawl
    disabled: []

chunking:
  default_chunk_size: 500
  overlap: 50

embedding:
  model: "all-MiniLM-L6-v2"

storage:
  chroma_persist_dir: "./data/chroma"

api:
  host: "0.0.0.0"
  port: 8000
```

- [ ] **Step 6: Create LICENSE file**

Use the standard Apache 2.0 license text. (Full text omitted for brevity — copy from https://www.apache.org/licenses/LICENSE-2.0.txt)

- [ ] **Step 7: Create directory structure**

Run:
```bash
mkdir -p src/casecrawler/{sources,pipeline,models,api/routes} tests ui/src/{pages,components,api}
touch src/casecrawler/sources/__init__.py src/casecrawler/pipeline/__init__.py src/casecrawler/models/__init__.py src/casecrawler/api/__init__.py src/casecrawler/api/routes/__init__.py
```

- [ ] **Step 8: Install and verify**

Run:
```bash
pip install -e ".[dev]"
python -c "import casecrawler; print(casecrawler.__version__)"
```

Expected: `0.1.0`

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml src/ tests/ .gitignore .env.example config.example.yaml LICENSE
git commit -m "feat: project scaffolding with dependencies and directory structure"
```

---

## Task 2: Data Models

**Files:**
- Create: `src/casecrawler/models/document.py`
- Create: `src/casecrawler/models/config.py`
- Create: `src/casecrawler/config.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write tests for data models**

```python
# tests/test_models.py
from datetime import date

from casecrawler.models.document import (
    Chunk,
    CredibilityLevel,
    Document,
    DocumentMetadata,
)


def test_document_creation():
    meta = DocumentMetadata(
        authors=["Smith J", "Doe A"],
        publication_date=date(2024, 1, 15),
        specialty=["neurosurgery"],
        credibility=CredibilityLevel.PEER_REVIEWED,
        url="https://example.com/article",
        doi="10.1234/test",
    )
    doc = Document(
        source="pubmed",
        source_id="12345678",
        title="Test Article",
        content="This is the abstract text.",
        content_type="abstract",
        metadata=meta,
    )
    assert doc.source == "pubmed"
    assert doc.metadata.credibility == CredibilityLevel.PEER_REVIEWED
    assert doc.metadata.authors == ["Smith J", "Doe A"]


def test_document_metadata_defaults():
    meta = DocumentMetadata(credibility=CredibilityLevel.PREPRINT)
    assert meta.authors == []
    assert meta.publication_date is None
    assert meta.specialty == []
    assert meta.url is None
    assert meta.doi is None


def test_credibility_ordering():
    levels = list(CredibilityLevel)
    assert CredibilityLevel.GUIDELINE in levels
    assert CredibilityLevel.PEER_REVIEWED in levels
    assert CredibilityLevel.PREPRINT in levels
    assert CredibilityLevel.CURATED in levels
    assert CredibilityLevel.FDA_LABEL in levels


def test_chunk_creation():
    meta = DocumentMetadata(credibility=CredibilityLevel.PEER_REVIEWED)
    chunk = Chunk(
        chunk_id="abc123",
        source_document_id="pubmed:12345678",
        text="Some chunk text here.",
        position=0,
        metadata=meta,
    )
    assert chunk.chunk_id == "abc123"
    assert chunk.position == 0
    assert chunk.metadata.credibility == CredibilityLevel.PEER_REVIEWED
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'casecrawler.models.document'`

- [ ] **Step 3: Implement data models**

```python
# src/casecrawler/models/document.py
from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel


class CredibilityLevel(str, Enum):
    GUIDELINE = "guideline"
    PEER_REVIEWED = "peer_reviewed"
    PREPRINT = "preprint"
    CURATED = "curated"
    FDA_LABEL = "fda_label"


class DocumentMetadata(BaseModel):
    authors: list[str] = []
    publication_date: date | None = None
    specialty: list[str] = []
    credibility: CredibilityLevel
    url: str | None = None
    doi: str | None = None


class Document(BaseModel):
    source: str
    source_id: str
    title: str
    content: str
    content_type: str
    metadata: DocumentMetadata


class Chunk(BaseModel):
    chunk_id: str
    source_document_id: str
    text: str
    position: int
    metadata: DocumentMetadata
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Implement config models**

```python
# src/casecrawler/models/config.py
from __future__ import annotations

from pydantic import BaseModel


class SourcesConfig(BaseModel):
    priority: list[str] = [
        "pubmed", "glass", "openfda", "annas_archive",
        "dailymed", "rxnorm", "medrxiv", "clinicaltrials", "firecrawl",
    ]
    disabled: list[str] = []


class IngestionConfig(BaseModel):
    default_limit_per_source: int = 20
    sources: SourcesConfig = SourcesConfig()


class ChunkingConfig(BaseModel):
    default_chunk_size: int = 500
    overlap: int = 50


class EmbeddingConfig(BaseModel):
    model: str = "all-MiniLM-L6-v2"


class StorageConfig(BaseModel):
    chroma_persist_dir: str = "./data/chroma"


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class AppConfig(BaseModel):
    ingestion: IngestionConfig = IngestionConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    storage: StorageConfig = StorageConfig()
    api: ApiConfig = ApiConfig()
```

- [ ] **Step 6: Implement config loading**

```python
# src/casecrawler/config.py
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from casecrawler.models.config import AppConfig

_config: AppConfig | None = None


def load_config(config_path: str | None = None) -> AppConfig:
    global _config

    load_dotenv()

    if config_path is None:
        candidates = ["config.yaml", "config.yml"]
        for candidate in candidates:
            if Path(candidate).exists():
                config_path = candidate
                break

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        _config = AppConfig(**raw)
    else:
        _config = AppConfig()

    return _config


def get_config() -> AppConfig:
    if _config is None:
        return load_config()
    return _config


def get_env(key: str) -> str | None:
    return os.environ.get(key)
```

- [ ] **Step 7: Write config test**

```python
# tests/test_config.py
from casecrawler.config import load_config
from casecrawler.models.config import AppConfig


def test_load_default_config():
    config = load_config(config_path="/nonexistent/path.yaml")
    assert isinstance(config, AppConfig)
    assert config.ingestion.default_limit_per_source == 20
    assert config.chunking.default_chunk_size == 500
    assert config.embedding.model == "all-MiniLM-L6-v2"


def test_load_config_from_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "ingestion:\n  default_limit_per_source: 50\nchunking:\n  default_chunk_size: 300\n"
    )
    config = load_config(config_path=str(config_file))
    assert config.ingestion.default_limit_per_source == 50
    assert config.chunking.default_chunk_size == 300
    # defaults still work for unset values
    assert config.embedding.model == "all-MiniLM-L6-v2"
```

- [ ] **Step 8: Run all tests**

Run: `pytest tests/test_models.py tests/test_config.py -v`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add src/casecrawler/models/ src/casecrawler/config.py tests/test_models.py tests/test_config.py
git commit -m "feat: data models and config loading"
```

---

## Task 3: Source Base Class + Registry

**Files:**
- Create: `src/casecrawler/sources/base.py`
- Create: `src/casecrawler/sources/registry.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write registry tests**

```python
# tests/test_registry.py
import os
from unittest.mock import patch

from casecrawler.models.document import (
    CredibilityLevel,
    Document,
    DocumentMetadata,
)
from casecrawler.sources.base import BaseSource
from casecrawler.sources.registry import SourceRegistry


class FakeFreeSrc(BaseSource):
    name = "fake_free"
    requires_keys: list[str] = []

    async def search(self, query: str, limit: int = 20) -> list[Document]:
        return [
            Document(
                source=self.name,
                source_id="1",
                title="Fake",
                content="Fake content",
                content_type="abstract",
                metadata=DocumentMetadata(credibility=CredibilityLevel.PEER_REVIEWED),
            )
        ]

    async def fetch(self, document_id: str) -> Document:
        return Document(
            source=self.name,
            source_id=document_id,
            title="Fake",
            content="Full content",
            content_type="abstract",
            metadata=DocumentMetadata(credibility=CredibilityLevel.PEER_REVIEWED),
        )


class FakePaidSrc(BaseSource):
    name = "fake_paid"
    requires_keys: list[str] = ["FAKE_PAID_API_KEY"]

    async def search(self, query: str, limit: int = 20) -> list[Document]:
        return []

    async def fetch(self, document_id: str) -> Document:
        raise NotImplementedError


def test_registry_discovers_free_source():
    registry = SourceRegistry()
    registry.discover()
    assert "fake_free" in registry.available_source_names


def test_registry_excludes_paid_without_key():
    with patch.dict(os.environ, {}, clear=True):
        registry = SourceRegistry()
        registry.discover()
        assert "fake_paid" not in registry.available_source_names


def test_registry_includes_paid_with_key():
    with patch.dict(os.environ, {"FAKE_PAID_API_KEY": "test-key"}):
        registry = SourceRegistry()
        registry.discover()
        assert "fake_paid" in registry.available_source_names


def test_registry_get_source():
    registry = SourceRegistry()
    registry.discover()
    source = registry.get("fake_free")
    assert source is not None
    assert source.name == "fake_free"


def test_registry_get_missing_source():
    registry = SourceRegistry()
    registry.discover()
    assert registry.get("nonexistent") is None


def test_registry_all_sources_info():
    with patch.dict(os.environ, {}, clear=True):
        registry = SourceRegistry()
        registry.discover()
        info = registry.all_sources_info()
        paid_info = [s for s in info if s["name"] == "fake_paid"]
        assert len(paid_info) == 1
        assert paid_info[0]["available"] is False
        assert paid_info[0]["missing_keys"] == ["FAKE_PAID_API_KEY"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement BaseSource**

```python
# src/casecrawler/sources/base.py
from __future__ import annotations

import os
from abc import ABC, abstractmethod

from casecrawler.models.document import Document


class BaseSource(ABC):
    name: str
    requires_keys: list[str] = []

    @abstractmethod
    async def search(self, query: str, limit: int = 20) -> list[Document]:
        """Search the source for documents matching a query."""

    @abstractmethod
    async def fetch(self, document_id: str) -> Document:
        """Fetch full content for a specific document."""

    @classmethod
    def is_available(cls) -> bool:
        """Check if all required API keys are present in env."""
        return all(os.environ.get(key) for key in cls.requires_keys)

    @classmethod
    def missing_keys(cls) -> list[str]:
        """Return list of missing required keys."""
        return [key for key in cls.requires_keys if not os.environ.get(key)]
```

- [ ] **Step 4: Implement SourceRegistry**

```python
# src/casecrawler/sources/registry.py
from __future__ import annotations

from casecrawler.sources.base import BaseSource


class SourceRegistry:
    def __init__(self) -> None:
        self._sources: dict[str, BaseSource] = {}

    def discover(self) -> None:
        """Find all BaseSource subclasses, instantiate those with valid credentials."""
        for source_cls in BaseSource.__subclasses__():
            if source_cls.is_available():
                self._sources[source_cls.name] = source_cls()

    @property
    def available_source_names(self) -> list[str]:
        return list(self._sources.keys())

    def get(self, name: str) -> BaseSource | None:
        return self._sources.get(name)

    def get_sources(self, names: list[str] | None = None) -> list[BaseSource]:
        """Get sources by name, or all available if names is None."""
        if names is None:
            return list(self._sources.values())
        return [self._sources[n] for n in names if n in self._sources]

    def all_sources_info(self) -> list[dict]:
        """Info about all known sources (available and unavailable)."""
        info = []
        for source_cls in BaseSource.__subclasses__():
            available = source_cls.is_available()
            entry = {
                "name": source_cls.name,
                "requires_keys": list(source_cls.requires_keys),
                "available": available,
            }
            if not available:
                entry["missing_keys"] = source_cls.missing_keys()
            info.append(entry)
        return info
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_registry.py -v`
Expected: All 6 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/casecrawler/sources/base.py src/casecrawler/sources/registry.py tests/test_registry.py
git commit -m "feat: source plugin base class and registry with auto-discovery"
```

---

## Task 4: PubMed Source Plugin

**Files:**
- Create: `src/casecrawler/sources/pubmed.py`
- Test: `tests/test_pubmed.py`

- [ ] **Step 1: Write PubMed tests with mocked HTTP**

```python
# tests/test_pubmed.py
import pytest

from casecrawler.models.document import CredibilityLevel
from casecrawler.sources.pubmed import PubMedSource

ESEARCH_RESPONSE = {
    "esearchresult": {
        "count": "2",
        "retmax": "2",
        "retstart": "0",
        "idlist": ["38901234", "38905678"],
    }
}

EFETCH_XML = """<?xml version="1.0" ?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>38901234</PMID>
      <Article>
        <ArticleTitle>Management of Subarachnoid Hemorrhage: A Clinical Guideline</ArticleTitle>
        <Abstract>
          <AbstractText>This guideline addresses the management of aneurysmal subarachnoid hemorrhage including diagnosis and treatment.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Smith</LastName>
            <ForeName>John</ForeName>
          </Author>
          <Author>
            <LastName>Doe</LastName>
            <ForeName>Jane</ForeName>
          </Author>
        </AuthorList>
        <Journal>
          <JournalIssue>
            <PubDate><Year>2024</Year><Month>Mar</Month></PubDate>
          </JournalIssue>
        </Journal>
        <ELocationID EIdType="doi">10.1234/sah.2024</ELocationID>
      </Article>
      <PublicationTypeList>
        <PublicationType>Practice Guideline</PublicationType>
      </PublicationTypeList>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>38905678</PMID>
      <Article>
        <ArticleTitle>Outcomes After SAH: A Retrospective Study</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">SAH is a severe condition.</AbstractText>
          <AbstractText Label="RESULTS">Outcomes varied significantly.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Lee</LastName>
            <ForeName>Alice</ForeName>
          </Author>
        </AuthorList>
        <Journal>
          <JournalIssue>
            <PubDate><Year>2024</Year><Month>Feb</Month></PubDate>
          </JournalIssue>
        </Journal>
        <ELocationID EIdType="doi">10.5678/sah.2024</ELocationID>
      </Article>
      <PublicationTypeList>
        <PublicationType>Journal Article</PublicationType>
      </PublicationTypeList>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


@pytest.fixture
def pubmed():
    return PubMedSource()


@pytest.mark.asyncio
async def test_pubmed_search(pubmed, httpx_mock):
    httpx_mock.add_response(
        url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        json=ESEARCH_RESPONSE,
    )
    httpx_mock.add_response(
        url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        text=EFETCH_XML,
    )
    docs = await pubmed.search("subarachnoid hemorrhage", limit=2)
    assert len(docs) == 2
    assert docs[0].source == "pubmed"
    assert docs[0].source_id == "38901234"
    assert docs[0].title == "Management of Subarachnoid Hemorrhage: A Clinical Guideline"
    assert "management" in docs[0].content.lower()
    assert docs[0].metadata.authors == ["Smith John", "Doe Jane"]
    assert docs[0].metadata.doi == "10.1234/sah.2024"
    assert docs[0].metadata.credibility == CredibilityLevel.GUIDELINE


@pytest.mark.asyncio
async def test_pubmed_structured_abstract(pubmed, httpx_mock):
    httpx_mock.add_response(
        url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        json=ESEARCH_RESPONSE,
    )
    httpx_mock.add_response(
        url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        text=EFETCH_XML,
    )
    docs = await pubmed.search("SAH", limit=2)
    # Second article has structured abstract with labels
    assert "BACKGROUND" in docs[1].content or "SAH is a severe" in docs[1].content


@pytest.mark.asyncio
async def test_pubmed_credibility_detection(pubmed, httpx_mock):
    httpx_mock.add_response(
        url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        json=ESEARCH_RESPONSE,
    )
    httpx_mock.add_response(
        url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        text=EFETCH_XML,
    )
    docs = await pubmed.search("SAH", limit=2)
    assert docs[0].metadata.credibility == CredibilityLevel.GUIDELINE
    assert docs[1].metadata.credibility == CredibilityLevel.PEER_REVIEWED


def test_pubmed_is_available():
    assert PubMedSource.is_available() is True


def test_pubmed_name():
    assert PubMedSource.name == "pubmed"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pubmed.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement PubMed source**

```python
# src/casecrawler/sources/pubmed.py
from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import date

import httpx

from casecrawler.config import get_env
from casecrawler.models.document import (
    CredibilityLevel,
    Document,
    DocumentMetadata,
)
from casecrawler.sources.base import BaseSource

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

GUIDELINE_TYPES = {"Practice Guideline", "Guideline"}


class PubMedSource(BaseSource):
    name = "pubmed"
    requires_keys: list[str] = []

    def _base_params(self) -> dict:
        params: dict = {"tool": "casecrawler", "email": "casecrawler@example.com"}
        api_key = get_env("NCBI_API_KEY")
        if api_key:
            params["api_key"] = api_key
        return params

    async def search(self, query: str, limit: int = 20) -> list[Document]:
        async with httpx.AsyncClient() as client:
            # Step 1: search for PMIDs
            params = {
                **self._base_params(),
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": str(limit),
                "sort": "relevance",
            }
            resp = await client.get(ESEARCH_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            pmids = data["esearchresult"]["idlist"]

            if not pmids:
                return []

            # Step 2: fetch article details
            params = {
                **self._base_params(),
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
            }
            resp = await client.get(EFETCH_URL, params=params)
            resp.raise_for_status()
            return self._parse_articles(resp.text)

    async def fetch(self, document_id: str) -> Document:
        async with httpx.AsyncClient() as client:
            params = {
                **self._base_params(),
                "db": "pubmed",
                "id": document_id,
                "retmode": "xml",
            }
            resp = await client.get(EFETCH_URL, params=params)
            resp.raise_for_status()
            docs = self._parse_articles(resp.text)
            if not docs:
                raise ValueError(f"No article found for PMID {document_id}")
            return docs[0]

    def _parse_articles(self, xml_text: str) -> list[Document]:
        root = ET.fromstring(xml_text)
        documents = []

        for article_elem in root.findall(".//PubmedArticle"):
            citation = article_elem.find("MedlineCitation")
            if citation is None:
                continue

            pmid_elem = citation.find("PMID")
            pmid = pmid_elem.text if pmid_elem is not None else ""

            art = citation.find("Article")
            if art is None:
                continue

            # Title
            title_elem = art.find("ArticleTitle")
            title = title_elem.text or "" if title_elem is not None else ""

            # Abstract — may be single or structured (multiple AbstractText with Label)
            abstract_parts = []
            abstract_elem = art.find("Abstract")
            if abstract_elem is not None:
                for at in abstract_elem.findall("AbstractText"):
                    label = at.get("Label")
                    text = at.text or ""
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
            content = "\n\n".join(abstract_parts)

            # Authors
            authors = []
            author_list = art.find("AuthorList")
            if author_list is not None:
                for author in author_list.findall("Author"):
                    last = author.findtext("LastName", "")
                    fore = author.findtext("ForeName", "")
                    collective = author.findtext("CollectiveName", "")
                    if collective:
                        authors.append(collective)
                    elif last:
                        authors.append(f"{last} {fore}".strip())

            # DOI
            doi = None
            eloc = art.find("ELocationID[@EIdType='doi']")
            if eloc is not None:
                doi = eloc.text

            # Publication date
            pub_date = self._parse_pub_date(art)

            # Publication types → credibility
            pub_types = set()
            pt_list = citation.find("PublicationTypeList")
            if pt_list is not None:
                for pt in pt_list.findall("PublicationType"):
                    if pt.text:
                        pub_types.add(pt.text)

            if pub_types & GUIDELINE_TYPES:
                credibility = CredibilityLevel.GUIDELINE
            else:
                credibility = CredibilityLevel.PEER_REVIEWED

            metadata = DocumentMetadata(
                authors=authors,
                publication_date=pub_date,
                specialty=[],
                credibility=credibility,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                doi=doi,
            )

            documents.append(
                Document(
                    source="pubmed",
                    source_id=pmid,
                    title=title,
                    content=content,
                    content_type="abstract",
                    metadata=metadata,
                )
            )

        return documents

    def _parse_pub_date(self, article_elem: ET.Element) -> date | None:
        journal_issue = article_elem.find(".//JournalIssue/PubDate")
        if journal_issue is None:
            return None
        year_text = journal_issue.findtext("Year")
        if not year_text:
            return None
        year = int(year_text)
        month_text = journal_issue.findtext("Month", "").lower()
        month = MONTH_MAP.get(month_text, 1)
        day_text = journal_issue.findtext("Day", "1")
        try:
            return date(year, month, int(day_text))
        except ValueError:
            return date(year, month, 1)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_pubmed.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/casecrawler/sources/pubmed.py tests/test_pubmed.py
git commit -m "feat: PubMed source plugin with E-utilities search and XML parsing"
```

---

## Task 5: Processing Pipeline — Chunker

**Files:**
- Create: `src/casecrawler/pipeline/chunker.py`
- Test: `tests/test_chunker.py`

- [ ] **Step 1: Write chunker tests**

```python
# tests/test_chunker.py
from casecrawler.models.document import (
    CredibilityLevel,
    Document,
    DocumentMetadata,
)
from casecrawler.pipeline.chunker import Chunker


def _make_doc(content: str, content_type: str = "abstract") -> Document:
    return Document(
        source="test",
        source_id="123",
        title="Test Doc",
        content=content,
        content_type=content_type,
        metadata=DocumentMetadata(credibility=CredibilityLevel.PEER_REVIEWED),
    )


def test_abstract_kept_whole():
    doc = _make_doc("Short abstract text about a disease.", "abstract")
    chunker = Chunker(chunk_size=500, overlap=50)
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "Short abstract text about a disease."
    assert chunks[0].position == 0


def test_chunk_id_deterministic():
    doc = _make_doc("Some content.", "abstract")
    chunker = Chunker(chunk_size=500, overlap=50)
    chunks_a = chunker.chunk(doc)
    chunks_b = chunker.chunk(doc)
    assert chunks_a[0].chunk_id == chunks_b[0].chunk_id


def test_full_text_split():
    paragraphs = ["Paragraph one. " * 50, "Paragraph two. " * 50, "Paragraph three. " * 50]
    content = "\n\n".join(paragraphs)
    doc = _make_doc(content, "full_text")
    chunker = Chunker(chunk_size=200, overlap=20)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1
    # Check ordering
    for i, chunk in enumerate(chunks):
        assert chunk.position == i
    # Check back-reference
    assert all(c.source_document_id == "test:123" for c in chunks)


def test_drug_label_split_by_section():
    content = (
        "INDICATIONS AND USAGE\nThis drug is indicated for pain.\n\n"
        "CONTRAINDICATIONS\nDo not use if allergic.\n\n"
        "ADVERSE REACTIONS\nMay cause nausea and headache."
    )
    doc = _make_doc(content, "drug_label")
    chunker = Chunker(chunk_size=500, overlap=50)
    chunks = chunker.chunk(doc)
    # Should split into sections
    assert len(chunks) >= 3
    assert "indicated for pain" in chunks[0].text
    assert "allergic" in chunks[1].text
    assert "nausea" in chunks[2].text


def test_metadata_inherited():
    doc = _make_doc("Text.", "abstract")
    chunker = Chunker(chunk_size=500, overlap=50)
    chunks = chunker.chunk(doc)
    assert chunks[0].metadata.credibility == CredibilityLevel.PEER_REVIEWED
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chunker.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement chunker**

```python
# src/casecrawler/pipeline/chunker.py
from __future__ import annotations

import hashlib
import re

from casecrawler.models.document import Chunk, Document

# Section headers commonly found in drug labels and clinical documents
DRUG_LABEL_SECTIONS = re.compile(
    r"^(INDICATIONS AND USAGE|DOSAGE AND ADMINISTRATION|CONTRAINDICATIONS|"
    r"WARNINGS AND PRECAUTIONS|WARNINGS|ADVERSE REACTIONS|DRUG INTERACTIONS|"
    r"DESCRIPTION|CLINICAL PHARMACOLOGY|OVERDOSAGE|BOXED WARNING)",
    re.MULTILINE | re.IGNORECASE,
)

TRIAL_SECTIONS = re.compile(
    r"^(ELIGIBILITY|INCLUSION CRITERIA|EXCLUSION CRITERIA|INTERVENTIONS|"
    r"PRIMARY OUTCOME|SECONDARY OUTCOME|STUDY DESIGN)",
    re.MULTILINE | re.IGNORECASE,
)


class Chunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        if doc.content_type == "abstract":
            texts = [doc.content]
        elif doc.content_type == "drug_label":
            texts = self._split_by_sections(doc.content, DRUG_LABEL_SECTIONS)
        elif doc.content_type == "trial_protocol":
            texts = self._split_by_sections(doc.content, TRIAL_SECTIONS)
        else:
            texts = self._split_by_size(doc.content)

        doc_ref = f"{doc.source}:{doc.source_id}"
        chunks = []
        for i, text in enumerate(texts):
            text = text.strip()
            if not text:
                continue
            chunk_id = self._make_chunk_id(doc_ref, i)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_document_id=doc_ref,
                    text=text,
                    position=i,
                    metadata=doc.metadata,
                )
            )
        return chunks

    def _split_by_sections(self, text: str, pattern: re.Pattern) -> list[str]:
        """Split text on section headers, keeping the header with its content."""
        splits = pattern.split(text)
        if len(splits) <= 1:
            return self._split_by_size(text)

        sections = []
        # splits[0] is text before first header (often empty)
        if splits[0].strip():
            sections.append(splits[0].strip())
        # After that, alternating: header, content
        for i in range(1, len(splits), 2):
            header = splits[i]
            content = splits[i + 1] if i + 1 < len(splits) else ""
            section_text = f"{header}\n{content.strip()}"
            # If section is too long, further split it
            if len(section_text) > self.chunk_size * 2:
                sections.extend(self._split_by_size(section_text))
            else:
                sections.append(section_text)
        return sections

    def _split_by_size(self, text: str) -> list[str]:
        """Split text into chunks of roughly chunk_size characters with overlap."""
        # Try to split on paragraph boundaries first
        paragraphs = re.split(r"\n\n+", text)

        chunks = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 2 > self.chunk_size and current:
                chunks.append(current)
                # Overlap: keep the last overlap chars
                if self.overlap > 0 and len(current) > self.overlap:
                    current = current[-self.overlap :] + "\n\n" + para
                else:
                    current = para
            else:
                current = current + "\n\n" + para if current else para

        if current.strip():
            chunks.append(current)

        return chunks

    def _make_chunk_id(self, doc_ref: str, position: int) -> str:
        raw = f"{doc_ref}:{position}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_chunker.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/casecrawler/pipeline/chunker.py tests/test_chunker.py
git commit -m "feat: content-aware chunker with section and size-based splitting"
```

---

## Task 6: Processing Pipeline — Tagger, Embedder, Store

**Files:**
- Create: `src/casecrawler/pipeline/tagger.py`
- Create: `src/casecrawler/pipeline/embedder.py`
- Create: `src/casecrawler/pipeline/store.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write tests for tagger, embedder, and store**

```python
# tests/test_pipeline.py
import tempfile

from casecrawler.models.document import Chunk, CredibilityLevel, DocumentMetadata
from casecrawler.pipeline.embedder import Embedder
from casecrawler.pipeline.store import ChunkStore
from casecrawler.pipeline.tagger import Tagger


def _make_chunk(text: str = "Sample text", specialty: list[str] | None = None) -> Chunk:
    return Chunk(
        chunk_id="test_chunk_1",
        source_document_id="pubmed:123",
        text=text,
        position=0,
        metadata=DocumentMetadata(
            credibility=CredibilityLevel.PEER_REVIEWED,
            specialty=specialty or [],
        ),
    )


# --- Tagger ---


def test_tagger_adds_specialty():
    chunk = _make_chunk("The patient presented with acute subarachnoid hemorrhage requiring neurosurgical intervention.")
    tagger = Tagger()
    tagged = tagger.tag(chunk)
    assert "neurosurgery" in tagged.metadata.specialty


def test_tagger_multiple_specialties():
    chunk = _make_chunk("Cardiac arrest during anesthesia required emergency cardiology and critical care intervention.")
    tagger = Tagger()
    tagged = tagger.tag(chunk)
    assert len(tagged.metadata.specialty) >= 1


def test_tagger_preserves_existing_specialty():
    chunk = _make_chunk("Some text.", specialty=["oncology"])
    tagger = Tagger()
    tagged = tagger.tag(chunk)
    assert "oncology" in tagged.metadata.specialty


# --- Embedder ---


def test_embedder_produces_vectors():
    chunk = _make_chunk("Subarachnoid hemorrhage management.")
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    results = embedder.embed([chunk])
    assert len(results) == 1
    chunk_out, embedding = results[0]
    assert chunk_out.chunk_id == "test_chunk_1"
    assert len(embedding) == 384  # MiniLM output dimension


def test_embedder_batch():
    chunks = [_make_chunk(f"Text {i}") for i in range(3)]
    # Give them unique IDs
    for i, c in enumerate(chunks):
        c.chunk_id = f"chunk_{i}"
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    results = embedder.embed(chunks)
    assert len(results) == 3


# --- Store ---


def test_store_and_search():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChunkStore(persist_dir=tmpdir)
        chunk = _make_chunk("Subarachnoid hemorrhage is a neurosurgical emergency.")
        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        embedded = embedder.embed([chunk])
        store.store(embedded)
        results = store.search("brain hemorrhage", n_results=5)
        assert len(results) >= 1
        assert results[0]["chunk_id"] == "test_chunk_1"


def test_store_deduplication():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChunkStore(persist_dir=tmpdir)
        chunk = _make_chunk("Duplicate test content.")
        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        embedded = embedder.embed([chunk])
        store.store(embedded)
        store.store(embedded)  # store again — should upsert, not duplicate
        results = store.search("duplicate test", n_results=10)
        matching = [r for r in results if r["chunk_id"] == "test_chunk_1"]
        assert len(matching) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement tagger**

```python
# src/casecrawler/pipeline/tagger.py
from __future__ import annotations

import re

from casecrawler.models.document import Chunk

# Keyword → specialty mapping for rule-based tagging
SPECIALTY_KEYWORDS: dict[str, list[str]] = {
    "neurosurgery": ["neurosurg", "craniotomy", "brain tumor", "spinal fusion", "laminectomy"],
    "neurology": ["stroke", "seizure", "epilepsy", "multiple sclerosis", "parkinson"],
    "cardiology": ["cardiac", "myocardial", "heart failure", "arrhythmia", "coronary"],
    "cardiothoracic_surgery": ["cabg", "valve replacement", "thoracotomy"],
    "orthopedics": ["fracture", "arthroplasty", "orthop", "acl", "meniscus"],
    "oncology": ["cancer", "tumor", "chemotherapy", "radiation therapy", "oncol"],
    "emergency_medicine": ["emergency department", "trauma", "resuscitation", "triage"],
    "critical_care": ["icu", "intensive care", "ventilator", "sepsis", "critical care"],
    "pulmonology": ["pulmonary", "asthma", "copd", "pneumonia", "bronch"],
    "gastroenterology": ["gi bleed", "endoscopy", "hepat", "cirrhosis", "pancreat"],
    "nephrology": ["renal", "dialysis", "kidney", "nephr", "glomerulo"],
    "endocrinology": ["diabetes", "thyroid", "adrenal", "insulin", "endocrin"],
    "infectious_disease": ["infection", "antibiotic", "sepsis", "hiv", "tuberculosis"],
    "psychiatry": ["depression", "anxiety", "schizophrenia", "psychiatric", "bipolar"],
    "pediatrics": ["pediatric", "neonatal", "infant", "child", "adolescent"],
    "obstetrics_gynecology": ["pregnan", "obstetric", "gynecol", "cesarean", "preeclampsia"],
    "anesthesiology": ["anesthes", "sedation", "intubation", "airway management"],
    "radiology": ["ct scan", "mri", "imaging", "radiograph", "ultrasound"],
    "dermatology": ["skin", "dermat", "melanoma", "psoriasis", "eczema"],
    "ophthalmology": ["eye", "retina", "cataract", "glaucoma", "ophthalm"],
    "urology": ["urolog", "prostate", "bladder", "kidney stone", "renal calcul"],
}


class Tagger:
    def tag(self, chunk: Chunk) -> Chunk:
        """Add specialty tags based on keyword matching. Preserves existing tags."""
        text_lower = chunk.text.lower()
        new_specialties = set(chunk.metadata.specialty)

        for specialty, keywords in SPECIALTY_KEYWORDS.items():
            for keyword in keywords:
                if re.search(re.escape(keyword), text_lower):
                    new_specialties.add(specialty)
                    break

        updated_metadata = chunk.metadata.model_copy(
            update={"specialty": sorted(new_specialties)}
        )
        return chunk.model_copy(update={"metadata": updated_metadata})
```

- [ ] **Step 4: Implement embedder**

```python
# src/casecrawler/pipeline/embedder.py
from __future__ import annotations

from sentence_transformers import SentenceTransformer

from casecrawler.models.document import Chunk


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)

    def embed(self, chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
        """Generate embeddings for a list of chunks. Returns (chunk, embedding) pairs."""
        texts = [c.text for c in chunks]
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return [(chunk, emb.tolist()) for chunk, emb in zip(chunks, embeddings)]
```

- [ ] **Step 5: Implement store**

```python
# src/casecrawler/pipeline/store.py
from __future__ import annotations

import chromadb

from casecrawler.models.document import Chunk


class ChunkStore:
    COLLECTION_NAME = "casecrawler"

    def __init__(self, persist_dir: str = "./data/chroma") -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def store(self, embedded_chunks: list[tuple[Chunk, list[float]]]) -> None:
        """Upsert chunks with their embeddings into ChromaDB."""
        if not embedded_chunks:
            return

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for chunk, embedding in embedded_chunks:
            ids.append(chunk.chunk_id)
            embeddings.append(embedding)
            documents.append(chunk.text)
            metadatas.append({
                "source_document_id": chunk.source_document_id,
                "source": chunk.source_document_id.split(":")[0] if ":" in chunk.source_document_id else "",
                "position": chunk.position,
                "credibility": chunk.metadata.credibility.value,
                "specialty": ",".join(chunk.metadata.specialty),
                "authors": ",".join(chunk.metadata.authors),
                "doi": chunk.metadata.doi or "",
                "url": chunk.metadata.url or "",
            })

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(
        self,
        query_text: str,
        n_results: int = 10,
        source: str | None = None,
        embedder: object | None = None,
    ) -> list[dict]:
        """Search for chunks matching query text. Optionally filter by source."""
        # Use the collection's built-in embedding if no embedder provided
        where = None
        if source:
            where = {"source": source}

        results = self._collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                output.append({
                    "chunk_id": chunk_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1.0 - results["distances"][0][i],  # cosine distance → similarity
                })
        return output

    @property
    def count(self) -> int:
        return self._collection.count()
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_pipeline.py -v`
Expected: All 7 tests PASS (note: first run will download the MiniLM model — may take a minute)

- [ ] **Step 7: Commit**

```bash
git add src/casecrawler/pipeline/tagger.py src/casecrawler/pipeline/embedder.py src/casecrawler/pipeline/store.py tests/test_pipeline.py
git commit -m "feat: pipeline stages — tagger, embedder, and ChromaDB store"
```

---

## Task 7: Pipeline Orchestrator

**Files:**
- Create: `src/casecrawler/pipeline/orchestrator.py`
- Test: `tests/test_orchestrator.py`

- [ ] **Step 1: Write orchestrator tests**

```python
# tests/test_orchestrator.py
import tempfile

from casecrawler.models.document import (
    CredibilityLevel,
    Document,
    DocumentMetadata,
)
from casecrawler.pipeline.orchestrator import PipelineOrchestrator


def _make_docs(n: int = 2) -> list[Document]:
    return [
        Document(
            source="test",
            source_id=str(i),
            title=f"Test Article {i}",
            content=f"This is test content about subarachnoid hemorrhage number {i}.",
            content_type="abstract",
            metadata=DocumentMetadata(credibility=CredibilityLevel.PEER_REVIEWED),
        )
        for i in range(n)
    ]


def test_orchestrator_processes_documents():
    with tempfile.TemporaryDirectory() as tmpdir:
        orch = PipelineOrchestrator(chroma_dir=tmpdir)
        docs = _make_docs(3)
        result = orch.process(docs)
        assert result["documents"] == 3
        assert result["chunks"] >= 3
        assert orch.store.count >= 3


def test_orchestrator_idempotent():
    with tempfile.TemporaryDirectory() as tmpdir:
        orch = PipelineOrchestrator(chroma_dir=tmpdir)
        docs = _make_docs(2)
        orch.process(docs)
        count_first = orch.store.count
        orch.process(docs)  # same docs again
        count_second = orch.store.count
        assert count_first == count_second  # dedup via chunk_id
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_orchestrator.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement orchestrator**

```python
# src/casecrawler/pipeline/orchestrator.py
from __future__ import annotations

from casecrawler.config import get_config
from casecrawler.models.document import Document
from casecrawler.pipeline.chunker import Chunker
from casecrawler.pipeline.embedder import Embedder
from casecrawler.pipeline.store import ChunkStore
from casecrawler.pipeline.tagger import Tagger


class PipelineOrchestrator:
    def __init__(
        self,
        chroma_dir: str | None = None,
        embedding_model: str | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> None:
        config = get_config()
        self.chunker = Chunker(
            chunk_size=chunk_size or config.chunking.default_chunk_size,
            overlap=overlap or config.chunking.overlap,
        )
        self.tagger = Tagger()
        self.embedder = Embedder(
            model_name=embedding_model or config.embedding.model,
        )
        self.store = ChunkStore(
            persist_dir=chroma_dir or config.storage.chroma_persist_dir,
        )

    def process(self, documents: list[Document]) -> dict:
        """Run documents through the full pipeline: chunk → tag → embed → store."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk(doc)
            chunks = [self.tagger.tag(c) for c in chunks]
            all_chunks.extend(chunks)

        if all_chunks:
            embedded = self.embedder.embed(all_chunks)
            self.store.store(embedded)

        return {
            "documents": len(documents),
            "chunks": len(all_chunks),
        }
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_orchestrator.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/casecrawler/pipeline/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: pipeline orchestrator tying chunk → tag → embed → store"
```

---

## Task 8: CLI

**Files:**
- Create: `src/casecrawler/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write CLI tests**

```python
# tests/test_cli.py
import tempfile
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from casecrawler.cli import cli
from casecrawler.models.document import (
    CredibilityLevel,
    Document,
    DocumentMetadata,
)


def _fake_doc(source: str = "pubmed", source_id: str = "1") -> Document:
    return Document(
        source=source,
        source_id=source_id,
        title="Test Article",
        content="Test content about hemorrhage.",
        content_type="abstract",
        metadata=DocumentMetadata(credibility=CredibilityLevel.PEER_REVIEWED),
    )


def test_cli_sources():
    runner = CliRunner()
    result = runner.invoke(cli, ["sources"])
    assert result.exit_code == 0
    # Should show at least the free sources
    assert "pubmed" in result.output.lower()


def test_cli_config():
    runner = CliRunner()
    result = runner.invoke(cli, ["config"])
    assert result.exit_code == 0
    assert "chunk_size" in result.output.lower() or "embedding" in result.output.lower()


def test_cli_ingest(tmp_path):
    runner = CliRunner()
    fake_search = AsyncMock(return_value=[_fake_doc()])

    with patch("casecrawler.cli.SourceRegistry") as MockRegistry, \
         patch("casecrawler.cli.PipelineOrchestrator") as MockPipeline:
        mock_reg = MockRegistry.return_value
        mock_reg.discover.return_value = None
        mock_source = AsyncMock()
        mock_source.name = "pubmed"
        mock_source.search = fake_search
        mock_reg.get_sources.return_value = [mock_source]

        mock_pipeline = MockPipeline.return_value
        mock_pipeline.process.return_value = {"documents": 1, "chunks": 3}

        result = runner.invoke(cli, ["ingest", "subarachnoid hemorrhage"])
        assert result.exit_code == 0
        assert "1" in result.output  # document count
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement CLI**

```python
# src/casecrawler/cli.py
from __future__ import annotations

import asyncio
import time

import click

from casecrawler.config import get_config, load_config
from casecrawler.pipeline.orchestrator import PipelineOrchestrator
from casecrawler.sources.registry import SourceRegistry

# Import all source modules so BaseSource.__subclasses__() discovers them
import casecrawler.sources.pubmed  # noqa: F401


@click.group()
@click.option("--config", "config_path", default=None, help="Path to config.yaml")
def cli(config_path: str | None) -> None:
    """CaseCrawler — Medical knowledge ingestion engine."""
    load_config(config_path)


@cli.command()
@click.argument("query")
@click.option("--sources", default=None, help="Comma-separated source names")
@click.option("--limit", default=None, type=int, help="Max results per source")
def ingest(query: str, sources: str | None, limit: int | None) -> None:
    """Ingest medical content for a topic from available sources."""
    config = get_config()
    limit = limit or config.ingestion.default_limit_per_source

    registry = SourceRegistry()
    registry.discover()

    source_names = sources.split(",") if sources else None
    active_sources = registry.get_sources(source_names)

    if not active_sources:
        click.echo("No sources available. Check your API keys with 'casecrawler sources'.")
        return

    click.echo(f"Ingesting '{query}' from {len(active_sources)} source(s)...")
    start = time.time()

    # Fan out searches in parallel
    all_docs = asyncio.run(_search_all(active_sources, query, limit))

    # Process through pipeline
    pipeline = PipelineOrchestrator()
    total_summary: dict[str, dict] = {}

    for source_name, docs in all_docs.items():
        if docs:
            result = pipeline.process(docs)
            total_summary[source_name] = result

    elapsed = time.time() - start

    # Print summary
    click.echo("\n--- Ingestion Summary ---")
    total_docs = 0
    total_chunks = 0
    for source_name, summary in total_summary.items():
        click.echo(f"  {source_name}: {summary['documents']} documents, {summary['chunks']} chunks")
        total_docs += summary["documents"]
        total_chunks += summary["chunks"]
    click.echo(f"\nTotal: {total_docs} documents, {total_chunks} chunks in {elapsed:.1f}s")


async def _search_all(
    sources: list, query: str, limit: int
) -> dict[str, list]:
    """Fan out search calls to all sources concurrently."""
    import asyncio

    async def _search_one(source):
        try:
            docs = await source.search(query, limit=limit)
            return source.name, docs
        except Exception as e:
            click.echo(f"  Warning: {source.name} failed: {e}")
            return source.name, []

    tasks = [_search_one(s) for s in sources]
    results = await asyncio.gather(*tasks)
    return dict(results)


@cli.command()
@click.argument("query")
@click.option("--source", default=None, help="Filter by source name")
@click.option("--limit", default=10, type=int, help="Max results")
def search(query: str, source: str | None, limit: int) -> None:
    """Search the knowledge base."""
    pipeline = PipelineOrchestrator()
    results = pipeline.store.search(query, n_results=limit, source=source)

    if not results:
        click.echo("No results found.")
        return

    for i, r in enumerate(results, 1):
        score = r["score"]
        meta = r["metadata"]
        text_preview = r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"]
        click.echo(f"\n[{i}] (score: {score:.3f}) [{meta.get('source', '?')}] [{meta.get('credibility', '?')}]")
        click.echo(f"    {text_preview}")


@cli.command()
def sources() -> None:
    """List available and unavailable data sources."""
    registry = SourceRegistry()
    info = registry.all_sources_info()

    available = [s for s in info if s["available"]]
    unavailable = [s for s in info if not s["available"]]

    click.echo("Available:")
    for s in available:
        keys_info = ", ".join(s["requires_keys"]) if s["requires_keys"] else "no key required"
        click.echo(f"  ✓ {s['name']:<18} ({keys_info})")

    if unavailable:
        click.echo("\nUnavailable:")
        for s in unavailable:
            missing = ", ".join(s.get("missing_keys", []))
            click.echo(f"  ✗ {s['name']:<18} (missing {missing})")


@cli.command("config")
def show_config() -> None:
    """Show current configuration."""
    config = get_config()
    click.echo(f"Ingestion limit per source: {config.ingestion.default_limit_per_source}")
    click.echo(f"Chunk size: {config.chunking.default_chunk_size}")
    click.echo(f"Chunk overlap: {config.chunking.overlap}")
    click.echo(f"Embedding model: {config.embedding.model}")
    click.echo(f"ChromaDB dir: {config.storage.chroma_persist_dir}")
    click.echo(f"API: {config.api.host}:{config.api.port}")


@cli.command()
def serve() -> None:
    """Start the FastAPI server."""
    import uvicorn

    config = get_config()
    uvicorn.run(
        "casecrawler.api.app:app",
        host=config.api.host,
        port=config.api.port,
        reload=True,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_cli.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Verify CLI works end-to-end**

Run: `casecrawler sources`
Expected: Shows `✓ pubmed (no key required)` and lists any unavailable sources.

Run: `casecrawler config`
Expected: Shows current config values.

- [ ] **Step 6: Commit**

```bash
git add src/casecrawler/cli.py tests/test_cli.py
git commit -m "feat: CLI with ingest, search, sources, config, and serve commands"
```

---

## Task 9: Free Source Plugins — OpenFDA + DailyMed

**Files:**
- Create: `src/casecrawler/sources/openfda.py`
- Create: `src/casecrawler/sources/dailymed.py`
- Test: `tests/test_openfda.py`
- Test: `tests/test_dailymed.py`

- [ ] **Step 1: Write OpenFDA tests**

```python
# tests/test_openfda.py
import pytest

from casecrawler.models.document import CredibilityLevel
from casecrawler.sources.openfda import OpenFDASource

LABEL_RESPONSE = {
    "meta": {"results": {"total": 1}},
    "results": [
        {
            "set_id": "abc-123",
            "openfda": {
                "brand_name": ["ADVIL"],
                "generic_name": ["IBUPROFEN"],
                "manufacturer_name": ["Pfizer"],
            },
            "indications_and_usage": ["For temporary relief of minor aches and pains."],
            "contraindications": ["Do not use if you have had an allergic reaction to ibuprofen."],
            "adverse_reactions": ["Nausea, dizziness, headache."],
            "dosage_and_administration": ["Adults: 200-400mg every 4-6 hours."],
            "drug_interactions": ["Avoid concurrent use with aspirin."],
            "warnings": ["Stomach bleeding warning."],
            "effective_time": "20240115",
        }
    ],
}


@pytest.fixture
def openfda():
    return OpenFDASource()


@pytest.mark.asyncio
async def test_openfda_search(openfda, httpx_mock):
    httpx_mock.add_response(
        url__startswith="https://api.fda.gov/drug/label.json",
        json=LABEL_RESPONSE,
    )
    docs = await openfda.search("ibuprofen", limit=5)
    assert len(docs) == 1
    assert docs[0].source == "openfda"
    assert docs[0].content_type == "drug_label"
    assert "INDICATIONS" in docs[0].content
    assert "aches and pains" in docs[0].content
    assert docs[0].metadata.credibility == CredibilityLevel.FDA_LABEL


def test_openfda_is_available():
    assert OpenFDASource.is_available() is True
```

- [ ] **Step 2: Write DailyMed tests**

```python
# tests/test_dailymed.py
import pytest

from casecrawler.models.document import CredibilityLevel
from casecrawler.sources.dailymed import DailyMedSource

SEARCH_RESPONSE = {
    "data": [
        {
            "setid": "23cb5001-6aeb-434a-b2e8-aaf8bbab9dc6",
            "title": "METFORMIN HYDROCHLORIDE tablet",
            "published_date": "Jan 15, 2024",
            "spl_version": 5,
        }
    ],
    "metadata": {"total_elements": 1},
}

SPL_XML = """<?xml version="1.0" encoding="UTF-8"?>
<document xmlns="urn:hl7-org:v3">
  <component>
    <structuredBody>
      <component>
        <section>
          <code code="34067-9"/>
          <title>INDICATIONS AND USAGE</title>
          <text>Metformin is indicated for type 2 diabetes.</text>
        </section>
      </component>
      <component>
        <section>
          <code code="34070-3"/>
          <title>CONTRAINDICATIONS</title>
          <text>Severe renal impairment.</text>
        </section>
      </component>
      <component>
        <section>
          <code code="34084-4"/>
          <title>ADVERSE REACTIONS</title>
          <text>Diarrhea, nausea, vomiting.</text>
        </section>
      </component>
    </structuredBody>
  </component>
</document>"""


@pytest.fixture
def dailymed():
    return DailyMedSource()


@pytest.mark.asyncio
async def test_dailymed_search(dailymed, httpx_mock):
    httpx_mock.add_response(
        url__startswith="https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json",
        json=SEARCH_RESPONSE,
    )
    httpx_mock.add_response(
        url__startswith="https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/23cb5001",
        text=SPL_XML,
        headers={"content-type": "application/xml"},
    )
    docs = await dailymed.search("metformin", limit=5)
    assert len(docs) == 1
    assert docs[0].source == "dailymed"
    assert docs[0].content_type == "drug_label"
    assert "type 2 diabetes" in docs[0].content
    assert "renal impairment" in docs[0].content
    assert docs[0].metadata.credibility == CredibilityLevel.FDA_LABEL


def test_dailymed_is_available():
    assert DailyMedSource.is_available() is True
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_openfda.py tests/test_dailymed.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement OpenFDA source**

```python
# src/casecrawler/sources/openfda.py
from __future__ import annotations

from datetime import date

import httpx

from casecrawler.config import get_env
from casecrawler.models.document import (
    CredibilityLevel,
    Document,
    DocumentMetadata,
)
from casecrawler.sources.base import BaseSource

LABEL_URL = "https://api.fda.gov/drug/label.json"


class OpenFDASource(BaseSource):
    name = "openfda"
    requires_keys: list[str] = []

    def _base_params(self) -> dict:
        params: dict = {}
        api_key = get_env("OPENFDA_API_KEY")
        if api_key:
            params["api_key"] = api_key
        return params

    async def search(self, query: str, limit: int = 20) -> list[Document]:
        async with httpx.AsyncClient() as client:
            params = {
                **self._base_params(),
                "search": f'openfda.generic_name:"{query}"+openfda.brand_name:"{query}"',
                "limit": str(min(limit, 100)),
            }
            resp = await client.get(LABEL_URL, params=params)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            data = resp.json()
            return [self._parse_label(r) for r in data.get("results", [])]

    async def fetch(self, document_id: str) -> Document:
        async with httpx.AsyncClient() as client:
            params = {
                **self._base_params(),
                "search": f'set_id:"{document_id}"',
                "limit": "1",
            }
            resp = await client.get(LABEL_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                raise ValueError(f"No label found for set_id {document_id}")
            return self._parse_label(results[0])

    def _parse_label(self, result: dict) -> Document:
        openfda = result.get("openfda", {})
        brand = openfda.get("brand_name", [""])[0] if openfda.get("brand_name") else ""
        generic = openfda.get("generic_name", [""])[0] if openfda.get("generic_name") else ""
        manufacturer = openfda.get("manufacturer_name", [""])[0] if openfda.get("manufacturer_name") else ""
        set_id = result.get("set_id", "")

        title = f"{brand} ({generic})" if brand and generic else brand or generic or set_id

        # Build content from label sections
        sections = []
        for field, header in [
            ("indications_and_usage", "INDICATIONS AND USAGE"),
            ("dosage_and_administration", "DOSAGE AND ADMINISTRATION"),
            ("contraindications", "CONTRAINDICATIONS"),
            ("warnings", "WARNINGS"),
            ("warnings_and_cautions", "WARNINGS AND PRECAUTIONS"),
            ("adverse_reactions", "ADVERSE REACTIONS"),
            ("drug_interactions", "DRUG INTERACTIONS"),
            ("boxed_warning", "BOXED WARNING"),
            ("overdosage", "OVERDOSAGE"),
        ]:
            values = result.get(field, [])
            if values:
                text = values[0] if isinstance(values, list) else values
                sections.append(f"{header}\n{text}")

        content = "\n\n".join(sections)

        # Parse date
        pub_date = None
        effective = result.get("effective_time", "")
        if len(effective) >= 8:
            try:
                pub_date = date(int(effective[:4]), int(effective[4:6]), int(effective[6:8]))
            except ValueError:
                pass

        metadata = DocumentMetadata(
            authors=[manufacturer] if manufacturer else [],
            publication_date=pub_date,
            specialty=[],
            credibility=CredibilityLevel.FDA_LABEL,
            url=f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}" if set_id else None,
        )

        return Document(
            source="openfda",
            source_id=set_id,
            title=title,
            content=content,
            content_type="drug_label",
            metadata=metadata,
        )
```

- [ ] **Step 5: Implement DailyMed source**

```python
# src/casecrawler/sources/dailymed.py
from __future__ import annotations

import xml.etree.ElementTree as ET

import httpx

from casecrawler.models.document import (
    CredibilityLevel,
    Document,
    DocumentMetadata,
)
from casecrawler.sources.base import BaseSource

BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

# LOINC codes for key drug label sections
SECTION_CODES = {
    "34067-9": "INDICATIONS AND USAGE",
    "34068-7": "DOSAGE AND ADMINISTRATION",
    "34070-3": "CONTRAINDICATIONS",
    "43685-7": "WARNINGS AND PRECAUTIONS",
    "34084-4": "ADVERSE REACTIONS",
    "34073-7": "DRUG INTERACTIONS",
    "34089-3": "DESCRIPTION",
    "34090-1": "CLINICAL PHARMACOLOGY",
    "34066-1": "BOXED WARNING",
    "34088-5": "OVERDOSAGE",
}

NS = {"hl7": "urn:hl7-org:v3"}


class DailyMedSource(BaseSource):
    name = "dailymed"
    requires_keys: list[str] = []

    async def search(self, query: str, limit: int = 20) -> list[Document]:
        async with httpx.AsyncClient() as client:
            # Step 1: Search for SPLs by drug name
            resp = await client.get(
                f"{BASE_URL}/spls.json",
                params={"drug_name": query, "pagesize": str(min(limit, 100)), "page": "1"},
            )
            resp.raise_for_status()
            data = resp.json()
            items = data.get("data", [])

            # Step 2: Fetch full SPL XML for each result
            documents = []
            for item in items:
                setid = item.get("setid", "")
                title = item.get("title", "")
                try:
                    doc = await self._fetch_spl(client, setid, title)
                    documents.append(doc)
                except Exception:
                    continue  # Skip SPLs that fail to parse

            return documents

    async def fetch(self, document_id: str) -> Document:
        async with httpx.AsyncClient() as client:
            return await self._fetch_spl(client, document_id, "")

    async def _fetch_spl(self, client: httpx.AsyncClient, setid: str, title: str) -> Document:
        resp = await client.get(f"{BASE_URL}/spls/{setid}.xml")
        resp.raise_for_status()
        content = self._parse_spl_xml(resp.text)

        metadata = DocumentMetadata(
            credibility=CredibilityLevel.FDA_LABEL,
            url=f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}",
        )

        return Document(
            source="dailymed",
            source_id=setid,
            title=title,
            content=content,
            content_type="drug_label",
            metadata=metadata,
        )

    def _parse_spl_xml(self, xml_text: str) -> str:
        root = ET.fromstring(xml_text)
        sections = []

        for section in root.iter("{urn:hl7-org:v3}section"):
            code_elem = section.find("hl7:code", NS)
            if code_elem is None:
                continue
            code = code_elem.get("code", "")
            header = SECTION_CODES.get(code)
            if header is None:
                continue

            # Extract all text content from the section
            text_elem = section.find("hl7:text", NS)
            if text_elem is not None:
                text = "".join(text_elem.itertext()).strip()
                if text:
                    sections.append(f"{header}\n{text}")

        return "\n\n".join(sections)
```

- [ ] **Step 6: Update CLI imports to register new sources**

```python
# In src/casecrawler/cli.py, add these imports near the top alongside pubmed:
import casecrawler.sources.openfda  # noqa: F401
import casecrawler.sources.dailymed  # noqa: F401
```

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_openfda.py tests/test_dailymed.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/casecrawler/sources/openfda.py src/casecrawler/sources/dailymed.py src/casecrawler/cli.py tests/test_openfda.py tests/test_dailymed.py
git commit -m "feat: OpenFDA and DailyMed source plugins"
```

---

## Task 10: Free Source Plugins — RxNorm, medRxiv, ClinicalTrials

**Files:**
- Create: `src/casecrawler/sources/rxnorm.py`
- Create: `src/casecrawler/sources/medrxiv.py`
- Create: `src/casecrawler/sources/clinicaltrials.py`
- Test: `tests/test_free_sources.py`

- [ ] **Step 1: Write tests for all three sources**

```python
# tests/test_free_sources.py
import pytest

from casecrawler.models.document import CredibilityLevel
from casecrawler.sources.rxnorm import RxNormSource
from casecrawler.sources.medrxiv import MedRxivSource
from casecrawler.sources.clinicaltrials import ClinicalTrialsSource


# --- RxNorm ---

RXNORM_DRUGS_RESPONSE = {
    "drugGroup": {
        "conceptGroup": [
            {
                "tty": "SBD",
                "conceptProperties": [
                    {
                        "rxcui": "596928",
                        "name": "duloxetine 20 MG Delayed Release Oral Capsule [Cymbalta]",
                        "tty": "SBD",
                    },
                    {
                        "rxcui": "596930",
                        "name": "duloxetine 30 MG Delayed Release Oral Capsule [Cymbalta]",
                        "tty": "SBD",
                    },
                ],
            }
        ]
    }
}


@pytest.fixture
def rxnorm():
    return RxNormSource()


@pytest.mark.asyncio
async def test_rxnorm_search(rxnorm, httpx_mock):
    httpx_mock.add_response(
        url__startswith="https://rxnav.nlm.nih.gov/REST/drugs.json",
        json=RXNORM_DRUGS_RESPONSE,
    )
    docs = await rxnorm.search("cymbalta", limit=5)
    assert len(docs) == 2
    assert docs[0].source == "rxnorm"
    assert docs[0].source_id == "596928"
    assert "duloxetine" in docs[0].content.lower()
    assert docs[0].metadata.credibility == CredibilityLevel.CURATED


# --- medRxiv ---

MEDRXIV_RESPONSE = {
    "messages": [{"status": "ok", "count": 1, "total": 1}],
    "collection": [
        {
            "doi": "10.1101/2024.01.01.123456",
            "title": "Novel Biomarkers in Subarachnoid Hemorrhage",
            "authors": "Smith, J.; Doe, A.",
            "date": "2024-01-15",
            "abstract": "We identified novel biomarkers for SAH prognosis.",
            "category": "neurology",
            "server": "medrxiv",
        }
    ],
}


@pytest.fixture
def medrxiv():
    return MedRxivSource()


@pytest.mark.asyncio
async def test_medrxiv_search(medrxiv, httpx_mock):
    httpx_mock.add_response(
        url__startswith="https://api.medrxiv.org/details/medrxiv",
        json=MEDRXIV_RESPONSE,
    )
    docs = await medrxiv.search("subarachnoid hemorrhage", limit=5)
    assert len(docs) == 1
    assert docs[0].source == "medrxiv"
    assert docs[0].metadata.credibility == CredibilityLevel.PREPRINT
    assert "biomarkers" in docs[0].content.lower()


# --- ClinicalTrials ---

CT_RESPONSE = {
    "studies": [
        {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT00000123",
                    "briefTitle": "Study of Drug X for SAH",
                    "officialTitle": "A Randomized Trial of Drug X in SAH Patients",
                },
                "statusModule": {
                    "overallStatus": "RECRUITING",
                },
                "eligibilityModule": {
                    "eligibilityCriteria": "Inclusion: Adults 18-65\nExclusion: Pregnancy",
                    "sex": "ALL",
                    "minimumAge": "18 Years",
                    "maximumAge": "65 Years",
                },
                "armsInterventionsModule": {
                    "interventions": [
                        {"type": "DRUG", "name": "Drug X", "description": "Experimental drug"},
                    ]
                },
                "outcomesModule": {
                    "primaryOutcomes": [
                        {"measure": "Mortality at 30 days", "timeFrame": "30 days"},
                    ]
                },
                "designModule": {
                    "studyType": "INTERVENTIONAL",
                    "phases": ["PHASE3"],
                },
            }
        }
    ],
    "nextPageToken": None,
}


@pytest.fixture
def ct():
    return ClinicalTrialsSource()


@pytest.mark.asyncio
async def test_clinicaltrials_search(ct, httpx_mock):
    httpx_mock.add_response(
        url__startswith="https://clinicaltrials.gov/api/v2/studies",
        json=CT_RESPONSE,
    )
    docs = await ct.search("subarachnoid hemorrhage", limit=5)
    assert len(docs) == 1
    assert docs[0].source == "clinicaltrials"
    assert docs[0].source_id == "NCT00000123"
    assert "Drug X" in docs[0].content
    assert "18-65" in docs[0].content or "18 Years" in docs[0].content
    assert docs[0].content_type == "trial_protocol"
    assert docs[0].metadata.credibility == CredibilityLevel.PEER_REVIEWED
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_free_sources.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement RxNorm source**

```python
# src/casecrawler/sources/rxnorm.py
from __future__ import annotations

import httpx

from casecrawler.models.document import (
    CredibilityLevel,
    Document,
    DocumentMetadata,
)
from casecrawler.sources.base import BaseSource

BASE_URL = "https://rxnav.nlm.nih.gov/REST"


class RxNormSource(BaseSource):
    name = "rxnorm"
    requires_keys: list[str] = []

    async def search(self, query: str, limit: int = 20) -> list[Document]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{BASE_URL}/drugs.json",
                params={"name": query},
            )
            resp.raise_for_status()
            data = resp.json()

            documents = []
            drug_group = data.get("drugGroup", {})
            for group in drug_group.get("conceptGroup", []):
                for prop in group.get("conceptProperties", []):
                    if len(documents) >= limit:
                        break
                    documents.append(self._parse_concept(prop))
            return documents

    async def fetch(self, document_id: str) -> Document:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{BASE_URL}/rxcui/{document_id}/properties.json",
            )
            resp.raise_for_status()
            data = resp.json()
            props = data.get("properties", {})
            return Document(
                source="rxnorm",
                source_id=document_id,
                title=props.get("name", ""),
                content=f"Drug: {props.get('name', '')}\nRxCUI: {document_id}\nTerm Type: {props.get('tty', '')}",
                content_type="drug_info",
                metadata=DocumentMetadata(
                    credibility=CredibilityLevel.CURATED,
                    url=f"https://mor.nlm.nih.gov/RxNav/search?searchBy=RXCUI&searchTerm={document_id}",
                ),
            )

    def _parse_concept(self, prop: dict) -> Document:
        rxcui = prop.get("rxcui", "")
        name = prop.get("name", "")
        tty = prop.get("tty", "")

        content = f"Drug: {name}\nRxCUI: {rxcui}\nTerm Type: {tty}"

        return Document(
            source="rxnorm",
            source_id=rxcui,
            title=name,
            content=content,
            content_type="drug_info",
            metadata=DocumentMetadata(
                credibility=CredibilityLevel.CURATED,
                url=f"https://mor.nlm.nih.gov/RxNav/search?searchBy=RXCUI&searchTerm={rxcui}",
            ),
        )
```

- [ ] **Step 4: Implement medRxiv source**

```python
# src/casecrawler/sources/medrxiv.py
from __future__ import annotations

from datetime import date, timedelta

import httpx

from casecrawler.models.document import (
    CredibilityLevel,
    Document,
    DocumentMetadata,
)
from casecrawler.sources.base import BaseSource

BASE_URL = "https://api.medrxiv.org"


class MedRxivSource(BaseSource):
    name = "medrxiv"
    requires_keys: list[str] = []

    async def search(self, query: str, limit: int = 20) -> list[Document]:
        """Fetch recent preprints and filter by query terms.

        Note: The medRxiv API has no keyword search endpoint.
        We fetch recent papers and filter client-side.
        """
        async with httpx.AsyncClient() as client:
            # Fetch recent papers (last 30 days)
            end_date = date.today()
            start_date = end_date - timedelta(days=30)
            url = f"{BASE_URL}/details/medrxiv/{start_date}/{end_date}/0/json"

            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

            collection = data.get("collection", [])
            query_lower = query.lower()
            query_terms = query_lower.split()

            # Filter: any query term appears in title or abstract
            matching = []
            for item in collection:
                title = item.get("title", "").lower()
                abstract = item.get("abstract", "").lower()
                if any(term in title or term in abstract for term in query_terms):
                    matching.append(self._parse_item(item))
                    if len(matching) >= limit:
                        break

            return matching

    async def fetch(self, document_id: str) -> Document:
        """Fetch a specific preprint by DOI."""
        async with httpx.AsyncClient() as client:
            url = f"{BASE_URL}/details/medrxiv/{document_id}/na/json"
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("collection", [])
            if not items:
                raise ValueError(f"No preprint found for DOI {document_id}")
            return self._parse_item(items[0])

    def _parse_item(self, item: dict) -> Document:
        doi = item.get("doi", "")
        title = item.get("title", "")
        abstract = item.get("abstract", "")
        authors_str = item.get("authors", "")
        date_str = item.get("date", "")
        category = item.get("category", "")

        authors = [a.strip() for a in authors_str.split(";") if a.strip()]

        pub_date = None
        if date_str:
            try:
                parts = date_str.split("-")
                pub_date = date(int(parts[0]), int(parts[1]), int(parts[2]))
            except (ValueError, IndexError):
                pass

        metadata = DocumentMetadata(
            authors=authors,
            publication_date=pub_date,
            specialty=[category] if category else [],
            credibility=CredibilityLevel.PREPRINT,
            url=f"https://www.medrxiv.org/content/{doi}",
            doi=doi,
        )

        return Document(
            source="medrxiv",
            source_id=doi,
            title=title,
            content=abstract,
            content_type="abstract",
            metadata=metadata,
        )
```

- [ ] **Step 5: Implement ClinicalTrials source**

```python
# src/casecrawler/sources/clinicaltrials.py
from __future__ import annotations

import httpx

from casecrawler.models.document import (
    CredibilityLevel,
    Document,
    DocumentMetadata,
)
from casecrawler.sources.base import BaseSource

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


class ClinicalTrialsSource(BaseSource):
    name = "clinicaltrials"
    requires_keys: list[str] = []

    async def search(self, query: str, limit: int = 20) -> list[Document]:
        async with httpx.AsyncClient() as client:
            params = {
                "query.cond": query,
                "pageSize": str(min(limit, 100)),
                "countTotal": "true",
                "format": "json",
            }
            resp = await client.get(BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            return [self._parse_study(s) for s in data.get("studies", [])]

    async def fetch(self, document_id: str) -> Document:
        """Fetch a single study by NCT ID."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{BASE_URL}/{document_id}", params={"format": "json"})
            resp.raise_for_status()
            data = resp.json()
            return self._parse_study(data)

    def _parse_study(self, study: dict) -> Document:
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status = proto.get("statusModule", {})
        eligibility = proto.get("eligibilityModule", {})
        arms = proto.get("armsInterventionsModule", {})
        outcomes = proto.get("outcomesModule", {})
        design = proto.get("designModule", {})

        nct_id = ident.get("nctId", "")
        title = ident.get("briefTitle", "") or ident.get("officialTitle", "")

        # Build structured content
        sections = []

        sections.append(f"Title: {title}")
        sections.append(f"NCT ID: {nct_id}")
        sections.append(f"Status: {status.get('overallStatus', 'Unknown')}")

        phases = design.get("phases", [])
        if phases:
            sections.append(f"Phase: {', '.join(phases)}")

        criteria = eligibility.get("eligibilityCriteria", "")
        if criteria:
            age_range = ""
            min_age = eligibility.get("minimumAge", "")
            max_age = eligibility.get("maximumAge", "")
            if min_age or max_age:
                age_range = f" (Age: {min_age} - {max_age})"
            sections.append(f"Eligibility{age_range}:\n{criteria}")

        interventions = arms.get("interventions", [])
        if interventions:
            interv_text = "\n".join(
                f"- {i.get('type', '')}: {i.get('name', '')} — {i.get('description', '')}"
                for i in interventions
            )
            sections.append(f"Interventions:\n{interv_text}")

        primary = outcomes.get("primaryOutcomes", [])
        if primary:
            out_text = "\n".join(
                f"- {o.get('measure', '')} ({o.get('timeFrame', '')})" for o in primary
            )
            sections.append(f"Primary Outcomes:\n{out_text}")

        secondary = outcomes.get("secondaryOutcomes", [])
        if secondary:
            out_text = "\n".join(
                f"- {o.get('measure', '')} ({o.get('timeFrame', '')})" for o in secondary
            )
            sections.append(f"Secondary Outcomes:\n{out_text}")

        content = "\n\n".join(sections)

        metadata = DocumentMetadata(
            credibility=CredibilityLevel.PEER_REVIEWED,
            url=f"https://clinicaltrials.gov/study/{nct_id}",
        )

        return Document(
            source="clinicaltrials",
            source_id=nct_id,
            title=title,
            content=content,
            content_type="trial_protocol",
            metadata=metadata,
        )
```

- [ ] **Step 6: Update CLI imports**

Add to `src/casecrawler/cli.py` alongside other source imports:

```python
import casecrawler.sources.rxnorm  # noqa: F401
import casecrawler.sources.medrxiv  # noqa: F401
import casecrawler.sources.clinicaltrials  # noqa: F401
```

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_free_sources.py -v`
Expected: All 3 tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/casecrawler/sources/rxnorm.py src/casecrawler/sources/medrxiv.py src/casecrawler/sources/clinicaltrials.py src/casecrawler/cli.py tests/test_free_sources.py
git commit -m "feat: RxNorm, medRxiv, and ClinicalTrials.gov source plugins"
```

---

## Task 11: Paid Source Plugins — Glass, Anna's Archive, Firecrawl

**Files:**
- Create: `src/casecrawler/sources/glass.py`
- Create: `src/casecrawler/sources/annas_archive.py`
- Create: `src/casecrawler/sources/firecrawl.py`
- Test: `tests/test_paid_sources.py`

Note: These sources require API keys. Tests use mocked HTTP and patched env vars. The actual API shapes may need adjustment once you have access to the real APIs — the Glass Health and Anna's Archive APIs are not publicly documented. These implementations provide the correct plugin structure and will work once the actual API endpoints and response formats are confirmed.

- [ ] **Step 1: Write tests**

```python
# tests/test_paid_sources.py
import os
from unittest.mock import patch

import pytest

from casecrawler.models.document import CredibilityLevel
from casecrawler.sources.glass import GlassHealthSource
from casecrawler.sources.annas_archive import AnnasArchiveSource
from casecrawler.sources.firecrawl import FirecrawlSource


# --- Glass Health ---


def test_glass_requires_key():
    assert GlassHealthSource.requires_keys == ["GLASS_API_KEY"]


def test_glass_unavailable_without_key():
    with patch.dict(os.environ, {}, clear=True):
        assert GlassHealthSource.is_available() is False


def test_glass_available_with_key():
    with patch.dict(os.environ, {"GLASS_API_KEY": "test-key"}):
        assert GlassHealthSource.is_available() is True


@pytest.mark.asyncio
async def test_glass_search(httpx_mock):
    with patch.dict(os.environ, {"GLASS_API_KEY": "test-key"}):
        httpx_mock.add_response(
            url__startswith="https://glass.health",
            json={
                "results": [
                    {
                        "id": "glass-123",
                        "title": "Subarachnoid Hemorrhage",
                        "content": "SAH is a type of stroke caused by bleeding into the subarachnoid space.",
                        "category": "neurosurgery",
                    }
                ]
            },
        )
        source = GlassHealthSource()
        docs = await source.search("subarachnoid hemorrhage", limit=5)
        assert len(docs) == 1
        assert docs[0].source == "glass"
        assert docs[0].metadata.credibility == CredibilityLevel.CURATED


# --- Anna's Archive ---


def test_annas_requires_key():
    assert AnnasArchiveSource.requires_keys == ["ANNAS_ARCHIVE_API_KEY"]


def test_annas_unavailable_without_key():
    with patch.dict(os.environ, {}, clear=True):
        assert AnnasArchiveSource.is_available() is False


@pytest.mark.asyncio
async def test_annas_search(httpx_mock):
    with patch.dict(os.environ, {"ANNAS_ARCHIVE_API_KEY": "test-key"}):
        httpx_mock.add_response(
            json={
                "results": [
                    {
                        "id": "md5:abc123",
                        "title": "Principles of Neurosurgery",
                        "author": "Smith J",
                        "content": "Chapter 12: Management of SAH...",
                    }
                ]
            },
        )
        source = AnnasArchiveSource()
        docs = await source.search("subarachnoid hemorrhage", limit=5)
        assert len(docs) == 1
        assert docs[0].source == "annas_archive"


# --- Firecrawl ---


def test_firecrawl_requires_key():
    assert FirecrawlSource.requires_keys == ["FIRECRAWL_API_KEY"]


def test_firecrawl_unavailable_without_key():
    with patch.dict(os.environ, {}, clear=True):
        assert FirecrawlSource.is_available() is False


@pytest.mark.asyncio
async def test_firecrawl_search(httpx_mock):
    with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "test-key"}):
        httpx_mock.add_response(
            json={
                "data": [
                    {
                        "url": "https://example.com/guideline",
                        "title": "SAH Management Guideline",
                        "markdown": "# SAH Guidelines\n\nManagement of SAH includes...",
                    }
                ]
            },
        )
        source = FirecrawlSource()
        docs = await source.search("subarachnoid hemorrhage guidelines", limit=5)
        assert len(docs) == 1
        assert docs[0].source == "firecrawl"
        assert docs[0].content_type == "full_text"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_paid_sources.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement Glass Health source**

```python
# src/casecrawler/sources/glass.py
from __future__ import annotations

import httpx

from casecrawler.config import get_env
from casecrawler.models.document import (
    CredibilityLevel,
    Document,
    DocumentMetadata,
)
from casecrawler.sources.base import BaseSource

# Note: Glass Health API URL and response format are based on their developer docs.
# Adjust once you have confirmed API access.
BASE_URL = "https://glass.health/api/v1"


class GlassHealthSource(BaseSource):
    name = "glass"
    requires_keys: list[str] = ["GLASS_API_KEY"]

    async def search(self, query: str, limit: int = 20) -> list[Document]:
        api_key = get_env("GLASS_API_KEY")
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{BASE_URL}/search",
                params={"q": query, "limit": str(limit)},
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            return [self._parse_result(r) for r in data.get("results", [])]

    async def fetch(self, document_id: str) -> Document:
        api_key = get_env("GLASS_API_KEY")
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{BASE_URL}/content/{document_id}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            return self._parse_result(resp.json())

    def _parse_result(self, result: dict) -> Document:
        return Document(
            source="glass",
            source_id=result.get("id", ""),
            title=result.get("title", ""),
            content=result.get("content", ""),
            content_type="curated",
            metadata=DocumentMetadata(
                specialty=[result["category"]] if result.get("category") else [],
                credibility=CredibilityLevel.CURATED,
                url=result.get("url"),
            ),
        )
```

- [ ] **Step 4: Implement Anna's Archive source**

```python
# src/casecrawler/sources/annas_archive.py
from __future__ import annotations

import httpx

from casecrawler.config import get_env
from casecrawler.models.document import (
    CredibilityLevel,
    Document,
    DocumentMetadata,
)
from casecrawler.sources.base import BaseSource

# Note: Anna's Archive API endpoint and response format should be confirmed
# against their actual API documentation once you have access.
BASE_URL = "https://annas-archive.gl/api/v1"


class AnnasArchiveSource(BaseSource):
    name = "annas_archive"
    requires_keys: list[str] = ["ANNAS_ARCHIVE_API_KEY"]

    async def search(self, query: str, limit: int = 20) -> list[Document]:
        api_key = get_env("ANNAS_ARCHIVE_API_KEY")
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{BASE_URL}/search",
                params={"q": query, "limit": str(limit), "content": "scidb"},
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            return [self._parse_result(r) for r in data.get("results", [])]

    async def fetch(self, document_id: str) -> Document:
        api_key = get_env("ANNAS_ARCHIVE_API_KEY")
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{BASE_URL}/content/{document_id}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            return self._parse_result(resp.json())

    def _parse_result(self, result: dict) -> Document:
        authors_raw = result.get("author", "")
        authors = [a.strip() for a in authors_raw.split(";") if a.strip()] if authors_raw else []

        return Document(
            source="annas_archive",
            source_id=result.get("id", ""),
            title=result.get("title", ""),
            content=result.get("content", ""),
            content_type="full_text",
            metadata=DocumentMetadata(
                authors=authors,
                credibility=CredibilityLevel.PEER_REVIEWED,
                doi=result.get("doi"),
            ),
        )
```

- [ ] **Step 5: Implement Firecrawl source**

```python
# src/casecrawler/sources/firecrawl.py
from __future__ import annotations

import httpx

from casecrawler.config import get_env
from casecrawler.models.document import (
    CredibilityLevel,
    Document,
    DocumentMetadata,
)
from casecrawler.sources.base import BaseSource

BASE_URL = "https://api.firecrawl.dev/v1"


class FirecrawlSource(BaseSource):
    name = "firecrawl"
    requires_keys: list[str] = ["FIRECRAWL_API_KEY"]

    async def search(self, query: str, limit: int = 20) -> list[Document]:
        api_key = get_env("FIRECRAWL_API_KEY")
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BASE_URL}/search",
                json={"query": query, "limit": limit},
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            return [self._parse_result(r) for r in data.get("data", [])]

    async def fetch(self, document_id: str) -> Document:
        """Scrape a specific URL."""
        api_key = get_env("FIRECRAWL_API_KEY")
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BASE_URL}/scrape",
                json={"url": document_id, "formats": ["markdown"]},
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
            return Document(
                source="firecrawl",
                source_id=document_id,
                title=data.get("title", document_id),
                content=data.get("markdown", ""),
                content_type="full_text",
                metadata=DocumentMetadata(
                    credibility=CredibilityLevel.PEER_REVIEWED,
                    url=document_id,
                ),
            )

    def _parse_result(self, result: dict) -> Document:
        url = result.get("url", "")
        return Document(
            source="firecrawl",
            source_id=url,
            title=result.get("title", url),
            content=result.get("markdown", ""),
            content_type="full_text",
            metadata=DocumentMetadata(
                credibility=CredibilityLevel.PEER_REVIEWED,
                url=url,
            ),
        )
```

- [ ] **Step 6: Update CLI imports**

Add to `src/casecrawler/cli.py`:

```python
import casecrawler.sources.glass  # noqa: F401
import casecrawler.sources.annas_archive  # noqa: F401
import casecrawler.sources.firecrawl  # noqa: F401
```

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_paid_sources.py -v`
Expected: All 8 tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/casecrawler/sources/glass.py src/casecrawler/sources/annas_archive.py src/casecrawler/sources/firecrawl.py src/casecrawler/cli.py tests/test_paid_sources.py
git commit -m "feat: paid source plugins — Glass Health, Anna's Archive, Firecrawl"
```

---

## Task 12: REST API

**Files:**
- Create: `src/casecrawler/api/app.py`
- Create: `src/casecrawler/api/routes/ingest.py`
- Create: `src/casecrawler/api/routes/search.py`
- Create: `src/casecrawler/api/routes/sources.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write API tests**

```python
# tests/test_api.py
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from casecrawler.api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_sources_endpoint(client):
    resp = client.get("/api/sources")
    assert resp.status_code == 200
    data = resp.json()
    assert "available" in data
    assert "unavailable" in data
    # pubmed should always be available
    names = [s["name"] for s in data["available"]]
    assert "pubmed" in names


def test_ingest_endpoint(client):
    with patch("casecrawler.api.routes.ingest.run_ingestion") as mock_run:
        mock_run.return_value = None
        resp = client.post("/api/ingest", json={"query": "test topic"})
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "running"


def test_ingest_status_not_found(client):
    resp = client.get("/api/ingest/nonexistent-id")
    assert resp.status_code == 404


def test_search_endpoint(client):
    with patch("casecrawler.api.routes.search.get_store") as mock_store:
        mock_instance = MagicMock()
        mock_instance.search.return_value = [
            {
                "chunk_id": "abc",
                "text": "Test result text.",
                "metadata": {"source": "pubmed", "credibility": "peer_reviewed"},
                "score": 0.95,
            }
        ]
        mock_store.return_value = mock_instance
        resp = client.get("/api/search", params={"q": "test query"})
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["score"] == 0.95
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement FastAPI app**

```python
# src/casecrawler/api/app.py
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from casecrawler.api.routes import ingest, search, sources
from casecrawler.config import load_config

# Import sources for registry discovery
import casecrawler.sources.pubmed  # noqa: F401
import casecrawler.sources.openfda  # noqa: F401
import casecrawler.sources.dailymed  # noqa: F401
import casecrawler.sources.rxnorm  # noqa: F401
import casecrawler.sources.medrxiv  # noqa: F401
import casecrawler.sources.clinicaltrials  # noqa: F401
import casecrawler.sources.glass  # noqa: F401
import casecrawler.sources.annas_archive  # noqa: F401
import casecrawler.sources.firecrawl  # noqa: F401


def create_app() -> FastAPI:
    load_config()

    app = FastAPI(
        title="CaseCrawler",
        description="Medical knowledge ingestion engine",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(ingest.router, prefix="/api")
    app.include_router(search.router, prefix="/api")
    app.include_router(sources.router, prefix="/api")

    return app


app = create_app()
```

- [ ] **Step 4: Implement ingest routes**

```python
# src/casecrawler/api/routes/ingest.py
from __future__ import annotations

import asyncio
import time
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from casecrawler.config import get_config
from casecrawler.pipeline.orchestrator import PipelineOrchestrator
from casecrawler.sources.registry import SourceRegistry

router = APIRouter()

# In-memory job store (sufficient for local single-process use)
_jobs: dict[str, dict] = {}


class IngestRequest(BaseModel):
    query: str
    sources: list[str] | None = None
    limit: int | None = None


class IngestResponse(BaseModel):
    job_id: str
    status: str


@router.post("/ingest", response_model=IngestResponse, status_code=202)
async def start_ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running", "summary": {}, "elapsed_seconds": 0}
    background_tasks.add_task(run_ingestion, job_id, req.query, req.sources, req.limit)
    return IngestResponse(job_id=job_id, status="running")


@router.get("/ingest/{job_id}")
async def get_ingest_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **_jobs[job_id]}


async def run_ingestion(
    job_id: str,
    query: str,
    source_names: list[str] | None,
    limit: int | None,
) -> None:
    config = get_config()
    limit = limit or config.ingestion.default_limit_per_source
    start = time.time()

    try:
        registry = SourceRegistry()
        registry.discover()
        active_sources = registry.get_sources(source_names)

        # Fan out searches
        async def search_one(source):
            try:
                return source.name, await source.search(query, limit=limit)
            except Exception:
                return source.name, []

        results = await asyncio.gather(*[search_one(s) for s in active_sources])

        pipeline = PipelineOrchestrator()
        summary = {}
        for source_name, docs in results:
            if docs:
                result = pipeline.process(docs)
                summary[source_name] = result

        _jobs[job_id] = {
            "status": "completed",
            "summary": summary,
            "elapsed_seconds": round(time.time() - start, 1),
        }
    except Exception as e:
        _jobs[job_id] = {
            "status": "failed",
            "error": str(e),
            "elapsed_seconds": round(time.time() - start, 1),
        }
```

- [ ] **Step 5: Implement search routes**

```python
# src/casecrawler/api/routes/search.py
from __future__ import annotations

from fastapi import APIRouter, Query

from casecrawler.pipeline.store import ChunkStore
from casecrawler.config import get_config

router = APIRouter()


def get_store() -> ChunkStore:
    config = get_config()
    return ChunkStore(persist_dir=config.storage.chroma_persist_dir)


@router.get("/search")
async def search_chunks(
    q: str = Query(..., description="Search query"),
    source: str | None = Query(None, description="Filter by source name"),
    limit: int = Query(10, description="Max results", le=100),
):
    store = get_store()
    results = store.search(q, n_results=limit, source=source)
    return {"results": results}
```

- [ ] **Step 6: Implement sources routes**

```python
# src/casecrawler/api/routes/sources.py
from __future__ import annotations

from fastapi import APIRouter

from casecrawler.sources.registry import SourceRegistry

router = APIRouter()


@router.get("/sources")
async def list_sources():
    registry = SourceRegistry()
    info = registry.all_sources_info()
    available = [s for s in info if s["available"]]
    unavailable = [s for s in info if not s["available"]]
    return {"available": available, "unavailable": unavailable}
```

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_api.py -v`
Expected: All 4 tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/casecrawler/api/ tests/test_api.py
git commit -m "feat: REST API with ingest, search, and sources endpoints"
```

---

## Task 13: Web UI

**Files:**
- Create: `ui/package.json`
- Create: `ui/tsconfig.json`
- Create: `ui/vite.config.ts`
- Create: `ui/tailwind.config.js`
- Create: `ui/index.html`
- Create: `ui/src/main.tsx`
- Create: `ui/src/App.tsx`
- Create: `ui/src/api/client.ts`
- Create: `ui/src/pages/IngestPage.tsx`
- Create: `ui/src/pages/SearchPage.tsx`
- Create: `ui/src/pages/SourcesPage.tsx`
- Create: `ui/src/components/SourceBadge.tsx`
- Create: `ui/src/components/ChunkCard.tsx`
- Create: `ui/src/components/JobProgress.tsx`
- Create: `ui/src/index.css`

- [ ] **Step 1: Initialize the React project**

Run:
```bash
cd ui && npm create vite@latest . -- --template react-ts
npm install react-router-dom @tanstack/react-query
npm install -D tailwindcss @tailwindcss/vite
```

- [ ] **Step 2: Configure Vite with Tailwind and API proxy**

```typescript
// ui/vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 3000,
    proxy: {
      "/api": {
        target: process.env.VITE_API_URL || "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
```

- [ ] **Step 3: Set up Tailwind CSS**

```css
/* ui/src/index.css */
@import "tailwindcss";
```

- [ ] **Step 4: Create API client**

```typescript
// ui/src/api/client.ts
const BASE = "/api";

export interface SourceInfo {
  name: string;
  requires_keys: string[];
  available: boolean;
  missing_keys?: string[];
}

export interface SourcesResponse {
  available: SourceInfo[];
  unavailable: SourceInfo[];
}

export interface IngestRequest {
  query: string;
  sources?: string[];
  limit?: number;
}

export interface IngestJobResponse {
  job_id: string;
  status: string;
  summary?: Record<string, { documents: number; chunks: number }>;
  elapsed_seconds?: number;
  error?: string;
}

export interface SearchResult {
  chunk_id: string;
  text: string;
  metadata: Record<string, string>;
  score: number;
}

export interface SearchResponse {
  results: SearchResult[];
}

export async function fetchSources(): Promise<SourcesResponse> {
  const resp = await fetch(`${BASE}/sources`);
  return resp.json();
}

export async function startIngest(req: IngestRequest): Promise<IngestJobResponse> {
  const resp = await fetch(`${BASE}/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  return resp.json();
}

export async function getIngestStatus(jobId: string): Promise<IngestJobResponse> {
  const resp = await fetch(`${BASE}/ingest/${jobId}`);
  return resp.json();
}

export async function searchChunks(
  q: string,
  source?: string,
  limit = 10
): Promise<SearchResponse> {
  const params = new URLSearchParams({ q, limit: String(limit) });
  if (source) params.set("source", source);
  const resp = await fetch(`${BASE}/search?${params}`);
  return resp.json();
}
```

- [ ] **Step 5: Create shared components**

```tsx
// ui/src/components/SourceBadge.tsx
const CREDIBILITY_COLORS: Record<string, string> = {
  guideline: "bg-green-100 text-green-800",
  peer_reviewed: "bg-blue-100 text-blue-800",
  preprint: "bg-yellow-100 text-yellow-800",
  curated: "bg-purple-100 text-purple-800",
  fda_label: "bg-red-100 text-red-800",
};

export function SourceBadge({ source, credibility }: { source: string; credibility?: string }) {
  const colorClass = credibility ? CREDIBILITY_COLORS[credibility] || "bg-gray-100 text-gray-800" : "bg-gray-100 text-gray-800";
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${colorClass}`}>
      {source}
      {credibility && <span className="opacity-60">({credibility})</span>}
    </span>
  );
}
```

```tsx
// ui/src/components/ChunkCard.tsx
import { useState } from "react";
import type { SearchResult } from "../api/client";
import { SourceBadge } from "./SourceBadge";

export function ChunkCard({ result }: { result: SearchResult }) {
  const [expanded, setExpanded] = useState(false);
  const preview = result.text.length > 200 ? result.text.slice(0, 200) + "..." : result.text;

  return (
    <div
      className="rounded-lg border border-gray-200 p-4 hover:border-gray-300 cursor-pointer"
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-center justify-between mb-2">
        <SourceBadge source={result.metadata.source || "?"} credibility={result.metadata.credibility} />
        <span className="text-xs text-gray-500">score: {result.score.toFixed(3)}</span>
      </div>
      <p className="text-sm text-gray-700 whitespace-pre-wrap">{expanded ? result.text : preview}</p>
      {result.metadata.url && (
        <a
          href={result.metadata.url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-blue-600 hover:underline mt-2 inline-block"
          onClick={(e) => e.stopPropagation()}
        >
          View source
        </a>
      )}
    </div>
  );
}
```

```tsx
// ui/src/components/JobProgress.tsx
import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { getIngestStatus } from "../api/client";
import type { IngestJobResponse } from "../api/client";

export function JobProgress({ jobId, onComplete }: { jobId: string; onComplete: (job: IngestJobResponse) => void }) {
  const { data } = useQuery({
    queryKey: ["ingest-status", jobId],
    queryFn: () => getIngestStatus(jobId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "running" ? 1000 : false;
    },
  });

  useEffect(() => {
    if (data && data.status !== "running") {
      onComplete(data);
    }
  }, [data, onComplete]);

  if (!data) return <div className="text-sm text-gray-500">Starting...</div>;

  if (data.status === "running") {
    return (
      <div className="flex items-center gap-2">
        <div className="h-2 w-32 bg-gray-200 rounded-full overflow-hidden">
          <div className="h-full bg-blue-500 rounded-full animate-pulse w-2/3" />
        </div>
        <span className="text-sm text-gray-600">Ingesting...</span>
      </div>
    );
  }

  if (data.status === "failed") {
    return <div className="text-sm text-red-600">Failed: {data.error}</div>;
  }

  return (
    <div className="text-sm text-green-700">
      Done in {data.elapsed_seconds}s.
      {data.summary && Object.entries(data.summary).map(([src, s]) => (
        <span key={src} className="ml-2">{src}: {s.documents} docs, {s.chunks} chunks</span>
      ))}
    </div>
  );
}
```

- [ ] **Step 6: Create pages**

```tsx
// ui/src/pages/IngestPage.tsx
import { useState, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchSources, startIngest } from "../api/client";
import type { IngestJobResponse } from "../api/client";
import { JobProgress } from "../components/JobProgress";

export function IngestPage() {
  const [query, setQuery] = useState("");
  const [selectedSources, setSelectedSources] = useState<string[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<IngestJobResponse | null>(null);

  const { data: sourcesData } = useQuery({ queryKey: ["sources"], queryFn: fetchSources });

  const handleIngest = async () => {
    if (!query.trim()) return;
    setLastResult(null);
    const resp = await startIngest({
      query: query.trim(),
      sources: selectedSources.length > 0 ? selectedSources : undefined,
    });
    setJobId(resp.job_id);
  };

  const toggleSource = (name: string) => {
    setSelectedSources((prev) =>
      prev.includes(name) ? prev.filter((s) => s !== name) : [...prev, name]
    );
  };

  const handleComplete = useCallback((job: IngestJobResponse) => {
    setLastResult(job);
    setJobId(null);
  }, []);

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold">Ingest</h1>

      <div className="flex gap-2">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g. subarachnoid hemorrhage"
          className="flex-1 rounded-lg border border-gray-300 px-4 py-2 focus:border-blue-500 focus:outline-none"
          onKeyDown={(e) => e.key === "Enter" && handleIngest()}
        />
        <button
          onClick={handleIngest}
          disabled={!query.trim() || !!jobId}
          className="rounded-lg bg-blue-600 px-6 py-2 text-white hover:bg-blue-700 disabled:opacity-50"
        >
          Ingest
        </button>
      </div>

      {sourcesData && (
        <div className="space-y-2">
          <p className="text-sm font-medium text-gray-700">Sources:</p>
          <div className="flex flex-wrap gap-2">
            {sourcesData.available.map((s) => (
              <label key={s.name} className="flex items-center gap-1 text-sm">
                <input
                  type="checkbox"
                  checked={selectedSources.length === 0 || selectedSources.includes(s.name)}
                  onChange={() => toggleSource(s.name)}
                />
                {s.name}
              </label>
            ))}
            {sourcesData.unavailable.map((s) => (
              <span key={s.name} className="text-sm text-gray-400" title={`Missing: ${s.missing_keys?.join(", ")}`}>
                {s.name} (unavailable)
              </span>
            ))}
          </div>
        </div>
      )}

      {jobId && <JobProgress jobId={jobId} onComplete={handleComplete} />}

      {lastResult && lastResult.status === "completed" && lastResult.summary && (
        <div className="rounded-lg bg-green-50 border border-green-200 p-4">
          <p className="font-medium text-green-800">Ingestion complete ({lastResult.elapsed_seconds}s)</p>
          {Object.entries(lastResult.summary).map(([src, s]) => (
            <p key={src} className="text-sm text-green-700">{src}: {s.documents} documents, {s.chunks} chunks</p>
          ))}
        </div>
      )}
    </div>
  );
}
```

```tsx
// ui/src/pages/SearchPage.tsx
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { searchChunks } from "../api/client";
import { ChunkCard } from "../components/ChunkCard";

export function SearchPage() {
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [sourceFilter, setSourceFilter] = useState<string>("");

  const { data, isLoading } = useQuery({
    queryKey: ["search", submitted, sourceFilter],
    queryFn: () => searchChunks(submitted, sourceFilter || undefined, 20),
    enabled: !!submitted,
  });

  const handleSearch = () => {
    if (query.trim()) setSubmitted(query.trim());
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold">Search</h1>

      <div className="flex gap-2">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search the knowledge base..."
          className="flex-1 rounded-lg border border-gray-300 px-4 py-2 focus:border-blue-500 focus:outline-none"
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
        />
        <input
          type="text"
          value={sourceFilter}
          onChange={(e) => setSourceFilter(e.target.value)}
          placeholder="source filter"
          className="w-36 rounded-lg border border-gray-300 px-3 py-2 text-sm"
        />
        <button
          onClick={handleSearch}
          className="rounded-lg bg-blue-600 px-6 py-2 text-white hover:bg-blue-700"
        >
          Search
        </button>
      </div>

      {isLoading && <p className="text-sm text-gray-500">Searching...</p>}

      {data && (
        <div className="space-y-3">
          <p className="text-sm text-gray-500">{data.results.length} result(s)</p>
          {data.results.map((r) => (
            <ChunkCard key={r.chunk_id} result={r} />
          ))}
          {data.results.length === 0 && <p className="text-gray-500">No results found.</p>}
        </div>
      )}
    </div>
  );
}
```

```tsx
// ui/src/pages/SourcesPage.tsx
import { useQuery } from "@tanstack/react-query";
import { fetchSources } from "../api/client";

export function SourcesPage() {
  const { data, isLoading } = useQuery({ queryKey: ["sources"], queryFn: fetchSources });

  if (isLoading) return <p>Loading...</p>;
  if (!data) return null;

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold">Sources</h1>

      <div>
        <h2 className="text-lg font-semibold text-green-700 mb-2">Available</h2>
        <div className="space-y-1">
          {data.available.map((s) => (
            <div key={s.name} className="flex items-center gap-2 text-sm">
              <span className="text-green-600">&#10003;</span>
              <span className="font-medium">{s.name}</span>
              <span className="text-gray-500">
                ({s.requires_keys.length ? s.requires_keys.join(", ") + " set" : "no key required"})
              </span>
            </div>
          ))}
        </div>
      </div>

      {data.unavailable.length > 0 && (
        <div>
          <h2 className="text-lg font-semibold text-red-700 mb-2">Unavailable</h2>
          <div className="space-y-1">
            {data.unavailable.map((s) => (
              <div key={s.name} className="flex items-center gap-2 text-sm">
                <span className="text-red-500">&#10007;</span>
                <span className="font-medium">{s.name}</span>
                <span className="text-gray-500">(missing {s.missing_keys?.join(", ")})</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 7: Wire up App with router**

```tsx
// ui/src/App.tsx
import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { IngestPage } from "./pages/IngestPage";
import { SearchPage } from "./pages/SearchPage";
import { SourcesPage } from "./pages/SourcesPage";

const queryClient = new QueryClient();

function Nav() {
  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `px-3 py-1 rounded text-sm ${isActive ? "bg-blue-100 text-blue-800 font-medium" : "text-gray-600 hover:text-gray-900"}`;

  return (
    <nav className="border-b border-gray-200 px-6 py-3 flex items-center gap-4">
      <span className="font-bold text-lg mr-4">CaseCrawler</span>
      <NavLink to="/" className={linkClass}>Ingest</NavLink>
      <NavLink to="/search" className={linkClass}>Search</NavLink>
      <NavLink to="/sources" className={linkClass}>Sources</NavLink>
    </nav>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Nav />
        <main className="p-6">
          <Routes>
            <Route path="/" element={<IngestPage />} />
            <Route path="/search" element={<SearchPage />} />
            <Route path="/sources" element={<SourcesPage />} />
          </Routes>
        </main>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
```

```tsx
// ui/src/main.tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
```

- [ ] **Step 8: Verify UI builds**

Run:
```bash
cd ui && npm run build
```
Expected: Build completes without errors.

- [ ] **Step 9: Commit**

```bash
git add ui/
git commit -m "feat: React web UI with ingest, search, and sources pages"
```

---

## Task 14: Docker + Final Config

**Files:**
- Create: `Dockerfile`
- Create: `ui/Dockerfile`
- Create: `docker-compose.yml`

- [ ] **Step 1: Create backend Dockerfile**

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir .

COPY config.example.yaml config.yaml

EXPOSE 8000

CMD ["uvicorn", "casecrawler.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Create frontend Dockerfile**

```dockerfile
# ui/Dockerfile
FROM node:22-slim AS build

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 3000
```

```nginx
# ui/nginx.conf
server {
    listen 3000;
    root /usr/share/nginx/html;
    index index.html;

    location /api/ {
        proxy_pass http://api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

- [ ] **Step 3: Create docker-compose.yml**

```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./.env:/app/.env:ro
      - ./config.yaml:/app/config.yaml:ro
    environment:
      - PYTHONUNBUFFERED=1

  ui:
    build: ./ui
    ports:
      - "3000:3000"
    depends_on:
      - api
```

- [ ] **Step 4: Create a config.yaml for local use (copy from example)**

Run:
```bash
cp config.example.yaml config.yaml
cp .env.example .env
```

- [ ] **Step 5: Verify docker compose config is valid**

Run:
```bash
docker compose config
```
Expected: Outputs the resolved config without errors.

- [ ] **Step 6: Run all tests one final time**

Run:
```bash
pytest tests/ -v
```
Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
git add Dockerfile ui/Dockerfile ui/nginx.conf docker-compose.yml
git commit -m "feat: Docker setup with compose for API and UI"
```

---

## Summary

After completing all 14 tasks, you will have:

- **9 source plugins** (6 free, 3 paid) with mocked tests
- **4-stage processing pipeline** (chunk, tag, embed, store) with ChromaDB
- **CLI** with `ingest`, `search`, `sources`, `config`, and `serve` commands
- **REST API** with ingest (async), search, and sources endpoints
- **React web UI** with ingest, search, and sources pages
- **Docker Compose** for one-command deployment
- **Graceful degradation** — works with zero config, improves with each API key added
