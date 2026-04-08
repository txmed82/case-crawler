# CaseCrawler Ingestion Engine — Design Spec

**Date:** 2026-04-08
**Status:** Draft
**License:** Apache 2.0

---

## 1. Overview

CaseCrawler is an open-source agentic system that ingests medical knowledge from across the internet and generates realistic, decision-based clinical cases. This spec covers **v1: the ingestion engine** — the data foundation that crawls, normalizes, chunks, embeds, and stores medical content from multiple sources.

The system uses a **hybrid plugin-source + processing-pipeline architecture**. Source plugins handle the specifics of each data source (auth, fetching, parsing). A shared processing pipeline handles everything after: chunking, tagging, embedding, and storage into ChromaDB.

### Design Philosophy

- **Graceful degradation:** Works with zero API keys (free sources only). Each paid key unlocks richer data.
- **Open source, modular:** Contributors add sources by dropping in a plugin file. No manual registration.
- **Local-first:** Embedded vector DB, local embeddings, no external infrastructure required.

---

## 2. Data Sources

### Tiered Source Model

| Tier | Sources | Keys Required |
|------|---------|---------------|
| **Zero config** | PubMed, OpenFDA, DailyMed, RxNorm, medRxiv, ClinicalTrials.gov | None |
| **+ Glass Health** | Curated clinical reasoning content | `GLASS_API_KEY` |
| **+ Anna's Archive** | Full-text papers + medical textbooks | `ANNAS_ARCHIVE_API_KEY` |
| **+ Firecrawl** | Web scraping for guidelines sites, blogs, unstructured content | `FIRECRAWL_API_KEY` |

Any combination of keys works. The system adapts to whatever is available.

### Source Details

**PubMed E-utilities** (`eutils.ncbi.nlm.nih.gov`) — 35M+ biomedical citations. Can filter by "Practice Guideline" publication type. Free, 3 req/sec without key, 10 with key.

**OpenFDA** (`api.fda.gov`) — 20M+ adverse event reports, drug labeling, device data. Free, 240 req/min. Feeds the complication engine directly.

**DailyMed** (`dailymed.nlm.nih.gov`) — Structured drug labeling: indications, dosing, contraindications, adverse reactions. Free, no auth.

**RxNorm** (`rxnav.nlm.nih.gov/REST`) — Drug names, interactions, classes. Free, 20 req/sec.

**medRxiv** (`api.medrxiv.org`) — Preprints. Free, no auth.

**ClinicalTrials.gov v2** (`clinicaltrials.gov/api/v2/studies`) — Trial protocols, eligibility criteria, interventions, outcomes. Free, 50 req/min.

**Glass Health** (`glass.health/developer-api`) — Curated clinical reasoning content. Paid, requires API key.

**Anna's Archive** (`annas-archive.gl/scidb`) — Full-text papers via SciDB + medical textbooks. Cheap API, requires key.

**Firecrawl** — JS rendering, anti-bot, clean markdown output. Paid, enables scraping of guideline sites and other unstructured web content.

---

## 3. Architecture

### High-Level Flow

```
[Source Plugins] → [Normalized Documents] → [Processing Pipeline] → [ChromaDB]
     |                                            |
  per-source logic                    shared: chunking, tagging,
  (auth, fetch, parse)                embedding, dedup, storage
```

### Source Plugin Interface

Each source implements:

```python
class BaseSource(ABC):
    name: str                          # e.g. "pubmed"
    requires_keys: list[str]           # e.g. [] or ["GLASS_API_KEY"]

    @abstractmethod
    async def search(self, query: str, limit: int = 20) -> list[Document]:
        """Search the source for documents matching a query."""

    @abstractmethod
    async def fetch(self, document_id: str) -> Document:
        """Fetch full content for a specific document."""

    @classmethod
    def is_available(cls) -> bool:
        """Check if all required API keys are present in env."""
```

All sources are async for parallel fan-out.

### Normalized Document Model

The contract between sources and the pipeline:

```python
class Document(BaseModel):
    source: str                        # "pubmed", "glass", etc.
    source_id: str                     # ID within that source
    title: str
    content: str                       # Raw text content
    content_type: str                  # "abstract", "full_text", "drug_label", etc.
    metadata: DocumentMetadata

class DocumentMetadata(BaseModel):
    authors: list[str] = []
    publication_date: date | None = None
    specialty: list[str] = []          # e.g. ["neurosurgery", "emergency"]
    credibility: CredibilityLevel      # enum: guideline, peer_reviewed, preprint, curated, fda_label
    url: str | None = None
    doi: str | None = None
```

### Source Registry

Auto-discovers available sources at startup via subclass discovery:

```python
class SourceRegistry:
    def __init__(self):
        self._sources: dict[str, BaseSource] = {}

    def discover(self) -> None:
        """Find all BaseSource subclasses, instantiate those with valid credentials."""
        for source_cls in BaseSource.__subclasses__():
            if source_cls.is_available():
                self._sources[source_cls.name] = source_cls()

    @property
    def available_sources(self) -> list[str]:
        """What sources are active right now."""
```

Adding a source = add a file implementing `BaseSource`, import it. No manual registration.

---

## 4. Processing Pipeline

Documents flow through four stages in order:

```
Documents → Chunker → Tagger → Embedder → Store
```

### Chunker

Splits content into retrieval-sized pieces. Strategy varies by `content_type`:

- **Abstracts** — kept whole (usually <500 tokens)
- **Full text** — recursive splitting on headings/paragraphs, ~500 token chunks with overlap
- **Drug labels** — split by section (indications, contraindications, dosing, adverse reactions)
- **Trial protocols** — split by structured sections (eligibility, interventions, outcomes)

```python
class Chunk(BaseModel):
    chunk_id: str                      # deterministic hash of source_id + offset
    source_document_id: str            # back-reference
    text: str
    position: int                      # order within document
    metadata: DocumentMetadata         # inherited from parent document
```

Deterministic `chunk_id` ensures re-ingesting the same document doesn't create duplicates.

### Tagger

Enriches metadata that the source didn't provide. Rule-based for v1: keyword matching for specialty, mapping source types to credibility. LLM-based tagging is a future enhancement.

### Embedder

Generates vector embeddings. Default model: **`all-MiniLM-L6-v2`** via `sentence-transformers`. Free, local, no API key needed. Configurable via `config.yaml` for users who want OpenAI embeddings or a medical-specific model.

### Store

Writes to ChromaDB. Upserts by `chunk_id` for deduplication. All chunks go into a single unified collection with `source` as a metadata field. Filtering by source happens at query time via ChromaDB's metadata filtering, not via separate collections.

---

## 5. CLI Interface

Built with Click:

```
casecrawler ingest "subarachnoid hemorrhage"      # Ingest from all available sources
casecrawler ingest "ACDF" --sources pubmed,glass   # Specific sources only
casecrawler ingest "sepsis" --limit 50             # Cap results per source

casecrawler search "thunderclap headache"          # Query ChromaDB
casecrawler search "LP contraindications" --source openfda

casecrawler sources                                # List available/unavailable sources + why
casecrawler config                                 # Show current config, detected keys
casecrawler serve                                  # Start FastAPI server
```

### `casecrawler ingest` flow:

1. Registry discovers available sources
2. Fans out `search()` calls to all available sources (or filtered set) in parallel
3. Fetches full content where available
4. Feeds documents through the pipeline: Chunk -> Tag -> Embed -> Store
5. Prints summary: documents ingested per source, total chunks stored, time elapsed

### `casecrawler sources` output:

```
Available:
  ✓ pubmed          (no key required)
  ✓ openfda         (no key required)
  ✓ dailymed        (no key required)
  ✓ rxnorm          (no key required)
  ✓ medrxiv         (no key required)
  ✓ clinicaltrials  (no key required)
  ✓ glass           (GLASS_API_KEY set)

Unavailable:
  ✗ annas_archive   (missing ANNAS_ARCHIVE_API_KEY)
  ✗ firecrawl       (missing FIRECRAWL_API_KEY)
```

---

## 6. REST API

FastAPI server with three route groups:

### POST /api/ingest

```json
// Request
{
  "query": "subarachnoid hemorrhage",
  "sources": ["pubmed", "glass"],
  "limit": 20
}

// Response: 202 Accepted
{
  "job_id": "uuid",
  "status": "running"
}
```

Ingestion runs as a background task. Status is pollable via `GET /api/ingest/{job_id}`:

```json
{
  "job_id": "uuid",
  "status": "completed",
  "summary": {
    "pubmed": {"documents": 18, "chunks": 74},
    "glass": {"documents": 5, "chunks": 22}
  },
  "elapsed_seconds": 12.3
}
```

### GET /api/search

```
GET /api/search?q=thunderclap+headache&source=pubmed&limit=10
```

Returns ranked chunks with scores and metadata.

### GET /api/sources

Returns available and unavailable sources with status and missing key information.

No auth on the API for v1 — runs locally.

---

## 7. Web UI

React + TypeScript + Vite SPA. Three views:

### Dashboard / Ingest
- Text input for topic
- Checkboxes for available sources (unavailable greyed out with tooltip showing missing key)
- "Ingest" button with progress bar polling job status
- Summary card on completion: documents per source, total chunks, elapsed time

### Search / Browse
- Search bar querying `GET /api/search`
- Results as cards: chunk text, source badge (color-coded by credibility), metadata
- Filter sidebar: source, credibility level, specialty
- Click to expand: full chunk text + link to original source

### Sources / Config
- Available/unavailable sources list
- Links to docs for obtaining each API key
- Current config summary

### Tech Stack
- React + TypeScript + Vite
- Tailwind CSS
- TanStack Query for API state management
- No component library — keep dependencies light

---

## 8. Configuration

### `.env` — secrets only

```
# LLM providers (not required for ingestion, needed for future case generation)
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
OPENROUTER_API_KEY=

# Paid data sources (all optional)
GLASS_API_KEY=
ANNAS_ARCHIVE_API_KEY=
FIRECRAWL_API_KEY=

# Free sources (optional, increases rate limits)
PUBMED_API_KEY=
NCBI_API_KEY=
```

### `config.yaml` — preferences, safe to commit/share

```yaml
ingestion:
  default_limit_per_source: 20
  sources:
    priority: [pubmed, glass, openfda, annas_archive, dailymed, rxnorm, medrxiv, clinicaltrials, firecrawl]
    disabled: []

chunking:
  default_chunk_size: 500
  overlap: 50

embedding:
  model: "all-MiniLM-L6-v2"

storage:
  chroma_persist_dir: "./data/chroma"

llm:
  provider: "anthropic"
  model: "claude-sonnet-4-6"

api:
  host: "0.0.0.0"
  port: 8000
```

Source priority order determines ranking when multiple sources return similar content. The `disabled` list lets users turn off a source without removing its key.

---

## 9. Docker & Deployment

### docker-compose.yml

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./.env:/app/.env
      - ./config.yaml:/app/config.yaml
    command: uvicorn casecrawler.api.app:app --host 0.0.0.0 --port 8000

  ui:
    build: ./ui
    ports:
      - "3000:3000"
    environment:
      - VITE_API_URL=http://localhost:8000
```

Two containers. ChromaDB runs embedded inside the API container. Data persists via volume mount.

### Local dev (no Docker)

```bash
# Backend
pip install -e ".[dev]"
casecrawler serve

# Frontend
cd ui && npm install && npm run dev
```

### Quickstart

```bash
git clone <repo>
cp .env.example .env
docker compose up
```

---

## 10. Project Structure

```
casecrawler/
├── src/
│   ├── casecrawler/
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   ├── config.py
│   │   ├── sources/
│   │   │   ├── base.py
│   │   │   ├── registry.py
│   │   │   ├── pubmed.py
│   │   │   ├── openfda.py
│   │   │   ├── dailymed.py
│   │   │   ├── rxnorm.py
│   │   │   ├── medrxiv.py
│   │   │   ├── clinicaltrials.py
│   │   │   ├── glass.py
│   │   │   ├── annas_archive.py
│   │   │   └── firecrawl.py
│   │   ├── pipeline/
│   │   │   ├── chunker.py
│   │   │   ├── tagger.py
│   │   │   ├── embedder.py
│   │   │   └── store.py
│   │   ├── models/
│   │   │   ├── document.py
│   │   │   └── config.py
│   │   ├── api/
│   │   │   ├── app.py
│   │   │   └── routes/
│   │   │       ├── ingest.py
│   │   │       ├── search.py
│   │   │       └── sources.py
│   │   └── llm/
│   │       ├── base.py
│   │       ├── anthropic.py
│   │       ├── openai.py
│   │       ├── openrouter.py
│   │       └── ollama.py
├── ui/
│   ├── package.json
│   ├── src/
│   │   ├── App.tsx
│   │   ├── pages/
│   │   └── components/
│   └── vite.config.ts
├── tests/
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── config.example.yaml
├── .env.example
└── LICENSE
```

---

## 11. LLM Provider Abstraction

Not required for the ingestion engine, but the interface is defined now for the case generation phase:

```python
class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, system: str = "", **kwargs) -> str:
        """Generate a completion."""

    @abstractmethod
    async def generate_structured(self, prompt: str, schema: type[BaseModel], **kwargs) -> BaseModel:
        """Generate a structured response matching a Pydantic schema."""
```

Implementations: `AnthropicProvider`, `OpenAIProvider`, `OpenRouterProvider`, `OllamaProvider`. User selects via `config.yaml`. The system checks for the corresponding API key in `.env`.

---

## 12. Future Phases (Out of Scope for v1)

These are documented for context but **not built in this phase:**

- **Case Generation Agent** — takes topic + retrieved content, outputs structured clinical case
- **Decision Tree Generator** — builds correct path + plausible wrong paths for each case
- **Complication Engine** — adds delayed diagnosis, incorrect treatment, edge-case deterioration
- **Standardized Case JSON format** — structured output with vignette, decision prompt, ground truth, decision tree, complications, sources
- **UMLS integration** — terminology normalization across sources
- **LLM-based tagging** — specialty and concept extraction via LLM during ingestion
- **Hosted vector DB option** — Pinecone/Weaviate for users who want managed infrastructure
