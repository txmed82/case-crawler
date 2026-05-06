# CaseCrawler

CaseCrawler generates validated synthetic healthcare datasets for AI training and evaluation.

It combines grounded medical knowledge retrieval, structured clinical data generation, messy clinical text synthesis, labs, vitals, time-series scaffolding, optional medical imaging hooks, validation, and fine-tuning exports.

The goal is not to simulate a classroom case. The goal is to produce multimodal synthetic records that are ready to inspect, validate, and export as JSONL, FHIR NDJSON, parquet, or model-specific fine-tuning formats.

## Quick Start

```bash
git clone https://github.com/txmed82/case-crawler.git
cd case-crawler
pip install -e ".[dev]"

# See available knowledge sources
casecrawler sources

# Ingest medical knowledge for grounding
casecrawler ingest "sepsis"

# Generate synthetic healthcare records without an LLM key
casecrawler generate-dataset "sepsis" --count 10

# Search the grounded knowledge base
casecrawler search "sepsis lactate fluid resuscitation"
```

### With Docker

```bash
cp .env.example .env
docker compose up
```

## What It Generates

The synthetic dataset path produces `SyntheticRecord` objects with:

- Structured patient demographics and encounters
- Diagnoses and procedure/code slots
- Labs with units, reference ranges, flags, and timestamps
- Vitals with timestamps
- Clean clinical notes and messy note variants
- Time-series channels for longitudinal vitals, labs, or waveform-like data
- Imaging asset metadata and optional image-generation backend hooks
- Provenance metadata
- Validation reports with schema, clinical consistency, privacy, utility, and modality-alignment scores

The legacy case-generation path still exists for backward compatibility, but new work should use `generate-dataset` and the dataset APIs.

## Data Sources

CaseCrawler works with zero API keys using free public sources. Paid keys unlock richer data.

| Source | Key Required | What You Get |
|--------|-------------|--------------|
| PubMed | None | Biomedical citations and abstracts |
| OpenFDA | None | Drug adverse events and labeling |
| DailyMed | None | Structured drug labels |
| RxNorm | None | Drug names and classes |
| medRxiv | None | Medical preprints |
| ClinicalTrials.gov | None | Trial protocols, eligibility, outcomes |
| Glass Health | `GLASS_API_KEY` | Curated clinical reasoning content |
| Anna's Archive | `ANNAS_ARCHIVE_API_KEY` | Full-text papers and medical textbooks |
| Firecrawl | `FIRECRAWL_API_KEY` | Web scraping for guidelines and unstructured content |

Run `casecrawler sources` to see what is available with your current keys.

## Synthetic Generation Pipeline

The dataset-first path starts with a no-key deterministic slice and is designed for pluggable model backends:

```text
Topic + GenerationRequest
      |
[1. Structured Generator]  -> patient, encounter, labs, vitals
      |
[2. Text Generator]        -> clean and messy clinical notes
      |
[3. Validators]            -> schema, clinical rules, privacy, utility
      |
[4. DatasetStore]          -> SQLite synthetic_records
      |
[5. Exporters]             -> SFT/chat/multimodal/RL/FHIR/parquet profiles
```

Optional backends are intentionally lazy:

- `casecrawler[hf]` for Hugging Face helpers
- `casecrawler[imaging]` for diffusers/image validation backends
- `casecrawler[parquet]` for parquet exports
- Existing OpenAI, Anthropic, OpenRouter, and Ollama providers remain available for model-backed generation

## CLI Reference

```bash
# Knowledge ingestion and search
casecrawler ingest "sepsis"
casecrawler ingest "pulmonary embolism" --sources pubmed,openfda
casecrawler search "elevated lactate septic shock"
casecrawler sources
casecrawler config

# Synthetic healthcare dataset generation
casecrawler generate-dataset "sepsis" --count 25
casecrawler generate-dataset "heart failure exacerbation" --count 100 --complexity complex

# Legacy case generation remains available
casecrawler generate "subarachnoid hemorrhage" --difficulty resident
casecrawler cases export --output legacy_cases.jsonl
```

## REST API

Start the server with `casecrawler serve` or `docker compose up`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ingest` | POST | Ingest content for a topic |
| `/api/ingest/{job_id}` | GET | Poll ingestion status |
| `/api/search?q=...` | GET | Search the knowledge base |
| `/api/sources` | GET | List available sources |
| `/api/datasets/generate` | POST | Generate synthetic healthcare records |
| `/api/generate` | POST | Legacy clinical case generation |
| `/api/cases` | GET | Legacy case list/filter |
| `/api/cases/export` | GET | Legacy case JSONL stream |

## Configuration

### `.env`

```bash
# Optional LLM providers
ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...
# OPENROUTER_API_KEY=sk-or-...

# Optional paid data sources
# GLASS_API_KEY=
# ANNAS_ARCHIVE_API_KEY=
# FIRECRAWL_API_KEY=

# Optional free-source rate-limit keys
# NCBI_API_KEY=
# OPENFDA_API_KEY=
```

### `config.yaml`

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

generation:
  max_retries: 3
  review_threshold: 0.7
  default_difficulty: "resident"
  retriever_chunk_count: 25

api:
  host: "0.0.0.0"
  port: 8000
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check src tests
```

## Project Structure

```text
src/casecrawler/
  sources/       # Public and paid medical source adapters
  pipeline/      # Chunking, tagging, embedding, Chroma storage
  generation/    # Legacy cases plus synthetic dataset generators
  validation/    # Synthetic record validation
  storage/       # SQLite stores
  export/        # Fine-tuning and legacy exporters
  api/           # FastAPI routes
```
