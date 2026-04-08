# CaseCrawler

CaseCrawler ingests medical knowledge from across the internet and generates **realistic, decision-based clinical cases** — the kind where you have to make a call with incomplete information, just like real medicine.

Most medical AI tools summarize. CaseCrawler forces decisions under uncertainty.

## Why

Clinical reasoning is hard to teach and harder to evaluate. Textbook questions are too clean. Real patients are messy, incomplete, and full of distractors. CaseCrawler generates cases that feel like real encounters — with decision trees that branch into correct paths, common mistakes, and catastrophic errors.

Two use cases:

1. **AI training data** — Generate structured JSONL datasets for fine-tuning medical models, building eval benchmarks, and training clinical reasoning systems
2. **Medical education** — Interactive case player where students, residents, and nurses walk through clinical decisions and learn from realistic outcomes

## Quick Start

```bash
# Clone and install
git clone https://github.com/txmed82/case-crawler.git
cd case-crawler
pip install -e ".[dev]"

# See what data sources are available (works immediately, no keys needed)
casecrawler sources

# Ingest medical knowledge about a topic
casecrawler ingest "subarachnoid hemorrhage"

# Search what you've ingested
casecrawler search "thunderclap headache"

# Generate a clinical case (requires an LLM API key — see Configuration)
casecrawler generate "subarachnoid hemorrhage" --difficulty resident

# Browse generated cases
casecrawler cases
```

### With Docker

```bash
cp .env.example .env    # add your API keys
docker compose up       # API on :8000, UI on :3000
```

## Data Sources

CaseCrawler works with **zero API keys** using free public sources. Each paid key you add unlocks richer data.

| Source | Key Required | What You Get |
|--------|-------------|--------------|
| PubMed | None | 35M+ biomedical citations, guidelines |
| OpenFDA | None | Drug adverse events, labeling |
| DailyMed | None | Structured drug labels |
| RxNorm | None | Drug names, classes |
| medRxiv | None | Medical preprints |
| ClinicalTrials.gov | None | Trial protocols, eligibility, outcomes |
| Glass Health | `GLASS_API_KEY` | Curated clinical reasoning content |
| Anna's Archive | `ANNAS_ARCHIVE_API_KEY` | Full-text papers + medical textbooks |
| Firecrawl | `FIRECRAWL_API_KEY` | Web scraping for guidelines and unstructured content |

Run `casecrawler sources` to see what's available with your current keys.

## Case Generation

CaseCrawler uses a 4-stage LLM pipeline to generate cases:

```
Topic + Difficulty
      |
[1. Retriever]          — pulls relevant knowledge from ChromaDB
      |
[2. Case Generator]     — creates vignette, patient, ground truth
      |
[3. Decision Tree]      — builds correct + wrong paths with consequences
      |
[4. Clinical Reviewer]  — scores accuracy, pedagogy, and bias
      |
  rejected? --> retry with reviewer feedback (up to 3x)
      |
  approved --> saved to SQLite
```

The clinical reviewer checks three dimensions:
- **Accuracy** — Is the medicine correct?
- **Pedagogy** — Is this case actually teaching something?
- **Bias** — Does it avoid demographic stereotyping?

### Difficulty Levels

| Level | What Changes |
|-------|-------------|
| `medical_student` | Classic presentations, 2-3 choices, simple consequences |
| `resident` | Atypical presentations, 3-4 choices, multi-step cascades |
| `attending` | Rare variants, 4-5 subtle choices, system-level failures |

### LLM Providers

Use whichever LLM provider you already pay for:

| Provider | Key | Config |
|----------|-----|--------|
| Anthropic | `ANTHROPIC_API_KEY` | `provider: anthropic` |
| OpenAI | `OPENAI_API_KEY` | `provider: openai` |
| OpenRouter | `OPENROUTER_API_KEY` | `provider: openrouter` |
| Ollama (local) | None | `provider: ollama` |

## CLI Reference

```bash
# Ingestion
casecrawler ingest "sepsis"                          # ingest from all free sources
casecrawler ingest "ACDF" --sources pubmed,glass     # specific sources
casecrawler ingest "PE" --limit 50                   # more results per source

# Case generation
casecrawler generate "sepsis"                        # 1 case, default difficulty
casecrawler generate "MI" --difficulty attending --count 10
casecrawler generate "SAH" --count 50 --output training.jsonl
casecrawler generate "PE" --ingest                   # ingest first, then generate

# Case management
casecrawler cases                                    # list all cases
casecrawler cases --topic sepsis --difficulty resident
casecrawler cases show <case_id>                     # full case JSON
casecrawler cases export --output dataset.jsonl      # JSONL export

# Search & info
casecrawler search "thunderclap headache"
casecrawler search "LP contraindications" --source openfda
casecrawler sources                                  # available/unavailable sources
casecrawler config                                   # current settings

# Server
casecrawler serve                                    # start API on :8000
```

## REST API

Start the server with `casecrawler serve` or `docker compose up`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ingest` | POST | Ingest content for a topic (async) |
| `/api/ingest/{job_id}` | GET | Poll ingestion status |
| `/api/search?q=...` | GET | Search the knowledge base |
| `/api/sources` | GET | List available sources |
| `/api/generate` | POST | Generate cases (async) |
| `/api/generate/{job_id}` | GET | Poll generation status |
| `/api/cases` | GET | List/filter cases |
| `/api/cases/{case_id}` | GET | Get a single case |
| `/api/cases/export` | GET | JSONL stream |

## Web UI

The React frontend runs on port 3000 and gives you:

- **Ingest** — pick a topic, select sources, watch ingestion progress
- **Search** — query the knowledge base, filter by source and credibility
- **Sources** — see what's available and what keys you're missing
- **Generate** — set topic + difficulty + count, generate cases
- **Cases** — browse and filter generated cases
- **Play** — interactive case player: read the vignette, make a decision, see the outcome, review the full debrief

## Configuration

### `.env` — API keys (secrets, never committed)

```bash
# LLM provider (at least one needed for case generation)
ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...
# OPENROUTER_API_KEY=sk-or-...

# Paid data sources (all optional)
# GLASS_API_KEY=
# ANNAS_ARCHIVE_API_KEY=
# FIRECRAWL_API_KEY=

# Free sources (optional, increases rate limits)
# NCBI_API_KEY=
# OPENFDA_API_KEY=
```

### `config.yaml` — preferences (safe to commit)

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

## Case Output Format

Generated cases are structured JSON:

```json
{
  "case_id": "uuid",
  "topic": "subarachnoid hemorrhage",
  "difficulty": "resident",
  "specialty": ["neurosurgery", "emergency_medicine"],
  "patient": { "age": 42, "sex": "female", "demographics": "..." },
  "vignette": "A 42-year-old woman presents to the ED with...",
  "decision_prompt": "What would you do next?",
  "ground_truth": {
    "diagnosis": "aneurysmal subarachnoid hemorrhage",
    "optimal_next_step": "Non-contrast CT head",
    "rationale": "...",
    "key_findings": ["thunderclap headache", "neck stiffness"]
  },
  "decision_tree": [
    { "choice": "CT head", "is_correct": true, "outcome": "..." },
    { "choice": "MRI brain", "is_correct": false, "error_type": "common_mistake", "consequence": "..." },
    { "choice": "Discharge", "is_correct": false, "error_type": "catastrophic", "consequence": "..." }
  ],
  "complications": [...],
  "review": { "accuracy_score": 0.95, "pedagogy_score": 0.88, "bias_score": 0.92 },
  "sources": [{ "type": "pubmed", "reference": "PMID:12345678" }]
}
```

Export as JSONL for AI training: `casecrawler cases export --output training_data.jsonl`

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Frontend
cd ui && npm install && npm run dev

# Lint
ruff check src/ tests/
```

## Architecture

```
src/casecrawler/
  sources/        # 9 source plugins (BaseSource interface)
  pipeline/       # chunker, tagger, embedder, ChromaDB store
  generation/     # retriever, case generator, decision tree, clinical reviewer
  llm/            # Anthropic, OpenAI, OpenRouter, Ollama providers
  storage/        # SQLite case store
  api/            # FastAPI REST API
  cli.py          # Click CLI

ui/               # React + Vite + Tailwind
```

## License

Apache 2.0
