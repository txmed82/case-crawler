# CaseCrawler Case Generation Pipeline — Design Spec

**Date:** 2026-04-08
**Status:** Draft
**Depends on:** Ingestion Engine (v1, complete)

---

## 1. Overview

Phase 2 of CaseCrawler: a multi-agent LLM pipeline that takes medical topics and generates realistic, decision-forcing clinical cases. Cases serve two use cases:

1. **AI training data** — structured JSONL output for fine-tuning and evaluation benchmarks
2. **Medical education** — interactive case player where learners walk through clinical decisions

The pipeline retrieves relevant medical knowledge from the ingestion engine's ChromaDB store, generates cases through specialized LLM agents, validates them via a clinical reviewer agent, and stores approved cases in SQLite.

### Design Philosophy

- **Multi-agent with critic** — separate agents for generation, decision trees, and review. Each can be retried independently.
- **Retry with feedback** — critic rejection notes are fed back to the failed stage for targeted regeneration (max 3 retries).
- **Structured output everywhere** — Pydantic schemas enforce JSON format from LLMs. No parsing failures.
- **Two stores for two data types** — ChromaDB for vector knowledge, SQLite for structured cases.

---

## 2. Pipeline Architecture

Four stages in sequence:

```
User Input (topic + difficulty)
        |
[1. Retriever] — ChromaDB query + credibility ranking (no LLM)
        |
[2. Case Generator] — vignette + ground truth + patient demographics (LLM)
        |
[3. Decision Tree Builder] — choices, outcomes, complications (LLM)
        |
[4. Clinical Reviewer] — accuracy, pedagogy, bias scoring (LLM)
        |
   approved? --no--> feed critic notes to failed stage, retry (max 3)
        | yes
[SQLite Store] --> JSON/JSONL output
```

### Stage Details

**Stage 1: Retriever** (no LLM call, pure code)

- Takes topic string, queries ChromaDB with semantic search
- Fetches top 20-30 chunks, ranked by relevance score
- Groups by credibility: guidelines first, then peer-reviewed, then preprints
- Passes to generator as a structured context block with source attribution (chunk IDs, source type, credibility level)

**Stage 2: Case Generator** (LLM call)

- System prompt: "You are a clinical case author creating realistic, decision-forcing scenarios"
- Receives: topic, difficulty level, difficulty rules, retrieved chunks with citations
- Key instruction: "Create a case where a clinician must make a decision with incomplete information"
- Outputs: vignette, patient demographics, ground truth (diagnosis, optimal next step, rationale, key findings)
- Uses `generate_structured()` with Pydantic schema enforcement

**Stage 3: Decision Tree Builder** (LLM call)

- Receives: generated vignette + ground truth + source chunks
- Builds: 1 correct path + 2-4 wrong paths (count depends on difficulty)
- Each wrong path gets an `error_type` (common_mistake or catastrophic) with realistic consequences
- Also generates the complications layer (delayed diagnosis, incorrect treatment cascades)
- Uses `generate_structured()` with Pydantic schema enforcement

**Stage 4: Clinical Reviewer** (LLM call)

- Receives: complete case + source chunks it was built from
- Three scoring dimensions (0.0-1.0):
  - **Accuracy** — Is the diagnosis correct? Are treatments real? Are lab values physiologically possible?
  - **Pedagogy** — Is the case appropriately challenging? Are distractors plausible but distinguishable? Does the decision tree cover important learning points?
  - **Bias** — Does the case avoid demographic stereotyping? Is the patient presentation diverse? Are there cultural assumptions?
- Threshold for approval: all three scores >= 0.7
- On rejection: provides specific, actionable notes per failed dimension (e.g., "The vignette states troponin is elevated but no MI is in the differential — either add it as a distractor or remove the lab value")

### Retry Logic

When the critic rejects a case:
1. Identify which dimension(s) failed (accuracy, pedagogy, bias)
2. Map failure to the responsible stage (accuracy/bias issues → Case Generator or Decision Tree Builder; pedagogy → Decision Tree Builder)
3. Re-invoke that stage with the critic's notes injected into the prompt: "Previous attempt was rejected. Reviewer feedback: [notes]. Fix these specific issues."
4. Re-run the reviewer on the updated case
5. Max 3 retry cycles. After that, mark the case as failed and move on.

---

## 3. Difficulty System

Three tiers: `medical_student`, `resident`, `attending`.

| Dimension | Medical Student | Resident | Attending |
|---|---|---|---|
| **Vignette completeness** | Most key findings present, fewer distractors | Some findings missing, moderate distractors | Incomplete data, many distractors, red herrings |
| **Diagnosis** | Classic textbook presentation | Atypical or overlapping presentations | Rare variants, multiple concurrent pathologies |
| **Decision tree breadth** | 2-3 choices | 3-4 choices | 4-5 choices with subtle distinctions |
| **Complication depth** | Simple cause-effect | Multi-step cascades | System-level failures (missed consult -> delayed OR -> herniation) |
| **Expected knowledge** | Foundational pathophysiology | Management algorithms, workup prioritization | Nuanced judgment calls, resource constraints, ambiguity |

These rules are encoded in the Case Generator and Decision Tree Builder system prompts. The difficulty level is passed as a parameter and the prompt includes the specific rules for that tier.

---

## 4. Data Models

### Case Output Schema

```python
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
    error_type: str | None       # "common_mistake" or "catastrophic", None if correct
    reasoning: str
    outcome: str
    consequence: str | None
    next_decision: str | None    # for multi-step correct paths

class Complication(BaseModel):
    trigger: str                 # "delayed_diagnosis", "incorrect_treatment"
    detail: str                  # specific trigger (e.g., "Anticoagulation")
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
    review: ReviewResult
    sources: list[dict]          # {"type": "pubmed", "reference": "...", "chunk_ids": [...]}
    metadata: dict               # generated_at, model, version, retry_count
```

### SQLite Schema

```sql
CREATE TABLE cases (
    case_id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    difficulty TEXT NOT NULL,
    specialty TEXT NOT NULL,     -- comma-separated
    accuracy_score REAL,
    pedagogy_score REAL,
    bias_score REAL,
    model TEXT,
    generated_at TIMESTAMP,
    case_json TEXT NOT NULL      -- full GeneratedCase as JSON
);

CREATE INDEX idx_topic ON cases(topic);
CREATE INDEX idx_difficulty ON cases(difficulty);
CREATE INDEX idx_specialty ON cases(specialty);
```

Full case stored as JSON column for flexibility and easy JSONL export. Key fields pulled into columns for efficient filtering.

---

## 5. LLM Provider Implementation

The ingestion engine defined the LLM abstraction interface. This phase implements it.

### Interface

```python
class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, system: str = "", **kwargs) -> GenerationResult:
        """Generate a text completion."""

    @abstractmethod
    async def generate_structured(self, prompt: str, schema: type[BaseModel], system: str = "", **kwargs) -> StructuredGenerationResult:
        """Generate a response conforming to a Pydantic schema."""

class GenerationResult(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int
    model: str

class StructuredGenerationResult(BaseModel):
    data: BaseModel
    input_tokens: int
    output_tokens: int
    model: str
```

### Implementations

| Provider | SDK | Structured output method |
|---|---|---|
| Anthropic | `anthropic` | Tool use with JSON schema |
| OpenAI | `openai` | `response_format` with JSON schema |
| OpenRouter | `openai` (compatible API) | Same as OpenAI |
| Ollama | `httpx` (already a dependency) | JSON mode + Pydantic validation |

### Provider Selection

Reads `llm.provider` and `llm.model` from `config.yaml`. Checks for the corresponding API key in `.env`. If the configured provider's key is missing, provides a clear error message indicating which key to set.

### Token Tracking

Each provider returns token counts with every response. Stored in case metadata for cost-per-case tracking.

### Dependencies

Added as optional deps in `pyproject.toml`:

```toml
[project.optional-dependencies]
anthropic = ["anthropic>=0.40"]
openai = ["openai>=1.50"]
all-llm = ["anthropic>=0.40", "openai>=1.50"]
```

Ollama uses `httpx` (already installed). No extra dependency for local models.

---

## 6. CLI Extensions

```
casecrawler generate "subarachnoid hemorrhage"                    # 1 case, default difficulty
casecrawler generate "SAH" --difficulty resident --count 10       # Batch
casecrawler generate "SAH" --count 50 --ingest                   # Ingest first, then generate
casecrawler generate "SAH" --count 50 --output cases.jsonl       # Export to JSONL

casecrawler cases                                                 # List all cases
casecrawler cases --topic SAH --difficulty attending              # Filter
casecrawler cases export --output training_data.jsonl             # Export filtered set
casecrawler cases show <case_id>                                  # View single case
```

### `casecrawler generate` flow:

1. If `--ingest` flag: run ingestion for the topic first
2. Retriever queries ChromaDB for relevant chunks
3. For each case in `--count`:
   - Run Case Generator
   - Run Decision Tree Builder
   - Run Clinical Reviewer
   - If rejected: retry with feedback (max 3)
   - If approved: store in SQLite
4. Print summary: cases generated, cases failed, avg scores, elapsed time, tokens used

Batch generation runs cases sequentially (not parallel LLM calls) for v1. Parallel generation is a future optimization.

---

## 7. REST API Extensions

### POST /api/generate

```json
{
  "topic": "subarachnoid hemorrhage",
  "difficulty": "resident",
  "count": 10,
  "ingest_first": false
}

// Response: 202 Accepted
{
  "job_id": "uuid",
  "status": "running"
}
```

Generation runs as a background task (sequential case generation within the job, same as CLI). Pollable via:

### GET /api/generate/{job_id}

```json
{
  "job_id": "uuid",
  "status": "completed",
  "cases_generated": 8,
  "cases_failed": 2,
  "elapsed_seconds": 45.2
}
```

### GET /api/cases

```
GET /api/cases?topic=SAH&difficulty=resident&min_accuracy=0.8&limit=20

{
  "cases": [...],
  "total": 42
}
```

### GET /api/cases/{case_id}

Returns the full GeneratedCase JSON.

### GET /api/cases/export

```
GET /api/cases/export?topic=SAH&difficulty=resident

// Returns JSONL stream (Content-Type: application/x-ndjson)
```

---

## 8. Case Player UI

New route in the existing React app: `/play/{case_id}`

### Player Flow

1. **Presentation** — Shows vignette, patient info. "What would you do next?"
2. **Decision** — Choices as clickable cards (correct answer not revealed)
3. **Reveal** — Shows outcome of chosen path. Correct: green with reasoning. Wrong: shows error type, consequence, what they should have done.
4. **Debrief** — Ground truth, full decision tree (all paths color-coded), complications, source references with links.

### New React Components

| Component | Responsibility |
|---|---|
| `pages/PlayCasePage.tsx` | Orchestrates the case player flow (state machine) |
| `pages/CasesPage.tsx` | Browse/filter generated cases, click to play |
| `pages/GeneratePage.tsx` | Form to trigger case generation (like IngestPage) |
| `components/VignetteCard.tsx` | Renders case vignette + patient info |
| `components/DecisionCards.tsx` | Presents choices, handles selection |
| `components/OutcomeReveal.tsx` | Shows result of choice |
| `components/CaseDebrief.tsx` | Full case breakdown |
| `components/DecisionTreeViz.tsx` | Visual tree — correct green, mistake yellow, catastrophic red |

### Navigation Updates

Add "Generate" and "Cases" to the nav bar alongside existing "Ingest", "Search", "Sources".

---

## 9. Project Structure (New/Modified Files)

```
src/casecrawler/
├── llm/                              # NEW: LLM provider implementations
│   ├── base.py                       # BaseLLMProvider, GenerationResult
│   ├── anthropic.py
│   ├── openai.py
│   ├── openrouter.py
│   ├── ollama.py
│   └── factory.py                    # get_provider() from config
├── generation/                       # NEW: Case generation pipeline
│   ├── retriever.py                  # ChromaDB query + ranking
│   ├── case_generator.py             # Stage 2: vignette + ground truth
│   ├── decision_tree_builder.py      # Stage 3: choices + complications
│   ├── clinical_reviewer.py          # Stage 4: accuracy/pedagogy/bias
│   ├── pipeline.py                   # Orchestrates all stages + retry
│   └── prompts.py                    # System prompts for each stage
├── models/
│   ├── document.py                   # (existing)
│   ├── config.py                     # (modify: add llm optional deps config)
│   └── case.py                       # NEW: GeneratedCase, DifficultyLevel, etc.
├── storage/
│   └── case_store.py                 # NEW: SQLite CRUD for cases
├── cli.py                            # (modify: add generate + cases commands)
├── api/
│   ├── app.py                        # (modify: add new routers)
│   └── routes/
│       ├── generate.py               # NEW: generation endpoints
│       └── cases.py                  # NEW: case CRUD + export endpoints

ui/src/
├── pages/
│   ├── PlayCasePage.tsx              # NEW
│   ├── CasesPage.tsx                 # NEW
│   └── GeneratePage.tsx              # NEW
├── components/
│   ├── VignetteCard.tsx              # NEW
│   ├── DecisionCards.tsx             # NEW
│   ├── OutcomeReveal.tsx             # NEW
│   ├── CaseDebrief.tsx              # NEW
│   └── DecisionTreeViz.tsx           # NEW
└── App.tsx                           # (modify: add routes + nav links)
```

---

## 10. Configuration Updates

### config.yaml additions

```yaml
llm:
  provider: "anthropic"          # anthropic | openai | openrouter | ollama
  model: "claude-sonnet-4-6"
  # ollama_base_url: "http://localhost:11434"

generation:
  max_retries: 3
  review_threshold: 0.7          # minimum score for all three dimensions
  default_difficulty: "resident"
  retriever_chunk_count: 25
```

### .env additions

```
# LLM providers (at least one required for case generation)
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
OPENROUTER_API_KEY=
# Ollama needs no key — just ollama_base_url in config.yaml
```

---

## 11. Future Phases (Out of Scope)

- **Parallel batch generation** — concurrent LLM calls for faster batch output
- **Progress tracking / user accounts** — learner performance analytics
- **Spaced repetition** — resurface cases the learner struggled with
- **Multi-step cases** — cases that evolve over time (patient deteriorates, new information arrives)
- **Case editing UI** — manual review and correction of generated cases
- **Fine-tuned reviewer** — train the critic on expert-annotated case quality data
- **Diversity controls for batch generation** — demographic distribution rules, chief complaint variety
