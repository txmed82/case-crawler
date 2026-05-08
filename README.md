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

- Structured patient demographics, encounters, diagnoses, medication history, and allergy/intolerance safety facts
- Diagnoses and procedure/code slots
- Labs with units, reference ranges, flags, and timestamps
- Vitals with timestamps
- Clean clinical notes and messy note variants
- Time-series channels for longitudinal vitals, labs, ECG lead II, and pleth waveform-like data
- Imaging asset metadata, optional image-generation backend hooks, and inline image payloads in multimodal exports when image files exist
- Provenance metadata
- Validation reports with schema, clinical consistency, privacy, utility, and modality-alignment scores

The primary workflow is `generate-dataset`, dataset quality reports, reference benchmarks, human review, and export profiles.

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
[5. Exporters]             -> SFT/note-fact/chat/multimodal/RL/FHIR/parquet profiles
```

Optional backends are intentionally lazy:

- `casecrawler[hf]` for Hugging Face helpers
- `casecrawler[imaging]` for diffusers/image validation backends
- Imaging model profiles include chest X-ray focused adapters, CheXGenBench Sana (`raman07/CheXGenBench-Models-Sana-e20`) for chest-radiograph experiments, and `medisyn` (`hiesingerlab/MediSyn`) for broader text-guided medical image synthesis
- `casecrawler imaging-models` and `/api/datasets/capabilities` expose each
  image profile contract: diffusers command template, prompt inputs, generated
  `ImagingAsset` fields, and required image/report validation gates
- `BiomedCLIPImageValidator` scores generated image/report alignment when `casecrawler[imaging]` dependencies are installed
- `MedGemmaImageTextValidator` can use gated MedGemma multimodal models through `casecrawler[hf]` plus imaging dependencies for report/image consistency checks
- `casecrawler[parquet]` for parquet exports
- Time-series model profiles include TimeDiff, RawMed, and MIRA (`MIRA-Mode/MIRA`) wrappers for external generation, forecasting, or validation commands
- `casecrawler timeseries-models` and `casecrawler datasets capabilities` expose
  each external adapter contract: suggested command template, stdin JSON fields,
  expected `TimeSeriesChannel[]` stdout, and validation requirements
- Existing OpenAI, Anthropic, OpenRouter, and Ollama providers remain available for model-backed generation
- `synthetic.clinical_text_backend: llm` routes clinical document drafting through the configured LLM provider while the default deterministic backend remains no-key
- `synthetic.clinical_text_noise_profile` controls deterministic messy-note variants: `standard`, `message`, `ocr`, or `heavy`
- `synthetic.clinical_text_backend: external` wraps local or Hugging Face note generators as stdin/stdout commands and validates their returned `ClinicalDocument[]` records
- Clinical text model profiles include MedGemma (`google/medgemma-4b-it`), Meditron (`epfl-llm/meditron-7b`), and a generic external note-generator contract; `casecrawler clinical-text-models` lists the adapter metadata
- Registered Hugging Face references include BeTraC/Synth-DoPaCo (`BeTraC/betrac-2026`) for doctor-patient transcript to SOAP-note benchmarking

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
casecrawler generate-dataset "pulmonary embolism" \
  --count 50 \
  --modalities structured_ehr,clinical_text,labs,vitals,time_series,imaging \
  --age-min 45 --age-max 85 --sexes female,male
casecrawler generate-dataset "mixed acute care cohort" \
  --count 90 \
  --topic-mix "sepsis,pneumonia,heart failure exacerbation" \
  --modalities structured_ehr,clinical_text,labs,vitals,time_series
casecrawler datasets capabilities
casecrawler reference-datasets
casecrawler import-reference-dataset asclepius --dataset-id ds-asclepius-ref --limit 100
casecrawler import-reference-dataset betrac_2026 --dataset-id ds-betrac-ref --limit 100
casecrawler import-reference-dataset clinical_notes_to_fhir --dataset-id ds-fhir-ref --limit 100
casecrawler import-reference-dataset radiology_report_consistency --dataset-id ds-rad-ref --limit 100
casecrawler import-reference-dataset synthchex_75k --dataset-id ds-synthchex-ref --limit 100
casecrawler import-reference-dataset synthetic_chest_xray_pneumonia --dataset-id ds-cxr-pneumonia-ref --limit 100
casecrawler import-synthea ./synthea/output/fhir --dataset-id ds-synthea-ref
# The Synthea import accepts FHIR JSON bundles, FHIR NDJSON resource directories,
# and standard Synthea CSV directories such as output/csv with patients.csv.
casecrawler run-synthea \
  --synthea-executable ./synthea/run_synthea \
  --output-dir ./synthea/output/fhir \
  --dataset-id ds-synthea-ref \
  --population 100
casecrawler import-reference-dataset \
  --repo-id org/custom-synthetic-notes \
  --dataset-id ds-custom-ref \
  --note-field clinical_note \
  --question-field prompt \
  --answer-field completion \
  --split eval \
  --limit 100
casecrawler import-reference-dataset \
  --repo-id org/custom-image-caption-dataset \
  --dataset-id ds-custom-image-ref \
  --note-field caption \
  --image-field image \
  --image-label-field label \
  --image-label-map '{"0":"normal","1":"pneumonia"}' \
  --split train \
  --limit 100
casecrawler import-reference-dataset local-validation-notes \
  --path ./validation/local-notes.jsonl \
  --dataset-id ds-local-ref \
  --note-field clinical_note \
  --question-field prompt \
  --answer-field completion \
  --lab-values-field labs \
  --limit 100
casecrawler benchmark-dataset \
  --dataset-id <dataset_id> \
  --reference-dataset-id ds-asclepius-ref \
  --min-overall-score 0.8 \
  --min-metric-score 0.5
casecrawler datasets quality <dataset_id>
casecrawler export-dataset \
  --dataset-id <dataset_id> \
  --reference-dataset-id ds-asclepius-ref \
  --min-overall-score 0.8 \
  --min-metric-score 0.5 \
  --format sft_jsonl \
  --output train.jsonl
casecrawler export-dataset --dataset-id <dataset_id> --format note_fact_sft_jsonl --output note_facts.jsonl
casecrawler export-dataset --dataset-id <dataset_id> --format tool_call_jsonl --output tools.jsonl
casecrawler export-dataset --dataset-id <dataset_id> --format time_series_jsonl --output time_series.jsonl
casecrawler export-dataset --dataset-id <dataset_id> --format dpo_jsonl --output preference.jsonl
casecrawler export-dataset --dataset-id <dataset_id> --format rl_jsonl --output episodes.jsonl
casecrawler export-dataset --dataset-id <dataset_id> --format fhir_ndjson --output fhir.ndjson
casecrawler export-dataset --dataset-id <dataset_id> --format parquet --output records.parquet
casecrawler export-dataset-splits \
  --dataset-id <dataset_id> \
  --format clinical_observation_jsonl \
  --reference-dataset-id ds-asclepius-ref \
  --min-overall-score 0.8 \
  --min-metric-score 0.5 \
  --output-dir finetune-package
casecrawler generate-release-package "mixed acute care cohort" \
  --count 25 \
  --output-dir release-package \
  --format multimodal_jsonl \
  --seed casecrawler \
  --age-min 45 \
  --age-max 88 \
  --encounter-count 3
casecrawler verify-split-package --require-multimodal-release release-package
```

Datasets generated with `--require-human-review` are blocked from export until
each record is approved through `casecrawler reviews mark <record_id> --status
approved` or the matching REST review endpoint. Quality reports surface missing
human approvals as `human_review.missing` blockers.

`generate-release-package` is the shortest offline smoke path for a fine-tuning
ready multimodal package. It generates with the full multimodal acute-care
recipe by default, writes file-backed synthetic radiology images, seeds bundled
reference fixtures for the generated recipe, runs the benchmark gate, writes
dataset/model cards plus quality and benchmark audit reports, copies file-backed
radiology images into the package `images/` directory, exports
train/validation/test JSONL splits, and verifies strict multimodal release
readiness. The default benchmark thresholds are a non-zero bundled-fixture smoke
gate: `--min-overall-score 0.1` and `--min-metric-score 0.0`. Raise them when
benchmarking against larger imported reference datasets.

Benchmark reports compare generated cohorts to stored reference datasets and
return explicit pass/fail gates plus failing metric names. They compare across
demographics, note types, artifact density, declared-modality artifact coverage,
extracted clinical fact targets, labs, vitals, medication history, time-series
channels and backend provenance, imaging findings and backend provenance, and
approval rates. Export commands and API downloads can require the same benchmark
gate by passing a reference dataset id and thresholds, which prevents unbenchmarked
or underperforming synthetic data from silently becoming fine-tuning input.

Registered Hugging Face reference datasets include synthetic clinical notes,
doctor-patient dialogue to SOAP-note rows, clinical-note-to-FHIR rows,
radiology consistency rows, de-identification and ICD-coding references such as
Technetium-I, Synthea imports, and image-reference datasets such as
SynthCheX-75K-v2 and synthetic chest X-ray pneumonia. Custom
Hugging Face imports can map text fields, FHIR answer fields, PHI annotations,
diagnosis-code fields, image fields, image-label fields, explicit lab/vital
arrays, medication-history arrays, and time-series channel arrays into the local
`SyntheticRecord` schema. The same field mapping works for local JSONL, NDJSON,
JSON arrays, or `{"rows": [...]}` files via `--path` or the REST import `path`
field, so private validation sets can stay local.

The current external landscape and model/dataset research notes are tracked in
`docs/research/2026-05-08-synthetic-healthcare-data-landscape.md`.

## REST API

Start the server with `casecrawler serve` or `docker compose up`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ingest` | POST | Ingest content for a topic |
| `/api/ingest/{job_id}` | GET | Poll ingestion status |
| `/api/search?q=...` | GET | Search the knowledge base |
| `/api/sources` | GET | List available sources |
| `/api/datasets/capabilities` | GET | List modalities, export formats, strict release coverage requirements, validators, and model/profile adapters |
| `/api/datasets/generate` | POST | Generate synthetic healthcare records |
| `/api/datasets/release-package` | POST | Generate, benchmark, export, and return a multimodal release package zip |
| `/api/datasets/reference-catalog` | GET | List registered Hugging Face reference datasets |
| `/api/datasets/reference-import` | POST | Import registered reference datasets into local storage |
| `/api/datasets/synthea-import` | POST | Import Synthea FHIR JSON bundles, FHIR NDJSON directories, or CSV directories into local storage |
| `/api/datasets/{dataset_id}/benchmark` | GET | Compare a generated dataset to a reference dataset with configurable pass/fail thresholds |
| `/api/datasets/{dataset_id}/reference-fixtures` | POST | Seed bundled benchmark fixtures for a generated dataset recipe |
| `/api/datasets/{dataset_id}/benchmark-plan` | GET | Show recommended reference readiness for a generated dataset |
| `/api/datasets/{dataset_id}/quality` | GET | Summarize validation and fine-tuning export readiness |
| `/api/datasets/{dataset_id}/export` | GET | Stream fine-tuning/export records |
| `/api/datasets/{dataset_id}/export-splits` | GET | Download train/validation/test split package zip |

Example one-call API release package:

```bash
curl -X POST http://localhost:8000/api/datasets/release-package \
  -H 'Content-Type: application/json' \
  -o release-package.zip \
  -d '{
    "topic": "mixed acute care cohort",
    "count": 25,
    "recipe": "full_multimodal_acute_care",
    "export_format": "multimodal_jsonl",
    "seed": "casecrawler"
  }'
```

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

synthetic:
  clinical_text_backend: "deterministic" # or "llm" or "external"
  clinical_text_noise_profile: "standard" # standard, message, ocr, or heavy
  clinical_text_model_profile: null # e.g. medgemma_4b_it or meditron_7b
  clinical_text_command: null # e.g. ["hf-note-sample", "--model", "local-notes"]
  # GenerationRequest can override clinical_text_backend, llm_provider,
  # llm_model, ollama_base_url, clinical_text_noise_profile,
  # clinical_text_model_profile, and clinical_text_command for one dataset run.
  # External clinical text commands receive stdin JSON with record and must print
  # ClinicalDocument[] or {"documents": ClinicalDocument[]} to stdout.
  imaging_backend: "placeholder" # or "diffusers" or "external"
  imaging_model_profile: null # e.g. cxr_pneumonia_dreambooth
  diffusers_model_id: "stabilityai/stable-diffusion-2-1"
  imaging_command: null # e.g. ["hf-image-sample", "--model", "local-cxr"]
  # Imaging profiles declare prompt inputs, generated ImagingAsset output fields,
  # licensing/use policy, and validation gates for file integrity and alignment.
  # External imaging commands receive stdin JSON with output_dir, prompt, modality,
  # and body_region. They must print an ImagingAsset JSON object or
  # {"asset": ImagingAsset} to stdout.
  time_series_backend: "deterministic" # or "external"
  time_series_model_profile: null # e.g. timediff or rawmed
  time_series_command: null # e.g. ["timediff-sample", "--checkpoint", "local.pt"]
  # External commands receive stdin JSON with record, channels, and points.
  # They must print a JSON array of TimeSeriesChannel objects or
  # {"channels": [TimeSeriesChannel, ...]} to stdout.
  synthea_executable: null
  # GenerationRequest.cohort_constraints supports age_min, age_max, sexes,
  # sex_cycle, topic_mix, and base_time for deterministic cohort composition.
  # GenerationRequest can also override imaging_backend, imaging_model_profile,
  # and diffusers_model_id for a single dataset generation run.
  # It can also override time_series_backend, time_series_model_profile,
  # and time_series_command for external EHR time-series adapters.
  export_formats:
    - raw_jsonl
    - sft_jsonl
    - note_fact_sft_jsonl
    - chat_jsonl
    - tool_call_jsonl
    - multimodal_jsonl
    - time_series_jsonl
    - dpo_jsonl
    - rl_jsonl
    - fhir_ndjson
    - parquet

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
  generation/    # Synthetic dataset generators and backend adapters
  validation/    # Synthetic record validation
  storage/       # SQLite stores
  export/        # Fine-tuning and benchmark-ready export profiles
  api/           # FastAPI routes
```
