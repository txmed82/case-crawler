# Open Source Roadmap

CaseCrawler is an open-source synthetic healthcare data platform for research,
training-pipeline prototyping, benchmark construction, and release-package
experimentation. It is not a medical device, and generated outputs require
validation before any downstream clinical or operational use.

## Maturity Levels

### Level 1: Offline Synthetic Records

- Deterministic structured EHR records.
- Clinical notes with controlled variation.
- Schema, privacy, and clinical-consistency validation.
- Basic JSONL, FHIR NDJSON, parquet, and fine-tuning exports.

### Level 2: Evidence-Aware Dataset Generation

- Grounding metadata from public references and local fixtures.
- Public reference fixture import for benchmark comparisons.
- Benchmark profile scoring.
- Human review queues and review summaries.

### Level 3: Training-Ready Release Packages

- Train, validation, and test split packages.
- Dataset cards, model cards, manifests, checksums, and provenance.
- Objective coverage audits.
- Strict placeholder rejection for release-gated multimodal packages.
- Export transparency summaries that describe record origins and limitations.

### Level 4: Clinician-Reviewed Benchmark Suites

- Clinician-reviewed golden cases.
- Specialty-specific coverage targets.
- Condition-specific clinical content packs.
- Contributor-maintained benchmark and validation fixtures.

### Level 5: Multimodal Research Platform

- Real image backend contracts and policy metadata.
- Time-series model adapters.
- Image/report alignment validators.
- Optional external judge support.
- Public evaluation suites for generated release packages.

## Current Strengths

- No-key deterministic generation path for local development.
- Multimodal record model covering structured EHR, notes, labs, vitals,
  medications, allergies, orders, time series, and imaging metadata.
- Fine-tuning export profiles for SFT, chat, tool use, note-fact extraction,
  clinical observations, medication reconciliation, multimodal tasks,
  time-series tasks, DPO/RL, FHIR NDJSON, and parquet.
- Release-readiness gates with quality reports, audit artifacts, and human
  review workflow support.
- Pull-request CI tiers for backend, UI, and optional backend coverage.

## Near-Term Priorities

1. Convert hard-coded clinical profiles into contributor-friendly clinical
   content packs.
2. Expand clinical coverage validation for condition-specific required
   artifacts.
3. Grow public benchmark-suite fixtures and golden regression cases.
4. Improve UI visibility into release readiness, human review status, and
   benchmark failures.
5. Add more examples for open-source contributors building new conditions,
   model adapters, and export profiles.

## Out Of Scope

- Real patient data ingestion for training datasets.
- Clinical decision support claims.
- Automated diagnosis, treatment recommendation, or patient-care workflows.
- Shipping generated data without provenance, validation, and review metadata.

## Contributor Entry Points

- [Getting started](getting-started.md)
- [Testing](testing.md)
- [Contributing clinical content](contributing-clinical-packs.md)
- [Validation and benchmarking](validation-and-benchmarking.md)
- [Release packages](release-packages.md)
