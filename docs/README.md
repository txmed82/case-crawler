# CaseCrawler Documentation

CaseCrawler is an open-source synthetic healthcare data engine for clinical AI training and evaluation. It generates dataset-first synthetic records, validates them, compares them to reference datasets, and exports fine-tuning-ready packages.

## Core Guides

- [Getting started](getting-started.md): install, run the no-key workflow, generate a release package, and verify it.
- [Architecture](architecture.md): how generation, validation, storage, adapters, API, UI, and exports fit together.
- [Release packages](release-packages.md): what a release-ready multimodal package contains and how the hard gate works.
- [Validation and benchmarking](validation-and-benchmarking.md): quality reports, benchmark references, objective coverage, privacy checks, and human review.
- [Reference data and model adapters](reference-data-and-models.md): Hugging Face datasets, Synthea, imaging profiles, clinical text adapters, time-series adapters, and validators.
- [CLI and API guide](api-and-cli.md): commands and REST endpoints for generation, import, benchmark, review, export, and verification.

## Planning And Research

- [Healthcare synthetic data platform implementation plan](superpowers/plans/2026-05-06-healthcare-synthetic-data-platform.md)
- [Synthetic healthcare data landscape research](research/2026-05-08-synthetic-healthcare-data-landscape.md)

## Main Artifacts

- `SyntheticRecord`: dataset-first patient record with structured EHR, documents, labs, vitals, time series, imaging, provenance, and validation.
- `GenerationRequest`: topic, count, modalities, recipe, cohort constraints, model backends, validation threshold, and retry settings.
- `DatasetQualityReport`: export readiness, release readiness, artifact coverage, modality distributions, benchmark readiness, and blockers.
- Release package: train/validation/test JSONL splits plus images, time-series files, manifest, quality report, benchmark reports, dataset card, model card, and objective coverage audit.
