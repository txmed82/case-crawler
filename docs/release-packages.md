# Release Packages

A release package is the strictest CaseCrawler output. It is designed to be handed to a training or evaluation pipeline with enough metadata to inspect provenance, benchmark results, and safety status.

## Create One

```bash
casecrawler generate-release-package "mixed acute care cohort" \
  --count 25 \
  --max-validation-retries 2 \
  --output-dir release-package \
  --format multimodal_jsonl \
  --seed casecrawler

casecrawler verify-split-package --require-multimodal-release release-package
```

## Contents

Release packages include:

- `manifest.json`: split metadata, file checksums, task coverage, image artifacts, and time-series artifacts.
- `train.jsonl`, `validation.jsonl`, `test.jsonl`: deterministic splits.
- `images/`: file-backed radiology image artifacts.
- `time_series/`: exported time-series channel artifacts.
- `quality_report.json`: quality, coverage, policy, and release-readiness report.
- `benchmark_profile.json`: generated dataset profile.
- `benchmark_report.json`: primary reference comparison.
- `benchmark_suite_report.json`: recommended reference suite comparison.
- `dataset_card.md`: dataset card for release review.
- `model_card.md`: model/training-use card.
- `release_package_summary.json`: compact release summary and objective coverage audit.

## Hard Release Gate

`--require-multimodal-release` verifies that the package has:

- approved records
- structured EHR facts
- labs and lab reports
- vitals and vital-sign flowsheets
- medication history and medication administration records
- allergies and clinical orders
- messy clinical text
- physician and nursing notes
- time-series channels
- radiology reports and file-backed radiology images
- image/report alignment scores
- clinical text and imaging model policy metadata
- validation reports
- benchmark references
- task-specific reference coverage
- no blocking quality issues
- required audit artifacts
- complete objective coverage

The verifier rejects release packages with missing audit artifacts, invalid checksums, incomplete objective coverage, failed benchmarks, missing image metadata, or inconsistent release-readiness flags.

## Retry Generation

Use validation retries to regenerate records that fail validation before they enter the package:

```bash
casecrawler generate-release-package "sepsis" \
  --count 50 \
  --max-validation-retries 3 \
  --output-dir release-package \
  --format multimodal_jsonl
```

Each record stores retry metadata so downstream audits can identify whether regeneration occurred.
