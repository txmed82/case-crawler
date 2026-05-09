# Validation And Benchmarking

CaseCrawler treats validation as a core product surface, not a post-processing script.

## Record Validation

`SyntheticValidator` checks generated records for:

- schema completeness
- clinical consistency
- lab flags and reference ranges
- vital plausibility
- temporal consistency
- privacy and PHI-like leakage
- utility for downstream training tasks
- optional image/report alignment

Generation can retry failed records with `--max-validation-retries`.

## Quality Reports

```bash
casecrawler datasets quality <dataset_id>
```

Quality reports summarize:

- approval rate and blocking issues
- modality counts
- artifact density
- labs, vitals, medications, allergies, orders, notes, imaging, and time-series coverage
- noisy text coverage
- model policy metadata
- export profile readiness
- multimodal release readiness

## Benchmark Datasets

Generated datasets can be compared against stored references:

```bash
casecrawler benchmark-dataset \
  --dataset-id <generated_dataset_id> \
  --reference-dataset-id <reference_dataset_id> \
  --min-overall-score 0.8 \
  --min-metric-score 0.5
```

Benchmark reports compare distributions and task coverage across:

- record counts and demographics
- note types and document density
- labs, vitals, medication history, allergies, orders
- time-series channels and backend provenance
- imaging findings, labels, dimensions, backend provenance, and validator policy
- extracted clinical fact targets
- approval rates and blocking issues

## Recommended Reference Suite

Recipes declare recommended references. CaseCrawler can seed bundled fixtures for smoke tests and can import Hugging Face or local references for stronger gates.

```bash
casecrawler datasets benchmark-plan <dataset_id>
casecrawler datasets seed-reference-fixtures <dataset_id>
casecrawler export-dataset-splits \
  --dataset-id <dataset_id> \
  --auto-benchmark \
  --require-multimodal-release \
  --format multimodal_jsonl \
  --output-dir package
```

## Objective Coverage Audit

Release packages include an objective coverage audit that maps the platform goal to evidence:

- records
- structured EHR
- labs
- vitals
- medication history
- allergies
- clinical orders
- messy clinical text
- physician notes
- nursing notes
- time series
- radiology reports
- radiology images
- privacy safety
- validation references
- fine-tuning exports
- release audit artifacts
- cohort similarity

The package verifier treats incomplete objective coverage as a release failure when the package claims multimodal release readiness.
