# CLI And API Guide

## CLI Overview

```bash
casecrawler --help
```

Important commands:

- `generate-dataset`: generate synthetic healthcare records.
- `generate-release-package`: generate, benchmark, export, and verify a multimodal package.
- `verify-split-package`: verify split package integrity and release readiness.
- `datasets`: list, inspect, quality-check, benchmark-plan, and manage datasets.
- `reviews`: queue and mark human review decisions.
- `export-dataset`: export a stored dataset to one file.
- `export-dataset-splits`: export train/validation/test split packages.
- `benchmark-dataset`: compare a generated dataset against a reference dataset.
- `reference-datasets`: list reference datasets.
- `import-reference-dataset`: import Hugging Face or local references.
- `import-synthea` and `run-synthea`: import or run Synthea output.
- `imaging-models`, `clinical-text-models`, `timeseries-models`: inspect adapter contracts.

## Common CLI Workflows

Generate and export:

```bash
casecrawler generate-dataset "heart failure exacerbation" --count 100 --complexity complex
casecrawler datasets quality <dataset_id>
casecrawler export-dataset --dataset-id <dataset_id> --format sft_jsonl --output train.jsonl
```

Generate a multimodal cohort:

```bash
casecrawler generate-dataset "pulmonary embolism" \
  --count 50 \
  --modalities structured_ehr,clinical_text,labs,vitals,time_series,imaging \
  --age-min 45 \
  --age-max 85 \
  --sexes female,male
```

Create a release package:

```bash
casecrawler generate-release-package "mixed acute care cohort" \
  --count 25 \
  --output-dir release-package \
  --format multimodal_jsonl \
  --max-validation-retries 2

casecrawler verify-split-package --require-multimodal-release release-package
```

Import references and benchmark:

```bash
casecrawler import-reference-dataset synthchex_75k --dataset-id ds-synthchex-ref --limit 100
casecrawler benchmark-dataset \
  --dataset-id <dataset_id> \
  --reference-dataset-id ds-synthchex-ref \
  --min-overall-score 0.8 \
  --min-metric-score 0.5
```

## API Overview

Start the server:

```bash
casecrawler serve
```

Core endpoints:

| Endpoint | Method | Purpose |
|---|---:|---|
| `/api/datasets/capabilities` | GET | Modalities, export formats, release requirements, model profiles, validators |
| `/api/datasets/generate` | POST | Generate synthetic records |
| `/api/datasets/release-package` | POST | Generate and download a release package zip |
| `/api/datasets/reference-catalog` | GET | List reference datasets |
| `/api/datasets/reference-import` | POST | Import Hugging Face or local references |
| `/api/datasets/synthea-import` | POST | Import Synthea output |
| `/api/datasets/{dataset_id}/quality` | GET | Dataset quality and export readiness |
| `/api/datasets/{dataset_id}/benchmark` | GET | Benchmark against a reference dataset |
| `/api/datasets/{dataset_id}/benchmark-plan` | GET | Recommended reference readiness |
| `/api/datasets/{dataset_id}/reference-fixtures` | POST | Seed bundled reference fixtures |
| `/api/datasets/{dataset_id}/export` | GET | Stream export records |
| `/api/datasets/{dataset_id}/export-splits` | GET | Download split package zip |
| `/api/datasets/{dataset_id}/reviews` | GET | Review queue |
| `/api/records/{record_id}/review` | POST | Save review decision |

Example release-package API call:

```bash
curl -X POST http://localhost:8000/api/datasets/release-package \
  -H 'Content-Type: application/json' \
  -o release-package.zip \
  -d '{
    "topic": "mixed acute care cohort",
    "count": 25,
    "recipe": "full_multimodal_acute_care",
    "export_format": "multimodal_jsonl",
    "max_validation_retries": 2,
    "seed": "casecrawler"
  }'
```
