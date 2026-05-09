# Getting Started

## Install

```bash
git clone https://github.com/txmed82/case-crawler.git
cd case-crawler
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[hf,imaging,parquet]"
```

Use `hf` for Hugging Face dataset/model helpers, `imaging` for image generation and image-text validators, and `parquet` for parquet export.

## Generate A Dataset Without API Keys

```bash
casecrawler generate-dataset "sepsis" --count 10
casecrawler datasets
casecrawler datasets quality <dataset_id>
```

The default path is deterministic and offline. It creates structured EHR facts, labs, vitals, clinical documents, validation reports, and export-ready records without requiring an LLM key.

## Generate A Multimodal Release Package

```bash
casecrawler generate-release-package "mixed acute care cohort" \
  --count 25 \
  --max-validation-retries 2 \
  --output-dir release-package \
  --format multimodal_jsonl \
  --seed casecrawler

casecrawler verify-split-package --require-multimodal-release release-package
```

`generate-release-package` runs the full multimodal recipe, seeds bundled reference fixtures, runs benchmark gates, writes train/validation/test splits, copies file-backed image and time-series artifacts, creates dataset/model cards, and verifies the package.

## Run The API And UI

```bash
casecrawler serve
```

Or with Docker:

```bash
cp .env.example .env
docker compose up
```

The API defaults to `http://localhost:8000`. The frontend is served by the Docker stack.

## Human Review

Datasets generated with `--require-human-review` are blocked from export until records are approved:

```bash
casecrawler generate-dataset "sepsis" --count 5 --require-human-review
casecrawler reviews queue --dataset-id <dataset_id>
casecrawler reviews mark <record_id> --status approved --reviewer clinical-reviewer
```
