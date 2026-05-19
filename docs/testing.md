# Testing

Case Crawler uses a pull-request test tier for fast feedback and a scheduled
optional-backend tier for heavier adapter coverage.

## Backend

Install the package with development extras and run the default backend tier:

```bash
python -m pip install -e ".[dev,anthropic,openai,hf]"
python -m pytest -q -m "not optional_backend and not network and not slow"
```

Run the full local suite, including optional-backend marker tests:

```bash
python -m pytest -q
```

## Optional Backends

Optional backend tests exercise heavier adapters and integration contracts. They
are not required for every pull request, but should pass before release work:

```bash
python -m pip install -e ".[dev,anthropic,openai,hf,parquet]"
python -m pytest -q -m "optional_backend or slow"
```

## Network Tests

Network tests are opt-in and must use the `network` marker. Keep them out of the
default pull-request tier unless the test uses a mock transport.

```bash
CASECRAWLER_RUN_NETWORK_TESTS=1 python -m pytest -q -m network
```

## UI

The UI tier runs from the `ui/` directory:

```bash
cd ui
npm ci
npm run lint
npm run build
```

## Release Package Smoke

Use this smoke test before publishing sample generated data:

```bash
casecrawler generate-release-package "mixed acute care cohort" \
  --count 5 \
  --max-validation-retries 1 \
  --output-dir .context/release-smoke \
  --format sft_jsonl

casecrawler verify-split-package .context/release-smoke
```
