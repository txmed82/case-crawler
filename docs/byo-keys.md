# BYO Keys

CaseCrawler is open source and BYO-key. There is no proxy; every external
call goes from your machine to your provider against your account, and
you pay them directly. The offline path needs no keys at all — see
[Getting Started](getting-started.md) — but adding keys unlocks richer
generation, imaging, and judging.

This page covers each key, what it unlocks, where to put it, and how to
verify it works.

## Don't commit your keys

`.env` and any other file holding secrets must never be checked in.
The repo's `.gitignore` excludes `.env` by default; verify with:

```bash
git check-ignore -v .env       # should print a .gitignore line
git log --all --source -- .env # should return nothing
```

If a key has already been committed, **rotate it at the provider**
(revoke the leaked key, issue a new one) before scrubbing history —
git history rewrites are slow and don't help once the key has been
seen. For CI and production, prefer platform secret managers (GitHub
Actions Secrets, Vercel environment variables, AWS Secrets Manager,
etc.) over checked-in files.

## At a glance

| Key | What it unlocks | Where to put it |
|-----|-----------------|-----------------|
| `ANTHROPIC_API_KEY` | LLM-backed clinical text, LLM judges | `.env` |
| `OPENAI_API_KEY` | LLM-backed clinical text, LLM judges | `.env` |
| `OPENROUTER_API_KEY` | LLM-backed clinical text + model variety | `.env` |
| `OLLAMA_BASE_URL` | Fully-local LLM-backed generation | `.env` |
| `HF_TOKEN` | Gated imaging models, gated datasets, `suggest-imaging-models` | `.env` |
| `GLASS_API_KEY`, `ANNAS_ARCHIVE_API_KEY`, `FIRECRAWL_API_KEY` | Paid ingestion sources | `.env` |
| `NCBI_API_KEY`, `OPENFDA_API_KEY` | Higher rate limits on free sources | `.env` |

Run `casecrawler sources` to confirm which sources your current `.env`
exposes.

## LLM provider keys

CaseCrawler talks to four LLM provider shapes through the same interface:
Anthropic, OpenAI, OpenRouter (OpenAI-compatible), and Ollama (local).

### Anthropic

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
```

Get a key at https://console.anthropic.com/settings/keys. Then in
`config.yaml`:

```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-6
```

### OpenAI

```bash
# .env
OPENAI_API_KEY=sk-...
```

Get a key at https://platform.openai.com/api-keys. Then:

```yaml
llm:
  provider: openai
  model: gpt-4.1
```

Structured generation uses native `response_format={"type": "json_schema"}`
where supported, with automatic fallback to `json_object` for older
endpoints — you don't have to configure that.

### OpenRouter

OpenRouter exposes many models behind a single OpenAI-compatible API.
Useful when you want to compare model families without juggling four keys.

```bash
# .env
OPENROUTER_API_KEY=sk-or-...
```

```yaml
llm:
  provider: openrouter
  model: anthropic/claude-sonnet-4-6
```

### Ollama (local, no cloud key)

If you have Ollama running locally, no API key is needed — just point
CaseCrawler at it:

```bash
# .env
OLLAMA_BASE_URL=http://localhost:11434
```

```yaml
llm:
  provider: ollama
  model: llama3.1
```

`casecrawler` will surface `provider=ollama` in pipeline summaries so
you can confirm runs stayed local.

## Hugging Face token

Set `HF_TOKEN` when you want any of:

- Gated medical imaging models like MediSyn (`hiesingerlab/MediSyn`),
  MedGemma multimodal validators, or any Hub repo behind a license click-through.
- Gated reference datasets (some of the registered HF references gate
  their licenses).
- `casecrawler suggest-imaging-models` to enumerate gated repos rather
  than silently skip them.

```bash
# .env
HF_TOKEN=hf_...
```

Create the token at https://huggingface.co/settings/tokens with `read`
scope. The token is read on demand — there is no login dance and nothing
is cached outside your process.

### Picking an imaging model

`casecrawler suggest-imaging-models` queries the HF Hub for medical
imaging models by modality, ranked by downloads:

```bash
casecrawler suggest-imaging-models --modality chest_xray --limit 10
casecrawler suggest-imaging-models --modality pathology --limit 5
casecrawler suggest-imaging-models --modality mri --pipeline-tag text-to-image
```

The output shows each repo's license, gated-ness, and last-modified date.
The command is **print-only** — it never writes your config. Pick a repo,
read the license, then pin it yourself:

```yaml
# config.yaml
synthetic:
  imaging_backend: diffusers
  diffusers_model_id: raman07/CheXGenBench-Models-Sana-e20
  imaging_model_profile: cxr_sana_chexgenbench
```

For external (non-diffusers) backends, point CaseCrawler at a shell
command that reads stdin JSON and prints an `ImagingAsset` to stdout:

```yaml
synthetic:
  imaging_backend: external
  imaging_command: ["hf-image-sample", "--model", "local-cxr"]
```

## Paid ingestion sources

| Key | Source | Why bother |
|-----|--------|------------|
| `GLASS_API_KEY` | Glass Health | Curated clinical reasoning content with explicit case structure |
| `ANNAS_ARCHIVE_API_KEY` | Anna's Archive | Full-text papers and medical textbooks |
| `FIRECRAWL_API_KEY` | Firecrawl | Guidelines and unstructured web content |

The free sources (PubMed, OpenFDA, DailyMed, RxNorm, medRxiv,
ClinicalTrials.gov) cover the default ingestion path. Add paid keys when
you want depth in specific topics that the free sources don't cover well.

## Free-source rate-limit keys

These don't change what data you can access — they just raise per-source
rate limits, which matters when you ingest large topic mixes.

- `NCBI_API_KEY`: 10 PubMed reqs/sec instead of 3. Get one at
  https://www.ncbi.nlm.nih.gov/account/settings/.
- `OPENFDA_API_KEY`: 240 reqs/min instead of 40. Get one at
  https://open.fda.gov/apis/authentication/.

## Verifying

```bash
casecrawler sources    # which ingestion sources are unlocked
casecrawler config     # which LLM provider + model is configured
```

If a key is missing from `.env`, the corresponding feature path silently
degrades (e.g. clinical text falls back to the deterministic backend; the
DPO judge slot is left unfilled). Nothing crashes — you just don't get
the LLM-enhanced output.

## Costs

See [Costs and tokens](cost-and-tokens.md) for rough tokens-per-case
estimates so you can budget LLM-backed runs before you start them.
