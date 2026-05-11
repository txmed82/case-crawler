# Costs and tokens

CaseCrawler is BYO-key. Nothing is proxied; your provider bills you
directly. This page gives rough orders of magnitude so you can budget a
run before you start it.

> **Numbers are approximate.** They depend on the topic, complexity
> profile, modality mix, and whether grounding is enabled. Treat the
> figures below as a sanity check, not a quote.

## When tokens are spent

The default offline path spends **zero tokens** — the deterministic
clinical text backend and the deterministic candidate factory in the
preference pipeline don't call any LLM. Tokens are spent only when:

1. `synthetic.clinical_text_backend: llm` — every generated note is an
   LLM call.
2. The DPO preference pipeline runs with an LLM judge (the judge slot
   is empty by default).
3. You pass `--judge` to a generate-release-package or DPO export run.

If none of those are on, you're free.

## Rough tokens per case

For a *standard moderate-complexity record* with one ED note, one
progress note, and one discharge summary (the default
`full_multimodal_acute_care` recipe):

| Operation | Input tokens | Output tokens | Notes |
|-----------|-------------:|--------------:|-------|
| 1 clinical note (LLM backend) | ~1,200 | ~700 | Prompt includes patient summary, structured facts, and grounding excerpts |
| 3-note record (typical) | ~3,600 | ~2,100 | The structured patient context is reused, so this isn't 3× the single-note cost in practice — closer to 2.5× |
| LLM judge per DPO candidate | ~1,800 | ~150 | Judge sees the prompt + candidate; output is a short rationale + score |
| DPO pair (4 candidates, judged) | ~9,000 | ~600 | 4 judge calls × per-candidate cost |

Multiply by your `--count` and recipe complexity. A 100-record dataset
with LLM-backed notes and no judge is **~600k input + ~210k output**
tokens. With a 4-candidate judged DPO export on the same dataset, add
**~900k input + ~60k output** tokens.

## Rough dollar estimates (May 2026)

These are list prices; volume discounts and free tiers vary. **Verify
current prices** at:
[Anthropic](https://www.anthropic.com/pricing) ·
[OpenAI](https://openai.com/api/pricing/) ·
[OpenRouter](https://openrouter.ai/models) ·
[Ollama](https://ollama.com/) (self-hosted, no per-token pricing).

| Provider / model | 600k in + 210k out (text only) | + DPO judging |
|------------------|-------------------------------:|--------------:|
| [Anthropic Claude Sonnet 4.6](https://www.anthropic.com/pricing) (`$3 / $15` per Mtok) | **~$5.00** | ~$5.00 + ~$3.60 |
| [OpenAI GPT-4.1](https://openai.com/api/pricing/) (`$2 / $8` per Mtok) | **~$2.90** | ~$2.90 + ~$2.30 |
| [OpenRouter Claude Sonnet 4.6](https://openrouter.ai/models) | same as Anthropic | same |
| [Ollama](https://ollama.com/) (local) | **$0** (your hardware) | $0 |

For most exploration, **Ollama or a small OpenAI model is fine**.
Reach for Claude Sonnet when you need clinical fidelity in the notes
and your downstream evaluation rewards it.

## Reducing cost

- Stay on the deterministic backend until you have an evaluation signal
  that says LLM notes are buying you something.
- Use `--max-validation-retries 0` while iterating. Retries multiply
  cost.
- For DPO exports, use `recommend_judges()` (in
  `casecrawler.generation.judges`) — it suggests a cheap judge model
  per provider so you don't accidentally judge with the same expensive
  model that wrote the candidate.
- Cap `n_candidates` in the preference pipeline. Two candidates per
  record halves the judge cost vs four with little impact on the
  quality of the chosen/rejected pair when the validator-only scoring
  already separates them.

## What CaseCrawler doesn't bill for

- Image generation (HF diffusers): you spend GPU time, not tokens.
  Local diffusers models are free at inference; gated hosted endpoints
  bill per-image.
- Ingestion: free sources (PubMed, OpenFDA, DailyMed, etc.) are free.
  Paid sources (Glass, Anna's Archive, Firecrawl) bill per-request.
- Validation: validators run locally on your CPU.
