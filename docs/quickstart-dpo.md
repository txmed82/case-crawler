# Quickstart: DPO and RL exports

This is the short path from "I have a generated dataset" to "I have
`preference.jsonl` ready for DPO training" (or `episodes.jsonl` for RL).

Both exports run on the same preference pipeline — see
`casecrawler/generation/preference_pipeline.py` for the contract.

## Prerequisites

- A generated dataset (`dataset_id` of any
  `casecrawler generate-dataset` or `generate-release-package` run).
- Optional: an LLM key and `synthetic.clinical_text_backend: llm` if
  you want non-deterministic candidate variation. The default
  deterministic candidate factory works without keys.
- Optional: a judge LLM if you want LLM scoring on top of the
  validator. The validator-only path is free and fast.

## Minimal DPO export

```bash
# 1. Generate a dataset (offline, deterministic — no keys needed).
casecrawler generate-dataset "sepsis" --count 50

# 2. Note the dataset_id from the output.
casecrawler datasets   # lists dataset_ids

# 3. Export DPO pairs.
casecrawler export-dataset \
  --dataset-id <dataset_id> \
  --format dpo_jsonl \
  --output preference.jsonl
```

Each line in `preference.jsonl` is:

```json
{
  "prompt": {"system": "...", "user": "..."},
  "chosen": "candidate text the validator scored highest",
  "rejected": "candidate text the validator scored lowest",
  "scores": {"chosen": {...}, "rejected": {...}},
  "citations": [...],
  "metadata": {"record_id": "...", "export_profile": "dpo_jsonl"}
}
```

Drop straight into the trainer of your choice (TRL, axolotl, etc.) —
the field names are the standard DPO shape.

## Minimal RL export

```bash
casecrawler export-dataset \
  --dataset-id <dataset_id> \
  --format rl_jsonl \
  --output episodes.jsonl
```

`rl_jsonl` uses the same preference pipeline but writes one row per
candidate with a scalar reward derived from the validator components
(see `DEFAULT_REWARD_WEIGHTS`). Suitable for PPO/GRPO style trainers
that consume `(prompt, response, reward)` tuples.

## Selecting candidates

Two selection strategies are baked in:

- **Score-based** (default): highest validator score wins as `chosen`,
  lowest wins as `rejected`.
- **Abnormal-aware** (RRG-DPO style): if any candidate fails to surface
  an abnormal clinical finding present in the record while another
  covers it, the abnormal-missing candidate is preferred as `rejected`
  regardless of the score tie. Useful for radiology / report tasks
  where "noticing the abnormality" is the thing you care about.

The abnormal-aware path triggers automatically when the record has
abnormal lab/vital findings and the candidates differ in coverage.

## Adding an LLM judge

The validator-only scorer is the default. To add an LLM judge:

```python
from casecrawler.generation.judges import recommend_judges
from casecrawler.generation.preference_pipeline import (
    build_preference_pair,
)

# Get a cheap judge for your generator's provider.
judge_provider, judge_model = recommend_judges("anthropic")[0]
# ... wire judge_provider/judge_model into your candidate factory or
# directly into build_preference_pair.
```

`recommend_judges()` deliberately avoids returning the same model that
generated the candidates — judging your own output is a known
preference-leak path.

## DPO from a release package

`generate-release-package` doesn't write DPO/RL by default. Once a
release package is built, the dataset_id is in
`release-package/manifest.json`; pass it to `export-dataset` with
`--format dpo_jsonl` as above.

## Costs

A 4-candidate judged DPO export of 100 records costs roughly $3-$4 on
Claude Sonnet 4.6 or $2-$3 on GPT-4.1. See
[Costs and tokens](cost-and-tokens.md) for more.

If you don't want judge cost, skip the judge — the validator-only path
is what every example above runs.
