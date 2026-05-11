# Examples

Drop-in starter configurations. Copy whichever matches your use case to
the project root as `config.yaml`, edit, and run.

| Config | Use case |
|--------|----------|
| [`configs/minimal-rag.yaml`](configs/minimal-rag.yaml) | Text-only synthetic records grounded in retrieved literature; no imaging, no LLM keys required |
| [`configs/minimal-imaging.yaml`](configs/minimal-imaging.yaml) | Chest-radiograph synthetic records via Hugging Face diffusers; needs `HF_TOKEN` for gated models |
| [`configs/dpo-export.yaml`](configs/dpo-export.yaml) | Generate + export `dpo_jsonl` / `rl_jsonl` preference data ready for TRL / axolotl trainers |

All three pass `AppConfig` validation as written — CI exercises them.
