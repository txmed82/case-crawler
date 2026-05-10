"""Curated cheap-judge defaults per LLM provider.

Used by:

- :func:`recommend_judges` -- backs the ``casecrawler suggest-judges`` CLI.
- :func:`warn_if_judge_collides_with_generator` -- emits a warning when
  the configured judge provider matches the generator provider, since
  self-judging skews preference data.

This is the deliberately offline path. A future PR can add a
``casecrawler refresh-judges`` command that uses the configured LLM's
web tools to update this list against current pricing pages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JudgeRecommendation:
    provider: str
    model: str
    notes: str
    is_default: bool = False


_CURATED: dict[str, list[JudgeRecommendation]] = {
    "anthropic": [
        JudgeRecommendation(
            provider="anthropic",
            model="claude-haiku-4-5",
            notes="cheap, fast, strong instruction-following",
            is_default=True,
        ),
        JudgeRecommendation(
            provider="anthropic",
            model="claude-sonnet-4-6",
            notes="higher quality if budget allows",
        ),
    ],
    "openai": [
        JudgeRecommendation(
            provider="openai",
            model="gpt-4.1-mini",
            notes="cheap-tier with native json_schema response_format",
            is_default=True,
        ),
        JudgeRecommendation(
            provider="openai",
            model="gpt-5-mini",
            notes="successor cheap tier; check availability per account",
        ),
    ],
    "openrouter": [
        JudgeRecommendation(
            provider="openrouter",
            model="anthropic/claude-haiku-4-5",
            notes="route to Anthropic via OpenRouter for unified billing",
            is_default=True,
        ),
        JudgeRecommendation(
            provider="openrouter",
            model="meta-llama/llama-3.3-70b-instruct",
            notes="fully open-weights; cheapest tier on most aggregators",
        ),
        JudgeRecommendation(
            provider="openrouter",
            model="qwen/qwen2.5-72b-instruct",
            notes="strong instruction-following; competitive pricing",
        ),
    ],
    "ollama": [
        JudgeRecommendation(
            provider="ollama",
            model="medgemma:4b",
            notes="local, free; medical-tuned, recommended for clinical judging",
            is_default=True,
        ),
        JudgeRecommendation(
            provider="ollama",
            model="llama3.1:8b",
            notes="local, free; general-purpose",
        ),
        JudgeRecommendation(
            provider="ollama",
            model="qwen2.5:7b",
            notes="local, free; strong long-context handling",
        ),
    ],
}


def recommend_judges(provider: str | None = None) -> list[JudgeRecommendation]:
    """Return curated judge recommendations.

    With no ``provider`` argument, returns the default model for every
    provider. With a provider, returns every recommendation for that
    provider (defaults first).
    """
    if provider:
        key = provider.strip().lower()
        if key not in _CURATED:
            raise KeyError(
                f"Unknown provider {provider!r}. Known: "
                f"{', '.join(sorted(_CURATED))}"
            )
        return list(_CURATED[key])
    return [recs[0] for recs in _CURATED.values() if recs]


def warn_if_judge_collides_with_generator(
    *,
    judge_provider: str | None,
    generator_provider: str | None,
) -> str | None:
    """If judge and generator share a provider, log a warning + return it.

    Returning the warning string lets callers (CLI, API) surface it to the
    user without having to capture log output. Self-judging is a known
    bias in the DPO / preference-learning literature; we don't refuse to
    proceed (the user may have a good reason) but we make sure they
    notice.
    """
    if not judge_provider or not generator_provider:
        return None
    if judge_provider.strip().lower() != generator_provider.strip().lower():
        return None
    message = (
        f"synthetic.judge.provider={judge_provider!r} matches "
        f"llm.provider={generator_provider!r}. Self-judging biases "
        "preference pairs toward the generator's stylistic quirks rather "
        "than clinical correctness. Configure a different provider for "
        "the judge -- see `casecrawler suggest-judges` for cheap options."
    )
    logger.warning(message)
    return message
