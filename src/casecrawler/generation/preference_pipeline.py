"""Preference-learning data pipeline for synthetic clinical records.

This is the carrier the DPO and RL exports run on. It replaces the
previous placeholder implementations -- a one-word swap and a hand-built
"reward table" -- with a real construction pattern based on RS-DPO
(Khaki et al., NAACL 2024) and RRG-DPO (MICCAI 2025) for radiology
report preference learning.

Key shape
---------

For each synthetic record we want a preference pair, the pipeline:

1. Builds an *anchor prompt* (system + user message) from the record.
2. Generates ``n_candidates`` candidate responses. The default candidate
   factory is fully deterministic and runs without any LLM provider --
   it derives variation from the record itself, so offline / CI mode
   still produces real (chosen, rejected) pairs. When a real LLM is
   wired in, callers pass an ``async_candidate_factory`` that samples
   N completions at varied temperature / system prompt; the rest of
   the pipeline is unchanged.
3. Scores each candidate with the existing :class:`SyntheticValidator`
   (clinical / privacy / utility / modality alignment) plus, when
   provided, an LLM judge. The default scorer is the validator alone;
   the judge slot is open for future wiring.
4. Selects ``chosen`` and ``rejected`` via *abnormal-aware* selection
   (RRG-DPO): if any candidate fails to surface an abnormal finding
   present in the record while another candidate covers it, the
   rejected candidate is preferred to be the abnormal-missing one.
   Otherwise the highest- and lowest-scored candidates are used.

The resulting :class:`PreferencePair` carries the prompt, the chosen /
rejected response texts, every candidate's scores, and the citations
from any RAG grounding bundle on the record. That is the contract the
``dpo_jsonl`` and ``rl_jsonl`` exporters serialize.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from casecrawler.models.synthetic import SyntheticRecord
from casecrawler.validation.synthetic_validator import SyntheticValidator

logger = logging.getLogger(__name__)


# Tunable weights for the RL reward + chosen/rejected scoring. Sum need
# not be 1; the resulting reward is normalized to [0, 1] by the validator's
# own bounds (each component already lives in [0, 1]).
DEFAULT_REWARD_WEIGHTS: dict[str, float] = {
    "clinical_consistency_score": 0.45,
    "privacy_score": 0.25,
    "utility_score": 0.15,
    "modality_alignment_score": 0.15,
}


CandidateFactory = Callable[[SyntheticRecord, int], list[str]]
AsyncCandidateFactory = Callable[
    [SyntheticRecord, int], Awaitable[list[str]]
]


class PreferenceCandidate(BaseModel):
    """A single candidate response, scored by the validator + judge."""

    text: str
    score: float = Field(ge=0.0, le=1.0)
    component_scores: dict[str, float] = Field(default_factory=dict)
    abnormal_findings_covered: int = 0
    judge_score: float | None = None
    judge_rationale: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PreferencePair(BaseModel):
    """A typed preference triple ready for DPO / KTO export."""

    record_id: str
    dataset_id: str
    prompt: list[dict[str, str]]
    chosen: PreferenceCandidate
    rejected: PreferenceCandidate
    candidates: list[PreferenceCandidate]
    citations: list[dict[str, Any]] = Field(default_factory=list)
    selection_strategy: str
    abnormal_findings: list[str] = Field(default_factory=list)
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class PreferenceConfig:
    n_candidates: int = 4
    reward_weights: dict[str, float] | None = None
    require_judge: bool = False  # noqa: D401 -- opt-in gate (future)


# --- Construction --------------------------------------------------------


def build_preference_pair(
    record: SyntheticRecord,
    *,
    config: PreferenceConfig | None = None,
    candidate_factory: CandidateFactory | None = None,
    judge: Callable[[str, SyntheticRecord], tuple[float, str]] | None = None,
    validator: SyntheticValidator | None = None,
) -> PreferencePair:
    """Synchronous entry point. Used by deterministic / offline pipelines.

    For LLM-backed sampling, see :func:`build_preference_pair_async` --
    the only difference is that the candidate factory is awaited.
    """

    cfg = config or PreferenceConfig()
    factory = candidate_factory or _deterministic_candidates
    candidates_text = factory(record, cfg.n_candidates)
    return _build_pair_from_texts(
        record=record,
        candidates_text=candidates_text,
        config=cfg,
        judge=judge,
        validator=validator,
    )


async def build_preference_pair_async(
    record: SyntheticRecord,
    *,
    config: PreferenceConfig | None = None,
    candidate_factory: AsyncCandidateFactory | None = None,
    judge: Callable[[str, SyntheticRecord], tuple[float, str]] | None = None,
    validator: SyntheticValidator | None = None,
) -> PreferencePair:
    cfg = config or PreferenceConfig()
    factory = candidate_factory or _deterministic_candidates_async
    candidates_text = await factory(record, cfg.n_candidates)
    return _build_pair_from_texts(
        record=record,
        candidates_text=candidates_text,
        config=cfg,
        judge=judge,
        validator=validator,
    )


# --- Internals -----------------------------------------------------------


def _build_pair_from_texts(
    *,
    record: SyntheticRecord,
    candidates_text: list[str],
    config: PreferenceConfig,
    judge: Callable[[str, SyntheticRecord], tuple[float, str]] | None,
    validator: SyntheticValidator | None,
) -> PreferencePair:
    if len(candidates_text) < 2:
        raise ValueError(
            "Preference construction requires at least 2 candidates; "
            f"got {len(candidates_text)}."
        )
    weights = config.reward_weights or DEFAULT_REWARD_WEIGHTS
    abnormal_findings = _abnormal_findings(record)
    candidates = [
        _score_candidate(
            text=text,
            record=record,
            weights=weights,
            judge=judge,
            abnormal_findings=abnormal_findings,
            validator=validator,
        )
        for text in candidates_text
    ]
    chosen, rejected, strategy = _select_chosen_rejected(
        candidates, abnormal_findings
    )
    citations = _record_citations(record)
    prompt = _build_prompt(record)
    return PreferencePair(
        record_id=record.record_id,
        dataset_id=record.dataset_id,
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        candidates=candidates,
        citations=citations,
        selection_strategy=strategy,
        abnormal_findings=abnormal_findings,
    )


def _score_candidate(
    *,
    text: str,
    record: SyntheticRecord,
    weights: dict[str, float],
    judge: Callable[[str, SyntheticRecord], tuple[float, str]] | None,
    abnormal_findings: list[str],
    validator: SyntheticValidator | None,
) -> PreferenceCandidate:
    # The validator scores the *record* not the candidate text; it gives
    # us a baseline score the candidate inherits. The candidate-specific
    # signal then comes from (a) the judge and (b) abnormal-coverage.
    val = validator or SyntheticValidator(threshold=0.0)
    report = val.validate(record)
    component_scores = {
        "clinical_consistency_score": float(report.clinical_consistency_score),
        "privacy_score": float(report.privacy_score),
        "utility_score": float(report.utility_score),
        "modality_alignment_score": float(report.modality_alignment_score or 0.0),
    }
    base = sum(
        component_scores[name] * weights.get(name, 0.0)
        for name in component_scores
    )
    # The pure-validator score lives in [0, sum(weights)]. Normalise to
    # [0, 1] so the reward and DPO consumers don't have to special-case
    # weight totals.
    weight_total = sum(weights.values()) or 1.0
    base = max(0.0, min(1.0, base / weight_total))

    # Abnormal-coverage bonus: each finding the candidate text mentions
    # adds a small bump capped at 0.15. This makes RRG-DPO selection
    # actually distinguish candidates even when validator scores tie.
    coverage = _count_findings(text, abnormal_findings)
    if abnormal_findings:
        base = min(1.0, base + 0.15 * (coverage / max(len(abnormal_findings), 1)))

    judge_score: float | None = None
    judge_rationale: str | None = None
    if judge is not None:
        try:
            judge_score, judge_rationale = judge(text, record)
            if judge_score is not None:
                judge_score = max(0.0, min(1.0, float(judge_score)))
                base = max(0.0, min(1.0, 0.5 * base + 0.5 * judge_score))
        except Exception:  # judge failures must not crash the pipeline
            logger.exception(
                "Preference judge raised; falling back to validator-only score."
            )

    return PreferenceCandidate(
        text=text,
        score=base,
        component_scores=component_scores,
        abnormal_findings_covered=coverage,
        judge_score=judge_score,
        judge_rationale=judge_rationale,
    )


def _select_chosen_rejected(
    candidates: list[PreferenceCandidate],
    abnormal_findings: list[str],
) -> tuple[PreferenceCandidate, PreferenceCandidate, str]:
    if abnormal_findings:
        # RRG-DPO abnormal-aware: prefer rejected to be the candidate that
        # missed the most abnormal findings, breaking ties with score.
        ordered = sorted(
            candidates,
            key=lambda c: (-c.abnormal_findings_covered, c.score),
        )
        chosen = ordered[0]
        rejected = ordered[-1]
        if chosen is not rejected and chosen.score > rejected.score:
            return chosen, rejected, "abnormal_aware"
    by_score = sorted(candidates, key=lambda c: c.score)
    chosen = by_score[-1]
    rejected = by_score[0]
    return chosen, rejected, "rs_dpo"


def _build_prompt(record: SyntheticRecord) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a clinical AI assistant trained only on synthetic "
                "healthcare records for model development. Surface abnormal "
                "findings clearly and preserve synthetic provenance."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Topic: {record.topic}. Patient: "
                f"{record.patient.age}-year-old {record.patient.sex}. "
                "Summarize the clinical facts, flag abnormal findings, and "
                "cite the labs / vitals you relied on."
            ),
        },
    ]


def _record_citations(record: SyntheticRecord) -> list[dict[str, Any]]:
    grounding = record.metadata.get("grounding")
    if not isinstance(grounding, dict):
        return []
    citations = grounding.get("citations") or []
    return [c for c in citations if isinstance(c, dict)]


def _abnormal_findings(record: SyntheticRecord) -> list[str]:
    abnormal_flags = {
        "h", "high",
        "l", "low",
        "a", "abnormal",
        "critical",
    }
    findings: list[str] = []
    for lab in record.labs:
        if (lab.flag or "").strip().lower() in abnormal_flags:
            findings.append(lab.name)
    # VitalObservation doesn't have a flag field; the structured-validator
    # plausibility checks already drive validator-side scoring, so we limit
    # the abnormal-coverage feature to flagged labs for now.
    # Stable ordering for reproducible selection.
    seen: list[str] = []
    for name in findings:
        if name not in seen:
            seen.append(name)
    return seen


_FINDING_RE_CACHE: dict[str, re.Pattern[str]] = {}


def _count_findings(text: str, findings: Sequence[str]) -> int:
    if not findings:
        return 0
    text_lower = text.lower()
    count = 0
    for name in findings:
        key = name.lower()
        pattern = _FINDING_RE_CACHE.get(key)
        if pattern is None:
            pattern = re.compile(rf"\b{re.escape(key)}\b")
            _FINDING_RE_CACHE[key] = pattern
        if pattern.search(text_lower):
            count += 1
    return count


# --- Deterministic candidate factory (offline / CI baseline) -------------


def _deterministic_candidates(record: SyntheticRecord, n: int) -> list[str]:
    """Produce ``n`` text candidates with controlled variation.

    We deliberately span the quality spectrum so the validator + abnormal
    coverage actually distinguish them:

    - high-quality: full structured summary including labs / vitals + flags
    - mid: same body without the abnormal-flag callout
    - low: ignores abnormal findings entirely (intentional rejected target)
    - terse: extra short variant (drives variation when ``n >= 4``)

    Real LLM-backed pipelines override this via ``candidate_factory=...``.
    """

    if n < 2:
        raise ValueError("n_candidates must be >= 2")
    abnormal_summary = ", ".join(_abnormal_findings(record)) or "none flagged"
    labs_summary = ", ".join(
        f"{lab.name}={lab.value}{lab.unit or ''}" for lab in record.labs[:4]
    )
    vitals_summary = ", ".join(
        f"{v.name}={v.value}{v.unit or ''}" for v in record.vitals[:4]
    )
    high = (
        f"Synthetic {record.topic} record. {record.patient.age}-year-old "
        f"{record.patient.sex}. Abnormal findings: {abnormal_summary}. "
        f"Key labs: {labs_summary}. Vitals: {vitals_summary}. "
        "Use is limited to model development on synthetic data."
    )
    mid = (
        f"Synthetic {record.topic} record. {record.patient.age}-year-old "
        f"{record.patient.sex}. Key labs: {labs_summary}. "
        f"Vitals: {vitals_summary}. Synthetic provenance preserved."
    )
    low = (
        f"Patient with {record.topic}. Age {record.patient.age}. "
        "No specific findings to highlight. Routine review."
    )
    terse = (
        f"{record.topic} -- {record.patient.age}{record.patient.sex[:1]}. "
        f"Findings: {abnormal_summary}."
    )
    pool = [high, mid, low, terse]
    while len(pool) < n:
        pool.append(f"{high} (variant {len(pool)})")
    return pool[:n]


async def _deterministic_candidates_async(
    record: SyntheticRecord, n: int
) -> list[str]:
    return _deterministic_candidates(record, n)
