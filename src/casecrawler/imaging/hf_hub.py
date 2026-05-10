"""Hugging Face Hub helpers for imaging adapters.

This module is the single place that talks to the HF Hub for the imaging
stack. It covers two needs:

1. ``fetch_model_card_metadata(repo_id, token=...)`` -- pull a repo's model
   card YAML so we can surface its license / use-policy / gated status on
   every generated image's metadata. Auto-fetched per the open-source
   thesis: we don't curate licenses by hand for arbitrary user-chosen
   models.

2. ``suggest_imaging_models(modality, ...)`` -- query the Hub's public
   ``/api/models`` endpoint for medical-imaging models, ranked by recent
   downloads, and present a ranked list. Backs the
   ``casecrawler suggest-imaging-models`` CLI.

Both helpers degrade gracefully when ``huggingface_hub`` isn't installed
(this is an optional dependency under ``casecrawler[hf]``) -- they raise a
typed :class:`HFHubUnavailable` so callers can present a setup hint.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)


HF_API_BASE = "https://huggingface.co/api"

# Tags we expect on legitimate medical imaging generators on the Hub.
_MEDICAL_TAGS = {"medical", "medical-imaging", "radiology"}

# Lightweight per-modality query hints. These map a caller-friendly
# modality ("chest_xray", "ct", "pathology", ...) to a Hub search filter
# tuple (tag, fallback search string).
_MODALITY_QUERIES: dict[str, tuple[str | None, str]] = {
    "chest_xray": ("chest-xray", "chest x-ray"),
    "cxr": ("chest-xray", "chest x-ray"),
    "ct": ("ct", "computed tomography"),
    "mri": ("mri", "magnetic resonance"),
    "pathology": ("histopathology", "pathology"),
    "histopathology": ("histopathology", "pathology"),
    "ultrasound": ("ultrasound", "ultrasound"),
    "fundus": ("fundus", "retinal fundus"),
    "dermatology": ("dermatology", "skin lesion"),
}


class HFHubUnavailable(RuntimeError):
    """Raised when the Hub cannot be reached or the optional dep is missing."""


class HFGatedModelError(RuntimeError):
    """Raised when a model on the Hub requires explicit access approval.

    The error message includes a direct URL the user can open to request
    access. We catch this at every download / model_info call site so users
    see something actionable instead of an opaque 401.
    """

    def __init__(self, repo_id: str) -> None:
        url = f"https://huggingface.co/{repo_id}"
        super().__init__(
            f"Model {repo_id!r} is gated. Visit {url} to request access, "
            "wait for approval, then re-run this command. If you've already "
            "been approved, ensure HF_TOKEN is set and the token has "
            "permission to read this repo."
        )
        self.repo_id = repo_id
        self.url = url


@dataclass(frozen=True)
class ModelCardMetadata:
    repo_id: str
    license: str | None
    gated: bool
    use_policy: str | None
    pipeline_tag: str | None
    tags: tuple[str, ...]
    last_modified: str | None
    fetched_at: str

    def as_image_source(self) -> dict[str, Any]:
        """Render as the ``image_source`` payload attached to image metadata."""
        return {
            "repo_id": self.repo_id,
            "license": self.license or "unspecified",
            "gated": self.gated,
            "use_policy": self.use_policy or "see_model_card",
            "model_card_url": f"https://huggingface.co/{self.repo_id}",
            "pipeline_tag": self.pipeline_tag,
            "tags": list(self.tags),
            "last_modified": self.last_modified,
            "fetched_at": self.fetched_at,
        }


@dataclass(frozen=True)
class ModelSuggestion:
    repo_id: str
    downloads: int
    likes: int
    last_modified: str | None
    license: str | None
    gated: bool
    pipeline_tag: str | None
    tags: tuple[str, ...]


def fetch_model_card_metadata(
    repo_id: str,
    *,
    token: str | None = None,
) -> ModelCardMetadata:
    """Fetch the model card YAML for ``repo_id`` and surface license info.

    Uses ``huggingface_hub.model_info`` when available (handles auth and
    caching) and falls back to a plain HTTP GET otherwise. Returns a typed
    :class:`ModelCardMetadata` with everything we need to embed into image
    metadata. License is reported verbatim from the card; we never invent
    a license for a model that doesn't declare one.
    """

    try:
        from huggingface_hub import model_info
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        return _fetch_via_http(repo_id, token=token)

    try:
        info = model_info(repo_id=repo_id, token=token)
    except GatedRepoError as exc:
        raise HFGatedModelError(repo_id) from exc
    except RepositoryNotFoundError as exc:
        raise HFHubUnavailable(
            f"Hugging Face repo {repo_id!r} not found. Check the spelling and "
            "whether the model is private (requires HF_TOKEN)."
        ) from exc
    except Exception as exc:
        logger.exception(
            "Failed to fetch model_info for %r; falling back to HTTP.", repo_id
        )
        try:
            return _fetch_via_http(repo_id, token=token)
        except HFHubUnavailable:
            raise HFHubUnavailable(
                f"Could not fetch model card for {repo_id!r}: {exc}"
            ) from exc

    # `huggingface_hub.ModelInfo` exposes this as `card_data` (snake_case),
    # not `cardData`. The HTTP fallback returns `cardData` (the public API
    # response is camelCase). Prefer the snake_case attribute when present
    # and accept either a Mapping or a `ModelCardData` object that supports
    # attribute access for the fields we need.
    card_data = getattr(info, "card_data", None)
    if card_data is None:
        card_data = getattr(info, "cardData", None)
    license_str = _card_field(card_data, "license")
    use_policy = _card_field(card_data, "use_policy")
    return ModelCardMetadata(
        repo_id=repo_id,
        license=license_str,
        gated=bool(getattr(info, "gated", False)),
        use_policy=use_policy,
        pipeline_tag=getattr(info, "pipeline_tag", None),
        tags=tuple(getattr(info, "tags", []) or []),
        last_modified=_format_dt(getattr(info, "last_modified", None)),
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )


def _fetch_via_http(
    repo_id: str,
    *,
    token: str | None = None,
) -> ModelCardMetadata:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{HF_API_BASE}/models/{repo_id}", headers=headers)
    except httpx.HTTPError as exc:
        raise HFHubUnavailable(
            f"Could not reach Hugging Face Hub: {exc}"
        ) from exc
    if resp.status_code == 401 or resp.status_code == 403:
        raise HFGatedModelError(repo_id)
    if resp.status_code == 404:
        raise HFHubUnavailable(
            f"Hugging Face repo {repo_id!r} not found."
        )
    resp.raise_for_status()
    payload = resp.json()
    card_data = payload.get("cardData")
    return ModelCardMetadata(
        repo_id=repo_id,
        license=_card_field(card_data, "license"),
        gated=bool(payload.get("gated") or False),
        use_policy=_card_field(card_data, "use_policy"),
        pipeline_tag=payload.get("pipeline_tag"),
        tags=tuple(payload.get("tags") or []),
        last_modified=payload.get("lastModified"),
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )


def suggest_imaging_models(
    modality: str,
    *,
    limit: int = 10,
    pipeline_tag: str = "text-to-image",
    token: str | None = None,
) -> list[ModelSuggestion]:
    """Query the Hub for medical-imaging models for ``modality``.

    Returns a ranked list (downloads desc) of repos tagged ``medical`` plus
    a modality-specific tag, restricted to ``pipeline_tag`` (default
    ``text-to-image``). Print-only output for the
    ``casecrawler suggest-imaging-models`` CLI; we never auto-write the
    user's config.
    """

    if limit <= 0:
        raise ValueError("suggest_imaging_models requires limit > 0")
    tag, search = _MODALITY_QUERIES.get(modality.lower(), (None, modality))
    params: dict[str, Any] = {
        "search": search,
        "filter": ["medical"],
        "pipeline_tag": pipeline_tag,
        "sort": "downloads",
        "direction": "-1",
        "limit": min(limit, 50),
        "full": "true",
    }
    if tag:
        params["filter"] = [*params["filter"], tag]

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{HF_API_BASE}/models", params=params, headers=headers)
            resp.raise_for_status()
            rows = resp.json()
    except httpx.HTTPError as exc:
        raise HFHubUnavailable(
            f"Could not query Hugging Face Hub: {exc}"
        ) from exc

    suggestions: list[ModelSuggestion] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        tags = tuple(row.get("tags") or [])
        if not _MEDICAL_TAGS.intersection({t.lower() for t in tags}):
            continue
        card_data = row.get("cardData") or {}
        suggestions.append(
            ModelSuggestion(
                repo_id=row.get("id") or row.get("modelId") or "",
                downloads=int(row.get("downloads") or 0),
                likes=int(row.get("likes") or 0),
                last_modified=row.get("lastModified"),
                license=(card_data.get("license") if isinstance(card_data, dict) else None),
                gated=bool(row.get("gated") or False),
                pipeline_tag=row.get("pipeline_tag"),
                tags=tags,
            )
        )
    return suggestions


def _card_field(card_data: Any, name: str) -> Any:
    """Read ``name`` from a Hub model card whether it's a Mapping or object."""
    if card_data is None:
        return None
    if isinstance(card_data, Mapping):
        return card_data.get(name)
    return getattr(card_data, name, None)


def _format_dt(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)
