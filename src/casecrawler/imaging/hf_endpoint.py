"""Hugging Face Inference Endpoint imaging backend.

Users with no local GPU can deploy a diffusers model to a managed HF
Inference Endpoint and point this adapter at the resulting URL. The
adapter does a single POST per image, writes the returned bytes to the
output dir, and stamps the asset's ``image_source`` metadata with the
license + use-policy auto-fetched from the model card.

This is the first of the HF-first imaging backends to land. The
``hf_local`` (download-via-snapshot + diffusers) backend is intentionally
deferred to a follow-up because it pulls in heavy optional deps; this
endpoint adapter works against any HF-hosted model the user can hit
remotely with a token.
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

import httpx

from casecrawler.generation.imaging_templates import (
    build_imaging_report,
    infer_imaging_labels,
)
from casecrawler.imaging.file_metadata import image_file_metadata
from casecrawler.imaging.hf_hub import (
    HFGatedModelError,
    HFHubUnavailable,
    fetch_model_card_metadata,
)
from casecrawler.models.synthetic import ImagingAsset

logger = logging.getLogger(__name__)


HF_INFERENCE_TIMEOUT_SECONDS = 120.0


class HFEndpointImagingBackend:
    """Drives image generation by POSTing to a user-provided HF endpoint.

    Required configuration:

    - ``endpoint_url`` -- the HF Inference Endpoint URL (e.g.
      ``https://xxxx.us-east-1.aws.endpoints.huggingface.cloud``)
    - ``token`` -- HF token with read access to the endpoint
    - ``repo_id`` -- the underlying HF model id (used for license / use-
      policy lookup; not for the actual inference request)

    Returns a typed :class:`ImagingAsset` with ``image_source`` metadata
    auto-fetched from the model card. The endpoint is expected to return
    PNG / JPEG bytes; the adapter writes them to ``output_dir`` and stamps
    the metadata.
    """

    def __init__(
        self,
        *,
        endpoint_url: str,
        token: str,
        repo_id: str,
        timeout_seconds: float = HF_INFERENCE_TIMEOUT_SECONDS,
        client: httpx.Client | None = None,
    ) -> None:
        if not endpoint_url:
            raise ValueError("hf_endpoint backend requires endpoint_url.")
        if not token:
            raise ValueError(
                "hf_endpoint backend requires an HF token; set HF_TOKEN or "
                "pass token=... to the backend constructor."
            )
        if not repo_id:
            raise ValueError(
                "hf_endpoint backend requires the underlying model's repo_id "
                "so license / use_policy can be auto-fetched from the model card."
            )
        self._endpoint_url = endpoint_url
        self._token = token
        self._repo_id = repo_id
        self._timeout = timeout_seconds
        self._client = client
        self._card_metadata = None  # lazily fetched on first call

    def generate(
        self,
        *,
        output_dir: str,
        prompt: str,
        modality: str = "XR",
        body_region: str = "chest",
        negative_prompt: str | None = None,
    ) -> ImagingAsset:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        image_id = f"img-hfendpoint-{uuid4()}"

        body: dict[str, object] = {"inputs": prompt}
        if negative_prompt:
            body["parameters"] = {"negative_prompt": negative_prompt}

        headers = {"Authorization": f"Bearer {self._token}"}
        client = self._client
        owns_client = client is None
        try:
            client = client or httpx.Client(timeout=self._timeout)
            try:
                resp = client.post(
                    self._endpoint_url, json=body, headers=headers
                )
            except httpx.HTTPError as exc:
                raise RuntimeError(
                    f"hf_endpoint POST to {self._endpoint_url!r} failed: {exc}"
                ) from exc
        finally:
            if owns_client and client is not None:
                client.close()

        if resp.status_code in (401, 403):
            raise HFGatedModelError(self._repo_id)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"hf_endpoint returned HTTP {resp.status_code}: "
                f"{resp.text[:300]}"
            )
        if not resp.content:
            raise RuntimeError("hf_endpoint returned empty response body.")
        # Pick a file extension that matches the bytes the endpoint returned.
        # Diffusers-backed endpoints can return PNG or JPEG depending on the
        # serving config; persisting JPEG bytes under a `.png` path confuses
        # downstream consumers and breaks ImageMagick/PIL identify calls.
        extension = _infer_image_extension(resp)
        file_path = Path(output_dir) / f"{image_id}{extension}"
        file_path.write_bytes(resp.content)

        card = self._get_card_metadata()
        labels = infer_imaging_labels(prompt, modality)
        backend = f"hf_endpoint:{self._repo_id}"
        metadata = {
            "generation_backend": backend,
            "artifact_contract": "casecrawler.models.synthetic.ImagingAsset",
            "image_source": card.as_image_source(),
            "endpoint": self._endpoint_url,
            "is_placeholder": False,
        }
        if file_path.exists():
            metadata["file"] = image_file_metadata(file_path)
        return ImagingAsset(
            image_id=image_id,
            modality=modality,
            body_region=body_region,
            prompt=prompt,
            file_path=str(file_path),
            report_text=build_imaging_report(
                prompt=prompt,
                modality=modality,
                body_region=body_region,
                labels=labels,
            ),
            labels=labels,
            generation_backend=backend,
            metadata=metadata,
        )

    def _get_card_metadata(self):
        if self._card_metadata is not None:
            return self._card_metadata
        try:
            self._card_metadata = fetch_model_card_metadata(
                self._repo_id, token=self._token
            )
        except HFGatedModelError:
            raise
        except HFHubUnavailable as exc:
            logger.warning(
                "Could not fetch model card for %r (%s); embedding minimal "
                "image_source metadata. Set HF_TOKEN or check connectivity.",
                self._repo_id,
                exc,
            )
            from datetime import datetime, timezone

            from casecrawler.imaging.hf_hub import ModelCardMetadata

            self._card_metadata = ModelCardMetadata(
                repo_id=self._repo_id,
                license=None,
                gated=False,
                use_policy=None,
                pipeline_tag=None,
                tags=(),
                last_modified=None,
                fetched_at=datetime.now(timezone.utc).isoformat(),
            )
        return self._card_metadata


def _infer_image_extension(resp: httpx.Response) -> str:
    """Pick a file extension matching the response's image bytes.

    Looks at ``Content-Type`` first (the standards-compliant signal) and
    falls back to magic bytes when the header is missing or generic. We
    keep the decision small and conservative -- PNG, JPEG, and WebP cover
    every diffusers serving config we've seen on HF Inference Endpoints.
    """
    content_type = (resp.headers.get("content-type") or "").lower().split(";")[0].strip()
    by_type = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
    }
    if content_type in by_type:
        return by_type[content_type]
    # Magic-byte fallback for endpoints that don't set Content-Type.
    head = resp.content[:12]
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if head.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return ".webp"
    # Default -- preserve the previous behaviour for byte streams we can't
    # identify rather than refusing to write.
    return ".png"
