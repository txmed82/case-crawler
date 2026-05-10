"""Regression tests for Phase 4b (HF-first imaging adapters)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from casecrawler.export.fine_tuning import (
    StrictExportError,
    export_multimodal_record,
    has_placeholder_imaging,
)
from casecrawler.generation.imaging_generator import ImagingGenerator
from casecrawler.imaging.hf_endpoint import HFEndpointImagingBackend
from casecrawler.imaging.hf_hub import (
    HFGatedModelError,
    HFHubUnavailable,
    fetch_model_card_metadata,
    suggest_imaging_models,
)
from casecrawler.models.synthetic import (
    ComplexityProfile,
    ImagingAsset,
    Modality,
    Provenance,
)


# ---------- Placeholder marking on imaging assets --------------------------


def test_generate_placeholder_marks_metadata(tmp_path):
    asset = ImagingGenerator().generate_placeholder(
        output_dir=str(tmp_path),
        prompt="chest x-ray with consolidation",
        modality="XR",
        body_region="chest",
    )
    assert asset.metadata["is_placeholder"] is True
    assert "placeholder_reason" in asset.metadata
    assert asset.generation_backend == "placeholder"


def test_has_placeholder_imaging_detects_marker():
    placeholder_asset = ImagingAsset(
        image_id="img-1",
        modality="XR",
        body_region="chest",
        prompt="x",
        report_text="r",
        labels=[],
        generation_backend="placeholder",
        metadata={"is_placeholder": True},
    )
    real_asset = ImagingAsset(
        image_id="img-2",
        modality="XR",
        body_region="chest",
        prompt="x",
        report_text="r",
        labels=[],
        generation_backend="hf_endpoint:foo/bar",
        metadata={"is_placeholder": False},
    )

    record_with_placeholder = _make_record_with_imaging([placeholder_asset])
    record_clean = _make_record_with_imaging([real_asset])
    record_mixed = _make_record_with_imaging([placeholder_asset, real_asset])

    assert has_placeholder_imaging(record_with_placeholder) is True
    assert has_placeholder_imaging(record_clean) is False
    assert has_placeholder_imaging(record_mixed) is True


def test_strict_multimodal_export_rejects_placeholder():
    placeholder_asset = ImagingAsset(
        image_id="img-1",
        modality="XR",
        body_region="chest",
        prompt="x",
        report_text="r",
        labels=[],
        generation_backend="placeholder",
        metadata={"is_placeholder": True},
    )
    record = _make_record_with_imaging([placeholder_asset])

    # Without strict, the export proceeds.
    payload = export_multimodal_record(record, strict=False)
    assert payload["images"]

    # With strict, the export raises StrictExportError.
    with pytest.raises(StrictExportError, match="placeholder imaging"):
        export_multimodal_record(record, strict=True)


def test_strict_multimodal_export_accepts_real_image():
    real_asset = ImagingAsset(
        image_id="img-2",
        modality="XR",
        body_region="chest",
        prompt="x",
        report_text="r",
        labels=[],
        generation_backend="hf_endpoint:foo/bar",
        metadata={"is_placeholder": False},
    )
    record = _make_record_with_imaging([real_asset])
    payload = export_multimodal_record(record, strict=True)
    assert len(payload["images"]) == 1


# ---------- HF Hub helpers -------------------------------------------------


def test_fetch_model_card_metadata_reports_license_and_gated():
    # huggingface_hub.ModelInfo exposes the field as `card_data`
    # (snake_case). The fallback HTTP path sees the wire-format `cardData`.
    fake_info = MagicMock()
    fake_info.card_data = {"license": "apache-2.0", "use_policy": "research-only"}
    # Force the camelCase attribute to be absent so the fallback isn't used.
    del fake_info.cardData
    fake_info.gated = False
    fake_info.pipeline_tag = "text-to-image"
    fake_info.tags = ["medical", "chest-xray"]
    fake_info.last_modified = None

    with patch("huggingface_hub.model_info", return_value=fake_info):
        meta = fetch_model_card_metadata("foo/bar")

    assert meta.repo_id == "foo/bar"
    assert meta.license == "apache-2.0"
    assert meta.gated is False
    assert meta.use_policy == "research-only"
    payload = meta.as_image_source()
    assert payload["repo_id"] == "foo/bar"
    assert payload["license"] == "apache-2.0"
    assert payload["model_card_url"] == "https://huggingface.co/foo/bar"


def test_fetch_model_card_metadata_handles_modelcarddata_object():
    """ModelInfo.card_data can be a ModelCardData instance with attribute
    access rather than a plain dict. The helper must handle both."""
    class _CardData:
        license = "mit"
        use_policy = "non-commercial"

    fake_info = MagicMock()
    fake_info.card_data = _CardData()
    del fake_info.cardData
    fake_info.gated = True
    fake_info.pipeline_tag = "text-to-image"
    fake_info.tags = ["medical"]
    fake_info.last_modified = None

    with patch("huggingface_hub.model_info", return_value=fake_info):
        meta = fetch_model_card_metadata("foo/object")

    assert meta.license == "mit"
    assert meta.use_policy == "non-commercial"
    assert meta.gated is True


def test_fetch_model_card_metadata_raises_on_gated_repo():
    from huggingface_hub.utils import GatedRepoError

    fake_response = MagicMock()
    fake_response.status_code = 403
    gated_exc = GatedRepoError("gated", response=fake_response)

    with patch("huggingface_hub.model_info", side_effect=gated_exc):
        with pytest.raises(HFGatedModelError) as exc:
            fetch_model_card_metadata("gated/repo")
    assert "gated/repo" in str(exc.value)
    assert "https://huggingface.co/gated/repo" in str(exc.value)


def test_suggest_imaging_models_filters_to_medical_tags(monkeypatch):
    rows = [
        {
            "id": "a/medical-good",
            "downloads": 1000,
            "likes": 50,
            "lastModified": "2026-04-01",
            "tags": ["medical", "chest-xray"],
            "pipeline_tag": "text-to-image",
            "cardData": {"license": "apache-2.0"},
            "gated": False,
        },
        {
            "id": "b/non-medical",
            "downloads": 5000,
            "likes": 100,
            "tags": ["text-to-image"],  # no medical tag → filtered out
            "pipeline_tag": "text-to-image",
        },
        {
            "id": "c/medical-gated",
            "downloads": 500,
            "tags": ["medical-imaging"],
            "pipeline_tag": "text-to-image",
            "gated": True,
        },
    ]

    class _FakeResp:
        status_code = 200

        def raise_for_status(self):  # noqa: D401
            return None

        def json(self):
            return rows

    class _FakeClient:
        def __init__(self, *a, **kw):
            self.timeout = 15.0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, *a, **kw):
            return _FakeResp()

    monkeypatch.setattr("casecrawler.imaging.hf_hub.httpx.Client", _FakeClient)

    suggestions = suggest_imaging_models("chest_xray", limit=5)
    repo_ids = {s.repo_id for s in suggestions}
    assert "a/medical-good" in repo_ids
    assert "b/non-medical" not in repo_ids
    assert "c/medical-gated" in repo_ids
    gated = next(s for s in suggestions if s.repo_id == "c/medical-gated")
    assert gated.gated is True


def test_suggest_imaging_models_raises_when_hub_unreachable(monkeypatch):
    class _FailingClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, *a, **kw):
            raise httpx.ConnectError("network down")

    monkeypatch.setattr("casecrawler.imaging.hf_hub.httpx.Client", _FailingClient)
    with pytest.raises(HFHubUnavailable, match="Hugging Face Hub"):
        suggest_imaging_models("chest_xray")


# ---------- HF Endpoint imaging backend ------------------------------------


def test_hf_endpoint_backend_writes_image_and_stamps_metadata(tmp_path):
    fake_card = MagicMock()
    fake_card.cardData = {"license": "mit"}
    fake_card.gated = False
    fake_card.pipeline_tag = "text-to-image"
    fake_card.tags = ["medical"]
    fake_card.last_modified = None

    fake_resp = MagicMock(spec=httpx.Response)
    fake_resp.status_code = 200
    fake_resp.content = b"\x89PNG\r\n\x1a\nfake-png-bytes"
    fake_resp.text = ""
    fake_resp.headers = {"content-type": "image/png"}

    fake_client = MagicMock()
    fake_client.post.return_value = fake_resp

    fake_card.card_data = fake_card.cardData  # snake_case for ModelInfo
    with patch("huggingface_hub.model_info", return_value=fake_card):
        backend = HFEndpointImagingBackend(
            endpoint_url="https://test.endpoints.huggingface.cloud",
            token="hf_test",
            repo_id="foo/cxr-model",
            client=fake_client,
        )
        asset = backend.generate(
            output_dir=str(tmp_path),
            prompt="chest x-ray with pneumonia",
            modality="XR",
            body_region="chest",
        )

    assert asset.generation_backend == "hf_endpoint:foo/cxr-model"
    assert asset.metadata["is_placeholder"] is False
    assert asset.metadata["image_source"]["repo_id"] == "foo/cxr-model"
    assert asset.metadata["image_source"]["license"] == "mit"
    assert asset.metadata["endpoint"].startswith("https://test.endpoints")
    fake_client.post.assert_called_once()
    assert fake_client.post.call_args.kwargs["headers"]["Authorization"] == "Bearer hf_test"
    assert (tmp_path / f"{asset.image_id}.png").exists()


def test_hf_endpoint_backend_writes_jpeg_when_endpoint_returns_jpeg(tmp_path):
    """If the endpoint returns JPEG bytes, the file must be written as
    .jpg, not .png. Otherwise downstream consumers (PIL, ImageMagick)
    fail to identify the file."""
    fake_card = MagicMock()
    fake_card.card_data = {"license": "mit"}
    fake_card.gated = False
    fake_card.pipeline_tag = "text-to-image"
    fake_card.tags = ["medical"]
    fake_card.last_modified = None
    del fake_card.cardData

    fake_resp = MagicMock(spec=httpx.Response)
    fake_resp.status_code = 200
    fake_resp.content = b"\xff\xd8\xfffake-jpeg-bytes"
    fake_resp.text = ""
    fake_resp.headers = {"content-type": "image/jpeg"}

    fake_client = MagicMock()
    fake_client.post.return_value = fake_resp

    with patch("huggingface_hub.model_info", return_value=fake_card):
        backend = HFEndpointImagingBackend(
            endpoint_url="https://test.endpoints.huggingface.cloud",
            token="hf_test",
            repo_id="foo/jpeg-model",
            client=fake_client,
        )
        asset = backend.generate(
            output_dir=str(tmp_path),
            prompt="x",
            modality="XR",
            body_region="chest",
        )
    assert (tmp_path / f"{asset.image_id}.jpg").exists()
    assert not (tmp_path / f"{asset.image_id}.png").exists()


def test_hf_endpoint_backend_raises_gated_on_401(tmp_path):
    fake_resp = MagicMock(spec=httpx.Response)
    fake_resp.status_code = 401
    fake_resp.text = "unauthorized"

    fake_client = MagicMock()
    fake_client.post.return_value = fake_resp

    backend = HFEndpointImagingBackend(
        endpoint_url="https://test.endpoints.huggingface.cloud",
        token="hf_test",
        repo_id="gated/cxr-model",
        client=fake_client,
    )
    with pytest.raises(HFGatedModelError) as exc:
        backend.generate(
            output_dir=str(tmp_path),
            prompt="x",
            modality="XR",
            body_region="chest",
        )
    assert "gated/cxr-model" in str(exc.value)


def test_hf_endpoint_backend_validates_required_args():
    with pytest.raises(ValueError, match="endpoint_url"):
        HFEndpointImagingBackend(endpoint_url="", token="t", repo_id="a/b")
    with pytest.raises(ValueError, match="HF token"):
        HFEndpointImagingBackend(
            endpoint_url="https://x", token="", repo_id="a/b"
        )
    with pytest.raises(ValueError, match="repo_id"):
        HFEndpointImagingBackend(
            endpoint_url="https://x", token="t", repo_id=""
        )


# ---------- helpers --------------------------------------------------------


def _make_record_with_imaging(assets):
    from casecrawler.models.synthetic import SyntheticPatient, SyntheticRecord

    return SyntheticRecord(
        record_id="rec-1",
        dataset_id="ds-1",
        topic="pneumonia",
        complexity=ComplexityProfile.MODERATE,
        modalities=[Modality.IMAGING, Modality.CLINICAL_TEXT],
        patient=SyntheticPatient(patient_id="pat-1", age=40, sex="female"),
        encounters=[],
        labs=[],
        vitals=[],
        imaging=assets,
        provenance=Provenance(
            generator="unit-test", created_at="2026-05-06T09:00:00"
        ),
    )


