import json

import pytest

from casecrawler.generation.imaging_generator import ImagingGenerator

def test_imaging_placeholder_does_not_require_diffusers(tmp_path):
    asset = ImagingGenerator().generate_placeholder(
        str(tmp_path),
        "portable chest x-ray with pulmonary edema",
    )

    assert asset.generation_backend == "placeholder"
    assert asset.modality == "XR"
    assert asset.file_path is not None
    assert (tmp_path / f"{asset.image_id}.png").read_bytes().startswith(b"\x89PNG")
    assert asset.labels[0].display == "Pulmonary edema"
    assert "Pulmonary edema" in asset.report_text
    assert asset.metadata["generation_backend"] == "placeholder"
    assert asset.metadata["artifact_contract"] == (
        "casecrawler.models.synthetic.ImagingAsset"
    )
    assert asset.metadata["file"]["mime_type"] == "image/png"
    assert asset.metadata["file"]["width"] == 128
    assert asset.metadata["file"]["height"] == 128
    assert len(asset.metadata["file"]["sha256"]) == 64


class FakeImage:
    def save(self, path):
        with open(path, "wb") as f:
            f.write(b"fake-png")


class FakeDiffusersResult:
    images = [FakeImage()]


class FakeDiffusersPipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return FakeDiffusersResult()


@pytest.mark.optional_backend
def test_diffusers_backend_uses_injected_pipeline(tmp_path):
    pipeline = FakeDiffusersPipeline()
    generator = ImagingGenerator(diffusers_pipeline=pipeline, diffusers_model_id="test/xray")

    asset = generator.generate_diffusers(
        str(tmp_path),
        "portable chest x-ray with pulmonary edema",
        negative_prompt="real patient identifiers",
    )

    assert asset.generation_backend == "diffusers:test/xray"
    assert asset.file_path is not None
    assert (tmp_path / f"{asset.image_id}.png").read_bytes() == b"fake-png"
    assert pipeline.calls[0]["negative_prompt"] == "real patient identifiers"
    assert asset.labels[0].display == "Pulmonary edema"
    assert "Pulmonary edema" in asset.report_text
    assert asset.metadata["generation_backend"] == "diffusers:test/xray"
    assert asset.metadata["file"]["byte_size"] == len(b"fake-png")
    assert len(asset.metadata["file"]["sha256"]) == 64


@pytest.mark.optional_backend
def test_diffusers_backend_generates_unique_files(tmp_path):
    pipeline = FakeDiffusersPipeline()
    generator = ImagingGenerator(diffusers_pipeline=pipeline, diffusers_model_id="test/xray")

    first = generator.generate_diffusers(str(tmp_path), "portable chest x-ray")
    second = generator.generate_diffusers(str(tmp_path), "portable chest x-ray")

    assert first.image_id != second.image_id
    assert first.file_path != second.file_path


@pytest.mark.optional_backend
def test_diffusers_backend_caches_loaded_pipeline(tmp_path):
    class LoadingGenerator(ImagingGenerator):
        def __init__(self):
            super().__init__(diffusers_model_id="test/xray")
            self.load_count = 0

        def _load_diffusers_pipeline(self):
            self.load_count += 1
            return FakeDiffusersPipeline()

    generator = LoadingGenerator()

    generator.generate_diffusers(str(tmp_path), "portable chest x-ray")
    generator.generate_diffusers(str(tmp_path), "portable chest x-ray")

    assert generator.load_count == 1


@pytest.mark.optional_backend
def test_diffusers_backend_requires_imaging_extra_when_not_injected(monkeypatch, tmp_path):
    def fake_require_package(import_name: str, extra: str):
        raise RuntimeError(f"Install casecrawler[{extra}] to use this backend.")

    monkeypatch.setattr("casecrawler.generation.imaging_generator.require_package", fake_require_package)
    generator = ImagingGenerator()

    try:
        generator.generate_diffusers(str(tmp_path), "portable chest x-ray")
    except RuntimeError as exc:
        assert "casecrawler[imaging]" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for missing imaging extra.")


@pytest.mark.optional_backend
def test_diffusers_backend_uses_imaging_model_profile(tmp_path):
    pipeline = FakeDiffusersPipeline()
    generator = ImagingGenerator(
        diffusers_pipeline=pipeline,
        imaging_model_profile="cxr_pneumonia_dreambooth",
    )

    asset = generator.generate_diffusers(str(tmp_path), "right lower lobe infiltrate")

    assert asset.generation_backend.startswith(
        "diffusers:cxr_pneumonia_dreambooth:chimbiwide/cxr-pneumonia-dreambooth"
    )
    assert asset.modality == "XR"
    assert asset.body_region == "chest"
    assert "pneumonia infection" in pipeline.calls[0]["prompt"]
    assert "right lower lobe infiltrate" in pipeline.calls[0]["prompt"]
    assert "patient identifiers" in pipeline.calls[0]["negative_prompt"]
    assert asset.metadata["model_profile"] == {
        "name": "cxr_pneumonia_dreambooth",
        "model_id": "chimbiwide/cxr-pneumonia-dreambooth",
        "adapter_type": "diffusers",
        "license": "openrail++",
        "gated": False,
        "use_policy": "openrail_review_outputs_before_release",
        "validation_requirements": [
            "image_file_signature",
            "image_dimensions_min_32x32",
            "radiology_label_evidence",
            "privacy_screen",
            "image_text_alignment_if_configured",
        ],
    }


@pytest.mark.optional_backend
def test_diffusers_backend_rejects_incompatible_imaging_model_profile(tmp_path):
    generator = ImagingGenerator(
        diffusers_pipeline=FakeDiffusersPipeline(),
        imaging_model_profile="cxr_pneumonia_dreambooth",
    )

    with pytest.raises(ValueError, match="incompatible with requested CT abdomen"):
        generator.generate_diffusers(
            str(tmp_path),
            "acute appendicitis fat stranding",
            modality="CT",
            body_region="abdomen",
        )


@pytest.mark.optional_backend
def test_external_imaging_backend_accepts_asset_envelope(tmp_path):
    calls = []

    def fake_runner(command, payload):
        calls.append((command, payload))
        return (
            '{"asset":{"image_id":"img-external","modality":"XR",'
            '"body_region":"chest","prompt":"generated prompt",'
            '"file_path":"external.png","report_text":"Right lower lobe pneumonia.",'
            '"labels":[{"system":"external","code":"pneumonia",'
            '"display":"Pneumonia"}]}}'
        )

    asset = ImagingGenerator(
        external_command=["hf-image-sample"],
        external_runner=fake_runner,
    ).generate_external(
        str(tmp_path),
        "portable chest x-ray with pneumonia",
    )

    payload = json.loads(calls[0][1])
    assert calls[0][0] == ["hf-image-sample"]
    assert payload["output_dir"] == str(tmp_path)
    assert payload["modality"] == "XR"
    assert asset.image_id == "img-external"
    assert asset.generation_backend == "external:hf-image-sample"
    assert asset.labels[0].display == "Pneumonia"
    assert asset.metadata["generation_backend"] == "external:hf-image-sample"
    assert asset.metadata["external_command"] == ["hf-image-sample"]
    assert asset.metadata["external_contract"]["stdout"] == (
        "ImagingAsset JSON or {'asset': ImagingAsset JSON}"
    )


@pytest.mark.optional_backend
def test_external_imaging_backend_rejects_empty_command():
    with pytest.raises(ValueError, match="external_command must not be empty"):
        ImagingGenerator(external_command=[])
