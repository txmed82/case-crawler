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


def test_diffusers_backend_generates_unique_files(tmp_path):
    pipeline = FakeDiffusersPipeline()
    generator = ImagingGenerator(diffusers_pipeline=pipeline, diffusers_model_id="test/xray")

    first = generator.generate_diffusers(str(tmp_path), "portable chest x-ray")
    second = generator.generate_diffusers(str(tmp_path), "portable chest x-ray")

    assert first.image_id != second.image_id
    assert first.file_path != second.file_path


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


def test_external_imaging_backend_rejects_empty_command():
    with pytest.raises(ValueError, match="external_command must not be empty"):
        ImagingGenerator(external_command=[])
