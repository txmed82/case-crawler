import pytest

from casecrawler.generation.synthetic_pipeline import SyntheticPipeline
from casecrawler.models.dataset import GenerationRequest
from casecrawler.models.synthetic import ImagingAsset, Modality
from casecrawler.validation.synthetic_validator import SyntheticValidator


@pytest.mark.asyncio
async def test_synthetic_pipeline_generates_valid_records():
    pipeline = SyntheticPipeline(validator=SyntheticValidator())

    result = await pipeline.generate(GenerationRequest(topic="sepsis", count=2))

    assert result["generated"] == 2
    assert result["approved"] == 2
    assert len(result["records"]) == 2
    assert result["records"][0].documents
    assert result["records"][0].labs


class FakeImagingGenerator:
    def __init__(self):
        self.diffusers_prompts = []

    def generate_diffusers(self, output_dir: str, prompt: str):
        self.diffusers_prompts.append((output_dir, prompt))
        return ImagingAsset(
            image_id="img-test",
            modality="XR",
            body_region="chest",
            prompt=prompt,
            file_path="fake.png",
            report_text=f"Synthetic XR image for {prompt}",
            generation_backend="diffusers:test",
        )

    def generate_placeholder(self, output_dir: str, prompt: str):
        raise AssertionError("Expected diffusers backend.")


@pytest.mark.asyncio
async def test_synthetic_pipeline_uses_configured_diffusers_backend(tmp_path):
    imaging_generator = FakeImagingGenerator()
    pipeline = SyntheticPipeline(
        imaging_generator=imaging_generator,
        validator=SyntheticValidator(),
        image_output_dir=str(tmp_path),
        image_backend="diffusers",
    )

    result = await pipeline.generate(
        GenerationRequest(topic="pneumonia", count=1, modalities=[Modality.IMAGING])
    )

    assert result["records"][0].imaging[0].generation_backend == "diffusers:test"
    assert imaging_generator.diffusers_prompts[0][0] == str(tmp_path)
