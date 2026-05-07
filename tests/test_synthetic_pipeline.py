import pytest

from casecrawler.generation.synthetic_pipeline import SyntheticPipeline
from casecrawler.models.dataset import GenerationRequest
from casecrawler.models.synthetic import Code, ImagingAsset, Modality
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
        self.diffusers_calls = []

    def generate_diffusers(
        self,
        output_dir: str,
        prompt: str,
        modality: str = "XR",
        body_region: str = "chest",
        negative_prompt: str | None = None,
    ):
        self.diffusers_calls.append((output_dir, prompt, modality, body_region))
        return ImagingAsset(
            image_id="img-test",
            modality=modality,
            body_region=body_region,
            prompt=prompt,
            file_path="fake.png",
            report_text=f"Synthetic {modality} image for {prompt}",
            labels=[
                Code(
                    system="synthetic",
                    code="pneumonia",
                    display="Pneumonia",
                )
            ],
            generation_backend="diffusers:test",
        )

    def generate_placeholder(
        self,
        output_dir: str,
        prompt: str,
        modality: str = "XR",
        body_region: str = "chest",
    ):
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
    assert imaging_generator.diffusers_calls[0][0] == str(tmp_path)
    assert "right lower lobe opacity" in imaging_generator.diffusers_calls[0][1]
    assert imaging_generator.diffusers_calls[0][2:] == ("XR", "chest")


@pytest.mark.asyncio
async def test_synthetic_pipeline_placeholder_imaging_uses_topic_aware_labels(tmp_path):
    pipeline = SyntheticPipeline(
        validator=SyntheticValidator(),
        image_output_dir=str(tmp_path),
        image_backend="placeholder",
    )

    result = await pipeline.generate(
        GenerationRequest(topic="heart failure", count=1, modalities=[Modality.IMAGING])
    )

    image = result["records"][0].imaging[0]

    assert "pulmonary edema" in image.prompt
    assert "Pulmonary edema" in image.report_text
    assert any(label.display == "Pulmonary edema" for label in image.labels)


@pytest.mark.asyncio
async def test_synthetic_pipeline_uses_topic_specific_imaging_modalities(tmp_path):
    pipeline = SyntheticPipeline(
        validator=SyntheticValidator(),
        image_output_dir=str(tmp_path),
        image_backend="placeholder",
    )

    stroke = await pipeline.generate(
        GenerationRequest(topic="ischemic stroke", count=1, modalities=[Modality.IMAGING])
    )
    pe = await pipeline.generate(
        GenerationRequest(topic="pulmonary embolism", count=1, modalities=[Modality.IMAGING])
    )
    aki = await pipeline.generate(
        GenerationRequest(topic="acute kidney injury", count=1, modalities=[Modality.IMAGING])
    )

    assert stroke["records"][0].imaging[0].modality == "CT"
    assert stroke["records"][0].imaging[0].body_region == "head"
    assert pe["records"][0].imaging[0].modality == "CTA"
    assert pe["records"][0].imaging[0].body_region == "chest"
    assert aki["records"][0].imaging[0].modality == "US"
    assert aki["records"][0].imaging[0].body_region == "abdomen"


@pytest.mark.asyncio
async def test_synthetic_pipeline_rejects_unknown_image_backend(tmp_path):
    pipeline = SyntheticPipeline(
        validator=SyntheticValidator(),
        image_output_dir=str(tmp_path),
        image_backend="unknown",
    )

    with pytest.raises(ValueError, match="Unknown synthetic imaging backend"):
        await pipeline.generate(
            GenerationRequest(topic="pneumonia", count=1, modalities=[Modality.IMAGING])
        )
