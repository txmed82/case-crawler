import pytest

from casecrawler.generation.synthetic_pipeline import SyntheticPipeline
from casecrawler.models.dataset import GenerationRequest
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

