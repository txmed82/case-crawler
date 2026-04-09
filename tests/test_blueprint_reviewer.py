from unittest.mock import AsyncMock
import pytest
from casecrawler.generation.blueprint_reviewer import BlueprintReviewerAgent, BlueprintReviewResult
from casecrawler.llm.base import StructuredGenerationResult

def _mock_approved():
    return StructuredGenerationResult(data=BlueprintReviewResult(approved=True, notes=[]), input_tokens=300, output_tokens=50, model="test-model")

def _mock_rejected():
    return StructuredGenerationResult(data=BlueprintReviewResult(approved=False, notes=["Phase count too low for attending"]), input_tokens=300, output_tokens=80, model="test-model")

@pytest.mark.asyncio
async def test_blueprint_reviewer_approves():
    provider = AsyncMock()
    provider.generate_structured.return_value = _mock_approved()
    reviewer = BlueprintReviewerAgent(provider=provider)
    result = await reviewer.review(blueprint_json='{"diagnosis":"SAH"}', context="SAH content")
    assert result.data.approved is True
    assert result.data.notes == []

@pytest.mark.asyncio
async def test_blueprint_reviewer_rejects():
    provider = AsyncMock()
    provider.generate_structured.return_value = _mock_rejected()
    reviewer = BlueprintReviewerAgent(provider=provider)
    result = await reviewer.review(blueprint_json='{"diagnosis":"SAH"}', context="SAH content")
    assert result.data.approved is False
    assert len(result.data.notes) == 1
