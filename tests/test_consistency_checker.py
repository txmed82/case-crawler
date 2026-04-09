from unittest.mock import AsyncMock
import pytest
from casecrawler.generation.consistency_checker import ConsistencyCheckerAgent, ConsistencyCheckerOutput, ConsistencyIssue
from casecrawler.llm.base import StructuredGenerationResult

def _mock_no_issues():
    return StructuredGenerationResult(data=ConsistencyCheckerOutput(issues=[]), input_tokens=500, output_tokens=50, model="test-model")

def _mock_with_issues():
    return StructuredGenerationResult(data=ConsistencyCheckerOutput(issues=[ConsistencyIssue(phase_number=3, field="vitals.hr", issue="HR drops from 120 to 68 without treatment", suggested_fix="Set HR to 108 in phase 3 or add treatment in phase 2")]), input_tokens=500, output_tokens=100, model="test-model")

@pytest.mark.asyncio
async def test_consistency_checker_no_issues():
    provider = AsyncMock()
    provider.generate_structured.return_value = _mock_no_issues()
    checker = ConsistencyCheckerAgent(provider=provider)
    result = await checker.check(phases_json='[{"phase_number": 1}]')
    assert result.data.issues == []

@pytest.mark.asyncio
async def test_consistency_checker_finds_issues():
    provider = AsyncMock()
    provider.generate_structured.return_value = _mock_with_issues()
    checker = ConsistencyCheckerAgent(provider=provider)
    result = await checker.check(phases_json='[{"phase_number": 1}, {"phase_number": 2}]')
    assert len(result.data.issues) == 1
    assert result.data.issues[0].phase_number == 3
    assert "HR" in result.data.issues[0].issue
