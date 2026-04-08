import tempfile
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from casecrawler.cli import cli
from casecrawler.models.document import (
    CredibilityLevel,
    Document,
    DocumentMetadata,
)


def _fake_doc(source: str = "pubmed", source_id: str = "1") -> Document:
    return Document(
        source=source,
        source_id=source_id,
        title="Test Article",
        content="Test content about hemorrhage.",
        content_type="abstract",
        metadata=DocumentMetadata(credibility=CredibilityLevel.PEER_REVIEWED),
    )


def test_cli_sources():
    runner = CliRunner()
    result = runner.invoke(cli, ["sources"])
    assert result.exit_code == 0
    # Should show at least the free sources
    assert "pubmed" in result.output.lower()


def test_cli_config():
    runner = CliRunner()
    result = runner.invoke(cli, ["config"])
    assert result.exit_code == 0
    assert "chunk_size" in result.output.lower() or "embedding" in result.output.lower()


def test_cli_ingest(tmp_path):
    runner = CliRunner()
    fake_search = AsyncMock(return_value=[_fake_doc()])

    with patch("casecrawler.cli.SourceRegistry") as MockRegistry, \
         patch("casecrawler.cli.PipelineOrchestrator") as MockPipeline:
        mock_reg = MockRegistry.return_value
        mock_reg.discover.return_value = None
        mock_source = AsyncMock()
        mock_source.name = "pubmed"
        mock_source.search = fake_search
        mock_reg.get_sources.return_value = [mock_source]

        mock_pipeline = MockPipeline.return_value
        mock_pipeline.process.return_value = {"documents": 1, "chunks": 3}

        result = runner.invoke(cli, ["ingest", "subarachnoid hemorrhage"])
        assert result.exit_code == 0
        assert "1" in result.output  # document count
