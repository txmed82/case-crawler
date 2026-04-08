from __future__ import annotations

import tempfile

import pytest
from casecrawler.models.document import Document, DocumentMetadata, CredibilityLevel
from casecrawler.pipeline.orchestrator import PipelineOrchestrator


def _doc(source_id: str, content: str = "Some text. " * 20, content_type: str = "abstract") -> Document:
    return Document(
        source="test",
        source_id=source_id,
        title=f"Doc {source_id}",
        content=content,
        content_type=content_type,
        metadata=DocumentMetadata(
            credibility=CredibilityLevel.PEER_REVIEWED,
            specialty=[],
        ),
    )


def test_orchestrator_processes_documents():
    with tempfile.TemporaryDirectory() as tmpdir:
        orch = PipelineOrchestrator(chroma_dir=tmpdir)
        docs = [_doc("d1"), _doc("d2"), _doc("d3")]
        result = orch.process(docs)
        assert result["documents"] == 3
        assert result["chunks"] >= 3
        assert orch.store.count >= 3


def test_orchestrator_idempotent():
    with tempfile.TemporaryDirectory() as tmpdir:
        orch = PipelineOrchestrator(chroma_dir=tmpdir)
        docs = [_doc("d1"), _doc("d2"), _doc("d3")]
        result1 = orch.process(docs)
        result2 = orch.process(docs)
        # Same docs processed twice — chunk count should be stable (upsert)
        assert result1["chunks"] == result2["chunks"]
        assert orch.store.count == result1["chunks"]
