from __future__ import annotations

from datetime import datetime

from casecrawler.models.synthetic import GroundingBundle, GroundingCitation
from casecrawler.pipeline.store import Store

CREDIBILITY_ORDER = {
    "guideline": 0,
    "fda_label": 1,
    "peer_reviewed": 2,
    "curated": 3,
    "preprint": 4,
}

# How many characters of each chunk we keep on the citation as a snippet.
# This is for traceability/debugging; the full chunk text stays in Chroma.
_SNIPPET_CHARS = 280


class Retriever:
    def __init__(self, store: Store) -> None:
        self._store = store

    def retrieve(self, topic: str, limit: int = 25) -> list[dict]:
        """Query ChromaDB and return chunks ranked by relevance then credibility."""
        results = self._store.search(topic, n_results=limit)

        enriched = []
        for r in results:
            credibility = r["metadata"].get("credibility", "preprint")
            enriched.append({
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "score": r["score"],
                "credibility": credibility,
                "credibility_rank": CREDIBILITY_ORDER.get(credibility, 99),
                "source_document_id": r["metadata"].get("source_document_id", ""),
                "source": r["metadata"].get("source", ""),
                "specialty": r["metadata"].get("specialty", ""),
                "doi": r["metadata"].get("doi", ""),
                "url": r["metadata"].get("url", ""),
            })

        enriched.sort(key=lambda x: (x["credibility_rank"], -x["score"]))
        return enriched

    def format_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into a context string for LLM prompts."""
        sections = []
        for i, chunk in enumerate(chunks, 1):
            sections.append(
                f"[Source {i}] ({chunk['credibility']}, {chunk['source']})\n{chunk['text']}"
            )
        return "\n\n---\n\n".join(sections)

    def fetch_grounding(
        self,
        topic: str,
        modalities: list[str] | None = None,  # noqa: ARG002 - reserved for source filtering
        k: int = 8,
    ) -> GroundingBundle:
        """Retrieve a citation bundle for a topic.

        The returned :class:`GroundingBundle` is the carrier the synthetic
        pipeline attaches to ``record.metadata["grounding"]`` so every
        generated record can be traced back to the chunks that grounded it.

        ``modalities`` is accepted for forward-compatibility -- the current
        Chroma collection does not tag chunks with a modality, so we ignore
        the filter and return the full set. Sources are still ranked by
        credibility (guideline > FDA label > peer reviewed > preprint).
        """

        chunks = self.retrieve(topic=topic, limit=k)
        citations = [
            GroundingCitation(
                chunk_id=chunk["chunk_id"],
                source=chunk["source"] or "unknown",
                source_document_id=chunk["source_document_id"],
                score=float(chunk["score"]),
                credibility=chunk["credibility"],
                snippet=chunk["text"][:_SNIPPET_CHARS],
                doi=chunk["doi"] or None,
                url=chunk["url"] or None,
                specialty=chunk["specialty"] or None,
            )
            for chunk in chunks
        ]
        return GroundingBundle(
            topic=topic,
            retrieved_at=datetime.now().isoformat(),
            citations=citations,
        )
