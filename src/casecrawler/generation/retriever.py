from __future__ import annotations

from casecrawler.pipeline.store import Store

CREDIBILITY_ORDER = {
    "guideline": 0,
    "fda_label": 1,
    "peer_reviewed": 2,
    "curated": 3,
    "preprint": 4,
}


class Retriever:
    def __init__(self, store: Store) -> None:
        self._store = store

    def retrieve(self, topic: str, limit: int = 25) -> list[dict]:
        """Query ChromaDB and return chunks ranked by relevance then credibility."""
        results = self._store.search(topic, n_results=limit)

        # Enrich and sort: credibility first, then relevance score
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
