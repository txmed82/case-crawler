from __future__ import annotations

import hashlib
import math

from casecrawler.models.document import Chunk

EMBEDDING_DIM = 384


def deterministic_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    vector = [0.0] * dim
    tokens = text.lower().split()
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[idx] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
        except Exception:
            self.model = None

    def embed(self, chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
        texts = [c.text for c in chunks]
        if self.model is None:
            vectors = [deterministic_embedding(text) for text in texts]
            return list(zip(chunks, vectors))

        vectors = self.model.encode(texts, show_progress_bar=False)
        return [(chunk, vec.tolist()) for chunk, vec in zip(chunks, vectors)]
