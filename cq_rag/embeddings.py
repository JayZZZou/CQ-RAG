from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

class EmbeddingClient(Protocol):
    def embed(self, model: str, text: str, dimensions: int) -> list[float]: ...


class TextEmbeddings:
    def __init__(self, client: EmbeddingClient) -> None:
        self._client = client

    def embed_texts(
        self,
        texts: Sequence[str],
        model: str,
        dimensions: int,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, dimensions), dtype=np.float32)
        vectors: list[list[float]] = []
        for text in texts:
            vector = self._client.embed(model, text, dimensions)
            if len(vector) != dimensions:
                raise RuntimeError("Embedding response dimensions do not match the requested dimensions")
            vectors.append(vector)
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.shape != (len(texts), dimensions):
            raise RuntimeError(f"Unexpected embedding result shape: {matrix.shape}")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise RuntimeError("Embedding response contains a zero vector")
        return matrix / norms