"""Embedding engine using sentence-transformers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    """Wraps a sentence-transformer model for encoding and similarity."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        """Encode a list of texts into embeddings."""
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def similarity(self, query: str, texts: list[str]) -> NDArray[np.float64]:
        """Compute cosine similarity between a query and each text.

        Returns a 1-D array of similarity scores in [−1, 1].
        """
        query_emb = self.encode([query])
        text_embs = self.encode(texts)
        sims = cosine_similarity(query_emb, text_embs).flatten()
        return sims

    def encode_with_embeddings(
        self, query: str, texts: list[str]
    ) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
        """Return (similarity_scores, text_embeddings) for visualization."""
        query_emb = self.encode([query])
        text_embs = self.encode(texts)
        sims = cosine_similarity(query_emb, text_embs).flatten()
        return sims, text_embs
