"""SentenceTransformer adapter -- ACL wrapping sentence-transformers."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SentenceTransformerAdapter:
    """ACL: Wraps sentence-transformers behind the EmbeddingModel protocol.

    Default model: all-MiniLM-L6-v2 (22M params, 384 dims, CPU-friendly).
    Alternative: all-mpnet-base-v2 (768 dims) for higher quality.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        show_progress_bar: bool = False,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        self._show_progress_bar = show_progress_bar

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        dim = self._model.get_sentence_embedding_dimension()
        assert isinstance(dim, int)
        return dim

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        result: NDArray[np.float32] = self._model.encode(
            texts,
            show_progress_bar=self._show_progress_bar,
            normalize_embeddings=True,
        )
        return result
