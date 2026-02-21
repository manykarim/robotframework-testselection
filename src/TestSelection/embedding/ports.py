"""Embedding model protocol -- the port for embedding adapters."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol for embedding models.

    Decouples the domain from sentence-transformers or any other
    embedding backend. Implementations must be L2-normalizing.
    """

    @property
    def model_name(self) -> str: ...

    @property
    def embedding_dim(self) -> int: ...

    def encode(self, texts: list[str]) -> NDArray[np.float32]: ...
