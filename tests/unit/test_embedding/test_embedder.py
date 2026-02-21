"""Tests for the embedding adapter (ACL) using a fake implementation."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from TestSelection.embedding.ports import EmbeddingModel


class FakeEmbedder:
    """A fake embedder that satisfies the EmbeddingModel protocol.

    Returns random but deterministic L2-normalized vectors.
    """

    def __init__(
        self,
        model_name: str = "fake-model",
        embedding_dim: int = 384,
    ) -> None:
        self._model_name = model_name
        self._embedding_dim = embedding_dim

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        rng = np.random.default_rng(len(texts))
        vectors = rng.random((len(texts), self._embedding_dim)).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms


class TestFakeEmbedderProtocol:
    def test_satisfies_embedding_model_protocol(self) -> None:
        embedder = FakeEmbedder()
        assert isinstance(embedder, EmbeddingModel)

    def test_model_name(self) -> None:
        embedder = FakeEmbedder(model_name="my-test-model")
        assert embedder.model_name == "my-test-model"

    def test_embedding_dim(self) -> None:
        embedder = FakeEmbedder(embedding_dim=768)
        assert embedder.embedding_dim == 768

    def test_encode_returns_correct_shape(self) -> None:
        embedder = FakeEmbedder(embedding_dim=384)
        texts = ["hello world", "foo bar", "test case"]
        result = embedder.encode(texts)
        assert result.shape == (3, 384)
        assert result.dtype == np.float32

    def test_encode_returns_normalized_vectors(self) -> None:
        embedder = FakeEmbedder(embedding_dim=128)
        texts = ["one", "two", "three", "four"]
        result = embedder.encode(texts)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_encode_empty_list(self) -> None:
        embedder = FakeEmbedder(embedding_dim=64)
        result = embedder.encode([])
        assert result.shape == (0, 64)


class TestSentenceTransformerAdapter:
    """Tests that require the sentence-transformers package."""

    @pytest.mark.slow
    def test_adapter_satisfies_protocol(self) -> None:
        from TestSelection.embedding.embedder import SentenceTransformerAdapter

        try:
            adapter = SentenceTransformerAdapter()
        except (ImportError, ModuleNotFoundError):
            pytest.skip("sentence-transformers not installed")

        assert isinstance(adapter, EmbeddingModel)

    @pytest.mark.slow
    def test_adapter_encode(self) -> None:
        from TestSelection.embedding.embedder import SentenceTransformerAdapter

        try:
            adapter = SentenceTransformerAdapter()
        except (ImportError, ModuleNotFoundError):
            pytest.skip("sentence-transformers not installed")

        result = adapter.encode(["test case one", "test case two"])
        assert result.shape[0] == 2
        assert result.shape[1] == adapter.embedding_dim
