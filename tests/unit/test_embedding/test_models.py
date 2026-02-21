"""Tests for embedding context domain objects."""
from __future__ import annotations

import numpy as np
import pytest

from TestSelection.embedding.models import (
    ArtifactManifest,
    Embedding,
    EmbeddingMatrix,
    ManifestEntry,
)


def _make_manifest_entries(n: int) -> tuple[ManifestEntry, ...]:
    return tuple(
        ManifestEntry(
            id=f"id-{i}",
            name=f"Test Case {i}",
            tags=("smoke", "regression"),
            suite=f"/path/to/suite_{i}.robot",
            suite_name=f"Suite {i}",
            is_datadriver=i % 2 == 0,
        )
        for i in range(n)
    )


class TestEmbedding:
    def test_is_normalized_true_for_unit_vector(self) -> None:
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb = Embedding(vector=vec)
        assert emb.is_normalized is True

    def test_is_normalized_true_for_l2_normalized_vector(self) -> None:
        vec = np.random.default_rng(42).random(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        emb = Embedding(vector=vec)
        assert emb.is_normalized is True

    def test_is_normalized_false_for_unnormalized_vector(self) -> None:
        vec = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        emb = Embedding(vector=vec)
        assert emb.is_normalized is False

    def test_dimensionality(self) -> None:
        vec = np.zeros(384, dtype=np.float32)
        emb = Embedding(vector=vec)
        assert emb.dimensionality == 384


class TestManifestEntry:
    def test_construction_with_all_fields(self) -> None:
        entry = ManifestEntry(
            id="abc123",
            name="Login Test",
            tags=("smoke", "auth"),
            suite="/tests/login.robot",
            suite_name="Login Suite",
            is_datadriver=True,
        )
        assert entry.id == "abc123"
        assert entry.name == "Login Test"
        assert entry.tags == ("smoke", "auth")
        assert entry.suite == "/tests/login.robot"
        assert entry.suite_name == "Login Suite"
        assert entry.is_datadriver is True


class TestArtifactManifest:
    def test_test_count_matches_entries(self) -> None:
        entries = _make_manifest_entries(5)
        manifest = ArtifactManifest(
            model="all-MiniLM-L6-v2",
            embedding_dim=384,
            test_count=5,
            resolve_depth=0,
            tests=entries,
        )
        assert manifest.test_count == len(manifest.tests)


class TestEmbeddingMatrix:
    def test_valid_construction(self) -> None:
        n, dim = 3, 384
        vectors = np.random.default_rng(42).random((n, dim)).astype(np.float32)
        ids = tuple(f"id-{i}" for i in range(n))
        matrix = EmbeddingMatrix(
            model_name="test-model",
            embedding_dim=dim,
            vectors=vectors,
            test_ids=ids,
        )
        assert matrix.test_count == n
        assert matrix.model_name == "test-model"
        assert matrix.embedding_dim == dim

    def test_raises_on_vector_count_mismatch(self) -> None:
        vectors = np.random.default_rng(42).random((5, 384)).astype(np.float32)
        ids = ("id-0", "id-1", "id-2")
        with pytest.raises(ValueError, match="Vector count 5 != test ID count 3"):
            EmbeddingMatrix(
                model_name="test-model",
                embedding_dim=384,
                vectors=vectors,
                test_ids=ids,
            )

    def test_raises_on_embedding_dim_mismatch(self) -> None:
        vectors = np.random.default_rng(42).random((3, 256)).astype(np.float32)
        ids = ("id-0", "id-1", "id-2")
        with pytest.raises(ValueError, match="Vector dim 256 != expected dim 384"):
            EmbeddingMatrix(
                model_name="test-model",
                embedding_dim=384,
                vectors=vectors,
                test_ids=ids,
            )

    def test_to_artifact_creates_files(self, tmp_path: pytest.TempPathFactory) -> None:
        n, dim = 3, 384
        vectors = np.random.default_rng(42).random((n, dim)).astype(np.float32)
        ids = tuple(f"id-{i}" for i in range(n))
        entries = _make_manifest_entries(n)
        matrix = EmbeddingMatrix(
            model_name="test-model",
            embedding_dim=dim,
            vectors=vectors,
            test_ids=ids,
        )

        artifact = matrix.to_artifact(tmp_path, entries, resolve_depth=2)  # type: ignore[arg-type]

        assert artifact.embeddings_path.exists()
        assert artifact.manifest_path.exists()
        assert artifact.model_name == "test-model"
        assert artifact.embedding_dim == dim
        assert artifact.test_count == n

    def test_round_trip_artifact(self, tmp_path: pytest.TempPathFactory) -> None:
        n, dim = 4, 128
        rng = np.random.default_rng(99)
        vectors = rng.random((n, dim)).astype(np.float32)
        ids = tuple(f"id-{i}" for i in range(n))
        entries = _make_manifest_entries(n)

        original = EmbeddingMatrix(
            model_name="round-trip-model",
            embedding_dim=dim,
            vectors=vectors,
            test_ids=ids,
        )
        original.to_artifact(tmp_path, entries, resolve_depth=1)  # type: ignore[arg-type]

        loaded_matrix, loaded_manifest = EmbeddingMatrix.from_artifact(tmp_path)  # type: ignore[arg-type]

        assert loaded_matrix.model_name == original.model_name
        assert loaded_matrix.embedding_dim == original.embedding_dim
        assert loaded_matrix.test_ids == original.test_ids
        np.testing.assert_array_almost_equal(
            loaded_matrix.vectors, original.vectors, decimal=5
        )
        assert loaded_manifest.model == "round-trip-model"
        assert loaded_manifest.test_count == n
        assert loaded_manifest.resolve_depth == 1
        assert len(loaded_manifest.tests) == n
        assert loaded_manifest.tests[0].name == "Test Case 0"
