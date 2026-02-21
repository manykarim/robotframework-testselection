"""Tests for pipeline artifact management."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from TestSelection.pipeline.artifacts import ArtifactManager


def _create_test_artifacts(
    artifact_dir: Path,
    n_tests: int = 20,
    dim: int = 384,
    seed: int = 42,
) -> None:
    """Create test artifacts programmatically."""
    rng = np.random.RandomState(seed)
    vectors = rng.randn(n_tests, dim).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = np.array([f"t{i}" for i in range(n_tests)])

    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_dir / "embeddings.npz",
        vectors=vectors,
        ids=ids,
    )

    manifest = {
        "model": "all-MiniLM-L6-v2",
        "embedding_dim": dim,
        "test_count": n_tests,
        "resolve_depth": 0,
        "tests": [
            {
                "id": f"t{i}",
                "name": f"Test {i}",
                "tags": ["smoke"] if i % 2 == 0 else ["regression"],
                "suite": f"/suite/test_{i // 5}.robot",
                "suite_name": f"Suite{i // 5}",
                "is_datadriver": False,
            }
            for i in range(n_tests)
        ],
    }
    (artifact_dir / "test_manifest.json").write_text(json.dumps(manifest, indent=2))


class TestArtifactManager:
    """Tests for ArtifactManager."""

    def test_creates_directory_on_init(self, tmp_path: Path) -> None:
        d = tmp_path / "new_dir" / "artifacts"
        ArtifactManager(d)
        assert d.exists()

    def test_has_embedding_artifacts_false_when_empty(self, tmp_path: Path) -> None:
        manager = ArtifactManager(tmp_path / "empty")
        assert manager.has_embedding_artifacts() is False

    def test_has_embedding_artifacts_true_after_creating(self, tmp_path: Path) -> None:
        d = tmp_path / "arts"
        _create_test_artifacts(d)
        manager = ArtifactManager(d)
        assert manager.has_embedding_artifacts() is True

    def test_load_manifest_parses_correctly(self, tmp_path: Path) -> None:
        d = tmp_path / "arts"
        _create_test_artifacts(d, n_tests=10)
        manager = ArtifactManager(d)
        manifest = manager.load_manifest()

        assert manifest.model == "all-MiniLM-L6-v2"
        assert manifest.embedding_dim == 384
        assert manifest.test_count == 10
        assert len(manifest.tests) == 10
        assert manifest.tests[0].name == "Test 0"

    def test_load_vectors_correct_shape(self, tmp_path: Path) -> None:
        d = tmp_path / "arts"
        _create_test_artifacts(d, n_tests=15, dim=384)
        manager = ArtifactManager(d)
        vectors = manager.load_vectors()

        assert vectors.shape == (15, 384)
        assert vectors.dtype == np.float32

    def test_validate_artifacts_detects_shape_mismatch(self, tmp_path: Path) -> None:
        d = tmp_path / "arts"
        _create_test_artifacts(d, n_tests=20)

        # Overwrite manifest with wrong test_count
        manifest_path = d / "test_manifest.json"
        data = json.loads(manifest_path.read_text())
        data["test_count"] = 10  # mismatch with actual 20 vectors
        manifest_path.write_text(json.dumps(data))

        manager = ArtifactManager(d)
        valid, message = manager.validate_artifacts()

        assert valid is False
        assert "mismatch" in message.lower()

    def test_validate_artifacts_passes_for_valid(self, tmp_path: Path) -> None:
        d = tmp_path / "arts"
        _create_test_artifacts(d, n_tests=20)
        manager = ArtifactManager(d)
        valid, message = manager.validate_artifacts()

        assert valid is True
        assert message == "Artifacts valid"
