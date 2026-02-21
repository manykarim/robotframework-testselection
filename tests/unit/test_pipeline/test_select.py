"""Tests for pipeline selection stage."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from TestSelection.pipeline.errors import SelectionError
from TestSelection.pipeline.select import run_select
from TestSelection.selection.strategy import SelectionResult


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


class TestRunSelect:
    """Tests for run_select."""

    def test_valid_artifacts_produces_result(self, tmp_path: Path) -> None:
        d = tmp_path / "arts"
        _create_test_artifacts(d, n_tests=20)

        result = run_select(artifact_dir=d, k=5)

        assert isinstance(result, SelectionResult)
        assert len(result.selected) == 5
        assert result.strategy == "fps"
        assert result.total_tests == 20

    def test_fps_deterministic(self, tmp_path: Path) -> None:
        d = tmp_path / "arts"
        _create_test_artifacts(d, n_tests=20)

        result1 = run_select(artifact_dir=d, k=5, seed=42)
        result2 = run_select(artifact_dir=d, k=5, seed=42)

        names1 = [t.name for t in result1.selected]
        names2 = [t.name for t in result2.selected]
        assert names1 == names2

    def test_invalid_artifacts_raises_error(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()

        with pytest.raises(SelectionError, match="[Aa]rtifact"):
            run_select(artifact_dir=d, k=5)

    def test_selection_result_json_roundtrip(self, tmp_path: Path) -> None:
        d = tmp_path / "arts"
        _create_test_artifacts(d, n_tests=20)
        output_file = tmp_path / "selected.json"

        result = run_select(artifact_dir=d, k=5, output_file=output_file)

        assert output_file.exists()
        loaded = SelectionResult.from_json(output_file)

        assert loaded.strategy == result.strategy
        assert loaded.k == result.k
        assert loaded.seed == result.seed
        assert len(loaded.selected) == len(result.selected)
        assert [t.name for t in loaded.selected] == [t.name for t in result.selected]

    def test_k_clamped_to_total(self, tmp_path: Path) -> None:
        d = tmp_path / "arts"
        _create_test_artifacts(d, n_tests=5)

        result = run_select(artifact_dir=d, k=100)

        assert len(result.selected) == 5

    def test_diversity_metrics_computed(self, tmp_path: Path) -> None:
        d = tmp_path / "arts"
        _create_test_artifacts(d, n_tests=20)

        result = run_select(artifact_dir=d, k=10)

        assert result.diversity_metrics.avg_pairwise_distance > 0
        assert result.diversity_metrics.min_pairwise_distance > 0
        assert result.diversity_metrics.suite_coverage > 0
