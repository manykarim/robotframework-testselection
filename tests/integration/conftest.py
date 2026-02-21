from __future__ import annotations

import json

import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def sample_suite_path():
    return Path(__file__).parent.parent / "fixtures" / "sample.robot"


@pytest.fixture
def sample_csv_path():
    return Path(__file__).parent.parent / "fixtures" / "sample_datadriver.csv"


@pytest.fixture
def fake_artifacts(tmp_path, sample_suite_path):
    """Create realistic fake vector artifacts for testing Stage 2 and 3 without ML model."""
    from TestSelection.parsing.suite_collector import RobotApiAdapter

    # Parse real tests
    adapter = RobotApiAdapter()
    raw_tests, kw_map = adapter.parse_suite(sample_suite_path)

    # Generate fake 384-dim embeddings (normalized random vectors)
    n_tests = len(raw_tests)
    rng = np.random.RandomState(42)
    vectors = rng.randn(n_tests, 384).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    # Write embeddings.npz
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    np.savez_compressed(
        artifact_dir / "embeddings.npz",
        vectors=vectors,
        ids=np.array([f"test_{i:04d}" for i in range(n_tests)]),
    )

    # Write test_manifest.json
    manifest = {
        "model": "fake-test-model",
        "embedding_dim": 384,
        "test_count": n_tests,
        "resolve_depth": 0,
        "tests": [
            {
                "id": f"test_{i:04d}",
                "name": t["name"],
                "tags": t.get("tags", []),
                "suite": t.get("source", ""),
                "suite_name": t.get("suite_name", ""),
                "is_datadriver": False,
            }
            for i, t in enumerate(raw_tests)
        ],
    }
    (artifact_dir / "test_manifest.json").write_text(json.dumps(manifest, indent=2))

    return artifact_dir, raw_tests
