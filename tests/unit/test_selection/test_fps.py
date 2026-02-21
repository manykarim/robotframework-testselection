"""Tests for FPS and FPSMultiStart selection strategies."""
from __future__ import annotations

import numpy as np
import pytest

from TestSelection.selection.fps import FarthestPointSampling, FPSMultiStart


@pytest.fixture()
def fps() -> FarthestPointSampling:
    return FarthestPointSampling()


@pytest.fixture()
def random_vectors() -> np.ndarray:
    rng = np.random.RandomState(0)
    return rng.randn(20, 8).astype(np.float32)


@pytest.fixture()
def corner_vectors() -> np.ndarray:
    """4 corners of a square + center point in 2D, embedded in float32."""
    return np.array(
        [
            [1.0, 1.0],   # corner 0
            [1.0, -1.0],  # corner 1
            [-1.0, 1.0],  # corner 2
            [-1.0, -1.0], # corner 3
            [0.0, 0.01],  # near-center (avoid exact zero for cosine)
        ],
        dtype=np.float32,
    )


class TestFarthestPointSampling:
    def test_deterministic_with_seed(
        self, fps: FarthestPointSampling, random_vectors: np.ndarray,
    ) -> None:
        result1 = fps.select(random_vectors, k=5, seed=42)
        result2 = fps.select(random_vectors, k=5, seed=42)
        assert result1 == result2

    def test_k_equals_1_returns_single_index(
        self, fps: FarthestPointSampling, random_vectors: np.ndarray,
    ) -> None:
        result = fps.select(random_vectors, k=1, seed=42)
        assert len(result) == 1
        assert 0 <= result[0] < random_vectors.shape[0]

    def test_k_equals_n_returns_all_indices(
        self, fps: FarthestPointSampling, random_vectors: np.ndarray,
    ) -> None:
        n = random_vectors.shape[0]
        result = fps.select(random_vectors, k=n, seed=42)
        assert len(result) == n
        assert set(result) == set(range(n))

    def test_selected_indices_within_valid_range(
        self, fps: FarthestPointSampling, random_vectors: np.ndarray,
    ) -> None:
        result = fps.select(random_vectors, k=7, seed=42)
        assert len(result) == 7
        for idx in result:
            assert 0 <= idx < random_vectors.shape[0]

    def test_no_duplicate_indices(
        self, fps: FarthestPointSampling, random_vectors: np.ndarray,
    ) -> None:
        result = fps.select(random_vectors, k=10, seed=42)
        assert len(result) == len(set(result))

    def test_corner_geometry_selects_corners(
        self, fps: FarthestPointSampling, corner_vectors: np.ndarray,
    ) -> None:
        """With 4 corners + center, k=4 should select the 4 corners."""
        result = fps.select(corner_vectors, k=4, seed=42)
        assert len(result) == 4
        # All 4 corner indices (0-3) should be selected; center (4) excluded
        assert set(result) == {0, 1, 2, 3}


class TestFPSMultiStart:
    def test_returns_correct_count(self, random_vectors: np.ndarray) -> None:
        multi = FPSMultiStart(n_starts=3)
        result = multi.select(random_vectors, k=5, seed=42)
        assert len(result) == 5

    def test_indices_within_range(self, random_vectors: np.ndarray) -> None:
        multi = FPSMultiStart(n_starts=3)
        result = multi.select(random_vectors, k=5, seed=42)
        for idx in result:
            assert 0 <= idx < random_vectors.shape[0]

    def test_deterministic_with_seed(self, random_vectors: np.ndarray) -> None:
        multi = FPSMultiStart(n_starts=3)
        result1 = multi.select(random_vectors, k=5, seed=42)
        result2 = multi.select(random_vectors, k=5, seed=42)
        assert result1 == result2

    def test_no_duplicate_indices(self, random_vectors: np.ndarray) -> None:
        multi = FPSMultiStart(n_starts=5)
        result = multi.select(random_vectors, k=8, seed=42)
        assert len(result) == len(set(result))
