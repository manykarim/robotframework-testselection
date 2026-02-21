"""Tests for facility location selection strategy (optional dependency)."""
from __future__ import annotations

import numpy as np

apricot = __import__("pytest").importorskip("apricot")

from TestSelection.selection.facility import FacilityLocationSelection


class TestFacilityLocationSelection:
    def test_returns_correct_number_of_indices(self) -> None:
        rng = np.random.RandomState(0)
        vectors = rng.randn(30, 8).astype(np.float32)
        strategy = FacilityLocationSelection()
        result = strategy.select(vectors, k=5, seed=42)
        assert len(result) == 5

    def test_indices_within_range(self) -> None:
        rng = np.random.RandomState(0)
        vectors = rng.randn(30, 8).astype(np.float32)
        strategy = FacilityLocationSelection()
        result = strategy.select(vectors, k=5, seed=42)
        for idx in result:
            assert 0 <= idx < vectors.shape[0]
