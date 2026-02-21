"""Benchmarks for selection algorithms at various scales."""
from __future__ import annotations

import pytest

from TestSelection.selection.fps import FarthestPointSampling, FPSMultiStart


@pytest.mark.benchmark
class TestFPSBenchmark:
    """Benchmark FPS at various scales."""

    @pytest.mark.parametrize("n", [100, 500, 1000, 5000])
    def test_fps_scale(self, benchmark, make_normalized_vectors, n):
        k = max(1, n // 5)  # 20% of N
        vectors = make_normalized_vectors(n)
        fps = FarthestPointSampling()
        result = benchmark(fps.select, vectors, k)
        assert len(result) == k

    def test_fps_multi_n1000(self, benchmark, make_normalized_vectors):
        n = 1000
        k = 200
        vectors = make_normalized_vectors(n)
        fps_multi = FPSMultiStart(n_starts=5)
        result = benchmark(fps_multi.select, vectors, k)
        assert len(result) == k


@pytest.mark.benchmark
class TestFPSMultiBenchmark:
    """Benchmark FPSMultiStart with different start counts."""

    @pytest.mark.parametrize("n_starts", [3, 5, 10])
    def test_fps_multi_starts(self, benchmark, make_normalized_vectors, n_starts):
        n = 500
        k = 100
        vectors = make_normalized_vectors(n)
        fps_multi = FPSMultiStart(n_starts=n_starts)
        result = benchmark(fps_multi.select, vectors, k)
        assert len(result) == k
