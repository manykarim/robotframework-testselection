"""Farthest Point Sampling selection strategies."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class FarthestPointSampling:
    """Greedy farthest-first traversal for maximum dispersion.

    2-approximation guarantee for max-min dispersion (Gonzalez, 1985).
    Time: O(N * k * d). Deterministic given seed.
    """

    name = "fps"

    def select(
        self, vectors: NDArray[np.float32], k: int, seed: int = 42
    ) -> list[int]:
        from sklearn.metrics.pairwise import cosine_distances

        n = vectors.shape[0]
        k = min(k, n)
        rng = np.random.RandomState(seed)
        initial = rng.randint(n)
        selected = [initial]
        min_distances = cosine_distances(vectors[initial : initial + 1], vectors)[0]
        min_distances[initial] = -np.inf
        for _ in range(k - 1):
            next_idx = int(np.argmax(min_distances))
            selected.append(next_idx)
            new_dists = cosine_distances(vectors[next_idx : next_idx + 1], vectors)[0]
            min_distances = np.minimum(min_distances, new_dists)
            min_distances[next_idx] = -np.inf
        return selected


class FPSMultiStart:
    """FPS from multiple starting points, keeps best result.

    'Best' = maximizes minimum pairwise distance in selected set.
    Mitigates initial-point sensitivity.
    """

    name = "fps_multi"

    def __init__(self, n_starts: int = 5) -> None:
        self._n_starts = n_starts

    def select(
        self, vectors: NDArray[np.float32], k: int, seed: int = 42
    ) -> list[int]:
        from sklearn.metrics.pairwise import cosine_distances

        fps = FarthestPointSampling()
        best_selected: list[int] | None = None
        best_min_dist = -1.0
        for i in range(self._n_starts):
            selected = fps.select(vectors, k, seed=seed + i)
            sel_vectors = vectors[selected]
            pairwise = cosine_distances(sel_vectors, sel_vectors)
            np.fill_diagonal(pairwise, np.inf)
            min_dist = float(pairwise.min())
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_selected = selected
        return best_selected  # type: ignore[return-value]
