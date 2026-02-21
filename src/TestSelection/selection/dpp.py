"""Determinantal Point Process selection strategy (requires dppy)."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class DPPSelection:
    """Determinantal Point Process for probabilistic diverse sampling.

    Produces genuinely random diverse subsets.
    Useful for nightly CI runs that collectively cover more ground.
    Requires: pip install dppy
    """

    name = "dpp"

    def select(
        self, vectors: NDArray[np.float32], k: int, seed: int = 42
    ) -> list[int]:
        from dppy.finite_dpps import FiniteDPP
        from sklearn.preprocessing import normalize

        rng = np.random.RandomState(seed)
        np.random.seed(rng.randint(2**31))
        x_norm = normalize(vectors, norm="l2")
        kernel = x_norm @ x_norm.T
        kernel = (kernel + kernel.T) / 2
        dpp = FiniteDPP("likelihood", **{"L": kernel})
        dpp.sample_exact_k_dpp(size=k)
        return list(dpp.list_of_samples[-1])
