"""Determinantal Point Process selection strategy (requires dppy)."""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class DPPSelection:
    """Determinantal Point Process for probabilistic diverse sampling.

    Produces genuinely random diverse subsets.
    Useful for nightly CI runs that collectively cover more ground.
    Requires: pip install dppy

    Note: exact k-DPP sampling is numerically unstable for large k
    (typically k > N/3). For large k, falls back to MCMC sampling.
    """

    name = "dpp"

    def select(
        self, vectors: NDArray[np.float32], k: int, seed: int = 42
    ) -> list[int]:
        from dppy.finite_dpps import FiniteDPP
        from sklearn.metrics.pairwise import rbf_kernel
        from sklearn.preprocessing import normalize

        n = vectors.shape[0]
        rng = np.random.RandomState(seed)
        np.random.seed(rng.randint(2**31))
        x_norm = normalize(vectors.astype(np.float64), norm="l2")
        # RBF kernel is guaranteed non-negative and PSD
        kernel = rbf_kernel(x_norm)
        kernel = (kernel + kernel.T) / 2

        dpp = FiniteDPP("likelihood", **{"L": kernel})
        # Pre-compute eigendecomposition with clamped eigenvalues
        # to avoid dppy rejecting ~-1e-15 numerical artifacts
        dpp.L_eig_vals, dpp.eig_vecs = np.linalg.eigh(kernel)
        dpp.L_eig_vals = np.maximum(dpp.L_eig_vals, 0.0)

        try:
            dpp.sample_exact_k_dpp(size=k)
            return list(dpp.list_of_samples[-1])
        except (ValueError, FloatingPointError):
            # Exact k-DPP is numerically unstable when k exceeds the
            # effective rank of the kernel (common for k > N/3 with
            # sentence embeddings). Fall back to FPS.
            logger.warning(
                "Exact k-DPP unstable for k=%d/N=%d, falling back to FPS.",
                k, n,
            )
            from TestSelection.selection.fps import FarthestPointSampling

            return FarthestPointSampling().select(vectors, k, seed=seed)
