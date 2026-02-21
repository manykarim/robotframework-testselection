"""k-Medoids selection strategy (requires sklearn-extra)."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class KMedoidsSelection:
    """Cluster representatives via k-medoids (PAM algorithm).

    Optimizes representativeness, not dispersion.
    Each medoid is a real data point.
    Requires: pip install scikit-learn-extra
    """

    name = "kmedoids"

    def select(
        self, vectors: NDArray[np.float32], k: int, seed: int = 42
    ) -> list[int]:
        from sklearn_extra.cluster import KMedoids

        kmed = KMedoids(
            n_clusters=k,
            metric="cosine",
            method="pam",
            init="k-medoids++",
            random_state=seed,
            max_iter=300,
        )
        kmed.fit(vectors)
        return list(kmed.medoid_indices_)
