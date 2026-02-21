"""Facility location selection strategy (requires apricot-select)."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class FacilityLocationSelection:
    """Submodular facility location for representative selection.

    (1-1/e) ~ 0.632 approximation guarantee.
    Ensures no cluster goes unrepresented.
    Requires: pip install apricot-select
    """

    name = "facility"

    def select(
        self, vectors: NDArray[np.float32], k: int, seed: int = 42
    ) -> list[int]:
        from apricot import FacilityLocationSelection as ApricotFL

        selector = ApricotFL(k, metric="cosine", verbose=False)
        selector.fit(vectors)
        return list(selector.ranking)
