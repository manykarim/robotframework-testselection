"""Fixtures for benchmark tests generating large synthetic datasets."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def make_normalized_vectors():
    """Factory fixture that generates random normalized vectors."""

    def _make(n: int, dim: int = 384, seed: int = 42) -> np.ndarray:
        rng = np.random.RandomState(seed)
        vectors = rng.randn(n, dim).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors

    return _make


@pytest.fixture
def synthetic_test_dicts():
    """Generate 100 synthetic test case dictionaries for parsing benchmarks."""
    tests = []
    for i in range(100):
        tests.append(
            {
                "name": f"Test Case {i:03d} - Verify Feature {i % 10}",
                "tags": [f"tag_{i % 5}", f"group_{i % 3}"],
                "body": [],
                "source": f"/suite/test_{i // 10}.robot",
                "suite_name": f"Suite {i // 10}",
            }
        )
    return tests


@pytest.fixture
def synthetic_keyword_map():
    """Generate a nested keyword map for resolver benchmarks."""
    from TestSelection.shared.types import UserKeywordRef

    class FakeBodyItem:
        def __init__(self, name: str, args: tuple[str, ...] = ()) -> None:
            self.name = name
            self.args = args

    kw_map: dict[str, UserKeywordRef] = {}

    # Create 20 leaf keywords
    for i in range(20):
        kw_map[f"leaf_keyword_{i}"] = UserKeywordRef(
            name=f"Leaf Keyword {i}",
            normalized_name=f"leaf_keyword_{i}",
            body_items=(),
        )

    # Create 10 mid-level keywords that call leaf keywords
    for i in range(10):
        children = tuple(
            FakeBodyItem(f"Leaf Keyword {j}") for j in range(i, min(i + 3, 20))
        )
        kw_map[f"mid_keyword_{i}"] = UserKeywordRef(
            name=f"Mid Keyword {i}",
            normalized_name=f"mid_keyword_{i}",
            body_items=children,
        )

    # Create 5 top-level keywords that call mid keywords
    for i in range(5):
        children = tuple(
            FakeBodyItem(f"Mid Keyword {j}") for j in range(i * 2, min(i * 2 + 2, 10))
        )
        kw_map[f"top_keyword_{i}"] = UserKeywordRef(
            name=f"Top Keyword {i}",
            normalized_name=f"top_keyword_{i}",
            body_items=children,
        )

    return kw_map
