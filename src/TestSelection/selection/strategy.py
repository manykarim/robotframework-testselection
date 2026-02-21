"""Selection strategy protocol and domain value objects."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class SelectionStrategy(Protocol):
    """Protocol for diversity selection algorithms.

    All implementations take a (N, d) embedding matrix and return
    k indices into that matrix.
    """

    @property
    def name(self) -> str: ...

    def select(
        self, vectors: NDArray[np.float32], k: int, seed: int = 42
    ) -> list[int]: ...


@dataclass(frozen=True)
class SelectedTest:
    """A test case that was selected by the diversity algorithm."""

    name: str
    id: str
    suite: str
    is_datadriver: bool


@dataclass(frozen=True)
class DiversityMetrics:
    """Statistics describing the diversity of a selected test subset."""

    avg_pairwise_distance: float
    min_pairwise_distance: float
    suite_coverage: int
    suite_total: int

    @property
    def suite_coverage_ratio(self) -> float:
        if self.suite_total == 0:
            return 0.0
        return self.suite_coverage / self.suite_total


@dataclass(frozen=True)
class TagFilter:
    """Pre-filter criteria applied before diversity selection."""

    include_tags: frozenset[str] = frozenset()
    exclude_tags: frozenset[str] = frozenset()
    include_datadriver: bool = True

    def matches(self, tags: frozenset[str], is_datadriver: bool) -> bool:
        if not self.include_datadriver and is_datadriver:
            return False
        normalized_tags = frozenset(t.lower() for t in tags)
        if self.include_tags and not (normalized_tags & self.include_tags):
            return False
        return not (self.exclude_tags and (normalized_tags & self.exclude_tags))


@dataclass(frozen=True)
class SelectionResult:
    """Aggregate root: the output of the selection pipeline stage."""

    strategy: str
    k: int
    seed: int
    total_tests: int
    filtered_tests: int
    selected: tuple[SelectedTest, ...]
    diversity_metrics: DiversityMetrics

    def to_json(self, path: Path) -> None:
        data = {
            "strategy": self.strategy,
            "k": self.k,
            "seed": self.seed,
            "total_tests": self.total_tests,
            "filtered_tests": self.filtered_tests,
            "selected": [
                {
                    "name": t.name,
                    "id": t.id,
                    "suite": t.suite,
                    "is_datadriver": t.is_datadriver,
                }
                for t in self.selected
            ],
            "diversity_metrics": {
                "avg_pairwise_distance": self.diversity_metrics.avg_pairwise_distance,
                "min_pairwise_distance": self.diversity_metrics.min_pairwise_distance,
                "suite_coverage": self.diversity_metrics.suite_coverage,
                "suite_total": self.diversity_metrics.suite_total,
            },
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> SelectionResult:
        data = json.loads(path.read_text())
        selected = tuple(
            SelectedTest(
                name=t["name"],
                id=t["id"],
                suite=t.get("suite", ""),
                is_datadriver=t.get("is_datadriver", False),
            )
            for t in data["selected"]
        )
        return cls(
            strategy=data["strategy"],
            k=data["k"],
            seed=data.get("seed", 42),
            total_tests=data["total_tests"],
            filtered_tests=data.get("filtered_tests", data["total_tests"]),
            selected=selected,
            diversity_metrics=DiversityMetrics(
                avg_pairwise_distance=data.get("diversity_metrics", {}).get(
                    "avg_pairwise_distance", 0.0
                ),
                min_pairwise_distance=data.get("diversity_metrics", {}).get(
                    "min_pairwise_distance", 0.0
                ),
                suite_coverage=data.get("diversity_metrics", {}).get(
                    "suite_coverage", 0
                ),
                suite_total=data.get("diversity_metrics", {}).get(
                    "suite_total", 0
                ),
            ),
        )
