"""Selection bounded context: diversity-based test subset selection algorithms."""

from TestSelection.selection.filtering import filter_by_tags
from TestSelection.selection.fps import FarthestPointSampling, FPSMultiStart
from TestSelection.selection.registry import StrategyRegistry, default_registry
from TestSelection.selection.strategy import (
    DiversityMetrics,
    SelectedTest,
    SelectionResult,
    SelectionStrategy,
    TagFilter,
)

__all__ = [
    "DiversityMetrics",
    "FPSMultiStart",
    "FarthestPointSampling",
    "SelectedTest",
    "SelectionResult",
    "SelectionStrategy",
    "StrategyRegistry",
    "TagFilter",
    "default_registry",
    "filter_by_tags",
]
