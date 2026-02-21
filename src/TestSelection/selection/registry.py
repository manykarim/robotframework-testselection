"""Strategy registry for selection algorithms."""
from __future__ import annotations

from TestSelection.selection.fps import FarthestPointSampling, FPSMultiStart
from TestSelection.selection.strategy import SelectionStrategy


class StrategyRegistry:
    """Registry mapping strategy names to their implementation classes."""

    def __init__(self) -> None:
        self._strategies: dict[str, type] = {}

    def register(self, strategy_class: type) -> None:
        """Register a strategy class by its name attribute."""
        self._strategies[strategy_class.name] = strategy_class

    def get(self, name: str) -> SelectionStrategy:
        """Instantiate and return a strategy by name."""
        if name not in self._strategies:
            available = ", ".join(sorted(self._strategies))
            msg = (
                f"Unknown strategy {name!r}. "
                f"Available: {available}"
            )
            raise KeyError(msg)
        return self._strategies[name]()

    def available(self) -> list[str]:
        """Return names of all registered strategies."""
        return sorted(self._strategies)

    def is_available(self, name: str) -> bool:
        """Check if a strategy is registered."""
        return name in self._strategies


def _build_default_registry() -> StrategyRegistry:
    """Build the default registry with core + optional strategies."""
    registry = StrategyRegistry()
    registry.register(FarthestPointSampling)
    registry.register(FPSMultiStart)

    try:
        from TestSelection.selection.kmedoids import KMedoidsSelection

        registry.register(KMedoidsSelection)
    except ImportError:
        pass

    try:
        from TestSelection.selection.dpp import DPPSelection

        registry.register(DPPSelection)
    except ImportError:
        pass

    try:
        from TestSelection.selection.facility import FacilityLocationSelection

        registry.register(FacilityLocationSelection)
    except ImportError:
        pass

    return registry


default_registry = _build_default_registry()
