"""Tests for the strategy registry."""
from __future__ import annotations

import pytest

from TestSelection.selection.fps import FarthestPointSampling
from TestSelection.selection.registry import StrategyRegistry, default_registry


class TestStrategyRegistry:
    def test_fps_always_available(self) -> None:
        assert default_registry.is_available("fps")

    def test_fps_multi_always_available(self) -> None:
        assert default_registry.is_available("fps_multi")

    def test_get_fps_returns_instance(self) -> None:
        strategy = default_registry.get("fps")
        assert isinstance(strategy, FarthestPointSampling)

    def test_get_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Unknown strategy"):
            default_registry.get("unknown_strategy")

    def test_available_includes_core_strategies(self) -> None:
        available = default_registry.available()
        assert "fps" in available
        assert "fps_multi" in available

    def test_register_custom_strategy(self) -> None:
        registry = StrategyRegistry()

        class DummyStrategy:
            name = "dummy"

            def select(self, vectors, k, seed=42):
                return list(range(k))

        registry.register(DummyStrategy)
        assert registry.is_available("dummy")
        instance = registry.get("dummy")
        assert instance.name == "dummy"

    def test_is_available_false_for_unregistered(self) -> None:
        registry = StrategyRegistry()
        assert not registry.is_available("nonexistent")
