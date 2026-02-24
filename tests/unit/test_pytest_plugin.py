"""Tests for TestSelection.pytest.plugin."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from TestSelection.pytest.plugin import pytest_addoption, pytest_collection_modifyitems


class _FakeOption:
    """Simple container returned by FakeParser.getgroup().addoption()."""

    def __init__(self, name: str, **kwargs: object) -> None:
        self.name = name
        self.kwargs = kwargs


class _FakeGroup:
    """Mimics the object returned by parser.getgroup()."""

    def __init__(self) -> None:
        self.options: list[_FakeOption] = []

    def addoption(self, name: str, **kwargs: object) -> None:
        self.options.append(_FakeOption(name, **kwargs))


class _FakeParser:
    """Minimal mock for pytest.Parser that records registered options."""

    def __init__(self) -> None:
        self.groups: dict[str, _FakeGroup] = {}

    def getgroup(self, name: str, description: str = "") -> _FakeGroup:
        if name not in self.groups:
            self.groups[name] = _FakeGroup()
        return self.groups[name]


class TestPytestAddoption:
    def test_registers_diverse_k(self) -> None:
        parser = _FakeParser()
        pytest_addoption(parser)  # type: ignore[arg-type]
        group = parser.groups["diverse"]
        option_names = [o.name for o in group.options]
        assert "--diverse-k" in option_names

    def test_registers_diverse_strategy(self) -> None:
        parser = _FakeParser()
        pytest_addoption(parser)  # type: ignore[arg-type]
        group = parser.groups["diverse"]
        option_names = [o.name for o in group.options]
        assert "--diverse-strategy" in option_names

    def test_registers_diverse_seed(self) -> None:
        parser = _FakeParser()
        pytest_addoption(parser)  # type: ignore[arg-type]
        group = parser.groups["diverse"]
        option_names = [o.name for o in group.options]
        assert "--diverse-seed" in option_names

    def test_registers_diverse_cache_dir(self) -> None:
        parser = _FakeParser()
        pytest_addoption(parser)  # type: ignore[arg-type]
        group = parser.groups["diverse"]
        option_names = [o.name for o in group.options]
        assert "--diverse-cache-dir" in option_names

    def test_registers_diverse_model(self) -> None:
        parser = _FakeParser()
        pytest_addoption(parser)  # type: ignore[arg-type]
        group = parser.groups["diverse"]
        option_names = [o.name for o in group.options]
        assert "--diverse-model" in option_names

    def test_registers_all_expected_options(self) -> None:
        parser = _FakeParser()
        pytest_addoption(parser)  # type: ignore[arg-type]
        group = parser.groups["diverse"]
        option_names = {o.name for o in group.options}
        expected = {
            "--diverse-k",
            "--diverse-strategy",
            "--diverse-seed",
            "--diverse-cache-dir",
            "--diverse-model",
            "--diverse-include-markers",
            "--diverse-exclude-markers",
            "--diverse-group-parametrize",
        }
        assert expected == option_names


class TestPytestCollectionModifyItems:
    def _make_config(self, **option_overrides: object) -> MagicMock:
        """Build a mock pytest.Config with getoption support."""
        defaults = {
            "--diverse-k": 0,
            "--diverse-strategy": "fps",
            "--diverse-seed": 42,
            "--diverse-model": "all-MiniLM-L6-v2",
            "--diverse-group-parametrize": False,
        }
        defaults.update(option_overrides)

        config = MagicMock(spec=["getoption", "hook"])
        config.getoption = lambda key, default=None: defaults.get(key, default)
        return config

    def _make_items(self, count: int) -> list[MagicMock]:
        """Build a list of mock pytest.Item objects."""
        items = []
        for i in range(count):
            item = MagicMock()
            item.nodeid = f"tests/test_example.py::test_{i}"
            item.name = f"test_{i}"
            items.append(item)
        return items

    def test_k_zero_does_nothing(self) -> None:
        config = self._make_config(**{"--diverse-k": 0})
        items = self._make_items(10)
        original_items = list(items)

        pytest_collection_modifyitems(config, items)

        assert items == original_items

    def test_k_negative_does_nothing(self) -> None:
        config = self._make_config(**{"--diverse-k": -1})
        items = self._make_items(10)
        original_items = list(items)

        pytest_collection_modifyitems(config, items)

        assert items == original_items

    def test_items_count_less_than_or_equal_k_keeps_all(self) -> None:
        config = self._make_config(**{"--diverse-k": 20})
        items = self._make_items(5)
        original_items = list(items)

        pytest_collection_modifyitems(config, items)

        assert items == original_items

    def test_items_count_equal_k_keeps_all(self) -> None:
        config = self._make_config(**{"--diverse-k": 5})
        items = self._make_items(5)
        original_items = list(items)

        pytest_collection_modifyitems(config, items)

        assert items == original_items

    @patch("TestSelection.pytest.plugin._select_diverse")
    def test_selection_invoked_when_k_less_than_items(
        self, mock_select: MagicMock,
    ) -> None:
        """When k < len(items), _select_diverse is called and items are pruned."""
        mock_select.return_value = [0, 2, 4]

        config = self._make_config(**{"--diverse-k": 3})
        config.hook = MagicMock()
        items = self._make_items(10)

        pytest_collection_modifyitems(config, items)

        mock_select.assert_called_once()
        assert len(items) == 3

    @patch("TestSelection.pytest.plugin._select_diverse")
    def test_deselected_items_reported(
        self, mock_select: MagicMock,
    ) -> None:
        mock_select.return_value = [1, 3]

        config = self._make_config(**{"--diverse-k": 2})
        config.hook = MagicMock()
        items = self._make_items(5)

        pytest_collection_modifyitems(config, items)

        config.hook.pytest_deselected.assert_called_once()
        deselected = config.hook.pytest_deselected.call_args[1]["items"]
        assert len(deselected) == 3

    @patch("TestSelection.pytest.plugin._select_diverse")
    def test_selection_failure_keeps_all_items(
        self, mock_select: MagicMock,
    ) -> None:
        """When _select_diverse raises, all items should be kept."""
        mock_select.side_effect = RuntimeError("embedding failed")

        config = self._make_config(**{"--diverse-k": 3})
        items = self._make_items(10)
        original_count = len(items)

        pytest_collection_modifyitems(config, items)

        assert len(items) == original_count
