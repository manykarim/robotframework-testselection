from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from TestSelection.execution.listener import DiverseDataDriverListener


def _write_selection(tmp_path: Path, selected: list[dict]) -> Path:
    """Write a selection JSON file and return its path."""
    selection_file = tmp_path / "selected_tests.json"
    data = {
        "strategy": "fps",
        "k": len(selected),
        "total_tests": 10,
        "filtered_tests": 10,
        "seed": 42,
        "selected": selected,
    }
    selection_file.write_text(json.dumps(data))
    return selection_file


def _make_test(name: str) -> SimpleNamespace:
    """Create a minimal test-like object with a name attribute."""
    return SimpleNamespace(name=name)


def _make_suite_data(test_names: list[str]) -> SimpleNamespace:
    """Create a minimal suite data object with a mutable tests list."""
    return SimpleNamespace(tests=[_make_test(n) for n in test_names])


class TestDiverseDataDriverListener:
    def test_api_version_is_3(self) -> None:
        """Listener declares API version 3."""
        assert DiverseDataDriverListener.ROBOT_LISTENER_API_VERSION == 3

    def test_priority_is_50(self) -> None:
        """Listener priority is 50 (runs after DataDriver default)."""
        assert DiverseDataDriverListener.ROBOT_LISTENER_PRIORITY == 50

    def test_no_dd_tests_is_noop(self, tmp_path: Path) -> None:
        """With no DataDriver tests selected, start_suite is a no-op."""
        selection_file = _write_selection(
            tmp_path,
            [{"name": "Standard Test", "is_datadriver": False}],
        )
        listener = DiverseDataDriverListener(str(selection_file))

        suite_data = _make_suite_data(["Gen Test 1", "Gen Test 2", "Gen Test 3"])
        original_tests = list(suite_data.tests)

        listener.start_suite(suite_data, None)

        # Tests should be unchanged
        assert len(suite_data.tests) == len(original_tests)

    def test_filters_datadriver_tests(self, tmp_path: Path) -> None:
        """Filters generated tests to only those in selection."""
        selection_file = _write_selection(
            tmp_path,
            [
                {"name": "Gen Test 1", "is_datadriver": True},
                {"name": "Gen Test 3", "is_datadriver": True},
            ],
        )
        listener = DiverseDataDriverListener(str(selection_file))

        suite_data = _make_suite_data([
            "Gen Test 1",
            "Gen Test 2",
            "Gen Test 3",
            "Gen Test 4",
        ])

        listener.start_suite(suite_data, None)

        remaining = [t.name for t in suite_data.tests]
        assert remaining == ["Gen Test 1", "Gen Test 3"]

    def test_single_test_suite_skipped(self, tmp_path: Path) -> None:
        """Suites with <=1 test are skipped (not DataDriver suites)."""
        selection_file = _write_selection(
            tmp_path,
            [{"name": "Only Test", "is_datadriver": True}],
        )
        listener = DiverseDataDriverListener(str(selection_file))

        suite_data = _make_suite_data(["Only Test"])
        listener.start_suite(suite_data, None)

        assert len(suite_data.tests) == 1
        assert suite_data.tests[0].name == "Only Test"

    def test_stats_tracking(self, tmp_path: Path) -> None:
        """Stats track suites processed and tests filtered."""
        selection_file = _write_selection(
            tmp_path,
            [{"name": "Keep Me", "is_datadriver": True}],
        )
        listener = DiverseDataDriverListener(str(selection_file))

        suite_data = _make_suite_data(["Keep Me", "Remove 1", "Remove 2"])
        listener.start_suite(suite_data, None)

        assert listener.stats["suites_processed"] == 1
        assert listener.stats["tests_filtered"] == 2

    def test_constructor_reads_selection_file(self, tmp_path: Path) -> None:
        """Constructor correctly reads and parses the selection file."""
        selection_file = _write_selection(
            tmp_path,
            [
                {"name": "DD Test A", "is_datadriver": True},
                {"name": "Standard Test", "is_datadriver": False},
                {"name": "DD Test B", "is_datadriver": True},
            ],
        )
        listener = DiverseDataDriverListener(str(selection_file))

        assert listener._has_dd_tests is True
        assert listener._selected_dd_names == {"DD Test A", "DD Test B"}

    def test_no_match_preserves_tests(self, tmp_path: Path) -> None:
        """When DD tests are selected but none match suite, tests are preserved."""
        selection_file = _write_selection(
            tmp_path,
            [{"name": "Nonexistent DD Test", "is_datadriver": True}],
        )
        listener = DiverseDataDriverListener(str(selection_file))

        suite_data = _make_suite_data(["Test A", "Test B", "Test C"])
        listener.start_suite(suite_data, None)

        # No matches -> tests are not modified
        assert len(suite_data.tests) == 3
