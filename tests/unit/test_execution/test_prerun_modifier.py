from __future__ import annotations

import json
from pathlib import Path

import pytest
from robot.api import TestSuite

from TestSelection.execution.prerun_modifier import DiversePreRunModifier

SAMPLE_ROBOT = Path(__file__).resolve().parents[2] / "fixtures" / "sample.robot"


def _write_selection(tmp_path: Path, selected: list[dict]) -> Path:
    """Write a selection JSON file and return its path."""
    selection_file = tmp_path / "selected_tests.json"
    data = {
        "strategy": "fps",
        "k": len(selected),
        "total_tests": 14,
        "filtered_tests": 14,
        "seed": 42,
        "selected": selected,
    }
    selection_file.write_text(json.dumps(data))
    return selection_file


class TestDiversePreRunModifier:
    def test_filters_correct_tests(self, tmp_path: Path) -> None:
        """Select 3 specific tests and verify only those remain."""
        selected_names = [
            "Login With Valid Credentials",
            "Search Products By Name",
            "Complete Purchase Flow",
        ]
        selection_file = _write_selection(
            tmp_path,
            [{"name": n, "is_datadriver": False} for n in selected_names],
        )

        suite = TestSuite.from_file_system(str(SAMPLE_ROBOT))
        modifier = DiversePreRunModifier(str(selection_file))
        suite.visit(modifier)

        remaining = [t.name for t in suite.tests]
        assert sorted(remaining) == sorted(selected_names)

    def test_prunes_empty_suites(self, tmp_path: Path) -> None:
        """When no tests match a child suite, that suite is removed."""
        # Select a test that does not exist -> all suites become empty
        selection_file = _write_selection(
            tmp_path,
            [{"name": "Nonexistent Test", "is_datadriver": False}],
        )

        suite = TestSuite.from_file_system(str(SAMPLE_ROBOT))
        modifier = DiversePreRunModifier(str(selection_file))
        suite.visit(modifier)

        # The top-level suite's tests should be empty
        assert len(suite.tests) == 0

    def test_stats_track_kept_and_removed(self, tmp_path: Path) -> None:
        """Stats correctly track kept/removed counts."""
        selected_names = [
            "Login With Valid Credentials",
            "Export Report As CSV",
        ]
        selection_file = _write_selection(
            tmp_path,
            [{"name": n, "is_datadriver": False} for n in selected_names],
        )

        suite = TestSuite.from_file_system(str(SAMPLE_ROBOT))
        modifier = DiversePreRunModifier(str(selection_file))
        suite.visit(modifier)

        assert modifier.stats["kept"] == 2
        assert modifier.stats["removed"] == 12  # 14 total - 2 kept

    def test_empty_selection(self, tmp_path: Path) -> None:
        """When no tests are selected, all tests are removed."""
        selection_file = _write_selection(tmp_path, [])

        suite = TestSuite.from_file_system(str(SAMPLE_ROBOT))
        modifier = DiversePreRunModifier(str(selection_file))
        suite.visit(modifier)

        assert len(suite.tests) == 0
        assert modifier.stats["kept"] == 0
        assert modifier.stats["removed"] == 14

    def test_skips_datadriver_tests_in_selection(self, tmp_path: Path) -> None:
        """DataDriver tests in the selection are ignored by PreRunModifier."""
        selection_file = _write_selection(
            tmp_path,
            [
                {"name": "Login With Valid Credentials", "is_datadriver": False},
                {"name": "DD Generated Test 1", "is_datadriver": True},
            ],
        )

        suite = TestSuite.from_file_system(str(SAMPLE_ROBOT))
        modifier = DiversePreRunModifier(str(selection_file))
        suite.visit(modifier)

        remaining = [t.name for t in suite.tests]
        assert remaining == ["Login With Valid Credentials"]

    def test_programmatic_suite_visit(self, tmp_path: Path) -> None:
        """Verify the programmatic suite.visit() approach works."""
        selection_file = _write_selection(
            tmp_path,
            [{"name": "Generate Sales Report", "is_datadriver": False}],
        )

        suite = TestSuite.from_file_system(str(SAMPLE_ROBOT))
        modifier = DiversePreRunModifier(str(selection_file))
        suite.visit(modifier)

        assert len(suite.tests) == 1
        assert suite.tests[0].name == "Generate Sales Report"
