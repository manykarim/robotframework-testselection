from __future__ import annotations

import json
from pathlib import Path

from robot.api import SuiteVisitor


class DiversePreRunModifier(SuiteVisitor):
    """PreRunModifier that filters standard (non-DataDriver) tests.

    Usage CLI::

        robot --prerunmodifier module.DiversePreRunModifier:file tests/

    Usage programmatic:
        suite.visit(DiversePreRunModifier('selected_tests.json'))
    """

    def __init__(self, selection_file: str) -> None:
        data = json.loads(Path(selection_file).read_text())
        self._selected_names: set[str] = set(
            t["name"]
            for t in data["selected"]
            if not t.get("is_datadriver", False)
        )
        self._stats = {"kept": 0, "removed": 0}

    def start_suite(self, suite) -> None:  # type: ignore[override]
        original = len(suite.tests)
        suite.tests = [t for t in suite.tests if t.name in self._selected_names]
        self._stats["kept"] += len(suite.tests)
        self._stats["removed"] += original - len(suite.tests)

    def end_suite(self, suite) -> None:  # type: ignore[override]
        suite.suites = [s for s in suite.suites if s.test_count > 0]

    def visit_test(self, test) -> None:  # type: ignore[override]
        pass  # skip internals for performance

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)
