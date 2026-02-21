from __future__ import annotations

import json
from pathlib import Path


class DiverseDataDriverListener:
    """Listener v3 that filters DataDriver-generated tests.

    Must run AFTER DataDriver. Lower priority = runs later.

    Usage::

        robot --listener module.DiverseDataDriverListener:file tests/
    """

    ROBOT_LISTENER_API_VERSION = 3
    ROBOT_LISTENER_PRIORITY = 50  # lower than default -> runs after DataDriver

    def __init__(self, selection_file: str) -> None:
        data = json.loads(Path(selection_file).read_text())
        self._selected_dd_names: set[str] = set(
            t["name"]
            for t in data["selected"]
            if t.get("is_datadriver", False)
        )
        self._has_dd_tests = bool(self._selected_dd_names)
        self._stats = {"suites_processed": 0, "tests_filtered": 0}

    def start_suite(self, data, result) -> None:  # noqa: ARG002
        if not self._has_dd_tests:
            return
        if len(data.tests) <= 1:
            return  # not a DataDriver suite
        original_count = len(data.tests)
        matches = [t for t in data.tests if t.name in self._selected_dd_names]
        if matches:
            data.tests = matches
            self._stats["suites_processed"] += 1
            self._stats["tests_filtered"] += original_count - len(matches)

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)
