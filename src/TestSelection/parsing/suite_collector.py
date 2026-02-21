from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from robot.api import TestSuite as RobotTestSuite

from TestSelection.shared.types import FileHash, UserKeywordRef


class RobotApiAdapter:
    """ACL: Translates robot.api types into domain objects.

    Uses TestSuite.from_file_system() as the primary entry point,
    which is the recommended robot.api approach for building a
    complete suite model from .robot files or directories.
    """

    def parse_suite(
        self, suite_path: Path
    ) -> tuple[list[dict[str, Any]], dict[str, UserKeywordRef]]:
        """Parse a suite path and return (raw_tests, keyword_map).

        Returns raw test dicts and a domain keyword map.
        Callers use TextRepresentationBuilder to convert to TestCaseRecords.
        """
        robot_suite = RobotTestSuite.from_file_system(str(suite_path))
        kw_map = self._build_keyword_map(robot_suite)
        raw_tests = self._collect_tests(robot_suite)
        return raw_tests, kw_map

    def _build_keyword_map(
        self, suite: Any
    ) -> dict[str, UserKeywordRef]:
        kw_map: dict[str, UserKeywordRef] = {}
        if hasattr(suite, "resource") and suite.resource:
            for uk in suite.resource.keywords:
                ref = UserKeywordRef(
                    name=uk.name,
                    normalized_name=uk.name.lower().replace(" ", "_"),
                    body_items=tuple(uk.body),
                )
                kw_map[ref.normalized_name] = ref
        for child in suite.suites:
            kw_map.update(self._build_keyword_map(child))
        return kw_map

    def _collect_tests(self, suite: Any) -> list[dict[str, Any]]:
        """Collect raw test data from suite hierarchy."""
        tests: list[dict[str, Any]] = []
        for test in suite.tests:
            tests.append(
                {
                    "name": test.name,
                    "tags": [str(t) for t in test.tags],
                    "body": list(test.body),
                    "source": str(suite.source) if suite.source else suite.name,
                    "suite_name": suite.name,
                }
            )
        for child in suite.suites:
            tests.extend(self._collect_tests(child))
        return tests

    def compute_file_hashes(
        self, suite_path: Path
    ) -> dict[str, FileHash]:
        """Hash all .robot files for change detection."""
        hashes: dict[str, FileHash] = {}
        target = suite_path if suite_path.is_dir() else suite_path.parent
        patterns = ["*.robot"]
        if not suite_path.is_dir():
            patterns = []
            if suite_path.suffix == ".robot":
                md5 = hashlib.md5(suite_path.read_bytes()).hexdigest()
                hashes[str(suite_path)] = FileHash(
                    path=str(suite_path), md5=md5
                )
                return hashes
        for pattern in patterns:
            for p in target.rglob(pattern):
                md5 = hashlib.md5(p.read_bytes()).hexdigest()
                hashes[str(p)] = FileHash(path=str(p), md5=md5)
        return hashes
