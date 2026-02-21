"""Tests for pipeline caching service."""
from __future__ import annotations

import json
from pathlib import Path

from TestSelection.pipeline.cache import CacheInvalidator


class TestCacheInvalidator:
    """Tests for CacheInvalidator."""

    def test_compute_hashes_returns_dict(self, tmp_path: Path) -> None:
        suite_dir = tmp_path / "suite"
        suite_dir.mkdir()
        (suite_dir / "test.robot").write_text("*** Test Cases ***\nTest A\n    Log    hello")
        (suite_dir / "data.csv").write_text("a,b\n1,2")

        cache = CacheInvalidator(tmp_path / "hashes.json")
        hashes = cache._compute_hashes(suite_dir)

        assert isinstance(hashes, dict)
        assert len(hashes) == 2
        for key, val in hashes.items():
            assert isinstance(key, str)
            assert isinstance(val, str)
            assert len(val) == 32  # MD5 hex digest

    def test_has_changes_true_when_no_stored_hashes(self, tmp_path: Path) -> None:
        suite_dir = tmp_path / "suite"
        suite_dir.mkdir()
        (suite_dir / "test.robot").write_text("*** Test Cases ***\nTest A\n    Log    hello")

        cache = CacheInvalidator(tmp_path / "hashes.json")
        assert cache.has_changes(suite_dir) is True

    def test_has_changes_false_when_hashes_match(self, tmp_path: Path) -> None:
        suite_dir = tmp_path / "suite"
        suite_dir.mkdir()
        (suite_dir / "test.robot").write_text("*** Test Cases ***\nTest A\n    Log    hello")

        hash_file = tmp_path / "hashes.json"
        cache = CacheInvalidator(hash_file)
        cache.save_hashes(suite_dir)

        assert cache.has_changes(suite_dir) is False

    def test_has_changes_true_when_file_changes(self, tmp_path: Path) -> None:
        suite_dir = tmp_path / "suite"
        suite_dir.mkdir()
        robot_file = suite_dir / "test.robot"
        robot_file.write_text("*** Test Cases ***\nTest A\n    Log    hello")

        hash_file = tmp_path / "hashes.json"
        cache = CacheInvalidator(hash_file)
        cache.save_hashes(suite_dir)

        # Modify the file
        robot_file.write_text("*** Test Cases ***\nTest A\n    Log    world")

        assert cache.has_changes(suite_dir) is True

    def test_save_hashes_creates_file(self, tmp_path: Path) -> None:
        suite_dir = tmp_path / "suite"
        suite_dir.mkdir()
        (suite_dir / "test.robot").write_text("*** Test Cases ***\nTest A\n    Log    hello")

        hash_file = tmp_path / "output" / "hashes.json"
        cache = CacheInvalidator(hash_file)
        cache.save_hashes(suite_dir)

        assert hash_file.exists()
        data = json.loads(hash_file.read_text())
        assert isinstance(data, dict)
        assert len(data) == 1
