from __future__ import annotations

from pathlib import Path

from TestSelection.parsing.suite_collector import RobotApiAdapter

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"
SAMPLE_ROBOT = FIXTURES_DIR / "sample.robot"


class TestParseSuite:
    def test_parse_succeeds(self) -> None:
        adapter = RobotApiAdapter()
        tests, kw_map = adapter.parse_suite(SAMPLE_ROBOT)
        assert isinstance(tests, list)
        assert isinstance(kw_map, dict)

    def test_returns_correct_number_of_tests(self) -> None:
        adapter = RobotApiAdapter()
        tests, _ = adapter.parse_suite(SAMPLE_ROBOT)
        assert len(tests) == 14

    def test_each_test_has_required_keys(self) -> None:
        adapter = RobotApiAdapter()
        tests, _ = adapter.parse_suite(SAMPLE_ROBOT)
        required_keys = {"name", "tags", "body", "source", "suite_name"}
        for test in tests:
            assert required_keys.issubset(
                test.keys()
            ), f"Test {test.get('name', '?')} missing keys: {required_keys - test.keys()}"

    def test_keyword_map_contains_user_keywords(self) -> None:
        adapter = RobotApiAdapter()
        _, kw_map = adapter.parse_suite(SAMPLE_ROBOT)
        assert "login_as_user" in kw_map
        assert "open_application" in kw_map


class TestComputeFileHashes:
    def test_file_hash_for_single_file(self) -> None:
        adapter = RobotApiAdapter()
        hashes = adapter.compute_file_hashes(SAMPLE_ROBOT)
        assert len(hashes) == 1
        key = str(SAMPLE_ROBOT)
        assert key in hashes
        assert len(hashes[key].md5) == 32

    def test_file_hash_for_directory(self) -> None:
        adapter = RobotApiAdapter()
        hashes = adapter.compute_file_hashes(FIXTURES_DIR)
        assert len(hashes) >= 1
        for fh in hashes.values():
            assert fh.path.endswith(".robot")
            assert len(fh.md5) == 32
