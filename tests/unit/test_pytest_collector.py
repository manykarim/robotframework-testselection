"""Tests for TestSelection.pytest.collector."""
from __future__ import annotations

from pathlib import Path

from TestSelection.pytest.collector import (
    PytestTestRecord,
    collect_and_extract,
    collect_tests,
    extract_record,
)

SAMPLE_SUITE = (
    Path(__file__).resolve().parent.parent / "fixtures" / "sample_pytest_suite"
)


class TestCollectTests:
    def test_returns_items_from_sample_suite(self) -> None:
        items = collect_tests(SAMPLE_SUITE)
        assert len(items) > 0
        # Every item should have a nodeid that contains the suite path
        for item in items:
            assert hasattr(item, "nodeid")

    def test_collected_count_matches_expected(self) -> None:
        """The sample suite has ~14 test functions (including parametrize variants)."""
        items = collect_tests(SAMPLE_SUITE)
        # test_auth: 4, test_api: 4, test_helpers: 2 + 3 parametrize = 5 -> 13
        assert len(items) >= 13


class TestExtractRecord:
    def test_extract_returns_pytest_test_record(self) -> None:
        items = collect_tests(SAMPLE_SUITE)
        record = extract_record(items[0])
        assert isinstance(record, PytestTestRecord)

    def test_extracted_fields_are_populated(self) -> None:
        items = collect_tests(SAMPLE_SUITE)
        record = extract_record(items[0])
        assert record.nodeid
        assert record.name
        assert record.module_path
        assert record.module_name
        assert isinstance(record.markers, tuple)
        assert isinstance(record.fixtures, tuple)

    def test_docstring_extracted_when_present(self) -> None:
        """test_login_valid has a docstring."""
        items = collect_tests(SAMPLE_SUITE)
        login_items = [i for i in items if "test_login_valid" in i.nodeid]
        assert login_items, "Expected to find test_login_valid in sample suite"
        record = extract_record(login_items[0])
        assert record.docstring is not None
        assert "valid credentials" in record.docstring.lower()

    def test_source_code_extracted(self) -> None:
        items = collect_tests(SAMPLE_SUITE)
        record = extract_record(items[0])
        assert record.source_code is not None
        assert "def " in record.source_code

    def test_markers_extracted(self) -> None:
        """test_session_timeout has @pytest.mark.slow."""
        items = collect_tests(SAMPLE_SUITE)
        slow_items = [i for i in items if "test_session_timeout" in i.nodeid]
        assert slow_items, "Expected to find test_session_timeout"
        record = extract_record(slow_items[0])
        assert "slow" in record.markers

    def test_fixtures_extracted(self) -> None:
        """test_get_users uses the sample_data fixture."""
        items = collect_tests(SAMPLE_SUITE)
        api_items = [i for i in items if "test_get_users" in i.nodeid]
        assert api_items, "Expected to find test_get_users"
        record = extract_record(api_items[0])
        assert "sample_data" in record.fixtures


class TestParametrizeHandling:
    def test_parametrized_tests_have_parametrize_id(self) -> None:
        items = collect_tests(SAMPLE_SUITE)
        parse_items = [i for i in items if "test_parse_config" in i.nodeid]
        assert len(parse_items) == 3, "Expected 3 parametrize variants"
        for item in parse_items:
            record = extract_record(item)
            assert record.parametrize_id is not None

    def test_non_parametrized_tests_have_none_parametrize_id(self) -> None:
        items = collect_tests(SAMPLE_SUITE)
        login_items = [i for i in items if "test_login_valid" in i.nodeid]
        record = extract_record(login_items[0])
        assert record.parametrize_id is None


class TestClassBasedTests:
    """Verify class_name handling.

    The sample suite uses only module-level functions, so class_name should be None.
    """

    def test_class_name_is_none_for_function_tests(self) -> None:
        items = collect_tests(SAMPLE_SUITE)
        record = extract_record(items[0])
        assert record.class_name is None


class TestCollectAndExtract:
    def test_end_to_end(self) -> None:
        records = collect_and_extract(SAMPLE_SUITE)
        assert len(records) >= 13
        assert all(isinstance(r, PytestTestRecord) for r in records)

    def test_all_records_have_nodeids(self) -> None:
        records = collect_and_extract(SAMPLE_SUITE)
        for record in records:
            assert record.nodeid
            assert record.name
