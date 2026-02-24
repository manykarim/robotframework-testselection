"""Tests for TestSelection.pytest.text_builder."""
from __future__ import annotations

from TestSelection.pytest.collector import PytestTestRecord
from TestSelection.pytest.text_builder import (
    _extract_ast_features,
    build_combined_text,
)


def _make_record(**overrides: object) -> PytestTestRecord:
    """Create a PytestTestRecord with sensible defaults."""
    defaults: dict[str, object] = {
        "nodeid": "tests/test_example.py::test_something",
        "name": "test_something",
        "originalname": "test_something",
        "module_path": "tests/test_example.py",
        "module_name": "test_example",
        "class_name": None,
        "docstring": None,
        "source_code": None,
        "markers": (),
        "fixtures": (),
        "parametrize_id": None,
        "lineno": 0,
    }
    defaults.update(overrides)
    return PytestTestRecord(**defaults)  # type: ignore[arg-type]


class TestBuildCombinedText:
    def test_includes_test_name(self) -> None:
        record = _make_record(originalname="test_login_valid")
        text = build_combined_text(record)
        assert "Test: test_login_valid." in text

    def test_includes_class_name_when_present(self) -> None:
        record = _make_record(
            class_name="TestAuth",
            originalname="test_login_valid",
        )
        text = build_combined_text(record)
        assert "Test: TestAuth.test_login_valid." in text

    def test_includes_markers(self) -> None:
        record = _make_record(markers=("slow", "integration"))
        text = build_combined_text(record)
        assert "Markers: slow, integration." in text

    def test_includes_docstring(self) -> None:
        record = _make_record(
            docstring="Verify that login succeeds with valid creds.",
        )
        text = build_combined_text(record)
        assert "Verify that login succeeds with valid creds." in text

    def test_includes_fixtures(self) -> None:
        record = _make_record(fixtures=("sample_data", "db_session"))
        text = build_combined_text(record)
        assert "Uses fixtures: sample_data, db_session." in text

    def test_includes_ast_features_when_source_present(self) -> None:
        source = (
            "def test_add():\n"
            "    result = compute(1, 2)\n"
            "    assert result == 3\n"
        )
        record = _make_record(source_code=source)
        text = build_combined_text(record)
        assert "Calls:" in text
        assert "compute" in text
        assert "Verifies:" in text

    def test_no_source_code_handled_gracefully(self) -> None:
        record = _make_record(source_code=None)
        text = build_combined_text(record)
        assert "Test: test_something." in text
        assert "Calls:" not in text
        assert "Verifies:" not in text

    def test_empty_markers_not_included(self) -> None:
        record = _make_record(markers=())
        text = build_combined_text(record)
        assert "Markers:" not in text

    def test_empty_fixtures_not_included(self) -> None:
        record = _make_record(fixtures=())
        text = build_combined_text(record)
        assert "Uses fixtures:" not in text


class TestExtractAstFeatures:
    def test_extracts_function_calls(self) -> None:
        source = (
            "def test_example():\n"
            "    client.get('/users')\n"
            "    assert True\n"
        )
        calls, _assertions = _extract_ast_features(source)
        assert "client.get" in calls

    def test_assert_equality_classified(self) -> None:
        source = (
            "def test_eq():\n"
            "    assert x == y\n"
        )
        _calls, assertions = _extract_ast_features(source)
        assert "checks equality" in assertions

    def test_assert_identity_classified(self) -> None:
        source = (
            "def test_is():\n"
            "    assert x is None\n"
        )
        _calls, assertions = _extract_ast_features(source)
        assert "checks identity" in assertions

    def test_assert_membership_classified(self) -> None:
        source = (
            "def test_in():\n"
            "    assert 'a' in collection\n"
        )
        _calls, assertions = _extract_ast_features(source)
        assert "checks membership" in assertions

    def test_assert_ordering_classified(self) -> None:
        source = (
            "def test_gt():\n"
            "    assert x > 0\n"
        )
        _calls, assertions = _extract_ast_features(source)
        assert "checks ordering" in assertions

    def test_pytest_raises_classified_as_checks_exception(self) -> None:
        source = (
            "def test_raises():\n"
            "    with pytest.raises(ValueError):\n"
            "        do_thing()\n"
        )
        _calls, assertions = _extract_ast_features(source)
        assert "checks exception" in assertions

    def test_noise_functions_excluded_from_calls(self) -> None:
        source = (
            "def test_noise():\n"
            "    x = len([1, 2, 3])\n"
            "    y = str(x)\n"
            "    assert y == '3'\n"
        )
        calls, _assertions = _extract_ast_features(source)
        assert "len" not in calls
        assert "str" not in calls

    def test_syntax_error_returns_empty(self) -> None:
        source = "def test_broken(\n"
        calls, assertions = _extract_ast_features(source)
        assert calls == []
        assert assertions == []

    def test_no_function_def_returns_empty(self) -> None:
        source = "x = 1\ny = 2\n"
        calls, assertions = _extract_ast_features(source)
        assert calls == []
        assert assertions == []

    def test_deduplicates_calls(self) -> None:
        source = (
            "def test_dup():\n"
            "    do_thing()\n"
            "    do_thing()\n"
            "    assert True\n"
        )
        calls, _assertions = _extract_ast_features(source)
        assert calls.count("do_thing") == 1
