from __future__ import annotations

from pathlib import Path

from TestSelection.parsing.datadriver_reader import read_datadriver_csv

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"
SAMPLE_CSV = FIXTURES_DIR / "sample_datadriver.csv"


class TestReadDatadriverCsv:
    def test_reads_csv_successfully(self) -> None:
        tests = read_datadriver_csv(SAMPLE_CSV, "Login Template")
        assert isinstance(tests, list)
        assert len(tests) > 0

    def test_returns_five_entries(self) -> None:
        tests = read_datadriver_csv(SAMPLE_CSV, "Login Template")
        assert len(tests) == 5

    def test_each_entry_has_required_keys(self) -> None:
        tests = read_datadriver_csv(SAMPLE_CSV, "Login Template")
        for entry in tests:
            assert "name" in entry
            assert "description" in entry
            assert "source" in entry
            assert entry["is_datadriver"] is True

    def test_first_entry_name(self) -> None:
        tests = read_datadriver_csv(SAMPLE_CSV, "Login Template")
        assert tests[0]["name"] == "Login admin"

    def test_comment_rows_are_skipped(self) -> None:
        tests = read_datadriver_csv(SAMPLE_CSV, "Login Template")
        names = [t["name"] for t in tests]
        assert not any(n.startswith("#") for n in names)

    def test_description_includes_template_name(self) -> None:
        tests = read_datadriver_csv(SAMPLE_CSV, "Login Template")
        for entry in tests:
            assert "Template: Login Template" in entry["description"]

    def test_source_is_csv_path(self) -> None:
        tests = read_datadriver_csv(SAMPLE_CSV, "Login Template")
        for entry in tests:
            assert entry["source"] == str(SAMPLE_CSV)
