"""Benchmarks for pytest text building and collection."""
from __future__ import annotations

import pytest

from TestSelection.pytest.collector import PytestTestRecord
from TestSelection.pytest.text_builder import (
    _extract_ast_features,
    build_combined_text,
)


def _make_record(
    name: str = "test_example",
    class_name: str | None = "TestSuite",
    markers: tuple[str, ...] = ("slow",),
    fixtures: tuple[str, ...] = ("tmp_path",),
    docstring: str | None = "Verify example behavior.",
    source_code: str | None = None,
) -> PytestTestRecord:
    if source_code is None:
        source_code = (
            f"def {name}(self, tmp_path):\n"
            f"    result = do_something(tmp_path)\n"
            f"    assert result == 42\n"
        )
    return PytestTestRecord(
        nodeid=(
            f"tests/test_example.py::{class_name}.{name}"
            if class_name
            else f"tests/test_example.py::{name}"
        ),
        name=name,
        originalname=name,
        module_path="tests/test_example.py",
        module_name="test_example",
        class_name=class_name,
        docstring=docstring,
        source_code=source_code,
        markers=markers,
        fixtures=fixtures,
        parametrize_id=None,
        lineno=10,
    )


@pytest.mark.benchmark
class TestTextBuilderBenchmark:
    """Benchmark text building for pytest tests."""

    def test_build_combined_text_single(self, benchmark):
        record = _make_record()
        result = benchmark(build_combined_text, record)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_build_combined_text_100(self, benchmark):
        records = [
            _make_record(
                name=f"test_case_{i:03d}",
                class_name=f"TestGroup{i % 5}",
                markers=(f"tag_{i % 3}",),
                fixtures=(f"fixture_{i % 4}",),
                docstring=f"Verify feature {i} works correctly.",
                source_code=(
                    f"def test_case_{i:03d}(self, fixture_{i % 4}):\n"
                    f"    result = process_{i % 10}(fixture_{i % 4})\n"
                    f"    assert result is not None\n"
                    f"    assert len(result) > 0\n"
                ),
            )
            for i in range(100)
        ]

        def _build_all():
            return [build_combined_text(r) for r in records]

        results = benchmark(_build_all)
        assert len(results) == 100

    def test_build_combined_text_no_source(self, benchmark):
        record = _make_record(source_code=None, docstring=None)
        # Override to actually have None source
        record = PytestTestRecord(
            nodeid=record.nodeid,
            name=record.name,
            originalname=record.originalname,
            module_path=record.module_path,
            module_name=record.module_name,
            class_name=record.class_name,
            docstring=None,
            source_code=None,
            markers=record.markers,
            fixtures=record.fixtures,
            parametrize_id=None,
            lineno=10,
        )
        result = benchmark(build_combined_text, record)
        assert isinstance(result, str)


@pytest.mark.benchmark
class TestASTExtractionBenchmark:
    """Benchmark AST feature extraction at various complexity levels."""

    @pytest.mark.parametrize("n_lines", [5, 20, 50])
    def test_extract_features_scale(self, benchmark, n_lines):
        lines = ["def test_example():\n"]
        for i in range(n_lines):
            lines.append(f"    result_{i} = process_{i}(data)\n")
            if i % 3 == 0:
                lines.append(f"    assert result_{i} == expected_{i}\n")
        source = "".join(lines)
        calls, assertions = benchmark(_extract_ast_features, source)
        assert isinstance(calls, list)
        assert isinstance(assertions, list)

    def test_extract_features_complex(self, benchmark):
        source = (
            "def test_complex_flow():\n"
            "    client = APIClient()\n"
            "    response = client.get('/api/users')\n"
            "    assert response.status_code == 200\n"
            "    data = response.json()\n"
            "    assert isinstance(data, list)\n"
            "    for user in data:\n"
            "        assert 'name' in user\n"
            "        validate_email(user['email'])\n"
            "    with pytest.raises(ValueError):\n"
            "        client.get('/api/invalid')\n"
        )
        calls, assertions = benchmark(_extract_ast_features, source)
        assert len(calls) > 0
        assert len(assertions) > 0
