"""Experiment: Programmatic pytest test collection and metadata extraction.

This script demonstrates how to use pytest's internal API to:
1. Collect all tests from a directory without running them
2. Extract rich metadata from each collected test item
3. Report what information is available for vectorization

Usage:
    uv run python scripts/experiments/pytest_collection_experiment.py
"""
from __future__ import annotations

import inspect
import json
import textwrap
from pathlib import Path

import pytest


class CollectorPlugin:
    """A minimal pytest plugin that captures collected items without running them."""

    def __init__(self) -> None:
        self.items: list[pytest.Item] = []

    def pytest_collection_modifyitems(self, items: list[pytest.Item]) -> None:
        """Hook called after collection. We capture items and deselect all
        so nothing actually runs."""
        self.items.extend(items)
        # Deselect everything -- we only want to collect, not run
        items[:] = []


def collect_tests(test_dir: str) -> list[pytest.Item]:
    """Programmatically collect all tests from a directory without running them.

    Uses pytest.main() with a custom plugin that intercepts collected items.
    This is the cleanest and most reliable approach because it goes through
    the full pytest startup sequence (plugin loading, conftest parsing, etc.)
    without actually executing any tests.

    Alternative approaches:
    - `pytest --collect-only` via subprocess (simple but requires parsing output)
    - Direct Session.perform_collect() (lower-level, plugin compat issues)
    - pytest.Config + Session manually (most control but most fragile)
    """
    collector = CollectorPlugin()

    # Run pytest with --collect-only and our custom plugin.
    # The pyproject.toml has addopts = "-ra --strict-markers" so we
    # must keep those args compatible. --collect-only prevents execution.
    exit_code = pytest.main(
        [
            test_dir,
            "--collect-only",
            "-q",
        ],
        plugins=[collector],
    )

    print(f"  pytest.main() exit code: {exit_code}")
    return collector.items


def extract_metadata(item: pytest.Item) -> dict:
    """Extract all available metadata from a single pytest test item.

    This demonstrates the full range of metadata accessible from
    pytest's Function (test) items.
    """
    metadata = {}

    # --- 1. Basic identification ---
    metadata["nodeid"] = item.nodeid  # e.g. "tests/unit/test_foo.py::TestClass::test_method[param]"
    metadata["name"] = item.name  # e.g. "test_method[param0]"
    metadata["originalname"] = getattr(item, "originalname", item.name)  # Without parametrize suffix
    metadata["path"] = str(item.path)  # Absolute file path
    metadata["module_name"] = item.module.__name__ if hasattr(item, "module") else None

    # --- 2. Location (file, line number, test id) ---
    # reportinfo() returns (path, lineno, test_id_string)
    fspath, lineno, test_id = item.reportinfo()
    metadata["file_path"] = str(fspath)
    metadata["line_number"] = lineno
    metadata["test_id"] = test_id

    # --- 3. Class information ---
    # item.cls is the test class (if the test is inside a class), or None
    cls = getattr(item, "cls", None)
    metadata["class_name"] = cls.__name__ if cls else None
    metadata["class_qualname"] = cls.__qualname__ if cls else None

    # --- 4. Docstring ---
    # The actual test function object
    func = getattr(item, "function", None) or getattr(item, "obj", None)
    metadata["docstring"] = inspect.getdoc(func) if func else None

    # --- 5. Source code of the test function body ---
    if func is not None:
        try:
            source = inspect.getsource(func)
            metadata["source_code"] = source
            metadata["source_lines"] = len(source.splitlines())
        except (OSError, TypeError):
            metadata["source_code"] = None
            metadata["source_lines"] = None
    else:
        metadata["source_code"] = None
        metadata["source_lines"] = None

    # --- 6. Markers / Tags ---
    # iter_markers() yields all markers (own + inherited from class/module)
    markers = []
    for mark in item.iter_markers():
        markers.append({
            "name": mark.name,
            "args": [repr(a) for a in mark.args],
            "kwargs": {k: repr(v) for k, v in mark.kwargs.items()},
        })
    metadata["markers"] = markers
    metadata["marker_names"] = [m["name"] for m in markers]

    # Get specific well-known markers
    metadata["has_skip"] = item.get_closest_marker("skip") is not None
    metadata["has_skipif"] = item.get_closest_marker("skipif") is not None
    metadata["has_xfail"] = item.get_closest_marker("xfail") is not None

    # --- 7. Parametrize information ---
    callspec = getattr(item, "callspec", None)
    if callspec is not None:
        metadata["parametrize"] = {
            "params": {k: repr(v) for k, v in callspec.params.items()},
            "param_id": callspec.id,
            "param_indices": dict(callspec.indices),
        }
    else:
        metadata["parametrize"] = None

    # --- 8. Fixtures ---
    # fixturenames contains the transitive closure of all required fixtures
    fixtureinfo = getattr(item, "_fixtureinfo", None)
    if fixtureinfo is not None:
        metadata["fixtures"] = {
            "argnames": list(fixtureinfo.argnames),  # Direct function params
            "initialnames": list(fixtureinfo.initialnames),  # + usefixtures + autouse
            "all_fixtures": list(fixtureinfo.names_closure),  # Full transitive closure
        }
    else:
        metadata["fixtures"] = {
            "argnames": list(getattr(item, "fixturenames", [])),
            "initialnames": [],
            "all_fixtures": [],
        }

    # --- 9. Keywords (markers + extra keywords for -k filtering) ---
    metadata["keywords"] = sorted(str(k) for k in item.keywords)

    # --- 10. Node chain (hierarchy: Session > Package > Module > Class > Function) ---
    chain = item.listchain()
    metadata["node_chain"] = [
        {"type": type(node).__name__, "name": node.name}
        for node in chain
    ]

    # --- 11. Extra keywords (used for -k expression matching) ---
    metadata["extra_keywords"] = sorted(item.listextrakeywords())

    return metadata


def print_summary(all_metadata: list[dict]) -> None:
    """Print a summary of what metadata is available across all tests."""
    total = len(all_metadata)
    print(f"\n{'=' * 80}")
    print(f"COLLECTION SUMMARY: {total} tests collected")
    print(f"{'=' * 80}")

    # Count tests with various metadata
    with_docstring = sum(1 for m in all_metadata if m["docstring"])
    with_class = sum(1 for m in all_metadata if m["class_name"])
    with_parametrize = sum(1 for m in all_metadata if m["parametrize"])
    with_markers = sum(1 for m in all_metadata if m["markers"])
    with_source = sum(1 for m in all_metadata if m["source_code"])

    print(f"\n  Tests with docstrings:     {with_docstring:>4} / {total}")
    print(f"  Tests inside classes:      {with_class:>4} / {total}")
    print(f"  Tests with parametrize:    {with_parametrize:>4} / {total}")
    print(f"  Tests with markers:        {with_markers:>4} / {total}")
    print(f"  Tests with source code:    {with_source:>4} / {total}")

    # Unique markers
    all_markers = set()
    for m in all_metadata:
        all_markers.update(m["marker_names"])
    print(f"\n  Unique markers found: {sorted(all_markers)}")

    # Unique fixtures
    all_fixtures = set()
    for m in all_metadata:
        all_fixtures.update(m["fixtures"]["argnames"])
    # Filter out 'self' and 'request' which are not real fixtures
    real_fixtures = sorted(all_fixtures - {"self", "request"})
    print(f"  Unique fixtures used: {real_fixtures}")

    # Modules
    modules = set(m["module_name"] for m in all_metadata if m["module_name"])
    print(f"  Unique test modules:  {len(modules)}")
    for mod in sorted(modules):
        print(f"    - {mod}")

    # Source code line stats
    source_lines = [m["source_lines"] for m in all_metadata if m["source_lines"]]
    if source_lines:
        print(f"\n  Source code line stats:")
        print(f"    Min:  {min(source_lines)} lines")
        print(f"    Max:  {max(source_lines)} lines")
        print(f"    Mean: {sum(source_lines) / len(source_lines):.1f} lines")


def print_sample_items(all_metadata: list[dict], n: int = 3) -> None:
    """Print detailed metadata for a few sample items."""
    print(f"\n{'=' * 80}")
    print(f"DETAILED SAMPLE ITEMS (first {n} tests)")
    print(f"{'=' * 80}")

    for i, meta in enumerate(all_metadata[:n]):
        print(f"\n--- Test {i + 1} ---")
        print(f"  Node ID:       {meta['nodeid']}")
        print(f"  Name:          {meta['name']}")
        print(f"  Original Name: {meta['originalname']}")
        print(f"  File:          {meta['file_path']}")
        print(f"  Line:          {meta['line_number']}")
        print(f"  Class:         {meta['class_name']}")
        print(f"  Module:        {meta['module_name']}")
        print(f"  Docstring:     {meta['docstring']}")
        print(f"  Markers:       {meta['marker_names']}")
        print(f"  Parametrize:   {meta['parametrize']}")
        print(f"  Fixtures (direct):  {meta['fixtures']['argnames']}")
        print(f"  Fixtures (all):     {meta['fixtures']['all_fixtures']}")
        print(f"  Source lines:  {meta['source_lines']}")
        print(f"  Node chain:    {' > '.join(n['type'] + ':' + n['name'] for n in meta['node_chain'])}")

        if meta["source_code"]:
            truncated = "\n".join(meta["source_code"].splitlines()[:10])
            print(f"  Source (first 10 lines):")
            print(textwrap.indent(truncated, "    "))


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    test_dir = str(project_root / "tests")

    print(f"Project root: {project_root}")
    print(f"Test directory: {test_dir}")
    print(f"Pytest version: {pytest.__version__}")

    # Collect all tests
    print(f"\nCollecting tests from: {test_dir}")
    items = collect_tests(test_dir)
    print(f"Collected {len(items)} test items")

    # Extract metadata from each test
    all_metadata = []
    for item in items:
        try:
            meta = extract_metadata(item)
            all_metadata.append(meta)
        except Exception as e:
            print(f"  WARNING: Failed to extract metadata for {item.nodeid}: {e}")

    # Print detailed samples
    print_sample_items(all_metadata)

    # Print summary
    print_summary(all_metadata)

    # Print metadata keys available for vectorization
    if all_metadata:
        print(f"\n{'=' * 80}")
        print("METADATA FIELDS AVAILABLE FOR VECTORIZATION")
        print(f"{'=' * 80}")
        sample = all_metadata[0]
        for key, value in sample.items():
            vtype = type(value).__name__
            if isinstance(value, str):
                preview = value[:60] + "..." if len(value) > 60 else value
            elif isinstance(value, list):
                preview = f"[{len(value)} items]"
            elif isinstance(value, dict):
                preview = f"{{{len(value)} keys}}"
            else:
                preview = repr(value)[:60]
            print(f"  {key:25s}  ({vtype:10s})  {preview}")

    # Save full metadata to JSON for inspection
    output_path = project_root / "scripts" / "experiments" / "collected_metadata.json"
    serializable = []
    for m in all_metadata:
        s = dict(m)
        # Source code can be very large, truncate for JSON output
        if s.get("source_code") and len(s["source_code"]) > 500:
            s["source_code_truncated"] = s["source_code"][:500] + "..."
            del s["source_code"]
        serializable.append(s)

    output_path.write_text(json.dumps(serializable, indent=2, default=str))
    print(f"\nFull metadata saved to: {output_path}")


if __name__ == "__main__":
    main()
