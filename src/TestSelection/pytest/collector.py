"""Programmatic pytest test collection without execution."""
from __future__ import annotations

import contextlib
import inspect
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True)
class PytestTestRecord:
    """Metadata extracted from a single collected pytest test item."""

    nodeid: str
    name: str
    originalname: str
    module_path: str
    module_name: str
    class_name: str | None
    docstring: str | None
    source_code: str | None
    markers: tuple[str, ...]
    fixtures: tuple[str, ...]
    parametrize_id: str | None
    lineno: int


class _CollectorPlugin:
    """Internal pytest plugin that captures collected items."""

    def __init__(self) -> None:
        self.items: list[pytest.Item] = []

    def pytest_collection_modifyitems(self, items: list[pytest.Item]) -> None:
        self.items.extend(items)
        items[:] = []  # Deselect all — collect only


def collect_tests(test_dir: str | Path) -> list[pytest.Item]:
    """Collect all pytest tests from a directory without running them.

    Uses pytest.main() with a plugin that intercepts collected items.
    Goes through the full pytest startup sequence (conftest parsing,
    plugin loading, etc.).
    """
    collector = _CollectorPlugin()
    pytest.main(
        [str(test_dir), "--collect-only", "-q", "--no-header"],
        plugins=[collector],
    )
    return collector.items


def extract_record(item: pytest.Item) -> PytestTestRecord:
    """Extract metadata from a pytest test item into a PytestTestRecord."""
    func = getattr(item, "function", None) or getattr(item, "obj", None)

    # Docstring
    docstring = inspect.getdoc(func) if func else None

    # Source code
    source_code = None
    if func is not None:
        with contextlib.suppress(OSError, TypeError):
            source_code = inspect.getsource(func)

    # Markers (own + inherited)
    marker_names: list[str] = []
    for mark in item.iter_markers():
        if mark.name not in ("parametrize", "usefixtures"):
            marker_names.append(mark.name)

    # Fixtures (direct arguments)
    fixtures: list[str] = []
    fixtureinfo = getattr(item, "_fixtureinfo", None)
    if fixtureinfo is not None:
        fixtures = [
            name for name in fixtureinfo.argnames
            if name not in ("self", "cls", "request")
        ]
    else:
        fixtures = [
            name for name in getattr(item, "fixturenames", [])
            if name not in ("self", "cls", "request")
        ]

    # Parametrize id
    callspec = getattr(item, "callspec", None)
    parametrize_id = callspec.id if callspec is not None else None

    # Module info
    fspath, lineno, _ = item.reportinfo()
    cls = getattr(item, "cls", None)

    return PytestTestRecord(
        nodeid=item.nodeid,
        name=item.name,
        originalname=getattr(item, "originalname", item.name),
        module_path=str(item.path),
        module_name=item.module.__name__ if hasattr(item, "module") else "",
        class_name=cls.__name__ if cls else None,
        docstring=docstring,
        source_code=source_code,
        markers=tuple(marker_names),
        fixtures=tuple(fixtures),
        parametrize_id=parametrize_id,
        lineno=lineno if isinstance(lineno, int) else 0,
    )


def collect_and_extract(test_dir: str | Path) -> list[PytestTestRecord]:
    """Collect tests and extract records in one call."""
    items = collect_tests(test_dir)
    records: list[PytestTestRecord] = []
    for item in items:
        with contextlib.suppress(Exception):
            records.append(extract_record(item))
    return records
