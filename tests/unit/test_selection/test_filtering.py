"""Tests for tag-based filtering."""
from __future__ import annotations

from dataclasses import dataclass

from TestSelection.selection.filtering import filter_by_tags
from TestSelection.selection.strategy import TagFilter


@dataclass(frozen=True)
class FakeManifestEntry:
    """Minimal manifest entry for testing."""

    name: str
    tags: tuple[str, ...]
    is_datadriver: bool


def _make_entries() -> list[FakeManifestEntry]:
    return [
        FakeManifestEntry(name="Login Test", tags=("smoke", "auth"), is_datadriver=False),
        FakeManifestEntry(name="DD Checkout", tags=("checkout",), is_datadriver=True),
        FakeManifestEntry(name="Search Test", tags=("search", "smoke"), is_datadriver=False),
        FakeManifestEntry(name="Admin Panel", tags=("admin",), is_datadriver=False),
        FakeManifestEntry(name="DD Report", tags=("report",), is_datadriver=True),
    ]


class TestFilterByTags:
    def test_empty_filter_matches_everything(self) -> None:
        entries = _make_entries()
        tag_filter = TagFilter()
        result = filter_by_tags(entries, tag_filter)
        assert result == [0, 1, 2, 3, 4]

    def test_include_tags_filters_correctly(self) -> None:
        entries = _make_entries()
        tag_filter = TagFilter(include_tags=frozenset({"smoke"}))
        result = filter_by_tags(entries, tag_filter)
        assert result == [0, 2]

    def test_exclude_tags_filters_correctly(self) -> None:
        entries = _make_entries()
        tag_filter = TagFilter(exclude_tags=frozenset({"admin"}))
        result = filter_by_tags(entries, tag_filter)
        assert result == [0, 1, 2, 4]

    def test_include_datadriver_false_excludes_dd(self) -> None:
        entries = _make_entries()
        tag_filter = TagFilter(include_datadriver=False)
        result = filter_by_tags(entries, tag_filter)
        assert result == [0, 2, 3]

    def test_combined_include_and_exclude(self) -> None:
        entries = _make_entries()
        tag_filter = TagFilter(
            include_tags=frozenset({"smoke"}),
            exclude_tags=frozenset({"auth"}),
        )
        result = filter_by_tags(entries, tag_filter)
        # "Login Test" has smoke+auth -> excluded by auth
        # "Search Test" has search+smoke -> included
        assert result == [2]

    def test_exclude_with_datadriver_false(self) -> None:
        entries = _make_entries()
        tag_filter = TagFilter(
            exclude_tags=frozenset({"smoke"}),
            include_datadriver=False,
        )
        result = filter_by_tags(entries, tag_filter)
        # Only non-DD, non-smoke: Admin Panel
        assert result == [3]
