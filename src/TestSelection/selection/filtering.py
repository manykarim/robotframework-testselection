"""Tag-based filtering for test manifest entries."""
from __future__ import annotations

from typing import TYPE_CHECKING

from TestSelection.selection.strategy import TagFilter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from TestSelection.selection.strategy import TagFilter


class _ManifestEntryLike:
    """Structural type hint for manifest entries with tags and is_datadriver."""

    tags: tuple[str, ...]
    is_datadriver: bool


def filter_by_tags(
    manifest_entries: Sequence[_ManifestEntryLike],
    tag_filter: TagFilter,
) -> list[int]:
    """Return indices of manifest entries that match the tag filter."""
    return [
        i
        for i, entry in enumerate(manifest_entries)
        if tag_filter.matches(frozenset(entry.tags), entry.is_datadriver)
    ]
