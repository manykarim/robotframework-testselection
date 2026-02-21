"""Tag-based filtering for test manifest entries."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from TestSelection.selection.strategy import TagFilter

if TYPE_CHECKING:
    from collections.abc import Sequence


def filter_by_tags(
    manifest_entries: Sequence[Any],
    tag_filter: TagFilter,
) -> list[int]:
    """Return indices of manifest entries that match the tag filter."""
    return [
        i
        for i, entry in enumerate(manifest_entries)
        if tag_filter.matches(frozenset(entry.tags), entry.is_datadriver)
    ]
