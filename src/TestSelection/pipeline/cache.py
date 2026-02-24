"""Caching service for change detection via content hashing."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CacheInvalidator:
    """Detects changes in .robot and .csv files via MD5 content hashing."""

    def __init__(self, hash_store_path: Path) -> None:
        self._hash_store_path = hash_store_path

    def has_changes(
        self,
        suite_path: Path,
        glob_patterns: tuple[str, ...] | None = None,
    ) -> bool:
        """Compare current file hashes with stored hashes.

        Returns True if there are changes or no stored hashes exist.
        If glob_patterns is provided, uses those instead of the default
        .robot/.csv patterns.
        """
        if not self._hash_store_path.exists():
            logger.info(
                "[DIVERSE-SELECT] stage=vectorize event=cache_miss "
                "reason=no_stored_hashes"
            )
            return True

        stored = json.loads(self._hash_store_path.read_text())
        current = self._compute_hashes(suite_path, glob_patterns)

        if current != stored:
            changed = set(current.keys()) ^ set(stored.keys())
            for key in set(current.keys()) & set(stored.keys()):
                if current[key] != stored[key]:
                    changed.add(key)
            logger.info(
                "[DIVERSE-SELECT] stage=vectorize event=cache_miss "
                "changed_files=%d",
                len(changed),
            )
            return True

        logger.info(
            "[DIVERSE-SELECT] stage=vectorize event=cache_hit "
            "files=%d",
            len(current),
        )
        return False

    def save_hashes(
        self,
        suite_path: Path,
        glob_patterns: tuple[str, ...] | None = None,
    ) -> None:
        """Save current file hashes to the hash store."""
        hashes = self._compute_hashes(suite_path, glob_patterns)
        self._hash_store_path.parent.mkdir(parents=True, exist_ok=True)
        self._hash_store_path.write_text(json.dumps(hashes, indent=2))

    def _compute_hashes(
        self,
        suite_path: Path,
        glob_patterns: tuple[str, ...] | None = None,
    ) -> dict[str, str]:
        """Compute MD5 hashes for source files.

        By default hashes .robot and .csv files. Pass glob_patterns
        to override (e.g., ("**/*.py",) for pytest).
        """
        hashes: dict[str, str] = {}
        target = suite_path if suite_path.is_dir() else suite_path.parent

        if suite_path.is_dir():
            patterns = list(glob_patterns) if glob_patterns else ["*.robot", "*.csv"]
            for pattern in patterns:
                for p in sorted(target.rglob(pattern)):
                    md5 = hashlib.md5(p.read_bytes()).hexdigest()
                    hashes[str(p)] = md5
        else:
            if suite_path.exists():
                md5 = hashlib.md5(suite_path.read_bytes()).hexdigest()
                hashes[str(suite_path)] = md5

        return hashes
