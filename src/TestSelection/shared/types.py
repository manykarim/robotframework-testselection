from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TestCaseId:
    """Stable identity for a test case, derived from source file + name."""

    value: str

    @classmethod
    def from_source_and_name(cls, source: str, name: str) -> TestCaseId:
        raw = f"{source}::{name}"
        return cls(value=hashlib.md5(raw.encode()).hexdigest())


@dataclass(frozen=True)
class Tag:
    """A Robot Framework test tag, normalized for comparison."""

    value: str

    @property
    def normalized(self) -> str:
        return self.value.lower().strip()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Tag):
            return self.normalized == other.normalized
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.normalized)


@dataclass(frozen=True)
class SuitePath:
    """Path to a .robot file or test directory."""

    value: Path

    @property
    def is_directory(self) -> bool:
        return self.value.is_dir()


@dataclass(frozen=True)
class FileHash:
    """Content hash of a .robot file for change detection."""

    path: str
    md5: str


@dataclass(frozen=True)
class TextRepresentation:
    """Embeddable text representation of a test case.

    Built from test name, tags, keyword names, and semantic arguments.
    Excludes DOM locators, variable placeholders, and XPaths.
    """

    text: str
    resolve_depth: int = 0
    includes_tags: bool = True
    includes_keyword_args: bool = True


NOISE_PREFIXES = ("id:", "css:", "xpath:", "//", "${", "@{", "%{", "&{")


@dataclass(frozen=True)
class KeywordTree:
    """Resolved keyword call tree for a single keyword invocation."""

    keyword_name: str
    args: tuple[str, ...]
    children: tuple[KeywordTree, ...] = ()

    def flatten(self) -> str:
        """Convert to natural language, filtering noise arguments."""
        kw = self.keyword_name.replace("_", " ")
        semantic_args = [
            a
            for a in self.args
            if not any(a.startswith(p) for p in NOISE_PREFIXES)
        ]
        text = kw
        if semantic_args:
            text += f" with {', '.join(semantic_args)}"
        children_text = " ".join(c.flatten() for c in self.children)
        return f"{text} {children_text}".strip()


@dataclass(frozen=True)
class UserKeywordRef:
    """Domain representation of a user keyword from robot.api."""

    name: str
    normalized_name: str
    body_items: tuple


@dataclass(frozen=True)
class TestCaseRecord:
    """Domain entity representing a parsed test case ready for embedding."""

    test_id: TestCaseId
    name: str
    tags: frozenset[Tag]
    suite_source: SuitePath
    suite_name: str
    text_representation: TextRepresentation
    is_datadriver: bool = False
