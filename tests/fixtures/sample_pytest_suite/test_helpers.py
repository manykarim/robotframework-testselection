"""Utility / helper function tests for the sample pytest suite fixture."""
from __future__ import annotations

import pytest


def test_format_name() -> None:
    """Format a first and last name into a display name."""
    first, last = "jane", "doe"
    display = f"{first.title()} {last.title()}"
    assert display == "Jane Doe"


def test_validate_email() -> None:
    valid = "user@example.com"
    invalid = "not-an-email"
    assert "@" in valid
    assert "@" not in invalid


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("key=value", {"key": "value"}),
        ("a=1", {"a": "1"}),
        ("x=hello world", {"x": "hello world"}),
    ],
)
def test_parse_config(raw: str, expected: dict) -> None:
    key, value = raw.split("=", maxsplit=1)
    assert {key: value} == expected
