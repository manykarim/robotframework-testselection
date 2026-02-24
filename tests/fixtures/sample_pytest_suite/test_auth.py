"""Authentication tests for the sample pytest suite fixture."""
from __future__ import annotations

import pytest


def test_login_valid() -> None:
    """Verify that a user can log in with valid credentials."""
    _username = "admin"  # noqa: F841
    _password = "secret123"  # noqa: F841
    result = {"authenticated": True, "token": "abc"}
    assert result["authenticated"] is True
    assert "token" in result


def test_login_invalid() -> None:
    """Verify that invalid credentials are rejected."""
    _username = "admin"  # noqa: F841
    _password = "wrong"  # noqa: F841
    result = {"authenticated": False, "error": "bad credentials"}
    assert result["authenticated"] is False
    assert "error" in result


def test_logout() -> None:
    _token = "abc"  # noqa: F841
    revoked = True
    assert revoked is True


@pytest.mark.slow
def test_session_timeout() -> None:
    """Verify that sessions expire after the configured timeout."""
    session_start = 1000
    timeout = 5
    current_time = session_start + timeout + 1
    assert current_time > session_start + timeout
