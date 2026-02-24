"""Shared fixtures for the sample pytest suite."""
from __future__ import annotations

import pytest


@pytest.fixture()
def sample_data() -> dict[str, object]:
    """Provide sample data shared across test modules."""
    return {
        "users": [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"},
        ],
        "api_base": "https://api.example.com",
        "timeout": 30,
    }
