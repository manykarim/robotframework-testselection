"""API operation tests for the sample pytest suite fixture."""
from __future__ import annotations


def test_get_users(sample_data: dict) -> None:
    """Retrieve all users from the API."""
    users = sample_data["users"]
    assert len(users) == 2
    assert users[0]["name"] == "Alice"


def test_create_user(sample_data: dict) -> None:
    """Create a new user via the API."""
    new_user = {"id": 3, "name": "Charlie", "email": "charlie@example.com"}
    users = list(sample_data["users"])
    users.append(new_user)
    assert len(users) == 3
    assert users[-1]["name"] == "Charlie"


def test_delete_user() -> None:
    users = [{"id": 1}, {"id": 2}]
    users = [u for u in users if u["id"] != 1]
    assert len(users) == 1
    assert users[0]["id"] == 2


def test_search_users(sample_data: dict) -> None:
    users = sample_data["users"]
    results = [u for u in users if "alice" in u["name"].lower()]
    assert len(results) == 1
    assert results[0]["email"] == "alice@example.com"
