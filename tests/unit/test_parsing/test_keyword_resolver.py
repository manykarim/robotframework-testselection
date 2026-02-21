from __future__ import annotations

from dataclasses import dataclass

from TestSelection.parsing.keyword_resolver import KeywordTreeResolver
from TestSelection.shared.types import UserKeywordRef


@dataclass
class FakeBodyItem:
    """Minimal stand-in for a robot keyword body item."""

    name: str
    args: tuple[str, ...]


def _make_kw_map(
    definitions: dict[str, list[FakeBodyItem]],
) -> dict[str, UserKeywordRef]:
    kw_map: dict[str, UserKeywordRef] = {}
    for name, body in definitions.items():
        normalized = name.lower().replace(" ", "_")
        kw_map[normalized] = UserKeywordRef(
            name=name,
            normalized_name=normalized,
            body_items=tuple(body),
        )
    return kw_map


class TestKeywordTreeResolverDepthZero:
    def test_depth_zero_returns_top_level_only(self) -> None:
        kw_map = _make_kw_map(
            {
                "Login As User": [
                    FakeBodyItem(name="Open Browser", args=("http://x",)),
                    FakeBodyItem(name="Input Text", args=("id:user", "admin")),
                ],
            }
        )
        resolver = KeywordTreeResolver(kw_map)
        tree = resolver.resolve("Login As User", ("admin", "pass"), max_depth=0)

        assert tree.keyword_name == "Login As User"
        assert tree.args == ("admin", "pass")
        assert tree.children == ()


class TestKeywordTreeResolverDepthTwo:
    def test_depth_two_resolves_nested_keywords(self) -> None:
        kw_map = _make_kw_map(
            {
                "Login As User": [
                    FakeBodyItem(name="Open Application", args=("http://x",)),
                    FakeBodyItem(name="Input Credentials", args=("admin", "pass")),
                ],
                "Input Credentials": [
                    FakeBodyItem(name="Type Username", args=("admin",)),
                    FakeBodyItem(name="Type Password", args=("pass",)),
                ],
            }
        )
        resolver = KeywordTreeResolver(kw_map)
        tree = resolver.resolve("Login As User", (), max_depth=2)

        assert tree.keyword_name == "Login As User"
        assert len(tree.children) == 2

        input_creds = tree.children[1]
        assert input_creds.keyword_name == "Input Credentials"
        assert len(input_creds.children) == 2
        assert input_creds.children[0].keyword_name == "Type Username"
        assert input_creds.children[1].keyword_name == "Type Password"


class TestKeywordTreeResolverCircularProtection:
    def test_circular_reference_stops_at_max_depth(self) -> None:
        kw_map = _make_kw_map(
            {
                "Keyword A": [
                    FakeBodyItem(name="Keyword B", args=()),
                ],
                "Keyword B": [
                    FakeBodyItem(name="Keyword A", args=()),
                ],
            }
        )
        resolver = KeywordTreeResolver(kw_map)
        tree = resolver.resolve("Keyword A", (), max_depth=3)

        # Depth 0: Keyword A -> resolves children
        # Depth 1: Keyword B -> resolves children
        # Depth 2: Keyword A -> resolves children
        # Depth 3: Keyword B -> max_depth reached, no children
        assert tree.keyword_name == "Keyword A"
        level1 = tree.children[0]
        assert level1.keyword_name == "Keyword B"
        level2 = level1.children[0]
        assert level2.keyword_name == "Keyword A"
        level3 = level2.children[0]
        assert level3.keyword_name == "Keyword B"
        assert level3.children == ()


class TestKeywordTreeResolverUnknownKeyword:
    def test_unknown_keyword_returns_leaf_node(self) -> None:
        kw_map = _make_kw_map({})
        resolver = KeywordTreeResolver(kw_map)
        tree = resolver.resolve("Click Element", ("id:btn",), max_depth=5)

        assert tree.keyword_name == "Click Element"
        assert tree.args == ("id:btn",)
        assert tree.children == ()
