from __future__ import annotations

from dataclasses import dataclass

from TestSelection.parsing.keyword_resolver import KeywordTreeResolver
from TestSelection.parsing.text_builder import TextRepresentationBuilder
from TestSelection.shared.config import TextBuilderConfig
from TestSelection.shared.types import Tag, UserKeywordRef


@dataclass
class FakeBodyItem:
    """Minimal stand-in for a robot keyword body item."""

    name: str
    args: tuple[str, ...]


def _make_builder(
    resolve_depth: int = 0,
    kw_defs: dict[str, list[FakeBodyItem]] | None = None,
) -> TextRepresentationBuilder:
    kw_map: dict[str, UserKeywordRef] = {}
    if kw_defs:
        for name, body in kw_defs.items():
            normalized = name.lower().replace(" ", "_")
            kw_map[normalized] = UserKeywordRef(
                name=name,
                normalized_name=normalized,
                body_items=tuple(body),
            )
    resolver = KeywordTreeResolver(kw_map)
    config = TextBuilderConfig(resolve_depth=resolve_depth)
    return TextRepresentationBuilder(resolver, config)


class TestNoiseFiltering:
    def test_variable_args_are_filtered(self) -> None:
        builder = _make_builder()
        body = [
            FakeBodyItem(name="Input Text", args=("${USERNAME}", "admin")),
        ]
        result = builder.build("My Test", frozenset(), body)
        assert "${USERNAME}" not in result.text
        assert "admin" in result.text

    def test_id_locator_args_are_filtered(self) -> None:
        builder = _make_builder()
        body = [
            FakeBodyItem(name="Click Element", args=("id:submit-btn",)),
        ]
        result = builder.build("My Test", frozenset(), body)
        assert "id:submit-btn" not in result.text
        assert "Click Element" in result.text

    def test_xpath_args_are_filtered(self) -> None:
        builder = _make_builder()
        body = [
            FakeBodyItem(name="Click Element", args=("xpath://div[@class='x']",)),
            FakeBodyItem(name="Wait For", args=("//span[@id='y']",)),
        ]
        result = builder.build("My Test", frozenset(), body)
        assert "xpath:" not in result.text
        assert "//" not in result.text

    def test_list_dict_env_vars_are_filtered(self) -> None:
        builder = _make_builder()
        body = [
            FakeBodyItem(name="Log", args=("@{ITEMS}",)),
            FakeBodyItem(name="Log", args=("%{HOME}",)),
            FakeBodyItem(name="Log", args=("&{CONFIG}",)),
        ]
        result = builder.build("My Test", frozenset(), body)
        assert "@{ITEMS}" not in result.text
        assert "%{HOME}" not in result.text
        assert "&{CONFIG}" not in result.text


class TestNameAndTagInclusion:
    def test_test_name_is_included(self) -> None:
        builder = _make_builder()
        result = builder.build("Login With Valid Creds", frozenset(), [])
        assert "Test: Login With Valid Creds." in result.text

    def test_tags_are_included(self) -> None:
        builder = _make_builder()
        tags = frozenset({Tag(value="smoke"), Tag(value="regression")})
        result = builder.build("My Test", tags, [])
        assert "Tags:" in result.text
        assert "smoke" in result.text
        assert "regression" in result.text


class TestKeywordNameFormatting:
    def test_underscores_replaced_with_spaces(self) -> None:
        builder = _make_builder()
        body = [
            FakeBodyItem(name="Open_Browser_To_Page", args=()),
        ]
        result = builder.build("My Test", frozenset(), body)
        assert "Open Browser To Page" in result.text
        assert "Open_Browser_To_Page" not in result.text


class TestEmptyBody:
    def test_empty_body_produces_name_and_tags_only(self) -> None:
        builder = _make_builder()
        tags = frozenset({Tag(value="api")})
        result = builder.build("Empty Test", tags, [])
        assert result.text == "Test: Empty Test. Tags: api."

    def test_empty_body_no_tags(self) -> None:
        builder = _make_builder()
        result = builder.build("Empty Test", frozenset(), [])
        assert result.text == "Test: Empty Test."
