from __future__ import annotations

from TestSelection.parsing.keyword_resolver import KeywordTreeResolver
from TestSelection.shared.config import TextBuilderConfig
from TestSelection.shared.types import (
    Tag,
    TextRepresentation,
    UserKeywordRef,
)


class TextRepresentationBuilder:
    """Builds embeddable text from a test case and its keyword tree.

    Includes test name, tags, keyword names, and semantic arguments.
    Filters DOM locators, variable placeholders, and XPaths -- these
    are noise that dilutes embedding quality.
    """

    def __init__(
        self,
        resolver: KeywordTreeResolver,
        config: TextBuilderConfig | None = None,
    ) -> None:
        self._resolver = resolver
        self._config = config or TextBuilderConfig()

    def build(
        self,
        test_name: str,
        tags: frozenset[Tag],
        body_items: list,
    ) -> TextRepresentation:
        parts = [f"Test: {test_name}."]
        if self._config.include_tags and tags:
            sorted_tags = sorted(tags, key=lambda t: t.normalized)
            parts.append(
                f"Tags: {', '.join(t.value for t in sorted_tags)}."
            )
        for item in body_items:
            if hasattr(item, "name") and item.name:
                if self._config.resolve_depth > 0:
                    tree = self._resolver.resolve(
                        item.name,
                        tuple(item.args),
                        max_depth=self._config.resolve_depth,
                    )
                    parts.append(tree.flatten())
                else:
                    kw_text = item.name.replace("_", " ")
                    semantic_args = [
                        str(a)
                        for a in item.args
                        if not any(
                            str(a).startswith(p)
                            for p in self._config.noise_prefixes
                        )
                    ]
                    if semantic_args:
                        kw_text += f" with {', '.join(semantic_args)}"
                    parts.append(kw_text)
        return TextRepresentation(
            text=" ".join(parts),
            resolve_depth=self._config.resolve_depth,
            includes_tags=self._config.include_tags,
        )

    def build_from_record(
        self,
        test_dict: dict,
        keyword_map: dict[str, UserKeywordRef],
    ) -> TextRepresentation:
        """Build a TextRepresentation from a raw test dict and keyword map."""
        tags = frozenset(Tag(value=t) for t in test_dict.get("tags", []))
        return self.build(
            test_name=test_dict["name"],
            tags=tags,
            body_items=test_dict.get("body", []),
        )
