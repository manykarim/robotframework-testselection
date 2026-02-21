from __future__ import annotations

from TestSelection.shared.types import KeywordTree, UserKeywordRef


class KeywordTreeResolver:
    """Resolves keyword names to their full sub-keyword trees.

    Wraps the manual resolution logic required because robot.api
    does not resolve keyword implementations at parse time.
    """

    def __init__(self, keyword_map: dict[str, UserKeywordRef]) -> None:
        self._kw_map = keyword_map

    def resolve(
        self,
        kw_name: str,
        kw_args: tuple[str, ...],
        max_depth: int = 10,
    ) -> KeywordTree:
        return self._resolve_recursive(
            kw_name, kw_args, depth=0, max_depth=max_depth
        )

    def _resolve_recursive(
        self,
        kw_name: str,
        kw_args: tuple[str, ...],
        depth: int,
        max_depth: int,
    ) -> KeywordTree:
        children: list[KeywordTree] = []
        if depth < max_depth:
            normalized = kw_name.lower().replace(" ", "_")
            uk = self._kw_map.get(normalized)
            if uk is not None:
                for item in uk.body_items:
                    if hasattr(item, "name") and item.name:
                        child = self._resolve_recursive(
                            item.name,
                            tuple(item.args),
                            depth + 1,
                            max_depth,
                        )
                        children.append(child)
        return KeywordTree(
            keyword_name=kw_name,
            args=kw_args,
            children=tuple(children),
        )
