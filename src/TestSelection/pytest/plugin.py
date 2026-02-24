"""pytest plugin for diversity-based test selection.

Registered as a ``pytest11`` entry point. Activated via CLI options:

    pytest --diverse-k=20 --diverse-strategy=fps tests/

When ``--diverse-k`` is 0 (default), the plugin is inactive and imposes
zero overhead.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("TestSelection.pytest")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register CLI options for diverse test selection."""
    group = parser.getgroup("diverse", "Diversity-based test selection")
    group.addoption(
        "--diverse-k",
        type=int,
        default=0,
        help="Select k most diverse tests. 0 disables selection (default: 0).",
    )
    group.addoption(
        "--diverse-strategy",
        default="fps",
        help="Selection algorithm: fps, fps_multi, kmedoids, "
        "dpp, facility (default: fps).",
    )
    group.addoption(
        "--diverse-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    group.addoption(
        "--diverse-cache-dir",
        default=".diverse-cache",
        help="Directory for embedding cache (default: .diverse-cache).",
    )
    group.addoption(
        "--diverse-model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name (default: all-MiniLM-L6-v2).",
    )
    group.addoption(
        "--diverse-include-markers",
        nargs="*",
        default=None,
        help="Only include tests with these markers before selection.",
    )
    group.addoption(
        "--diverse-exclude-markers",
        nargs="*",
        default=None,
        help="Exclude tests with these markers before selection.",
    )
    group.addoption(
        "--diverse-group-parametrize",
        action="store_true",
        default=False,
        help="Group parametrized tests and select at group level.",
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item],
) -> None:
    """Select the most diverse subset of collected tests.

    Runs after all other collection hooks (trylast=True) so we operate
    on the final filtered set.
    """
    k = config.getoption("--diverse-k", default=0)
    if not k or k <= 0:
        return  # Plugin disabled

    if len(items) <= k:
        logger.info(
            "[DIVERSE] Suite has %d tests <= k=%d, running all.",
            len(items), k,
        )
        return

    strategy_name = config.getoption("--diverse-strategy", default="fps")
    seed = config.getoption("--diverse-seed", default=42)
    model_name = config.getoption("--diverse-model", default="all-MiniLM-L6-v2")
    group_parametrize = config.getoption("--diverse-group-parametrize", default=False)

    try:
        indices = _select_diverse(
            items=items,
            k=k,
            strategy_name=strategy_name,
            seed=seed,
            model_name=model_name,
            config=config,
            group_parametrize=group_parametrize,
        )
    except Exception as exc:
        logger.warning(
            "[DIVERSE] Selection failed, running all tests: %s", exc,
        )
        return

    # Deselect non-selected items
    selected_set = set(indices)
    deselected = [item for i, item in enumerate(items) if i not in selected_set]
    config.hook.pytest_deselected(items=deselected)
    items[:] = [items[i] for i in sorted(indices)]

    logger.info(
        "[DIVERSE] Selected %d/%d tests (strategy=%s, seed=%d).",
        len(items), len(items) + len(deselected), strategy_name, seed,
    )


def _select_diverse(
    items: list[pytest.Item],
    k: int,
    strategy_name: str,
    seed: int,
    model_name: str,
    config: pytest.Config,
    group_parametrize: bool,
) -> list[int]:
    """Run the diversity selection pipeline and return selected indices."""
    from TestSelection.pytest.text_builder import build_combined_text_from_item

    if group_parametrize:
        return _select_grouped(
            items, k, strategy_name, seed, model_name, config,
        )

    # 1. Build text representations
    texts = [build_combined_text_from_item(item) for item in items]

    # 2. Embed
    vectors = _embed_texts(texts, model_name, config)

    # 3. Select
    from TestSelection.selection.registry import default_registry

    strategy = default_registry.get(strategy_name)
    return strategy.select(vectors, k, seed=seed)


def _select_grouped(
    items: list[pytest.Item],
    k: int,
    strategy_name: str,
    seed: int,
    model_name: str,
    config: pytest.Config,
) -> list[int]:
    """Select at the group level for parametrized tests."""
    from TestSelection.pytest.text_builder import build_combined_text_from_item

    # Group items by originalname
    groups: dict[str, list[int]] = {}
    for i, item in enumerate(items):
        key = getattr(item, "originalname", item.name)
        groups.setdefault(key, []).append(i)

    # Build one text per group (use first item as representative)
    group_keys = list(groups.keys())
    group_texts = []
    for key in group_keys:
        representative_idx = groups[key][0]
        group_texts.append(build_combined_text_from_item(items[representative_idx]))

    # Embed and select groups
    vectors = _embed_texts(group_texts, model_name, config)

    from TestSelection.selection.registry import default_registry

    strategy = default_registry.get(strategy_name)
    actual_k = min(k, len(group_keys))
    selected_group_indices = strategy.select(vectors, actual_k, seed=seed)

    # Expand selected groups back to individual items
    selected_item_indices: list[int] = []
    for gi in selected_group_indices:
        selected_item_indices.extend(groups[group_keys[gi]])

    return selected_item_indices


def _embed_texts(
    texts: list[str],
    model_name: str,
    config: pytest.Config,
) -> np.ndarray:
    """Embed texts, using cache if available."""
    import numpy as np

    cache_dir = Path(config.getoption("--diverse-cache-dir", default=".diverse-cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Compute hash of all texts for cache key
    content_hash = hashlib.md5(
        "\n".join(texts).encode(), usedforsecurity=False,
    ).hexdigest()
    cache_file = cache_dir / f"embeddings_{model_name}_{content_hash}.npz"

    if cache_file.exists():
        logger.info("[DIVERSE] Loading cached embeddings from %s", cache_file)
        data = np.load(cache_file)
        return data["vectors"]

    # Embed fresh
    try:
        from TestSelection.embedding.embedder import SentenceTransformerAdapter
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for diverse selection. "
            "Install with: pip install robotframework-testselection[vectorize]"
        ) from exc

    model = SentenceTransformerAdapter(model_name)
    vectors = model.encode(texts)

    # Save to cache
    np.savez_compressed(cache_file, vectors=vectors)
    logger.info("[DIVERSE] Cached embeddings to %s", cache_file)

    return vectors
