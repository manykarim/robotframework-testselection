from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from TestSelection.shared.types import NOISE_PREFIXES


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for a complete pipeline run."""

    model_name: str = "all-MiniLM-L6-v2"
    resolve_depth: int = 0
    k: int = 50
    strategy: str = "fps"
    seed: int = 42
    output_dir: Path = Path("./results")
    force_reindex: bool = False


@dataclass(frozen=True)
class TextBuilderConfig:
    """Configuration for TextRepresentationBuilder."""

    resolve_depth: int = 0
    include_tags: bool = True
    include_suite_name: bool = False
    noise_prefixes: frozenset[str] = field(
        default_factory=lambda: frozenset(NOISE_PREFIXES)
    )
