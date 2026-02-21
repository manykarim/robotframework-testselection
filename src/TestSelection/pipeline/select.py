"""Stage 2 orchestrator: load artifacts, filter, select, and output."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from TestSelection.pipeline.artifacts import ArtifactManager
from TestSelection.pipeline.errors import SelectionError
from TestSelection.selection.filtering import filter_by_tags
from TestSelection.selection.registry import default_registry
from TestSelection.selection.strategy import (
    DiversityMetrics,
    SelectedTest,
    SelectionResult,
    TagFilter,
)

logger = logging.getLogger(__name__)


def run_select(
    artifact_dir: Path,
    k: int,
    strategy: str = "fps",
    output_file: Path | None = None,
    include_tags: list[str] | None = None,
    exclude_tags: list[str] | None = None,
    seed: int = 42,
    include_datadriver: bool = True,
) -> SelectionResult:
    """Run the selection stage.

    Returns SelectionResult. Raises SelectionError on failure.
    """
    try:
        manager = ArtifactManager(artifact_dir)

        # Validate artifacts
        valid, message = manager.validate_artifacts()
        if not valid:
            raise SelectionError(f"Artifact validation failed: {message}")

        manifest = manager.load_manifest()
        vectors = manager.load_vectors()

        logger.info(
            "[DIVERSE-SELECT] stage=select event=artifacts_loaded "
            "tests=%d dim=%d",
            manifest.test_count,
            manifest.embedding_dim,
        )

        # Build tag filter
        tag_filter = TagFilter(
            include_tags=frozenset(
                t.lower() for t in (include_tags or [])
            ),
            exclude_tags=frozenset(
                t.lower() for t in (exclude_tags or [])
            ),
            include_datadriver=include_datadriver,
        )

        # Filter entries
        filtered_indices = filter_by_tags(manifest.tests, tag_filter)
        filtered_count = len(filtered_indices)

        if filtered_count == 0:
            raise SelectionError(
                "No tests remain after tag filtering"
            )

        filtered_vectors = vectors[filtered_indices]
        actual_k = min(k, filtered_count)

        logger.info(
            "[DIVERSE-SELECT] stage=select event=filter_complete "
            "total=%d filtered=%d k=%d",
            manifest.test_count,
            filtered_count,
            actual_k,
        )

        # Get strategy and run selection
        algo = default_registry.get(strategy)
        selected_indices = algo.select(filtered_vectors, actual_k, seed=seed)

        # Map back to manifest entries
        selected_entries = []
        for idx in selected_indices:
            original_idx = filtered_indices[idx]
            entry = manifest.tests[original_idx]
            selected_entries.append(
                SelectedTest(
                    name=entry.name,
                    id=entry.id,
                    suite=entry.suite,
                    is_datadriver=entry.is_datadriver,
                )
            )

        # Compute diversity metrics
        selected_vectors = filtered_vectors[selected_indices]
        metrics = _compute_diversity_metrics(
            selected_vectors, selected_entries, manifest
        )

        result = SelectionResult(
            strategy=strategy,
            k=actual_k,
            seed=seed,
            total_tests=manifest.test_count,
            filtered_tests=filtered_count,
            selected=tuple(selected_entries),
            diversity_metrics=metrics,
        )

        # Write output
        out_path = output_file or manager.selection_path
        result.to_json(out_path)

        logger.info(
            "[DIVERSE-SELECT] stage=select event=complete "
            "strategy=%s k=%d avg_dist=%.4f min_dist=%.4f "
            "suite_coverage=%d/%d",
            strategy,
            actual_k,
            metrics.avg_pairwise_distance,
            metrics.min_pairwise_distance,
            metrics.suite_coverage,
            metrics.suite_total,
        )

        return result

    except SelectionError:
        raise
    except KeyError as exc:
        logger.warning(
            "[DIVERSE-SELECT] stage=select event=error error=%s",
            str(exc),
        )
        raise SelectionError(str(exc)) from exc
    except Exception as exc:
        logger.warning(
            "[DIVERSE-SELECT] stage=select event=error error=%s",
            str(exc),
        )
        raise SelectionError(str(exc)) from exc


def _compute_diversity_metrics(
    selected_vectors: np.ndarray,
    selected_entries: list[SelectedTest],
    manifest,
) -> DiversityMetrics:
    """Compute pairwise cosine distance metrics for selected tests."""
    from sklearn.metrics.pairwise import cosine_distances

    n = selected_vectors.shape[0]
    if n < 2:
        return DiversityMetrics(
            avg_pairwise_distance=0.0,
            min_pairwise_distance=0.0,
            suite_coverage=len({e.suite for e in selected_entries}),
            suite_total=len({e.suite for e in manifest.tests}),
        )

    pairwise = cosine_distances(selected_vectors, selected_vectors)
    mask = np.triu(np.ones_like(pairwise, dtype=bool), k=1)
    upper_dists = pairwise[mask]

    all_suites = {e.suite for e in manifest.tests}
    selected_suites = {e.suite for e in selected_entries}

    return DiversityMetrics(
        avg_pairwise_distance=float(np.mean(upper_dists)),
        min_pairwise_distance=float(np.min(upper_dists)),
        suite_coverage=len(selected_suites),
        suite_total=len(all_suites),
    )
