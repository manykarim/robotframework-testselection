"""Stage 1 orchestrator for pytest: collect, build text, embed, and store."""
from __future__ import annotations

import logging
from pathlib import Path

from TestSelection.pipeline.artifacts import ArtifactManager
from TestSelection.pipeline.cache import CacheInvalidator
from TestSelection.pipeline.errors import VectorizationError

logger = logging.getLogger(__name__)


def run_vectorize_pytest(
    suite_path: Path,
    artifact_dir: Path,
    model_name: str = "all-MiniLM-L6-v2",
    force: bool = False,
) -> bool:
    """Run the vectorization stage for pytest test suites.

    Returns True if indexing was performed, False if skipped (cache hit).
    Raises VectorizationError on failure.
    """
    try:
        manager = ArtifactManager(artifact_dir)
        cache = CacheInvalidator(manager.hash_store_path)

        if (
            not force
            and not cache.has_changes(suite_path, glob_patterns=("**/*.py",))
            and manager.has_embedding_artifacts()
        ):
            logger.info(
                "[DIVERSE-SELECT] stage=vectorize framework=pytest "
                "event=skipped reason=no_changes"
            )
            return False

        # Collect and extract pytest test metadata
        from TestSelection.pytest.collector import collect_and_extract
        from TestSelection.pytest.text_builder import build_combined_text

        records = collect_and_extract(suite_path)
        logger.info(
            "[DIVERSE-SELECT] stage=vectorize framework=pytest "
            "event=collect_complete tests_found=%d",
            len(records),
        )

        if not records:
            raise VectorizationError("No pytest test cases found to vectorize")

        # Build text representations
        texts = [build_combined_text(r) for r in records]

        # Encode via embedding model
        try:
            from TestSelection.embedding.embedder import (
                SentenceTransformerAdapter,
            )
        except ImportError as exc:
            raise VectorizationError(
                "sentence-transformers is required for vectorization. "
                "Install with: pip install robotframework-testselection[vectorize]"
            ) from exc

        model = SentenceTransformerAdapter(model_name)
        vectors = model.encode(texts)

        logger.info(
            "[DIVERSE-SELECT] stage=vectorize framework=pytest "
            "event=embed_complete model=%s dim=%d tests=%d",
            model.model_name,
            model.embedding_dim,
            len(texts),
        )

        # Build manifest entries and save artifacts
        from TestSelection.embedding.models import (
            EmbeddingMatrix,
            ManifestEntry,
        )
        from TestSelection.shared.types import TestCaseId

        test_ids: list[str] = []
        manifest_entries: list[ManifestEntry] = []

        for record in records:
            tid = TestCaseId.from_source_and_name(
                record.module_path, record.originalname,
            )
            test_ids.append(tid.value)
            manifest_entries.append(
                ManifestEntry(
                    id=tid.value,
                    name=record.nodeid,
                    tags=record.markers,
                    suite=record.module_path,
                    suite_name=record.module_name,
                    is_datadriver=False,
                )
            )

        matrix = EmbeddingMatrix(
            model_name=model.model_name,
            embedding_dim=model.embedding_dim,
            vectors=vectors,
            test_ids=tuple(test_ids),
        )
        matrix.to_artifact(artifact_dir, tuple(manifest_entries), 0)
        cache.save_hashes(suite_path, glob_patterns=("**/*.py",))

        logger.info(
            "[DIVERSE-SELECT] stage=vectorize framework=pytest "
            "event=complete tests_indexed=%d",
            len(records),
        )
        return True

    except VectorizationError:
        raise
    except Exception as exc:
        logger.warning(
            "[DIVERSE-SELECT] stage=vectorize framework=pytest "
            "event=error error=%s",
            str(exc),
        )
        raise VectorizationError(str(exc)) from exc
