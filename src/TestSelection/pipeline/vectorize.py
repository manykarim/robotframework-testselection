"""Stage 1 orchestrator: parse, embed, and store test vectors."""
from __future__ import annotations

import logging
from pathlib import Path

from TestSelection.pipeline.artifacts import ArtifactManager
from TestSelection.pipeline.cache import CacheInvalidator
from TestSelection.pipeline.errors import VectorizationError

logger = logging.getLogger(__name__)


def run_vectorize(
    suite_path: Path,
    artifact_dir: Path,
    model_name: str = "all-MiniLM-L6-v2",
    resolve_depth: int = 0,
    force: bool = False,
    datadriver_csvs: list[Path] | None = None,
) -> bool:
    """Run the vectorization stage.

    Returns True if indexing was performed, False if skipped (cache hit).
    Raises VectorizationError on failure.
    """
    try:
        manager = ArtifactManager(artifact_dir)
        cache = CacheInvalidator(manager.hash_store_path)

        if (
            not force
            and not cache.has_changes(suite_path)
            and manager.has_embedding_artifacts()
        ):
            logger.info(
                "[DIVERSE-SELECT] stage=vectorize event=skipped "
                "reason=no_changes"
            )
            return False

        # Parse suite
        from TestSelection.parsing.suite_collector import RobotApiAdapter

        adapter = RobotApiAdapter()
        raw_tests, kw_map = adapter.parse_suite(suite_path)
        logger.info(
            "[DIVERSE-SELECT] stage=vectorize event=parse_complete "
            "tests_found=%d",
            len(raw_tests),
        )

        # Build text representations
        from TestSelection.parsing.keyword_resolver import (
            KeywordTreeResolver,
        )
        from TestSelection.parsing.text_builder import (
            TextRepresentationBuilder,
        )
        from TestSelection.shared.config import TextBuilderConfig
        from TestSelection.shared.types import (
            SuitePath,
            Tag,
            TestCaseId,
            TestCaseRecord,
        )

        resolver = KeywordTreeResolver(kw_map)
        config = TextBuilderConfig(resolve_depth=resolve_depth)
        builder = TextRepresentationBuilder(resolver, config)

        records: list[TestCaseRecord] = []
        for test_dict in raw_tests:
            tags = frozenset(Tag(value=t) for t in test_dict.get("tags", []))
            text_rep = builder.build(
                test_name=test_dict["name"],
                tags=tags,
                body_items=test_dict.get("body", []),
            )
            test_id = TestCaseId.from_source_and_name(
                test_dict["source"], test_dict["name"]
            )
            records.append(
                TestCaseRecord(
                    test_id=test_id,
                    name=test_dict["name"],
                    tags=tags,
                    suite_source=SuitePath(Path(test_dict["source"])),
                    suite_name=test_dict.get("suite_name", ""),
                    text_representation=text_rep,
                )
            )

        # Handle DataDriver CSVs
        if datadriver_csvs:
            from TestSelection.parsing.datadriver_reader import (
                read_datadriver_csv,
            )
            from TestSelection.shared.types import TextRepresentation

            for csv_path in datadriver_csvs:
                dd_tests = read_datadriver_csv(csv_path, template_name="Template")
                for dd in dd_tests:
                    test_id = TestCaseId.from_source_and_name(
                        dd["source"], dd["name"]
                    )
                    records.append(
                        TestCaseRecord(
                            test_id=test_id,
                            name=dd["name"],
                            tags=frozenset(),
                            suite_source=SuitePath(Path(dd["source"])),
                            suite_name="DataDriver",
                            text_representation=TextRepresentation(
                                text=dd["description"]
                            ),
                            is_datadriver=True,
                        )
                    )

        if not records:
            raise VectorizationError("No test cases found to vectorize")

        # Encode via embedding model
        try:
            from TestSelection.embedding.embedder import (
                SentenceTransformerAdapter,
            )
        except ImportError as exc:
            raise VectorizationError(
                "sentence-transformers is required for vectorization. "
                "Install with: pip install testcase-selection[vectorize]"
            ) from exc

        model = SentenceTransformerAdapter(model_name)
        texts = [r.text_representation.text for r in records]
        vectors = model.encode(texts)

        logger.info(
            "[DIVERSE-SELECT] stage=vectorize event=embed_complete "
            "model=%s dim=%d tests=%d",
            model.model_name,
            model.embedding_dim,
            len(texts),
        )

        # Build EmbeddingMatrix and save artifacts
        from TestSelection.embedding.models import (
            EmbeddingMatrix,
            ManifestEntry,
        )

        test_ids = tuple(r.test_id.value for r in records)
        matrix = EmbeddingMatrix(
            model_name=model.model_name,
            embedding_dim=model.embedding_dim,
            vectors=vectors,
            test_ids=test_ids,
        )

        manifest_entries = tuple(
            ManifestEntry(
                id=r.test_id.value,
                name=r.name,
                tags=tuple(t.value for t in r.tags),
                suite=str(r.suite_source.value),
                suite_name=r.suite_name,
                is_datadriver=r.is_datadriver,
            )
            for r in records
        )

        matrix.to_artifact(artifact_dir, manifest_entries, resolve_depth)
        cache.save_hashes(suite_path)

        logger.info(
            "[DIVERSE-SELECT] stage=vectorize event=complete "
            "tests_indexed=%d",
            len(records),
        )
        return True

    except VectorizationError:
        raise
    except Exception as exc:
        logger.warning(
            "[DIVERSE-SELECT] stage=vectorize event=error error=%s",
            str(exc),
        )
        raise VectorizationError(str(exc)) from exc
