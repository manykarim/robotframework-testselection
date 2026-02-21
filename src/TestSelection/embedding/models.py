"""Core domain objects for the Embedding bounded context."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Embedding:
    """A single test case embedding vector (value object)."""

    vector: NDArray[np.float32]

    @property
    def dimensionality(self) -> int:
        return int(self.vector.shape[0])

    @property
    def is_normalized(self) -> bool:
        norm = float(np.linalg.norm(self.vector))
        return abs(norm - 1.0) < 1e-6


@dataclass(frozen=True)
class ManifestEntry:
    """Metadata for a single test in the artifact manifest."""

    id: str
    name: str
    tags: tuple[str, ...]
    suite: str
    suite_name: str
    is_datadriver: bool


@dataclass(frozen=True)
class ArtifactManifest:
    """Describes the contents of an EmbeddingArtifact."""

    model: str
    embedding_dim: int
    test_count: int
    resolve_depth: int
    tests: tuple[ManifestEntry, ...]


@dataclass(frozen=True)
class EmbeddingArtifact:
    """Portable artifact produced by the Embedding Context.

    Consists of two files:
    - embeddings.npz: numpy archive with 'vectors' (N x dim) and 'ids' (N,)
    - test_manifest.json: human-readable metadata
    """

    embeddings_path: Path
    manifest_path: Path
    model_name: str
    embedding_dim: int
    test_count: int


class EmbeddingMatrix:
    """Aggregate root: holds all embeddings for a test suite.

    Invariants:
    - vectors.shape[0] == len(test_ids)  (one vector per test)
    - vectors.shape[1] == embedding_dim  (consistent dimensionality)
    - All vectors are L2-normalized for cosine distance computation
    """

    def __init__(
        self,
        model_name: str,
        embedding_dim: int,
        vectors: NDArray[np.float32],
        test_ids: tuple[str, ...],
    ) -> None:
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.vectors = vectors
        self.test_ids = test_ids
        self.validate_dimensions()

    @property
    def test_count(self) -> int:
        return len(self.test_ids)

    def validate_dimensions(self) -> None:
        """Assert that vectors shape matches test_ids length and embedding_dim."""
        if self.vectors.shape[0] != len(self.test_ids):
            raise ValueError(
                f"Vector count {self.vectors.shape[0]} != "
                f"test ID count {len(self.test_ids)}"
            )
        if self.vectors.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Vector dim {self.vectors.shape[1]} != "
                f"expected dim {self.embedding_dim}"
            )

    def to_artifact(
        self,
        output_dir: Path,
        manifest_entries: tuple[ManifestEntry, ...],
        resolve_depth: int = 0,
    ) -> EmbeddingArtifact:
        """Serialize to portable artifact files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        emb_path = output_dir / "embeddings.npz"
        manifest_path = output_dir / "test_manifest.json"

        np.savez_compressed(
            emb_path,
            vectors=self.vectors,
            ids=np.array(list(self.test_ids)),
        )

        manifest = ArtifactManifest(
            model=self.model_name,
            embedding_dim=self.embedding_dim,
            test_count=self.test_count,
            resolve_depth=resolve_depth,
            tests=manifest_entries,
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "model": manifest.model,
                    "embedding_dim": manifest.embedding_dim,
                    "test_count": manifest.test_count,
                    "resolve_depth": manifest.resolve_depth,
                    "tests": [
                        {
                            "id": e.id,
                            "name": e.name,
                            "tags": list(e.tags),
                            "suite": e.suite,
                            "suite_name": e.suite_name,
                            "is_datadriver": e.is_datadriver,
                        }
                        for e in manifest.tests
                    ],
                },
                indent=2,
            )
        )

        return EmbeddingArtifact(
            embeddings_path=emb_path,
            manifest_path=manifest_path,
            model_name=self.model_name,
            embedding_dim=self.embedding_dim,
            test_count=self.test_count,
        )

    @classmethod
    def from_artifact(
        cls, artifact_dir: Path
    ) -> tuple[EmbeddingMatrix, ArtifactManifest]:
        """Load an EmbeddingMatrix and ArtifactManifest from artifact files."""
        emb_path = artifact_dir / "embeddings.npz"
        manifest_path = artifact_dir / "test_manifest.json"

        data = np.load(emb_path, allow_pickle=True)
        vectors: NDArray[np.float32] = data["vectors"]
        ids: tuple[str, ...] = tuple(str(i) for i in data["ids"])

        raw = json.loads(manifest_path.read_text())
        manifest = ArtifactManifest(
            model=raw["model"],
            embedding_dim=raw["embedding_dim"],
            test_count=raw["test_count"],
            resolve_depth=raw.get("resolve_depth", 0),
            tests=tuple(
                ManifestEntry(
                    id=t["id"],
                    name=t["name"],
                    tags=tuple(t.get("tags", [])),
                    suite=t.get("suite", ""),
                    suite_name=t.get("suite_name", ""),
                    is_datadriver=t.get("is_datadriver", False),
                )
                for t in raw["tests"]
            ),
        )

        matrix = cls(
            model_name=raw["model"],
            embedding_dim=raw["embedding_dim"],
            vectors=vectors,
            test_ids=ids,
        )
        return matrix, manifest
