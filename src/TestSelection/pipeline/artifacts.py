"""Artifact management for the pipeline stages."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from TestSelection.embedding.models import ArtifactManifest, ManifestEntry
from TestSelection.pipeline.errors import ArtifactError

logger = logging.getLogger(__name__)


class ArtifactManager:
    """Manages artifact storage and retrieval between pipeline stages."""

    def __init__(self, artifact_dir: Path) -> None:
        self._artifact_dir = artifact_dir
        self._artifact_dir.mkdir(parents=True, exist_ok=True)

    @property
    def embeddings_path(self) -> Path:
        return self._artifact_dir / "embeddings.npz"

    @property
    def manifest_path(self) -> Path:
        return self._artifact_dir / "test_manifest.json"

    @property
    def selection_path(self) -> Path:
        return self._artifact_dir / "selected_tests.json"

    @property
    def hash_store_path(self) -> Path:
        return self._artifact_dir / "file_hashes.json"

    def has_embedding_artifacts(self) -> bool:
        """Check if both embedding artifacts exist."""
        return self.embeddings_path.exists() and self.manifest_path.exists()

    def has_selection_artifact(self) -> bool:
        """Check if the selection artifact exists."""
        return self.selection_path.exists()

    def load_manifest(self) -> ArtifactManifest:
        """Load and parse test_manifest.json into ArtifactManifest."""
        if not self.manifest_path.exists():
            raise ArtifactError(
                f"Manifest not found: {self.manifest_path}"
            )
        raw = json.loads(self.manifest_path.read_text())
        return ArtifactManifest(
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

    def load_vectors(self) -> NDArray:
        """Load embedding vectors from embeddings.npz."""
        if not self.embeddings_path.exists():
            raise ArtifactError(
                f"Embeddings not found: {self.embeddings_path}"
            )
        data = np.load(self.embeddings_path, allow_pickle=True)
        return data["vectors"]

    def validate_artifacts(self) -> tuple[bool, str]:
        """Validate that embedding artifacts are consistent.

        Returns (valid, message) where message describes any issues.
        """
        if not self.has_embedding_artifacts():
            return False, "Missing embedding artifacts"

        try:
            manifest = self.load_manifest()
            vectors = self.load_vectors()
        except Exception as exc:
            return False, f"Failed to load artifacts: {exc}"

        if vectors.shape[0] != manifest.test_count:
            return (
                False,
                f"Shape mismatch: vectors has {vectors.shape[0]} rows "
                f"but manifest has {manifest.test_count} tests",
            )

        if vectors.shape[1] != manifest.embedding_dim:
            return (
                False,
                f"Dimension mismatch: vectors has {vectors.shape[1]} dims "
                f"but manifest expects {manifest.embedding_dim}",
            )

        return True, "Artifacts valid"
