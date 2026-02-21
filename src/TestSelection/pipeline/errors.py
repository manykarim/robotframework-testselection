"""Custom exception hierarchy for the diverse test selection pipeline."""
from __future__ import annotations


class DiverseSelectionError(Exception):
    """Base exception for the diverse test selection pipeline."""


class VectorizationError(DiverseSelectionError):
    """Raised when Stage 1 (vectorization) encounters an unrecoverable error."""


class SelectionError(DiverseSelectionError):
    """Raised when Stage 2 (selection) encounters an unrecoverable error."""


class ArtifactError(DiverseSelectionError):
    """Raised when artifact validation fails between stages."""
