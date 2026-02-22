# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-22

### Added

- Robot CLI passthrough via `--` separator — all arguments after `--` are forwarded directly to `robot` (variables, tag filters, log levels, listeners, metadata, etc.)
- GitHub Actions CI workflow with ruff lint, mypy type checking, pytest matrix (Python 3.10–3.13), coverage gating, and build verification
- 10 new CLI unit tests covering passthrough splitting, parser integration, and end-to-end arg handling

### Changed

- License changed from MIT to Apache 2.0
- Improved type annotations in `selection/registry.py` and `selection/filtering.py` for mypy compatibility
- Added `# type: ignore[attr-defined]` for `robot.run_cli()` calls (Robot Framework lacks type stubs)

### Fixed

- Ruff lint violations: long lines (E501), unused variables (F841), unused imports (F401), import ordering (I001)
- Mypy type errors: missing stubs for optional dependencies, incompatible argument types in filtering, unresolved `.name` attribute in registry

## [0.1.0] - 2025-02-21

### Added

- 3-stage pipeline: vectorize, select, execute
- Sentence-transformers embedding via `all-MiniLM-L6-v2`
- Farthest Point Sampling (FPS) and multi-start FPS selection strategies
- Optional k-Medoids, DPP, and Facility Location strategies
- Robot Framework PreRunModifier for standard test filtering
- Robot Framework Listener v3 for DataDriver test filtering
- Cache invalidation via content hashing
- Tag-based pre-filtering
- CLI with `testcase-select` entry point
- Graceful degradation to full suite on pipeline failure
