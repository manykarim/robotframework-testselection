# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
