# Implementation Roadmap: Vector-Based Diverse Test Selection

**Date**: 2026-02-21
**Prerequisite Reading**: All 8 ADRs + Domain Model

This roadmap synthesizes the ADR decisions and domain model into a phased, file-by-file implementation plan with dependency ordering, testing strategy, and performance targets.

---

## Phase 0: Project Scaffolding (Foundation)

**Goal**: Establish the uv-managed Python project with the full DDD package structure. No business logic yet.

**Depends on**: Nothing (entry point)

### Files to Create

| File | Purpose | ADR Reference |
|------|---------|--------------|
| `pyproject.toml` | Project metadata, dependency groups, CLI entry point, tool config | ADR-005 |
| `src/testcase_selection/__init__.py` | Package root, version | ADR-005 |
| `src/testcase_selection/shared/__init__.py` | Shared kernel exports | Domain Model |
| `src/testcase_selection/shared/types.py` | `TestCaseId`, `SuitePath`, `Tag`, `FileHash`, `TextRepresentation`, `KeywordTree` value objects | Domain Model, Parsing Context |
| `src/testcase_selection/shared/config.py` | `PipelineConfig`, `SelectionConfig` dataclasses | Domain Model, Pipeline Context |
| `src/testcase_selection/parsing/__init__.py` | Parsing context exports | ADR-005 |
| `src/testcase_selection/embedding/__init__.py` | Embedding context exports | ADR-005 |
| `src/testcase_selection/selection/__init__.py` | Selection context exports | ADR-005 |
| `src/testcase_selection/execution/__init__.py` | Execution context exports | ADR-005 |
| `src/testcase_selection/pipeline/__init__.py` | Pipeline context exports | ADR-005 |
| `tests/fixtures/sample.robot` | Minimal Robot Framework test file for testing | ADR-005 |
| `tests/fixtures/sample_datadriver.csv` | Sample DataDriver CSV for testing | ADR-005 |

### Commands

```bash
uv init
uv venv
uv sync --extra dev
uv run pytest  # should pass with 0 tests
uv run ruff check src/
uv run mypy src/
```

### Acceptance Criteria
- `uv sync` succeeds
- `uv run pytest` exits 0
- `uv run ruff check src/` exits 0
- `uv run mypy src/` exits 0
- All 5 bounded context packages importable

---

## Phase 1: Shared Kernel + Parsing Context

**Goal**: Implement value objects, the `RobotApiAdapter` ACL, `KeywordTreeResolver`, and `TextRepresentationBuilder`. This is the foundation that all downstream contexts consume.

**Depends on**: Phase 0

### Files to Implement

| File | Key Types/Functions | ADR Reference |
|------|-------------------|--------------|
| `src/testcase_selection/shared/types.py` | `TestCaseId.from_source_and_name()`, `Tag`, `SuitePath`, `FileHash`, `TextRepresentation`, `KeywordTree.flatten()` | Domain Model |
| `src/testcase_selection/shared/config.py` | `PipelineConfig`, `TextBuilderConfig(resolve_depth, include_tags, noise_prefixes)` | ADR-006 |
| `src/testcase_selection/parsing/suite_collector.py` | `RobotApiAdapter.parse_suite()`, `._build_keyword_map()`, `._collect_tests()`, `.compute_file_hashes()` | Domain Model, ADR-004 |
| `src/testcase_selection/parsing/keyword_resolver.py` | `KeywordTreeResolver.resolve()` with configurable `max_depth` | Domain Model, ADR-006 |
| `src/testcase_selection/parsing/text_builder.py` | `TextRepresentationBuilder.build()` -- noise filtering, natural language conversion | ADR-006 |
| `src/testcase_selection/parsing/datadriver_reader.py` | `read_datadriver_csv()` -- pre-reads CSV, generates `TestCaseRecord`-compatible dicts | ADR-004, ADR-006 |

### Key Design Points
- Use `robot.api.TestSuite.from_file_system()` exclusively (ADR-004)
- `KeywordTree` is a frozen dataclass with recursive `flatten()` (Domain Model)
- Noise prefixes: `id:`, `css:`, `xpath:`, `//`, `${`, `@{`, `%{`, `&{` (ADR-006)
- `resolve_depth=0` as default -- top-level keywords only (ADR-006)

### Tests

| Test File | What It Covers |
|-----------|---------------|
| `tests/unit/test_parsing/test_keyword_resolver.py` | Depth 0/1/2 resolution, circular reference protection, missing keyword handling |
| `tests/unit/test_parsing/test_text_builder.py` | Noise filtering, tag inclusion, keyword name normalization, empty test body |
| `tests/unit/test_parsing/test_suite_collector.py` | Mock robot.api, verify TestCaseRecord production, file hash computation |
| `tests/unit/test_parsing/test_datadriver_reader.py` | CSV parsing, delimiter handling, comment row skipping, template name insertion |

### Performance Target
- Parse + text-build 1000 tests in < 2s

---

## Phase 2: Embedding Context

**Goal**: Implement the `SentenceTransformerAdapter` ACL, `EmbeddingMatrix` aggregate, and artifact serialization (NPZ + manifest JSON).

**Depends on**: Phase 1 (consumes `TestCaseRecord` / text representations)

### Files to Implement

| File | Key Types/Functions | ADR Reference |
|------|-------------------|--------------|
| `src/testcase_selection/embedding/ports.py` | `EmbeddingModel` Protocol (`encode()`, `embedding_dim`) | Domain Model |
| `src/testcase_selection/embedding/embedder.py` | `SentenceTransformerAdapter` implementing `EmbeddingModel` -- wraps `sentence-transformers` | ADR-002, Domain Model |
| `src/testcase_selection/embedding/models.py` | `Embedding`, `EmbeddingMatrix` (aggregate root with `validate_dimensions()`, `to_artifact()`), `EmbeddingArtifact`, `ArtifactManifest`, `ManifestEntry` | Domain Model |

### Key Design Points
- Lazy import of `sentence_transformers` inside `__init__` (ADR-002, ADR-005)
- L2-normalize at encode time: `normalize_embeddings=True` (ADR-002)
- `EmbeddingMatrix.validate_dimensions()` enforces: `vectors.shape[0] == len(test_ids)` and `vectors.shape[1] == embedding_dim` (Domain Model invariant)
- `to_artifact()` writes `embeddings.npz` + `test_manifest.json` (ADR-001 artifact contract)

### Tests

| Test File | What It Covers |
|-----------|---------------|
| `tests/unit/test_embedding/test_models.py` | EmbeddingMatrix dimension validation, to_artifact serialization/deserialization round-trip, ManifestEntry construction |
| `tests/unit/test_embedding/test_embedder.py` | Mock SentenceTransformer, verify encode call, normalization flag, dimension property |

### Performance Target
- Batch encode 1000 test descriptions in < 1s on CPU (model load ~2s one-time)
- NPZ write < 100ms for 5000 vectors

---

## Phase 3: Selection Context

**Goal**: Implement all 5 selection strategies behind the `SelectionStrategy` protocol, the strategy registry, tag filtering, `DiversitySelector` service, and `SelectionResult` aggregate.

**Depends on**: Phase 2 (consumes `EmbeddingArtifact` / vectors)

### Files to Implement

| File | Key Types/Functions | ADR Reference |
|------|-------------------|--------------|
| `src/testcase_selection/selection/strategy.py` | `SelectionStrategy` Protocol (`select(vectors, k, seed) -> list[int]`) | ADR-003, Domain Model |
| `src/testcase_selection/selection/fps.py` | `FarthestPointSampling`, `FPSMultiStart(n_starts=5)` -- core, no optional deps | ADR-003 |
| `src/testcase_selection/selection/kmedoids.py` | `KMedoidsSelection` -- requires `sklearn-extra` | ADR-003 |
| `src/testcase_selection/selection/dpp.py` | `DPPSelection` -- requires `dppy` | ADR-003 |
| `src/testcase_selection/selection/facility.py` | `FacilityLocationSelection` -- requires `apricot-select` | ADR-003 |
| `src/testcase_selection/selection/registry.py` | `StrategyRegistry.get(name)` with lazy ImportError handling | ADR-003 |
| `src/testcase_selection/selection/filtering.py` | `TagFilter.matches()`, `filter_by_tags()` | Domain Model |

### Key Design Points
- FPS uses `numpy.random.RandomState(seed)` not global seed (ADR-008)
- Optional strategies catch `ImportError` at registration, not at import time (ADR-003, ADR-005)
- `DiversitySelector` computes metrics: avg/min pairwise cosine distance, suite coverage (ADR-008)
- `SelectionResult.to_json()` / `.from_json()` for artifact contract (ADR-001)
- `DiversityMetrics` value object with `suite_coverage_ratio` property (Domain Model)

### Tests

| Test File | What It Covers |
|-----------|---------------|
| `tests/unit/test_selection/test_fps.py` | Deterministic output with seed, k=1 edge case, k=N edge case, 2-approximation spot check |
| `tests/unit/test_selection/test_kmedoids.py` | Skip if sklearn-extra missing, cluster representative property |
| `tests/unit/test_selection/test_dpp.py` | Skip if dppy missing, probabilistic output varies without seed |
| `tests/unit/test_selection/test_facility.py` | Skip if apricot missing, all clusters represented |
| `tests/unit/test_selection/test_registry.py` | FPS always available, optional strategies unavailable without deps, unknown name error |
| `tests/unit/test_selection/test_filtering.py` | Include/exclude tags, DataDriver filter, empty filter passes all |

### Performance Targets
- FPS: < 50ms for N=5000, k=200, d=384
- k-Medoids: < 5s for N=5000, k=200
- DPP: < 15s for N=5000, k=200

---

## Phase 4: Execution Context

**Goal**: Implement `DiversePreRunModifier` (SuiteVisitor), `DiverseDataDriverListener` (Listener v3), `TestFilter` value object, and `ExecutionPlan` aggregate.

**Depends on**: Phase 3 (consumes `SelectionResult` / `selected_tests.json`)

### Files to Implement

| File | Key Types/Functions | ADR Reference |
|------|-------------------|--------------|
| `src/testcase_selection/execution/prerun_modifier.py` | `DiversePreRunModifier(SuiteVisitor)` -- `start_suite`, `end_suite`, `visit_test` | ADR-004, Domain Model |
| `src/testcase_selection/execution/listener.py` | `DiverseDataDriverListener` -- `ROBOT_LISTENER_API_VERSION=3`, `ROBOT_LISTENER_PRIORITY=50` | ADR-004, Domain Model |
| `src/testcase_selection/execution/runner.py` | `ExecutionPlan.execute()` -- builds `robot.run_cli()` args with appropriate `--prerunmodifier` and `--listener` | Domain Model |

### Key Design Points
- `PreRunModifier` uses `suite.visit()` for programmatic execution (ADR-004 gotcha)
- Listener priority 50 ensures it fires AFTER DataDriver (ADR-004)
- `TestFilter.from_selection_result(result, datadriver_only=bool)` splits standard vs DD tests (Domain Model)
- `visit_test` is a no-op for performance (ADR-004)
- Empty suite pruning in `end_suite` (ADR-004)

### Tests

| Test File | What It Covers |
|-----------|---------------|
| `tests/unit/test_execution/test_prerun_modifier.py` | Filters correct tests, prunes empty suites, handles missing names gracefully |
| `tests/unit/test_execution/test_listener.py` | Priority value, filters after DataDriver, no-op when no DD tests selected |

---

## Phase 5: Pipeline Orchestration

**Goal**: Wire all contexts together via stage orchestrators, caching, and artifact management. Implement the CLI.

**Depends on**: Phases 1-4 (orchestrates all contexts)

### Files to Implement

| File | Key Types/Functions | ADR Reference |
|------|-------------------|--------------|
| `src/testcase_selection/pipeline/cache.py` | `CacheInvalidator.has_changes()`, `.save_hashes()` | Domain Model, ADR-007 |
| `src/testcase_selection/pipeline/artifacts.py` | `ArtifactManager` -- paths, load/save, validation | Domain Model, ADR-008 |
| `src/testcase_selection/pipeline/vectorize.py` | Stage 1 orchestrator: check cache -> parse -> embed -> save artifacts | ADR-001 |
| `src/testcase_selection/pipeline/select.py` | Stage 2 orchestrator: load artifacts -> validate -> filter -> select -> save | ADR-001 |
| `src/testcase_selection/pipeline/execute.py` | Stage 3 orchestrator: load selection -> build ExecutionPlan -> run | ADR-001 |
| `src/testcase_selection/cli.py` | `testcase-select vectorize/select/execute` subcommands | ADR-005 |

### Key Design Points
- Graceful degradation: catch errors, log, exit code 2, fall back to run-all (ADR-008)
- Artifact contract validation: shape check, dimension check, name existence check (ADR-008)
- Structured logging with `[DIVERSE-SELECT]` prefix and `stage=` field (ADR-008)
- Environment variable support: `DIVERSE_K`, `DIVERSE_STRATEGY`, `DIVERSE_OUTPUT`, `DIVERSE_SEED` (ADR-007)
- Custom exceptions: `SelectionError`, `VectorizationError`, `ArtifactError` (ADR-008)
- Wall-clock timing per stage (ADR-008)

### Tests

| Test File | What It Covers |
|-----------|---------------|
| `tests/unit/test_pipeline/test_cache.py` | Hash computation, change detection, hash file persistence |
| `tests/unit/test_pipeline/test_artifacts.py` | Load/save round-trip, validation failures, missing file handling |
| `tests/unit/test_pipeline/test_vectorize.py` | Cache hit skips embedding, force flag overrides, error fallback |
| `tests/unit/test_pipeline/test_select.py` | Artifact validation, strategy dispatch, metrics computation |

### Integration Tests

| Test File | What It Covers |
|-----------|---------------|
| `tests/integration/test_stage1_vectorize.py` | End-to-end: sample.robot -> embeddings.npz + manifest.json |
| `tests/integration/test_stage2_select.py` | End-to-end: artifacts -> selected_tests.json with correct structure |
| `tests/integration/test_stage3_execute.py` | End-to-end: selection file -> robot execution with filtered tests |

---

## Phase 6: CI/CD Integration

**Goal**: Provide ready-to-use CI pipeline configurations.

**Depends on**: Phase 5

### Files to Create

| File | Purpose | ADR Reference |
|------|---------|--------------|
| `config/github-actions.yml` | 3-job GitHub Actions workflow with caching + artifacts + PR annotations | ADR-007 |
| `config/gitlab-ci.yml` | 3-stage GitLab CI with `cache:key:files` + junit reports | ADR-007 |
| `config/Jenkinsfile` | Declarative pipeline with stash/unstash + Robot plugin | ADR-007 |

---

## Dependency Graph (Implementation Order)

```
Phase 0: Scaffolding
    |
Phase 1: Shared Kernel + Parsing Context
    |
Phase 2: Embedding Context
    |
Phase 3: Selection Context
    |
Phase 4: Execution Context
    |
Phase 5: Pipeline Orchestration + CLI
    |
Phase 6: CI/CD Integration
```

Phases 2, 3, and 4 could be parallelized across developers since they consume interfaces defined in Phase 1 and write to well-defined artifact contracts. However, integration testing requires sequential availability.

---

## Testing Strategy

### Unit Tests (per bounded context, mock-first TDD)

| Context | Mock Boundaries | Key Assertions |
|---------|----------------|----------------|
| Parsing | Mock `robot.api.TestSuite` | Correct `TestCaseRecord` production, noise filtered, keyword resolution depth |
| Embedding | Mock `SentenceTransformer` | Correct shape, normalization, artifact format |
| Selection | Synthetic embedding matrices | Deterministic output, approximation bounds, missing dep handling |
| Execution | Mock `robot.run_cli` + mock suite | Correct CLI args, filter application, priority values |
| Pipeline | Mock all context services | Cache hit/miss, graceful degradation, artifact validation |

### Integration Tests (end-to-end per stage)

- Stage 1: Real `sample.robot` -> real `sentence-transformers` -> real NPZ output
- Stage 2: Real NPZ -> real FPS -> real JSON output
- Stage 3: Real selection JSON -> real `robot` execution -> real `output.xml`

### Marks

```ini
[tool.pytest.ini_options]
markers = [
    "slow: loads ML models or large datasets",
    "integration: end-to-end stage tests",
]
```

Run fast tests: `uv run pytest -m "not slow and not integration"`
Run all: `uv run pytest`

---

## Performance Targets Summary

| Operation | Target | Scale | Reference |
|-----------|--------|-------|-----------|
| Parse + text-build | < 2s | 1000 tests | ADR-006 |
| Embedding (batch) | < 1s (+ 2s model load) | 1000 tests | ADR-002 |
| FPS selection | < 50ms | N=5000, k=200 | ADR-003 |
| k-Medoids selection | < 5s | N=5000, k=200 | ADR-003 |
| DPP selection | < 15s | N=5000, k=200 | ADR-003 |
| Artifact write (NPZ+JSON) | < 100ms | 5000 tests | ADR-001 |
| Cache check (file hashes) | < 200ms | 500 .robot files | ADR-007 |
| Warm pipeline overhead | < 5s total | Steady-state CI | ADR-001 |

---

## File Count Summary

| Category | Count |
|----------|-------|
| Source files (src/) | 22 |
| Unit test files | 16 |
| Integration test files | 3 |
| Test fixtures | 2 |
| CI configs | 3 |
| ADR documents | 8 |
| Architecture documents | 2 |
| **Total** | **56** |

---

## ADR Cross-Reference Index

| ADR | Key Decision | Phases Affected |
|-----|-------------|----------------|
| ADR-001 | 3-stage pipeline with artifact contracts | All |
| ADR-002 | MiniLM-L6-v2 + NPZ storage | Phase 2 |
| ADR-003 | FPS default + Strategy pattern | Phase 3 |
| ADR-004 | PreRunModifier + Listener v3 for DataDriver | Phase 4 |
| ADR-005 | uv + src-layout + optional deps | Phase 0 |
| ADR-006 | Natural language text representation | Phase 1 |
| ADR-007 | Content-hash caching + CI configs | Phase 5, 6 |
| ADR-008 | Graceful degradation + structured logging | Phase 5 |

---

## Risk Register

| Risk | Impact | Mitigation | ADR |
|------|--------|-----------|-----|
| Semantic diversity != fault-detection diversity | Selection misses real bugs | Empirical validation on defect data; facility location as alternative | ADR-003 |
| DataDriver test name prediction mismatch | DD tests not filtered | Pre-read CSV at index time; log unmatched names | ADR-004, ADR-008 |
| sentence-transformers download fails in CI | Stage 1 blocks | Graceful degradation to cached or run-all | ADR-008 |
| Variable-heavy tests cluster together | Poor diversity for parameterized tests | Noise filtering + higher resolve_depth | ADR-006 |
| N^2 pairwise matrix for large suites | OOM for N>50k | Out of scope (target <5000); warn in docs | ADR-003 |
| uv lockfile not reproducible across platforms | CI/dev mismatch | Commit uv.lock; test on CI runners | ADR-005 |
