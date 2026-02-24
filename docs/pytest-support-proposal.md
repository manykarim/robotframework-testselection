# Proposal: pytest Support for robotframework-testselection

**Branch**: `feature/add_pytest_support`
**Date**: 2026-02-24
**Status**: Proposal (not yet implemented)

## Executive Summary

Extend `robotframework-testselection` to support **pytest** test suites in addition to Robot Framework. The core insight: the selection algorithms (FPS, k-Medoids, DPP, Facility Location) and embedding infrastructure are framework-agnostic. Only the **parsing** and **execution** stages need framework-specific adapters.

This proposal adds a pytest plugin (`pytest --diverse-k=20`) and a new `pytest` bounded context within the existing package, reusing all shared infrastructure.

## Motivation

- Robot Framework users often have mixed test portfolios (RF + pytest)
- pytest is the dominant Python testing framework — supporting it greatly expands the user base
- The mathematical foundation (vector embeddings + diversity selection) applies identically to any collection of test case texts

## Architecture

### Current Architecture (Robot Framework only)

```
Parsing (RF-specific)  →  Embedding (generic)  →  Selection (generic)  →  Execution (RF-specific)
```

### Proposed Architecture (multi-framework)

```
                      ┌─ RF Parsing ──────────┐
                      │  (existing)            │
Test Discovery ──────►│                        ├──► Embedding ──► Selection ──► Execution
                      │  pytest Parsing (NEW)  │    (shared)      (shared)     ├─ RF Runner
                      └────────────────────────┘                               └─ pytest Plugin (NEW)
```

### Package Layout

```
src/TestSelection/
  shared/              # Unchanged — types, config
  parsing/             # RF-specific parsing (unchanged)
  embedding/           # Unchanged — EmbeddingModel protocol, embedder
  selection/           # Unchanged — strategies, registry, filtering
  pipeline/            # Orchestration (minor additions)
  execution/           # RF execution (unchanged)
  cli.py               # Extended with --framework pytest

  pytest/              # NEW bounded context
    __init__.py
    plugin.py          # pytest plugin (pytest11 entry point)
    collector.py       # Programmatic test collection via pytest API
    text_builder.py    # AST-based text representation for pytest tests
    runner.py          # pytest execution with deselection
```

## Detailed Design

### 1. Test Collection (`pytest/collector.py`)

Collects pytest test items programmatically without executing them:

```python
class CollectorPlugin:
    """Intercepts collected items via pytest_collection_modifyitems."""

    def __init__(self) -> None:
        self.items: list[pytest.Item] = []

    def pytest_collection_modifyitems(self, items: list[pytest.Item]) -> None:
        self.items.extend(items)
        items[:] = []  # Deselect all — collect only

def collect_tests(test_dir: str) -> list[pytest.Item]:
    collector = CollectorPlugin()
    pytest.main([test_dir, "--collect-only", "-q"], plugins=[collector])
    return collector.items
```

**Metadata extracted per test** (validated experimentally — see `scripts/experiments/pytest_collection_experiment.py`):

| Field | Source | Example |
|-------|--------|---------|
| `nodeid` | `item.nodeid` | `tests/unit/test_cli.py::TestRun::test_basic` |
| `name` | `item.name` | `test_basic` |
| `docstring` | `inspect.getdoc(item.function)` | `"Verify basic CLI invocation."` |
| `source_code` | `inspect.getsource(item.function)` | Full function body |
| `markers` | `item.iter_markers()` | `["slow", "integration"]` |
| `fixtures` | `item._fixtureinfo.argnames` | `["tmp_path", "mock_model"]` |
| `class_name` | `item.cls.__name__` | `TestRun` |
| `parametrize` | `item.callspec.params` | `{"k": 10, "strategy": "fps"}` |

**Experiment result**: Successfully collected **154 tests** from this project with all metadata fields populated.

### 2. Text Representation (`pytest/text_builder.py`)

Converts pytest test metadata into embeddable text strings. Four strategies were evaluated experimentally (see `scripts/experiments/vectorize_pytest_experiment.py`):

| Strategy | Mean Distance | Std | Spread | Inter/Intra Ratio |
|----------|--------------|-----|--------|-------------------|
| A: Name + Markers | 0.3215 | 0.1194 | 0.4978 | 2.1x |
| B: Full Source Code | 0.2847 | 0.0892 | 0.3961 | 1.5x |
| C: AST-Based | 0.4102 | 0.1441 | 0.5312 | 3.8x |
| **D: Combined** | **0.3891** | **0.1387** | **0.5645** | **5.9x** |

**Recommendation**: **Strategy D (Combined)** — best inter/intra class ratio (5.9x), meaning tests within the same functional group cluster together while tests from different groups spread apart. This maximizes the signal for diversity selection.

Combined text template:

```
Test: {qualname}. Markers: {markers}. {docstring}
Uses fixtures: {fixtures}. Calls: {meaningful_function_calls}.
Verifies: {assertion_summary}.
```

Example output:
```
Test: TestSelectCommand.test_select_basic. Markers: pytest.mark.slow.
Select command runs with default strategy. Uses fixtures: tmp_path, mock_embeddings.
Calls: CliRunner, runner.invoke, json.loads. Verifies: checks equality, checks identity/none.
```

**Why not full source code?** Option B had the worst inter/intra ratio (1.5x). Source code contains syntactic noise (variable names, string literals, imports) that drowns the semantic signal. The AST-based approach extracts what the test *does*, not how it's spelled.

### 3. pytest Plugin (`pytest/plugin.py`)

Registered as a `pytest11` entry point, activated via CLI options:

```python
def pytest_addoption(parser):
    group = parser.getgroup("diverse", "Diversity-based test selection")
    group.addoption("--diverse-k", type=int, default=0,
                    help="Select k most diverse tests (0 = disabled)")
    group.addoption("--diverse-strategy", default="fps",
                    help="Selection algorithm: fps, fps_multi, kmedoids, dpp, facility")
    group.addoption("--diverse-seed", type=int, default=42,
                    help="Random seed for reproducibility")
    group.addoption("--diverse-cache-dir", default=".diverse-cache",
                    help="Directory for embedding cache")
    group.addoption("--diverse-model", default="all-MiniLM-L6-v2",
                    help="Sentence-transformer model name")
    group.addoption("--diverse-include-markers", nargs="*", default=None,
                    help="Only include tests with these markers")
    group.addoption("--diverse-exclude-markers", nargs="*", default=None,
                    help="Exclude tests with these markers")

@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    k = config.getoption("--diverse-k")
    if k <= 0:
        return  # Plugin disabled

    # 1. Build text representations
    texts = [build_combined_text(item) for item in items]

    # 2. Embed (with caching)
    embedder = get_or_create_embedder(config)
    vectors = embedder.encode(texts)

    # 3. Select diverse subset
    strategy = get_strategy(config.getoption("--diverse-strategy"))
    indices = strategy.select(vectors, k=k, seed=config.getoption("--diverse-seed"))

    # 4. Deselect non-selected items (in-place mutation)
    selected_set = set(indices)
    deselected = [item for i, item in enumerate(items) if i not in selected_set]
    config.hook.pytest_deselected(items=deselected)
    items[:] = [items[i] for i in sorted(indices)]
```

Key design decisions:

- **`trylast=True`**: Ensures our hook runs after all other collection hooks (markers, -k expressions, etc.), so we select from an already-filtered set
- **In-place `items[:] =`**: Standard pytest pattern for deselecting items. We call `pytest_deselected` so other plugins (e.g., pytest-html) can track what was deselected
- **Caching**: Embeddings stored in `.diverse-cache/` using the same content-hash scheme as the RF pipeline. Subsequent runs skip vectorization when source files haven't changed
- **Zero-config activation**: Simply `pytest --diverse-k=20` to select 20 most diverse tests

### 4. Embedding Cache for pytest

Reuse the existing `pipeline/cache.py` content-hash invalidation:

```python
# Hash each .py test file, store alongside embeddings
# On subsequent runs, only re-embed changed files and merge with cached vectors
```

Cache directory structure:
```
.diverse-cache/
  embeddings.npz         # N x 384 float32 matrix
  test_manifest.json     # Test metadata (nodeids, markers, etc.)
  file_hashes.json       # MD5 hashes for cache invalidation
```

### 5. CLI Integration

Extend the existing CLI to support pytest as a framework:

```bash
# pytest via plugin (primary usage — no CLI wrapper needed)
pytest --diverse-k=20 --diverse-strategy=fps tests/

# Via testcase-select CLI (wraps pytest internally)
testcase-select run --framework pytest --suite tests/ --k 20 --strategy fps

# Stage-by-stage (also works)
testcase-select vectorize --framework pytest --suite tests/ --output ./artifacts/
testcase-select select --artifacts ./artifacts/ --k 20 --strategy fps
# Execute stage uses pytest instead of robot:
testcase-select execute --framework pytest --suite tests/ --selection selected_tests.json
```

### 6. Shared Types Extension

Add a framework-agnostic `TestCaseRecord` variant or extend the existing one:

```python
@dataclass(frozen=True)
class TestCaseRecord:
    test_id: TestCaseId
    name: str
    tags: frozenset[Tag]           # markers for pytest
    suite_source: SuitePath        # module path for pytest
    suite_name: str                # module name for pytest
    text_representation: TextRepresentation
    is_datadriver: bool = False
    framework: str = "robot"       # NEW: "robot" | "pytest"
    node_id: str | None = None     # NEW: pytest nodeid for deselection
```

Alternatively, keep `TestCaseRecord` unchanged and add `PytestTestRecord` as a parallel type that maps to the same `EmbeddingMatrix` input.

## Scalability

### Algorithm Performance (validated experimentally)

| Suite Size | FPS Time | Memory | Recommendation |
|-----------|----------|--------|----------------|
| 100 | 12 ms | ~150 KB | FPS (default) |
| 500 | 99 ms | ~750 KB | FPS |
| 1,000 | 304 ms | ~1.5 MB | FPS |
| 5,000 | 8.0 s | ~7.5 MB | FPS (still practical) |
| 10,000 | ~24 s | ~15 MB | FPS or centroid-FPS |
| 50,000 | ~5 min | ~75 MB | Centroid-FPS (1.7x speedup) |

### Centroid-FPS for Large Suites

For suites > 10K tests, a two-phase approach:

1. **Cluster**: MiniBatchKMeans into `sqrt(N)` clusters (~224 for 50K)
2. **Select from centroids**: FPS on cluster centroids, then pick nearest real test per centroid

This provides 1.7x speedup at 50K tests with negligible quality loss.

### Embedding Time (dominates for large suites)

| Suite Size | Embedding Time (all-MiniLM-L6-v2, CPU) |
|-----------|----------------------------------------|
| 1,000 | ~5 s |
| 10,000 | ~50 s |
| 50,000 | ~4 min |

Content-hash caching makes this a one-time cost per test change.

## Implementation Plan

### Phase 1: Core pytest Collection & Text Building (Week 1-2)

1. Create `src/TestSelection/pytest/` sub-package
2. Implement `collector.py` — programmatic test collection
3. Implement `text_builder.py` — Combined strategy (D) text builder
4. Unit tests for collection and text building
5. Ensure existing RF tests still pass

### Phase 2: pytest Plugin (Week 2-3)

1. Implement `plugin.py` — `pytest_addoption` + `pytest_collection_modifyitems`
2. Add `pytest11` entry point to `pyproject.toml`
3. Implement embedding caching for pytest
4. Integration test: `pytest --diverse-k=10 tests/unit/`
5. Verify compatibility with common plugins (pytest-cov, pytest-xdist)

### Phase 3: CLI Integration (Week 3-4)

1. Add `--framework pytest` option to `testcase-select` CLI
2. Implement `pytest/runner.py` for programmatic pytest invocation
3. Adapt `pipeline/vectorize.py` and `pipeline/execute.py` for multi-framework dispatch
4. End-to-end test: `testcase-select run --framework pytest --suite tests/ --k 20`

### Phase 4: Documentation & Packaging (Week 4)

1. Update README with pytest usage section
2. ADR-009: Multi-Framework Support Architecture
3. Move `robotframework` to `[robot]` optional dependency group
4. Update CI to test both frameworks
5. Release as v0.3.0

## Dependency Changes

### Current `pyproject.toml`

```toml
dependencies = [
    "robotframework>=7.0",   # Required
    "numpy>=1.24",
    "scikit-learn>=1.3",
]
```

### Proposed

```toml
dependencies = [
    "numpy>=1.24",
    "scikit-learn>=1.3",
]

[project.optional-dependencies]
robot = ["robotframework>=7.0"]
pytest = []  # pytest is already in the test environment; plugin activates on import
vectorize = ["sentence-transformers>=2.2"]
all = [
    "robotframework-testselection[robot]",
    "robotframework-testselection[vectorize]",
    "robotframework-testselection[selection-extras]",
]
```

This makes `robotframework` optional, allowing pure-pytest users to install without it.

### Entry Points

```toml
[project.entry-points.pytest11]
diverse-selection = "TestSelection.pytest.plugin"

[project.scripts]
testcase-select = "TestSelection.cli:main"
```

## pytest-xdist Compatibility

When `pytest-xdist` is active (parallel test execution), our `pytest_collection_modifyitems` hook runs on the **controller node** before items are distributed to workers. This means:

- Selection happens once, before distribution
- Workers only receive pre-selected items
- No special handling needed for xdist compatibility

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `inspect.getsource()` fails for dynamically generated tests | Low | Fall back to name-only text representation |
| Large monorepo (50K+ tests) embedding is slow | Medium | Content-hash caching + centroid-FPS |
| Plugin conflicts with other pytest plugins | Low | `trylast=True` ordering + integration tests |
| Parametrized test explosion (1000 params) | Medium | Collapse parametrized variants to single entry before selection |
| Breaking existing RF users | Low | No changes to RF code paths; pytest is additive |

## Parametrized Test Handling

For heavily parametrized tests (`@pytest.mark.parametrize` with many values), each parameter combination appears as a separate test item. Options:

**A. Select from all items individually** (default)
- Simple, each parametrized variant is embedded separately
- Pro: Fine-grained selection
- Con: Can over-represent one parametrized test if its variants are diverse

**B. Group by `originalname`, embed once, expand selected**
- Group items by `item.originalname` (without `[param0]` suffix)
- Embed the base test once, select at group level
- Expand selected groups back to individual items
- Pro: Better balance across test functions
- Con: Loses parameter-level diversity

**Recommendation**: Default to **A** with a `--diverse-group-parametrize` flag to enable **B** for projects with heavy parametrization.

## Testing Strategy

```
tests/
  unit/
    test_pytest_collector.py    # Collection API tests
    test_pytest_text_builder.py # Text representation tests
    test_pytest_plugin.py       # Plugin hook tests (pytester fixture)
  integration/
    test_pytest_pipeline.py     # End-to-end: collect → embed → select → run
  fixtures/
    sample_pytest_suite/        # Small pytest suite for testing
      conftest.py
      test_auth.py
      test_api.py
      test_helpers.py
```

Use pytest's `pytester` fixture for plugin integration tests — it spawns a subprocess pytest invocation and validates the outcome.

## Experiment Scripts

Two experiment scripts validate the approach (already on this branch):

1. **`scripts/experiments/pytest_collection_experiment.py`** — Demonstrates programmatic collection of 154 tests with full metadata extraction
2. **`scripts/experiments/vectorize_pytest_experiment.py`** — Compares 4 text representation strategies and validates embedding quality with inter/intra class ratio analysis

Run with:
```bash
uv run python scripts/experiments/pytest_collection_experiment.py
uv run python scripts/experiments/vectorize_pytest_experiment.py
```

## Success Criteria

- [ ] `pytest --diverse-k=20 tests/` selects exactly 20 tests and runs them
- [ ] Selection is deterministic (same seed = same tests)
- [ ] Embedding cache avoids re-vectorization on unchanged code
- [ ] `testcase-select run --framework pytest` works end-to-end
- [ ] Existing Robot Framework functionality is unaffected
- [ ] All existing tests pass
- [ ] New pytest-specific tests achieve >80% coverage of the new code
- [ ] Works with pytest-xdist, pytest-cov, and pytest-html
