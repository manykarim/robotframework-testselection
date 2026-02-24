# robotframework-testselection

Vector-based diverse test case selection for Robot Framework and pytest. Embeds test cases as semantic vectors and selects maximally diverse subsets to reduce test suite execution time while preserving coverage breadth.

## How It Works

The system operates as a **3-stage pipeline**:

```
Vectorize              Select                Execute
(.robot files) ──►  (embeddings.npz   ──►  (selected_tests.json
                     test_manifest.json)     + robot --prerunmodifier)
```

1. **Vectorize** — Parses `.robot` files via the Robot Framework API, converts each test case to a natural-language text representation (name + tags + resolved keyword tree), then encodes with `all-MiniLM-L6-v2` (384-dim sentence embeddings).
2. **Select** — Loads embedding vectors and applies a diversity-maximizing selection algorithm (default: Farthest Point Sampling) to choose *k* tests that are as semantically different from each other as possible.
3. **Execute** — Runs the selected tests via Robot Framework using a `PreRunModifier` (for standard tests) and a `Listener v3` (for DataDriver-generated tests).

If any stage fails, the pipeline **gracefully degrades** by running all tests (exit code 2).

## Installation

```bash
# From PyPI
pip install robotframework-testselection

# With sentence-transformers for vectorization
pip install robotframework-testselection[vectorize]

# With all optional selection algorithms
pip install robotframework-testselection[selection-extras]

# Everything
pip install robotframework-testselection[all]
```

### Development Install (from source)

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra vectorize
uv sync --extra all
```

The pytest plugin is included in the base install and activates when you pass `--diverse-k` to pytest.

### Dependency Groups

| Group | Packages | Purpose |
|-------|----------|---------|
| *(base)* | `robotframework`, `numpy`, `scikit-learn` | Core pipeline |
| `vectorize` | `sentence-transformers` | Embedding model (Stage 1) |
| `selection-extras` | `scikit-learn-extra`, `dppy`, `apricot-select` | k-Medoids, DPP, Facility Location strategies |
| `chromadb` | `chromadb` | Alternative vector storage |
| `dev` | `pytest`, `pytest-cov`, `pytest-benchmark`, `ruff`, `mypy` | Development |

## Quick Start

### Full Pipeline (one command)

```bash
testcase-select run \
  --suite tests/robot/ \
  --k 20 \
  --strategy fps \
  --seed 42 \
  --output-dir ./results/
```

### Stage-by-Stage

```bash
# Stage 1: Vectorize
testcase-select vectorize \
  --suite tests/robot/ \
  --output ./artifacts/ \
  --model all-MiniLM-L6-v2 \
  --resolve-depth 2

# Stage 2: Select
testcase-select select \
  --artifacts ./artifacts/ \
  --k 20 \
  --strategy fps \
  --seed 42 \
  --output selected_tests.json

# Stage 3: Execute
testcase-select execute \
  --suite tests/robot/ \
  --selection selected_tests.json \
  --output-dir ./results/
```

### With DataDriver CSV Files

```bash
testcase-select vectorize \
  --suite tests/robot/ \
  --output ./artifacts/ \
  --datadriver-csv tests/data/login.csv tests/data/search.csv
```

### Passing Robot Framework Options

All arguments after `--` are forwarded directly to `robot`. This lets you combine diversity selection with any Robot Framework option — variables, tag filters, log levels, listeners, etc.

```bash
# Pass variables and set log level
testcase-select run --suite tests/ --k 20 \
  -- --variable ENV:staging --variable USER:admin --loglevel DEBUG

# Use Robot's own tag filtering on top of diversity selection
testcase-select execute --suite tests/ --selection sel.json \
  -- --include smoke --exclude manual

# Add metadata and custom output name
testcase-select run --suite tests/ --k 30 \
  -- --metadata Version:2.1 --name "Smoke Regression"

# Set variables file and debug log
testcase-select run --suite tests/ --k 50 \
  -- --variablefile config/env_staging.py --debugfile debug.log
```

This works with both `run` and `execute` subcommands, including during graceful fallback (when selection fails and all tests are run).

### pytest Support

The diversity selection algorithm also works with pytest test suites, via a pytest plugin:

```bash
# Select 20 most diverse tests
pytest --diverse-k=20 tests/

# With custom strategy and seed
pytest --diverse-k=30 --diverse-strategy=fps_multi --diverse-seed=123 tests/

# Filter by markers before selection
pytest --diverse-k=20 --diverse-include-markers slow integration tests/

# Group parametrized tests (select at group level)
pytest --diverse-k=20 --diverse-group-parametrize tests/
```

Or via the `testcase-select` CLI:

```bash
# Full pipeline with pytest
testcase-select run --framework pytest --suite tests/ --k 20 --strategy fps

# Stage-by-stage
testcase-select vectorize --framework pytest --suite tests/ --output ./artifacts/
testcase-select select --artifacts ./artifacts/ --k 20
testcase-select execute --framework pytest --suite tests/ --selection selected_tests.json
```

The pytest plugin is installed automatically and activated by `--diverse-k`. It uses AST-based text representation with sentence embeddings for test diversity analysis.

### Direct Robot Framework Integration

You can also use the components directly with `robot`:

```bash
# PreRunModifier for standard tests
robot --prerunmodifier TestSelection.execution.prerun_modifier.DiversePreRunModifier:selected_tests.json tests/

# Listener v3 for DataDriver tests
robot --listener TestSelection.execution.listener.DiverseDataDriverListener:selected_tests.json tests/

# Both together
robot \
  --prerunmodifier TestSelection.execution.prerun_modifier.DiversePreRunModifier:selected_tests.json \
  --listener TestSelection.execution.listener.DiverseDataDriverListener:selected_tests.json \
  tests/
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DIVERSE_K` | `50` | Number of tests to select |
| `DIVERSE_STRATEGY` | `fps` | Selection algorithm |
| `DIVERSE_SEED` | `42` | Random seed for reproducibility |
| `DIVERSE_OUTPUT` | *(none)* | Output file path for selection JSON |

## Selection Strategies

| Strategy | Name | Dependencies | Description |
|----------|------|-------------|-------------|
| **FPS** | `fps` | *(base)* | Farthest Point Sampling. Greedy farthest-first traversal. O(N*k*d). 2-approximation guarantee for max-min dispersion. **Default.** |
| **FPS Multi-Start** | `fps_multi` | *(base)* | Runs FPS from multiple random starting points, keeps the result with the highest minimum pairwise distance. Mitigates initial-point sensitivity. |
| **k-Medoids** | `kmedoids` | `selection-extras` | PAM algorithm for medoid-based clustering. Better centroid representativeness. |
| **DPP** | `dpp` | `selection-extras` | Determinantal Point Process. Probabilistic repulsion-based sampling. |
| **Facility Location** | `facility` | `selection-extras` | Submodular facility location maximization. Optimizes for both diversity and representativeness. |

## Tag Filtering

Filter tests by tags before selection:

```bash
# Include only smoke and regression tests
testcase-select select \
  --artifacts ./artifacts/ \
  --k 20 \
  --include-tags smoke regression

# Exclude slow tests
testcase-select select \
  --artifacts ./artifacts/ \
  --k 20 \
  --exclude-tags slow manual

# Exclude DataDriver tests
testcase-select select \
  --artifacts ./artifacts/ \
  --k 20 \
  --no-datadriver
```

## Caching

Stage 1 uses **content-hash caching**. It computes MD5 hashes of all `.robot` and `.csv` files and stores them alongside artifacts. On subsequent runs, vectorization is skipped when no source files have changed. Use `--force` to bypass the cache:

```bash
testcase-select vectorize --suite tests/ --output ./artifacts/ --force
```

## Artifacts

The pipeline produces these artifacts:

| File | Stage | Format | Contents |
|------|-------|--------|----------|
| `embeddings.npz` | Vectorize | NumPy compressed | N x 384 float32 matrix |
| `test_manifest.json` | Vectorize | JSON | Test metadata (names, tags, suites, IDs) |
| `file_hashes.json` | Vectorize | JSON | Source file MD5 hashes for cache invalidation |
| `selected_tests.json` | Select | JSON | Selected test list + diversity metrics |

### Selection Output Format

```json
{
  "strategy": "fps",
  "k": 20,
  "seed": 42,
  "total_tests": 150,
  "filtered_tests": 120,
  "selected": [
    {
      "name": "Login With Valid Credentials",
      "id": "a1b2c3d4...",
      "suite": "tests/login.robot",
      "is_datadriver": false
    }
  ],
  "diversity_metrics": {
    "avg_pairwise_distance": 0.8234,
    "min_pairwise_distance": 0.4512,
    "suite_coverage": 8,
    "suite_total": 10
  }
}
```

## CI/CD Integration

Pre-built configurations are provided in `config/`:

### GitHub Actions

```yaml
# .github/workflows/diverse-tests.yml
# Copy from config/github-actions.yml
```

Features:
- 3-job pipeline with artifact transfer between stages
- Content-hash caching (`actions/cache@v4`) to skip vectorization
- Automatic PR annotations with selection summary
- `workflow_dispatch` for manual runs with custom k/strategy

### GitLab CI

```yaml
# .gitlab-ci.yml
# Copy from config/gitlab-ci.yml
```

### Jenkins

```groovy
// Jenkinsfile
// Copy from config/Jenkinsfile
```

## Project Structure

```
src/TestSelection/
  shared/              # Shared kernel (types, config)
    types.py           # TestCaseId, Tag, TestCaseRecord, KeywordTree, ...
    config.py          # PipelineConfig, TextBuilderConfig
  parsing/             # Bounded context: Robot Framework parsing
    suite_collector.py # RobotApiAdapter (TestSuite.from_file_system)
    keyword_resolver.py# Recursive keyword tree resolution
    text_builder.py    # Natural language text representation
    datadriver_reader.py # DataDriver CSV reader
  embedding/           # Bounded context: vector embedding
    ports.py           # EmbeddingModel protocol
    embedder.py        # SentenceTransformerAdapter (ACL)
    models.py          # EmbeddingMatrix aggregate, ManifestEntry
  selection/           # Bounded context: diversity selection
    strategy.py        # SelectionStrategy protocol, SelectionResult
    fps.py             # FarthestPointSampling, FPSMultiStart
    kmedoids.py        # KMedoidsSelection (optional)
    dpp.py             # DPPSelection (optional)
    facility.py        # FacilityLocationSelection (optional)
    registry.py        # StrategyRegistry with auto-discovery
    filtering.py       # Tag-based pre-selection filtering
  execution/           # Bounded context: Robot Framework execution
    prerun_modifier.py # DiversePreRunModifier (SuiteVisitor)
    listener.py        # DiverseDataDriverListener (Listener v3)
    runner.py          # ExecutionRunner
  pytest/              # pytest integration
    plugin.py          # pytest plugin (pytest11 entry point)
    collector.py       # Programmatic test collection
    text_builder.py    # AST-based text representation
    runner.py          # pytest execution
  pipeline/            # Orchestration layer
    vectorize.py       # Stage 1 orchestrator
    select.py          # Stage 2 orchestrator
    execute.py         # Stage 3 orchestrator
    cache.py           # Content-hash cache invalidator
    artifacts.py       # Artifact storage and validation
    errors.py          # Domain error hierarchy
  cli.py               # CLI entry point (argparse)

tests/
  fixtures/            # Sample .robot and .csv files
  unit/                # Unit tests
  integration/         # Integration tests
  benchmarks/          # Performance benchmarks
```

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run all tests
uv run pytest tests/ -v --benchmark-disable

# Run unit tests only
uv run pytest tests/unit/ -v

# Run integration tests only
uv run pytest tests/integration/ -v

# Run benchmarks
uv run pytest tests/benchmarks/ -v

# Lint
uv run ruff check src/

# Type check
uv run mypy src/
```

### Test Markers

```bash
# Skip slow tests (ML model loading)
uv run pytest -m "not slow"

# Only integration tests
uv run pytest -m integration

# Only benchmarks
uv run pytest -m benchmark
```

## Coverage Retention Analysis

Measured on [robotframework-doctestlibrary](https://github.com/manykarim/robotframework-doctestlibrary) (5,045 statements in the DocTest package). All strategies use `all-MiniLM-L6-v2` embeddings (384-dim), seed=42.

### pytest (594 tests, 70.1% full coverage)

| Strategy | Selection | Tests | Coverage | Retention | Speedup |
|---|---|---|---|---|---|
| *(full suite)* | 100% | 594 | 70.1% | baseline | 1.0x |
| **fps** | **50%** | **297** | **65.9%** | **93.9%** | **3.4x** |
| dpp | 50% | 297 | 65.9% | 93.9% | 3.5x |
| kmedoids | 50% | 297 | 65.8% | 93.8% | 3.1x |
| fps_multi | 50% | 297 | 65.7% | 93.7% | 3.0x |
| facility | 50% | 297 | 64.9% | 92.6% | 2.9x |
| random | 50% | 297 | 64.2% | 91.5% | 2.3x |
| **facility** | **20%** | **119** | **58.6%** | **83.6%** | **7.5x** |
| kmedoids | 20% | 119 | 58.3% | 83.1% | 6.7x |
| fps | 20% | 119 | 56.1% | 80.0% | 6.7x |
| fps_multi | 20% | 119 | 53.8% | 76.8% | 8.0x |
| random | 20% | 119 | 53.2% | 75.8% | 5.4x |
| dpp | 20% | 119 | 47.6% | 67.8% | 6.0x |
| **fps_multi** | **10%** | **60** | **47.7%** | **67.9%** | **18.5x** |
| kmedoids | 10% | 60 | 46.9% | 66.9% | 14.9x |
| facility | 10% | 60 | 46.8% | 66.8% | 11.5x |
| random | 10% | 60 | 44.3% | 63.1% | 10.3x |
| dpp | 10% | 60 | 42.9% | 61.1% | 13.7x |
| fps | 10% | 60 | 40.6% | 57.9% | 21.4x |

### Robot Framework (126 tests, 63.5% full coverage)

| Strategy | Selection | Tests | Coverage | Retention | Speedup |
|---|---|---|---|---|---|
| *(full suite)* | 100% | 126 | 63.5% | baseline | 1.0x |
| **fps_multi** | **50%** | **63** | **61.6%** | **97.0%** | **1.5x** |
| facility | 50% | 63 | 61.6% | 96.9% | 1.5x |
| fps | 50% | 63 | 61.2% | 96.4% | 1.6x |
| kmedoids | 50% | 63 | 60.5% | 95.3% | 1.5x |
| dpp | 50% | 63 | 58.5% | 92.0% | 1.7x |
| random | 50% | 63 | 57.2% | 90.1% | 2.3x |
| **facility** | **20%** | **26** | **51.8%** | **81.5%** | **4.4x** |
| kmedoids | 20% | 26 | 47.5% | 74.8% | 5.5x |
| random | 20% | 26 | 46.0% | 72.5% | 5.7x |
| fps | 20% | 26 | 45.3% | 71.4% | 3.1x |
| fps_multi | 20% | 26 | 45.3% | 71.4% | 2.9x |
| dpp | 20% | 26 | 43.9% | 69.0% | 4.6x |
| **kmedoids** | **10%** | **13** | **42.3%** | **66.6%** | **11.3x** |
| dpp | 10% | 13 | 40.9% | 64.3% | 10.7x |
| facility | 10% | 13 | 40.2% | 63.3% | 12.9x |
| random | 10% | 13 | 37.6% | 59.2% | 11.3x |
| fps_multi | 10% | 13 | 37.5% | 59.1% | 10.7x |
| fps | 10% | 13 | 33.2% | 52.3% | 8.3x |

### Key Findings

- **50% selection retains 92-97% of coverage** across all strategies — the diversity algorithms select tests maximizing behavioral breadth, with minimal redundancy loss
- **Facility location excels at 20% selection** — 83.6% retention on pytest, 81.5% on Robot Framework, making it the best choice for aggressive smoke-test suites
- **kmedoids matches facility at 20%** on pytest (83.1% retention) and leads at 10% on Robot Framework (66.6%)
- **fps_multi leads at 10% pytest** (67.9% retention) where initial-point sensitivity matters most
- **DPP performs well at 50%** (93.9% retention, matching FPS) but drops at 20% due to FPS fallback for large k values
- **All diversity strategies outperform random** at 50% by 2-7 percentage points of retention
- At 50% robot selection, fps_multi achieves **97% retention** — meaning half the tests contribute almost no incremental coverage

## Performance

Benchmarked on 384-dimensional normalized vectors:

| N (tests) | FPS Time | FPS Multi (5 starts) |
|-----------|----------|---------------------|
| 100 | 12 ms | — |
| 500 | 99 ms | — |
| 1,000 | 304 ms | 376 ms (3 starts) |
| 5,000 | 8.0 s | — |

Keyword resolution: ~18 us/depth-1 resolve, ~61 us/depth-3 resolve. Text building for 100 tests: ~458 us.

## Architecture Decision Records

| ADR | Title |
|-----|-------|
| [ADR-001](docs/adr/ADR-001-pipeline-architecture.md) | 3-Stage Pipeline Architecture |
| [ADR-002](docs/adr/ADR-002-embedding-model-and-storage.md) | Embedding Model and Storage |
| [ADR-003](docs/adr/ADR-003-selection-algorithm-strategy.md) | Selection Algorithm Strategy Pattern |
| [ADR-004](docs/adr/ADR-004-robot-framework-integration.md) | Robot Framework Integration |
| [ADR-005](docs/adr/ADR-005-project-structure-and-uv.md) | Project Structure and uv |
| [ADR-006](docs/adr/ADR-006-text-representation-strategy.md) | Text Representation Strategy |
| [ADR-007](docs/adr/ADR-007-cicd-integration-and-caching.md) | CI/CD Integration and Caching |
| [ADR-008](docs/adr/ADR-008-reliability-and-observability.md) | Reliability and Observability |

## License

Apache 2.0
