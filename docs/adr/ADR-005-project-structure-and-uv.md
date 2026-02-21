# ADR-005: Python Project Structure and Dependency Management with uv

## Status

Proposed

## Date

2026-02-21

## Context

The Vector-Based Diverse Test Selection system for Robot Framework requires a Python project that:

1. **Maps Domain-Driven Design bounded contexts to packages.** The system has five distinct bounded contexts identified during design -- Parsing (keyword tree resolution and text representation), Embedding (sentence-transformer vectorization), Selection (diversity algorithms like FPS, k-Medoids, DPP, facility location), Execution (PreRunModifier and Listener v3 integration with Robot Framework), and Pipeline Orchestration (three-stage coordination with caching and artifacts).

2. **Manages a heterogeneous dependency graph.** Core dependencies (robotframework, numpy) are lightweight, but the vectorization stage pulls in sentence-transformers (~500 MB including torch), and several selection strategies depend on optional packages (scikit-learn-extra for k-Medoids, dppy for DPP sampling, apricot-select for facility location). CI environments running only Stage 3 (execution) should not need to install ML dependencies.

3. **Supports fast, reproducible builds in CI/CD.** The three-stage pipeline architecture (ADR TBD) means each CI job installs only its required dependency subset. Slow dependency resolution directly increases pipeline wall-clock time.

4. **Targets Python 3.11+ only.** No legacy Python support is required. The team can use modern language features and stdlib additions.

5. **Follows Python packaging best practices** for a tool that will be installed via `pip install` or `uv pip install` from source or eventually from PyPI.

Several Python dependency managers were evaluated:

- **pip + requirements.txt**: No lockfile, no dependency resolution guarantee, no optional groups without maintaining multiple files.
- **Poetry**: Mature lockfile support but slow resolution (minutes for large dependency trees involving torch), non-standard `pyproject.toml` extensions (`[tool.poetry]`), and the resolver struggles with packages that have platform-specific wheels (sentence-transformers/torch).
- **PDM**: PEP 621 compliant, lockfile support, but smaller ecosystem and less CI tooling integration.
- **uv**: Written in Rust, 10-100x faster than pip for resolution and installation, native lockfile (`uv.lock`), built-in virtual environment management, built-in Python version management, full PEP 621 compliance, pip-compatible interface, actively maintained by Astral (the ruff team).

## Decision

### 1. Use `uv` as the sole dependency management and virtual environment tool

All dependency installation, virtual environment creation, lockfile management, and script execution will use `uv`. The project will not include `requirements.txt` files or Poetry/PDM configuration.

### 2. Use src-layout with bounded context packages

The project follows the `src/` layout convention, with a single top-level package `testcase_selection` containing sub-packages that map one-to-one to DDD bounded contexts.

### 3. Use `pyproject.toml` with optional dependency groups

Heavy and optional dependencies are isolated into named groups so that CI jobs and users install only what they need.

### 4. Target Python 3.11+

The minimum Python version is 3.11.

### Project Layout

```
testcase-selection/
├── pyproject.toml                  # Single source of truth for project metadata
├── uv.lock                         # Deterministic lockfile (committed to VCS)
├── src/
│   └── testcase_selection/
│       ├── __init__.py             # Package version, public API re-exports
│       ├── cli.py                  # CLI entry points (click or argparse)
│       │
│       ├── parsing/                # Bounded Context: Parsing
│       │   ├── __init__.py
│       │   ├── keyword_resolver.py #   KeywordTreeResolver -- recursive keyword map building
│       │   ├── text_builder.py     #   TextRepresentationBuilder -- test-to-text conversion
│       │   ├── suite_collector.py  #   Collects tests via robot.api TestSuite.from_file_system
│       │   └── datadriver_reader.py#   Pre-reads DataDriver CSV/Excel to generate test names
│       │
│       ├── embedding/              # Bounded Context: Embedding
│       │   ├── __init__.py
│       │   ├── embedder.py         #   SentenceTransformerEmbedder -- wraps encode()
│       │   ├── models.py           #   EmbeddingMatrix, Embedding value objects
│       │   └── ports.py            #   Embedder protocol (typing.Protocol for DI)
│       │
│       ├── selection/              # Bounded Context: Selection
│       │   ├── __init__.py
│       │   ├── strategy.py         #   SelectionStrategy protocol
│       │   ├── fps.py              #   FPS + multi-start FPS (core, no optional deps)
│       │   ├── kmedoids.py         #   k-Medoids via sklearn-extra (optional dep)
│       │   ├── dpp.py              #   DPP via dppy (optional dep)
│       │   ├── facility.py         #   Facility location via apricot-select (optional dep)
│       │   ├── registry.py         #   Strategy registry -- name-to-implementation lookup
│       │   └── filtering.py        #   Tag-based pre-filtering before selection
│       │
│       ├── execution/              # Bounded Context: Execution
│       │   ├── __init__.py
│       │   ├── prerun_modifier.py  #   DiversePreRunModifier (SuiteVisitor)
│       │   ├── listener.py         #   DataDriver Listener v3 (priority-aware)
│       │   └── runner.py           #   Execution orchestrator (robot.run_cli wrapper)
│       │
│       ├── pipeline/               # Bounded Context: Pipeline Orchestration
│       │   ├── __init__.py
│       │   ├── vectorize.py        #   Stage 1 orchestrator (parse -> embed -> save)
│       │   ├── select.py           #   Stage 2 orchestrator (load -> filter -> select -> save)
│       │   ├── execute.py          #   Stage 3 orchestrator (load selection -> run)
│       │   ├── cache.py            #   File hash caching for change detection
│       │   └── artifacts.py        #   Artifact read/write (NPZ for embeddings, JSON for manifests)
│       │
│       └── shared/                 # Shared Kernel
│           ├── __init__.py
│           ├── types.py            #   TestCaseId, SuitePath, TagSet, EmbeddingDim etc.
│           └── config.py           #   Configuration dataclasses (PipelineConfig, SelectionConfig)
│
├── tests/
│   ├── unit/
│   │   ├── test_parsing/
│   │   │   ├── test_keyword_resolver.py
│   │   │   ├── test_text_builder.py
│   │   │   ├── test_suite_collector.py
│   │   │   └── test_datadriver_reader.py
│   │   ├── test_embedding/
│   │   │   ├── test_embedder.py
│   │   │   └── test_models.py
│   │   ├── test_selection/
│   │   │   ├── test_fps.py
│   │   │   ├── test_kmedoids.py
│   │   │   ├── test_dpp.py
│   │   │   ├── test_facility.py
│   │   │   ├── test_registry.py
│   │   │   └── test_filtering.py
│   │   ├── test_execution/
│   │   │   ├── test_prerun_modifier.py
│   │   │   └── test_listener.py
│   │   └── test_pipeline/
│   │       ├── test_vectorize.py
│   │       ├── test_select.py
│   │       ├── test_cache.py
│   │       └── test_artifacts.py
│   ├── integration/
│   │   ├── test_stage1_vectorize.py
│   │   ├── test_stage2_select.py
│   │   └── test_stage3_execute.py
│   └── fixtures/
│       ├── sample.robot
│       └── sample_datadriver.csv
│
├── docs/
│   ├── adr/
│   └── architecture/
├── scripts/
└── config/
```

### pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "testcase-selection"
version = "0.1.0"
description = "Vector-based diverse test case selection for Robot Framework"
requires-python = ">=3.11"
license = { text = "MIT" }
readme = "README.md"
dependencies = [
    "robotframework>=7.0",
    "numpy>=1.24",
]

[project.optional-dependencies]
vectorize = [
    "sentence-transformers>=2.2",
]
selection-extras = [
    "scikit-learn-extra>=0.3",
    "dppy>=0.3",
    "apricot-select>=0.6",
]
chromadb = [
    "chromadb>=0.4",
]
dev = [
    "pytest>=8.0",
    "pytest-cov",
    "ruff",
    "mypy",
]
all = [
    "testcase-selection[vectorize,selection-extras,chromadb,dev]",
]

[project.scripts]
testcase-select = "testcase_selection.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/testcase_selection"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
markers = [
    "slow: marks tests that load ML models or large datasets",
    "integration: marks integration tests requiring full pipeline",
]

[tool.ruff]
target-version = "py311"
src = ["src"]

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "SIM", "TCH"]

[tool.mypy]
python_version = "3.11"
strict = true
packages = ["testcase_selection"]
mypy_path = "src"
```

### uv Workflow

```bash
# Initial setup -- create venv and install core dependencies
uv venv
uv sync

# Install with vectorization support (Stage 1)
uv sync --extra vectorize

# Install with all optional selection strategies
uv sync --extra selection-extras

# Install everything (development)
uv sync --extra all

# Run tests
uv run pytest

# Run only unit tests (no ML model loading)
uv run pytest tests/unit/

# Run the CLI
uv run testcase-select vectorize --suite ./tests/
uv run testcase-select select --k 50 --strategy fps
uv run testcase-select execute --selection selected_tests.json --suite ./tests/

# Update lockfile after changing pyproject.toml
uv lock

# Install specific Python version (if not already available)
uv python install 3.11
```

### CI Dependency Installation by Stage

| CI Stage | Install Command | Packages Installed |
|----------|----------------|-------------------|
| Stage 1 (Vectorize) | `uv sync --extra vectorize` | robotframework, numpy, sentence-transformers (+ torch) |
| Stage 2 (Select, FPS only) | `uv sync` | robotframework, numpy (scikit-learn is a numpy transitive) |
| Stage 2 (Select, all strategies) | `uv sync --extra selection-extras` | + scikit-learn-extra, dppy, apricot-select |
| Stage 3 (Execute) | `uv sync` | robotframework, numpy |
| Development | `uv sync --extra all` | Everything |

## Consequences

### Positive

- **Fast CI builds.** uv resolves and installs dependencies 10-100x faster than pip. For a cold install of the `vectorize` group (sentence-transformers + torch + dependencies), uv typically completes in 5-15 seconds vs. 60-120 seconds with pip. This matters because Stage 1 may re-install on cache miss.

- **Deterministic lockfile.** `uv.lock` is committed to version control and ensures every CI run and every developer machine uses identical dependency versions. This eliminates "works on my machine" class issues and satisfies reproducibility requirements.

- **Minimal per-stage installs.** Optional dependency groups mean Stage 3 (execution) never installs sentence-transformers or torch. A CI job running only the execution stage installs ~5 packages instead of ~50.

- **Clean import boundaries.** The src-layout prevents accidental imports from the project root. Running `python -c "import testcase_selection"` from the project root will fail unless the package is properly installed, which catches packaging errors early.

- **Bounded context isolation.** Each sub-package has a clear responsibility boundary. The `selection/` package never imports from `execution/`. The `embedding/` package exposes a protocol (`ports.py`) so the embedding implementation can be swapped without touching downstream code. Dependencies flow inward: `pipeline/` orchestrates the other contexts but they do not depend on `pipeline/`.

- **Optional strategies degrade gracefully.** The `selection/registry.py` module registers strategies at import time. Strategies backed by optional dependencies (k-Medoids, DPP, facility location) catch `ImportError` at registration and are simply unavailable if their dependencies are not installed. FPS is always available since it depends only on numpy.

- **Python 3.11+ features available.** `tomllib` in stdlib (no `toml` package needed for config parsing), `ExceptionGroup` for aggregating errors across pipeline stages, `TaskGroup` from asyncio (if async embedding is needed later), and performance improvements (CPython 3.11 is ~25% faster than 3.10 on average).

- **Standard PEP 621 metadata.** The `pyproject.toml` uses only standard fields. The project can be built with any PEP 517 build backend (hatchling chosen for simplicity). Users who prefer pip can still install via `pip install -e ".[dev]"` -- uv is not required for consumers, only for development workflow.

- **Single entry point.** The `testcase-select` CLI command dispatches to subcommands for each pipeline stage (`vectorize`, `select`, `execute`), providing a unified interface rather than requiring users to invoke separate scripts.

### Negative

- **uv is relatively new.** While backed by Astral (the ruff team) and rapidly adopted, uv has a shorter track record than pip or Poetry. Mitigation: the project remains pip-compatible since `pyproject.toml` uses standard PEP 621 metadata. Falling back to `pip install -e ".[all]"` works without any configuration changes.

- **Lockfile format is uv-specific.** `uv.lock` is not interchangeable with Poetry's `poetry.lock` or PDM's `pdm.lock`. Teams already using Poetry or PDM for other projects will need to learn uv's workflow for this project. Mitigation: uv's CLI surface is intentionally close to pip, reducing learning curve.

- **Optional dependency sprawl.** Five optional groups add cognitive load when deciding what to install. Mitigation: the `all` group installs everything, and the CLI reports which strategies are available at runtime.

- **src-layout adds one level of nesting.** Imports during development require the package to be installed (`uv sync` or `pip install -e .`). Running `python src/testcase_selection/cli.py` directly will not work. Mitigation: `uv run` handles this transparently, and all developers use `uv sync` during setup.

### Neutral

- The `uv.lock` file should be committed to version control. It is a generated artifact but critical for reproducibility.
- The project does not use a monorepo structure. All bounded contexts live under a single installable package. If contexts diverge significantly in the future (e.g., the embedding service becomes a standalone microservice), they can be extracted into separate packages at that point.
- Test fixtures (`.robot` files, `.csv` files) live under `tests/fixtures/` rather than alongside source code, keeping the src tree clean.
