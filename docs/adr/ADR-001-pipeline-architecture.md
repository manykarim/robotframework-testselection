# ADR-001: Three-Stage Pipeline Architecture

- **Status**: Proposed
- **Date**: 2026-02-21
- **Authors**: Architecture Team
- **Deciders**: Engineering Team

## Context

We are building a system that parses Robot Framework test suites, embeds them as vectors using transformer-based NLP models, selects maximally diverse subsets via dispersion algorithms, and executes the selected tests. The goal is to reduce CI execution time while maintaining broad test coverage without requiring historical execution data or coverage instrumentation.

A critical architectural observation is that the three core operations -- vectorization, selection, and execution -- have **fundamentally different lifecycle requirements**. Vectorization is triggered by source file changes and carries heavy ML dependencies. Selection is parameterized per CI run and requires only numerical computation. Execution depends on the full Robot Framework runtime and test infrastructure. Combining these into a single monolithic process wastes time re-embedding unchanged tests on every run, makes debugging harder (which stage failed?), prevents independent scaling, and couples unrelated dependency trees.

### Lifecycle Comparison

| Concern | Stage 1: Vectorize | Stage 2: Select | Stage 3: Execute |
|---------|-------------------|-----------------|------------------|
| **Trigger** | Test file changes (.robot, CSV) | Every CI run | Every CI run |
| **Duration** | 10-60s (depends on suite size) | < 2s | Minutes to hours |
| **Dependencies** | `sentence-transformers`, `robot.api`, `numpy` | `numpy`, `scikit-learn` | Robot Framework + test libraries |
| **Compute** | CPU (embedding model) | CPU (lightweight) | Test infrastructure |
| **Artifacts produced** | `embeddings.npz`, `test_manifest.json` | `selected_tests.json` | `output.xml`, logs, reports |
| **Cache lifetime** | Until test files change | Per-run (ephemeral) | Per-run (ephemeral) |
| **Failure impact** | Blocks selection (can fall back to cached) | Blocks execution (can fall back to run-all) | End of pipeline |

This separation aligns with how CI/CD systems natively operate: jobs produce artifacts, downstream jobs consume them, and caching is keyed on content hashes.

## Decision

Adopt a **three-stage pipeline architecture** with well-defined artifact contracts between stages:

```
Stage 1 (Vectorize) --> Stage 2 (Select) --> Stage 3 (Execute)
```

Each stage is an independent, self-contained process with its own dependencies, inputs, and outputs. Stages communicate exclusively through file-based artifact contracts.

### Stage 1: Vectorize (The Indexer)

**Purpose**: Parse all `.robot` files, resolve keyword trees, convert tests to natural language text, embed with `all-MiniLM-L6-v2`, and produce a portable vector store snapshot.

**Trigger**: Runs only when `.robot` files or DataDriver CSV sources change, detected via content hashing. Skips entirely on cache hit.

**Dependencies**: `sentence-transformers`, `robot.api`, `numpy`

**Process**:
1. Compute MD5 hashes of all `.robot` files; compare against stored hashes
2. If no changes detected and not forced, skip (exit with cached artifacts)
3. Parse test suite via `TestSuite.from_file_system()`
4. Build keyword map, resolve keyword trees to configurable depth
5. Convert each test to embeddable natural language text (test name, tags, keyword names, semantic arguments; excluding DOM locators, variable placeholders, XPaths)
6. Pre-read DataDriver CSV sources to generate test descriptions for data-driven tests
7. Batch embed all text descriptions using `all-MiniLM-L6-v2` (384-dimensional, 22M parameters, runs on CPU)
8. Write artifacts to output directory; update file hash cache

**Output artifacts** (consumed by Stage 2):

| File | Format | Contents |
|------|--------|----------|
| `embeddings.npz` | NumPy compressed archive | `vectors` (N x 384 float32), `ids` (N strings) |
| `test_manifest.json` | JSON | Test catalog: names, tags, suites, DataDriver flags, model metadata |
| `file_hashes.json` | JSON | Content hashes for change detection |

Artifacts are portable, self-contained, and small -- typically under 5 MB for 5,000 tests.

### Stage 2: Select (The Selector)

**Purpose**: Load vector artifacts, optionally filter by tags, apply a diversity selection algorithm, and output a selection file.

**Trigger**: Every CI run. Selection parameters (k, strategy, tag filters) can vary per run.

**Dependencies**: `numpy`, `scikit-learn` (no ML model loading, no Robot Framework)

**Process**:
1. Load `embeddings.npz` and `test_manifest.json` from Stage 1
2. Apply optional tag-based pre-filtering (include/exclude)
3. Optionally filter out DataDriver tests
4. Clamp k to filtered test count
5. Run the selected diversity algorithm on the filtered embedding matrix
6. Write selection output with metadata and diversity metrics

**Supported selection algorithms**:

| Algorithm | Time Complexity | Guarantee | Deterministic | Best For |
|-----------|----------------|-----------|---------------|----------|
| FPS (farthest point sampling) | O(N*k*d) | 2-approximation (max-min dispersion) | Yes (with seed) | Fast, guaranteed boundary coverage |
| FPS multi-start | O(N*k*d*starts) | Best of multiple 2-approx | Yes (with seed) | More robust FPS |
| k-Medoids (PAM) | O(N^2*k*iter) | Local optima | No | Cluster representativeness |
| Facility location | O(N^2*k) | (1-1/e) approx ~0.632 | Yes | No cluster unrepresented |
| k-DPP | O(N^3 + N*k^3) | Probabilistic | No | Varied ensemble sampling across runs |

**Default**: FPS with seed=42 for deterministic, reproducible selections.

**Output artifact** (consumed by Stage 3):

| File | Format | Contents |
|------|--------|----------|
| `selected_tests.json` | JSON | Strategy metadata, k, seed, list of selected test names/IDs with suite and DataDriver flags |

### Stage 3: Execute (The Runner)

**Purpose**: Run only the selected tests using Robot Framework's native extension points.

**Trigger**: Every CI run, after Stage 2 completes.

**Dependencies**: `robotframework`, test-specific libraries (e.g., `robotframework-seleniumlibrary`, `robotframework-datadriver`)

**Process**:
1. Load `selected_tests.json` from Stage 2
2. Detect whether DataDriver tests are in the selection
3. For standard tests: apply `DiversePreRunModifier` (a `SuiteVisitor` that filters `suite.tests` by selected names)
4. For DataDriver tests: additionally apply `DiverseDataDriverListener` (Listener v3 with `ROBOT_LISTENER_PRIORITY = 50`, running after DataDriver's `start_suite` generates tests)
5. Execute via `robot.run_cli()` or direct `robot` CLI invocation
6. Write selection coverage report alongside standard Robot Framework outputs

**Output artifacts**:

| File | Format | Contents |
|------|--------|----------|
| `output.xml` | Robot Framework XML | Test execution results |
| `log.html` | HTML | Detailed execution log |
| `report.html` | HTML | Summary report |
| `selection_report.json` | JSON | Selection metadata + return code |

### Artifact Contract Summary

```
Stage 1                   Stage 2                   Stage 3
(Vectorize)               (Select)                  (Execute)
    |                         |                         |
    |-- embeddings.npz ------>|                         |
    |-- test_manifest.json -->|                         |
    |                         |-- selected_tests.json ->|
    |                         |                         |-- output.xml
    |                         |                         |-- log.html
    |                         |                         |-- report.html
    |                         |                         |-- selection_report.json
```

### Dependency Isolation

Each stage has a minimal, non-overlapping dependency set:

| Stage | Python Packages | Approximate Install Size |
|-------|----------------|--------------------------|
| Stage 1 | `sentence-transformers`, `torch`, `robot.api`, `numpy` | ~1.5 GB (includes PyTorch) |
| Stage 2 | `numpy`, `scikit-learn` | ~80 MB |
| Stage 3 | `robotframework`, test-specific libraries | Varies |

This isolation means Stage 2 never needs to load an ML model, Stage 3 never needs `sentence-transformers`, and CI environments can use different Docker images or virtual environments per stage.

### Performance Characteristics

For a suite of approximately 2,000 test cases with `all-MiniLM-L6-v2`:

| Stage | Cold (no cache) | Warm (cached) |
|-------|-----------------|---------------|
| Stage 1: Vectorize | 15-30s (model load + embed) | 0s (cache hit, skipped entirely) |
| Stage 2: Select (FPS, k=100) | 0.3s | 0.3s |
| Stage 2: Select (k-Medoids, k=100) | 2-5s | 2-5s |
| Stage 2: Select (DPP, k=100) | 8-15s | 8-15s |
| Stage 3: Execute | Depends on test suite | Depends on test suite |
| Artifact transfer overhead | 3-5s total (across all stages) | 3-5s total |

Conditional vectorization is the single largest performance optimization. By hashing `.robot` file contents and comparing against stored hashes, Stage 1 skips entirely when tests have not changed. In steady-state CI (code changes without test changes), the pipeline adds only ~4s overhead (artifact transfer + selection) before test execution.

### Stability and Reproducibility

- **Deterministic selections**: FPS with a fixed seed produces identical selections given identical inputs. The seed is recorded in `selected_tests.json`.
- **Portable artifacts**: NPZ and JSON are platform-independent formats. Artifacts produced on one machine can be consumed on another.
- **Reproducible builds**: Content-hash caching ensures that the same test files always produce the same embeddings, regardless of when or where Stage 1 runs.
- **Auditability**: `selected_tests.json` records the strategy, k, seed, total/filtered counts, and diversity metrics, making every selection fully traceable.

### Graceful Degradation

- **Stage 1 failure**: If vectorization fails (e.g., model download issue), the pipeline can fall back to cached artifacts from a previous successful run. If no cache exists, the pipeline falls back to running all tests.
- **Stage 2 failure**: If selection fails, the pipeline falls back to running all tests (a safe default that sacrifices speed but not correctness).
- **Stage 3 failure**: Standard Robot Framework failure handling applies. The selection report records the return code for downstream analysis.

### CI/CD Integration

The three-stage design maps directly to CI/CD pipeline primitives:

- **GitHub Actions**: Three jobs with `actions/upload-artifact` and `actions/download-artifact` for artifact passing, `actions/cache` for vectorization caching keyed on test file hashes.
- **GitLab CI**: Three stages with `artifacts:paths` for inter-stage transfer, `cache:key:files` for vectorization caching.
- **Jenkins**: Three stages with `stash`/`unstash` for artifact passing within a pipeline, `jobCachingPlugin` for cross-build caching.

Each CI system's native artifact and caching mechanisms align with the pipeline's artifact contracts without requiring custom infrastructure.

## Consequences

### Positive

- **Independent scaling**: Each stage can run on different hardware. Stage 1 benefits from faster CPUs for embedding; Stage 3 may need browser infrastructure for Selenium tests.
- **Caching eliminates redundant work**: Conditional vectorization means the most expensive stage (15-30s) runs only when tests actually change, which in practice is a small fraction of CI runs.
- **Failure isolation**: A Stage 1 failure (e.g., model download timeout) does not crash test execution. Graceful degradation to cached artifacts or run-all-tests preserves pipeline reliability.
- **CI-native artifact flow**: The artifact contract (NPZ + JSON files) maps directly to how CI systems transfer data between jobs. No custom infrastructure, databases, or services required.
- **Independent testability**: Each stage can be unit-tested and integration-tested in isolation. Stage 2 can be tested with synthetic embedding matrices. Stage 3 can be tested with hand-crafted selection files.
- **Strategy flexibility**: Different CI triggers can use different selection strategies (FPS for PRs, DPP for nightly, facility location for pre-release) without changing any infrastructure -- only Stage 2 parameters change.
- **Replaceability**: Any stage can be swapped independently. Stage 1 could switch from `all-MiniLM-L6-v2` to a fine-tuned model. Stage 2 could add new algorithms. Stage 3 could integrate with a different test runner. As long as artifact contracts are maintained, other stages are unaffected.
- **Parallel strategy comparison**: Stage 2 can fan out into parallel jobs running different strategies, with a merge step that unions or intersects selections, enabling empirical strategy comparison over time.

### Negative

- **Artifact transfer overhead**: Passing artifacts between CI jobs adds approximately 3-5 seconds of total overhead (upload + download across stages). This is negligible compared to test execution time but is a fixed cost on every pipeline run.
- **Contract maintenance burden**: The artifact contracts (`embeddings.npz` schema, `test_manifest.json` schema, `selected_tests.json` schema) must be kept in sync across stages. A breaking change to the NPZ format in Stage 1 requires a corresponding update in Stage 2. Versioning the contracts (e.g., a `version` field in manifest JSON) mitigates but does not eliminate this.
- **Slightly more complex than a single script**: A monolithic script that embeds, selects, and runs in one process is conceptually simpler and has zero artifact transfer overhead. For very small test suites (under 100 tests) where vectorization takes under 5 seconds, the three-stage separation may be over-engineering. The pipeline design pays off at scale (hundreds to thousands of tests) and in CI environments where caching and failure isolation matter.
- **Debugging across stages**: When end-to-end behavior is unexpected, the developer must inspect artifacts at each stage boundary to locate the issue. This is mitigated by the human-readable `test_manifest.json` and `selected_tests.json` files, and by each stage producing clear log output.
- **Initial setup complexity**: Configuring three CI jobs with artifact passing requires more pipeline YAML than a single job. However, this is a one-time setup cost, and CI templates can be provided for GitHub Actions, GitLab CI, and Jenkins.

## References

- Gonzalez, T. (1985). Clustering to minimize the maximum intercluster distance. *Theoretical Computer Science*, 38, 293-306.
- Cruciani et al. (2019). Scalable Approaches for Test Suite Reduction. *ICSE 2019*.
- Elgendy (2025). Systematic mapping study of diversity-based testing techniques. *Wiley STVR*.
- Research: `/docs/research/compass_artifact_wf-fd8b16a0-0008-4bc3-9105-e4553444d251_text_markdown.md`
- Research: `/docs/research/multistage_pipeline_report.md`
