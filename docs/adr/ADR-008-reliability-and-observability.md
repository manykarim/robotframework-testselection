# ADR-008: Error Handling, Observability, and Reliability

- **Status**: Proposed
- **Date**: 2026-02-21
- **Authors**: Architecture Team
- **Deciders**: Engineering Team

## Context

A test selection system that fails silently or corrupts selections can be worse than running all tests. If Stage 2 silently produces an empty selection, or Stage 1 generates embeddings with the wrong dimensionality, or Stage 3 receives a selection file referencing tests that no longer exist, the result is a CI run that appears green while providing no actual coverage. This is strictly worse than not having test selection at all.

The three-stage pipeline defined in ADR-001 introduces three independent failure domains. Each stage can fail for distinct reasons -- model download timeouts in Stage 1, numerical errors in Stage 2, malformed selection files in Stage 3 -- and failures must not cascade into silent test gaps. The system must degrade gracefully, provide structured observability into every selection decision, and guarantee deterministic reproducibility so that any reported selection can be exactly reproduced for debugging.

Additionally, the pipeline operates at a system boundary: it consumes user-provided file paths, Robot Framework suites of unknown structure, and CSV data sources of varying quality. Input validation must be rigorous at these boundaries while internal code can trust validated data flowing through artifact contracts.

## Decision

### 1. Graceful Degradation: Always Fall Back to Running All Tests

The cardinal rule is: **never silently skip tests**. Any failure in the selection pipeline must result in running all tests, not running no tests.

**Stage 1 (Vectorize) failure**:
- If the embedding model fails to load (download timeout, corrupted cache, incompatible Python version), log the error and exit with code 2.
- The CI pipeline detects exit code 2 and skips Stage 2, proceeding directly to Stage 3 with no `--prerunmodifier` argument. All tests run.
- If cached artifacts exist from a previous successful Stage 1 run, the pipeline may use those instead, with a logged warning noting the cache staleness.

**Stage 2 (Select) failure**:
- If artifact loading fails (missing files, corrupt NPZ, shape mismatch), log the error and exit with code 2.
- If the selection algorithm raises an exception (numerical instability, invalid k), log the error and exit with code 2.
- The CI pipeline detects exit code 2 and runs Stage 3 without selection filtering. All tests run.

**Stage 3 (Execute) with bad selection file**:
- If `selected_tests.json` is missing, malformed, or references zero valid test names, the PreRunModifier logs a warning and does not filter any tests. All tests run.
- If some selected test names do not match any tests in the suite (e.g., tests were renamed since Stage 1 ran), log each unmatched name as a warning. The matched tests still run, and the unmatched names are recorded in the selection report.

**Exit code convention**:

| Code | Meaning |
|------|---------|
| 0 | Success -- selected tests ran and all passed |
| 1 | Test failures -- selected tests ran but some failed (standard Robot Framework behavior) |
| 2 | Selection error -- pipeline fell back to running all tests due to a selection system failure |

Exit code 2 is distinct from both success and test failure so that CI systems can distinguish "selection worked but tests failed" from "selection itself broke." A pipeline that consistently returns exit code 2 indicates a systemic issue with the selection infrastructure that needs investigation.

### 2. Artifact Contract Validation

Each stage validates the artifacts it receives before processing them. Validation failures trigger the graceful degradation described above.

**Stage 2 validates Stage 1 artifacts**:
- `embeddings.npz` exists and contains a `vectors` key with a 2D float array
- `test_manifest.json` exists and conforms to the expected JSON schema (has `tests` array, `model` string, `embedding_dim` integer, `test_count` integer)
- The number of rows in `vectors` matches `test_count` in the manifest
- The number of columns in `vectors` matches `embedding_dim` in the manifest (384 for `all-MiniLM-L6-v2`)
- Every entry in `manifest.tests` has required fields: `id`, `name`
- File integrity: optionally verify an SHA-256 hash stored alongside artifacts (written by Stage 1 as `artifacts.sha256`)

**Stage 3 validates Stage 2 artifacts**:
- `selected_tests.json` exists and is valid JSON
- The JSON object has a `selected` array with at least one entry
- Each entry in `selected` has a `name` string
- At least one selected test name matches a test in the suite being executed (if zero match, fall back to all tests with a warning)

**Schema validation example for `selected_tests.json`**:

```python
SELECTED_TESTS_SCHEMA = {
    "type": "object",
    "required": ["strategy", "k", "seed", "selected"],
    "properties": {
        "strategy": {"type": "string"},
        "k": {"type": "integer", "minimum": 1},
        "total_tests": {"type": "integer", "minimum": 1},
        "filtered_tests": {"type": "integer", "minimum": 1},
        "seed": {"type": "integer"},
        "selected": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"},
                    "id": {"type": "string"},
                    "suite": {"type": "string"},
                    "is_datadriver": {"type": "boolean"}
                }
            }
        }
    }
}
```

### 3. Deterministic Reproducibility

All selection algorithms must produce identical results given identical inputs and the same seed. This is essential for debugging ("why was test X excluded from yesterday's run?") and for CI reproducibility guarantees.

**Seeding strategy**:
- All algorithms accept a `seed` parameter (default: 42).
- Algorithms use `numpy.random.RandomState(seed)` instance-level RNG, not the global `numpy.random.seed()`. This ensures thread safety and prevents interference between concurrent pipeline runs or test processes.
- The seed is recorded in `selected_tests.json` output so that any selection can be exactly reproduced.
- The same inputs (embedding matrix + manifest) combined with the same seed and strategy must always produce the same `selected_tests.json` output, byte-for-byte (after JSON normalization).

**Reproducibility contract**:

```
identical(.robot files) + identical(seed) + identical(strategy) + identical(k)
    => identical(selected_tests.json)
```

For non-deterministic algorithms (k-DPP, k-Medoids), the `RandomState` seed makes them deterministic for a given seed. Different seeds produce different (but equally valid) selections, which is useful for nightly DPP runs that intentionally vary coverage.

### 4. Observability and Metrics

Every selection run produces structured metrics that enable monitoring selection quality over time and diagnosing issues.

**Selection diversity metrics** (computed by Stage 2, recorded in `selected_tests.json`):

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| `avg_pairwise_cosine_distance` | Mean cosine distance across all pairs in selected set | 0.3 -- 1.2 (higher = more diverse) |
| `min_pairwise_cosine_distance` | Cosine distance of the most similar pair (the bottleneck) | > 0.05 (very low indicates near-duplicates) |
| `suite_coverage` | Fraction of test suites with at least one selected test | Target > 0.8 for broad coverage |
| `tag_coverage` | Fraction of unique tags represented in selected set | Context-dependent; monitor for regressions |
| `reduction_ratio` | k / total_tests | Configured per environment; typically 0.1 -- 0.3 |

**Per-stage structured logging**:

Stage 1 (Vectorize):
- Number of `.robot` files parsed
- Number of tests discovered (standard and DataDriver separately)
- Embedding model name and dimension
- Embedding throughput: tests/second
- Wall-clock time for parsing, embedding, and artifact writing
- Cache status: hit (skipped) or miss (re-indexed), with count of changed files

Stage 2 (Select):
- Artifact load time
- Number of tests before and after tag filtering
- Strategy name, k, seed
- Selection algorithm wall-clock time
- All diversity metrics listed above
- Names of selected tests (at debug log level)

Stage 3 (Execute):
- Number of tests in selection file
- Number of tests matched in suite vs. unmatched (with unmatched names logged as warnings)
- Number of DataDriver tests handled via Listener v3
- Wall-clock time for suite modification

**Log format**: All log messages use structured key-value format for machine parsing:

```
[DIVERSE-SELECT] stage=vectorize event=parse_complete tests_found=847 robot_files=23 elapsed_ms=1200
[DIVERSE-SELECT] stage=vectorize event=embed_complete model=all-MiniLM-L6-v2 dim=384 throughput_tps=2115 elapsed_ms=400
[DIVERSE-SELECT] stage=select event=selection_complete strategy=fps k=50 total=847 filtered=623 avg_dist=0.7842 min_dist=0.1523 suite_coverage=0.91 elapsed_ms=312
[DIVERSE-SELECT] stage=execute event=filter_applied matched=48 unmatched=2 unmatched_names=["Deleted Test A","Renamed Test B"]
```

The `[DIVERSE-SELECT]` prefix makes log lines easy to filter from Robot Framework's own output. The `stage=` field enables per-stage log aggregation.

### 5. Input Validation at System Boundaries

The pipeline validates all external inputs at the point of entry. Internal data flowing through artifact contracts is trusted after contract validation.

**Stage 1 input validation**:
- Suite path exists and is a directory
- Suite path contains at least one `.robot` file (error if zero found)
- DataDriver CSV paths, if specified, exist and are readable
- DataDriver CSV files parse without errors (valid CSV structure, expected columns present)
- Model name is a recognized sentence-transformers model identifier (warn on unrecognized, proceed anyway)

**Stage 2 input validation**:
- k is a positive integer. If k > total tests after filtering, clamp to total (select all) and log a warning. Never error on a valid positive k.
- Strategy name is in the set of known strategies. Error with available options listed if unknown.
- Tag filter values, if provided, are non-empty strings

**Stage 3 input validation**:
- Selection file path exists and is a readable file
- Selection file is valid JSON (not truncated, not binary)
- Selection file conforms to the expected schema (see Section 2)

**Clamping vs. erroring**: For parameters where a reasonable default exists (e.g., k > total_tests), clamp to the safe value and log a warning. For parameters where no reasonable default exists (e.g., unknown strategy name), error immediately with actionable guidance. The principle is: never fail when the user's intent is unambiguous; always fail when it is ambiguous.

### 6. Custom Exception Hierarchy

All pipeline-specific errors inherit from a common base class. This enables catch-all handling at stage boundaries while preserving specific error context for logging.

```python
class DiverseSelectionError(Exception):
    """Base exception for the diverse test selection pipeline."""
    pass

class VectorizationError(DiverseSelectionError):
    """Raised when Stage 1 (vectorization) encounters an unrecoverable error."""
    pass

class ArtifactError(DiverseSelectionError):
    """Raised when artifact validation fails between stages."""
    def __init__(self, artifact_path, expected, actual, message=""):
        self.artifact_path = artifact_path
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Artifact validation failed for {artifact_path}: "
            f"expected {expected}, got {actual}. {message}"
        )

class SelectionError(DiverseSelectionError):
    """Raised when Stage 2 (selection) encounters an unrecoverable error."""
    pass
```

**Error handling pattern**: Each stage's `main()` function wraps its core logic in a `try/except DiverseSelectionError` block. On catch, it logs the structured error context and exits with code 2, triggering the graceful degradation path.

**Stack trace policy**: Full stack traces are logged only at `DEBUG` level. At `INFO` level (the default), errors produce a single structured log line with the error class, message, and relevant context (file path, stage, test name). This keeps CI output clean while preserving full diagnostics when `--loglevel DEBUG` is set.

### 7. Performance Monitoring

Each stage records wall-clock timing for its major operations. These timings are included in stage output (log messages and artifact metadata) to enable performance regression detection.

**Stage 1 performance metrics**:
- `parse_time_ms`: Time to parse `.robot` files into suite model
- `embed_time_ms`: Time to run the embedding model
- `embed_throughput_tps`: Tests embedded per second
- `artifact_write_time_ms`: Time to write NPZ and JSON artifacts
- `total_time_ms`: End-to-end Stage 1 time

**Stage 2 performance metrics**:
- `artifact_load_time_ms`: Time to load NPZ and JSON from disk
- `filter_time_ms`: Time for tag filtering
- `selection_time_ms`: Time for the diversity algorithm itself
- `total_time_ms`: End-to-end Stage 2 time

**Memory monitoring**: For suites exceeding 5,000 tests, Stage 2 logs peak memory usage (via `tracemalloc` or `resource.getrusage`). The pairwise distance matrix for N tests at 384 dimensions requires approximately N^2 * 4 bytes of memory (e.g., 10,000 tests = ~400 MB). This metric helps operators determine whether the suite has outgrown the current hardware allocation.

**Embedding cache hit rate**: Stage 1 tracks the ratio of skipped (cached) runs to total runs over time. A cache hit rate below 50% may indicate excessive test churn or misconfigured cache keys. This metric is useful for CI pipeline tuning but is tracked externally (e.g., via CI dashboard) rather than within the pipeline itself.

## Consequences

### Positive

- **No silent test gaps**: The fall-back-to-all-tests policy ensures that a failure in the selection system never results in reduced test coverage. The worst outcome of a pipeline failure is running all tests (slower but safe), never running fewer tests than intended.
- **Debuggable selections**: Structured logging with per-stage context and diversity metrics makes it straightforward to answer "why was test X excluded?" by inspecting Stage 2 logs and the selection file.
- **Reproducible results**: Instance-level seeding with `RandomState` guarantees that any reported selection can be exactly reproduced, which is critical for diagnosing false negatives ("did the selection miss a relevant test?").
- **Early failure detection**: Artifact contract validation catches data corruption, schema drift between stages, and stale artifacts before they cause downstream confusion. A dimension mismatch in Stage 2 produces a clear `ArtifactError` rather than a mysterious `numpy` broadcast error deep in the selection algorithm.
- **CI-friendly exit codes**: The three-code convention (0/1/2) integrates cleanly with CI systems' conditional step execution (`if: steps.select.outcome == 'failure'` in GitHub Actions) and alerting rules.
- **Performance visibility**: Per-stage timing metrics enable detection of performance regressions (e.g., embedding throughput dropping due to model updates) and capacity planning (e.g., when does the suite outgrow the current memory allocation?).

### Negative

- **Overhead of validation**: Artifact contract validation adds a small fixed cost (typically < 100ms) to each stage. For very small suites (under 50 tests) where selection itself takes < 100ms, validation represents a meaningful fraction of Stage 2 time. This is acceptable because the validation cost is absolute (does not scale with suite size) and prevents potentially costly debugging sessions.
- **Fall-back masks failures**: Running all tests on selection failure means the CI pipeline stays green even when the selection system is broken. Teams must monitor for exit code 2 and treat persistent selection failures as actionable incidents, not ignore them because "tests still pass." This requires CI alerting configuration beyond the pipeline itself.
- **Logging volume**: Structured per-stage logging produces more output than a silent pipeline. In suites with thousands of tests, debug-level logging (which includes all selected test names) can generate significant output. The default `INFO` level keeps output concise; `DEBUG` should be used only for targeted investigation.
- **Exception hierarchy maintenance**: The custom exception hierarchy adds code that must be kept in sync with the actual failure modes of each stage. New failure modes (e.g., a future Stage 2 algorithm with a new error class) require corresponding exception subclasses.
- **Seed management responsibility**: Deterministic reproducibility requires that the seed be recorded and communicated. If a team member runs a local selection without recording the seed, the result cannot be reproduced. The default seed (42) mitigates this for the common case, but ad-hoc runs with custom seeds must be logged.

## References

- ADR-001: Three-Stage Pipeline Architecture (artifact contracts and stage definitions)
- ADR-002: Embedding Model and Storage (model specifications, 384-dimension contract)
- ADR-003: Selection Algorithm Strategy (algorithm guarantees and complexity)
- ADR-004: Robot Framework Integration (PreRunModifier and Listener v3 mechanisms)
- Research: `/docs/research/compass_artifact_wf-fd8b16a0-0008-4bc3-9105-e4553444d251_text_markdown.md`
- Research: `/docs/research/multistage_pipeline_report.md`
