# ADR-007: CI/CD Integration and Caching Strategy

- **Status**: Proposed
- **Date**: 2026-02-21
- **Authors**: Architecture Team
- **Deciders**: Engineering Team

## Context

The three-stage pipeline defined in ADR-001 must integrate with real CI/CD systems -- GitHub Actions, GitLab CI, and Jenkins -- each with their own artifact passing, caching, and job coordination mechanisms. The single largest performance optimization in the pipeline is skipping Stage 1 (vectorization) when test files have not changed, which transforms a 15-30 second embedding operation into a 0-second cache hit. Getting caching right is therefore the most impactful architectural decision for day-to-day CI performance.

Each pipeline stage has distinct caching and artifact requirements:

| Stage | Caching Need | Artifact Transfer | Retention |
|-------|-------------|-------------------|-----------|
| Stage 1 (Vectorize) | Content-hash keyed; skip entirely on cache hit | Produces `embeddings.npz` + `test_manifest.json` for Stage 2 | 7 days |
| Stage 2 (Select) | No caching (ephemeral, < 2s runtime) | Produces `selected_tests.json` for Stage 3 | 1 day |
| Stage 3 (Execute) | No caching (unique per run) | Produces `output.xml`, logs, reports | 30 days |

Beyond caching, CI integration requires runtime configuration via environment variables, support for parallel strategy fan-out at Stage 2, PR annotation with selection summaries, and manual override via `workflow_dispatch` (or equivalent). Each CI system maps these requirements differently to its native primitives.

## Decision

### 1. Content-Hash Cache Keys for Vectorization

Use a composite hash of all `.robot` files and DataDriver CSV source files as the cache key for Stage 1 artifacts. This hash determines whether vectorization can be skipped entirely.

**Cache key computation**:

```bash
HASH=$(find tests/ -name "*.robot" -o -name "*.csv" | \
       sort | xargs md5sum | md5sum | cut -d' ' -f1)
```

The hash includes CSV files because DataDriver tests are derived from CSV data sources. A change to a CSV file changes the test population and must invalidate the vectorization cache.

**File hash persistence**: Stage 1 also writes a `file_hashes.json` file mapping individual file paths to their MD5 hashes. This serves as a secondary change detection mechanism within the stage itself (the `has_changes()` function in `stage1_vectorize.py`), providing a defense-in-depth check even when the CI-level cache is restored.

**Cache restoration strategy**: Use fallback/restore keys so that a partial cache hit (e.g., most files unchanged) can still provide a warm start. In practice, Stage 1's internal `has_changes()` function detects the individual changed files and only re-embeds those.

### 2. CI-System-Specific Caching Mechanisms

#### GitHub Actions

Use `actions/cache@v4` with the computed hash as the cache key and a fallback restore key:

```yaml
- name: Hash test files for cache key
  id: test-hash
  run: |
    HASH=$(find tests/ -name "*.robot" -o -name "*.csv" | \
           sort | xargs md5sum | md5sum | cut -d' ' -f1)
    echo "hash=$HASH" >> $GITHUB_OUTPUT

- name: Cache vector artifacts
  id: cache-vectors
  uses: actions/cache@v4
  with:
    path: vector_artifacts/
    key: vectors-${{ steps.test-hash.outputs.hash }}
    restore-keys: |
      vectors-
```

When `cache-hit` is `true`, the vectorization step is conditionally skipped:

```yaml
- name: Run vectorization
  if: steps.cache-vectors.outputs.cache-hit != 'true'
  run: python stage1_vectorize.py --suite ./tests/ --output ./vector_artifacts/
```

Artifact transfer between jobs uses `actions/upload-artifact@v4` and `actions/download-artifact@v4`.

#### GitLab CI

Use the `cache:key:files` directive, which natively hashes the specified file patterns:

```yaml
vectorize:
  stage: vectorize
  cache:
    key:
      files:
        - tests/**/*.robot
    paths:
      - vector_artifacts/
      - .pip-cache/
  artifacts:
    paths:
      - vector_artifacts/embeddings.npz
      - vector_artifacts/test_manifest.json
    expire_in: 7 days
```

GitLab's `artifacts` mechanism handles inter-stage transfers. The `needs` directive with `artifacts: true` provides explicit dependency declaration.

#### Jenkins

Use `stash`/`unstash` for artifact passing within a single pipeline run, and the `jobCachingPlugin` for cross-build caching of vector artifacts:

```groovy
stage('Vectorize') {
    steps {
        script {
            def testsHash = sh(
                script: 'find tests/ -name "*.robot" -exec md5sum {} \\; | sort | md5sum | cut -d" " -f1',
                returnStdout: true
            ).trim()
            def cached = fileExists("vector_artifacts/.hash_${testsHash}")
            if (!cached) {
                sh 'python stage1_vectorize.py --suite ./tests/ --output ./vector_artifacts/ --force'
                sh "touch vector_artifacts/.hash_${testsHash}"
            }
        }
    }
    post {
        success { stash name: 'vectors', includes: 'vector_artifacts/**' }
    }
}

stage('Select') {
    steps {
        unstash 'vectors'
        // ...
    }
}
```

### 3. Environment Variable Configuration

All runtime parameters are configurable via environment variables, enabling CI pipelines to override defaults without code changes:

| Variable | Purpose | Default |
|----------|---------|---------|
| `DIVERSE_K` | Number of tests to select | `50` |
| `DIVERSE_STRATEGY` | Selection algorithm (`fps`, `fps_multi`, `kmedoids`, `facility`, `dpp`) | `fps` |
| `DIVERSE_OUTPUT` | Output file path for `selected_tests.json` | `selected_tests.json` |
| `DIVERSE_SEED` | Random seed for reproducibility | `42` |

These map directly to Stage 2 CLI arguments and can be set at the pipeline, job, or step level in any CI system. For GitHub Actions, `workflow_dispatch` inputs provide a UI for manual override:

```yaml
on:
  workflow_dispatch:
    inputs:
      k:
        description: 'Number of tests to select'
        default: '50'
      strategy:
        description: 'Selection strategy'
        default: 'fps'
        type: choice
        options: [fps, fps_multi, kmedoids, facility, dpp]
```

### 4. Artifact Flow and Retention

The artifact flow across stages is:

```
Stage 1                       Stage 2                       Stage 3
(Vectorize)                   (Select)                      (Execute)
    |                             |                             |
    |-- embeddings.npz ---------> |                             |
    |-- test_manifest.json -----> |                             |
    |                             |-- selected_tests.json ----> |
    |                             |                             |-- output.xml
    |                             |                             |-- log.html
    |                             |                             |-- report.html
    |                             |                             |-- selection_report.json
```

**Retention policy**:

| Artifact | Retention | Rationale |
|----------|-----------|-----------|
| `embeddings.npz`, `test_manifest.json` | 7 days | Reusable across multiple CI runs as long as tests unchanged; 7 days covers a typical sprint |
| `selected_tests.json` | 1 day | Ephemeral; only meaningful for the pipeline run that produced it |
| `output.xml`, logs, reports | 30 days | Test results need longer retention for trend analysis and failure investigation |

**Transfer overhead**: Artifact upload and download adds approximately 1-3 seconds per stage boundary, totaling 3-5 seconds across the pipeline. This is negligible relative to test execution time and is a fixed cost regardless of suite size.

### 5. Parallel Strategy Fan-Out

Stage 2 can fan out into parallel jobs running different selection strategies. This is useful for empirical comparison of strategies or for combining multiple perspectives on diversity.

**GitHub Actions pattern** using reusable workflows:

```yaml
select-fps:
  name: "Select (FPS)"
  needs: vectorize
  uses: ./.github/workflows/select-reusable.yml
  with:
    strategy: fps
    k: 50
    artifact-name: selection-fps

select-kmedoids:
  name: "Select (k-Medoids)"
  needs: vectorize
  uses: ./.github/workflows/select-reusable.yml
  with:
    strategy: kmedoids
    k: 50
    artifact-name: selection-kmedoids

select-facility:
  name: "Select (Facility Location)"
  needs: vectorize
  uses: ./.github/workflows/select-reusable.yml
  with:
    strategy: facility
    k: 50
    artifact-name: selection-facility
```

**Merge strategies** for combining parallel selections:

| Merge | Behavior | Use Case |
|-------|----------|----------|
| Union | Run tests selected by ANY strategy | Maximum coverage, larger test set |
| Intersection | Run tests selected by ALL strategies | Consensus diverse core, smallest set |
| Weighted | Score each test by how many strategies selected it, pick top-k | Balanced approach |

The merge step downloads all selection artifacts, combines them, and produces a single `selected_tests.json` for Stage 3.

### 6. PR Annotations

On pull request events, the pipeline annotates the PR with a selection summary using `actions/github-script` (GitHub Actions) or merge request comments (GitLab CI):

```yaml
- name: Annotate PR with selection summary
  if: github.event_name == 'pull_request'
  uses: actions/github-script@v7
  with:
    script: |
      const fs = require('fs');
      const sel = JSON.parse(fs.readFileSync('selected_tests.json'));
      const reduction = (100*(1-sel.k/sel.total_tests)).toFixed(1);
      const body = [
        '### Diverse Test Selection',
        '| Metric | Value |',
        '|--------|-------|',
        `| Strategy | \`${sel.strategy}\` |`,
        `| Selected | ${sel.k} / ${sel.total_tests} |`,
        `| Reduction | ${reduction}% |`,
        `| Seed | ${sel.seed} |`,
      ].join('\n');
      github.rest.issues.createComment({
        issue_number: context.issue.number,
        owner: context.repo.owner,
        repo: context.repo.repo,
        body: body
      });
```

This provides immediate visibility into what percentage of the test suite was run for a given PR, aiding review confidence decisions.

### 7. Timing Benchmarks

Expected pipeline overhead for a suite of approximately 2,000 test cases:

| Operation | Cold (no cache) | Warm (cached) |
|-----------|-----------------|---------------|
| Stage 1: Vectorize | 15-30s | 0s (cache hit) |
| Stage 2: Select FPS | 0.3s | 0.3s |
| Stage 2: Select k-Medoids | 2-5s | 2-5s |
| Stage 2: Select DPP | 8-15s | 8-15s |
| Artifact transfer (total) | 3-5s | 3-5s |
| **Total CI overhead (FPS, cached)** | **18-35s** | **3-5s** |

In steady-state CI -- where code changes without test file changes are the norm -- the pipeline adds approximately 3-5 seconds of overhead before test execution. This is the expected common case.

### 8. Complete CI Configurations

Three reference pipeline configurations are provided as canonical implementations:

- **GitHub Actions** (`docs/ci/github-actions.yml`): Three-job pipeline with `actions/cache` for vectorization, `actions/upload-artifact`/`actions/download-artifact` for inter-job transfers, `actions/github-script` for PR annotations, and `workflow_dispatch` for manual parameter override.

- **GitLab CI** (`docs/ci/gitlab-ci.yml`): Three-stage pipeline with `cache:key:files` for vectorization caching, `artifacts` with `expire_in` for inter-stage transfers, and `reports:junit` for test result integration.

- **Jenkins** (`docs/ci/Jenkinsfile`): Declarative pipeline with `stash`/`unstash` for intra-pipeline artifact passing, file-hash sentinel files for cross-build caching, `robot` post-build plugin for test result visualization, and pipeline `parameters` for manual override.

## Consequences

### Positive

- **Near-zero overhead in steady state**: Content-hash caching eliminates the most expensive stage (vectorization) on the vast majority of CI runs where test files have not changed. The pipeline adds only 3-5 seconds of artifact transfer and selection overhead.
- **CI-native integration**: Each CI system uses its own native caching and artifact mechanisms rather than custom infrastructure. No external cache service, database, or shared filesystem is required.
- **Manual override without code changes**: Environment variables and `workflow_dispatch` inputs allow engineers to adjust k, strategy, and seed per pipeline run without modifying pipeline configuration or application code.
- **Parallel strategy evaluation**: The fan-out pattern at Stage 2 enables empirical comparison of selection strategies over time, helping teams converge on the best strategy for their test suite without committing to one upfront.
- **PR visibility**: Automated annotations give reviewers immediate context about test coverage reduction, supporting informed merge decisions.
- **Portable across CI systems**: The artifact contracts (NPZ + JSON files) are CI-system-agnostic. The three reference configurations demonstrate that the same pipeline architecture maps cleanly to GitHub Actions, GitLab CI, and Jenkins with only syntactic differences.

### Negative

- **Cache invalidation granularity**: The composite hash of all `.robot` and `.csv` files means that changing a single file invalidates the entire vectorization cache, forcing a full re-embed. For very large test suites (10,000+ tests), this could mean occasional 30-60 second cache misses even for small changes. Stage 1's internal `file_hashes.json` mechanism provides per-file change detection for incremental re-embedding, but the CI-level cache key does not distinguish between "one file changed" and "all files changed." A future optimization could use a two-tier cache strategy with per-file granularity.
- **CI configuration maintenance**: Three reference configurations must be maintained in parallel. Changes to artifact contracts, stage arguments, or environment variables must be reflected across all three. This is mitigated by keeping configurations as close to the reference implementations as possible and by treating the artifact contract (defined in ADR-001) as the single source of truth.
- **Parallel fan-out cost**: Running multiple selection strategies in parallel multiplies Stage 2 job count. While each job is lightweight (< 5s), CI systems may impose job limits or billing implications. The fan-out pattern should be reserved for nightly or pre-release pipelines, not every PR.
- **PR annotation noise**: Automated PR comments on every push can create noise in pull request threads. Consider using `if: github.event.action == 'opened'` or a find-and-update pattern to limit annotations to one per PR rather than one per push.

## References

- ADR-001: Three-Stage Pipeline Architecture
- Research: `/docs/research/compass_artifact_wf-fd8b16a0-0008-4bc3-9105-e4553444d251_text_markdown.md`
- Research: `/docs/research/multistage_pipeline_report.md`
- GitHub Actions Cache: https://github.com/actions/cache
- GitLab CI Caching: https://docs.gitlab.com/ee/ci/caching/
