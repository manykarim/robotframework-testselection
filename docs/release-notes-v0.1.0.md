# robotframework-testselection v0.1.0

**Vector-based diverse test case selection for Robot Framework**

Select fewer tests, keep more coverage. `robotframework-testselection` embeds your Robot Framework test cases as semantic vectors and picks maximally diverse subsets — reducing execution time while preserving coverage breadth.

## Highlights

- **3-stage pipeline**: Vectorize → Select → Execute, each stage usable independently or as a single `testcase-select run` command
- **Semantic embeddings**: Converts test cases to 384-dim vectors using `all-MiniLM-L6-v2` (sentence-transformers), capturing test name, tags, and resolved keyword tree
- **Farthest Point Sampling (FPS)**: Greedy diversity-maximizing selection with 2-approximation guarantee for max-min dispersion. Multi-start variant included
- **Optional algorithms**: k-Medoids, Determinantal Point Process (DPP), and Facility Location via `[selection-extras]`
- **Robot Framework integration**: `PreRunModifier` for standard tests, `Listener v3` for DataDriver-generated tests — works with `robot` directly or via the CLI
- **Content-hash caching**: Skips re-vectorization when source files haven't changed
- **Tag-based filtering**: Include/exclude tests by tag before selection
- **Graceful degradation**: If any pipeline stage fails, falls back to running the full suite (exit code 2)

## Coverage Results

Tested against [robotframework-doctestlibrary](https://github.com/manykarim/robotframework-doctestlibrary) (125 test cases):

| Metric | Full Suite | Diverse Selection |
|--------|-----------|-------------------|
| Tests run | 125 | 51 (40.8%) |
| Coverage | 58.0% | 55.7% (96.0% retained) |
| Execution time | ~55.6% faster with selection |

## Installation

```bash
# Core package
pip install robotframework-testselection

# With sentence-transformers for vectorization
pip install robotframework-testselection[vectorize]

# With all optional selection algorithms
pip install robotframework-testselection[selection-extras]

# Everything (including dev tools)
pip install robotframework-testselection[all]
```

Requires Python 3.10+.

## Quick Start

```bash
# Full pipeline in one command
testcase-select run \
  --suite tests/robot/ \
  --k 20 \
  --strategy fps \
  --seed 42 \
  --output-dir ./results/

# Or stage by stage
testcase-select vectorize --suite tests/robot/ --output ./artifacts/
testcase-select select --artifacts ./artifacts/ --k 20 --strategy fps
testcase-select execute --suite tests/robot/ --selection selected_tests.json
```

## Selection Strategies

| Strategy | ID | Dependencies | Description |
|----------|----|-------------|-------------|
| Farthest Point Sampling | `fps` | *(base)* | Greedy farthest-first traversal. **Default.** |
| FPS Multi-Start | `fps_multi` | *(base)* | Multiple random starts, keeps best result |
| k-Medoids | `kmedoids` | `[selection-extras]` | PAM algorithm for medoid-based clustering |
| DPP | `dpp` | `[selection-extras]` | Determinantal Point Process sampling |
| Facility Location | `facility` | `[selection-extras]` | Submodular facility location maximization |

## Robot Framework Integration

```bash
# Use directly with robot
robot \
  --prerunmodifier TestSelection.execution.prerun_modifier.DiversePreRunModifier:selected_tests.json \
  --listener TestSelection.execution.listener.DiverseDataDriverListener:selected_tests.json \
  tests/
```

## CI/CD

Pre-built configurations included for GitHub Actions, GitLab CI, and Jenkins in `config/`.

## Links

- **Repository**: https://github.com/manykarim/robotframework-testselection
- **Documentation**: https://github.com/manykarim/robotframework-testselection#readme
- **Issues**: https://github.com/manykarim/robotframework-testselection/issues
