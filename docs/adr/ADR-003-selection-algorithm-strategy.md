# ADR-003: Diversity Selection Algorithm Strategy

- **Status**: Proposed
- **Date**: 2026-02-21
- **Deciders**: Architecture Team
- **Relates to**: ADR-001 (Embedding Model Selection), ADR-002 (Vector Storage)

## Context

Given N embedded test cases represented as vectors in a d-dimensional space, we need to select k maximally diverse tests for execution. The goal is to maximize the semantic coverage of the selected subset while respecting time budgets imposed by different CI/CD scenarios (PR validation, daily regression, nightly runs, pre-release).

Five candidate algorithms are available, each optimizing a different diversity objective with distinct tradeoffs in time complexity, approximation guarantees, determinism, and selection characteristics. The system must support switching between algorithms without modifying calling code, since different CI scenarios benefit from different strategies.

Additionally, three of the five algorithms require optional third-party dependencies (`sklearn-extra`, `dppy`, `apricot-select`) that should not be mandatory for basic operation.

### Problem Dimensions

1. **Objective mismatch**: Max-min dispersion (spread boundary points apart) vs. representativeness (cover every cluster) vs. probabilistic diversity (varied sampling across runs) are different optimization targets. No single algorithm is best for all scenarios.
2. **Performance constraints**: Daily CI needs sub-second selection; nightly runs can tolerate seconds. At N=5,000 and k=200, algorithm choice determines whether selection takes milliseconds or minutes.
3. **Determinism requirements**: PR validation demands reproducible selections for debugging; nightly diversity runs benefit from stochastic variation.
4. **Dependency management**: Core installations should require only `numpy` and `scikit-learn`. Heavier dependencies should load lazily only when their strategy is requested.

## Decision

### 1. Use Farthest Point Sampling (FPS) as the default algorithm

FPS (also called farthest-first traversal) provides the best overall tradeoff for the default case:

- **2-approximation guarantee** for the max-min dispersion problem (Gonzalez, 1985): the minimum pairwise distance in the selected subset is at least half the optimal.
- **O(N*k*d) time complexity** with O(N) memory beyond the embedding matrix. For N=5,000, k=200, d=384, this runs in single-digit milliseconds.
- **Deterministic** given a fixed seed (controls the initial point).
- **Outlier-first property**: boundary points are always selected first, ensuring coverage of semantic extremes in the test suite.

The algorithm iteratively selects the point maximally distant (cosine distance) from all previously selected points, maintaining a running minimum-distance vector that updates in O(N*d) per iteration.

### 2. Implement a Strategy pattern with pluggable algorithms

Define an abstract interface that all selection algorithms implement:

```
SelectionStrategy.select(embeddings: ndarray, k: int, seed: int) -> list[int]
```

Where:
- `embeddings` is an (N, d) numpy array of test case vectors
- `k` is the number of tests to select
- `seed` is the random seed for reproducibility
- Returns a list of k integer indices into the embeddings array

Each strategy is a separate class implementing this interface, registered by name in a strategy registry for lookup via configuration or CLI parameter.

### 3. Support five strategies selectable by name

| Strategy | Name | Time Complexity | Guarantee | Deterministic | Dependency | Best For |
|----------|------|-----------------|-----------|---------------|------------|----------|
| Farthest Point Sampling | `fps` | O(N*k*d) | 2-approx (max-min) | Yes | numpy, sklearn | Fast guaranteed coverage |
| FPS Multi-Start | `fps_multi` | O(N*k*d*S) where S=starts | 2-approx (best of S) | Yes | numpy, sklearn | Robust edge-case coverage |
| k-Medoids (PAM) | `kmedoids` | O(N^2*k*iter) | Local optima | No | sklearn-extra | Cluster representative selection |
| Facility Location | `facility` | O(N^2*k) | (1-1/e) ~= 0.632 | Yes | apricot-select | Ensuring no cluster unrepresented |
| k-DPP | `dpp` | O(N^3 + N*k^3) | Probabilistic | No | dppy | Varied ensemble sampling across CI runs |

A sixth algorithm, **Maximal Marginal Relevance (MMR)**, is noted but deferred. At lambda=0 or with uniform relevance, MMR reduces exactly to FPS. MMR becomes valuable only when query-biased selection is needed (e.g., selecting tests relevant to changed code areas), which is a future enhancement requiring integration with code-change analysis.

### 4. Strategy-specific design notes

**FPS** (default): Greedy farthest-first traversal. Initial point selected by seeded RNG. Marks selected points with -inf distance to prevent reselection. Uses `sklearn.metrics.pairwise.cosine_distances` for batch distance computation.

**FPS Multi-Start**: Runs FPS `S` times (default S=5) from different random starting points. Evaluates each result by the minimum pairwise cosine distance among selected points. Returns the result that maximizes this metric, mitigating sensitivity to initial-point choice.

**k-Medoids (PAM)**: Uses `sklearn_extra.cluster.KMedoids` with `metric='cosine'`, `method='pam'`, and `init='k-medoids++'`. Optimizes representativeness (minimizes average intra-cluster distance to medoid), not dispersion. Each medoid is a real test case, making interpretation natural. O(N^2*k*iter) complexity limits practical use to N < 10,000.

**Facility Location**: Uses `apricot.FacilityLocationSelection` with `metric='cosine'`. Maximizes a submodular objective ensuring every test in the full suite has a similar representative in the selected subset. The (1-1/e) approximation guarantee comes from the greedy algorithm's property on submodular functions. Subtly different from FPS: facility location prevents unrepresented clusters while FPS maximizes spread.

**k-DPP**: Constructs an L-kernel from L2-normalized embeddings: `L = X_norm @ X_norm.T`, symmetrized. Uses `dppy.finite_dpps.FiniteDPP` for exact k-DPP sampling via eigendecomposition. The O(N^3) cost makes this the slowest option, but it produces genuinely random diverse subsets where similar items are anti-correlated in the sampling distribution. Different seeds produce different valid diverse subsets, making it ideal for nightly runs where cumulative coverage across runs matters.

### 5. Lazy loading for optional dependencies

The three optional dependencies (`sklearn-extra`, `dppy`, `apricot-select`) are imported only when their corresponding strategy is requested. If a strategy is requested and its dependency is not installed, the system raises a clear error message naming the missing package and the pip install command. Core operation with `fps` and `fps_multi` requires only `numpy` and `scikit-learn`.

### 6. Strategy registry pattern

Strategies are registered by name in a dictionary-based registry. Selection by name occurs at the pipeline's Stage 2 entry point, allowing CI configuration, CLI flags, or environment variables to control algorithm choice without code changes.

```
STRATEGIES = {
    "fps": FarthestPointSampling,
    "fps_multi": FPSMultiStart,
    "kmedoids": KMedoidsSelection,
    "facility": FacilityLocationSelection,
    "dpp": DPPSelection,
}
```

### 7. Scenario-based strategy recommendations

| CI Scenario | Recommended Strategy | k Guidance | Rationale |
|-------------|---------------------|------------|-----------|
| Daily CI on main | `fps` | k = 20% of N | Fast, deterministic, good boundary coverage |
| PR validation | `fps_multi` | k = 15% of N | More robust against initial-point sensitivity |
| Nightly full diversity | `dpp` | k = 30% of N | Stochastic variation; cumulative multi-night coverage |
| Pre-release regression | `facility` | k = 50% of N | Ensures every test cluster is represented |
| Quick smoke test | `fps` | k = 10 (absolute) | The 10 most semantically distant tests |
| Strategy comparison | Fan-out parallel | k = same across all | Run multiple strategies, compare fault detection over time |

## Consequences

### Positive

- **FPS as default provides immediate value**: 2-approximation guarantee, millisecond performance, zero optional dependencies, deterministic output. Teams can adopt the tool without configuring anything.
- **Strategy pattern enables evolution**: New algorithms (e.g., MMR with code-change queries, graph-based diversity) can be added by implementing a single interface and registering a name.
- **Lazy dependency loading keeps the core lightweight**: The base installation requires only `numpy` and `scikit-learn` (~30 MB). Optional strategies add dependencies only when explicitly chosen.
- **Scenario-based recommendations reduce decision fatigue**: Teams can map their CI pipeline type directly to a strategy without understanding the algorithmic details.
- **Deterministic strategies (fps, fps_multi, facility) enable reproducible debugging**: Given the same embeddings, k, and seed, the same tests are always selected.
- **DPP enables cumulative coverage**: Over multiple nightly runs with different seeds, probabilistic diverse sampling covers more of the test space than any single deterministic selection.

### Negative

- **FPS outlier-first property can be undesirable**: If the test suite contains spurious outlier tests (e.g., one-off configuration tests), FPS always selects them, potentially wasting slots. Mitigation: use `fps_multi` or pre-filter outliers before selection.
- **k-Medoids and DPP scale poorly**: k-Medoids at O(N^2*k*iter) and DPP at O(N^3) become impractical above N=10,000. For large suites, only `fps`, `fps_multi`, and `facility` are viable. This must be documented clearly.
- **Strategy choice adds a configuration decision**: Teams must choose or accept the default. The scenario-based decision framework mitigates this, but misconfiguration is possible (e.g., using DPP for time-sensitive PR checks).
- **Optional dependencies are not validated at install time**: A user requesting `kmedoids` without `sklearn-extra` installed encounters a runtime error. Mitigation: clear error messages with exact pip install commands.
- **No query-biased selection yet**: MMR (the natural fit for "select tests relevant to changed code") is deferred. Until implemented, selection is purely diversity-driven with no relevance weighting.

### Risks

- **Semantic diversity may not equal fault-detection diversity**: The assumption that maximally diverse embeddings correspond to maximal fault-detection coverage is supported by the FAST-R literature (Cruciani et al., ICSE 2019) and the broader diversity testing field (167 papers cataloged in Elgendy, Wiley STVR, 2025), but should be validated empirically on real defect data for this project.
- **Embedding quality is the ceiling**: All strategies operate on the embedding space. If the embedding model fails to capture meaningful test-case distinctions (e.g., due to domain-specific vocabulary), no algorithm choice compensates. Embedding quality should be evaluated separately.

## References

- Gonzalez, T. (1985). "Clustering to minimize the maximum intercluster distance." *Theoretical Computer Science*, 38, 293-306.
- Cruciani, E. et al. (2019). "Scalable Approaches for Test Suite Reduction." *ICSE 2019*. (FAST-R)
- Kulesza, A. & Taskar, B. (2012). "Determinantal Point Processes for Machine Learning." *Foundations and Trends in Machine Learning*.
- Elgendy, M. (2025). "Diversity-based Testing Techniques: A Systematic Mapping Study." *Wiley STVR*.
