# Multistage Pipeline Architecture for Vector-Based Diverse Test Selection

## Architecture Overview: Why Separate Stages?

The core insight for production deployment is that **vectorization, selection, and execution have fundamentally different lifecycle requirements**:

| Concern | Vectorization | Selection | Execution |
|---------|--------------|-----------|-----------|
| **Trigger** | Test file changes | Every CI run | Every CI run |
| **Duration** | 10–60s (depends on suite size) | < 2s | Minutes to hours |
| **Dependencies** | `sentence-transformers`, `chromadb`, `robot.api` | `numpy`, `scikit-learn` | Robot Framework + test libs |
| **Compute** | CPU (embedding model) | CPU (lightweight) | Test infrastructure |
| **Artifacts produced** | Vector store snapshot | `selected_tests.json` | `output.xml`, logs |
| **Cache lifetime** | Until test files change | Per-run (ephemeral) | Per-run (ephemeral) |
| **Failure impact** | Blocks selection | Blocks execution | End of pipeline |

Combining these into a single process wastes time re-embedding unchanged tests on every run, makes debugging harder (which stage failed?), and prevents parallelizing selection strategies.

---

## Stage 1: Vectorization (The Indexer)

This stage parses all `.robot` files, resolves keyword trees, embeds them, and produces a **portable vector store snapshot**. It runs only when test content changes.

### Implementation: `stage1_vectorize.py`

```python
#!/usr/bin/env python3
"""Stage 1: Parse Robot Framework tests and produce vector embeddings.

Produces two artifacts:
  - embeddings.npz: numpy archive with vectors + metadata
  - test_manifest.json: human-readable test catalog with names, tags, suites

Trigger: Run when .robot files change (detected via content hashing).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from robot.api import TestSuite
from sentence_transformers import SentenceTransformer


def build_keyword_map(suite):
    """Recursively collect all user keywords from suite and resources."""
    kw_map = {}
    if hasattr(suite, 'resource') and suite.resource:
        for uk in suite.resource.keywords:
            kw_map[uk.name.lower().replace(' ', '_')] = uk
    for child in suite.suites:
        kw_map.update(build_keyword_map(child))
    return kw_map


def resolve_keyword_tree(kw_name, kw_args, kw_map, depth=0, max_depth=10):
    """Recursively resolve a keyword call into its full sub-keyword tree."""
    result = {"keyword": kw_name, "args": list(kw_args), "children": []}
    if depth >= max_depth:
        return result
    normalized = kw_name.lower().replace(' ', '_')
    if normalized in kw_map:
        uk = kw_map[normalized]
        for item in uk.body:
            if hasattr(item, 'name') and item.name:
                child = resolve_keyword_tree(
                    item.name, item.args, kw_map, depth + 1
                )
                result["children"].append(child)
    return result


def flatten_tree(node):
    """Convert keyword tree node into natural language for embedding."""
    kw = node["keyword"].replace('_', ' ')
    # Filter out DOM locators and variable placeholders - they're noise
    semantic_args = [
        a for a in node["args"]
        if not any(str(a).startswith(p) for p in [
            'id:', 'css:', 'xpath:', '//', '${', '@{', '%{', '&{'
        ])
    ]
    text = kw
    if semantic_args:
        text += f" with {', '.join(str(a) for a in semantic_args)}"
    children_text = " ".join(flatten_tree(c) for c in node["children"])
    return f"{text} {children_text}".strip()


def test_to_text(test, kw_map, resolve_depth=0):
    """Convert a test case to an embeddable text description.
    
    Args:
        test: Robot Framework test case object
        kw_map: Dictionary of user keywords for resolution
        resolve_depth: Max depth for keyword resolution (0 = top-level only)
    """
    parts = [f"Test: {test.name}."]
    if test.tags:
        parts.append(f"Tags: {', '.join(str(t) for t in test.tags)}.")
    
    for item in test.body:
        if hasattr(item, 'name') and item.name:
            if resolve_depth > 0:
                tree = resolve_keyword_tree(
                    item.name, item.args, kw_map, max_depth=resolve_depth
                )
                parts.append(flatten_tree(tree))
            else:
                kw_text = item.name.replace('_', ' ')
                semantic_args = [
                    str(a) for a in item.args
                    if not any(str(a).startswith(p) for p in [
                        'id:', 'css:', 'xpath:', '//', '${'
                    ])
                ]
                if semantic_args:
                    kw_text += f" with {', '.join(semantic_args)}"
                parts.append(kw_text)
    
    return " ".join(parts)


def read_datadriver_csv(csv_path, template_name, delimiter=','):
    """Pre-read DataDriver CSV to generate test descriptions.
    
    DataDriver generates tests dynamically at runtime, so we must
    read the data source directly to embed those tests ahead of time.
    """
    import csv
    tests = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            test_name = row.get('*** Test Cases ***', '').strip()
            if not test_name or test_name.startswith('#'):
                continue
            args = {k: v for k, v in row.items()
                    if k.startswith('${') and v}
            description = f"Template: {template_name}. Test: {test_name}. "
            description += " ".join(f"{k}={v}" for k, v in args.items())
            tests.append({
                "name": test_name,
                "description": description,
                "source": str(csv_path),
                "is_datadriver": True
            })
    return tests


def collect_all_tests(suite, kw_map, resolve_depth=0):
    """Recursively collect all tests from a suite hierarchy."""
    tests_data = []
    for test in suite.tests:
        text = test_to_text(test, kw_map, resolve_depth)
        test_id = hashlib.md5(
            f"{suite.source}::{test.name}".encode()
        ).hexdigest()
        tests_data.append({
            "id": test_id,
            "name": test.name,
            "text": text,
            "tags": [str(t) for t in test.tags],
            "suite": str(suite.source or suite.name),
            "suite_name": suite.name,
            "is_datadriver": False
        })
    for child in suite.suites:
        tests_data.extend(collect_all_tests(child, kw_map, resolve_depth))
    return tests_data


def compute_file_hashes(suite_path):
    """Hash all .robot files to detect changes."""
    hashes = {}
    for p in Path(suite_path).rglob("*.robot"):
        hashes[str(p)] = hashlib.md5(p.read_bytes()).hexdigest()
    return hashes


def has_changes(suite_path, hash_file):
    """Check if any .robot files changed since last indexing."""
    current = compute_file_hashes(suite_path)
    if not Path(hash_file).exists():
        return True, current
    previous = json.loads(Path(hash_file).read_text())
    return current != previous, current


def vectorize(suite_path, output_dir, model_name="all-MiniLM-L6-v2",
              resolve_depth=0, datadriver_csvs=None, force=False):
    """Main vectorization pipeline.
    
    Args:
        suite_path: Path to Robot Framework test directory
        output_dir: Where to write artifacts
        model_name: Sentence transformer model name
        resolve_depth: How deep to resolve keyword trees (0=top-level)
        datadriver_csvs: List of dicts with {path, template, delimiter}
        force: Re-index even if no changes detected
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hash_file = output_dir / "file_hashes.json"
    
    # Check for changes (skip if nothing changed)
    changed, current_hashes = has_changes(suite_path, hash_file)
    if not changed and not force:
        print("SKIP: No .robot files changed since last indexing")
        print(f"  Existing artifacts in {output_dir}")
        return False
    
    # Parse suite
    print(f"Parsing test suite from {suite_path}...")
    suite = TestSuite.from_file_system(suite_path)
    kw_map = build_keyword_map(suite)
    
    # Collect standard tests
    tests_data = collect_all_tests(suite, kw_map, resolve_depth)
    print(f"  Found {len(tests_data)} standard test cases")
    
    # Collect DataDriver tests from CSV sources
    if datadriver_csvs:
        for dd_config in datadriver_csvs:
            dd_tests = read_datadriver_csv(
                dd_config["path"],
                dd_config["template"],
                dd_config.get("delimiter", ",")
            )
            for dt in dd_tests:
                dt["id"] = hashlib.md5(
                    f"dd::{dt['source']}::{dt['name']}".encode()
                ).hexdigest()
                dt["tags"] = []
            tests_data.extend(dd_tests)
            print(f"  Found {len(dd_tests)} DataDriver tests from {dd_config['path']}")
    
    if not tests_data:
        print("ERROR: No tests found!")
        sys.exit(1)
    
    # Embed
    print(f"Embedding {len(tests_data)} tests with {model_name}...")
    model = SentenceTransformer(model_name)
    texts = [t.get("text") or t.get("description", "") for t in tests_data]
    embeddings = model.encode(texts, show_progress_bar=True,
                              normalize_embeddings=True)
    
    # Save artifacts
    # 1. Numpy archive with embeddings
    np.savez_compressed(
        output_dir / "embeddings.npz",
        vectors=embeddings,
        ids=np.array([t["id"] for t in tests_data]),
    )
    
    # 2. Test manifest (JSON) with all metadata
    manifest = {
        "model": model_name,
        "embedding_dim": embeddings.shape[1],
        "test_count": len(tests_data),
        "resolve_depth": resolve_depth,
        "tests": [
            {
                "id": t["id"],
                "name": t["name"],
                "tags": t.get("tags", []),
                "suite": t.get("suite", ""),
                "suite_name": t.get("suite_name", ""),
                "is_datadriver": t.get("is_datadriver", False),
            }
            for t in tests_data
        ]
    }
    (output_dir / "test_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    
    # 3. Save file hashes for change detection
    Path(hash_file).write_text(json.dumps(current_hashes))
    
    print(f"Artifacts written to {output_dir}/")
    print(f"  embeddings.npz: {embeddings.shape}")
    print(f"  test_manifest.json: {len(tests_data)} tests")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Vectorize RF tests")
    parser.add_argument("--suite", required=True, help="Path to test suite")
    parser.add_argument("--output", default="./vector_artifacts",
                        help="Output directory for artifacts")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--resolve-depth", type=int, default=0,
                        help="Keyword tree resolution depth (0=top-level)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-indexing even without changes")
    parser.add_argument("--datadriver-csv", action="append", nargs=3,
                        metavar=("CSV_PATH", "TEMPLATE_NAME", "DELIMITER"),
                        help="DataDriver CSV source (repeatable)")
    args = parser.parse_args()
    
    dd_csvs = None
    if args.datadriver_csv:
        dd_csvs = [
            {"path": c[0], "template": c[1], "delimiter": c[2]}
            for c in args.datadriver_csv
        ]
    
    vectorize(
        args.suite, args.output, args.model,
        args.resolve_depth, dd_csvs, args.force
    )
```

### Artifact Contract

Stage 1 produces two files consumed by Stage 2:

| File | Format | Contents |
|------|--------|----------|
| `embeddings.npz` | NumPy compressed archive | `vectors` (N×384 float32), `ids` (N strings) |
| `test_manifest.json` | JSON | Test metadata: names, tags, suites, DataDriver flags |

These are portable, self-contained, and small — typically under 5 MB for 5,000 tests.

---

## Stage 2: Selection (The Selector)

This stage loads the vector artifacts, applies a diversity algorithm, and outputs a selection file. It's fast (< 2 seconds) and can be parameterized differently per CI run.

### Implementation: `stage2_select.py`

```python
#!/usr/bin/env python3
"""Stage 2: Select maximally diverse test subset from vector artifacts.

Consumes: embeddings.npz, test_manifest.json (from Stage 1)
Produces: selected_tests.json (consumed by Stage 3)

Supports multiple selection strategies and tag-based pre-filtering.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_distances


# ─── Selection Algorithms ───────────────────────────────────────────

def farthest_point_sampling(X, k, seed=42):
    """Greedy farthest-first traversal for maximum dispersion.
    
    Guarantees: 2-approximation for max-min dispersion (Gonzalez, 1985).
    Complexity: O(N·k·d) time, O(N) memory.
    """
    N = X.shape[0]
    rng = np.random.RandomState(seed)
    initial = rng.randint(N)
    
    selected = [initial]
    # Track minimum distance from each point to any selected point
    min_distances = cosine_distances(X[initial:initial+1], X)[0]
    min_distances[initial] = -np.inf
    
    for _ in range(k - 1):
        next_idx = int(np.argmax(min_distances))
        selected.append(next_idx)
        new_dists = cosine_distances(X[next_idx:next_idx+1], X)[0]
        min_distances = np.minimum(min_distances, new_dists)
        min_distances[next_idx] = -np.inf
    
    return selected


def fps_multi_start(X, k, n_starts=5, seed=42):
    """Run FPS from multiple starting points, keep best result.
    
    'Best' = maximizes the minimum pairwise distance in selected set.
    Mitigates sensitivity to initial point choice.
    """
    rng = np.random.RandomState(seed)
    best_selected = None
    best_min_dist = -1
    
    for i in range(n_starts):
        start = rng.randint(X.shape[0])
        selected = farthest_point_sampling(X, k, seed=seed + i)
        # Evaluate: minimum pairwise distance among selected
        sel_vectors = X[selected]
        pairwise = cosine_distances(sel_vectors, sel_vectors)
        np.fill_diagonal(pairwise, np.inf)
        min_dist = pairwise.min()
        
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_selected = selected
    
    return best_selected


def kmedoids_selection(X, k, seed=42):
    """Select cluster representatives via k-medoids (PAM algorithm).
    
    Optimizes representativeness (every test is close to a medoid),
    not dispersion. Best when you want cluster coverage.
    """
    try:
        from sklearn_extra.cluster import KMedoids
    except ImportError:
        print("ERROR: sklearn-extra required for k-medoids")
        print("  pip install scikit-learn-extra")
        sys.exit(1)
    
    kmed = KMedoids(
        n_clusters=k, metric='cosine', method='pam',
        init='k-medoids++', random_state=seed, max_iter=300
    )
    kmed.fit(X)
    return list(kmed.medoid_indices_)


def facility_location_selection(X, k):
    """Submodular facility location for representative selection.
    
    Guarantees: (1-1/e) ≈ 0.632 approximation.
    Ensures no cluster in the test space goes unrepresented.
    """
    try:
        from apricot import FacilityLocationSelection
    except ImportError:
        print("ERROR: apricot-select required for facility location")
        print("  pip install apricot-select")
        sys.exit(1)
    
    selector = FacilityLocationSelection(k, metric='cosine', verbose=False)
    selector.fit(X)
    return list(selector.ranking)


def dpp_selection(X, k, seed=42):
    """Determinantal Point Process for probabilistic diverse sampling.
    
    Produces genuinely random diverse subsets - useful for varied
    CI runs that collectively cover more ground.
    """
    try:
        from dppy.finite_dpps import FiniteDPP
    except ImportError:
        print("ERROR: dppy required for DPP selection")
        print("  pip install dppy")
        sys.exit(1)
    
    from sklearn.preprocessing import normalize
    np.random.seed(seed)
    
    X_norm = normalize(X, norm='l2')
    L = X_norm @ X_norm.T
    L = (L + L.T) / 2  # ensure symmetry
    
    dpp = FiniteDPP('likelihood', **{'L': L})
    dpp.sample_exact_k_dpp(size=k)
    return list(dpp.list_of_samples[-1])


STRATEGIES = {
    "fps": farthest_point_sampling,
    "fps_multi": fps_multi_start,
    "kmedoids": kmedoids_selection,
    "facility": facility_location_selection,
    "dpp": dpp_selection,
}


# ─── Tag Filtering ──────────────────────────────────────────────────

def filter_by_tags(manifest, include_tags=None, exclude_tags=None):
    """Pre-filter tests by tags before diversity selection.
    
    Returns indices of tests that pass the tag filter.
    """
    indices = []
    for i, test in enumerate(manifest["tests"]):
        tags = set(t.lower() for t in test.get("tags", []))
        
        if include_tags:
            include = set(t.lower() for t in include_tags)
            if not tags & include:  # no overlap
                continue
        
        if exclude_tags:
            exclude = set(t.lower() for t in exclude_tags)
            if tags & exclude:  # has excluded tag
                continue
        
        indices.append(i)
    return indices


# ─── Main Selection Pipeline ────────────────────────────────────────

def select(artifact_dir, k, strategy="fps", output_file="selected_tests.json",
           include_tags=None, exclude_tags=None, seed=42,
           include_datadriver=True):
    """Load artifacts, apply filters, select diverse subset, write output.
    
    Args:
        artifact_dir: Directory containing Stage 1 artifacts
        k: Number of tests to select
        strategy: Selection algorithm name
        output_file: Path for output JSON
        include_tags: Only consider tests with these tags
        exclude_tags: Exclude tests with these tags
        seed: Random seed for reproducibility
        include_datadriver: Whether to include DataDriver tests
    """
    artifact_dir = Path(artifact_dir)
    
    # Load artifacts
    data = np.load(artifact_dir / "embeddings.npz", allow_pickle=True)
    vectors = data["vectors"]
    ids = data["ids"]
    
    with open(artifact_dir / "test_manifest.json") as f:
        manifest = json.load(f)
    
    print(f"Loaded {len(manifest['tests'])} tests "
          f"({vectors.shape[1]}d embeddings)")
    
    # Apply tag filters
    valid_indices = filter_by_tags(manifest, include_tags, exclude_tags)
    
    # Optionally exclude DataDriver tests
    if not include_datadriver:
        valid_indices = [
            i for i in valid_indices
            if not manifest["tests"][i].get("is_datadriver", False)
        ]
    
    if not valid_indices:
        print("ERROR: No tests remain after filtering!")
        sys.exit(1)
    
    filtered_vectors = vectors[valid_indices]
    filtered_tests = [manifest["tests"][i] for i in valid_indices]
    
    print(f"After filtering: {len(filtered_tests)} tests")
    
    # Clamp k
    k = min(k, len(filtered_tests))
    if k == len(filtered_tests):
        print(f"  k={k} equals filtered test count, selecting all")
        selected_indices = list(range(len(filtered_tests)))
    else:
        # Run selection algorithm
        algo = STRATEGIES.get(strategy)
        if not algo:
            print(f"ERROR: Unknown strategy '{strategy}'")
            print(f"  Available: {', '.join(STRATEGIES.keys())}")
            sys.exit(1)
        
        print(f"Selecting {k} tests via '{strategy}'...")
        selected_indices = algo(filtered_vectors, k, seed=seed)
    
    # Build output
    selected_tests = [filtered_tests[i] for i in selected_indices]
    
    output = {
        "strategy": strategy,
        "k": k,
        "total_tests": len(manifest["tests"]),
        "filtered_tests": len(filtered_tests),
        "seed": seed,
        "selected": [
            {
                "name": t["name"],
                "id": t["id"],
                "suite": t.get("suite", ""),
                "is_datadriver": t.get("is_datadriver", False),
            }
            for t in selected_tests
        ]
    }
    
    Path(output_file).write_text(json.dumps(output, indent=2))
    
    # Print summary statistics
    print(f"\nSelection complete:")
    print(f"  Strategy: {strategy}")
    print(f"  Selected: {k} / {len(filtered_tests)} "
          f"({100*k/len(filtered_tests):.1f}%)")
    
    dd_count = sum(1 for t in selected_tests
                   if t.get("is_datadriver", False))
    if dd_count:
        print(f"  DataDriver tests: {dd_count}")
    
    # Coverage statistics
    suites_covered = len(set(t.get("suite", "") for t in selected_tests))
    suites_total = len(set(t.get("suite", "") for t in filtered_tests))
    print(f"  Suite coverage: {suites_covered}/{suites_total}")
    
    # Diversity metric: average pairwise cosine distance
    sel_vectors = filtered_vectors[selected_indices]
    pairwise = cosine_distances(sel_vectors, sel_vectors)
    np.fill_diagonal(pairwise, np.nan)
    avg_dist = np.nanmean(pairwise)
    min_dist = np.nanmin(pairwise)
    print(f"  Avg pairwise distance: {avg_dist:.4f}")
    print(f"  Min pairwise distance: {min_dist:.4f}")
    
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Select diverse tests")
    parser.add_argument("--artifacts", default="./vector_artifacts",
                        help="Directory with Stage 1 artifacts")
    parser.add_argument("--k", type=int, required=True,
                        help="Number of tests to select")
    parser.add_argument("--strategy", default="fps",
                        choices=list(STRATEGIES.keys()),
                        help="Selection algorithm")
    parser.add_argument("--output", default="selected_tests.json",
                        help="Output file path")
    parser.add_argument("--include-tags", nargs="*",
                        help="Only include tests with these tags")
    parser.add_argument("--exclude-tags", nargs="*",
                        help="Exclude tests with these tags")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-datadriver", action="store_true",
                        help="Exclude DataDriver-generated tests")
    args = parser.parse_args()
    
    select(
        args.artifacts, args.k, args.strategy, args.output,
        args.include_tags, args.exclude_tags, args.seed,
        include_datadriver=not args.no_datadriver
    )
```

### Artifact Contract

Stage 2 consumes Stage 1 artifacts and produces:

| File | Format | Contents |
|------|--------|----------|
| `selected_tests.json` | JSON | Selection metadata + list of selected test names/IDs |

Example output:
```json
{
  "strategy": "fps",
  "k": 50,
  "total_tests": 847,
  "filtered_tests": 623,
  "seed": 42,
  "selected": [
    {"name": "Login With Valid Credentials", "id": "a3f1...", "suite": "tests/login.robot", "is_datadriver": false},
    {"name": "Login with user admin and password secret", "id": "b7c2...", "suite": "tests/dd_login.robot", "is_datadriver": true}
  ]
}
```

---

## Stage 3: Execution (The Runner)

This stage receives the selection file and runs only the selected tests. It provides **two integration mechanisms**: a PreRunModifier for standard suites and a Listener v3 for DataDriver suites.

### Implementation: `stage3_run.py` — Unified Runner

```python
#!/usr/bin/env python3
"""Stage 3: Execute selected Robot Framework tests.

Consumes: selected_tests.json (from Stage 2)
Produces: output.xml, log.html, report.html

Handles both standard tests (via PreRunModifier) and DataDriver tests
(via Listener v3) transparently.
"""
import argparse
import json
import sys
from pathlib import Path

from robot.api import SuiteVisitor


class DiversePreRunModifier(SuiteVisitor):
    """PreRunModifier that filters standard (non-DataDriver) tests.
    
    Usage via command line:
        robot --prerunmodifier stage3_run.DiversePreRunModifier:selected_tests.json tests/
    
    Usage programmatically:
        suite.visit(DiversePreRunModifier('selected_tests.json'))
    """
    
    def __init__(self, selection_file):
        with open(selection_file) as f:
            data = json.load(f)
        
        # Build lookup sets
        self.selected_names = set(t["name"] for t in data["selected"]
                                  if not t.get("is_datadriver", False))
        self.selected_ids = set(t["id"] for t in data["selected"]
                                if not t.get("is_datadriver", False))
        self._stats = {"kept": 0, "removed": 0}
    
    def start_suite(self, suite):
        original_count = len(suite.tests)
        suite.tests = [
            t for t in suite.tests
            if t.name in self.selected_names
        ]
        self._stats["kept"] += len(suite.tests)
        self._stats["removed"] += original_count - len(suite.tests)
    
    def end_suite(self, suite):
        # Remove empty child suites to keep output clean
        suite.suites = [s for s in suite.suites if s.test_count > 0]
    
    def visit_test(self, test):
        pass  # skip visiting test internals for speed


class DiverseDataDriverListener:
    """Listener v3 for filtering DataDriver-generated tests.
    
    DataDriver creates tests via its own start_suite listener event,
    which fires AFTER PreRunModifiers. This listener must therefore
    run AFTER DataDriver (lower ROBOT_LISTENER_PRIORITY value).
    
    Usage:
        robot --listener stage3_run.DiverseDataDriverListener:selected_tests.json tests/
    """
    ROBOT_LISTENER_API_VERSION = 3
    # DataDriver uses default priority. Lower number = runs later.
    # We need to run after DataDriver has generated its tests.
    ROBOT_LISTENER_PRIORITY = 50
    
    def __init__(self, selection_file):
        with open(selection_file) as f:
            data = json.load(f)
        
        self.selected_dd_names = set(
            t["name"] for t in data["selected"]
            if t.get("is_datadriver", False)
        )
        self._has_dd_tests = bool(self.selected_dd_names)
    
    def start_suite(self, data, result):
        """Filter tests after DataDriver has generated them."""
        if not self._has_dd_tests:
            return
        
        # Detect DataDriver suites: they have many tests that weren't
        # in the original .robot file (generated dynamically)
        original_count = len(data.tests)
        if original_count <= 1:
            return  # not a DataDriver suite
        
        # Check if any test names match our DataDriver selection
        dd_matches = [
            t for t in data.tests
            if t.name in self.selected_dd_names
        ]
        
        if dd_matches:
            data.tests = dd_matches
            print(f"* DataDriverSelector: {len(dd_matches)}/{original_count} "
                  f"tests selected in '{data.name}'")


def run_selected(suite_path, selection_file, output_dir="./results",
                 extra_robot_args=None):
    """Execute the full Stage 3 pipeline.
    
    For suites with ONLY standard tests, uses PreRunModifier.
    For suites with DataDriver tests, uses both PreRunModifier + Listener.
    """
    from robot.api import TestSuite
    import robot
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(selection_file) as f:
        selection = json.load(f)
    
    has_dd = any(t.get("is_datadriver") for t in selection["selected"])
    
    print(f"Stage 3: Executing {selection['k']} selected tests")
    print(f"  Strategy: {selection['strategy']}")
    print(f"  Has DataDriver tests: {has_dd}")
    
    # Build robot command arguments
    args = [
        '--outputdir', str(output_dir),
        '--prerunmodifier',
        f'stage3_run.DiversePreRunModifier:{selection_file}',
    ]
    
    if has_dd:
        args.extend([
            '--listener',
            f'stage3_run.DiverseDataDriverListener:{selection_file}',
        ])
    
    if extra_robot_args:
        args.extend(extra_robot_args)
    
    args.append(str(suite_path))
    
    # Run
    rc = robot.run_cli(args, exit=False)
    
    # Generate selection coverage report
    report = {
        "return_code": rc,
        "selection": {
            "strategy": selection["strategy"],
            "k": selection["k"],
            "total_available": selection["total_tests"],
            "reduction": f"{100*(1-selection['k']/selection['total_tests']):.1f}%"
        },
        "output_dir": str(output_dir),
    }
    (output_dir / "selection_report.json").write_text(
        json.dumps(report, indent=2)
    )
    
    return rc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Execute selected tests")
    parser.add_argument("--suite", required=True, help="Path to test suite")
    parser.add_argument("--selection", default="selected_tests.json",
                        help="Selection file from Stage 2")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("robot_args", nargs="*",
                        help="Additional robot arguments")
    args = parser.parse_args()
    
    rc = run_selected(
        args.suite, args.selection, args.output_dir, args.robot_args
    )
    sys.exit(rc)
```

### Alternative: Pure Command-Line Stage 3

For teams that prefer not adding a Python wrapper, Stage 3 can be pure `robot` CLI:

```bash
# Standard suites only:
robot --prerunmodifier stage3_run.DiversePreRunModifier:selected_tests.json \
      --outputdir ./results \
      ./tests/

# With DataDriver suites:
robot --prerunmodifier stage3_run.DiversePreRunModifier:selected_tests.json \
      --listener stage3_run.DiverseDataDriverListener:selected_tests.json \
      --outputdir ./results \
      ./tests/
```

---

## CI/CD Pipeline Configurations

### GitHub Actions: Three-Job Pipeline

```yaml
# .github/workflows/diverse-tests.yml
name: Diverse Test Selection Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
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

env:
  PYTHON_VERSION: '3.11'
  DEFAULT_K: '50'
  DEFAULT_STRATEGY: 'fps'

jobs:
  # ═══════════════════════════════════════════════════════════════
  # STAGE 1: Vectorize test cases
  # Runs only when .robot files change (cached otherwise)
  # ═══════════════════════════════════════════════════════════════
  vectorize:
    name: "Stage 1: Vectorize Tests"
    runs-on: ubuntu-latest
    outputs:
      cache-hit: ${{ steps.cache-vectors.outputs.cache-hit }}
    steps:
      - uses: actions/checkout@v4

      - name: Hash test files for cache key
        id: test-hash
        run: |
          HASH=$(find tests/ -name "*.robot" -exec md5sum {} \; | sort | md5sum | cut -d' ' -f1)
          echo "hash=$HASH" >> $GITHUB_OUTPUT

      - name: Cache vector artifacts
        id: cache-vectors
        uses: actions/cache@v4
        with:
          path: vector_artifacts/
          key: vectors-${{ steps.test-hash.outputs.hash }}

      - name: Set up Python
        if: steps.cache-vectors.outputs.cache-hit != 'true'
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install vectorization dependencies
        if: steps.cache-vectors.outputs.cache-hit != 'true'
        run: |
          pip install sentence-transformers robotframework numpy

      - name: Run vectorization
        if: steps.cache-vectors.outputs.cache-hit != 'true'
        run: |
          python stage1_vectorize.py \
            --suite ./tests/ \
            --output ./vector_artifacts/ \
            --resolve-depth 2

      - name: Upload vector artifacts
        uses: actions/upload-artifact@v4
        with:
          name: vector-artifacts
          path: vector_artifacts/
          retention-days: 7

  # ═══════════════════════════════════════════════════════════════
  # STAGE 2: Select diverse subset
  # Always runs (selection parameters may change per run)
  # ═══════════════════════════════════════════════════════════════
  select:
    name: "Stage 2: Select Diverse Tests"
    needs: vectorize
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Download vector artifacts
        uses: actions/download-artifact@v4
        with:
          name: vector-artifacts
          path: vector_artifacts/

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install selection dependencies
        run: pip install numpy scikit-learn

      - name: Run selection
        run: |
          K=${{ github.event.inputs.k || env.DEFAULT_K }}
          STRATEGY=${{ github.event.inputs.strategy || env.DEFAULT_STRATEGY }}
          python stage2_select.py \
            --artifacts ./vector_artifacts/ \
            --k $K \
            --strategy $STRATEGY \
            --output selected_tests.json

      - name: Upload selection artifact
        uses: actions/upload-artifact@v4
        with:
          name: test-selection
          path: selected_tests.json
          retention-days: 1

      - name: Annotate PR with selection summary
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const sel = JSON.parse(fs.readFileSync('selected_tests.json'));
            const body = `### 🎯 Diverse Test Selection
            | Metric | Value |
            |--------|-------|
            | Strategy | \`${sel.strategy}\` |
            | Selected | ${sel.k} / ${sel.total_tests} |
            | Reduction | ${(100*(1-sel.k/sel.total_tests)).toFixed(1)}% |`;
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: body
            });

  # ═══════════════════════════════════════════════════════════════
  # STAGE 3: Execute selected tests
  # ═══════════════════════════════════════════════════════════════
  execute:
    name: "Stage 3: Execute Tests"
    needs: select
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Download selection artifact
        uses: actions/download-artifact@v4
        with:
          name: test-selection

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install test dependencies
        run: |
          pip install robotframework robotframework-datadriver \
                      robotframework-seleniumlibrary  # add your libs

      - name: Execute selected tests
        run: |
          python stage3_run.py \
            --suite ./tests/ \
            --selection selected_tests.json \
            --output-dir ./results

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-results
          path: |
            results/output.xml
            results/log.html
            results/report.html
            results/selection_report.json
          retention-days: 30
```

### GitLab CI: Three-Stage Pipeline

```yaml
# .gitlab-ci.yml
stages:
  - vectorize
  - select
  - execute

variables:
  PYTHON_VERSION: "3.11"
  DEFAULT_K: "50"
  DEFAULT_STRATEGY: "fps"
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.pip-cache"

# ─── STAGE 1: Vectorize ────────────────────────────────────────
vectorize:
  stage: vectorize
  image: python:${PYTHON_VERSION}
  cache:
    key:
      files:
        - tests/**/*.robot  # cache invalidates when tests change
    paths:
      - vector_artifacts/
      - .pip-cache/
  script:
    - pip install sentence-transformers robotframework numpy
    - |
      python stage1_vectorize.py \
        --suite ./tests/ \
        --output ./vector_artifacts/ \
        --resolve-depth 2
  artifacts:
    paths:
      - vector_artifacts/embeddings.npz
      - vector_artifacts/test_manifest.json
    expire_in: 7 days
  rules:
    # Run on any pipeline, but skip if artifacts are cached
    - changes:
        - tests/**/*.robot
      when: always
    - when: always  # fallback: always run, script detects no-change

# ─── STAGE 2: Select ───────────────────────────────────────────
select:
  stage: select
  image: python:${PYTHON_VERSION}-slim
  needs:
    - job: vectorize
      artifacts: true
  script:
    - pip install numpy scikit-learn
    - |
      K=${SELECT_K:-$DEFAULT_K}
      STRATEGY=${SELECT_STRATEGY:-$DEFAULT_STRATEGY}
      python stage2_select.py \
        --artifacts ./vector_artifacts/ \
        --k $K \
        --strategy $STRATEGY \
        --output selected_tests.json
  artifacts:
    paths:
      - selected_tests.json
    expire_in: 1 day
    reports:
      # Expose selection as downloadable from MR
      dotenv: selection.env

# ─── STAGE 3: Execute ──────────────────────────────────────────
execute:
  stage: execute
  image: python:${PYTHON_VERSION}
  needs:
    - job: select
      artifacts: true
  script:
    - pip install robotframework robotframework-datadriver
    - |
      python stage3_run.py \
        --suite ./tests/ \
        --selection selected_tests.json \
        --output-dir ./results
  artifacts:
    paths:
      - results/
    expire_in: 30 days
    reports:
      junit: results/output.xml
    when: always
```

### Jenkins: Declarative Pipeline

```groovy
// Jenkinsfile
pipeline {
    agent any

    parameters {
        integer(name: 'K', defaultValue: 50, description: 'Number of tests to select')
        choice(name: 'STRATEGY', choices: ['fps', 'fps_multi', 'kmedoids', 'facility', 'dpp'],
               description: 'Diversity selection strategy')
    }

    environment {
        PYTHON = 'python3'
    }

    stages {
        stage('Vectorize') {
            steps {
                script {
                    // Check if vectors need updating
                    def testsHash = sh(
                        script: 'find tests/ -name "*.robot" -exec md5sum {} \\; | sort | md5sum | cut -d" " -f1',
                        returnStdout: true
                    ).trim()

                    def cached = fileExists("vector_artifacts/.hash_${testsHash}")

                    if (!cached) {
                        sh """
                            ${PYTHON} -m pip install sentence-transformers robotframework numpy
                            ${PYTHON} stage1_vectorize.py \
                                --suite ./tests/ \
                                --output ./vector_artifacts/ \
                                --resolve-depth 2 \
                                --force
                            touch vector_artifacts/.hash_${testsHash}
                        """
                    } else {
                        echo "Vectors cached, skipping vectorization"
                    }
                }
            }
            post {
                success {
                    stash name: 'vectors', includes: 'vector_artifacts/**'
                }
            }
        }

        stage('Select') {
            steps {
                unstash 'vectors'
                sh """
                    ${PYTHON} -m pip install numpy scikit-learn
                    ${PYTHON} stage2_select.py \
                        --artifacts ./vector_artifacts/ \
                        --k ${params.K} \
                        --strategy ${params.STRATEGY} \
                        --output selected_tests.json
                """
            }
            post {
                success {
                    stash name: 'selection', includes: 'selected_tests.json'
                    archiveArtifacts artifacts: 'selected_tests.json'
                }
            }
        }

        stage('Execute') {
            steps {
                unstash 'selection'
                sh """
                    ${PYTHON} -m pip install robotframework robotframework-datadriver
                    ${PYTHON} stage3_run.py \
                        --suite ./tests/ \
                        --selection selected_tests.json \
                        --output-dir ./results
                """
            }
            post {
                always {
                    robot outputPath: 'results/',
                          logFileName: 'log.html',
                          outputFileName: 'output.xml',
                          reportFileName: 'report.html',
                          passThreshold: 95.0
                    archiveArtifacts artifacts: 'results/**'
                }
            }
        }
    }
}
```

---

## Advanced: Parallel Strategy Comparison

A powerful extension of the multistage design is running **multiple selection strategies in parallel** at Stage 2, then either merging them or comparing their fault-detection rates:

```yaml
# GitHub Actions: parallel strategy fan-out
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

  compare-or-merge:
    name: "Merge/Compare Strategies"
    needs: [select-fps, select-kmedoids, select-facility]
    runs-on: ubuntu-latest
    steps:
      - name: Download all selections
        uses: actions/download-artifact@v4
        with:
          pattern: selection-*
          merge-multiple: true

      - name: Union merge (run tests selected by ANY strategy)
        run: |
          python -c "
          import json, glob
          all_names = set()
          for f in glob.glob('selection-*/selected_tests.json'):
              data = json.load(open(f))
              all_names.update(t['name'] for t in data['selected'])
          merged = {'strategy': 'union_merge', 'k': len(all_names),
                    'selected': [{'name': n} for n in sorted(all_names)]}
          json.dump(merged, open('selected_tests.json', 'w'), indent=2)
          print(f'Union: {len(all_names)} unique tests from 3 strategies')
          "
```

You can also do **intersection merge** (only run tests selected by ALL strategies — the consensus diverse core) or **weighted merge** (score each test by how many strategies selected it).

---

## Optimization: Conditional Vectorization

The biggest performance win is skipping Stage 1 entirely when tests haven't changed. In GitHub Actions, this uses the cache action with a hash key:

```yaml
      - name: Hash test files for cache key
        id: test-hash
        run: |
          # Hash all .robot files + any DataDriver CSV sources
          HASH=$(find tests/ -name "*.robot" -o -name "*.csv" | \
                 sort | xargs md5sum | md5sum | cut -d' ' -f1)
          echo "hash=$HASH" >> $GITHUB_OUTPUT

      - name: Restore cached vectors
        id: cache-vectors
        uses: actions/cache@v4
        with:
          path: vector_artifacts/
          key: vectors-${{ steps.test-hash.outputs.hash }}
          restore-keys: |
            vectors-  # fallback to most recent cache
```

In GitLab CI, the `cache:key:files` directive handles this natively. In Jenkins, you use `stash`/`unstash` across stages within a single pipeline, or the `jobCachingPlugin` for cross-build caching.

---

## Pipeline Timing (Typical Benchmarks)

For a suite of ~2,000 test cases with `all-MiniLM-L6-v2`:

| Stage | Cold (no cache) | Warm (cached) |
|-------|-----------------|---------------|
| **Vectorize** | 15–30s (model load + embed) | **0s** (cache hit) |
| **Select** (FPS, k=100) | 0.3s | 0.3s |
| **Select** (k-Medoids, k=100) | 2–5s | 2–5s |
| **Select** (DPP, k=100) | 8–15s | 8–15s |
| **Execute** | Depends on tests | Depends on tests |
| **Artifact transfer** | 1–3s per stage | 1–3s per stage |

The CI overhead (artifact upload/download between stages) adds ~3–5 seconds total, which is trivial compared to the time saved by running fewer tests.

---

## Selecting the Right Strategy Per Stage

Here's a decision framework for choosing strategies at Stage 2:

| Scenario | Strategy | Why |
|----------|----------|-----|
| **Daily CI** on main | `fps` with k=20% | Fast, deterministic, good boundary coverage |
| **PR validation** | `fps_multi` with k=15% | More robust, catches edge cases |
| **Nightly full diversity** | `dpp` with k=30% | Random diverse sample, varies nightly for cumulative coverage |
| **Pre-release regression** | `facility` with k=50% | Ensures every test cluster is represented |
| **Quick smoke test** | `fps` with k=10 (absolute) | Just the 10 most different tests |
| **Comparing strategies** | Fan-out parallel | Run all, compare fault detection over time |
