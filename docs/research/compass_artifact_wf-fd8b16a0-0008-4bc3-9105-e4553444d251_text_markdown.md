# Vector-based diversity test selection for Robot Framework

**Embedding Robot Framework test cases into vector space and selecting maximally diverse subsets for execution is a viable, novel approach that fills a significant gap in the testing ecosystem.** No existing tool—commercial or open-source—combines modern NLP embeddings with diversity-based selection for keyword-driven test frameworks. The approach works immediately on any test suite without historical execution data (unlike Launchable or Meta's Predictive Test Selection), requires no coverage instrumentation, and exploits the fact that Robot Framework's keyword-driven syntax is essentially structured natural language—ideal for transformer-based embeddings. This report provides complete implementation details: from parsing keyword trees via `robot.api`, through embedding and storage in vector databases, to diversity selection algorithms and integration with Robot Framework's execution pipeline via PreRunModifiers and Listener v3.

## Parsing keyword trees with the Robot Framework Python API

The `robot.api` module provides three primary entry points for parsing test suites into traversable models. `TestSuite.from_file_system()` is the recommended approach, building a complete suite model from `.robot` files or directories. The resulting `TestSuite` object exposes `tests`, `suites` (child suites), and `resource` (containing user keywords, imports, and variables).

```python
from robot.api import TestSuite

suite = TestSuite.from_file_system('/path/to/tests/')

for test in suite.tests:
    print(f"Test: {test.name}, Tags: {test.tags}")
    for item in test.body:
        if hasattr(item, 'name'):  # Keyword call
            print(f"  KW: {item.name}, Args: {item.args}")
```

**A critical limitation** is that at parse time, the model contains only keyword call references (name + args as strings)—it does not resolve which implementation will execute. User keywords calling other user keywords are stored as `Keyword` objects referencing callees by name. To build a fully resolved keyword tree, you must manually match names against the user keyword registry:

```python
from robot.api import TestSuite

def build_keyword_map(suite):
    """Build a lookup of all user keywords across suite and resources."""
    kw_map = {}
    for uk in suite.resource.keywords:
        kw_map[uk.name.lower().replace(' ', '_')] = uk
    for child in suite.suites:
        kw_map.update(build_keyword_map(child))
    return kw_map

def resolve_keyword_tree(kw_name, kw_args, kw_map, depth=0, max_depth=10):
    """Recursively resolve a keyword call into its full tree."""
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

def test_to_text(test, kw_map):
    """Convert a test case's full keyword tree to embeddable text."""
    parts = [f"Test: {test.name}."]
    if test.tags:
        parts.append(f"Tags: {', '.join(str(t) for t in test.tags)}.")
    
    for item in test.body:
        if hasattr(item, 'name') and item.name:
            tree = resolve_keyword_tree(item.name, item.args, kw_map)
            parts.append(flatten_tree(tree))
    return " ".join(parts)

def flatten_tree(node):
    """Flatten keyword tree node into natural language."""
    kw = node["keyword"].replace('_', ' ')
    # Filter out DOM locators and variable placeholders
    semantic_args = [
        a for a in node["args"]
        if not any(a.startswith(p) for p in ['id:', 'css:', 'xpath:', '//', '${'])
    ]
    text = kw
    if semantic_args:
        text += f" with {', '.join(semantic_args)}"
    children_text = " ".join(flatten_tree(c) for c in node["children"])
    return f"{text} {children_text}".strip()
```

The `SuiteVisitor` pattern provides an alternative traversal mechanism. Subclassing `SuiteVisitor` and overriding `start_test`, `start_keyword`, and related methods lets you walk the entire suite hierarchy with a visitor pattern—useful for extracting all test information in a single pass.

**What to include in the text representation** matters significantly for embedding quality. Research shows that converting structured data to natural language before embedding yields **19–27% improvement** in retrieval metrics compared to embedding raw structured formats. Include test names (highest semantic value), tags, suite paths, keyword names, and semantic arguments like page titles or expected text. Exclude DOM locators (`id:username`, `//div[@class='foo']`), variable placeholders (`${VAR}`), and implementation-specific XPaths—these are noise that dilutes embedding quality.

## Embedding models and vector database selection

**For embedding test case text, `all-MiniLM-L6-v2` is the optimal choice** for this use case. At just **22M parameters** and **384 dimensions**, it runs on CPU in sub-millisecond time per embedding, requires no API key or cloud dependency, and handles the typical 20–100 token test descriptions well within its 256-token sequence limit. For higher quality at modest compute cost, `all-mpnet-base-v2` (768 dims, 109M params) offers the best general-purpose performance in the sentence-transformers family.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# Batch encoding is efficient—encode all test descriptions at once
descriptions = [test_to_text(t, kw_map) for t in all_tests]
embeddings = model.encode(descriptions, show_progress_bar=True)
# Returns numpy array of shape (N, 384)
```

OpenAI's `text-embedding-3-small` (1536 dims, supports truncation to 512 via the `dimensions` parameter) is a cloud-based alternative that scores higher on benchmarks but adds API cost and latency. For fully local operation with no external dependencies, `all-MiniLM-L6-v2` via sentence-transformers is the clear winner.

### Choosing a vector store for test embeddings

For the target scale of **hundreds to low thousands of test cases**, five lightweight options were evaluated. The comparison focuses on local-first operation, metadata support, and ability to retrieve raw vectors (essential for diversity calculations).

| Feature | FAISS | ChromaDB | Qdrant | LanceDB | Weaviate |
|---------|-------|----------|--------|---------|----------|
| Setup complexity | Low | Very low | Low | Very low | High |
| Native metadata | No (manual) | Yes | Yes (rich) | Yes | Yes |
| Raw vector retrieval | `reconstruct()` | `include=["embeddings"]` | `with_vectors=True` | Column access | `include_vector=True` |
| Built-in embedding | No | Yes (MiniLM default) | Yes (FastEmbed) | Yes (registry) | Yes |
| Persistence | Manual save/load | `PersistentClient` | `path=` | Automatic | Automatic |
| Install size | ~30 MB | ~100 MB | ~50 MB | ~50 MB | ~500 MB+ |

**ChromaDB is the top recommendation** for this use case: zero configuration, built-in `all-MiniLM-L6-v2` embedding, native metadata with MongoDB-style filtering, and trivial raw vector retrieval. **LanceDB** is the second choice if you want Pandas/Polars DataFrame integration—its "SQLite for vectors" design makes analytics straightforward. **FAISS** wins when you want minimal dependencies and full control, but requires managing metadata separately. **Qdrant** suits teams planning to scale beyond prototyping (same API locally and in production). **Weaviate** is overkill at this scale.

```python
import chromadb

client = chromadb.PersistentClient(path="./rf_test_vectors")
collection = client.get_or_create_collection(
    name="test_cases",
    metadata={"hnsw:space": "cosine"}
)

# Store test embeddings with metadata
for i, test in enumerate(all_tests):
    text = test_to_text(test, kw_map)
    collection.add(
        ids=[f"tc_{i:04d}"],
        documents=[text],
        metadatas=[{
            "name": test.name,
            "tags": ",".join(str(t) for t in test.tags),
            "suite": str(test.parent.source) if test.parent else ""
        }]
    )

# Retrieve all vectors for diversity calculation
all_data = collection.get(include=["embeddings", "metadatas"])
vectors = np.array(all_data["embeddings"])  # shape: (N, 384)
```

## Diversity-based test selection algorithms

Five algorithms were evaluated for selecting a maximally diverse subset of k tests from N total embedded test cases. The mathematical foundations, approximation guarantees, and practical performance differ significantly.

### Farthest point sampling delivers the best tradeoff

**Farthest-first traversal** (also called farthest point sampling, FPS) is the recommended algorithm. It iteratively selects the point maximally distant from all previously selected points, providing a **2-approximation** for the max-min dispersion problem (Gonzalez, 1985). For N=5,000 and k=200, it runs in milliseconds.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

def farthest_point_sampling(X, k, initial_idx=None):
    """
    Select k maximally dispersed points via greedy farthest-first traversal.
    
    Args:
        X: (N, d) embedding matrix
        k: number of points to select
        initial_idx: starting point (random if None)
    Returns:
        List of k selected indices
    """
    N = X.shape[0]
    if initial_idx is None:
        initial_idx = np.random.randint(N)
    
    selected = [initial_idx]
    min_distances = cosine_distances(X[initial_idx:initial_idx+1], X)[0]
    min_distances[initial_idx] = -1  # mark selected
    
    for _ in range(k - 1):
        next_idx = int(np.argmax(min_distances))
        selected.append(next_idx)
        new_dists = cosine_distances(X[next_idx:next_idx+1], X)[0]
        min_distances = np.minimum(min_distances, new_dists)
        min_distances[next_idx] = -1
    
    return selected
```

The time complexity is **O(N·k·d)** with O(N) memory beyond the embedding matrix. A key property: outliers are always selected first, which ensures boundary coverage but may be undesirable if the test suite contains spurious outliers. Running FPS multiple times from different starting points and taking the best result (measured by minimum pairwise distance) mitigates initial-point sensitivity.

### Four alternative algorithms and when to use them

**k-Medoids** (via `sklearn-extra`) selects cluster representatives rather than boundary points. Each medoid is a real data point, making interpretation natural ("this test represents the login cluster"). However, it optimizes representativeness (minimize average distance), not dispersion, so it misses outlier/boundary cases. PAM runs in O(N²·k) per iteration—slower than FPS at scale.

```python
from sklearn_extra.cluster import KMedoids

def select_via_kmedoids(X, k):
    kmed = KMedoids(n_clusters=k, metric='cosine', method='pam',
                    init='k-medoids++', random_state=42)
    kmed.fit(X)
    return list(kmed.medoid_indices_)
```

**Determinantal Point Processes** (DPPs) offer the most principled probabilistic model of diversity—similar items are anti-correlated in the sampling distribution. The L-kernel is constructed from normalized embeddings: L = V̂·V̂ᵀ. Exact k-DPP sampling requires O(N³) eigendecomposition, making it the slowest option, but it produces genuinely random diverse subsets (useful for ensemble testing across CI runs).

```python
from dppy.finite_dpps import FiniteDPP
from sklearn.preprocessing import normalize

def select_via_dpp(X, k):
    X_norm = normalize(X, norm='l2')
    L = X_norm @ X_norm.T
    L = (L + L.T) / 2  # ensure symmetry
    dpp = FiniteDPP('likelihood', **{'L': L})
    dpp.sample_exact_k_dpp(size=k)
    return dpp.list_of_samples[-1]
```

**Maximal Marginal Relevance** (MMR) adds a tunable λ parameter balancing relevance and diversity. With λ=0 or uniform relevance, **MMR reduces exactly to FPS**. MMR becomes valuable when you want to bias selection toward a query—for example, selecting tests diverse but relevant to a recent code change or bug report area.

**Facility location** (submodular optimization, via `apricot-select`) maximizes representativeness: every test in the full suite should have a similar representative in the selected subset. It provides a **(1-1/e) ≈ 0.632 approximation** guarantee. This is subtly different from FPS's max-min dispersion objective—facility location ensures no cluster is unrepresented, while FPS maximizes spread.

| Algorithm | Time | Guarantee | Deterministic | Best for |
|-----------|------|-----------|---------------|----------|
| FPS | O(N·k·d) | 2-approx (max-min) | Yes | Fast, guaranteed coverage |
| k-Medoids | O(N²·k·iter) | Local optima | No | Representative selection |
| k-DPP | O(N³ + N·k³) | Probabilistic | No | Varied ensemble sampling |
| MMR | O(N²·d + N·k) | Same as FPS at λ=0 | Yes | Query-biased diversity |
| Facility location | O(N²·k) | (1-1/e) | Yes | Cluster representativeness |

## Integration with Robot Framework execution

### PreRunModifier for standard test suites

A `PreRunModifier` extending `robot.api.SuiteVisitor` is the primary mechanism for filtering tests before execution. It receives the parsed `TestSuite` model and can modify `suite.tests` in-place. The modifier runs **after parsing but before execution and before tag filtering** (`--include`/`--exclude`).

```python
# diverse_selector.py
import json
import numpy as np
from robot.api import SuiteVisitor

class DiverseTestSelector(SuiteVisitor):
    """PreRunModifier that selects a diverse subset of tests.
    
    Usage:
        robot --prerunmodifier diverse_selector.py:selected_tests.json tests/
    
    Where selected_tests.json contains a list of test names to run.
    """
    
    def __init__(self, selection_file):
        with open(selection_file) as f:
            self.selected_names = set(json.load(f))
    
    def start_suite(self, suite):
        suite.tests = [
            t for t in suite.tests 
            if t.name in self.selected_names
        ]
    
    def end_suite(self, suite):
        suite.suites = [s for s in suite.suites if s.test_count > 0]
    
    def visit_test(self, test):
        pass  # skip visiting test internals for performance
```

For programmatic execution (not via command line), **`--prerunmodifier` is silently ignored** when passed to `TestSuite.run()`. Instead, apply the visitor directly:

```python
suite = TestSuite.from_file_system('tests/')
suite.visit(DiverseTestSelector('selected_tests.json'))
suite.run(output='output.xml')
```

### The DataDriver challenge requires Listener v3

**DataDriver presents a fundamental timing problem.** It is a library listener that generates test cases via the Listener v3 `start_suite` event—this happens **after PreRunModifiers have already been applied**. A PreRunModifier applied via `--prerunmodifier` sees only the single template test case, not the hundreds of data-driven tests that DataDriver will create.

The execution order is:

1. Robot Framework parses `.robot` files → builds `TestSuite` model
2. **PreRunModifiers run** → only template test exists
3. Tag filtering (`--include`/`--exclude`) is processed
4. **Execution begins** → Listener v3 `start_suite` fires
5. **DataDriver's `start_suite` fires** → generates tests from CSV/Excel
6. Generated tests execute

To filter DataDriver-generated tests, you must use a **Listener v3 that fires after DataDriver**, controlled via `ROBOT_LISTENER_PRIORITY`:

```python
# datadriver_diverse_selector.py
import json

class DataDriverDiverseSelector:
    """Listener v3 that filters DataDriver-generated tests.
    
    Must run AFTER DataDriver (lower priority number = runs later).
    DataDriver has default priority; set ours lower to run after it.
    
    Usage:
        robot --listener datadriver_diverse_selector.py:selected_tests.json tests/
    """
    ROBOT_LISTENER_API_VERSION = 3
    ROBOT_LISTENER_PRIORITY = 50  # lower than default (higher runs first)
    
    def __init__(self, selection_file):
        with open(selection_file) as f:
            self.selected_names = set(json.load(f))
    
    def start_suite(self, data, result):
        """Filter tests after DataDriver has generated them."""
        # At this point, DataDriver has already populated data.tests
        original_count = len(data.tests)
        data.tests = [
            t for t in data.tests 
            if t.name in self.selected_names
        ]
        filtered_count = len(data.tests)
        if original_count != filtered_count:
            print(f"* DiverseSelector: {filtered_count}/{original_count} "
                  f"tests selected in {data.name}")
```

A subtlety: DataDriver-generated test names follow a pattern derived from the template and data row values (e.g., `Login with user admin and password secret`). The embedding and selection pipeline must account for this by either pre-reading the CSV/Excel data source to generate expected test names, or by embedding at the data-row level.

```python
# Pre-read DataDriver CSV to generate test descriptions for embedding
import csv

def read_datadriver_tests(csv_path, template_name):
    """Read DataDriver CSV and generate expected test names + descriptions."""
    tests = []
    with open(csv_path) as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            test_name = row.get('*** Test Cases ***', '').strip()
            if not test_name or test_name.startswith('#'):
                continue
            # Collect argument values
            args = {k: v for k, v in row.items() 
                    if k.startswith('${') and v}
            description = f"Template: {template_name}. Test: {test_name}. "
            description += " ".join(f"{k}={v}" for k, v in args.items())
            tests.append({"name": test_name, "description": description})
    return tests
```

## Three end-to-end architecture proposals

### Approach 1: offline indexing with PreRunModifier selection

This is the simplest and most robust architecture. Embedding happens as a separate offline step; test execution uses a PreRunModifier (or Listener v3 for DataDriver suites) with a pre-computed selection file.

**Pipeline stages:**

1. **Index phase** (runs on-demand or on test file change): Parse all `.robot` files, resolve keyword trees, generate text representations, embed with `all-MiniLM-L6-v2`, store in ChromaDB with metadata.
2. **Select phase** (runs before each CI execution): Load all vectors from ChromaDB, run FPS to select k diverse tests, write selected test names to JSON file.
3. **Execute phase**: Robot Framework runs with `--prerunmodifier` pointing to the JSON file (or `--listener` for DataDriver suites).

```python
# index_tests.py — Run once or when tests change
from robot.api import TestSuite
from sentence_transformers import SentenceTransformer
import chromadb
import json
import hashlib

def index_test_suite(suite_path, db_path="./rf_vectors"):
    suite = TestSuite.from_file_system(suite_path)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        "test_cases", metadata={"hnsw:space": "cosine"}
    )
    
    kw_map = build_keyword_map(suite)
    tests_data = []
    
    def collect_tests(s):
        for test in s.tests:
            text = test_to_text(test, kw_map)
            test_id = hashlib.md5(
                f"{s.source}::{test.name}".encode()
            ).hexdigest()
            tests_data.append({
                "id": test_id,
                "name": test.name,
                "text": text,
                "tags": ",".join(str(t) for t in test.tags),
                "suite": str(s.source or s.name),
            })
        for child in s.suites:
            collect_tests(child)
    
    collect_tests(suite)
    
    # Batch embed
    texts = [t["text"] for t in tests_data]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Upsert into ChromaDB
    collection.upsert(
        ids=[t["id"] for t in tests_data],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[{"name": t["name"], "tags": t["tags"], 
                    "suite": t["suite"]} for t in tests_data]
    )
    print(f"Indexed {len(tests_data)} test cases")

# select_diverse.py — Run before each execution
import numpy as np

def select_diverse_tests(db_path, k, output_file, method="fps"):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection("test_cases")
    
    all_data = collection.get(include=["embeddings", "metadatas"])
    vectors = np.array(all_data["embeddings"])
    names = [m["name"] for m in all_data["metadatas"]]
    
    if method == "fps":
        selected_idx = farthest_point_sampling(vectors, k)
    
    selected_names = [names[i] for i in selected_idx]
    
    with open(output_file, 'w') as f:
        json.dump(selected_names, f, indent=2)
    print(f"Selected {k} diverse tests from {len(names)} total")
    return selected_names
```

```bash
# CI/CD pipeline usage
python index_tests.py --suite ./tests/
python select_diverse.py --k 50 --output selected.json
robot --prerunmodifier diverse_selector.py:selected.json ./tests/
# For DataDriver suites:
robot --listener datadriver_diverse_selector.py:selected.json ./tests/
```

### Approach 2: real-time embedding with Listener-based selection

This approach performs everything at runtime—a Listener v3 parses tests as they are discovered, embeds them on-the-fly, runs diversity selection, and removes non-selected tests. No separate indexing step is needed. This is ideal for development-time use where the overhead of a separate pipeline is undesirable.

```python
# realtime_diverse_listener.py
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

class RealtimeDiverseSelector:
    """Listener v3 that embeds tests at runtime and selects diverse subset.
    
    Usage: robot --listener realtime_diverse_listener.py:50 tests/
    """
    ROBOT_LISTENER_API_VERSION = 3
    ROBOT_LISTENER_PRIORITY = 50  # run after DataDriver if present
    
    def __init__(self, k=50):
        self.k = int(k)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self._processed_suites = set()
    
    def start_suite(self, data, result):
        suite_id = id(data)
        if suite_id in self._processed_suites:
            return
        self._processed_suites.add(suite_id)
        
        if not data.tests:
            return
        
        # Generate text descriptions
        descriptions = []
        for test in data.tests:
            parts = [test.name]
            parts.extend(str(t) for t in test.tags)
            for item in test.body:
                if hasattr(item, 'name') and item.name:
                    parts.append(item.name)
                    parts.extend(str(a) for a in item.args 
                                if not str(a).startswith('${'))
            descriptions.append(" ".join(parts))
        
        if len(data.tests) <= self.k:
            return  # not enough tests to filter
        
        # Embed and select
        embeddings = self.model.encode(descriptions)
        selected_idx = self._farthest_point_sampling(embeddings, self.k)
        selected_set = set(selected_idx)
        
        original = len(data.tests)
        data.tests = [t for i, t in enumerate(data.tests) 
                      if i in selected_set]
        print(f"* RealtimeSelector: {len(data.tests)}/{original} "
              f"tests in {data.name}")
    
    @staticmethod
    def _farthest_point_sampling(X, k):
        N = X.shape[0]
        selected = [np.random.randint(N)]
        min_dist = cosine_distances(X[selected[0]:selected[0]+1], X)[0]
        min_dist[selected[0]] = -1
        for _ in range(k - 1):
            next_idx = int(np.argmax(min_dist))
            selected.append(next_idx)
            new_d = cosine_distances(X[next_idx:next_idx+1], X)[0]
            min_dist = np.minimum(min_dist, new_d)
            min_dist[next_idx] = -1
        return selected
```

The tradeoff: model loading adds **~2 seconds** at startup (one-time cost for MiniLM-L6), and embedding N=1,000 descriptions takes roughly **0.5 seconds** on CPU. For suites under 5,000 tests, total overhead stays below 5 seconds—negligible compared to test execution time.

### Approach 3: CI/CD pipeline with cached embeddings and incremental updates

This production-grade approach uses cached embeddings that update incrementally when test files change, supports multiple selection strategies, and integrates with CI systems via environment variables.

```python
# ci_diverse_pipeline.py
import os
import json
import hashlib
import numpy as np
from pathlib import Path
from robot.api import TestSuite
from sentence_transformers import SentenceTransformer
import chromadb

class CIDiversePipeline:
    """Production CI/CD pipeline for diverse test selection."""
    
    def __init__(self, suite_path, db_path=".rf_vector_cache",
                 model_name="all-MiniLM-L6-v2"):
        self.suite_path = suite_path
        self.db_path = db_path
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            "test_cases", metadata={"hnsw:space": "cosine"}
        )
        self._file_hashes_path = Path(db_path) / "file_hashes.json"
    
    def _file_hash(self, path):
        return hashlib.md5(Path(path).read_bytes()).hexdigest()
    
    def _load_hashes(self):
        if self._file_hashes_path.exists():
            return json.loads(self._file_hashes_path.read_text())
        return {}
    
    def _save_hashes(self, hashes):
        self._file_hashes_path.write_text(json.dumps(hashes))
    
    def incremental_index(self):
        """Only re-embed tests from changed files."""
        old_hashes = self._load_hashes()
        new_hashes = {}
        changed_files = []
        
        for robot_file in Path(self.suite_path).rglob("*.robot"):
            h = self._file_hash(robot_file)
            new_hashes[str(robot_file)] = h
            if old_hashes.get(str(robot_file)) != h:
                changed_files.append(str(robot_file))
        
        if not changed_files:
            print("No test files changed, using cached embeddings")
            return
        
        # Remove old entries for changed files
        existing = self.collection.get(include=["metadatas"])
        ids_to_delete = [
            eid for eid, meta in zip(existing["ids"], existing["metadatas"])
            if meta.get("suite") in changed_files
        ]
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
        
        # Parse and embed only changed files
        for fpath in changed_files:
            suite = TestSuite.from_file_system(fpath)
            kw_map = build_keyword_map(suite)
            for test in suite.tests:
                text = test_to_text(test, kw_map)
                tid = hashlib.md5(
                    f"{fpath}::{test.name}".encode()
                ).hexdigest()
                emb = self.model.encode(text).tolist()
                self.collection.upsert(
                    ids=[tid], embeddings=[emb], documents=[text],
                    metadatas=[{"name": test.name, 
                               "tags": ",".join(str(t) for t in test.tags),
                               "suite": fpath}]
                )
        
        self._save_hashes(new_hashes)
        print(f"Re-indexed {len(changed_files)} changed files")
    
    def select(self, k=None, method="fps", tag_filter=None):
        """Select diverse tests, respecting optional tag filter."""
        k = k or int(os.environ.get("DIVERSE_K", "50"))
        
        where_filter = None
        if tag_filter:
            where_filter = {"tags": {"$contains": tag_filter}}
        
        all_data = self.collection.get(
            include=["embeddings", "metadatas"],
            where=where_filter
        )
        
        if not all_data["ids"]:
            return []
        
        vectors = np.array(all_data["embeddings"])
        names = [m["name"] for m in all_data["metadatas"]]
        k = min(k, len(names))
        
        selected_idx = farthest_point_sampling(vectors, k)
        selected_names = [names[i] for i in selected_idx]
        
        output_file = os.environ.get(
            "DIVERSE_OUTPUT", "selected_tests.json"
        )
        with open(output_file, 'w') as f:
            json.dump(selected_names, f)
        return selected_names
    
    def run(self, k=None, extra_args=None):
        """Full pipeline: index, select, execute."""
        self.incremental_index()
        selected = self.select(k)
        
        suite = TestSuite.from_file_system(self.suite_path)
        suite.visit(DiverseTestSelector._from_names(selected))
        
        args = {"output": "diverse_output.xml", "log": "diverse_log.html"}
        if extra_args:
            args.update(extra_args)
        suite.run(**args)

# CI usage with environment variables:
# DIVERSE_K=100 python -c "
#   from ci_diverse_pipeline import CIDiversePipeline
#   CIDiversePipeline('./tests/').run()
# "
```

For Jenkins, GitHub Actions, or GitLab CI integration, the pipeline wraps cleanly into shell commands:

```yaml
# .github/workflows/diverse-tests.yml
- name: Index and select diverse tests
  run: |
    python ci_diverse_pipeline.py index --suite ./tests/
    python ci_diverse_pipeline.py select --k ${{ vars.TEST_SUBSET_SIZE }}
- name: Run selected tests
  run: |
    robot --prerunmodifier diverse_selector.py:selected_tests.json \
          --output output.xml ./tests/
```

## Research landscape and what already exists

**No existing tool combines modern NLP embeddings with diversity-based selection for test suites.** The closest prior work is **FAST-R** (Cruciani et al., ICSE 2019), which models test cases as points in Euclidean space and selects evenly spread subsets—but it uses simple text shingling, not transformer embeddings. FAST-R demonstrated that diversity-based reduction scales to **one million test cases** in under 20 minutes, validating the core concept.

**Test2Vec** (Jabbar et al., 2022) embeds test execution traces via BiLSTM, achieving **41.8% improvement** over CodeBERT baselines for test prioritization, but it requires dynamic execution traces—inapplicable to keyword-driven frameworks without running tests first. **NNE-TCP** maps file-test relationships into embedding space, achieving **APFD of 0.74** (vs. 0.5 random baseline), but requires historical CI data. **Tscope** (ESEC/FSE 2022) uses BERT embeddings to detect redundant natural-language test cases but focuses on duplicate detection, not subset selection.

Commercial tools take a fundamentally different approach. **Launchable** (now CloudBees Smart Tests) and **Meta's Predictive Test Selection** use gradient-boosted decision trees on historical metadata—effective but requiring weeks of execution history for cold-start. No commercial tool uses embeddings or diversity algorithms. This proposed approach fills a unique niche: it works **immediately on any test suite** with zero historical data, requires no coverage instrumentation, and leverages the natural-language structure of Robot Framework keywords.

A 2025 systematic mapping study of diversity-based testing techniques (Elgendy, Wiley STVR) cataloged **167 papers** using 79 different similarity metrics across 22 types of software artifacts. The study confirms that diversity-based methods are well-validated academically but underutilized in practice. The specific combination of transformer embeddings + FPS + Robot Framework keyword parsing represents genuinely novel work.

## Conclusion

The recommended implementation path starts with **Approach 1** (offline indexing + PreRunModifier) using `all-MiniLM-L6-v2` for embeddings, **ChromaDB** for storage, and **farthest point sampling** for diversity selection. This combination delivers the best balance of simplicity, performance, and reliability. FPS's 2-approximation guarantee ensures that the minimum pairwise distance in the selected subset is at least half of optimal—strong coverage with zero tuning parameters beyond k.

For DataDriver suites, the critical insight is that selection must happen via **Listener v3 with lower priority than DataDriver**, not via PreRunModifier. The listener approach also works for standard suites, making it the universal integration mechanism when DataDriver is in the mix.

Three design decisions deserve attention going forward. First, **embedding quality for domain-specific test vocabulary** should be validated empirically—fine-tuning on test case corpora may yield meaningful improvements over the general-purpose MiniLM model. Second, **the correlation between semantic diversity and fault-detection diversity** is theoretically supported by the literature but should be measured on real defect data. Third, **facility location** (via `apricot-select`) may outperform FPS for suites where cluster representativeness matters more than boundary coverage, and the two algorithms can be offered as selectable strategies. The entire pipeline—parsing, embedding, selecting, and executing—adds under 10 seconds of overhead for a suite of 5,000 tests, making it practical for daily CI integration.