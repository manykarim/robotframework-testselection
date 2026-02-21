# ADR-002: Embedding Model and Vector Storage Selection

## Status

Proposed

## Date

2026-02-21

## Context

The Vector-Based Diverse Test Selection system needs to embed Robot Framework test case text representations into vector space so that diversity algorithms (FPS, k-Medoids, DPP, facility location) can compute meaningful distances between tests. The embedding and storage subsystem must satisfy several constraints:

1. **Offline operation** -- The pipeline must run without cloud API calls. Many CI environments have restricted or no internet access, and depending on a cloud embedding API introduces cost, latency, and a runtime dependency on third-party uptime.
2. **CPU-only execution** -- Not all CI runners have GPUs. The model must produce embeddings at interactive speed on commodity CPU hardware.
3. **Input characteristics** -- Robot Framework test descriptions, after keyword tree flattening and noise removal (DOM locators, variable placeholders), are typically 20-100 tokens of structured natural language. The embedding model's sequence limit must cover this range with headroom.
4. **Portable artifacts** -- Stage 1 (vectorization) and Stage 2 (selection) run as separate pipeline stages, potentially on different machines or CI jobs. The vector artifact format must be self-contained, require no running server process, and transfer efficiently between stages.
5. **Reproducibility** -- Embeddings must be deterministic for the same input text and model version, enabling cache invalidation based on content hashes of `.robot` files.
6. **Scale target** -- Hundreds to low thousands of test cases (typical Robot Framework suites). The system is not designed for millions of vectors.

We evaluated three categories of embedding models and five vector storage options against these constraints.

## Decision

### Embedding Model: `all-MiniLM-L6-v2` via sentence-transformers

We adopt `all-MiniLM-L6-v2` as the default embedding model. It is loaded and invoked through the `sentence-transformers` Python library.

### Primary Storage: NumPy NPZ files

We adopt NumPy compressed archives (`.npz`) as the primary portable artifact format for passing embeddings between pipeline stages. A companion `test_manifest.json` file carries test metadata (names, tags, suite paths, DataDriver flags).

### Optional Enhancement: ChromaDB for development workflows

ChromaDB may be used as an optional persistent indexed store for local development and exploratory queries. It is not required for the CI pipeline.

### Embedding Normalization

All embeddings are L2-normalized at encoding time (`normalize_embeddings=True` in sentence-transformers). This allows cosine similarity to be computed as a simple dot product, which is both faster and numerically stable.

## Rationale

### Model Comparison

Three embedding models were evaluated:

| Property | all-MiniLM-L6-v2 | all-mpnet-base-v2 | text-embedding-3-small (OpenAI) |
|---|---|---|---|
| Parameters | 22M | 109M | Unknown (cloud) |
| Dimensions | 384 | 768 | 1536 (truncatable to 512) |
| Max sequence length | 256 tokens | 384 tokens | 8191 tokens |
| Encoding speed (CPU, single) | Sub-millisecond | ~2-3ms | ~100-300ms (network) |
| Batch 1000 descriptions (CPU) | ~0.5s | ~2-4s | ~10-30s + API cost |
| Model load time | ~2s (one-time) | ~4s (one-time) | N/A |
| Offline capable | Yes | Yes | No |
| API key required | No | No | Yes |
| Install footprint | ~80MB (with sentence-transformers) | ~80MB (same library) | None (cloud) |
| MTEB retrieval score | Good | Best in class (sentence-transformers) | Higher overall |

**Why MiniLM wins for this use case:**

- **CPU sub-millisecond encoding** makes it practical to embed thousands of test descriptions in under a second, even on CI runners without GPUs.
- **No API key or cloud dependency** means zero configuration for new users and no runtime failure mode from network issues.
- **256-token sequence limit is sufficient.** Robot Framework test descriptions after keyword flattening are 20-100 tokens. The 256-token limit provides 2.5x-12x headroom. Descriptions exceeding this limit would indicate a flattening depth that is too aggressive (capturing implementation noise rather than semantic intent).
- **384 dimensions** are adequate for distinguishing test semantics at our scale. At N=5000 tests, 384 dimensions provide far more representational capacity than needed. The reduced dimensionality also means smaller artifact files and faster distance computations.
- **Batch encoding of 1000 descriptions takes ~0.5s on CPU**, and model load is a one-time ~2s cost. Total Stage 1 overhead for a 2000-test suite is 15-30s cold, 0s cached.

`all-mpnet-base-v2` is the recommended upgrade path when higher embedding quality is needed (e.g., if empirical evaluation shows MiniLM produces too many false "similar" pairs for a particular test vocabulary). The system supports model swapping via a `--model` configuration parameter.

`text-embedding-3-small` is ruled out for the default configuration due to the hard requirements for offline operation and no API keys. It remains a valid option for teams with API access who want to experiment with higher-quality embeddings.

### Storage Format Comparison

Five vector storage options were evaluated for the target scale:

| Feature | NPZ (NumPy) | FAISS | ChromaDB | Qdrant | LanceDB |
|---|---|---|---|---|---|
| Setup complexity | None (NumPy is a dependency) | Low | Very low | Low | Very low |
| Server process required | No | No | No (PersistentClient) | Optional | No |
| Portable artifact | Yes (single file) | Yes (manual save/load) | No (directory) | No (directory) | No (directory) |
| Native metadata | No (separate JSON) | No (manual) | Yes (MongoDB-style) | Yes (rich filtering) | Yes (DataFrame) |
| Raw vector retrieval | Direct array access | `reconstruct()` | `include=["embeddings"]` | `with_vectors=True` | Column access |
| Built-in embedding | No | No | Yes (MiniLM default) | Yes (FastEmbed) | Yes (registry) |
| HNSW indexing | No (brute force) | Yes | Yes | Yes | Yes |
| Install size | 0 (already required) | ~30MB | ~100MB | ~50MB | ~50MB |
| Artifact size (5000 tests, 384d) | ~5MB compressed | ~8MB | ~15MB (directory) | ~15MB (directory) | ~10MB (directory) |
| CI artifact transfer | Trivial (single file) | Simple (single file) | Complex (directory tree) | Complex (directory tree) | Moderate (directory) |

**Why NPZ as primary format:**

- **Zero additional dependencies.** NumPy is already a transitive dependency of sentence-transformers and scikit-learn. No new package is needed.
- **Self-contained portable artifact.** A single `embeddings.npz` file (~5MB for 5000 tests at 384 dimensions) transfers trivially between CI stages via artifact upload/download. No directory trees, no database files, no server processes.
- **Direct array access.** Stage 2 (selection) loads vectors with `np.load()` and immediately has an `(N, 384)` float32 matrix ready for distance computations. No query API, no deserialization overhead.
- **HNSW indexing is unnecessary.** Our selection algorithms (FPS, k-Medoids, facility location, DPP) all require access to the full pairwise distance structure or the complete vector matrix. Approximate nearest-neighbor search via HNSW does not help -- these algorithms are not nearest-neighbor queries. At N=5000, brute-force cosine distance computation takes milliseconds.

**Why ChromaDB as optional enhancement:**

- ChromaDB ships with `all-MiniLM-L6-v2` as its default embedding model, making it a natural fit for exploration.
- Its MongoDB-style metadata filtering (`{"tags": {"$contains": "smoke"}}`) enables ad-hoc queries during development: "show me tests similar to this one that have tag X."
- `PersistentClient` stores data locally with no server process, suitable for developer workstations.
- It is NOT used in the CI pipeline path because: (a) it adds ~100MB of dependencies to Stage 2, which only needs NumPy and scikit-learn; (b) its directory-based storage is harder to transfer between CI stages than a single NPZ file; (c) its HNSW index provides no benefit for our selection algorithms.

### Why not FAISS, Qdrant, LanceDB, or Weaviate?

- **FAISS** is the closest alternative to NPZ. It provides efficient similarity search and can save/load index files. However, it adds a compiled dependency (~30MB), requires manual metadata management, and its index formats are optimized for ANN queries we do not need. For our use case, raw NumPy arrays are simpler and equally fast.
- **Qdrant** is designed for production vector search services with rich filtering. At our scale (hundreds to low thousands), its client-server architecture is unnecessary overhead. It would be appropriate if the system scaled to millions of tests or needed real-time vector search.
- **LanceDB** has an appealing "SQLite for vectors" design with DataFrame integration. It is a strong second choice for teams wanting analytical queries over test embeddings. However, it adds a dependency without clear benefit over NPZ for the pipeline use case.
- **Weaviate** requires a running server process and has a ~500MB+ install footprint. It is overkill at this scale.

### Normalization Strategy

Embeddings are L2-normalized at encoding time. This is a deliberate architectural choice:

1. **Cosine similarity becomes a dot product.** For L2-normalized vectors, `cosine_similarity(a, b) = dot(a, b)`. This simplifies distance computations in Stage 2 and avoids redundant normalization in every algorithm.
2. **Consistent distance semantics.** All selection algorithms (FPS, k-Medoids, DPP, facility location) operate on cosine distance. Normalizing once at embedding time ensures consistency regardless of which algorithm is chosen at Stage 2.
3. **DPP kernel construction.** The DPP likelihood kernel `L = V_hat @ V_hat.T` requires normalized vectors. Pre-normalizing avoids a separate normalization step in the DPP code path.

## Consequences

### Positive

- **Zero-configuration embedding.** New users install `sentence-transformers` and get a working embedding pipeline with no API keys, no cloud accounts, and no model downloads beyond the automatic first-run fetch (cached thereafter by sentence-transformers in `~/.cache/torch/sentence_transformers/`).
- **Lightweight CI stages.** Stage 2 (selection) requires only `numpy` and `scikit-learn` -- no embedding model, no vector database. This makes the selection stage fast to install and execute (~0.3s for FPS on 5000 tests).
- **Portable artifacts.** The NPZ + JSON artifact pair is a self-contained, version-controlled snapshot. It can be committed to a repository, attached to a CI artifact store, or passed between stages without any runtime dependencies.
- **Artifact size is manageable.** At 384 dimensions and float32 precision, 5000 test embeddings occupy ~7.5MB uncompressed, ~5MB compressed. This is well within CI artifact size limits.
- **Deterministic pipeline.** Given the same `.robot` files and model version, the vectorization stage produces identical embeddings. Combined with content-hash-based cache keys, this enables reliable caching.
- **Separation of concerns.** The NPZ format decouples the embedding model from the selection algorithm. Stage 2 does not know or care which model produced the vectors -- it only sees an `(N, d)` matrix.

### Negative

- **Lower embedding quality than larger models.** MiniLM-L6 trades quality for speed. For test suites with highly specialized domain vocabulary (e.g., medical device testing, financial protocol testing), the general-purpose model may produce lower-quality similarity judgments than a domain-adapted or larger model. This risk is mitigated by the `--model` configuration parameter.
- **No incremental updates with NPZ.** The NPZ format is a monolithic file -- adding one new test requires re-writing the entire file. At our scale (milliseconds to write 5000 vectors), this is not a practical concern, but it would become one at 100K+ tests. ChromaDB or a database-backed store would be needed at that scale.
- **No built-in similarity search.** NPZ files support only full-matrix operations, not indexed queries like "find the 10 tests most similar to X." For that use case (e.g., change-impact analysis), ChromaDB or FAISS would be needed. This is an enhancement path, not a core pipeline requirement.
- **Model download on first run.** The first invocation of `SentenceTransformer("all-MiniLM-L6-v2")` downloads ~80MB from Hugging Face Hub. In air-gapped environments, the model must be pre-downloaded and referenced by local path. The sentence-transformers library supports this via `SentenceTransformer("/path/to/local/model")`.
- **256-token sequence limit.** If keyword tree resolution is set to high depth, some test descriptions may exceed 256 tokens and be silently truncated. The pipeline should log a warning when any description exceeds the model's max sequence length. In practice, resolve depth 0-2 keeps descriptions well under the limit.

### Future Path

1. **Model swapping via configuration.** The `--model` parameter already supports any sentence-transformers model name. Teams can experiment with `all-mpnet-base-v2` for higher quality or domain-specific fine-tuned models.
2. **Fine-tuned models.** For test suites with specialized vocabulary, a fine-tuned model trained on test-case pairs (similar/dissimilar) could improve embedding quality. The NPZ artifact format is model-agnostic, so upgrading the model requires only re-running Stage 1.
3. **ChromaDB integration for change-impact analysis.** A future ADR may promote ChromaDB from optional to recommended if the system adds a "find tests similar to changed code" feature that requires indexed similarity search.
4. **Dimensionality validation.** An automated check should verify that the embedding dimensionality in the NPZ file matches what Stage 2 expects, guarding against artifact/model version mismatches.
