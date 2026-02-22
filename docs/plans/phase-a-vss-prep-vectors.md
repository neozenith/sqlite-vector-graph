# Phase A: VSS Prep Vectors Update — GGUF Models

## Context

The benchmark prep pipeline currently uses `sentence-transformers` (Python) to precompute
document embeddings. This phase replaces it with GGUF models run via `llama-cpp-python` (with
Metal acceleration). The key improvements:

1. **Unified model ecosystem** — GGUF models match what muninn uses natively in C
2. **Asymmetric embedding support** — nomic and BGE-Large use different prefixes for queries vs documents
3. **Separate query embeddings** — precompute query vectors for proper recall measurement
4. **Drop sentence-transformers dependency** — the prep pipeline becomes GGUF-only

## GGUF Model Set

| Name | GGUF File | Source | Dim | Quant | Size | Doc Prefix | Query Prefix |
|------|-----------|--------|-----|-------|------|------------|--------------|
| MiniLM | all-MiniLM-L6-v2.Q8_0.gguf | leliuga | 384 | Q8_0 | ~36 MB | _(none)_ | _(none)_ |
| NomicEmbed | nomic-embed-text-v1.5.Q8_0.gguf | nomic-ai | 768 | Q8_0 | ~146 MB | `search_document: ` | `search_query: ` |
| BGE-Large | bge-large-en-v1.5-q8_0.gguf | CompendiumLabs | 1024 | Q8_0 | ~342 MB | _(none)_ | `Represent this sentence for searching relevant passages: ` |

## Output Files (New Naming)

```
benchmarks/vectors/
  {model}_{dataset}_docs.npy       # Document embedding pool (float32, shape [N_max, dim])
  {model}_{dataset}_queries.npy    # Query embedding pool (float32, shape [Q_max, dim])
  {dataset}_queries.json           # Query texts + source metadata
```

## Pool Strategy — Prep Maximizes, Benchmark Samples

The prep step creates **maximum-size embedding pools**. Each benchmark trial randomly
samples from these pools. This separation means:
- Embedding (expensive) happens once
- Benchmarks can run many trials with different random subsets cheaply

### Document Pools (per model x dataset)

| Dataset | Source | Pool Size |
|---------|--------|-----------|
| ag_news | HuggingFace train split | up to max(VSS_SIZES) = 100,000 |
| wealth_of_nations | Gutenberg #3300, 256-token chunks | ~2,500 |

### Query Pools (per model x dataset)

| Dataset | Source | Pool Size | Rationale |
|---------|--------|-----------|-----------|
| ag_news | HuggingFace **test** split (disjoint from train) | **7,600** | Full test split, completely separate from docs |
| wealth_of_nations | First sentence of each document chunk | **~2,500** | One query per chunk, textually distinct |

Query texts stored in `{dataset}_queries.json` (model-independent, generated once):
```json
{
  "dataset": "ag_news",
  "source": "ag_news HuggingFace test split (full, 7600 rows)",
  "n_queries": 7600,
  "queries": ["Reuters - Short-sellers...", "AP - Scientists at..."]
}
```

### How Each Benchmark Run Samples From Pools

Each VSS treatment run (one invocation of the harness) does:
1. **Generate a random seed** (or accept one via CLI `--seed`)
2. **Sample N docs** randomly from the doc pool using that seed
3. **Sample M queries** randomly from the query pool (M = `N_QUERIES`, default 100)
4. Compute brute-force ground truth on this specific (N, M) subset
5. Insert the N sampled docs into the VSS index
6. Search with the M sampled queries
7. Compute recall
8. **Store the seed in the JSONL result** (but NOT in `permutation_id`)

Repeated benchmark runs -> different seeds -> accumulating JSONL records -> analysis
computes mean +/- std across runs sharing the same `permutation_id`.

**Constants in `common.py`:**
- `N_QUERIES = 100` — queries sampled per run (M)
- `VSS_SIZES` — N values to benchmark (100, 500, 1K, 5K, 10K, 50K, 100K)

**Seed strategy:**
- Default: `random.randint(0, 2**32 - 1)` — unique each run
- Optional: `--seed 42` CLI flag for reproducibility
- Stored in JSONL output: `{"seed": 42, "permutation_id": "vss_muninn-hnsw_MiniLM_...", ...}`
- `permutation_id` remains `vss_{engine}_{model}_{dataset}_n{N}` — no seed component

## Recall Measurement — Full Data Flow

### Current Approach (what vss.py does today)

**Prep time** (`vectors.py`):
- Load ag_news *training* texts -> embed ALL with `SentenceTransformer.encode()` -> `MiniLM_ag_news.npy` shape `[N, 384]`
- Single embedding pass. No query/doc distinction.

**Benchmark setup** (`vss.py:setup()`):
1. `_load_vectors("MiniLM", "ag_news", N=1000)` -> loads first 1000 rows -> `vectors: {1: [...], 2: [...], ..., 1000: [...]}`
2. Pick 100 random rowids as query IDs: `query_ids = [42, 337, 678, ...]`
3. For each query_id, call `_brute_force_knn(vectors[query_id], vectors, K=10)`:
   - The query vector is **identical** to the document vector at that rowid
   - Distance to itself = 0 -> the query document is always rank 1 in ground truth
4. Ground truth = 100 sets of 10 rowids each

**Benchmark run** (`vss.py:run()`):
1. Insert all 1000 doc vectors into VSS index
2. For each query_id: search index with `vectors[query_id]` -> K result rowids
3. Recall = avg(|search_results intersect ground_truth| / K)

**Limitations:**
- **Self-retrieval**: queries ARE documents. Each query has a guaranteed distance-0 match in the index.
- **No asymmetric support**: same vector for doc and query, so prefixed models can't be tested properly.
- **Unrealistic**: real search queries are not corpus documents.

---

### New Approach — Pool-Based Random Sampling

#### Prep Time — Create Maximum-Size Pools

**Step 1: Query text generation** (once per dataset, model-independent):

For **ag_news**:
- Load HuggingFace `ag_news` *test* split (7,600 rows, disjoint from training docs)
- Save ALL 7,600 texts -> `ag_news_queries.json`

For **wealth_of_nations**:
- Load all ~2,500 document chunks (same chunking as doc pool)
- Extract first sentence of EACH chunk as a query text -> `wealth_of_nations_queries.json`
- ~2,500 queries, one per chunk (textually distinct from full chunk)

**Step 2: Document embedding pool** (per model x dataset):
```
For NomicEmbed + ag_news:
  texts = [ag_news TRAIN split, up to max(VSS_SIZES) = 100K]
  prefixed = ["search_document: " + t for t in texts]
  doc_pool = llm.embed(prefixed)  -> shape [100000, 768]
  normalize(doc_pool)
  np.save("NomicEmbed_ag_news_docs.npy", doc_pool)
```

**Step 3: Query embedding pool** (per model x dataset):
```
For NomicEmbed + ag_news:
  queries = load("ag_news_queries.json")["queries"]   # 7600 query texts
  prefixed = ["search_query: " + q for q in queries]
  query_pool = llm.embed(prefixed)  -> shape [7600, 768]
  normalize(query_pool)
  np.save("NomicEmbed_ag_news_queries.npy", query_pool)
```

For **MiniLM** (symmetric, no prefixes):
- doc_prefix = "" -> documents embedded as-is
- query_prefix = "" -> queries embedded as-is
- Still produces separate `_docs.npy` and `_queries.npy` (same code path, prefix is just empty)

#### Benchmark Setup — Random Sampling + Ground Truth

Each benchmark trial randomly samples from the precomputed pools. Ground truth is computed
on each specific sample because the doc subset changes.

**Example: NomicEmbed + ag_news + N=1000, trial seed=42**

```python
def setup(self, conn, db_path):
    # 1. Load full pools
    doc_pool = np.load("NomicEmbed_ag_news_docs.npy")      # shape [100000, 768]
    query_pool = np.load("NomicEmbed_ag_news_queries.npy")  # shape [7600, 768]

    # 2. Sample N=1000 docs randomly from pool
    rng = np.random.default_rng(seed=self._seed)  # random or CLI-provided
    doc_indices = rng.choice(len(doc_pool), size=1000, replace=False)
    doc_vectors = doc_pool[doc_indices]    # shape [1000, 768]
    # doc_indices maps position -> original pool index (used as rowid)

    # 3. Sample M=100 queries randomly from pool
    query_indices = rng.choice(len(query_pool), size=100, replace=False)
    query_vectors = query_pool[query_indices]  # shape [100, 768]

    # 4. Brute-force ground truth on THIS specific sample
    #    For each query, find top-K=10 among the 1000 sampled docs
    ground_truth = []
    for q in range(100):
        dists = np.sum((doc_vectors - query_vectors[q]) ** 2, axis=1)  # [1000]
        top_k_positions = np.argsort(dists)[:10]
        # Map back to rowids (1-based position in the sampled doc set)
        top_k_rowids = {pos + 1 for pos in top_k_positions}
        ground_truth.append(top_k_rowids)
```

**Key properties:**
- Query vectors do NOT exist in the document set -> no distance-0 self-match
- Each harness invocation uses a unique seed -> re-running accumulates results in JSONL
- Ground truth is specific to this (N docs, M queries) sample
- The seed is stored in the JSONL record for reproducibility

#### Benchmark Run — Approximate Search + Recall

```python
def run(self, conn):
    # 1. Insert 1000 sampled doc vectors into VSS index
    for pos, vec in enumerate(doc_vectors):
        conn.execute("INSERT INTO bench_vec(rowid, vector) VALUES(?, ?)",
                     (pos + 1, pack_vector(vec)))

    # 2. Search with each of the 100 sampled query vectors
    search_results = []
    for q in range(len(query_vectors)):
        rows = conn.execute(
            "SELECT rowid FROM bench_vec WHERE vector MATCH ? AND k = ?",
            (pack_vector(query_vectors[q]), 10)
        ).fetchall()
        search_results.append({r[0] for r in rows})

    # 3. Recall per query
    recall_per_query = [len(sr & gt) / len(gt) for sr, gt in zip(search_results, ground_truth)]
    avg_recall = sum(recall_per_query) / len(recall_per_query)
```

#### Repeated Runs via Harness Re-Invocation

The benchmark harness does NOT loop trials internally. Re-running the benchmark IS
how you get repeated trials. Results accumulate in `benchmarks/results/*.jsonl`:

```
Run 1: python -m benchmarks.harness.cli benchmark --id vss_muninn-hnsw_MiniLM_...
  -> seed=738291, recall=0.93  -> appended to results/vss.jsonl

Run 2: python -m benchmarks.harness.cli benchmark --id vss_muninn-hnsw_MiniLM_...
  -> seed=194572, recall=0.95  -> appended to results/vss.jsonl

Run 3: python -m benchmarks.harness.cli benchmark --id vss_muninn-hnsw_MiniLM_...
  -> seed=503814, recall=0.94  -> appended to results/vss.jsonl
```

Each JSONL record includes the seed alongside all other metrics:
```json
{
  "permutation_id": "vss_muninn-hnsw_MiniLM_ag-news_n1000",
  "seed": 738291,
  "recall_at_k": 0.93,
  "search_latency_ms": 0.19,
  "insert_rate_vps": 45000.0
}
```

Analysis scripts group by `permutation_id` and compute mean +/- std across runs.

#### Why Random Sampling Matters at Scale

| N | Doc Pool | Sampled | Query Pool | Sampled | Issue with fixed "first N" |
|---|----------|---------|------------|---------|---------------------------|
| 100 | 100K | 100 random | 7600 | 100 random | First 100 docs are always the same subset |
| 1000 | 100K | 1000 random | 7600 | 100 random | Fixed ordering biases toward early rows |
| 100K | 100K | all (no sampling) | 7600 | 100 random | At max N, doc sampling degenerates to full pool |

When N = pool size, doc sampling returns the full pool (no randomness). Query sampling
still provides variance.

#### Distance Metric Consistency

All vectors are L2-normalized during prep (`||v|| = 1`). For unit vectors:
```
L2^2(a, b) = ||a - b||^2 = ||a||^2 + ||b||^2 - 2(a.b) = 2 - 2*cos(a, b)
```
So **top-K by L2 = top-K by cosine similarity** (monotonic transformation).

Both the brute-force ground truth and the VSS index use L2 distance -> apples-to-apples comparison.
The HNSW index is created with `metric='l2'` and the brute-force uses `sum((a-b)^2)`.

#### What Recall Measures

Recall@K answers: **"Of the K documents that exact search would return, how many did the approximate index find?"**

- Recall = 1.0 -> the index found all the same documents as brute-force (perfect)
- Recall = 0.8 -> the index missed 2 out of 10 exact top-K documents
- This measures **index quality**, not **model quality** — the model's relevance ranking is taken as ground truth

## Changes by File

### 1. `benchmarks/harness/common.py`
- Replace `EMBEDDING_MODELS` dict with GGUF-aware model definitions:
  ```python
  EMBEDDING_MODELS = {
      "MiniLM": {
          "gguf_filename": "all-MiniLM-L6-v2.Q8_0.gguf",
          "dim": 384,
          "params": "22M",
          "doc_prefix": "",
          "query_prefix": "",
      },
      "NomicEmbed": {
          "gguf_filename": "nomic-embed-text-v1.5.Q8_0.gguf",
          "dim": 768,
          "params": "137M",
          "doc_prefix": "search_document: ",
          "query_prefix": "search_query: ",
      },
      "BGE-Large": {
          "gguf_filename": "bge-large-en-v1.5-q8_0.gguf",
          "dim": 1024,
          "params": "335M",
          "doc_prefix": "",
          "query_prefix": "Represent this sentence for searching relevant passages: ",
      },
  }
  ```
- Keep `N_QUERIES = 100` (now means "queries sampled per run from the pool", not pool size)
- Ensure `GGUF_MODELS_DIR` is still exported

### 2. `benchmarks/harness/prep/gguf_models.py`
- Update `GGUF_MODELS` list to include BGE-Large entry:
  ```python
  {
      "name": "BGE-Large",
      "filename": "bge-large-en-v1.5-q8_0.gguf",
      "url": "https://huggingface.co/CompendiumLabs/bge-large-en-v1.5-gguf/resolve/main/bge-large-en-v1.5-q8_0.gguf",
      "params": "335M",
      "dim": "1024",
      "quant": "Q8_0",
  }
  ```
- Add `doc_prefix` and `query_prefix` fields to each model entry
- This module remains responsible for downloading GGUF files only (prep gguf)

### 3. `benchmarks/harness/prep/vectors.py` (Major rewrite)
- **Remove** `sentence-transformers` import, replace with `llama_cpp`
- **Remove** `SentenceTransformer` model loading
- **New embedding function** using llama-cpp-python:
  ```python
  from llama_cpp import Llama

  def _embed_texts(model_path: str, texts: list[str], prefix: str = "") -> np.ndarray:
      llm = Llama(model_path=str(model_path), embedding=True, n_gpu_layers=-1, verbose=False)
      prefixed = [prefix + t for t in texts] if prefix else texts
      embeddings = llm.embed(prefixed)
      # Normalize to unit length
      arr = np.array(embeddings, dtype=np.float32)
      norms = np.linalg.norm(arr, axis=1, keepdims=True)
      arr = arr / np.maximum(norms, 1e-12)
      return arr
  ```
- **New query text loading** functions:
  - `_load_query_texts_ag_news()` — load FULL test split (7,600 texts)
  - `_load_query_texts_wealth_of_nations(doc_chunks)` — extract first sentence of ALL chunks (~2,500)
- **Updated prep loop** — creates maximum pools:
  1. Generate query texts for each dataset -> `{dataset}_queries.json` (once, model-independent)
  2. For each model: load GGUF model once via llama-cpp-python
  3. For each dataset:
     - Embed ALL doc texts with doc_prefix -> `{model}_{dataset}_docs.npy` (full pool)
     - Embed ALL query texts with query_prefix -> `{model}_{dataset}_queries.npy` (full pool)
  4. Unload model
- **Updated VectorPrepTask**: outputs list now includes both `_docs.npy` and `_queries.npy`
- **Updated status display**: show doc pool size, query pool size, and cache status
- **EMBEDDING_MODELS reference**: read model info from unified config in `common.py`

### 4. `benchmarks/harness/treatments/vss.py` (Significant refactor)

**Loading:**
- `_load_vectors()` -> `_load_pool(model, dataset, kind)` — loads `_docs.npy` or `_queries.npy`
- Returns full `np.ndarray` pool (not dict)

**New fields on VSSTreatment:**
- `self._doc_pool: np.ndarray | None` — full doc embedding pool
- `self._query_pool: np.ndarray | None` — full query embedding pool
- Remove `self._vectors` (dict), `self._query_ids` (list)

**setup() flow (single trial per invocation):**
1. Generate or accept a random seed
2. Load doc pool and query pool from `.npy` files
3. `rng.choice(len(doc_pool), size=N, replace=False)` -> doc sample indices
4. `rng.choice(len(query_pool), size=M, replace=False)` -> query sample indices
5. Compute brute-force ground truth on this specific (N, M) sample
6. Store sampled vectors + ground truth for `run()` to use

**run() flow (single trial):**
1. Insert N sampled doc vectors into VSS index
2. Search with M sampled query vectors
3. Compute recall vs ground truth
4. Return single-run metrics including the seed

**Reporting (per JSONL record):**
- `seed` — the RNG seed used for this run's sampling
- `recall_at_k` — single recall value for this run
- `search_latency_ms` — mean search latency across M queries
- `insert_rate_vps` — vectors per second during insert

**_brute_force_knn():**
- Now takes `query_vec: np.ndarray` and `doc_matrix: np.ndarray` (vectorized numpy)
- Returns set of rowids (1-based positions in the sampled doc set)

**_compute_ground_truth():**
- Takes `doc_vectors: np.ndarray [N, dim]` and `query_vectors: np.ndarray [M, dim]`
- Vectorized: `dists = np.sum((docs[None,:,:] - queries[:,None,:]) ** 2, axis=2)` -> `[M, N]`
- `np.argsort(dists, axis=1)[:, :K]` -> top-K indices per query

### 5. `benchmarks/harness/prep/__init__.py`
- No structural changes needed (exports remain the same)

### 6. `benchmarks/harness/cli.py`
- Update `prep vectors` to reference new model names if needed
- The `--model` filter should work with "MiniLM", "NomicEmbed", "BGE-Large"

### 7. `benchmarks/harness/tests/test_cli_prep.py`
- Update expected model names in status output assertions
- Update expected status table headers if format changes

### 8. `benchmarks/harness/tests/test_prep.py`
- Update `test_vector_prep_tasks_count()` — now 3 models x 2 datasets = 6 tasks (same count, different models)
- Update vector status display assertions for new column names
- Add test for query vector presence in status output

### 9. Remove `sentence-transformers` dependency
- Remove from `pyproject.toml` if listed there
- Update any install documentation referencing sentence-transformers
- Add `llama-cpp-python` to benchmark dependencies

## Installation Step

```bash
CMAKE_ARGS="-DGGML_METAL=on" uv pip install llama-cpp-python
```

## Verification

1. **Prep GGUF models**: `python -m benchmarks.harness.cli prep gguf --status` — all 3 models READY
2. **Prep vectors**: `python -m benchmarks.harness.cli prep vectors --status` — shows doc + query status
3. **Prep vectors (small)**: `python -m benchmarks.harness.cli prep vectors --model MiniLM --dataset wealth_of_nations` — generates both `MiniLM_wealth_of_nations_docs.npy` and `MiniLM_wealth_of_nations_queries.npy`
4. **Run tests**: `make -C benchmarks/harness test-quick`
5. **Run a small benchmark**: `python -m benchmarks.harness.cli benchmark --id vss_muninn-hnsw_MiniLM_wealth-of-nations_n100` — verify recall computed correctly

## Implementation Steps

1. Install `llama-cpp-python` with Metal
2. Add BGE-Large to `gguf_models.py` catalog + download it
3. Update `EMBEDDING_MODELS` in `common.py` with GGUF model definitions
4. Rewrite `vectors.py` embedding generation (doc + query pool vectors)
5. Update `vss.py` to consume separate query vectors with pool-based random sampling
6. Update tests
7. Run `prep vectors --dataset wealth_of_nations` end-to-end to verify
8. Run `prep vectors --status` to verify all outputs
