# Phase B: Embedding Performance Benchmark Category

> **Prerequisite:** Phase A (VSS Prep Vectors Update) must be completed first.
> See: `docs/plans/phase-a-vss-prep-vectors.md`

## Purpose

The VSS benchmark category (Phase A) **excludes embedding costs** — vectors are precomputed so
it can isolate HNSW index quality and search performance.

This new `embed` category measures the **full text->embedding->search pipeline** including the
cost of running the GGUF model. It answers: "How fast can muninn embed and search text
end-to-end, compared to sqlite-lembed?"

## Competitive Landscape — Full Cross-Product

Two independent axes: **embedding function** x **search backend**.

**Embedding functions:**
- `muninn_embed(name, text)` — muninn's native llama.cpp wrapper
- `lembed(name, text)` — sqlite-lembed's llama.cpp wrapper

**Search backends:**
- muninn HNSW — approximate, O(log N) indexed search
- sqlite-vector PQ — approximate, O(N) product quantization scan, no index build cost
- sqlite-vec brute — exact, O(N) brute-force baseline

**Full 2x3 permutation matrix:**

| Slug | Embedding | Search | What it tests |
|------|-----------|--------|---------------|
| `muninn-embed+muninn-hnsw` | `muninn_embed` | muninn HNSW | Homogeneous muninn stack |
| `muninn-embed+sqlite-vector-pq` | `muninn_embed` | sqlite-vector PQ | muninn embed speed + competitor search |
| `muninn-embed+sqlite-vec-brute` | `muninn_embed` | sqlite-vec brute | muninn embed speed + exact baseline |
| `lembed+muninn-hnsw` | `lembed` | muninn HNSW | Competitor embed + muninn search |
| `lembed+sqlite-vector-pq` | `lembed` | sqlite-vector PQ | Competitor embed + competitor search |
| `lembed+sqlite-vec-brute` | `lembed` | sqlite-vec brute | Full competitor stack |

Both muninn and sqlite-lembed wrap llama.cpp, so embedding speed differences reveal
implementation overhead (batching, tokenization, memory management), not model differences.

By mixing embedding functions and search backends, we can isolate:
- **Embedding function overhead**: compare rows with same search backend, different embed fn
- **Search backend overhead**: compare rows with same embed fn, different search backend

## Benchmark Tasks

### Task 1: Model Load Time
```sql
-- muninn
INSERT INTO temp.muninn_models(name, model) SELECT 'MiniLM', muninn_embed_model('path.gguf');

-- sqlite-lembed
INSERT INTO temp.lembed_models(name, model) SELECT 'MiniLM', lembed_model_from_file('path.gguf');
```
**Metric:** `model_load_time_ms` — time to load GGUF into memory and initialize inference context.

### Task 2: Bulk Embedding + Insert
Load N texts from a dataset, embed each one live, and insert into the search backend.
No pre-cached vectors — embedding cost is included.

```sql
-- muninn HNSW: embed + index insert
INSERT INTO hnsw_table(rowid, vector)
  SELECT id, muninn_embed('MiniLM', content) FROM documents;

-- sqlite-lembed + sqlite-vector: embed + raw insert (no index)
INSERT INTO vec_table(id, embedding)
  SELECT id, lembed('MiniLM', content) FROM documents;
```

**Metrics:**
- `bulk_embed_insert_rate_vps` — vectors per second (embedding + insert)
- `bulk_embed_insert_total_s` — total wall time
- This includes model inference, tokenization, normalization, and storage

### Task 3: Incremental Trigger Embedding
Test per-row overhead when a trigger auto-embeds on INSERT. This simulates
an application that embeds new content as it arrives.

```sql
-- Setup trigger
CREATE TEMP TRIGGER auto_embed AFTER INSERT ON documents
BEGIN
  INSERT INTO hnsw_table(rowid, vector)
    VALUES (NEW.id, muninn_embed('MiniLM', doc_prefix || NEW.content));
END;

-- Benchmark: insert rows one at a time
INSERT INTO documents(id, content) VALUES (?, ?);
```

**Metrics:**
- `trigger_embed_latency_ms` — mean per-row latency (embed + insert via trigger)
- Compared across engines: muninn trigger vs sqlite-lembed trigger

### Task 4: Live Query Embedding + Search
Embed the query text at query time and search the index. This measures
end-to-end query latency including the embedding step.

```sql
-- muninn: embed query + HNSW search
SELECT v.rowid, v.distance
FROM hnsw_table v
WHERE v.vector MATCH muninn_embed('MiniLM', query_prefix || ?) AND k = 10;

-- sqlite-lembed + sqlite-vector: embed query + PQ scan
SELECT rowid, distance
FROM vector_quantize_scan('vec_table', 'embedding', lembed('MiniLM', ?), 10);
```

**Metrics:**
- `query_embed_search_latency_ms` — total time (embed query + search)
- `query_embed_only_ms` — time to embed the query text alone (subtract from above)
- `query_search_only_ms` — time for the search step alone

## Permutation Axes

```
embed_fn x search_backend x model x dataset x N
```

| Axis | Values |
|------|--------|
| `embed_fn` | `muninn_embed`, `lembed` |
| `search_backend` | `muninn-hnsw`, `sqlite-vector-pq`, `sqlite-vec-brute` |
| `model` | `MiniLM` (384d), `NomicEmbed` (768d), `BGE-Large` (1024d) |
| `dataset` | `ag_news`, `wealth_of_nations` |
| `N` | Subset of VSS_SIZES — probably `[100, 500, 1000, 5000]` (smaller range since embedding is slow) |

**Total permutations:** 2 x 3 x 3 x 2 x 4 = 144 (per run)

## Data Flow — No Pre-Cached Vectors

```
                    +---------------------------------------------+
  Prep (existing)   |  GGUF models downloaded to models/          |
                    |  Text datasets downloaded (ag_news, WoN)    |
                    |  NO vector caching for this category        |
                    +--------------------+------------------------+
                                         |
                    +--------------------v------------------------+
  Benchmark Run     |  1. Load GGUF model (timed)                 |
                    |  2. Load raw texts from dataset             |
                    |  3. Bulk embed + insert N texts (timed)     |
                    |  4. Trigger embed individual rows (timed)   |
                    |  5. Live embed query + search (timed)       |
                    |  6. Record all metrics + seed to JSONL      |
                    +---------------------------------------------+
```

## Query Handling for Asymmetric Models

The embedding benchmark uses the same prefix convention as the VSS benchmark:
- Documents prefixed with `doc_prefix` before embedding
- Queries prefixed with `query_prefix` before embedding
- The prefix is part of the timed embedding call (realistic cost)

## Results File

`benchmarks/results/embed.jsonl` — one record per run:
```json
{
  "permutation_id": "embed_muninn-embed+muninn-hnsw_MiniLM_wealth-of-nations_n500",
  "seed": 738291,
  "embed_fn": "muninn_embed",
  "search_backend": "muninn-hnsw",
  "model": "MiniLM",
  "dim": 384,
  "dataset": "wealth_of_nations",
  "n": 500,
  "model_load_time_ms": 120.5,
  "bulk_embed_insert_rate_vps": 850.3,
  "bulk_embed_insert_total_s": 0.588,
  "trigger_embed_latency_ms": 1.8,
  "query_embed_search_latency_ms": 3.2,
  "query_embed_only_ms": 2.1,
  "query_search_only_ms": 1.1,
  "recall_at_k": 0.94
}
```

## Recall in Embedding Benchmarks

Since there are no pre-cached vectors, recall is computed differently:
1. After bulk insert, the index contains N embedded documents
2. Embed M query texts live with query_prefix
3. For each query, also compute brute-force distances in Python (embed all N docs in Python
   to get ground truth vectors) — **or** use the pre-cached doc pool from the VSS prep step
   if available for the same model+dataset
4. Compare HNSW results to brute-force ground truth

The pre-cached doc pool from VSS prep can serve double duty here: recall ground truth
for the embedding benchmark without re-embedding in Python. The live-embedded vectors
from muninn/lembed should produce nearly identical vectors to the prep-cached ones
(same GGUF model, same llama.cpp backend).

## New Files

- `benchmarks/harness/treatments/embed.py` — `EmbedTreatment` class
- `benchmarks/harness/common.py` — add `EMBED_ENGINES` list and `EMBED_SIZES` list
- `benchmarks/harness/cli.py` — add `benchmark embed` subcommand
- `benchmarks/harness/tests/test_cli_benchmark.py` — add embed benchmark tests

## Implementation Steps

1. Add `EMBED_ENGINES` and `EMBED_SIZES` to `common.py`
2. Create `benchmarks/harness/treatments/embed.py` with `EmbedTreatment`
3. Add `benchmark embed` CLI subcommand
4. Add embed benchmark tests
5. Run `benchmark embed` with MiniLM + wealth_of_nations to verify
