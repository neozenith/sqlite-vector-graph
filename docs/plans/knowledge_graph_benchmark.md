# Knowledge Graph Pipeline — Gap Analysis

A gap analysis of remaining work to build the **knowledge graph construction pipeline** that exercises all five of muninn's subsystems together in a GraphRAG pattern.

**Status:** Gap analysis. Updated 2026-02-15. All C prerequisites are complete. Documentation staleness resolved. Remaining work is the KG pipeline.

**Benchmark tasks** have been extracted to [`benchmark_backlog.md`](benchmark_backlog.md).

**Extraction & coalescing benchmark spec** (NER models, LLM APIs, Ollama, datasets, metrics) is in [`kg_extraction_benchmark_spec.md`](kg_extraction_benchmark_spec.md).

**Tokenisation & multilingual challenges** are in [`tokenisation_and_languages.md`](tokenisation_and_languages.md).

**Documentation staleness** has been extracted to [`documentation_staleness.md`](documentation_staleness.md).

---

## Table of Contents

1. [Implementation Status Summary](#implementation-status-summary)
2. [Gap 1: Knowledge Graph Pipeline — Not Started](#gap-1-knowledge-graph-pipeline--not-started)
3. [Resolved Prerequisites](#resolved-prerequisites)
4. [Remaining Open Questions](#remaining-open-questions)
5. [Original Plan Reference](#original-plan-reference)

---

## Implementation Status Summary

### C Extension — COMPLETE

All five subsystems are fully implemented with tests:

| Subsystem | TVF/Function | C Source | Python Tests | C Tests | Status |
|-----------|-------------|---------|-------------|---------|--------|
| HNSW vector index | `hnsw_index` | `hnsw_vtab.c`, `hnsw_algo.c` | `test_hnsw_vtab.py` | `test_hnsw_algo.c` | **Done** |
| Graph traversal | `graph_bfs`, `graph_dfs`, `graph_shortest_path`, `graph_components`, `graph_pagerank` | `graph_tvf.c` | `test_graph_tvf.py` | — | **Done** |
| Centrality | `graph_degree`, `graph_betweenness`, `graph_closeness` | `graph_centrality.c` | `test_graph_centrality.py` | — | **Done** |
| Community detection | `graph_leiden` | `graph_community.c` | `test_graph_community.py` | — | **Done** |
| Node2Vec | `node2vec_train()` | `node2vec.c` | `test_node2vec.py` | — | **Done** |

**Note:** The original plan called for Louvain first, then Leiden as an upgrade. Implementation went directly to Leiden (superior algorithm with connectivity guarantees), which is what Microsoft GraphRAG uses.

### Knowledge Graph Pipeline — NOT STARTED

| Component | Script | Status |
|-----------|--------|--------|
| Entity extraction (GLiNER + SVO + FTS5) | `kg_extract.py` | **Not started** |
| Entity coalescing | `kg_coalesce.py` | **Not started** |
| Gutenberg catalog + download | `kg_gutenberg.py` | **Not started** |

### Documentation — RESOLVED

See [`documentation_staleness.md`](documentation_staleness.md). All staleness items have been addressed.

---

## Gap 1: Knowledge Graph Pipeline — Not Started

This is the largest remaining gap. All C primitives exist, but the Python pipeline to construct a knowledge graph is entirely unbuilt.

### 1a. Entity Extraction Pipeline

**Files needed:**
- `benchmarks/scripts/kg_extract.py` — Main extraction pipeline (GLiNER + SVO + FTS5)
- `benchmarks/scripts/kg_coalesce.py` — Entity resolution via HNSW-based blocking
- `benchmarks/scripts/kg_gutenberg.py` — Project Gutenberg catalog + download + caching

**Python dependencies needed:**
- `gliner` — Zero-shot NER
- `spacy` + `en_core_web_lg` (or `en_core_web_md`) — Dependency parsing + SVO
- `textacy` — SVO triple extraction helper
- `sentence-transformers` — Already in benchmark deps
- `requests` — Already available

**Output:** Cached SQLite databases in `benchmarks/kg/{gutenberg_id}.db`, one per book.

### 1b. Makefile Targets

**Needed in `benchmarks/Makefile`:**
```makefile
kg-extract:           ## Extract KG from Wealth of Nations (reference)
kg-extract-random:    ## Extract KG from a random Gutenberg economics book
kg-extract-book:      ## Extract KG from a specific book (BOOK_ID=...)
kg-coalesce:          ## Entity resolution + dedup (all cached KGs)
```

---

## Resolved Prerequisites

These items from the original plan's "Prerequisites" section are now complete:

| Prerequisite | Original Status | Current Status |
|-------------|----------------|----------------|
| Betweenness centrality TVF | P1 | **Done** — `graph_betweenness` in `graph_centrality.c` |
| Louvain/Leiden community detection | P2/P3 | **Done** — `graph_leiden` in `graph_community.c` (skipped Louvain, went directly to Leiden) |
| Degree centrality TVF | P4 | **Done** — `graph_degree` in `graph_centrality.c` |
| Closeness centrality TVF | P4 | **Done** — `graph_closeness` in `graph_centrality.c` |
| Python + Node.js wrappers | Not in original plan | **Done** — `sqlite_muninn/` and `index.mjs`/`index.cjs` |
| CI/CD pipeline | Not in original plan | **Done** — Multi-platform builds, tests, docs deployment |
| Examples | Not in original plan | **Done** — 5 examples covering all subsystems |

---

## Remaining Open Questions

These questions from the original plan are **still open** (not resolved by implementation):

1. **Entity extraction quality**: How reliable is GLiNER zero-shot NER on 18th-century economic text? The manual seed list of ~50 key entities remains essential.

2. **Betweenness centrality scalability**: Brandes' O(VE) is fine for ~3K nodes / ~10K edges. At 100K+ nodes, approximate betweenness (random sampling) would be needed.

3. **REBEL vs SVO extraction**: Typed relations (REBEL, 1.6 GB) vs noisy SVO extraction (spaCy, 40 MB). Unclear if typed relations improve retrieval enough.

4. **Entity coalescing threshold**: Cosine similarity threshold for entity merging. Too low -> over-merging, too high -> fragmentation.

5. **Graph density impact**: Sparse (high-confidence edges only) vs dense (including co-occurrence edges) — different retrieval characteristics.

6. **Random book text quality**: NER/SVO pipeline performance on texts from different eras (1776-1920s).

7. **Cross-book concept alignment**: Can Node2Vec find structurally equivalent concepts across different books?

8. **Book length normalization**: Small books (Communist Manifesto, ~30 pages) vs large (WoN, ~400 pages).

9. **Temporal edge simulation**: Static text doesn't have temporal data — simulate via progressive agent sessions?

10. **Gutendex API reliability**: Third-party API; fallback to offline CSV catalog?

---

## Original Plan Reference

The original research, entity extraction approaches, HuggingFace model recommendations, dataset descriptions, and key references have been preserved below for reference.

<details>
<summary><strong>Click to expand: Original Research & Plan Details</strong></summary>

### Vision

The knowledge graph pipeline tests the *composition* of muninn's subsystems — the workflow where a vector similarity search provides an entry point into a graph, and graph traversal expands the context. This is the core pattern behind GraphRAG, where:

1. **VSS entry point**: Query vector -> HNSW search -> find nearest graph node
2. **Graph expansion**: From that node -> BFS/DFS to explore k-hop neighbors
3. **Context assembly**: Collect text from traversed nodes as retrieval context
4. **Graph analytics**: Betweenness centrality identifies bridge concepts; PageRank finds authoritative nodes
5. **Hierarchical retrieval** (advanced): Leiden communities -> supernode embeddings -> multi-level search

muninn is uniquely positioned here — it's the only SQLite extension combining HNSW + graph TVFs + centrality + community detection + Node2Vec in a single shared library.

### Research: State of the Art

#### Vector-Seeded Graph Traversal (The GraphRAG Pattern)

The core insight behind GraphRAG is that **vector similarity search alone misses relational context**. A query about "How does the division of labour affect wages?" might find passages about wages but miss causally-linked concepts like "productivity" or "market price" that are connected via graph edges but not semantically similar to the query.

The pattern works in stages:

```
Query "How does division of labour affect wages?"
  |
  +- [1] VSS Entry Point ------- HNSW search -> nearest passage nodes
  |                                (finds: "wages", "labour", "price")
  |
  +- [2] Graph Expansion ------- BFS 2-hop from entry points
  |                                (discovers: "productivity", "market price",
  |                                 "rent", "profit" via CAUSES/COMPOSED_OF edges)
  |
  +- [3] Centrality Ranking ---- Betweenness centrality scores bridge nodes
  |                                (ranks "market price" highest -- it bridges
  |                                 labour/wages cluster to rent/profit cluster)
  |
  +- [4] Context Assembly ------ Collect passage text from traversed+ranked nodes
                                   (richer context than VSS alone)
```

**Key research systems:**

- **Microsoft GraphRAG (2024)**: Extracts KG from documents via LLM, builds community hierarchy (Leiden algorithm), generates community summaries, retrieval via summary search -> drill into communities. The gold standard for hierarchical GraphRAG.
- **HybridRAG (2024)**: Combines vector DB + graph DB with joint scoring (vector similarity + graph distance). Exactly what muninn enables in a single SQLite extension.
- **NaviX (VLDB 2025)**: Native dual indexing in the DB kernel — vector index + graph index with pruning strategies that leverage both simultaneously. The research frontier.
- **Deep GraphRAG (2025)**: Multi-hop reasoning via graph-guided evidence chains starting from VSS results.

#### Entity Coalescing and Synonym Merging

A critical challenge in KG construction: "division of labour", "division of labor", and "the labour is divided" all refer to the same concept. Without merging, the graph becomes fragmented.

**Coalescing Pipeline:**

```
Raw Entities  ->  Blocking  ->  Matching  ->  Merging  ->  Canonical Graph
(many dupes)    (group by    (pairwise    (resolve     (clean nodes)
                 similarity)  comparison)  clusters)
```

- **Stage 1 -- Blocking**: HNSW-based embedding similarity (cosine < 0.3) or token overlap (>50%)
- **Stage 2 -- Matching**: Cascade: exact match -> fuzzy (Jaro-Winkler > 0.85) -> embedding similarity -> WordNet synsets
- **Stage 3 -- Merging**: Leiden clustering on match graph to prevent over-merging, select canonical form

#### Temporal Knowledge Graphs

Bi-temporal model (valid time + transaction time) per edge. No C extension changes needed — pure schema pattern on edge tables plus application-level query construction.

### Entity Extraction Approaches

1. **GLiNER zero-shot NER** — Custom entity types at inference time. Recommended: `urchade/gliner_small-v2.1` (350 MB).
2. **spaCy SVO extraction** — Dependency parse -> (subject, verb, object) triples -> graph edges.
3. **FTS5/BM25 concept discovery** — Zero-model approach using SQLite's built-in FTS5 to identify important terms.

**Hybrid Pipeline**: All three combined -> union -> coalesce -> edge construction -> output SQLite KG.

### HuggingFace Model Recommendations

| Tier | Entity Extraction | Relation Extraction | Node Embeddings | Total Size |
|------|------------------|-------------------|-----------------|------------|
| Tier 1 (Minimal) | GLiNER small | spaCy SVO | MiniLM-L6-v2 | < 500 MB |
| Tier 2 (Better) | GLiNER medium | REBEL | BGE-base-en-v1.5 | < 2.5 GB |
| Tier 3 (Fastest) | spaCy small | spaCy SVO | potion-base-8M | < 100 MB |

### Dataset: Economics Texts from Project Gutenberg

Primary: Wealth of Nations (Gutenberg #3300, ~2,500 passage chunks). Additional: random economics books from Gutendex API for generalization testing.

### Key References

- Microsoft GraphRAG (2024), Deep GraphRAG (2025), HybridRAG (2024), NaviX (VLDB 2025)
- Multi-Scale Node Embeddings (2024), Node2Vec (Grover & Leskovec 2016)
- Louvain (Blondel 2008), Leiden (Traag 2019), Brandes Betweenness (2001)
- GLiNER (NAACL 2024), REBEL (ACL 2021)
- Vesper-Memory (2025), Zep/Graphiti (arXiv:2501.13956)

### Implementation Sketch — File Structure

```
benchmarks/
  scripts/
    kg_extract.py              # Entity/relation extraction pipeline
    kg_coalesce.py             # Entity resolution + synonym merging
    kg_gutenberg.py            # Project Gutenberg catalog + download + caching
  kg/                          # Cached knowledge graphs (one SQLite DB per book)
  texts/                       # Cached plain text downloads
  vectors/kg_*.npy             # Pre-computed entity embeddings per book
```

</details>
