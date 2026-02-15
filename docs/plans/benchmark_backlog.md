# Benchmark Backlog

Actionable task list of missing benchmarks, extracted from `knowledge_graph_benchmark.md` gap analysis and verified against the codebase on 2026-02-12.

---

## Status Key

| Status | Meaning |
|--------|---------|
| **Ready to run** | Code + Makefile targets exist, just needs execution |
| **Code needed** | New script or significant additions to existing scripts required |
| **Blocked** | Depends on other work being completed first |

---

## 1. Primitive Benchmarks — Ready to Run

These have code in `benchmark_graph.py` and Makefile targets, but **zero JSONL results** exist. Just need to be executed.

### 1a. Centrality benchmarks (degree, betweenness, closeness)

**Status:** Ready to run

Operations `degree`, `betweenness`, `closeness` are already in `ALL_OPERATIONS` (`benchmark_graph.py:61-63`). Engine wrappers exist (`run_graph_muninn` handles all three at lines 437-468). Makefile targets are defined.

**Run commands:**
```bash
make -C benchmarks graph-centrality-small        # N<=1K, ~5 min
make -C benchmarks graph-centrality-medium       # N<=10K, ~20 min
make -C benchmarks graph-centrality-large        # N<=100K, ~1 hr
make -C benchmarks graph-centrality-scale-free   # Barabasi-Albert topology
```

**Follow-up:** Verify `benchmark_graph_analyze.py` handles centrality operations in its chart generation. May need chart additions for centrality-specific metrics.

### 1b. Community detection benchmarks (Leiden)

**Status:** Ready to run

Operation `leiden` is in `ALL_OPERATIONS` (`benchmark_graph.py:64`). Engine wrapper exists (`run_graph_muninn` at line 469). Makefile targets are defined.

**Run commands:**
```bash
make -C benchmarks graph-community-small         # N<=1K
make -C benchmarks graph-community-medium        # N<=10K
make -C benchmarks graph-community-large         # N<=100K
make -C benchmarks graph-community-scale-free    # Barabasi-Albert topology
```

**Follow-up:** Verify results capture modularity score and community count. Check `benchmark_graph_analyze.py` handles Leiden-specific metrics in charts.

---

## 2. Primitive Benchmarks — Code Needed

### 2a. Node2Vec training time + quality

**Status:** Code needed

No benchmark script exists. Node2Vec is integration-tested (`test_node2vec.py`) but not benchmarked for performance or quality.

**What to measure:**
- Training time vs graph size (N nodes, E edges)
- Embedding quality vs p,q hyperparameters — sweep `[0.25, 0.5, 1.0, 2.0, 4.0]` x `[0.25, 0.5, 1.0, 2.0, 4.0]` (25 combos)
- Embedding dimensionality trade-offs (32 vs 64 vs 128 vs 384)
- Cluster separation quality (silhouette score on known communities)

**Files to create:**
- `benchmarks/scripts/benchmark_node2vec.py` — runner
- `benchmarks/scripts/benchmark_node2vec_analyze.py` — charts

**Makefile targets to add:**
```makefile
node2vec-small:        ## Node2Vec: small graphs (N<=1K)
node2vec-medium:       ## Node2Vec: medium graphs (N<=10K)
node2vec-sweep:        ## Node2Vec: p,q hyperparameter sweep
analyze-node2vec:      ## Analyze Node2Vec results -> charts
```

**Estimated effort:** Medium — standalone script, follows existing benchmark patterns.

### 2b. CTE baseline — missing operations

**Status:** Code partially exists

`run_graph_cte()` implements BFS, shortest_path, and components. It explicitly skips DFS, PageRank, degree, betweenness, closeness, and Leiden (line 576: "not expressible as CTE").

**Work needed:**
- DFS via CTE is possible (stack-based traversal with `ORDER BY depth DESC LIMIT 1` pattern) — implement or document as infeasible
- Degree via CTE is trivial (`SELECT src, COUNT(*) FROM edges GROUP BY src`) — implement
- PageRank, betweenness, closeness, Leiden are genuinely impractical in pure SQL — document the limitation

**Run commands (once implemented):**
```bash
# CTE engine is already selectable via --engine cte
make -C benchmarks graph-small   # CTE results will be generated alongside muninn
```

**Estimated effort:** Small — degree is trivial, DFS is moderate, rest are documented skips.

### 2c. Centrality + community analysis charts

**Status:** Code needed (verification)

`benchmark_graph_analyze.py` exists and generates charts for BFS, DFS, shortest_path, components, PageRank. Needs verification that it handles centrality and community operations correctly when those results exist.

**Work needed:**
- Run centrality/community benchmarks (tasks 1a, 1b)
- Run `make -C benchmarks analyze-graph`
- If charts are incomplete, extend `benchmark_graph_analyze.py` with:
  - Centrality computation time vs graph size
  - Leiden modularity scores and community counts
  - Comparison across graph topologies (Erdos-Renyi vs Barabasi-Albert)

**Estimated effort:** Small — likely works already, just needs validation.

---

## 3. Comparison Baselines — Incomplete

### 3a. sqlite-vec and vectorlite VSS comparisons

**Status:** Framework ready, results need verification

The VSS benchmark framework (`benchmark_vss.py`) supports multiple engines. Need to verify that result JSONL files contain runs for `sqlite-vec` and `vectorlite` engines, not just `muninn` vs `sqlite-vector`.

**Action:** Check existing results, run any missing engine comparisons.

### 3b. GraphQLite graph comparisons

**Status:** Framework ready, results need verification

`run_graph_graphqlite()` is implemented in `benchmark_graph.py`. Depends on `HAS_GRAPHQLITE` flag. Verify results exist and cover all graph operations.

**Action:** Check existing results, run any missing engine comparisons.

### 3c. Integrated vs separate tools

**Status:** Code needed — blocked on KG pipeline

Key competitive benchmark: muninn (single extension) vs sqlite-vec + GraphQLite (two separate tools). Measures the integration value proposition.

**What to measure:**
- Setup complexity (connection count, extension loading)
- Query latency for combined vector+graph workflows
- Memory overhead (one extension vs two)

**Blocked by:** KG pipeline (section 4) — needs a realistic combined workload to be meaningful.

### 3d. VSS-only vs VSS+Graph (GraphRAG value)

**Status:** Code needed — blocked on KG pipeline

Measures whether graph expansion after VSS entry point improves retrieval quality.

**Blocked by:** KG pipeline (section 4) — needs knowledge graph with ground truth queries.

### 3e. BM25+Graph vs VSS+Graph

**Status:** Code needed — blocked on KG pipeline

Compares FTS5/BM25 entry point vs HNSW entry point for graph expansion workflows.

**Blocked by:** KG pipeline (section 4) — needs FTS5 index and ground truth.

---

## 4. Knowledge Graph Benchmark — Not Started

The end-to-end KG benchmark is the largest remaining gap. All C primitives exist, but the Python pipeline is entirely unbuilt.

### 4a. Entity extraction pipeline

**Files to create:**
- `benchmarks/scripts/kg_extract.py` — GLiNER + spaCy SVO + FTS5 extraction
- `benchmarks/scripts/kg_coalesce.py` — Entity resolution via HNSW-based blocking + Leiden clustering
- `benchmarks/scripts/kg_gutenberg.py` — Project Gutenberg catalog, download, caching

**Python dependencies:**
- `gliner` — Zero-shot NER
- `spacy` + `en_core_web_lg` — Dependency parsing + SVO
- `textacy` — SVO triple extraction helper
- `sentence-transformers` — Already in benchmark deps

**Output:** Cached SQLite databases in `benchmarks/kg/{gutenberg_id}.db`

### 4b. KG benchmark runner

**Files to create:**
- `benchmarks/scripts/benchmark_kg.py` — 6-phase benchmark:
  - Phase A: Build KG (load cached .db, insert into HNSW, Node2Vec)
  - Phase B: GraphRAG retrieval (VSS -> BFS expansion -> context assembly)
  - Phase C: Graph analytics (betweenness, PageRank, Leiden on the KG)
  - Phase D: Hierarchical retrieval (Leiden -> supernodes -> multi-level search)
  - Phase E: Node2Vec hyperparameter sweep
  - Phase F: Temporal KG queries (bi-temporal schema pattern)
- `benchmarks/scripts/benchmark_kg_analyze.py` — Aggregate JSONL results into charts

### 4c. Ground truth queries

50-100 retrieval questions about Wealth of Nations content with:
- Human-annotated relevant passages per question
- Relevance judgments at passage and entity level
- Ground-truth bridge concepts (for betweenness centrality validation)

**Approach options:**
- LLM-generated questions with human validation
- Manual curation from the table of contents
- ~50 seed questions as a starting point

### 4d. Makefile targets

```makefile
kg-extract:           ## Extract KG from Wealth of Nations (reference)
kg-extract-random:    ## Extract KG from a random Gutenberg economics book
kg-extract-book:      ## Extract KG from a specific book (BOOK_ID=...)
kg-coalesce:          ## Entity resolution + dedup (all cached KGs)
kg:                   ## Run KG benchmark on all cached KGs
kg-analyze:           ## Analyze KG results -> charts
```

---

## Suggested Execution Order

```
Phase 1 — Quick wins (ready to run, no code changes):
  1a. Run centrality benchmarks
  1b. Run community benchmarks
  2c. Validate analysis charts

Phase 2 — Small code additions:
  2b. Complete CTE baseline (degree + DFS)
  3a. Verify sqlite-vec/vectorlite results
  3b. Verify GraphQLite results

Phase 3 — New benchmark scripts:
  2a. Node2Vec benchmark suite

Phase 4 — Knowledge graph pipeline (largest effort):
  4a. Entity extraction pipeline
  4b. KG benchmark runner
  4c. Ground truth queries
  4d. Makefile targets
  3c-3e. Comparison baselines (unblocked by KG pipeline)
```
