# Knowledge Graph C Extension — Design Notes

## Architecture Boundary

muninn focuses on **high-performance supportive data structures** in C:
- HNSW vector index, graph traversal TVFs, centrality, Leiden, Node2Vec
- Loading and running ML models is **out of scope** for the C extension

ML-heavy tasks (entity extraction, embedding, LLM calls) stay in Python. muninn provides the storage and query layer.

## KG Pipeline Stages

1. **Text chunking** — split documents into overlapping windows
2. **Entity extraction** — NER (GLiNER, LLMs) + SVO triples (spaCy) + FTS5 concept discovery
3. **Entity relationship mapping** — co-occurrence, verb-mediated, LLM-extracted triples
4. **Entity coalescing** — HNSW blocking → Jaro-Winkler cascade → Leiden clustering
5. **Graph analytics** — centrality, community detection, Node2Vec structural embeddings

See [`kg_extraction_benchmark_spec.md`](kg_extraction_benchmark_spec.md) for the full benchmark specification covering NER models, LLM APIs, Ollama integration, datasets, and metrics.

## FTS5 Enhancement Candidates (pure C, no model dependency)

### Tunable BM25+ auxiliary function
SQLite's `bm25()` hard-codes k1=1.2, b=0.75 and lacks the BM25+ delta correction for long documents. A custom `muninn_bm25(fts_table, k1, b, delta, ...)` via `fts5_api->xCreateFunction` would fix this. ~100-200 lines C.

### TVF over FTS5 shadow tables
Expose term frequencies, document frequencies, and BM25+ scores for entity candidate discovery. Enables SQL-based corpus vocabulary inspection.

### The "Bank Problem" (polysemy)
FTS5_TOKEN_COLOCATED can emit synonyms at the same index position, but this only helps **synonymy** (curated alias lists), not **polysemy** (same word, different meanings). Polysemy resolution requires a contextual model (BERT/Word2Vec) — this stays in Python.

The most FTS5-compatible approach for context-aware retrieval is **DeepCT** (Dai & Callan, 2020): run BERT offline at indexing time, use context-aware term weights to repeat words N times in FTS5 proportional to their importance. +27% improvement on MS MARCO, no custom tokenizer needed.

See [`tokenisation_and_languages.md`](tokenisation_and_languages.md) for tokenizer internals and multilingual challenges.