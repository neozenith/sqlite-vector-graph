# Text Embeddings — Semantic Search with muninn

Zero-dependency end-to-end example: load GGUF embedding models, embed documents,
build HNSW indices, and perform semantic similarity search — all inside a single
SQLite extension. Compares three models side-by-side.

## Models

| Model | Params | Dim | Quantization | Size | Context | Architecture |
|-------|--------|-----|-------------|------|---------|--------------|
| **all-MiniLM-L6-v2** | 22M | 384 | Q8_0 | 23 MB | 512 | BERT (mean pooling) |
| **nomic-embed-text-v1.5** | 137M | 768 | Q8_0 | 146 MB | 8K | BERT (mean pooling) |
| **Qwen3-Embedding-8B** | 8B | 4096 | Q4_K_M | 4.7 GB | 32K | Decoder (last-token pooling) |

All models are auto-downloaded to `models/` on first run. The Qwen3 model is
large (4.7 GB) — the example gracefully skips any model that fails to download.

## What This Demonstrates

| Feature | SQL |
|---------|-----|
| Load GGUF model | `INSERT INTO temp.muninn_models(name, model) SELECT 'name', muninn_embed_model('path.gguf')` |
| Query dimension | `SELECT muninn_model_dim('MiniLM')` |
| Generate embedding | `SELECT muninn_embed('MiniLM', 'hello world')` |
| Prefixed embedding | `SELECT muninn_embed('NomicEmbed', 'search_query: hello world')` |
| Bulk embed + index | `INSERT INTO idx(rowid, vector) SELECT id, muninn_embed('MiniLM', content) FROM docs` |
| KNN semantic search | `WHERE vector MATCH muninn_embed('MiniLM', 'query') AND k = 3` |
| Auto-embed trigger | `CREATE TEMP TRIGGER ... muninn_embed(...)` on INSERT |
| Multi-model compare | Side-by-side search results from all models |

## Prerequisites

Build the muninn extension — that's it. No `pip install` needed.

```bash
make all
```

## Run

```bash
# Run the example (models auto-download on first run)
python examples/text_embeddings/example.py

# See llama.cpp internals (model loading, tensor info, etc.)
MUNINN_LOG_LEVEL=verbose python examples/text_embeddings/example.py
```

## Sections

| # | Section | What It Shows |
|---|---------|---------------|
| 1 | **Model Loading & Inspection** | Load all models, query dimensions, show prefixes |
| 2 | **Embed & Build HNSW Indices** | Per-model HNSW indices with bulk INSERT...SELECT |
| 3 | **Comparative Semantic Search** | Side-by-side KNN results for each query across all models |
| 4 | **Auto-Embed Trigger** | TEMP trigger that auto-embeds new rows on INSERT |

## Data

8 sentences across 4 topics:

- **Nature:** fox in the forest, wolves and bears
- **AI/ML:** neural networks, gradient descent
- **Food:** Italian pasta
- **Space:** Mars rover, stars and galaxies

3 semantic queries test cross-topic retrieval:

- "animals in the wild" — should match nature documents
- "machine learning and artificial intelligence" — should match AI documents
- "outer space exploration" — should match space documents

## Model Architecture Notes

### BERT vs Decoder embedding models

muninn automatically handles both architectures through GGUF metadata:

- **BERT models** (MiniLM, Nomic): bidirectional attention, **mean pooling** — the
  embedding is the average of all token hidden states.
- **Decoder models** (Qwen3-Embedding): causal attention, **last-token pooling** — the
  embedding lives at the final token position (`<|endoftext|>`).

The pooling type is baked into each GGUF file's metadata, so muninn reads it
automatically. No configuration needed.

### Task instruction prefixes

Some models use prefixes to specialize embeddings for different tasks:

| Model | Document prefix | Query prefix |
|-------|----------------|--------------|
| MiniLM | *(none)* | *(none)* |
| nomic-embed-text-v1.5 | `search_document: ` | `search_query: ` |
| Qwen3-Embedding-8B | *(none, optional instructions)* | *(none, optional instructions)* |

The example handles prefixes via `ModelConfig.doc_prefix` / `query_prefix`.
