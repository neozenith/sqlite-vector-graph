# Knowledge Graph Extraction & Coalescing — Benchmark Specification

Captured: 2026-02-16. Companion to `knowledge_graph_benchmark.md` and `benchmark_backlog.md`.

This document specifies the benchmark suite for evaluating entity extraction strategies, entity resolution (coalescing) quality, and LLM integration for the muninn KG pipeline.

---

## Table of Contents

1. [Entity Extraction Models](#1-entity-extraction-models)
2. [LLM API Integration](#2-llm-api-integration)
3. [Ollama Local Models](#3-ollama-local-models)
4. [Benchmark Datasets](#4-benchmark-datasets)
5. [Entity Resolution (Coalescing) Benchmarks](#5-entity-resolution-coalescing-benchmarks)
6. [FTS5 Enhancements](#6-fts5-enhancements)
7. [Measurement Framework](#7-measurement-framework)
8. [POC Validation Plan](#8-poc-validation-plan)

---

## 1. Entity Extraction Models

### 1a. GLiNER Family (all use `gliner` Python package)

All models share the same API: `GLiNER.from_pretrained(model_id)` → `model.predict_entities(text, labels, threshold=0.5)`.

| Model ID | Params | Disk | CPU ms/chunk | Zero-Shot F1 | License |
|----------|--------|------|-------------|-------------|---------|
| `urchade/gliner_small-v2.1` | 166M | 611 MB | 50-100 | ~52 | Apache-2.0 |
| `urchade/gliner_medium-v2.1` | 209M | 781 MB | 80-160 | ~56 | Apache-2.0 |
| `urchade/gliner_large-v2.1` | 459M | 1.78 GB | 300-500 | 60.9 | Apache-2.0 |
| `urchade/gliner_multi-v2.1` | 209M | 1.16 GB | 100-200 | ~54 | Apache-2.0 |
| `knowledgator/modern-gliner-bi-large-v1.0` | ~530M | 2.12 GB | 300-500 | >60.9 | Apache-2.0 |
| `knowledgator/gliner-multitask-large-v0.5` | ~440M | 1.76 GB | 300-500 | >60.9 | Apache-2.0 |

**Notes:**
- `modern-gliner-bi-large` uses ModernBERT backbone (8192 token context, 4x faster than DeBERTa). Requires dev `transformers`: `pip install git+https://github.com/huggingface/transformers.git`
- `gliner-multitask-large` supports 7 tasks (NER, RE, classification, summarization, sentiment, QA) via prompt tuning
- `gliner_multi-v2.1` covers 20+ languages including CJK, Arabic, Hindi, Finnish

### 1b. Competitors (GLiNER-compatible API)

| Model ID | Params | Disk | CPU ms/chunk | Zero-Shot F1 | License | Notes |
|----------|--------|------|-------------|-------------|---------|-------|
| `numind/NuNerZero` | ~400M | 1.8 GB | 300-500 | ~64 | MIT | Labels MUST be lowercase. Use `merge_entities()` post-processing for adjacent spans. |

**Loading code is identical to GLiNER:**
```python
from gliner import GLiNER
model = GLiNER.from_pretrained("numind/NuNerZero")
entities = model.predict_entities(text, [l.lower() for l in labels])
```

### 1c. Competitors (different API — generative)

| Model ID | Params | Disk | CPU ms/chunk | Zero-Shot F1 | License | Notes |
|----------|--------|------|-------------|-------------|---------|-------|
| `dyyyyyyyy/GNER-T5-base` | 248M | ~1.0 GB | 2,000-5,000 | 59.5 | MIT | Seq2seq, outputs BIO text. 10-50x slower than encoder models. |
| `dyyyyyyyy/GNER-T5-large` | 783M | ~3.1 GB | 5,000-15,000 | 63.5 | MIT | Include with caution — ~8-25 min for 100 chunks. |

**Loading code (different from GLiNER):**
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("dyyyyyyyy/GNER-T5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("dyyyyyyyy/GNER-T5-base")
# Requires instruction-formatted input, outputs BIO-tagged text that must be parsed
```

### 1d. Excluded Models

| Model | Reason |
|-------|--------|
| SpanMarker | Not zero-shot — fixed predefined label set, cannot accept arbitrary entity types |
| GNER-T5-xl (3B) | Too large/slow for CPU benchmarking |
| GNER-T5-xxl (11B) | Far too large for 16GB RAM |
| UniNER-7B/13B | Requires GPU, 7B+ parameters |

### 1e. Installation

```bash
# Core (covers all GLiNER family + NuNER)
pip install gliner

# For modern-gliner-bi-large (ModernBERT support)
pip install git+https://github.com/huggingface/transformers.git

# For GNER-T5 (usually already installed via gliner deps)
pip install torch transformers
```

---

## 2. LLM API Integration

### 2a. Provider Configuration

| Provider | Env Var | Package | Install | Import |
|----------|---------|---------|---------|--------|
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic` | `pip install anthropic` | `from anthropic import Anthropic` |
| OpenAI | `OPENAI_API_KEY` | `openai` | `pip install openai` | `from openai import OpenAI` |
| Google Gemini | `GEMINI_API_KEY` (or `GOOGLE_API_KEY` — if both set, `GOOGLE_API_KEY` wins) | `google-genai` | `pip install google-genai` | `from google import genai` |
| Ollama | `OLLAMA_HOST` (default: `http://127.0.0.1:11434`) | `ollama` | `pip install ollama` | `import ollama` |

**Note:** `google-generativeai` is deprecated since Aug 2025. Use `google-genai`.

### 2b. Token Usage Tracking

Each SDK reports token usage differently:

```python
# Anthropic
usage = {
    "input_tokens": response.usage.input_tokens,
    "output_tokens": response.usage.output_tokens,
}

# OpenAI
usage = {
    "input_tokens": response.usage.prompt_tokens,
    "output_tokens": response.usage.completion_tokens,
}

# Gemini
usage = {
    "input_tokens": response.usage_metadata.prompt_token_count,
    "output_tokens": response.usage_metadata.candidates_token_count,
}

# Ollama
usage = {
    "input_tokens": response["prompt_eval_count"],
    "output_tokens": response["eval_count"],
    "eval_duration_ns": response["eval_duration"],  # for tokens/sec
}
```

### 2c. Pricing Lookup Table (Feb 2026)

All prices per 1M tokens.

| Provider | Model | Input $/1M | Output $/1M |
|----------|-------|-----------|------------|
| Google | Gemini 2.0 Flash-Lite | 0.075 | 0.30 |
| Google | Gemini 2.5 Flash-Lite | 0.10 | 0.40 |
| OpenAI | GPT-4o-mini | 0.15 | 0.60 |
| Google | Gemini 2.5 Flash | 0.30 | 2.50 |
| Anthropic | Claude Haiku 4.5 | 1.00 | 5.00 |
| OpenAI | GPT-4o | 2.50 | 10.00 |
| Anthropic | Claude Sonnet 4.5 | 3.00 | 15.00 |
| Ollama | (any local model) | 0.00 | 0.00 |

**Cost calculation:**
```python
cost = (usage["input_tokens"] / 1_000_000) * price_input + \
       (usage["output_tokens"] / 1_000_000) * price_output
```

### 2d. Structured Output (JSON Schema)

All four providers support constrained JSON decoding. Use Pydantic for schema definition:

```python
from pydantic import BaseModel

class Entity(BaseModel):
    name: str
    entity_type: str
    confidence: float

class Relation(BaseModel):
    source: str
    target: str
    relation_type: str

class ExtractionResult(BaseModel):
    entities: list[Entity]
    relations: list[Relation]
```

| Provider | How to enforce schema |
|----------|---------------------|
| Anthropic | `client.messages.parse(output_format=ExtractionResult)` |
| OpenAI | `client.beta.chat.completions.parse(response_format=ExtractionResult)` |
| Gemini | `config={"response_mime_type": "application/json", "response_json_schema": ExtractionResult.model_json_schema()}` |
| Ollama | `format=ExtractionResult.model_json_schema()` + `options={"temperature": 0}` |

---

## 3. Ollama Local Models

### 3a. CLI Commands

```bash
# Start server (if not running as a system service)
ollama serve

# NuExtract — purpose-built for structured extraction (Phi-3 fine-tune)
ollama pull nuextract                      # 2.2 GB default (Q4_0)
ollama pull nuextract:3.8b-q8_0           # 4.1 GB higher quality
ollama pull sroecker/nuextract-tiny-v1.5   # 494M ultra-light variant

# Triplex — purpose-built for KG triple extraction (Phi-3 fine-tune)
ollama pull sciphi/triplex                 # 2.4 GB default (3.8B)
ollama pull sciphi/triplex:1.5b            # 1.1 GB tiny variant (32K context)
```

### 3b. Model Details

| Model | Base | Params | Q4 Size | Context | Purpose | License |
|-------|------|--------|---------|---------|---------|---------|
| `nuextract` | Phi-3-mini | 3.8B | 2.2 GB | 4K | Structured data extraction | MIT |
| `nuextract:3.8b-q8_0` | Phi-3-mini | 3.8B | 4.1 GB | 4K | Higher quality extraction | MIT |
| `sroecker/nuextract-tiny-v1.5` | Qwen 2.5 | 0.5B | ~0.4 GB | 4K | Ultra-light extraction | MIT |
| `sciphi/triplex` | Phi-3-mini | 3.8B | 2.4 GB | 4K | KG triple extraction | CC-BY-NC-SA-4.0* |
| `sciphi/triplex:1.5b` | — | 1.5B | 1.1 GB | 32K | Tiny triple extraction | CC-BY-NC-SA-4.0* |

*Triplex: NC license waived for orgs under $5M annual revenue.

### 3c. NuExtract Prompt Format

NuExtract uses a specific template:
```
<|input|>
{text to extract from}
<|output|>
```

**Critical:** Always set `temperature: 0` for extraction. NuExtract is purely extractive — all output text should be present in the input.

### 3d. Integration Architecture

```
┌────────────────────┐     HTTP      ┌──────────────────┐
│  Python benchmark  │ ──────────→   │  ollama serve    │
│  (ollama SDK)      │ ←──────────   │  (localhost:11434)│
│                    │   JSON        │  loads GGUF model │
└────────────────────┘               └──────────────────┘
```

The Python SDK is a thin HTTP client. It **cannot** load models directly — the server must be running. Configure via `OLLAMA_HOST` env var.

---

## 4. Benchmark Datasets

### 4a. Entity Extraction (NER)

| Dataset | Size | Types | Domain | Access | Priority |
|---------|------|-------|--------|--------|----------|
| **CoNLL-2003** | 22K sentences | 4 (PER/LOC/ORG/MISC) | Reuters news | HuggingFace `datasets` | **P0** — gold standard, required |
| **CrossNER** | Small per domain | Domain-specific | Politics, science, music, literature, AI | GitHub | **P1** — tests cross-domain generalization |
| **WNUT-17** | Small | 6 types | Social media (noisy) | HuggingFace | P2 — tests on novel entities |
| **Few-NERD** | 188K sentences | 66 fine-grained | Wikipedia | HuggingFace | P3 — large, run subset only |

**Rationale:** CoNLL-2003 is required for comparability (every NER paper reports against it). CrossNER tests domain transfer (relevant to our Gutenberg economics texts). The others are optional stretch targets.

### 4b. Relation Extraction

| Dataset | Size | Level | Relations | Access | Priority |
|---------|------|-------|-----------|--------|----------|
| **DocRED** | 5K docs, 63K triples | Document | 96 types | GitHub | **P1** — document-level, matches chunked pipeline |
| **WebNLG** | 17K triple sets | Text-to-triple | ~450 DBpedia props | Free | P2 — full KG extraction |
| **TACRED** | 106K sentences | Sentence | 42 types | LDC/mirrors | P3 — sentence-level only |

### 4c. Standard Metrics

| Metric | Scope | Description |
|--------|-------|-------------|
| **Entity-level micro F1** (strict) | NER | Span boundaries AND type must match exactly |
| **Precision** | NER | Fraction of predicted entities that are correct |
| **Recall** | NER | Fraction of gold entities that were found |
| **Triple F1** (strict) | RE | Subject + predicate + object must all match |
| **Inference time** | All | Wall-clock ms per chunk |
| **Cost** | LLM | USD per 1K chunks (from token usage × pricing) |

---

## 5. Entity Resolution (Coalescing) Benchmarks

### 5a. Why Separate Benchmarks

Entity extraction (NER) and entity resolution (coalescing) are **different tasks** evaluated separately:

- **NER:** "What spans in text are entities?" — measured by span-level F1
- **ER:** "Which extracted entities refer to the same real-world thing?" — measured by pairwise F1 and B-Cubed F1

The NER datasets (CoNLL-2003, etc.) have **no entity resolution annotations** and cannot be repurposed for coalescing benchmarks. Exception: OntoNotes 5.0 has within-document coreference annotations, but requires LDC access.

### 5b. Dedicated ER Datasets

**Tier 1 — Tiny, free, zero setup (start here):**

| Dataset | Entities | True Matches | Domain | Access |
|---------|----------|-------------|--------|--------|
| **Febrl 1** | 1,000 records | 500 pairs | Synthetic person names | `pip install recordlinkage` → `load_febrl1()` |
| **Febrl 4** | 10,000 records | 5,000 pairs | Synthetic person names | Same package → `load_febrl4a()`, `load_febrl4b()` |
| **BeerAdvo-RateBeer** | 450 pairs | 68 matches | Beer reviews | Direct download, <1MB |
| **Fodors-Zagats** | 946 pairs | 110 matches | Restaurants | Direct download, <1MB |

**Tier 2 — Small, free, directly analogous to KG coalescing:**

| Dataset | Entities | True Matches | Domain | Access | Why Relevant |
|---------|----------|-------------|--------|--------|-------------|
| **DBLP-ACM** | ~5K entities | 2,224 matches | Bibliographic (name variations) | Leipzig direct download | Name matching like KG entities |
| **Affiliations** | 2,260 entities | 330 clusters | Organization names | Leipzig direct download | **Closest to KG entity clustering** |
| **Abt-Buy** | 2,173 entities | 1,097 matches | E-commerce products | Leipzig direct download | Textual similarity matching |
| **MusicBrainz 20K** | 19K entities, 5 sources | 10K clusters | Music metadata | Leipzig direct download | Multi-source merging |

**Tier 3 — KG-specific:**

| Dataset | Size | Domain | Access | Why Relevant |
|---------|------|--------|--------|-------------|
| **MovieGraphBenchmark** | Varies | Movie KGs (IMDB/TMDB/TVDB) | `pip install moviegraphbenchmark` | **Actual KG entity resolution** across heterogeneous sources |
| **DBP15K** | ~15K aligned pairs | Cross-lingual DBpedia | GitHub | KG entity alignment with embeddings |
| **AIDA-CoNLL** | 1,393 docs | Entity linking to YAGO | Max Planck Institute | Entity linking (mention → KB entry) |

### 5c. ER Metrics

| Metric | Description | Package |
|--------|-------------|---------|
| **Pairwise F1** | Of all entity pairs, which were correctly merged/not merged | `recordlinkage` |
| **B-Cubed F1** | Per-entity: what fraction of its cluster members are correct | Custom or `coval` |
| **Cluster purity** | Fraction of dominant class in each cluster | Custom |

### 5d. What Our Pipeline Already Does

Current coalescing in `kg_coalesce.py`:

| Stage | Method | Corresponds To |
|-------|--------|---------------|
| 1. HNSW Blocking | KNN on entity embeddings, cosine < 0.4 | Standard ER blocking step |
| 2. Matching Cascade | Exact → substring → Jaro-Winkler → cosine | Standard ER matching step |
| 3. Leiden Clustering | `graph_leiden` on match edges | Standard ER clustering step |

**Raw vs coalesced data:** Both are preserved in the same DB:
- Raw: `entities` table + `relations` table
- Mapping: `entity_clusters` table (name → canonical)
- Clean: `nodes` table + `edges` table (canonical names, aggregated weights)

---

## 6. FTS5 Enhancements

### 6a. Tunable BM25+ Auxiliary Function (C extension)

SQLite's built-in `bm25()` hard-codes k1=1.2, b=0.75 and lacks the BM25+ delta correction. A custom auxiliary function via `fts5_api->xCreateFunction` would:

- Allow tunable k1, b parameters
- Add BM25+ delta (Lv & Zhai, 2011) fixing the long-document scoring bug
- Register as e.g. `muninn_bm25(fts_table, k1, b, delta, col_weight1, col_weight2, ...)`

**Complexity:** Low (~100-200 lines C). High value — directly useful for entity candidate scoring.

### 6b. TVF Over FTS5 Shadow Tables

Expose term frequencies, document frequencies, and BM25+ scores as a table-valued function for entity candidate discovery:

```sql
SELECT term, doc_freq, bm25_score
FROM muninn_fts_vocab('chunks_fts', 'labour wages profit')
WHERE bm25_score > 0.5
ORDER BY bm25_score DESC
LIMIT 20;
```

**Complexity:** Medium. Requires reading FTS5 shadow tables (`{table}_data`, `{table}_idx`, `{table}_content`).

### 6c. FTS5_TOKEN_COLOCATED — Limitations

`FTS5_TOKEN_COLOCATED` emits synonyms at the same position in the inverted index. Useful for:
- Stemming variants ("running" + "run" at same position)
- Curated alias expansion ("USA" + "United States" at same position)

**Does NOT solve polysemy** (the "bank problem"). Polysemy resolution requires a contextual model (BERT, Word2Vec) at index time to determine which sense of "bank" is meant. This stays in the Python layer — it cannot be done in pure C without embedding a model runtime.

### 6d. DeepCT-Style Contextual Weighting (Python pre-processing)

The most FTS5-compatible approach for context-aware retrieval (Dai & Callan, 2020):

1. Run text through BERT offline (Python, at indexing time)
2. BERT produces a context-aware weight per word
3. Repeat the word N times in the FTS5 document proportional to its weight
4. BM25 naturally ranks contextually-important terms higher

Published result: +27% on MS MARCO, +46% on TREC-CAR. No custom tokenizer needed — pure pre-processing. Could be a future enhancement to `kg_extract.py`.

---

## 7. Measurement Framework

### 7a. Per-Model Metrics

Every extraction benchmark run records:

```python
{
    "model_id": "urchade/gliner_small-v2.1",
    "model_type": "gliner",           # gliner | gner | llm_api | ollama
    "dataset": "conll2003",
    "chunk_count": 100,
    "timestamp": "2026-02-16T12:00:00Z",

    # Quality
    "entity_precision": 0.82,
    "entity_recall": 0.75,
    "entity_f1": 0.78,

    # Performance
    "total_time_s": 12.5,
    "avg_ms_per_chunk": 125.0,
    "peak_memory_mb": 1800,

    # Cost (LLM only)
    "input_tokens": 50000,
    "output_tokens": 12000,
    "cost_usd": 0.045,

    # Model metadata
    "params_millions": 166,
    "quantization": null,             # or "q4_0", "q8_0" for Ollama
}
```

### 7b. Per-Coalescing Metrics

```python
{
    "method": "hnsw_blocking+jaro_winkler+leiden",
    "dataset": "dblp_acm",
    "blocking_threshold": 0.4,

    # Quality
    "pairwise_precision": 0.91,
    "pairwise_recall": 0.87,
    "pairwise_f1": 0.89,
    "bcubed_f1": 0.85,

    # Graph quality
    "nodes_before": 5000,
    "nodes_after": 3200,
    "edges_before": 12000,
    "edges_after": 8500,
    "singleton_ratio": 0.12,
    "connected_components": 45,

    # Performance
    "blocking_time_s": 2.1,
    "matching_time_s": 0.8,
    "clustering_time_s": 0.3,
    "total_time_s": 3.2,
}
```

### 7c. Results Storage

All results accumulate in JSONL files:

```
benchmarks/results/
  kg_extraction.jsonl      # NER model benchmarks
  kg_coalescing.jsonl      # Entity resolution benchmarks
  kg_llm_extraction.jsonl  # LLM API extraction benchmarks
  kg_graphrag.jsonl        # End-to-end GraphRAG query benchmarks (existing)
```

---

## 8. POC Validation Plan

Before building the full benchmark suite, validate each integration path with a small proof-of-concept.

### 8a. POC Scope

- **Text:** 10 chunks from Wealth of Nations (already cached)
- **Entity types:** `["person", "organization", "location", "economic concept", "commodity"]`
- **Gold standard:** Manually annotate the 10 chunks (~30 min)
- **Measure:** F1 score, time, cost per provider

### 8b. POC Order

| Step | Integration | Validates | Risk |
|------|------------|-----------|------|
| 1 | GLiNER small (existing) | Baseline NER quality | None (already works) |
| 2 | Ollama + NuExtract | Local LLM structured extraction | Server setup, prompt format |
| 3 | Ollama + Triplex | Local LLM KG triple extraction | Different output schema |
| 4 | OpenAI GPT-4o-mini | Cheapest API provider | API key, network |
| 5 | Anthropic Claude Haiku | Second cheapest API | API key, structured output |
| 6 | Google Gemini Flash-Lite | Cheapest overall | New SDK (`google-genai`) |

### 8c. POC Success Criteria

- Each integration can process 10 chunks and return valid structured JSON
- Token usage is captured for cost calculation
- Wall-clock time is recorded
- Output can be scored against gold-standard annotations
- No integration requires more than 20 lines of provider-specific code

### 8d. POC Script Location

```
benchmarks/scripts/kg_extraction_poc.py
```

CLI interface:
```bash
# Run specific provider
python benchmarks/scripts/kg_extraction_poc.py --provider gliner --model urchade/gliner_small-v2.1
python benchmarks/scripts/kg_extraction_poc.py --provider ollama --model nuextract
python benchmarks/scripts/kg_extraction_poc.py --provider openai --model gpt-4o-mini
python benchmarks/scripts/kg_extraction_poc.py --provider anthropic --model claude-haiku-4-5-20251001
python benchmarks/scripts/kg_extraction_poc.py --provider gemini --model gemini-2.0-flash-lite

# Run all available providers
python benchmarks/scripts/kg_extraction_poc.py --all

# Compare results
python benchmarks/scripts/kg_extraction_poc.py --compare
```

---

## Appendix A: Full Ollama Model Matrix for NER Benchmarking

Beyond the purpose-built extraction models, these general-purpose models are worth benchmarking for entity extraction quality:

| Model | Ollama Tag | Params | Q4 Size | Good at Extraction? |
|-------|-----------|--------|---------|-------------------|
| NuExtract | `nuextract` | 3.8B | 2.2 GB | Best — purpose-built |
| Triplex | `sciphi/triplex` | 3.8B | 2.4 GB | Best for KG triples |
| Qwen 2.5 7B | `qwen2.5:7b` | 7B | ~4.5 GB | Strong JSON adherence |
| Llama 3.2 3B | `llama3.2:3b` | 3B | ~2.0 GB | Decent, good for person entities |
| Gemma 2 9B | `gemma2:9b` | 9B | ~5.2 GB | Highest accuracy, tight on RAM |
| Phi-3.5 mini | `phi3.5` | 3.8B | ~2.3 GB | Good instruction following |
| Mistral 7B | `mistral:7b` | 7B | ~4.1 GB | Strong unique entity extraction |

## Appendix B: Entity Resolution Dataset Download URLs

| Dataset | URL |
|---------|-----|
| Leipzig Benchmarks (DBLP-ACM, Abt-Buy, Affiliations, etc.) | https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution |
| DeepMatcher Benchmarks (BeerAdvo, Fodors, iTunes, etc.) | https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md |
| Febrl (built-in) | `pip install recordlinkage` |
| MovieGraphBenchmark | `pip install moviegraphbenchmark` |
| DBP15K | https://github.com/nju-websoft/OpenEA |
| WDC Products | https://webdatacommons.org/largescaleproductcorpus/wdc-products/ |

## Appendix C: References

### Entity Extraction
- GLiNER (Zaratiana et al., NAACL 2024)
- NuNER-Zero (NuMind, 2024)
- GNER (Ding et al., ACL 2024 Findings)
- SpanMarker (Aarsen, 2023)

### Entity Resolution
- Febrl (Christen & Pudjijono, 2008)
- DeepMatcher (Mudgal et al., SIGMOD 2018)
- Ditto (Li et al., VLDB 2020)
- MovieGraphBenchmark (ScaDS Leipzig, 2023)

### Information Retrieval
- BM25+ (Lv & Zhai, CIKM 2011)
- DeepCT (Dai & Callan, 2020) — context-aware term weighting
- SPLADE (Formal et al., SIGIR 2021) — sparse neural retrieval
- ColBERT (Khattab & Zaharia, SIGIR 2020) — late interaction

### Knowledge Graph Quality
- KGGen / MINE benchmark (NeurIPS 2025) — fact recovery from KGs
- Text2KGBench (ISWC 2023) — ontology-driven KG extraction
- DocRED (Yao et al., ACL 2019) — document-level relation extraction

### Tokenisation and Multilingual
- See companion doc: `tokenisation_and_languages.md`
