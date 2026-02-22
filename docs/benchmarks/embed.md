# Embed Benchmarks

Compares **[`llama.cpp`](https://github.com/ggml-org/llama.cpp)**-based embedding functions on metrics:

- end-to-end query latency,
- embedding-only latency,
- insert throughput,
- and recall

...across multiple **GGUF** embedding models, search backends, and corpus types.

> Both embedding functions use **llama.cpp** under the hood to run **GGUF**-format
> embedding models. This benchmark compares their end-to-end performance.

## Embedding Functions

| Function | Description | 
|---------|---------|
| **muninn_embed** | muninn native llama.cpp embedding wrapper | 
| **lembed** | sqlite-lembed llama.cpp embedding wrapper | 


## Search Backends

| Backend | Method | Strategy | 
|---------|---------|---------|
| **muninn HNSW** | HNSW graph index | Approximate, O(log N) search | 
| **sqlite-vector PQ** | Product Quantization | Approximate, O(N) scan | 
| **sqlite-vec brute** | Brute-force KNN | Exact, O(N) scan | 


## GGUF Embedding Models

| Model | Dimension | Params | GGUF File | 
|---------|---------|---------|---------|
| **MiniLM** | 384 | 22M | `all-MiniLM-L6-v2.Q8_0.gguf` | 
| **NomicEmbed** | 768 | 137M | `nomic-embed-text-v1.5.Q8_0.gguf` | 


## Datasets

| Dataset | Source | Passages | Topology | 
|---------|---------|---------|---------|
| [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) | HuggingFace | ~120K | 4 discrete clusters | 
| [Wealth of Nations](https://www.gutenberg.org/ebooks/3300) | Project Gutenberg | ~2,500 (256-token windows, 50-token overlap) | Continuous conceptual gradient | 




## Query+Search Latency — AG News


### Query Embed+Search Latency (MiniLM / AG News)

```plotly
--8<-- "benchmarks/charts/embed_ql_MiniLM_ag_news.json"
```


### Query Embed+Search Latency (NomicEmbed / AG News)

```plotly
--8<-- "benchmarks/charts/embed_ql_NomicEmbed_ag_news.json"
```



## Query+Search Latency — Wealth of Nations


### Query Embed+Search Latency (MiniLM / Wealth of Nations)

```plotly
--8<-- "benchmarks/charts/embed_ql_MiniLM_wealth_of_nations.json"
```


### Query Embed+Search Latency (NomicEmbed / Wealth of Nations)

```plotly
--8<-- "benchmarks/charts/embed_ql_NomicEmbed_wealth_of_nations.json"
```



## Cross-Model Comparison — AG News


### Cross-Model Comparison (AG News)

```plotly
--8<-- "benchmarks/charts/embed_xmodel_ag_news.json"
```



## Cross-Model Comparison — Wealth of Nations


### Cross-Model Comparison (Wealth of Nations)

```plotly
--8<-- "benchmarks/charts/embed_xmodel_wealth_of_nations.json"
```



## Embedding-Only Latency


### Embedding-Only Latency (AG News)

```plotly
--8<-- "benchmarks/charts/embed_only_ag_news.json"
```


### Embedding-Only Latency (Wealth of Nations)

```plotly
--8<-- "benchmarks/charts/embed_only_wealth_of_nations.json"
```



## Insert Throughput


### Embed+Insert Throughput (AG News)

```plotly
--8<-- "benchmarks/charts/embed_insert_ag_news.json"
```


### Embed+Insert Throughput (Wealth of Nations)

```plotly
--8<-- "benchmarks/charts/embed_insert_wealth_of_nations.json"
```



## Recall


### Recall@k (AG News)

```plotly
--8<-- "benchmarks/charts/embed_recall_ag_news.json"
```


### Recall@k (Wealth of Nations)

```plotly
--8<-- "benchmarks/charts/embed_recall_wealth_of_nations.json"
```


