# Vector Search Benchmarks

Compares vector similarity search engines on insert throughput, search latency, and recall across multiple embedding models, dataset sizes, and corpus types.

## Engines

| Engine | Library | Method | Strategy | 
|---------|---------|---------|---------|
| **muninn-hnsw** | [muninn](https://github.com/neozenith/sqlite-muninn) | HNSW graph index | Approximate, O(log N) search | 
| **sqlite-vector-quantize** | [sqlite-vector](https://github.com/sqliteai/sqlite-vector) | Product Quantization | Approximate, O(N) scan | 
| **sqlite-vector-fullscan** | [sqlite-vector](https://github.com/sqliteai/sqlite-vector) | Brute-force | Exact, O(N) scan | 
| **vectorlite-hnsw** | [vectorlite](https://github.com/1yefuwang1/vectorlite) | HNSW via hnswlib | Approximate, O(log N) search | 
| **sqlite-vec-brute** | [sqlite-vec](https://github.com/asg017/sqlite-vec) | Brute-force KNN | Exact, O(N) scan | 


## Datasets

| Dataset | Source | Passages | Topology | 
|---------|---------|---------|---------|
| [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) | HuggingFace | ~120K | 4 discrete clusters | 
| [Wealth of Nations](https://www.gutenberg.org/ebooks/3300) | Project Gutenberg | ~2,500 (256-token windows, 50-token overlap) | Continuous conceptual gradient | 


## Embedding Models

| Model | Dimension | Params | Doc Prefix | Query Prefix | 
|---------|---------|---------|---------|---------|
| **MiniLM** | 384 | 22M | _(none)_ | _(none)_ | 
| **NomicEmbed** | 768 | 137M | `"search_document: "` | `"search_query: "` | 
| **BGE-Large** | 1024 | 335M | _(none)_ | `"Represent this sentence for se..."` | 




## Search Latency — AG News


### Search Latency vs Dataset Size (MiniLM / AG News)

```plotly
--8<-- "benchmarks/charts/tipping_point_MiniLM_ag_news.json"
```


### Search Latency vs Dataset Size (NomicEmbed / AG News)

```plotly
--8<-- "benchmarks/charts/tipping_point_NomicEmbed_ag_news.json"
```


### Search Latency vs Dataset Size (BGE-Large / AG News)

```plotly
--8<-- "benchmarks/charts/tipping_point_BGE-Large_ag_news.json"
```



## Cross-Model Comparison — AG News


### Cross-Model Comparison (AG News)

```plotly
--8<-- "benchmarks/charts/model_comparison_ag_news.json"
```



## Search Latency — Wealth of Nations


### Search Latency vs Dataset Size (MiniLM / Wealth of Nations)

```plotly
--8<-- "benchmarks/charts/tipping_point_MiniLM_wealth_of_nations.json"
```


### Search Latency vs Dataset Size (NomicEmbed / Wealth of Nations)

```plotly
--8<-- "benchmarks/charts/tipping_point_NomicEmbed_wealth_of_nations.json"
```


### Search Latency vs Dataset Size (BGE-Large / Wealth of Nations)

```plotly
--8<-- "benchmarks/charts/tipping_point_BGE-Large_wealth_of_nations.json"
```



## Cross-Model Comparison — Wealth of Nations


### Cross-Model Comparison (Wealth of Nations)

```plotly
--8<-- "benchmarks/charts/model_comparison_wealth_of_nations.json"
```



## Dataset Comparison


### Dataset Comparison (MiniLM)

```plotly
--8<-- "benchmarks/charts/dataset_comparison_MiniLM.json"
```


### Dataset Comparison (NomicEmbed)

```plotly
--8<-- "benchmarks/charts/dataset_comparison_NomicEmbed.json"
```


### Dataset Comparison (BGE-Large)

```plotly
--8<-- "benchmarks/charts/dataset_comparison_BGE-Large.json"
```



## Recall


### Recall@k (AG News)

```plotly
--8<-- "benchmarks/charts/recall_models_ag_news.json"
```


### Recall@k (Wealth of Nations)

```plotly
--8<-- "benchmarks/charts/recall_models_wealth_of_nations.json"
```



## Insert Throughput


### Insert Throughput (AG News)

```plotly
--8<-- "benchmarks/charts/insert_throughput_models_ag_news.json"
```


### Insert Throughput (Wealth of Nations)

```plotly
--8<-- "benchmarks/charts/insert_throughput_models_wealth_of_nations.json"
```



## Storage


### Storage (AG News)

```plotly
--8<-- "benchmarks/charts/db_size_models_ag_news.json"
```


### Storage (Wealth of Nations)

```plotly
--8<-- "benchmarks/charts/db_size_models_wealth_of_nations.json"
```


