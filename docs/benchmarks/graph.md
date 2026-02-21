# Graph Benchmarks

Measures graph algorithm performance across traversal, centrality, community detection, and Node2Vec embedding generation. Benchmarks compare muninn's built-in TVFs against alternative SQLite graph engines where available.

## Engines

| Engine | Library | Traversal | Centrality | Community | Node2Vec | 
|---------|---------|---------|---------|---------|---------|
| **muninn** | [muninn](https://github.com/neozenith/sqlite-muninn) | BFS, DFS, shortest path, components, PageRank | degree, betweenness, closeness | Leiden | Yes | 
| **graphqlite** | [GraphQLite](https://github.com/colliery-io/graphqlite) | BFS, DFS, shortest path, components, PageRank | — | Leiden | — | 


## Graph Models

| Model | Description | Key Property | 
|---------|---------|---------|
| [Erdos-Renyi](https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model) | Random edges, uniform probability | Uniform degree distribution | 
| [Barabasi-Albert](https://en.wikipedia.org/wiki/Barab%C3%A1si%E2%80%93Albert_model) | Preferential attachment | Power-law (scale-free) degree distribution | 



## Benchmark Dimensions


Graphs are tested at multiple sizes (100 to 10,000 nodes) and average degrees (3 to 20 edges per node), producing a range from sparse to dense topologies.



### Traversal

BFS, DFS, shortest path, connected components, and PageRank. Compared across both engines.


### Centrality

Degree, betweenness (Brandes algorithm), and closeness centrality. Muninn-only — measures how centrality computation scales with graph density.


### Community Detection

Leiden algorithm for community detection. Compared across both engines.


### Node2Vec

Random walk generation and Skip-gram training with Negative Sampling. Tests vary walk parameters (p, q) and embedding dimensionality (64, 128).




## Traversal


### BFS

```plotly
--8<-- "benchmarks/charts/graph_query_time_bfs.json"
```


### DFS

```plotly
--8<-- "benchmarks/charts/graph_query_time_dfs.json"
```


### Shortest Path

```plotly
--8<-- "benchmarks/charts/graph_query_time_shortest_path.json"
```


### Connected Components

```plotly
--8<-- "benchmarks/charts/graph_query_time_components.json"
```


### PageRank

```plotly
--8<-- "benchmarks/charts/graph_query_time_pagerank.json"
```



## Centrality


### Degree Centrality

```plotly
--8<-- "benchmarks/charts/graph_query_time_degree.json"
```


### Betweenness Centrality

```plotly
--8<-- "benchmarks/charts/graph_query_time_betweenness.json"
```


### Closeness Centrality

```plotly
--8<-- "benchmarks/charts/graph_query_time_closeness.json"
```



## Community Detection


### Leiden Community Detection

```plotly
--8<-- "benchmarks/charts/graph_query_time_leiden.json"
```



## Insertion Throughput


### Insertion Throughput

```plotly
--8<-- "benchmarks/charts/graph_setup_time.json"
```


