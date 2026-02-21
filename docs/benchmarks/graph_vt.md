# Graph VT Benchmarks

Compares CSR adjacency caching strategies against direct TVF access after edge mutations. Each chart plots performance against graph size (node count) on a log-log scale.

## Methods

| Method | Description | 
|---------|---------|
| **TVF** | No cache — scans edge table via SQL on every query | 
| **CSR** | Persistent CSR cache; initial build from edge table | 
| **CSR — full rebuild** | Persistent CSR cache; full edge-table re-scan when stale | 
| **CSR — incremental** | Delta + merge; rebuilds all blocks (spread mutations) | 
| **CSR — blocked incremental** | Delta + merge; rebuilds only affected blocks (concentrated mutations) | 


## Workloads

| Size | Nodes | Edges | Graph Model | 
|---------|---------|---------|---------|
| xsmall | 500 | 2,000 | Erdos-Renyi | 
| small | 1,000 | 5,000 | Erdos-Renyi | 
| medium | 5,000 | 25,000 | Barabasi-Albert | 
| large | 10,000 | 50,000 | Barabasi-Albert | 




## How Blocked CSR Works

The CSR is partitioned into blocks of 4,096 nodes. Each block is a separate row in the shadow table. When edges change, only blocks containing affected nodes are rewritten — unaffected blocks require zero I/O.




## Algorithm Query Time


### Degree Query Time

```plotly
--8<-- "benchmarks/charts/graph_vt_degree.json"
```


### Betweenness Query Time

```plotly
--8<-- "benchmarks/charts/graph_vt_betweenness.json"
```


### Closeness Query Time

```plotly
--8<-- "benchmarks/charts/graph_vt_closeness.json"
```


### Leiden Query Time

```plotly
--8<-- "benchmarks/charts/graph_vt_leiden.json"
```



## Rebuild Performance


### Rebuild Time

```plotly
--8<-- "benchmarks/charts/graph_vt_rebuild.json"
```



## Build Performance


### Initial CSR Build Time

```plotly
--8<-- "benchmarks/charts/graph_vt_build.json"
```



## Storage


### Shadow Table Disk Usage

```plotly
--8<-- "benchmarks/charts/graph_vt_disk.json"
```



## Trigger Overhead


### Trigger Overhead

```plotly
--8<-- "benchmarks/charts/graph_vt_trigger.json"
```


