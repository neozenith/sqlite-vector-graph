######################################################################
# benchmarks/Makefile — invoke via: make -C benchmarks {target}
#   Override storage: make -C benchmarks vss-embeddings STORAGE=memory
######################################################################

PYTHON = ../.venv/bin/python
BENCH_VSS_SRC = scripts/benchmark_vss.py
BENCH_VSS_ANALYZE_SRC = scripts/benchmark_vss_analyze.py
BENCH_GRAPH_SRC = scripts/benchmark_graph.py
BENCH_GRAPH_ANALYZE_SRC = scripts/benchmark_graph_analyze.py
BENCH_VSS = $(PYTHON) $(BENCH_VSS_SRC)
BENCH_GRAPH = $(PYTHON) $(BENCH_GRAPH_SRC)
STORAGE ?= disk
BENCH_FLAGS = --storage $(STORAGE)

# Operation categories for selective graph benchmarking
CENTRALITY_OPS = degree,betweenness,closeness
COMMUNITY_OPS = leiden

# Platform detection (for extension prerequisite)
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    EXT = .dylib
else ifeq ($(UNAME_S),Linux)
    EXT = .so
else
    EXT = .dll
endif

EXTENSION = ../muninn$(EXT)

.PHONY: help \
        models-prep models-prep-manifest models vss-analyze clean clean-all \
        graph-small graph-medium graph-large graph-scale-free graph-analyze \
        graph-centrality-small graph-centrality-medium graph-centrality-large graph-centrality-scale-free \
        graph-community-small graph-community-medium graph-community-large graph-community-scale-free \
        vss-manifest manifest-vss-all graph-manifest manifest-graph-all manifest \
        analyze analyze-vss analyze-graph \
        vss-embeddings-prep vss-embeddings \
        vss-embeddings-agnews-all-MiniLM-L6-v2 vss-embeddings-agnews-all-mpnet-base-v2 vss-embeddings-agnews-BAAI-bge-large-en-v1.5 \
        vss-embeddings-wealth-of-nations-all-models vss-embeddings-all

help:                                              ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2}'

$(EXTENSION):
	$(MAKE) -C .. all

######################################################################
# VECTOR BENCHMARKS
######################################################################

vss-embeddings-prep:                                       ## Download models + datasets, pre-build all .npy embedding caches
	$(BENCH_VSS) --prep-models --prep-model MiniLM --prep-dataset ag_news
	$(BENCH_VSS) --prep-models --prep-model MPNet --prep-dataset ag_news
	$(BENCH_VSS) --prep-models --prep-model BGE-Large --prep-dataset ag_news
	$(BENCH_VSS) --prep-models --prep-model MiniLM --prep-dataset wealth_of_nations
	$(BENCH_VSS) --prep-models --prep-model MPNet --prep-dataset wealth_of_nations
	$(BENCH_VSS) --prep-models --prep-model BGE-Large --prep-dataset wealth_of_nations

vss-embeddings-agnews-all-MiniLM-L6-v2: $(EXTENSION)
	# AG News (120K passages)
	$(BENCH_VSS) --source model:all-MiniLM-L6-v2 --sizes 1000 --dataset ag_news $(BENCH_FLAGS)
	$(BENCH_VSS) --source model:all-MiniLM-L6-v2 --sizes 5000 --dataset ag_news $(BENCH_FLAGS)
	$(BENCH_VSS) --source model:all-MiniLM-L6-v2 --sizes 10000 --dataset ag_news $(BENCH_FLAGS)
	$(BENCH_VSS) --source model:all-MiniLM-L6-v2 --sizes 50000 --dataset ag_news $(BENCH_FLAGS)
	$(BENCH_VSS) --source model:all-MiniLM-L6-v2 --sizes 100000 --dataset ag_news $(BENCH_FLAGS)
	$(BENCH_VSS) --source model:all-MiniLM-L6-v2 --sizes 250000 --dataset ag_news $(BENCH_FLAGS)
	$(BENCH_VSS) --source model:all-mpnet-base-v2 --sizes 1000 --dataset ag_news $(BENCH_FLAGS)

vss-embeddings-agnews-all-mpnet-base-v2: $(EXTENSION)
	$(BENCH_VSS) --source model:all-mpnet-base-v2 --sizes 5000 --dataset ag_news $(BENCH_FLAGS)
	$(BENCH_VSS) --source model:all-mpnet-base-v2 --sizes 10000 --dataset ag_news $(BENCH_FLAGS)
	$(BENCH_VSS) --source model:all-mpnet-base-v2 --sizes 50000 --dataset ag_news $(BENCH_FLAGS)
	$(BENCH_VSS) --source model:all-mpnet-base-v2 --sizes 100000 --dataset ag_news $(BENCH_FLAGS)
	$(BENCH_VSS) --source model:all-mpnet-base-v2 --sizes 250000 --dataset ag_news $(BENCH_FLAGS)

vss-embeddings-agnews-BAAI-bge-large-en-v1.5: $(EXTENSION)
	$(BENCH_VSS) --source model:BAAI/bge-large-en-v1.5 --sizes 1000 --dataset ag_news $(BENCH_FLAGS)
	$(BENCH_VSS) --source model:BAAI/bge-large-en-v1.5 --sizes 5000 --dataset ag_news $(BENCH_FLAGS)
	$(BENCH_VSS) --source model:BAAI/bge-large-en-v1.5 --sizes 10000 --dataset ag_news $(BENCH_FLAGS)
	$(BENCH_VSS) --source model:BAAI/bge-large-en-v1.5 --sizes 50000 --dataset ag_news $(BENCH_FLAGS)
	$(BENCH_VSS) --source model:BAAI/bge-large-en-v1.5 --sizes 100000 --dataset ag_news $(BENCH_FLAGS)
	$(BENCH_VSS) --source model:BAAI/bge-large-en-v1.5 --sizes 250000 --dataset ag_news $(BENCH_FLAGS)

vss-embeddings-wealth-of-nations-all-models: $(EXTENSION)											# 	# Wealth of Nations (~2.5K passages)
	$(BENCH_VSS) --source model:all-MiniLM-L6-v2 --sizes 1000 --dataset wealth_of_nations $(BENCH_FLAGS)
	$(BENCH_VSS) --source model:all-MiniLM-L6-v2 --sizes 2500 --dataset wealth_of_nations $(BENCH_FLAGS)
	$(BENCH_VSS) --source model:all-mpnet-base-v2 --sizes 1000 --dataset wealth_of_nations $(BENCH_FLAGS)
	$(BENCH_VSS) --source model:all-mpnet-base-v2 --sizes 2500 --dataset wealth_of_nations $(BENCH_FLAGS)
	$(BENCH_VSS) --source model:BAAI/bge-large-en-v1.5 --sizes 1000 --dataset wealth_of_nations $(BENCH_FLAGS)
	$(BENCH_VSS) --source model:BAAI/bge-large-en-v1.5 --sizes 2500 --dataset wealth_of_nations $(BENCH_FLAGS)

vss-embeddings-all: vss-embeddings-agnews-all-MiniLM-L6-v2 vss-embeddings-agnews-all-mpnet-base-v2 vss-embeddings-agnews-BAAI-bge-large-en-v1.5 vss-embeddings-wealth-of-nations-all-models  ## Run all VSS embedding benchmarks


######################################################################
# GRAPH BENCHMARKS
######################################################################

graph-small: $(EXTENSION)                          ## Graph: small profile (N<=1K)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 100 --avg-degree 5 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 100 --avg-degree 20 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 500 --avg-degree 5 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 500 --avg-degree 20 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 1000 --avg-degree 5 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 1000 --avg-degree 20 $(BENCH_FLAGS)

graph-medium: $(EXTENSION)                         ## Graph: medium profile (N<=10K)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 1000 --avg-degree 5 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 1000 --avg-degree 20 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 1000 --avg-degree 50 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 5000 --avg-degree 5 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 5000 --avg-degree 20 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 5000 --avg-degree 50 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 10000 --avg-degree 5 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 10000 --avg-degree 20 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 10000 --avg-degree 50 $(BENCH_FLAGS)

graph-large: $(EXTENSION)                          ## Graph: large profile (N<=100K)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 10000 --avg-degree 5 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 10000 --avg-degree 20 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 50000 --avg-degree 5 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 50000 --avg-degree 20 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 100000 --avg-degree 5 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 100000 --avg-degree 20 $(BENCH_FLAGS)

graph-scale-free: $(EXTENSION)                     ## Graph: Barabasi-Albert scale-free topology
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 1000 --avg-degree 3 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 1000 --avg-degree 5 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 1000 --avg-degree 10 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 5000 --avg-degree 3 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 5000 --avg-degree 5 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 5000 --avg-degree 10 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 10000 --avg-degree 3 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 10000 --avg-degree 5 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 10000 --avg-degree 10 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 50000 --avg-degree 3 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 50000 --avg-degree 5 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 50000 --avg-degree 10 $(BENCH_FLAGS)

######################################################################
# GRAPH OPERATION-GROUP TARGETS
######################################################################

graph-centrality-small: $(EXTENSION)               ## Graph: centrality ops, small (N<=1K)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 100 --avg-degree 5 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 100 --avg-degree 20 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 500 --avg-degree 5 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 500 --avg-degree 20 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 1000 --avg-degree 5 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 1000 --avg-degree 20 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)

graph-centrality-medium: $(EXTENSION)              ## Graph: centrality ops, medium (N<=10K)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 1000 --avg-degree 5 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 1000 --avg-degree 20 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 1000 --avg-degree 50 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 5000 --avg-degree 5 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 5000 --avg-degree 20 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 5000 --avg-degree 50 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 10000 --avg-degree 5 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 10000 --avg-degree 20 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 10000 --avg-degree 50 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)

graph-centrality-large: $(EXTENSION)               ## Graph: centrality ops, large (N<=100K)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 10000 --avg-degree 5 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 10000 --avg-degree 20 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 50000 --avg-degree 5 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 50000 --avg-degree 20 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 100000 --avg-degree 5 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 100000 --avg-degree 20 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)

graph-centrality-scale-free: $(EXTENSION)          ## Graph: centrality ops, Barabasi-Albert
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 1000 --avg-degree 3 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 1000 --avg-degree 5 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 1000 --avg-degree 10 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 5000 --avg-degree 3 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 5000 --avg-degree 5 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 5000 --avg-degree 10 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 10000 --avg-degree 3 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 10000 --avg-degree 5 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 10000 --avg-degree 10 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 50000 --avg-degree 3 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 50000 --avg-degree 5 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 50000 --avg-degree 10 --operations $(CENTRALITY_OPS) $(BENCH_FLAGS)

graph-community-small: $(EXTENSION)                ## Graph: community ops, small (N<=1K)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 100 --avg-degree 5 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 100 --avg-degree 20 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 500 --avg-degree 5 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 500 --avg-degree 20 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 1000 --avg-degree 5 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 1000 --avg-degree 20 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)

graph-community-medium: $(EXTENSION)               ## Graph: community ops, medium (N<=10K)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 1000 --avg-degree 5 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 1000 --avg-degree 20 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 1000 --avg-degree 50 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 5000 --avg-degree 5 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 5000 --avg-degree 20 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 5000 --avg-degree 50 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 10000 --avg-degree 5 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 10000 --avg-degree 20 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 10000 --avg-degree 50 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)

graph-community-large: $(EXTENSION)                ## Graph: community ops, large (N<=100K)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 10000 --avg-degree 5 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 10000 --avg-degree 20 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 50000 --avg-degree 5 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 50000 --avg-degree 20 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 100000 --avg-degree 5 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 100000 --avg-degree 20 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)

graph-community-scale-free: $(EXTENSION)           ## Graph: community ops, Barabasi-Albert
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 1000 --avg-degree 3 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 1000 --avg-degree 5 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 1000 --avg-degree 10 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 5000 --avg-degree 3 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 5000 --avg-degree 5 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 5000 --avg-degree 10 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 10000 --avg-degree 3 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 10000 --avg-degree 5 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 10000 --avg-degree 10 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 50000 --avg-degree 3 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 50000 --avg-degree 5 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 50000 --avg-degree 10 --operations $(COMMUNITY_OPS) $(BENCH_FLAGS)

######################################################################
# AGGREGATED TARGETS
######################################################################
analyze-vss:                                       ## Analyze vector results -> tables + plotly charts
	$(PYTHON) scripts/benchmark_vss_analyze.py

analyze-graph:                                     ## Analyze graph results -> tables + charts
	$(PYTHON) scripts/benchmark_graph_analyze.py

analyze: analyze-vss analyze-graph                 ## Analyze all benchmark results

######################################################################
# BENCHMARK MANIFESTS
######################################################################
# Each benchmarking script is self describing about the benchmarks it SHOULD run and the outputs
# The manifest targets are a self check to see which are missing to be able to run individual benchmarks in isolation.

manifest-vss-embeddings-prep:                              ## Show models-prep cache completeness
	$(PYTHON) scripts/benchmark_vss_analyze.py --prep-manifest

manifest-vss:                                      ## Show missing VSS benchmarks (JSONL)
	@$(PYTHON) $(BENCH_VSS_ANALYZE_SRC) --manifest-missing $(BENCH_FLAGS)

manifest-vss-all:                                  ## Show all VSS benchmarks (JSONL)
	@$(PYTHON) $(BENCH_VSS_ANALYZE_SRC) --manifest-all $(BENCH_FLAGS)

manifest-graph:                                    ## Show missing graph benchmarks (JSONL)
	@$(PYTHON) $(BENCH_GRAPH_ANALYZE_SRC) --manifest-missing $(BENCH_FLAGS)

manifest-graph-all:                                ## Show all graph benchmarks (JSONL)
	@$(PYTHON) $(BENCH_GRAPH_ANALYZE_SRC) --manifest-all $(BENCH_FLAGS)

manifest: manifest-vss manifest-graph              ## Show all missing benchmarks (JSONL)


clean:                                             ## Remove generated charts (keeps JSONL results)
	rm -rf $(CURDIR)/charts/*.html 
	rm -rf $(CURDIR)/charts/*.json

clean-all: clean                                   ## Remove all generated files (including JSONL results)
	rm -rf $(CURDIR)/results/**/*.sqlite 
	rm -rf $(CURDIR)/results/*.jsonl