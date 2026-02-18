/*
 * graph_csr.h — Compressed Sparse Row (CSR) representation for graphs
 *
 * Provides in-memory CSR arrays built from GraphData adjacency lists,
 * plus serialization/deserialization for BLOB storage in shadow tables.
 *
 * CSR layout:
 *   offsets[V+1] — cumulative edge count; node i's neighbors are
 *                  targets[offsets[i] .. offsets[i+1])
 *   targets[E]  — neighbor indices (int32)
 *   weights[E]  — edge weights (double), NULL if unweighted
 */
#ifndef GRAPH_CSR_H
#define GRAPH_CSR_H

#include "graph_load.h"

#include <stdint.h>

/* In-memory CSR representation for one direction (forward or reverse) */
typedef struct {
    int32_t node_count;  /* V */
    int32_t edge_count;  /* E */
    int32_t *offsets;    /* [V+1] — offsets[i] = start of node i's neighbors */
    int32_t *targets;    /* [E]   — target node indices */
    double *weights;     /* [E] or NULL if unweighted */
    int has_weights;
} CsrArray;

/* A delta operation for incremental CSR merge */
typedef struct {
    int32_t src_idx;
    int32_t dst_idx;
    double weight;
    int op; /* 1 = INSERT, 2 = DELETE */
} CsrDelta;

/*
 * Build forward and reverse CSR arrays from a GraphData adjacency structure.
 * fwd is built from g->out[], rev from g->in[].
 * Returns 0 on success, -1 on allocation failure.
 * Caller must call csr_destroy() on both arrays.
 */
int csr_build(const GraphData *g, CsrArray *fwd, CsrArray *rev);

/* Free memory owned by a CsrArray. Safe to call on a zeroed struct. */
void csr_destroy(CsrArray *csr);

/* Get the degree (neighbor count) of node idx. Returns 0 if out of range. */
int32_t csr_degree(const CsrArray *csr, int32_t idx);

/*
 * Get pointer to neighbor target indices for node idx.
 * Sets *count to the number of neighbors.
 * Returns pointer into csr->targets, or NULL if idx is out of range.
 */
const int32_t *csr_neighbors(const CsrArray *csr, int32_t idx, int32_t *count);

/*
 * Deserialize a CSR from raw BLOB data (as stored in shadow tables).
 * Makes copies of the input data — caller owns the resulting CsrArray.
 * offsets_blob: int32_t[node_count+1], targets_blob: int32_t[edge_count],
 * weights_blob: double[edge_count] or NULL.
 * Returns 0 on success, -1 on error.
 */
int csr_deserialize(CsrArray *csr,
                    const void *offsets_blob, int offsets_bytes,
                    const void *targets_blob, int targets_bytes,
                    const void *weights_blob, int weights_bytes);

/*
 * Apply delta operations to an existing CSR, producing a new CSR.
 * new_node_count may be larger than old_csr->node_count if new nodes
 * were introduced by the delta.
 * Returns 0 on success, -1 on error.
 */
int csr_apply_delta(const CsrArray *old_csr,
                    const CsrDelta *deltas, int delta_count,
                    int32_t new_node_count,
                    CsrArray *new_csr);

#endif /* GRAPH_CSR_H */
