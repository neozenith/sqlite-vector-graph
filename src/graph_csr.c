/*
 * graph_csr.c — CSR build, serialize, deserialize, and delta merge
 *
 * Pure algorithmic module — no sqlite3_* function calls. However,
 * sqlite3ext.h is included because this file is compiled as part of
 * the extension and graph_load.h pulls in sqlite3.h types.
 */
#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT3

#include "graph_csr.h"

#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════
 * CSR Build from GraphData adjacency lists
 * ═══════════════════════════════════════════════════════════════ */

static int csr_build_one(const GraphAdjList *adj, int node_count,
                         int has_weights, CsrArray *csr) {
    memset(csr, 0, sizeof(CsrArray));
    csr->node_count = node_count;
    csr->has_weights = has_weights;

    /* Count total edges */
    int32_t total = 0;
    for (int i = 0; i < node_count; i++)
        total += adj[i].count;
    csr->edge_count = total;

    /* Allocate arrays */
    csr->offsets = (int32_t *)malloc((size_t)(node_count + 1) * sizeof(int32_t));
    if (!csr->offsets)
        return -1;

    if (total > 0) {
        csr->targets = (int32_t *)malloc((size_t)total * sizeof(int32_t));
        if (!csr->targets) {
            free(csr->offsets);
            csr->offsets = NULL;
            return -1;
        }
    }

    if (has_weights && total > 0) {
        csr->weights = (double *)malloc((size_t)total * sizeof(double));
        if (!csr->weights) {
            free(csr->offsets);
            free(csr->targets);
            csr->offsets = NULL;
            csr->targets = NULL;
            return -1;
        }
    }

    /* Fill offsets and targets */
    int32_t offset = 0;
    for (int i = 0; i < node_count; i++) {
        csr->offsets[i] = offset;
        for (int j = 0; j < adj[i].count; j++) {
            csr->targets[offset + j] = (int32_t)adj[i].edges[j].target;
            if (csr->weights)
                csr->weights[offset + j] = adj[i].edges[j].weight;
        }
        offset += adj[i].count;
    }
    csr->offsets[node_count] = offset;

    return 0;
}

int csr_build(const GraphData *g, CsrArray *fwd, CsrArray *rev) {
    memset(fwd, 0, sizeof(CsrArray));
    memset(rev, 0, sizeof(CsrArray));

    if (csr_build_one(g->out, g->node_count, g->has_weights, fwd) != 0)
        return -1;
    if (csr_build_one(g->in, g->node_count, g->has_weights, rev) != 0) {
        csr_destroy(fwd);
        return -1;
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * CSR Lifecycle
 * ═══════════════════════════════════════════════════════════════ */

void csr_destroy(CsrArray *csr) {
    free(csr->offsets);
    free(csr->targets);
    free(csr->weights);
    memset(csr, 0, sizeof(CsrArray));
}

/* ═══════════════════════════════════════════════════════════════
 * CSR Query
 * ═══════════════════════════════════════════════════════════════ */

int32_t csr_degree(const CsrArray *csr, int32_t idx) {
    if (!csr || idx < 0 || idx >= csr->node_count)
        return 0;
    return csr->offsets[idx + 1] - csr->offsets[idx];
}

const int32_t *csr_neighbors(const CsrArray *csr, int32_t idx, int32_t *count) {
    if (!csr || idx < 0 || idx >= csr->node_count) {
        if (count)
            *count = 0;
        return NULL;
    }
    int32_t start = csr->offsets[idx];
    if (count)
        *count = csr->offsets[idx + 1] - start;
    return csr->targets ? csr->targets + start : NULL;
}

/* ═══════════════════════════════════════════════════════════════
 * CSR Deserialization (from shadow table BLOBs)
 * ═══════════════════════════════════════════════════════════════ */

int csr_deserialize(CsrArray *csr,
                    const void *offsets_blob, int offsets_bytes,
                    const void *targets_blob, int targets_bytes,
                    const void *weights_blob, int weights_bytes) {
    memset(csr, 0, sizeof(CsrArray));

    if (!offsets_blob || offsets_bytes < (int)sizeof(int32_t))
        return -1;

    csr->node_count = offsets_bytes / (int)sizeof(int32_t) - 1;
    if (csr->node_count < 0)
        return -1;

    csr->edge_count = targets_bytes / (int)sizeof(int32_t);
    csr->has_weights = (weights_blob != NULL && weights_bytes > 0);

    /* Copy offsets */
    csr->offsets = (int32_t *)malloc((size_t)offsets_bytes);
    if (!csr->offsets)
        return -1;
    memcpy(csr->offsets, offsets_blob, (size_t)offsets_bytes);

    /* Copy targets */
    if (csr->edge_count > 0 && targets_blob) {
        csr->targets = (int32_t *)malloc((size_t)targets_bytes);
        if (!csr->targets) {
            csr_destroy(csr);
            return -1;
        }
        memcpy(csr->targets, targets_blob, (size_t)targets_bytes);
    }

    /* Copy weights */
    if (csr->has_weights && csr->edge_count > 0) {
        csr->weights = (double *)malloc((size_t)weights_bytes);
        if (!csr->weights) {
            csr_destroy(csr);
            return -1;
        }
        memcpy(csr->weights, weights_blob, (size_t)weights_bytes);
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * CSR Delta Merge (Phase 2: incremental rebuild)
 *
 * Applies insert/delete operations to an existing CSR.
 * Algorithm (GraphBLAS-style):
 *   1. Convert old CSR to per-node adjacency lists
 *   2. Apply deletes (remove edges) and inserts (add edges)
 *   3. Rebuild CSR from modified adjacency lists
 * ═══════════════════════════════════════════════════════════════ */

int csr_apply_delta(const CsrArray *old_csr,
                    const CsrDelta *deltas, int delta_count,
                    int32_t new_node_count,
                    CsrArray *new_csr) {
    memset(new_csr, 0, sizeof(CsrArray));

    if (new_node_count < old_csr->node_count)
        new_node_count = old_csr->node_count;

    /* Allocate temporary adjacency lists */
    int *adj_count = (int *)calloc((size_t)new_node_count, sizeof(int));
    int *adj_cap = (int *)calloc((size_t)new_node_count, sizeof(int));
    int32_t **adj_targets = (int32_t **)calloc((size_t)new_node_count, sizeof(int32_t *));
    double **adj_weights = NULL;
    int has_weights = old_csr->has_weights;

    if (!adj_count || !adj_cap || !adj_targets)
        goto fail;

    if (has_weights) {
        adj_weights = (double **)calloc((size_t)new_node_count, sizeof(double *));
        if (!adj_weights)
            goto fail;
    }

    /* Step 1: Convert old CSR to adjacency lists */
    for (int32_t i = 0; i < old_csr->node_count; i++) {
        int32_t deg = old_csr->offsets[i + 1] - old_csr->offsets[i];
        if (deg == 0)
            continue;

        adj_count[i] = deg;
        adj_cap[i] = deg + 8; /* room for inserts */
        adj_targets[i] = (int32_t *)malloc((size_t)adj_cap[i] * sizeof(int32_t));
        if (!adj_targets[i])
            goto fail;
        memcpy(adj_targets[i], old_csr->targets + old_csr->offsets[i],
               (size_t)deg * sizeof(int32_t));

        if (has_weights && old_csr->weights) {
            adj_weights[i] = (double *)malloc((size_t)adj_cap[i] * sizeof(double));
            if (!adj_weights[i])
                goto fail;
            memcpy(adj_weights[i], old_csr->weights + old_csr->offsets[i],
                   (size_t)deg * sizeof(double));
        }
    }

    /* Step 2: Apply deltas */
    for (int d = 0; d < delta_count; d++) {
        int32_t src = deltas[d].src_idx;
        int32_t dst = deltas[d].dst_idx;
        if (src < 0 || src >= new_node_count || dst < 0 || dst >= new_node_count)
            continue;

        if (deltas[d].op == 2) {
            /* DELETE: find and remove dst from src's neighbor list */
            for (int j = 0; j < adj_count[src]; j++) {
                if (adj_targets[src][j] == dst) {
                    /* Swap with last element */
                    adj_count[src]--;
                    if (j < adj_count[src]) {
                        adj_targets[src][j] = adj_targets[src][adj_count[src]];
                        if (has_weights && adj_weights && adj_weights[src])
                            adj_weights[src][j] = adj_weights[src][adj_count[src]];
                    }
                    break;
                }
            }
        } else if (deltas[d].op == 1) {
            /* INSERT: add dst to src's neighbor list */
            if (adj_count[src] >= adj_cap[src]) {
                int nc = adj_cap[src] == 0 ? 8 : adj_cap[src] * 2;
                int32_t *nt = (int32_t *)realloc(adj_targets[src],
                                                  (size_t)nc * sizeof(int32_t));
                if (!nt)
                    goto fail;
                adj_targets[src] = nt;
                if (has_weights) {
                    double *nw = (double *)realloc(
                        adj_weights ? adj_weights[src] : NULL,
                        (size_t)nc * sizeof(double));
                    if (!nw)
                        goto fail;
                    if (adj_weights)
                        adj_weights[src] = nw;
                }
                adj_cap[src] = nc;
            }
            adj_targets[src][adj_count[src]] = dst;
            if (has_weights && adj_weights && adj_weights[src])
                adj_weights[src][adj_count[src]] = deltas[d].weight;
            adj_count[src]++;
        }
    }

    /* Step 3: Rebuild CSR from modified adjacency lists */
    new_csr->node_count = new_node_count;
    new_csr->has_weights = has_weights;

    int32_t total_edges = 0;
    for (int32_t i = 0; i < new_node_count; i++)
        total_edges += adj_count[i];
    new_csr->edge_count = total_edges;

    new_csr->offsets = (int32_t *)malloc((size_t)(new_node_count + 1) * sizeof(int32_t));
    if (!new_csr->offsets)
        goto fail;

    if (total_edges > 0) {
        new_csr->targets = (int32_t *)malloc((size_t)total_edges * sizeof(int32_t));
        if (!new_csr->targets)
            goto fail;
    }
    if (has_weights && total_edges > 0) {
        new_csr->weights = (double *)malloc((size_t)total_edges * sizeof(double));
        if (!new_csr->weights)
            goto fail;
    }

    int32_t offset = 0;
    for (int32_t i = 0; i < new_node_count; i++) {
        new_csr->offsets[i] = offset;
        for (int j = 0; j < adj_count[i]; j++) {
            new_csr->targets[offset + j] = adj_targets[i][j];
            if (new_csr->weights && adj_weights && adj_weights[i])
                new_csr->weights[offset + j] = adj_weights[i][j];
        }
        offset += adj_count[i];
    }
    new_csr->offsets[new_node_count] = offset;

    /* Cleanup temporary adjacency lists */
    for (int32_t i = 0; i < new_node_count; i++) {
        free(adj_targets[i]);
        if (adj_weights)
            free(adj_weights[i]);
    }
    free(adj_count);
    free(adj_cap);
    free(adj_targets);
    free(adj_weights);
    return 0;

fail:
    for (int32_t i = 0; i < new_node_count; i++) {
        if (adj_targets)
            free(adj_targets[i]);
        if (adj_weights)
            free(adj_weights[i]);
    }
    free(adj_count);
    free(adj_cap);
    free(adj_targets);
    free(adj_weights);
    csr_destroy(new_csr);
    return -1;
}
