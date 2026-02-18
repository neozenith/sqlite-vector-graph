/*
 * graph_adjacency.c — Persistent graph adjacency index virtual table
 *
 * Maintains a CSR (Compressed Sparse Row) representation of a graph
 * in shadow tables, synchronized with a source edge table via triggers.
 *
 * Phase 1: Dirty-flag-based cached full rebuild
 * Phase 2: Delta log with incremental merge (GraphBLAS-style)
 *
 * The CSR is stored as BLOBs in shadow tables. Queries check a dirty
 * flag (Phase 1) or delta count (Phase 2), rebuilding only when stale.
 * Administrative commands follow the FTS5 pattern:
 *   INSERT INTO g(g) VALUES('rebuild');
 *   INSERT INTO g(g) VALUES('incremental_rebuild');
 */
#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT3

#include "graph_adjacency.h"
#include "graph_csr.h"
#include "graph_load.h"
#include "graph_common.h"
#include "id_validate.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ═══════════════════════════════════════════════════════════════
 * Virtual Table & Cursor Structures
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    sqlite3_vtab base;
    sqlite3 *db;
    char *vtab_name;     /* virtual table name (for shadow table prefixes) */
    char *edge_table;
    char *src_col;
    char *dst_col;
    char *weight_col;    /* NULL if unweighted */
    int64_t generation;  /* increments on each rebuild */
} AdjVtab;

typedef struct {
    sqlite3_vtab_cursor base;
    sqlite3_stmt *stmt;  /* iterates _nodes JOIN _degree */
    int eof;
} AdjCursor;

/* Output column indices */
enum {
    ADJ_COL_NODE = 0,
    ADJ_COL_NODE_IDX,
    ADJ_COL_IN_DEGREE,
    ADJ_COL_OUT_DEGREE,
    ADJ_COL_W_IN_DEGREE,
    ADJ_COL_W_OUT_DEGREE,
    ADJ_COL_COMMAND,     /* hidden: same name as table, for command pattern */
    ADJ_NUM_COLS
};

/* ═══════════════════════════════════════════════════════════════
 * Argument Parsing
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    char *edge_table;
    char *src_col;
    char *dst_col;
    char *weight_col;
} AdjParams;

/* Strip surrounding quotes from a string value (single or double quotes) */
static const char *strip_quotes(const char *val, char *buf, int bufsize) {
    int len = (int)strlen(val);
    if (len >= 2 && (val[0] == '\'' || val[0] == '"')) {
        len -= 2;
        if (len >= bufsize)
            len = bufsize - 1;
        memcpy(buf, val + 1, (size_t)len);
        buf[len] = '\0';
        return buf;
    }
    return val;
}

static int parse_adjacency_params(int argc, const char *const *argv,
                                  AdjParams *params, char **errmsg) {
    memset(params, 0, sizeof(AdjParams));
    char buf[256];

    for (int i = 3; i < argc; i++) {
        const char *arg = argv[i];

        if (strncmp(arg, "edge_table=", 11) == 0) {
            const char *val = strip_quotes(arg + 11, buf, (int)sizeof(buf));
            params->edge_table = sqlite3_mprintf("%s", val);
        } else if (strncmp(arg, "src_col=", 8) == 0) {
            const char *val = strip_quotes(arg + 8, buf, (int)sizeof(buf));
            params->src_col = sqlite3_mprintf("%s", val);
        } else if (strncmp(arg, "dst_col=", 8) == 0) {
            const char *val = strip_quotes(arg + 8, buf, (int)sizeof(buf));
            params->dst_col = sqlite3_mprintf("%s", val);
        } else if (strncmp(arg, "weight_col=", 11) == 0) {
            const char *val = strip_quotes(arg + 11, buf, (int)sizeof(buf));
            params->weight_col = sqlite3_mprintf("%s", val);
        } else {
            *errmsg = sqlite3_mprintf("graph_adjacency: unknown parameter '%s'", arg);
            return SQLITE_ERROR;
        }
    }

    if (!params->edge_table || !params->src_col || !params->dst_col) {
        *errmsg = sqlite3_mprintf("graph_adjacency: edge_table, src_col, and dst_col are required");
        sqlite3_free(params->edge_table);
        sqlite3_free(params->src_col);
        sqlite3_free(params->dst_col);
        sqlite3_free(params->weight_col);
        memset(params, 0, sizeof(AdjParams));
        return SQLITE_ERROR;
    }

    /* Validate identifiers */
    if (id_validate(params->edge_table) != 0 ||
        id_validate(params->src_col) != 0 ||
        id_validate(params->dst_col) != 0) {
        *errmsg = sqlite3_mprintf("graph_adjacency: invalid table/column identifier");
        sqlite3_free(params->edge_table);
        sqlite3_free(params->src_col);
        sqlite3_free(params->dst_col);
        sqlite3_free(params->weight_col);
        memset(params, 0, sizeof(AdjParams));
        return SQLITE_ERROR;
    }
    if (params->weight_col && id_validate(params->weight_col) != 0) {
        *errmsg = sqlite3_mprintf("graph_adjacency: invalid weight column identifier");
        sqlite3_free(params->edge_table);
        sqlite3_free(params->src_col);
        sqlite3_free(params->dst_col);
        sqlite3_free(params->weight_col);
        memset(params, 0, sizeof(AdjParams));
        return SQLITE_ERROR;
    }

    return SQLITE_OK;
}

/* ═══════════════════════════════════════════════════════════════
 * Shadow Table Management
 * ═══════════════════════════════════════════════════════════════ */

static int create_shadow_tables(sqlite3 *db, const char *name) {
    char *sql;
    int rc;

    /* Config table */
    sql = sqlite3_mprintf(
        "CREATE TABLE IF NOT EXISTS \"%w_config\" "
        "(key TEXT PRIMARY KEY, value TEXT)", name);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return rc;

    /* Node registry: string ID ↔ integer index */
    sql = sqlite3_mprintf(
        "CREATE TABLE IF NOT EXISTS \"%w_nodes\" "
        "(idx INTEGER PRIMARY KEY, id TEXT UNIQUE)", name);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return rc;

    /* Degree sequence */
    sql = sqlite3_mprintf(
        "CREATE TABLE IF NOT EXISTS \"%w_degree\" "
        "(idx INTEGER PRIMARY KEY, in_deg INTEGER, out_deg INTEGER, "
        "w_in_deg REAL, w_out_deg REAL)", name);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return rc;

    /* Forward CSR (outgoing edges) — block_id=0 for monolithic Phase 1 */
    sql = sqlite3_mprintf(
        "CREATE TABLE IF NOT EXISTS \"%w_csr_fwd\" "
        "(block_id INTEGER PRIMARY KEY, offsets BLOB, targets BLOB, weights BLOB)",
        name);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return rc;

    /* Reverse CSR (incoming edges) */
    sql = sqlite3_mprintf(
        "CREATE TABLE IF NOT EXISTS \"%w_csr_rev\" "
        "(block_id INTEGER PRIMARY KEY, offsets BLOB, targets BLOB, weights BLOB)",
        name);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return rc;

    /* Delta log for incremental merge */
    sql = sqlite3_mprintf(
        "CREATE TABLE IF NOT EXISTS \"%w_delta\" "
        "(rowid INTEGER PRIMARY KEY, src TEXT, dst TEXT, weight REAL, op INTEGER)",
        name);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    return rc;
}

static int drop_shadow_tables(sqlite3 *db, const char *name) {
    const char *suffixes[] = {"_config", "_nodes", "_degree",
                              "_csr_fwd", "_csr_rev", "_delta"};
    for (int i = 0; i < 6; i++) {
        char *sql = sqlite3_mprintf("DROP TABLE IF EXISTS \"%w%s\"",
                                    name, suffixes[i]);
        sqlite3_exec(db, sql, NULL, NULL, NULL);
        sqlite3_free(sql);
    }
    return SQLITE_OK;
}

/* ═══════════════════════════════════════════════════════════════
 * Trigger Management
 * ═══════════════════════════════════════════════════════════════ */

static int install_triggers(sqlite3 *db, const char *vtab_name,
                            const char *edge_table, const char *src_col,
                            const char *dst_col, const char *weight_col) {
    char *sql;
    int rc;
    const char *w_expr = weight_col ? weight_col : "NULL";

    /* AFTER INSERT */
    sql = sqlite3_mprintf(
        "CREATE TRIGGER IF NOT EXISTS \"%w_ai\" AFTER INSERT ON \"%w\" BEGIN "
        "INSERT INTO \"%w_delta\"(src, dst, weight, op) "
        "VALUES (NEW.\"%w\", NEW.\"%w\", NEW.\"%w\", 1); END",
        vtab_name, edge_table, vtab_name, src_col, dst_col, w_expr);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return rc;

    /* AFTER DELETE */
    sql = sqlite3_mprintf(
        "CREATE TRIGGER IF NOT EXISTS \"%w_ad\" AFTER DELETE ON \"%w\" BEGIN "
        "INSERT INTO \"%w_delta\"(src, dst, weight, op) "
        "VALUES (OLD.\"%w\", OLD.\"%w\", OLD.\"%w\", 2); END",
        vtab_name, edge_table, vtab_name, src_col, dst_col, w_expr);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return rc;

    /* AFTER UPDATE: log delete of old + insert of new */
    sql = sqlite3_mprintf(
        "CREATE TRIGGER IF NOT EXISTS \"%w_au\" AFTER UPDATE ON \"%w\" BEGIN "
        "INSERT INTO \"%w_delta\"(src, dst, weight, op) "
        "VALUES (OLD.\"%w\", OLD.\"%w\", OLD.\"%w\", 2); "
        "INSERT INTO \"%w_delta\"(src, dst, weight, op) "
        "VALUES (NEW.\"%w\", NEW.\"%w\", NEW.\"%w\", 1); END",
        vtab_name, edge_table,
        vtab_name, src_col, dst_col, w_expr,
        vtab_name, src_col, dst_col, w_expr);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    return rc;
}

static void remove_triggers(sqlite3 *db, const char *vtab_name) {
    const char *suffixes[] = {"_ai", "_ad", "_au"};
    for (int i = 0; i < 3; i++) {
        char *sql = sqlite3_mprintf("DROP TRIGGER IF EXISTS \"%w%s\"",
                                    vtab_name, suffixes[i]);
        sqlite3_exec(db, sql, NULL, NULL, NULL);
        sqlite3_free(sql);
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Config Helpers
 * ═══════════════════════════════════════════════════════════════ */

static int config_set(sqlite3 *db, const char *name,
                      const char *key, const char *value) {
    char *sql = sqlite3_mprintf(
        "INSERT OR REPLACE INTO \"%w_config\"(key, value) VALUES ('%w', '%w')",
        name, key, value);
    int rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    return rc;
}

static int config_set_int(sqlite3 *db, const char *name,
                          const char *key, int64_t value) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%lld", (long long)value);
    return config_set(db, name, key, buf);
}

static int64_t config_get_int(sqlite3 *db, const char *name,
                              const char *key, int64_t def) {
    sqlite3_stmt *stmt;
    char *sql = sqlite3_mprintf(
        "SELECT value FROM \"%w_config\" WHERE key='%w'", name, key);
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK)
        return def;

    int64_t result = def;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *val = (const char *)sqlite3_column_text(stmt, 0);
        if (val)
            result = atoll(val);
    }
    sqlite3_finalize(stmt);
    return result;
}

/* ═══════════════════════════════════════════════════════════════
 * Delta Table Queries
 * ═══════════════════════════════════════════════════════════════ */

static int64_t delta_count(sqlite3 *db, const char *name) {
    sqlite3_stmt *stmt;
    char *sql = sqlite3_mprintf("SELECT COUNT(*) FROM \"%w_delta\"", name);
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return 0;

    int64_t count = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW)
        count = sqlite3_column_int64(stmt, 0);
    sqlite3_finalize(stmt);
    return count;
}

static int delta_clear(sqlite3 *db, const char *name) {
    char *sql = sqlite3_mprintf("DELETE FROM \"%w_delta\"", name);
    int rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    return rc;
}

/* ═══════════════════════════════════════════════════════════════
 * CSR Storage (shadow table BLOBs)
 * ═══════════════════════════════════════════════════════════════ */

static int store_csr_blob(sqlite3 *db, const char *name,
                          const char *suffix, const CsrArray *csr) {
    sqlite3_stmt *stmt;
    char *sql = sqlite3_mprintf(
        "INSERT OR REPLACE INTO \"%w%s\"(block_id, offsets, targets, weights) "
        "VALUES (0, ?, ?, ?)", name, suffix);
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return rc;

    /* Bind offsets BLOB */
    int offsets_bytes = (csr->node_count + 1) * (int)sizeof(int32_t);
    sqlite3_bind_blob(stmt, 1, csr->offsets, offsets_bytes, SQLITE_STATIC);

    /* Bind targets BLOB */
    if (csr->edge_count > 0 && csr->targets) {
        int targets_bytes = csr->edge_count * (int)sizeof(int32_t);
        sqlite3_bind_blob(stmt, 2, csr->targets, targets_bytes, SQLITE_STATIC);
    } else {
        sqlite3_bind_blob(stmt, 2, "", 0, SQLITE_STATIC);
    }

    /* Bind weights BLOB */
    if (csr->has_weights && csr->weights && csr->edge_count > 0) {
        int weights_bytes = csr->edge_count * (int)sizeof(double);
        sqlite3_bind_blob(stmt, 3, csr->weights, weights_bytes, SQLITE_STATIC);
    } else {
        sqlite3_bind_null(stmt, 3);
    }

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return (rc == SQLITE_DONE) ? SQLITE_OK : rc;
}

static int load_csr_blob(sqlite3 *db, const char *name,
                         const char *suffix, CsrArray *csr) {
    sqlite3_stmt *stmt;
    char *sql = sqlite3_mprintf(
        "SELECT offsets, targets, weights FROM \"%w%s\" WHERE block_id=0",
        name, suffix);
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return rc;

    if (sqlite3_step(stmt) != SQLITE_ROW) {
        sqlite3_finalize(stmt);
        memset(csr, 0, sizeof(CsrArray));
        return SQLITE_OK; /* no data yet */
    }

    const void *offsets = sqlite3_column_blob(stmt, 0);
    int offsets_bytes = sqlite3_column_bytes(stmt, 0);
    const void *targets = sqlite3_column_blob(stmt, 1);
    int targets_bytes = sqlite3_column_bytes(stmt, 1);
    const void *weights = sqlite3_column_blob(stmt, 2);
    int weights_bytes = sqlite3_column_bytes(stmt, 2);

    rc = csr_deserialize(csr, offsets, offsets_bytes,
                         targets, targets_bytes, weights, weights_bytes);
    sqlite3_finalize(stmt);
    return (rc == 0) ? SQLITE_OK : SQLITE_ERROR;
}

/* ═══════════════════════════════════════════════════════════════
 * Node Registry & Degree Storage
 * ═══════════════════════════════════════════════════════════════ */

static int store_nodes(sqlite3 *db, const char *name, const GraphData *g) {
    /* Clear existing */
    char *sql = sqlite3_mprintf("DELETE FROM \"%w_nodes\"", name);
    sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);

    /* Insert all nodes */
    sql = sqlite3_mprintf(
        "INSERT INTO \"%w_nodes\"(idx, id) VALUES (?, ?)", name);
    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return rc;

    for (int i = 0; i < g->node_count; i++) {
        sqlite3_bind_int(stmt, 1, i);
        sqlite3_bind_text(stmt, 2, g->ids[i], -1, SQLITE_STATIC);
        sqlite3_step(stmt);
        sqlite3_reset(stmt);
    }
    sqlite3_finalize(stmt);
    return SQLITE_OK;
}

static int store_degrees(sqlite3 *db, const char *name,
                         const GraphData *g,
                         const CsrArray *fwd, const CsrArray *rev) {
    /* Clear existing */
    char *sql = sqlite3_mprintf("DELETE FROM \"%w_degree\"", name);
    sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);

    /* Insert degree sequence */
    sql = sqlite3_mprintf(
        "INSERT INTO \"%w_degree\"(idx, in_deg, out_deg, w_in_deg, w_out_deg) "
        "VALUES (?, ?, ?, ?, ?)", name);
    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return rc;

    for (int i = 0; i < g->node_count; i++) {
        int32_t out_deg = csr_degree(fwd, (int32_t)i);
        int32_t in_deg = csr_degree(rev, (int32_t)i);

        /* Compute weighted degrees */
        double w_out = 0.0, w_in = 0.0;
        if (fwd->has_weights && fwd->weights) {
            int32_t start = fwd->offsets[i];
            for (int32_t j = start; j < fwd->offsets[i + 1]; j++)
                w_out += fwd->weights[j];
        } else {
            w_out = (double)out_deg;
        }
        if (rev->has_weights && rev->weights) {
            int32_t start = rev->offsets[i];
            for (int32_t j = start; j < rev->offsets[i + 1]; j++)
                w_in += rev->weights[j];
        } else {
            w_in = (double)in_deg;
        }

        sqlite3_bind_int(stmt, 1, i);
        sqlite3_bind_int(stmt, 2, in_deg);
        sqlite3_bind_int(stmt, 3, out_deg);
        sqlite3_bind_double(stmt, 4, w_in);
        sqlite3_bind_double(stmt, 5, w_out);
        sqlite3_step(stmt);
        sqlite3_reset(stmt);
    }
    sqlite3_finalize(stmt);
    return SQLITE_OK;
}

/* ═══════════════════════════════════════════════════════════════
 * Full Rebuild
 * ═══════════════════════════════════════════════════════════════ */

static int adj_full_rebuild(AdjVtab *vtab) {
    GraphData g;
    GraphLoadConfig config;
    memset(&config, 0, sizeof(config));
    config.edge_table = vtab->edge_table;
    config.src_col = vtab->src_col;
    config.dst_col = vtab->dst_col;
    config.weight_col = vtab->weight_col;
    config.direction = "both";

    char *errmsg = NULL;
    graph_data_init(&g);
    int rc = graph_data_load(vtab->db, &config, &g, &errmsg);
    if (rc != SQLITE_OK) {
        vtab->base.zErrMsg = errmsg;
        graph_data_destroy(&g);
        return rc;
    }

    /* Build CSR */
    CsrArray fwd, rev;
    if (csr_build(&g, &fwd, &rev) != 0) {
        graph_data_destroy(&g);
        vtab->base.zErrMsg = sqlite3_mprintf("graph_adjacency: CSR build failed (OOM)");
        return SQLITE_NOMEM;
    }

    /* Wrap in a transaction for atomicity */
    sqlite3_exec(vtab->db, "SAVEPOINT adj_rebuild", NULL, NULL, NULL);

    /* Store node registry */
    rc = store_nodes(vtab->db, vtab->vtab_name, &g);
    if (rc != SQLITE_OK) goto rollback;

    /* Store CSR BLOBs */
    rc = store_csr_blob(vtab->db, vtab->vtab_name, "_csr_fwd", &fwd);
    if (rc != SQLITE_OK) goto rollback;
    rc = store_csr_blob(vtab->db, vtab->vtab_name, "_csr_rev", &rev);
    if (rc != SQLITE_OK) goto rollback;

    /* Store degree sequence */
    rc = store_degrees(vtab->db, vtab->vtab_name, &g, &fwd, &rev);
    if (rc != SQLITE_OK) goto rollback;

    /* Update config metadata */
    vtab->generation++;
    config_set_int(vtab->db, vtab->vtab_name, "generation", vtab->generation);
    config_set_int(vtab->db, vtab->vtab_name, "node_count", g.node_count);
    config_set_int(vtab->db, vtab->vtab_name, "edge_count", g.edge_count);

    /* Clear delta log */
    delta_clear(vtab->db, vtab->vtab_name);

    sqlite3_exec(vtab->db, "RELEASE adj_rebuild", NULL, NULL, NULL);

    csr_destroy(&fwd);
    csr_destroy(&rev);
    graph_data_destroy(&g);
    return SQLITE_OK;

rollback:
    sqlite3_exec(vtab->db, "ROLLBACK TO adj_rebuild", NULL, NULL, NULL);
    sqlite3_exec(vtab->db, "RELEASE adj_rebuild", NULL, NULL, NULL);
    csr_destroy(&fwd);
    csr_destroy(&rev);
    graph_data_destroy(&g);
    return rc;
}

/* ═══════════════════════════════════════════════════════════════
 * Incremental Rebuild (Phase 2: Delta Merge)
 * ═══════════════════════════════════════════════════════════════ */

static int adj_incremental_rebuild(AdjVtab *vtab) {
    int rc;

    /* Load existing CSR from shadow tables */
    CsrArray old_fwd, old_rev;
    rc = load_csr_blob(vtab->db, vtab->vtab_name, "_csr_fwd", &old_fwd);
    if (rc != SQLITE_OK) return rc;
    rc = load_csr_blob(vtab->db, vtab->vtab_name, "_csr_rev", &old_rev);
    if (rc != SQLITE_OK) { csr_destroy(&old_fwd); return rc; }

    /* Load node registry for string→index mapping */
    /* Build a lookup map: read _nodes, create hash map */
    GraphData node_reg;
    graph_data_init(&node_reg);

    {
        sqlite3_stmt *stmt;
        char *sql = sqlite3_mprintf(
            "SELECT idx, id FROM \"%w_nodes\" ORDER BY idx", vtab->vtab_name);
        rc = sqlite3_prepare_v2(vtab->db, sql, -1, &stmt, NULL);
        sqlite3_free(sql);
        if (rc != SQLITE_OK) {
            csr_destroy(&old_fwd);
            csr_destroy(&old_rev);
            graph_data_destroy(&node_reg);
            return rc;
        }
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char *id = (const char *)sqlite3_column_text(stmt, 1);
            if (id) graph_data_find_or_add(&node_reg, id);
        }
        sqlite3_finalize(stmt);
    }

    /* Read delta entries and resolve string IDs to indices */
    sqlite3_stmt *delta_stmt;
    char *sql = sqlite3_mprintf(
        "SELECT src, dst, weight, op FROM \"%w_delta\"", vtab->vtab_name);
    rc = sqlite3_prepare_v2(vtab->db, sql, -1, &delta_stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) {
        csr_destroy(&old_fwd);
        csr_destroy(&old_rev);
        graph_data_destroy(&node_reg);
        return rc;
    }

    /* Collect delta entries */
    int dcap = 64;
    int dcount = 0;
    CsrDelta *fwd_deltas = (CsrDelta *)malloc((size_t)dcap * sizeof(CsrDelta));
    CsrDelta *rev_deltas = (CsrDelta *)malloc((size_t)dcap * sizeof(CsrDelta));

    while (sqlite3_step(delta_stmt) == SQLITE_ROW) {
        const char *src = (const char *)sqlite3_column_text(delta_stmt, 0);
        const char *dst = (const char *)sqlite3_column_text(delta_stmt, 1);
        double weight = sqlite3_column_double(delta_stmt, 2);
        int op = sqlite3_column_int(delta_stmt, 3);

        if (!src || !dst) continue;

        /* Resolve or create node indices */
        int si = graph_data_find_or_add(&node_reg, src);
        int di = graph_data_find_or_add(&node_reg, dst);

        if (dcount >= dcap) {
            dcap *= 2;
            fwd_deltas = (CsrDelta *)realloc(fwd_deltas, (size_t)dcap * sizeof(CsrDelta));
            rev_deltas = (CsrDelta *)realloc(rev_deltas, (size_t)dcap * sizeof(CsrDelta));
        }

        /* Forward: src → dst */
        fwd_deltas[dcount].src_idx = (int32_t)si;
        fwd_deltas[dcount].dst_idx = (int32_t)di;
        fwd_deltas[dcount].weight = weight;
        fwd_deltas[dcount].op = op;

        /* Reverse: dst ← src */
        rev_deltas[dcount].src_idx = (int32_t)di;
        rev_deltas[dcount].dst_idx = (int32_t)si;
        rev_deltas[dcount].weight = weight;
        rev_deltas[dcount].op = op;

        dcount++;
    }
    sqlite3_finalize(delta_stmt);

    if (dcount == 0) {
        /* Nothing to merge */
        free(fwd_deltas);
        free(rev_deltas);
        csr_destroy(&old_fwd);
        csr_destroy(&old_rev);
        graph_data_destroy(&node_reg);
        return SQLITE_OK;
    }

    /* Apply deltas */
    int32_t new_node_count = (int32_t)node_reg.node_count;
    CsrArray new_fwd, new_rev;

    if (csr_apply_delta(&old_fwd, fwd_deltas, dcount, new_node_count, &new_fwd) != 0 ||
        csr_apply_delta(&old_rev, rev_deltas, dcount, new_node_count, &new_rev) != 0) {
        free(fwd_deltas);
        free(rev_deltas);
        csr_destroy(&old_fwd);
        csr_destroy(&old_rev);
        csr_destroy(&new_fwd);
        csr_destroy(&new_rev);
        graph_data_destroy(&node_reg);
        /* Fall back to full rebuild */
        return adj_full_rebuild(vtab);
    }

    free(fwd_deltas);
    free(rev_deltas);
    csr_destroy(&old_fwd);
    csr_destroy(&old_rev);

    /* Store updated data */
    sqlite3_exec(vtab->db, "SAVEPOINT adj_incr", NULL, NULL, NULL);

    /* Update node registry if new nodes were added */
    rc = store_nodes(vtab->db, vtab->vtab_name, &node_reg);
    if (rc != SQLITE_OK) goto incr_rollback;

    rc = store_csr_blob(vtab->db, vtab->vtab_name, "_csr_fwd", &new_fwd);
    if (rc != SQLITE_OK) goto incr_rollback;
    rc = store_csr_blob(vtab->db, vtab->vtab_name, "_csr_rev", &new_rev);
    if (rc != SQLITE_OK) goto incr_rollback;

    /* Recompute degrees from new CSR */
    {
        char *dsql = sqlite3_mprintf("DELETE FROM \"%w_degree\"", vtab->vtab_name);
        sqlite3_exec(vtab->db, dsql, NULL, NULL, NULL);
        sqlite3_free(dsql);

        dsql = sqlite3_mprintf(
            "INSERT INTO \"%w_degree\"(idx, in_deg, out_deg, w_in_deg, w_out_deg) "
            "VALUES (?, ?, ?, ?, ?)", vtab->vtab_name);
        sqlite3_stmt *dstmt;
        rc = sqlite3_prepare_v2(vtab->db, dsql, -1, &dstmt, NULL);
        sqlite3_free(dsql);
        if (rc != SQLITE_OK) goto incr_rollback;

        for (int32_t i = 0; i < new_node_count; i++) {
            int32_t out_deg = csr_degree(&new_fwd, i);
            int32_t in_deg = csr_degree(&new_rev, i);
            double w_out = 0.0, w_in = 0.0;

            if (new_fwd.has_weights && new_fwd.weights) {
                for (int32_t j = new_fwd.offsets[i]; j < new_fwd.offsets[i+1]; j++)
                    w_out += new_fwd.weights[j];
            } else {
                w_out = (double)out_deg;
            }
            if (new_rev.has_weights && new_rev.weights) {
                for (int32_t j = new_rev.offsets[i]; j < new_rev.offsets[i+1]; j++)
                    w_in += new_rev.weights[j];
            } else {
                w_in = (double)in_deg;
            }

            sqlite3_bind_int(dstmt, 1, i);
            sqlite3_bind_int(dstmt, 2, in_deg);
            sqlite3_bind_int(dstmt, 3, out_deg);
            sqlite3_bind_double(dstmt, 4, w_in);
            sqlite3_bind_double(dstmt, 5, w_out);
            sqlite3_step(dstmt);
            sqlite3_reset(dstmt);
        }
        sqlite3_finalize(dstmt);
    }

    vtab->generation++;
    config_set_int(vtab->db, vtab->vtab_name, "generation", vtab->generation);
    config_set_int(vtab->db, vtab->vtab_name, "node_count", new_node_count);
    {
        int64_t new_edge_count = new_fwd.edge_count;
        config_set_int(vtab->db, vtab->vtab_name, "edge_count", new_edge_count);
    }
    delta_clear(vtab->db, vtab->vtab_name);

    sqlite3_exec(vtab->db, "RELEASE adj_incr", NULL, NULL, NULL);

    csr_destroy(&new_fwd);
    csr_destroy(&new_rev);
    graph_data_destroy(&node_reg);
    return SQLITE_OK;

incr_rollback:
    sqlite3_exec(vtab->db, "ROLLBACK TO adj_incr", NULL, NULL, NULL);
    sqlite3_exec(vtab->db, "RELEASE adj_incr", NULL, NULL, NULL);
    csr_destroy(&new_fwd);
    csr_destroy(&new_rev);
    graph_data_destroy(&node_reg);
    return rc;
}

/* ═══════════════════════════════════════════════════════════════
 * Auto-rebuild: check delta and rebuild if stale
 * ═══════════════════════════════════════════════════════════════ */

static int adj_ensure_fresh(AdjVtab *vtab) {
    int64_t dc = delta_count(vtab->db, vtab->vtab_name);
    if (dc == 0)
        return SQLITE_OK; /* clean */

    /* Check if first build has happened */
    int64_t gen = config_get_int(vtab->db, vtab->vtab_name, "generation", 0);
    if (gen == 0) {
        /* Never built — full rebuild required */
        return adj_full_rebuild(vtab);
    }

    /* Decide: incremental vs full rebuild */
    int64_t edge_count = config_get_int(vtab->db, vtab->vtab_name, "edge_count", 0);
    int64_t threshold = edge_count / 10; /* 10% of edge count */
    if (threshold < 10)
        threshold = 10;

    if (dc <= threshold) {
        return adj_incremental_rebuild(vtab);
    } else {
        return adj_full_rebuild(vtab);
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Virtual Table Methods
 * ═══════════════════════════════════════════════════════════════ */

static int adj_init(sqlite3 *db, void *pAux, int argc,
                    const char *const *argv, sqlite3_vtab **ppVTab,
                    char **pzErr, int is_create) {
    (void)pAux;

    AdjParams params;
    int rc = parse_adjacency_params(argc, argv, &params, pzErr);
    if (rc != SQLITE_OK)
        return rc;

    /* Declare schema — command column is hidden, named after the table */
    char *schema = sqlite3_mprintf(
        "CREATE TABLE x("
        "node TEXT, node_idx INTEGER, "
        "in_degree INTEGER, out_degree INTEGER, "
        "weighted_in_degree REAL, weighted_out_degree REAL, "
        "\"%w\" HIDDEN)", argv[2]);
    rc = sqlite3_declare_vtab(db, schema);
    sqlite3_free(schema);
    if (rc != SQLITE_OK) {
        sqlite3_free(params.edge_table);
        sqlite3_free(params.src_col);
        sqlite3_free(params.dst_col);
        sqlite3_free(params.weight_col);
        return rc;
    }

    /* Allocate vtab */
    AdjVtab *vtab = (AdjVtab *)sqlite3_malloc(sizeof(AdjVtab));
    if (!vtab) {
        sqlite3_free(params.edge_table);
        sqlite3_free(params.src_col);
        sqlite3_free(params.dst_col);
        sqlite3_free(params.weight_col);
        return SQLITE_NOMEM;
    }
    memset(vtab, 0, sizeof(AdjVtab));
    vtab->db = db;
    vtab->vtab_name = sqlite3_mprintf("%s", argv[2]);
    vtab->edge_table = params.edge_table;
    vtab->src_col = params.src_col;
    vtab->dst_col = params.dst_col;
    vtab->weight_col = params.weight_col;

    if (is_create) {
        /* Create shadow tables */
        rc = create_shadow_tables(db, argv[2]);
        if (rc != SQLITE_OK) {
            *pzErr = sqlite3_mprintf("graph_adjacency: failed to create shadow tables");
            sqlite3_free(vtab->vtab_name);
            sqlite3_free(vtab->edge_table);
            sqlite3_free(vtab->src_col);
            sqlite3_free(vtab->dst_col);
            sqlite3_free(vtab->weight_col);
            sqlite3_free(vtab);
            return rc;
        }

        /* Save config */
        config_set(db, argv[2], "edge_table", params.edge_table);
        config_set(db, argv[2], "src_col", params.src_col);
        config_set(db, argv[2], "dst_col", params.dst_col);
        if (params.weight_col)
            config_set(db, argv[2], "weight_col", params.weight_col);
        config_set_int(db, argv[2], "generation", 0);

        /* Install triggers on edge table */
        rc = install_triggers(db, argv[2], params.edge_table,
                              params.src_col, params.dst_col, params.weight_col);
        if (rc != SQLITE_OK) {
            *pzErr = sqlite3_mprintf("graph_adjacency: failed to install triggers");
            drop_shadow_tables(db, argv[2]);
            sqlite3_free(vtab->vtab_name);
            sqlite3_free(vtab->edge_table);
            sqlite3_free(vtab->src_col);
            sqlite3_free(vtab->dst_col);
            sqlite3_free(vtab->weight_col);
            sqlite3_free(vtab);
            return rc;
        }

        /* Initial full rebuild */
        vtab->generation = 0;
        *ppVTab = &vtab->base;
        rc = adj_full_rebuild(vtab);
        if (rc != SQLITE_OK) {
            *pzErr = sqlite3_mprintf("graph_adjacency: initial rebuild failed");
            remove_triggers(db, argv[2]);
            drop_shadow_tables(db, argv[2]);
            sqlite3_free(vtab->vtab_name);
            sqlite3_free(vtab->edge_table);
            sqlite3_free(vtab->src_col);
            sqlite3_free(vtab->dst_col);
            sqlite3_free(vtab->weight_col);
            sqlite3_free(vtab);
            *ppVTab = NULL;
            return rc;
        }
        return SQLITE_OK;
    }

    /* xConnect: load generation from config */
    vtab->generation = config_get_int(db, argv[2], "generation", 0);
    *ppVTab = &vtab->base;
    return SQLITE_OK;
}

static int adj_xCreate(sqlite3 *db, void *pAux, int argc,
                       const char *const *argv, sqlite3_vtab **ppVTab,
                       char **pzErr) {
    return adj_init(db, pAux, argc, argv, ppVTab, pzErr, 1);
}

static int adj_xConnect(sqlite3 *db, void *pAux, int argc,
                        const char *const *argv, sqlite3_vtab **ppVTab,
                        char **pzErr) {
    return adj_init(db, pAux, argc, argv, ppVTab, pzErr, 0);
}

static int adj_xDisconnect(sqlite3_vtab *pVTab) {
    AdjVtab *vtab = (AdjVtab *)pVTab;
    sqlite3_free(vtab->vtab_name);
    sqlite3_free(vtab->edge_table);
    sqlite3_free(vtab->src_col);
    sqlite3_free(vtab->dst_col);
    sqlite3_free(vtab->weight_col);
    sqlite3_free(vtab);
    return SQLITE_OK;
}

static int adj_xDestroy(sqlite3_vtab *pVTab) {
    AdjVtab *vtab = (AdjVtab *)pVTab;

    /* Remove triggers */
    remove_triggers(vtab->db, vtab->vtab_name);

    /* Drop shadow tables */
    drop_shadow_tables(vtab->db, vtab->vtab_name);

    return adj_xDisconnect(pVTab);
}

/* ─── Cursor ───────────────────────────────────────────────── */

static int adj_xOpen(sqlite3_vtab *pVTab, sqlite3_vtab_cursor **ppCursor) {
    (void)pVTab;
    AdjCursor *cur = (AdjCursor *)sqlite3_malloc(sizeof(AdjCursor));
    if (!cur)
        return SQLITE_NOMEM;
    memset(cur, 0, sizeof(AdjCursor));
    *ppCursor = &cur->base;
    return SQLITE_OK;
}

static int adj_xClose(sqlite3_vtab_cursor *pCursor) {
    AdjCursor *cur = (AdjCursor *)pCursor;
    if (cur->stmt)
        sqlite3_finalize(cur->stmt);
    sqlite3_free(cur);
    return SQLITE_OK;
}

/* ─── xBestIndex ───────────────────────────────────────────── */

static int adj_xBestIndex(sqlite3_vtab *pVTab, sqlite3_index_info *pInfo) {
    (void)pVTab;
    int has_node = -1;

    for (int i = 0; i < pInfo->nConstraint; i++) {
        if (!pInfo->aConstraint[i].usable)
            continue;
        if (pInfo->aConstraint[i].iColumn == ADJ_COL_NODE &&
            pInfo->aConstraint[i].op == SQLITE_INDEX_CONSTRAINT_EQ) {
            has_node = i;
        }
    }

    if (has_node >= 0) {
        /* Point lookup: node = ? */
        pInfo->idxNum = 1;
        pInfo->aConstraintUsage[has_node].argvIndex = 1;
        pInfo->aConstraintUsage[has_node].omit = 1;
        pInfo->estimatedCost = 1.0;
        pInfo->estimatedRows = 1;
    } else {
        /* Full scan */
        pInfo->idxNum = 0;
        pInfo->estimatedCost = 1000.0;
        pInfo->estimatedRows = 1000;
    }
    return SQLITE_OK;
}

/* ─── xFilter ──────────────────────────────────────────────── */

static int adj_xFilter(sqlite3_vtab_cursor *pCursor, int idxNum,
                       const char *idxStr, int argc, sqlite3_value **argv) {
    (void)idxStr;
    AdjCursor *cur = (AdjCursor *)pCursor;
    AdjVtab *vtab = (AdjVtab *)cur->base.pVtab;

    /* Clean up previous query */
    if (cur->stmt) {
        sqlite3_finalize(cur->stmt);
        cur->stmt = NULL;
    }

    /* Ensure CSR is fresh */
    int rc = adj_ensure_fresh(vtab);
    if (rc != SQLITE_OK)
        return rc;

    /* Build query over shadow tables */
    char *sql;
    if (idxNum == 1 && argc >= 1) {
        /* Point lookup */
        sql = sqlite3_mprintf(
            "SELECT n.idx, n.id, COALESCE(d.in_deg, 0), COALESCE(d.out_deg, 0), "
            "COALESCE(d.w_in_deg, 0.0), COALESCE(d.w_out_deg, 0.0) "
            "FROM \"%w_nodes\" n "
            "LEFT JOIN \"%w_degree\" d ON n.idx = d.idx "
            "WHERE n.id = ?",
            vtab->vtab_name, vtab->vtab_name);
    } else {
        /* Full scan */
        sql = sqlite3_mprintf(
            "SELECT n.idx, n.id, COALESCE(d.in_deg, 0), COALESCE(d.out_deg, 0), "
            "COALESCE(d.w_in_deg, 0.0), COALESCE(d.w_out_deg, 0.0) "
            "FROM \"%w_nodes\" n "
            "LEFT JOIN \"%w_degree\" d ON n.idx = d.idx "
            "ORDER BY n.idx",
            vtab->vtab_name, vtab->vtab_name);
    }

    rc = sqlite3_prepare_v2(vtab->db, sql, -1, &cur->stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK)
        return rc;

    /* Bind node filter if point lookup */
    if (idxNum == 1 && argc >= 1) {
        sqlite3_bind_value(cur->stmt, 1, argv[0]);
    }

    /* Step to first row */
    rc = sqlite3_step(cur->stmt);
    cur->eof = (rc != SQLITE_ROW);
    return SQLITE_OK;
}

static int adj_xNext(sqlite3_vtab_cursor *pCursor) {
    AdjCursor *cur = (AdjCursor *)pCursor;
    int rc = sqlite3_step(cur->stmt);
    cur->eof = (rc != SQLITE_ROW);
    return SQLITE_OK;
}

static int adj_xEof(sqlite3_vtab_cursor *pCursor) {
    return ((AdjCursor *)pCursor)->eof;
}

static int adj_xColumn(sqlite3_vtab_cursor *pCursor, sqlite3_context *ctx, int col) {
    AdjCursor *cur = (AdjCursor *)pCursor;

    switch (col) {
    case ADJ_COL_NODE:
        sqlite3_result_value(ctx, sqlite3_column_value(cur->stmt, 1));
        break;
    case ADJ_COL_NODE_IDX:
        sqlite3_result_value(ctx, sqlite3_column_value(cur->stmt, 0));
        break;
    case ADJ_COL_IN_DEGREE:
        sqlite3_result_value(ctx, sqlite3_column_value(cur->stmt, 2));
        break;
    case ADJ_COL_OUT_DEGREE:
        sqlite3_result_value(ctx, sqlite3_column_value(cur->stmt, 3));
        break;
    case ADJ_COL_W_IN_DEGREE:
        sqlite3_result_value(ctx, sqlite3_column_value(cur->stmt, 4));
        break;
    case ADJ_COL_W_OUT_DEGREE:
        sqlite3_result_value(ctx, sqlite3_column_value(cur->stmt, 5));
        break;
    case ADJ_COL_COMMAND:
        sqlite3_result_null(ctx);
        break;
    default:
        sqlite3_result_null(ctx);
    }
    return SQLITE_OK;
}

static int adj_xRowid(sqlite3_vtab_cursor *pCursor, sqlite3_int64 *pRowid) {
    AdjCursor *cur = (AdjCursor *)pCursor;
    *pRowid = sqlite3_column_int64(cur->stmt, 0);
    return SQLITE_OK;
}

/* ─── xUpdate (command pattern) ────────────────────────────── */

static int adj_xUpdate(sqlite3_vtab *pVTab, int argc,
                       sqlite3_value **argv, sqlite3_int64 *pRowid) {
    (void)pRowid;
    AdjVtab *vtab = (AdjVtab *)pVTab;

    /* Only handle INSERT with command column set (FTS5 pattern) */
    if (argc < 2 || sqlite3_value_type(argv[0]) != SQLITE_NULL) {
        vtab->base.zErrMsg = sqlite3_mprintf(
            "graph_adjacency: direct INSERT/UPDATE/DELETE not supported; "
            "modify the source edge table instead");
        return SQLITE_ERROR;
    }

    /* Check if command column (last column) has a value */
    int cmd_col = 2 + ADJ_COL_COMMAND;
    if (cmd_col >= argc || sqlite3_value_type(argv[cmd_col]) == SQLITE_NULL) {
        vtab->base.zErrMsg = sqlite3_mprintf(
            "graph_adjacency: use INSERT INTO t(t) VALUES('rebuild') for commands");
        return SQLITE_ERROR;
    }

    const char *cmd = (const char *)sqlite3_value_text(argv[cmd_col]);
    if (!cmd) {
        vtab->base.zErrMsg = sqlite3_mprintf("graph_adjacency: NULL command");
        return SQLITE_ERROR;
    }

    if (strcmp(cmd, "rebuild") == 0) {
        return adj_full_rebuild(vtab);
    } else if (strcmp(cmd, "incremental_rebuild") == 0) {
        return adj_incremental_rebuild(vtab);
    } else if (strcmp(cmd, "stats") == 0) {
        /* Stats is a no-op for now; query _config directly */
        return SQLITE_OK;
    } else {
        vtab->base.zErrMsg = sqlite3_mprintf(
            "graph_adjacency: unknown command '%s' "
            "(valid: rebuild, incremental_rebuild, stats)", cmd);
        return SQLITE_ERROR;
    }
}

/* ─── xRename ──────────────────────────────────────────────── */

static int adj_xRename(sqlite3_vtab *pVTab, const char *zNew) {
    AdjVtab *vtab = (AdjVtab *)pVTab;
    const char *old_name = vtab->vtab_name;
    int rc;

    const char *suffixes[] = {"_config", "_nodes", "_degree",
                              "_csr_fwd", "_csr_rev", "_delta"};
    for (int i = 0; i < 6; i++) {
        char *sql = sqlite3_mprintf(
            "ALTER TABLE \"%w%s\" RENAME TO \"%w%s\"",
            old_name, suffixes[i], zNew, suffixes[i]);
        rc = sqlite3_exec(vtab->db, sql, NULL, NULL, NULL);
        sqlite3_free(sql);
        if (rc != SQLITE_OK)
            return rc;
    }

    /* Rename triggers */
    remove_triggers(vtab->db, old_name);
    install_triggers(vtab->db, zNew, vtab->edge_table,
                     vtab->src_col, vtab->dst_col, vtab->weight_col);

    /* Update vtab name */
    sqlite3_free(vtab->vtab_name);
    vtab->vtab_name = sqlite3_mprintf("%s", zNew);

    return SQLITE_OK;
}

/* ─── xShadowName ──────────────────────────────────────────── */

static int adj_xShadowName(const char *zName) {
    const char *suffixes[] = {"config", "nodes", "degree",
                              "csr_fwd", "csr_rev", "delta"};
    for (int i = 0; i < 6; i++) {
        if (strcmp(zName, suffixes[i]) == 0)
            return 1;
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * Public API: Detection + Loading for TVF consumers
 * ═══════════════════════════════════════════════════════════════ */

int is_graph_adjacency(sqlite3 *db, const char *name) {
    sqlite3_stmt *stmt;
    char *sql = sqlite3_mprintf(
        "SELECT value FROM \"%w_config\" WHERE key='edge_table'", name);
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK)
        return 0;
    int found = (sqlite3_step(stmt) == SQLITE_ROW);
    sqlite3_finalize(stmt);
    return found;
}

/* Helper: get a string config value (caller must sqlite3_free) */
static char *config_get_str(sqlite3 *db, const char *name, const char *key) {
    sqlite3_stmt *stmt;
    char *sql = sqlite3_mprintf(
        "SELECT value FROM \"%w_config\" WHERE key='%w'", name, key);
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return NULL;

    char *result = NULL;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *val = (const char *)sqlite3_column_text(stmt, 0);
        if (val) result = sqlite3_mprintf("%s", val);
    }
    sqlite3_finalize(stmt);
    return result;
}

/* Helper: add edge to adjacency list (local copy of graph_load.c adj_add) */
static void gadj_adj_add(GraphAdjList *adj, int target, double weight) {
    if (adj->count >= adj->capacity) {
        int nc = adj->capacity == 0 ? 8 : adj->capacity * 2;
        adj->edges = (GraphEdge *)realloc(adj->edges,
                                          (size_t)nc * sizeof(GraphEdge));
        adj->capacity = nc;
    }
    adj->edges[adj->count].target = target;
    adj->edges[adj->count].weight = weight;
    adj->count++;
}

/* Load GraphData from shadow tables (assumes data is fresh) */
static int load_graph_from_shadow(sqlite3 *db, const char *name,
                                  GraphData *g, char **pzErrMsg) {
    graph_data_init(g);
    int rc;

    /* Load node registry */
    sqlite3_stmt *stmt;
    char *sql = sqlite3_mprintf(
        "SELECT id FROM \"%w_nodes\" ORDER BY idx", name);
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) {
        if (pzErrMsg) *pzErrMsg = sqlite3_mprintf("failed to load nodes");
        return rc;
    }
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *id = (const char *)sqlite3_column_text(stmt, 0);
        if (id) graph_data_find_or_add(g, id);
    }
    sqlite3_finalize(stmt);

    if (g->node_count == 0)
        return SQLITE_OK; /* empty graph */

    /* Load forward CSR → out[] adjacency lists */
    CsrArray fwd;
    rc = load_csr_blob(db, name, "_csr_fwd", &fwd);
    if (rc != SQLITE_OK) {
        if (pzErrMsg) *pzErrMsg = sqlite3_mprintf("failed to load forward CSR");
        return rc;
    }

    for (int32_t i = 0; i < fwd.node_count && i < (int32_t)g->node_count; i++) {
        for (int32_t j = fwd.offsets[i]; j < fwd.offsets[i + 1]; j++) {
            double w = (fwd.weights && fwd.has_weights) ? fwd.weights[j] : 1.0;
            gadj_adj_add(&g->out[i], fwd.targets[j], w);
        }
    }
    g->edge_count = fwd.edge_count;
    g->has_weights = fwd.has_weights;
    csr_destroy(&fwd);

    /* Load reverse CSR → in[] adjacency lists */
    CsrArray rev;
    rc = load_csr_blob(db, name, "_csr_rev", &rev);
    if (rc != SQLITE_OK) {
        if (pzErrMsg) *pzErrMsg = sqlite3_mprintf("failed to load reverse CSR");
        return rc;
    }

    for (int32_t i = 0; i < rev.node_count && i < (int32_t)g->node_count; i++) {
        for (int32_t j = rev.offsets[i]; j < rev.offsets[i + 1]; j++) {
            double w = (rev.weights && rev.has_weights) ? rev.weights[j] : 1.0;
            gadj_adj_add(&g->in[i], rev.targets[j], w);
        }
    }
    csr_destroy(&rev);

    return SQLITE_OK;
}

int graph_data_load_from_adjacency(sqlite3 *db, const char *vtab_name,
                                   GraphData *g, char **pzErrMsg) {
    /* Check if data is stale */
    int64_t dc = delta_count(db, vtab_name);

    if (dc > 0) {
        /* Stale — fall back to loading from original edge table */
        char *edge_table = config_get_str(db, vtab_name, "edge_table");
        char *src_col = config_get_str(db, vtab_name, "src_col");
        char *dst_col = config_get_str(db, vtab_name, "dst_col");
        char *weight_col = config_get_str(db, vtab_name, "weight_col");

        if (!edge_table || !src_col || !dst_col) {
            sqlite3_free(edge_table);
            sqlite3_free(src_col);
            sqlite3_free(dst_col);
            sqlite3_free(weight_col);
            if (pzErrMsg)
                *pzErrMsg = sqlite3_mprintf("graph_adjacency '%s': missing config", vtab_name);
            return SQLITE_ERROR;
        }

        GraphLoadConfig config;
        memset(&config, 0, sizeof(config));
        config.edge_table = edge_table;
        config.src_col = src_col;
        config.dst_col = dst_col;
        config.weight_col = weight_col;
        config.direction = "both";

        graph_data_init(g);
        int rc = graph_data_load(db, &config, g, pzErrMsg);

        sqlite3_free(edge_table);
        sqlite3_free(src_col);
        sqlite3_free(dst_col);
        sqlite3_free(weight_col);
        return rc;
    }

    /* Fresh — load from shadow tables (fast BLOB read) */
    return load_graph_from_shadow(db, vtab_name, g, pzErrMsg);
}

/* ═══════════════════════════════════════════════════════════════
 * Module Registration
 * ═══════════════════════════════════════════════════════════════ */

static sqlite3_module adjacency_module = {
    .iVersion = 3,
    .xCreate = adj_xCreate,
    .xConnect = adj_xConnect,
    .xBestIndex = adj_xBestIndex,
    .xDisconnect = adj_xDisconnect,
    .xDestroy = adj_xDestroy,
    .xOpen = adj_xOpen,
    .xClose = adj_xClose,
    .xFilter = adj_xFilter,
    .xNext = adj_xNext,
    .xEof = adj_xEof,
    .xColumn = adj_xColumn,
    .xRowid = adj_xRowid,
    .xUpdate = adj_xUpdate,
    .xRename = adj_xRename,
    .xShadowName = adj_xShadowName,
};

int adjacency_register_module(sqlite3 *db) {
    return sqlite3_create_module(db, "graph_adjacency", &adjacency_module, NULL);
}
