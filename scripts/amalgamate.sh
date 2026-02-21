#!/bin/bash
# Generate the muninn amalgamation: a single muninn.c + muninn.h
# that can be compiled standalone without the full source tree.
#
# Note: The amalgamation includes embed_gguf.c which #include "llama.h".
# To build the amalgamation, users must provide the llama.cpp headers
# and link against the llama.cpp static libraries. See dist/Makefile
# for a convenience build script.
set -euo pipefail

VERSION=$(cat VERSION 2>/dev/null || echo "0.0.0")
OUTDIR=dist
OUT="${OUTDIR}/muninn.c"

mkdir -p "$OUTDIR"

# ── Header ordering matters: dependencies before dependents ──
INTERNAL_HEADERS=(
    src/vec_math.h
    src/priority_queue.h
    src/hnsw_algo.h
    src/id_validate.h
    src/hnsw_vtab.h
    src/graph_common.h
    src/graph_tvf.h
    src/graph_load.h
    src/graph_centrality.h
    src/graph_community.h
    src/graph_adjacency.h
    src/graph_csr.h
    src/graph_selector_parse.h
    src/graph_selector_eval.h
    src/graph_select_tvf.h
    src/node2vec.h
    src/embed_gguf.h
    src/muninn.h
)

# ── Source ordering: dependencies before dependents ──
SOURCES=(
    src/vec_math.c
    src/priority_queue.c
    src/hnsw_algo.c
    src/id_validate.c
    src/hnsw_vtab.c
    src/graph_tvf.c
    src/graph_load.c
    src/graph_centrality.c
    src/graph_community.c
    src/graph_adjacency.c
    src/graph_csr.c
    src/graph_selector_parse.c
    src/graph_selector_eval.c
    src/graph_select_tvf.c
    src/node2vec.c
    src/embed_gguf.c
    src/muninn.c
)

# ── Write file header ──
cat > "$OUT" <<HEADER
/*
 * muninn amalgamation — v${VERSION}
 * Generated $(date -u +%Y-%m-%d)
 *
 * HNSW vector search + graph traversal + Node2Vec + GGUF embeddings for SQLite.
 * https://github.com/user/sqlite-muninn
 *
 * IMPORTANT: This amalgamation requires llama.cpp for GGUF embedding support.
 * You must provide the llama.cpp headers and link against its static libraries.
 *
 * Build as loadable extension (requires llama.cpp pre-built):
 *   # Linux
 *   cmake -B vendor/llama.cpp/build -S vendor/llama.cpp -DBUILD_SHARED_LIBS=OFF ...
 *   cmake --build vendor/llama.cpp/build
 *   gcc -O2 -fPIC -shared muninn.c -o muninn.so \\
 *       -Ivendor/llama.cpp/include -Ivendor/llama.cpp/ggml/include \\
 *       vendor/llama.cpp/build/src/libllama.a \\
 *       vendor/llama.cpp/build/ggml/src/libggml-base.a \\
 *       vendor/llama.cpp/build/ggml/src/ggml-cpu/libggml-cpu.a \\
 *       -lstdc++ -lm -lpthread
 *
 *   # macOS
 *   cc -O2 -fPIC -dynamiclib muninn.c -o muninn.dylib \\
 *       -Ivendor/llama.cpp/include -Ivendor/llama.cpp/ggml/include \\
 *       vendor/llama.cpp/build/src/libllama.a \\
 *       vendor/llama.cpp/build/ggml/src/libggml-base.a \\
 *       vendor/llama.cpp/build/ggml/src/ggml-cpu/libggml-cpu.a \\
 *       -lc++ -framework Accelerate -lm
 */

/* Enable POSIX functions (strdup) on strict C11 compilers */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

/* SQLite extension API — required for all builds */
#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT1

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <ctype.h>

HEADER

# ── Inline all internal headers (strip #include "..." lines) ──
for f in "${INTERNAL_HEADERS[@]}"; do
    if [ -f "$f" ]; then
        echo "/* ──── $f ──── */" >> "$OUT"
        # Remove #include "..." (internal headers) and header guards we'll handle
        grep -v '#include "' "$f" >> "$OUT"
        echo "" >> "$OUT"
    fi
done

# ── Inline all source files (strip #include "..." lines) ──
for f in "${SOURCES[@]}"; do
    if [ -f "$f" ]; then
        echo "/* ──── $f ──── */" >> "$OUT"
        # Remove internal #include "..." and SQLITE_EXTENSION_INIT1 (already at top)
        # Keep #include <llama.h> since it's an external dependency
        grep -v '#include "' "$f" | grep -v 'SQLITE_EXTENSION_INIT1' >> "$OUT"
        echo "" >> "$OUT"
    fi
done

# ── Copy public header ──
cp src/muninn.h "${OUTDIR}/muninn.h"

LINES=$(wc -l < "$OUT")
echo "Amalgamation: ${OUT} (${LINES} lines)"
echo "Header:       ${OUTDIR}/muninn.h"
