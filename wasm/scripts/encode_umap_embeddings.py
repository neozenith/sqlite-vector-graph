#!/usr/bin/env python3
"""
Pre-calculate UMAP 2D and 3D coordinates for all HNSW vector tables in a muninn database.

Creates a `{table}_umap` table for each vector node table containing:
  - id: matches the vector node row id
  - x2d, y2d: 2D UMAP coordinates (for flat scatter plots)
  - x3d, y3d, z3d: 3D UMAP coordinates (for Deck.GL point clouds)

Usage:
    python wasm/scripts/encode_umap_embeddings.py [--db wasm/assets/3300.db]

The script auto-discovers all HNSW _nodes tables by scanning sqlite_master for
tables matching the pattern `*_nodes` that have (id, vector, level, deleted) columns.
"""

import argparse
import logging
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np
from umap import UMAP

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DEFAULT_DB = Path(__file__).resolve().parent.parent / "assets" / "3300.db"

# UMAP needs at least n_neighbors+1 samples; use sensible defaults
UMAP_PARAMS = {"random_state": 42, "n_neighbors": 15, "min_dist": 0.1}
MIN_SAMPLES = 20  # Skip tables with fewer vectors than this


def discover_vector_tables(conn: sqlite3.Connection) -> list[dict]:
    """Find all HNSW shadow _nodes tables and their dimensions."""
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_nodes'")
    tables = []
    for (name,) in cursor.fetchall():
        # Check it has the expected HNSW shadow table schema
        cols = {row[1] for row in conn.execute(f'PRAGMA table_info("{name}")').fetchall()}
        if not {"id", "vector", "level", "deleted"}.issubset(cols):
            continue

        # Read one vector to determine dimension
        row = conn.execute(f'SELECT vector FROM "{name}" WHERE deleted = 0 LIMIT 1').fetchone()
        if row is None:
            continue
        dim = len(row[0]) // 4  # float32 = 4 bytes each

        count = conn.execute(f'SELECT COUNT(*) FROM "{name}" WHERE deleted = 0').fetchone()[0]

        tables.append({"name": name, "dim": dim, "count": count})
    return tables


def load_vectors(conn: sqlite3.Connection, table: str, dim: int) -> tuple[list[int], np.ndarray]:
    """Load all non-deleted vectors from a shadow table."""
    rows = conn.execute(f'SELECT id, vector FROM "{table}" WHERE deleted = 0 ORDER BY id').fetchall()

    ids = []
    vectors = []
    for row_id, blob in rows:
        floats = struct.unpack(f"<{dim}f", blob)
        ids.append(row_id)
        vectors.append(floats)

    return ids, np.array(vectors, dtype=np.float32)


def compute_umap(vectors: np.ndarray, n_components: int) -> np.ndarray:
    """Run UMAP dimensionality reduction."""
    n_neighbors = min(UMAP_PARAMS["n_neighbors"], len(vectors) - 1)
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=UMAP_PARAMS["min_dist"],
        random_state=UMAP_PARAMS["random_state"],
    )
    return reducer.fit_transform(vectors)


def create_umap_table(
    conn: sqlite3.Connection, base_table: str, ids: list[int], coords_2d: np.ndarray, coords_3d: np.ndarray
):
    """Create or replace the _umap table with pre-calculated coordinates."""
    umap_table = base_table.replace("_nodes", "_umap")

    conn.execute(f'DROP TABLE IF EXISTS "{umap_table}"')
    conn.execute(f"""
        CREATE TABLE "{umap_table}" (
            id INTEGER PRIMARY KEY,
            x2d REAL NOT NULL,
            y2d REAL NOT NULL,
            x3d REAL NOT NULL,
            y3d REAL NOT NULL,
            z3d REAL NOT NULL
        )
    """)

    data = [
        (
            ids[i],
            float(coords_2d[i, 0]),
            float(coords_2d[i, 1]),
            float(coords_3d[i, 0]),
            float(coords_3d[i, 1]),
            float(coords_3d[i, 2]),
        )
        for i in range(len(ids))
    ]
    conn.executemany(
        f'INSERT INTO "{umap_table}" (id, x2d, y2d, x3d, y3d, z3d) VALUES (?, ?, ?, ?, ?, ?)',
        data,
    )
    conn.commit()
    log.info("  Created %s with %d rows", umap_table, len(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"Path to SQLite database (default: {DEFAULT_DB})")
    parser.add_argument("--scale", type=float, default=30.0, help="Scale factor for UMAP coordinates (default: 30)")
    args = parser.parse_args()

    if not args.db.exists():
        log.error("Database not found: %s", args.db)
        sys.exit(1)

    conn = sqlite3.connect(str(args.db))
    tables = discover_vector_tables(conn)

    if not tables:
        log.error("No HNSW vector tables found in %s", args.db)
        sys.exit(1)

    log.info("Found %d vector tables in %s:", len(tables), args.db.name)
    for t in tables:
        log.info("  %s: %d vectors, %d-dim", t["name"], t["count"], t["dim"])

    for t in tables:
        name, count, dim = t["name"], t["count"], t["dim"]

        if count < MIN_SAMPLES:
            log.warning("  Skipping %s: only %d vectors (need >= %d)", name, count, MIN_SAMPLES)
            continue

        log.info("Processing %s (%d vectors, %d-dim)...", name, count, dim)

        ids, vectors = load_vectors(conn, name, dim)

        log.info("  Computing 2D UMAP...")
        coords_2d = compute_umap(vectors, n_components=2) * args.scale

        log.info("  Computing 3D UMAP...")
        coords_3d = compute_umap(vectors, n_components=3) * args.scale

        create_umap_table(conn, name, ids, coords_2d, coords_3d)

    conn.close()
    log.info("Done. Database updated: %s", args.db)


if __name__ == "__main__":
    main()
