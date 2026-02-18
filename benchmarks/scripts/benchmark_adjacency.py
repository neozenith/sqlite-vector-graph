"""
Graph adjacency benchmark: TVF-only vs CSR-cached rebuild vs incremental merge.

Compares graph algorithm performance across three approaches:
    tvf_only      — Baseline: graph_data_load() from edge table on every query
    csr_cached    — Phase 1: graph_adjacency VT builds CSR once, algorithms read from cache
    delta_merge   — Phase 2: After small edge mutations, incremental merge vs full rebuild

Workloads:
    small       — V=500, E=2000 (Erdos-Renyi)
    medium      — V=10000, E=50000 (Barabasi-Albert)
    large       — V=50000, E=250000 (Barabasi-Albert)
    incremental — V=10000, E=50000 + 100 inserts (delta merge vs full rebuild)

Measurements: wall time, shadow table disk size, trigger overhead.

Run:
    python benchmarks/scripts/benchmark_adjacency.py
    python benchmarks/scripts/benchmark_adjacency.py --profile small
    python benchmarks/scripts/benchmark_adjacency.py --profile medium
"""

import argparse
import datetime
import json
import logging
import platform
import random
import sqlite3
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MUNINN_PATH = str(PROJECT_ROOT / "build" / "muninn")
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"

ALGORITHMS = ["degree", "betweenness", "closeness", "leiden"]


def load_extension(db):
    """Load the muninn extension into a connection."""
    db.enable_load_extension(True)
    db.load_extension(MUNINN_PATH)


def generate_erdos_renyi(db, n, avg_degree, seed=42):
    """Generate an Erdos-Renyi random graph."""
    rng = random.Random(seed)
    p = avg_degree / n
    db.execute("CREATE TABLE edges (src TEXT, dst TEXT, weight REAL DEFAULT 1.0)")
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < p:
                edges.append((f"n{i}", f"n{j}", round(rng.uniform(0.1, 5.0), 2)))
    db.executemany("INSERT INTO edges VALUES (?, ?, ?)", edges)
    db.commit()
    return len(edges)


def generate_barabasi_albert(db, n, m=5, seed=42):
    """Generate a Barabasi-Albert preferential attachment graph."""
    rng = random.Random(seed)
    db.execute("CREATE TABLE edges (src TEXT, dst TEXT, weight REAL DEFAULT 1.0)")

    # Start with a fully connected core of m+1 nodes
    adj = {i: set() for i in range(m + 1)}
    edges = []
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            adj[i].add(j)
            adj[j].add(i)
            edges.append((f"n{i}", f"n{j}", round(rng.uniform(0.1, 5.0), 2)))

    # Degree list for preferential attachment
    degree_list = []
    for i in range(m + 1):
        degree_list.extend([i] * len(adj[i]))

    # Add remaining nodes
    for new_node in range(m + 1, n):
        targets = set()
        while len(targets) < m:
            t = degree_list[rng.randint(0, len(degree_list) - 1)]
            if t != new_node:
                targets.add(t)

        adj[new_node] = set()
        for t in targets:
            adj[new_node].add(t)
            adj.setdefault(t, set()).add(new_node)
            w = round(rng.uniform(0.1, 5.0), 2)
            edges.append((f"n{new_node}", f"n{t}", w))
            degree_list.extend([new_node, t])

    db.executemany("INSERT INTO edges VALUES (?, ?, ?)", edges)
    db.commit()
    return len(edges)


def time_operation(func, *args, repeats=1):
    """Time a function, returning average wall time in ms."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func(*args)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return sum(times) / len(times)


def run_tvf_query(db, algo, edge_table="edges"):
    """Run a TVF algorithm query and return row count."""
    if algo == "degree":
        sql = f"SELECT COUNT(*) FROM graph_degree('{edge_table}', 'src', 'dst', 'weight')"
    elif algo == "betweenness":
        sql = f"SELECT COUNT(*) FROM graph_betweenness('{edge_table}', 'src', 'dst', 'weight')"
    elif algo == "closeness":
        sql = f"SELECT COUNT(*) FROM graph_closeness('{edge_table}', 'src', 'dst', 'weight')"
    elif algo == "leiden":
        sql = f"SELECT COUNT(*) FROM graph_leiden('{edge_table}', 'src', 'dst', 'weight')"
    else:
        return 0
    return db.execute(sql).fetchone()[0]


def measure_disk_usage(db, vtab_name):
    """Measure shadow table disk size in bytes using dbstat."""
    try:
        rows = db.execute(
            "SELECT SUM(pgsize) FROM dbstat WHERE name LIKE ?",
            (f"{vtab_name}_%",),
        ).fetchone()
        return rows[0] or 0
    except Exception:
        return -1  # dbstat not available


def benchmark_workload(n, edge_count_target, model, profile_name, algos):
    """Run a complete benchmark workload."""
    results = []
    repeats = 3 if n <= 1000 else 1

    log.info("=== Workload: %s (V=%d, target_E=%d, model=%s) ===", profile_name, n, edge_count_target, model)

    # --- Approach 1: TVF-only (baseline) ---
    log.info("  Approach: tvf_only")
    db_tvf = sqlite3.connect(":memory:")
    load_extension(db_tvf)
    if model == "erdos_renyi":
        actual_edges = generate_erdos_renyi(db_tvf, n, edge_count_target / n)
    else:
        actual_edges = generate_barabasi_albert(db_tvf, n, m=edge_count_target // n)

    log.info("    Generated %d edges", actual_edges)

    for algo in algos:
        try:
            ms = time_operation(run_tvf_query, db_tvf, algo, "edges", repeats=repeats)
            log.info("    %s: %.2f ms", algo, ms)
            results.append({
                "approach": "tvf_only",
                "workload": profile_name,
                "operation": algo,
                "wall_time_ms": round(ms, 3),
                "edge_count": actual_edges,
                "node_count": n,
                "model": model,
            })
        except Exception as e:
            log.warning("    %s failed: %s", algo, e)

    db_tvf.close()

    # --- Approach 2: CSR-cached (graph_adjacency) ---
    log.info("  Approach: csr_cached")
    db_file = str(RESULTS_DIR / f"_bench_{profile_name}.db")
    db_csr = sqlite3.connect(db_file)
    load_extension(db_csr)
    if model == "erdos_renyi":
        actual_edges = generate_erdos_renyi(db_csr, n, edge_count_target / n)
    else:
        actual_edges = generate_barabasi_albert(db_csr, n, m=edge_count_target // n)

    # Measure CSR build time
    build_ms = time_operation(
        lambda: db_csr.execute(
            "CREATE VIRTUAL TABLE g USING graph_adjacency("
            "edge_table='edges', src_col='src', dst_col='dst', weight_col='weight')"
        ),
    )
    log.info("    CSR build: %.2f ms (%d edges)", build_ms, actual_edges)
    results.append({
        "approach": "csr_cached",
        "workload": profile_name,
        "operation": "build",
        "wall_time_ms": round(build_ms, 3),
        "edge_count": actual_edges,
        "node_count": n,
        "model": model,
    })

    # Measure disk usage
    disk_bytes = measure_disk_usage(db_csr, "g")
    if disk_bytes >= 0:
        log.info("    Shadow table disk: %d bytes (%.1f KB)", disk_bytes, disk_bytes / 1024)
        results.append({
            "approach": "csr_cached",
            "workload": profile_name,
            "operation": "disk_usage",
            "disk_bytes": disk_bytes,
            "edge_count": actual_edges,
            "node_count": n,
            "model": model,
        })

    # Run algorithms using the adjacency VT
    for algo in algos:
        try:
            ms = time_operation(run_tvf_query, db_csr, algo, "g", repeats=repeats)
            log.info("    %s: %.2f ms", algo, ms)
            results.append({
                "approach": "csr_cached",
                "workload": profile_name,
                "operation": algo,
                "wall_time_ms": round(ms, 3),
                "edge_count": actual_edges,
                "node_count": n,
                "model": model,
            })
        except Exception as e:
            log.warning("    %s failed: %s", algo, e)

    # --- Approach 3: Incremental merge ---
    log.info("  Approach: delta_merge")

    # Insert some edges and measure merge time
    n_delta = max(10, actual_edges // 100)  # ~1% of edges
    rng = random.Random(99)
    delta_edges = [
        (f"n{rng.randint(0, n-1)}", f"n{rng.randint(0, n-1)}", round(rng.uniform(0.1, 5.0), 2))
        for _ in range(n_delta)
    ]
    db_csr.executemany("INSERT INTO edges VALUES (?, ?, ?)", delta_edges)
    db_csr.commit()

    # Measure incremental rebuild time
    incr_ms = time_operation(
        lambda: db_csr.execute("INSERT INTO g(g) VALUES ('incremental_rebuild')"),
    )
    log.info("    Incremental rebuild (%d deltas): %.2f ms", n_delta, incr_ms)
    results.append({
        "approach": "delta_merge",
        "workload": profile_name,
        "operation": "incremental_rebuild",
        "wall_time_ms": round(incr_ms, 3),
        "edge_count": actual_edges + n_delta,
        "delta_count": n_delta,
        "node_count": n,
        "model": model,
    })

    # Measure full rebuild for comparison
    db_csr.executemany("INSERT INTO edges VALUES (?, ?, ?)", delta_edges)
    db_csr.commit()
    full_ms = time_operation(
        lambda: db_csr.execute("INSERT INTO g(g) VALUES ('rebuild')"),
    )
    log.info("    Full rebuild (same delta): %.2f ms", full_ms)
    results.append({
        "approach": "delta_merge",
        "workload": profile_name,
        "operation": "full_rebuild",
        "wall_time_ms": round(full_ms, 3),
        "edge_count": actual_edges + 2 * n_delta,
        "delta_count": n_delta,
        "node_count": n,
        "model": model,
    })

    # Run algorithms after merge
    for algo in algos:
        try:
            ms = time_operation(run_tvf_query, db_csr, algo, "g", repeats=repeats)
            log.info("    %s (post-merge): %.2f ms", algo, ms)
            results.append({
                "approach": "delta_merge",
                "workload": profile_name,
                "operation": f"{algo}_post_merge",
                "wall_time_ms": round(ms, 3),
                "edge_count": actual_edges + 2 * n_delta,
                "delta_count": n_delta,
                "node_count": n,
                "model": model,
            })
        except Exception as e:
            log.warning("    %s post-merge failed: %s", algo, e)

    # Measure trigger overhead: time to insert N edges with vs without triggers
    log.info("  Measuring trigger overhead...")
    trigger_edges = [
        (f"t{rng.randint(0, n-1)}", f"t{rng.randint(0, n-1)}", 1.0)
        for _ in range(min(1000, actual_edges // 10))
    ]

    # With triggers (already installed)
    with_ms = time_operation(
        lambda: (
            db_csr.executemany("INSERT INTO edges VALUES (?, ?, ?)", trigger_edges),
            db_csr.execute("DELETE FROM edges WHERE src LIKE 't%'"),
            db_csr.commit(),
        ),
    )

    # Without triggers (drop them, re-create after)
    db_csr.execute("DROP TRIGGER IF EXISTS g_ai")
    db_csr.execute("DROP TRIGGER IF EXISTS g_ad")
    db_csr.execute("DROP TRIGGER IF EXISTS g_au")

    without_ms = time_operation(
        lambda: (
            db_csr.executemany("INSERT INTO edges VALUES (?, ?, ?)", trigger_edges),
            db_csr.execute("DELETE FROM edges WHERE src LIKE 't%'"),
            db_csr.commit(),
        ),
    )

    trigger_overhead = max(0, with_ms - without_ms)
    log.info(
        "    Trigger overhead: %.2f ms (with=%.2f, without=%.2f, %d ops)",
        trigger_overhead, with_ms, without_ms, len(trigger_edges),
    )
    results.append({
        "approach": "trigger_overhead",
        "workload": profile_name,
        "operation": "insert_batch",
        "wall_time_ms": round(trigger_overhead, 3),
        "with_trigger_ms": round(with_ms, 3),
        "without_trigger_ms": round(without_ms, 3),
        "batch_size": len(trigger_edges),
        "node_count": n,
        "model": model,
    })

    db_csr.close()
    # Clean up temp db
    Path(db_file).unlink(missing_ok=True)

    return results


# --- Profiles ---
PROFILES = {
    "small": {
        "workloads": [
            {"n": 500, "edges": 2000, "model": "erdos_renyi"},
        ],
        "algos": ["degree", "betweenness", "closeness", "leiden"],
    },
    "medium": {
        "workloads": [
            {"n": 5000, "edges": 25000, "model": "barabasi_albert"},
        ],
        "algos": ["degree", "betweenness", "leiden"],
    },
    "large": {
        "workloads": [
            {"n": 10000, "edges": 50000, "model": "barabasi_albert"},
        ],
        "algos": ["degree", "leiden"],
    },
}


def main():
    parser = argparse.ArgumentParser(description="Graph adjacency CSR benchmark")
    parser.add_argument("--profile", choices=list(PROFILES.keys()) + ["all"], default="small")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    profiles_to_run = list(PROFILES.keys()) if args.profile == "all" else [args.profile]

    all_results = []
    metadata = {
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "platform": platform.machine(),
        "python": sys.version.split()[0],
    }

    for profile_name in profiles_to_run:
        profile = PROFILES[profile_name]
        for wl in profile["workloads"]:
            results = benchmark_workload(
                n=wl["n"],
                edge_count_target=wl["edges"],
                model=wl["model"],
                profile_name=profile_name,
                algos=profile["algos"],
            )
            for r in results:
                r.update(metadata)
            all_results.extend(results)

    # Write results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = args.output or str(RESULTS_DIR / "adjacency_benchmark.jsonl")
    with open(out_file, "a") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    log.info("\n=== Results written to %s (%d entries) ===", out_file, len(all_results))

    # Print summary table
    log.info("\n%-15s %-12s %-20s %10s", "Approach", "Workload", "Operation", "Time (ms)")
    log.info("-" * 60)
    for r in all_results:
        if "wall_time_ms" in r:
            log.info(
                "%-15s %-12s %-20s %10.2f",
                r["approach"], r["workload"], r["operation"], r["wall_time_ms"],
            )


if __name__ == "__main__":
    main()
