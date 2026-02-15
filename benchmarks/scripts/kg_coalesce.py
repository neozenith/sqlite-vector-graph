"""
Knowledge Graph Coalescing + GraphRAG Demo — Entity resolution and query pipeline.

Three-stage entity coalescing:
1. HNSW Blocking — KNN search on entity embeddings to find candidate pairs [subsystem #1]
2. Matching Cascade — Score pairs via exact match, substring, Jaro-Winkler, cosine
3. Leiden Clustering — Cluster match graph to resolve synonym groups [subsystem #4]

Post-coalescing:
- Build clean graph (nodes + edges with canonical entity names)
- Train Node2Vec structural embeddings [subsystem #5]
- GraphRAG query demo exercising all 5 subsystems

Output: Modified benchmarks/kg/{book_id}.db with added tables + JSONL results.
"""

import argparse
import collections
import datetime
import json
import logging
import re
import sqlite3
import struct
import sys
import time
from pathlib import Path

import numpy as np

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MUNINN_PATH = str(PROJECT_ROOT / "muninn")
KG_DIR = PROJECT_ROOT / "benchmarks" / "kg"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Node2Vec parameters
N2V_DIM = 64
N2V_P = 0.5
N2V_Q = 0.5
N2V_WALKS = 10
N2V_WALK_LENGTH = 40
N2V_WINDOW = 5
N2V_NEGATIVE = 5
N2V_LR = 0.025
N2V_EPOCHS = 5

# HNSW parameters
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64


# ── Utilities ─────────────────────────────────────────────────────────


def pack_vector(v):
    """Pack a float list/array into a float32 BLOB for SQLite."""
    if isinstance(v, np.ndarray):
        return v.astype(np.float32).tobytes()
    return struct.pack(f"{len(v)}f", *v)


def normalize_name(name):
    """Normalize entity name: lowercase, collapse whitespace, strip articles."""
    name = name.lower().strip()
    name = re.sub(r"\s+", " ", name)
    for article in ("the ", "a ", "an "):
        if name.startswith(article):
            name = name[len(article):]
    return name


# ── Jaro-Winkler similarity (pure Python) ────────────────────────────


def _jaro_similarity(s1, s2):
    """Compute Jaro similarity between two strings."""
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3
    return jaro


def jaro_winkler_similarity(s1, s2, prefix_weight=0.1):
    """Compute Jaro-Winkler similarity between two strings.

    Gives higher scores to strings that match from the beginning,
    which is ideal for entity names like "Adam Smith" vs "Smith, Adam".
    """
    jaro = _jaro_similarity(s1, s2)

    # Find common prefix (up to 4 chars)
    prefix_len = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break

    return jaro + prefix_len * prefix_weight * (1 - jaro)


# ── Stage 1: HNSW Blocking ───────────────────────────────────────────


def hnsw_blocking(conn, threshold=0.4):
    """Find candidate entity pairs using HNSW KNN on entity embeddings.

    For each entity, search for K nearest neighbors in entities_vec.
    Returns list of (name_a, name_b, cosine_distance) tuples where
    distance < threshold.

    This is O(N * K * log N) vs O(N^2) pairwise — critical for scaling.
    """
    log.info("Stage 1: HNSW blocking (threshold=%.2f)...", threshold)

    # Load entity name -> rowid mapping
    name_map = {}
    rowid_map = {}
    rows = conn.execute("SELECT rowid, name FROM entity_vec_map").fetchall()
    for rowid, name in rows:
        name_map[name] = rowid
        rowid_map[rowid] = name

    if not name_map:
        log.warning("No entity embeddings found — skipping blocking")
        return []

    # For each entity, KNN search to find candidates
    candidates = []
    k = min(20, len(name_map))  # Search for up to 20 neighbors

    for name, rowid in name_map.items():
        vec_row = conn.execute("SELECT vector FROM entities_vec WHERE rowid = ?", (rowid,)).fetchone()
        if vec_row is None:
            continue

        results = conn.execute(
            "SELECT rowid, distance FROM entities_vec WHERE vector MATCH ? AND k = ? AND ef_search = ?",
            (vec_row[0], k, HNSW_EF_SEARCH),
        ).fetchall()

        for match_rowid, distance in results:
            if match_rowid == rowid:
                continue  # Skip self
            if distance > threshold:
                continue
            other_name = rowid_map.get(match_rowid)
            if other_name is None:
                continue
            # Canonical order to avoid duplicates
            pair = tuple(sorted([name, other_name]))
            candidates.append((*pair, distance))

    # Deduplicate
    seen = set()
    unique = []
    for a, b, dist in candidates:
        key = (a, b)
        if key not in seen:
            seen.add(key)
            unique.append((a, b, dist))

    log.info("HNSW blocking: %d candidate pairs (from %d entities)", len(unique), len(name_map))
    return unique


# ── Stage 2: Matching Cascade ─────────────────────────────────────────


def score_pair(name_a, name_b, cosine_distance):
    """Score an entity pair through the matching cascade.

    Returns a match score in [0, 1] or 0.0 for no match.
    """
    norm_a = normalize_name(name_a)
    norm_b = normalize_name(name_b)

    # 1. Exact normalized match
    if norm_a == norm_b:
        return 1.0

    # 2. Substring containment
    if norm_a in norm_b or norm_b in norm_a:
        return 0.9

    # 3. Jaro-Winkler similarity
    jw = jaro_winkler_similarity(norm_a, norm_b)
    if jw > 0.85:
        return jw

    # 4. High cosine similarity
    if cosine_distance < 0.15:
        return 1.0 - cosine_distance

    # 5. No match
    return 0.0


def matching_cascade(candidates):
    """Score all candidate pairs through the matching cascade.

    Returns list of (name_a, name_b, score) where score > 0.5.
    """
    log.info("Stage 2: Matching cascade on %d candidates...", len(candidates))

    matches = []
    for name_a, name_b, cosine_dist in candidates:
        score = score_pair(name_a, name_b, cosine_dist)
        if score > 0.5:
            matches.append((name_a, name_b, score))

    log.info("Matching cascade: %d matches above threshold", len(matches))
    return matches


# ── Stage 3: Leiden Clustering ────────────────────────────────────────


def leiden_clustering(conn, matches):
    """Cluster matched entities using graph_leiden.

    Insert match edges into a temp table, run Leiden community detection,
    then select canonical form per cluster (highest mention count, shortest name).
    """
    log.info("Stage 3: Leiden clustering on %d match edges...", len(matches))

    if not matches:
        log.info("No matches to cluster")
        return {}

    # Create match edges table
    conn.execute("DROP TABLE IF EXISTS _match_edges")
    conn.execute("CREATE TABLE _match_edges (src TEXT, dst TEXT, weight REAL)")
    conn.executemany(
        "INSERT INTO _match_edges (src, dst, weight) VALUES (?, ?, ?)",
        matches,
    )
    conn.commit()

    # Run Leiden community detection
    clusters_raw = conn.execute("""
        SELECT node, community_id FROM graph_leiden
        WHERE edge_table = '_match_edges'
          AND src_col = 'src'
          AND dst_col = 'dst'
          AND weight_col = 'weight'
    """).fetchall()

    # Group by community
    communities = collections.defaultdict(list)
    for node, community_id in clusters_raw:
        communities[community_id].append(node)

    log.info("Leiden found %d communities from %d nodes", len(communities), len(clusters_raw))

    # Get mention counts for canonical selection
    mention_counts = {}
    for row in conn.execute("SELECT name, count(*) FROM entities GROUP BY name").fetchall():
        mention_counts[row[0]] = row[1]

    # Select canonical form per cluster: highest mention count, tie-break shortest name
    entity_to_canonical = {}
    for community_id, members in communities.items():
        # Sort by (-mention_count, len(name)) to pick best canonical form
        members_scored = [(m, mention_counts.get(m, 0)) for m in members]
        members_scored.sort(key=lambda x: (-x[1], len(x[0])))
        canonical = members_scored[0][0]

        for member in members:
            entity_to_canonical[member] = canonical

    # Singletons: entities not in any match -> they are their own canonical
    all_entity_names = [r[0] for r in conn.execute("SELECT DISTINCT name FROM entities").fetchall()]
    for name in all_entity_names:
        if name not in entity_to_canonical:
            entity_to_canonical[name] = name

    # Clean up temp table
    conn.execute("DROP TABLE IF EXISTS _match_edges")
    conn.commit()

    n_merged = sum(1 for k, v in entity_to_canonical.items() if k != v)
    log.info("Entity resolution: %d entities merged into %d canonical forms",
             n_merged, len(set(entity_to_canonical.values())))

    return entity_to_canonical


# ── Post-coalescing: clean graph ──────────────────────────────────────


def save_clusters(conn, entity_to_canonical):
    """Save the entity clustering results to entity_clusters table."""
    conn.execute("DROP TABLE IF EXISTS entity_clusters")
    conn.execute("""
        CREATE TABLE entity_clusters (
            name TEXT PRIMARY KEY,
            canonical TEXT NOT NULL
        )
    """)
    conn.executemany(
        "INSERT INTO entity_clusters (name, canonical) VALUES (?, ?)",
        entity_to_canonical.items(),
    )
    conn.commit()
    log.info("Saved %d entity cluster mappings", len(entity_to_canonical))


def build_clean_graph(conn):
    """Build clean nodes + edges tables using canonical entity names.

    Aggregates relations through canonical name mapping, deduplicates,
    and sums weights for repeated edges.
    """
    log.info("Building clean graph from canonical entities...")

    conn.execute("DROP TABLE IF EXISTS nodes")
    conn.execute("DROP TABLE IF EXISTS edges")

    conn.execute("""
        CREATE TABLE nodes (
            node_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            entity_type TEXT,
            mention_count INTEGER DEFAULT 0
        )
    """)

    conn.execute("""
        CREATE TABLE edges (
            src TEXT NOT NULL,
            dst TEXT NOT NULL,
            rel_type TEXT,
            weight REAL DEFAULT 1.0,
            PRIMARY KEY (src, dst, rel_type)
        )
    """)

    # Insert canonical nodes with aggregated mention counts
    conn.execute("""
        INSERT INTO nodes (name, entity_type, mention_count)
        SELECT
            ec.canonical,
            (SELECT e2.entity_type FROM entities e2
             JOIN entity_clusters ec2 ON e2.name = ec2.name
             WHERE ec2.canonical = ec.canonical AND e2.entity_type IS NOT NULL
             LIMIT 1),
            count(*)
        FROM entities e
        JOIN entity_clusters ec ON e.name = ec.name
        GROUP BY ec.canonical
    """)

    # Insert edges mapped through canonical names, aggregating weights
    conn.execute("""
        INSERT OR REPLACE INTO edges (src, dst, rel_type, weight)
        SELECT
            COALESCE(ec_src.canonical, r.src),
            COALESCE(ec_dst.canonical, r.dst),
            r.rel_type,
            SUM(r.weight)
        FROM relations r
        LEFT JOIN entity_clusters ec_src ON r.src = ec_src.name
        LEFT JOIN entity_clusters ec_dst ON r.dst = ec_dst.name
        WHERE COALESCE(ec_src.canonical, r.src) != COALESCE(ec_dst.canonical, r.dst)
        GROUP BY COALESCE(ec_src.canonical, r.src), COALESCE(ec_dst.canonical, r.dst), r.rel_type
    """)

    conn.commit()

    node_count = conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
    edge_count = conn.execute("SELECT count(*) FROM edges").fetchone()[0]
    log.info("Clean graph: %d nodes, %d edges", node_count, edge_count)
    return node_count, edge_count


# ── Node2Vec structural embeddings ────────────────────────────────────


def build_rowid_mapping(conn, edge_table="edges", src_col="src", dst_col="dst"):
    """Replicate node2vec_train's first-seen ordering to map names to rowids.

    node2vec.c iterates SELECT src, dst FROM edge_table and assigns indices
    in first-seen order. Embeddings are inserted with rowid = i + 1 (1-indexed).
    """
    rows = conn.execute(f"SELECT {src_col}, {dst_col} FROM {edge_table}").fetchall()

    name_to_idx = {}
    for src, dst in rows:
        if src not in name_to_idx:
            name_to_idx[src] = len(name_to_idx)
        if dst not in name_to_idx:
            name_to_idx[dst] = len(name_to_idx)

    return {name: idx + 1 for name, idx in name_to_idx.items()}


def run_node2vec(conn):
    """Train Node2Vec on the clean edges table, output to node2vec_emb HNSW table.

    Uses subsystem #5. The resulting embeddings encode graph structure,
    so entities with similar neighborhoods get similar vectors.
    """
    edge_count = conn.execute("SELECT count(*) FROM edges").fetchone()[0]
    if edge_count == 0:
        log.warning("No edges in clean graph — skipping Node2Vec")
        return {}

    log.info("Training Node2Vec on %d edges (dim=%d, p=%.1f, q=%.1f)...", edge_count, N2V_DIM, N2V_P, N2V_Q)

    # Create HNSW table for Node2Vec embeddings
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS node2vec_emb USING hnsw_index(
            dimensions={N2V_DIM}, metric='cosine', m={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION}
        )
    """)

    t0 = time.time()
    result = conn.execute(f"""
        SELECT node2vec_train(
            'edges', 'src', 'dst', 'node2vec_emb',
            {N2V_DIM},         -- dimensions
            {N2V_P},           -- p
            {N2V_Q},           -- q
            {N2V_WALKS},       -- num_walks
            {N2V_WALK_LENGTH}, -- walk_length
            {N2V_WINDOW},      -- window_size
            {N2V_NEGATIVE},    -- negative_samples
            {N2V_LR},          -- learning_rate
            {N2V_EPOCHS}       -- epochs
        )
    """).fetchone()

    num_embedded = result[0]
    elapsed = time.time() - t0
    log.info("Node2Vec: embedded %d nodes (%.1fs)", num_embedded, elapsed)

    # Build rowid mapping for later lookup
    rowid_map = build_rowid_mapping(conn)
    log.info("Node2Vec rowid mapping: %d entries", len(rowid_map))

    return rowid_map


# ── GraphRAG Query ────────────────────────────────────────────────────


def graphrag_query(conn, query_text, embedding_model=DEFAULT_EMBEDDING_MODEL, n2v_rowid_map=None):
    """Execute a GraphRAG query exercising all five muninn subsystems.

    1. [HNSW] Query -> chunks_vec MATCH -> top-K passages
    2. [Lookup] Passages -> entity names via entity_clusters.canonical
    3. [Graph TVFs] BFS 2-hop from seed entities -> expanded set
    4. [Centrality] betweenness, closeness, degree on edges -> rank bridges
    5. [Leiden] graph_leiden on edges -> find communities containing seeds
    6. [Node2Vec] Seed entity structural embedding -> KNN on node2vec_emb
    7. [Assembly] Merge + rank + collect passages
    """
    log.info("=" * 60)
    log.info("GraphRAG Query: %r", query_text)
    log.info("=" * 60)

    result = {
        "query": query_text,
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "stages": {},
    }

    # ── [1] HNSW vector search on chunk embeddings ────────────────
    if not HAS_SENTENCE_TRANSFORMERS:
        log.error("sentence-transformers required for GraphRAG queries")
        return result

    model = SentenceTransformer(embedding_model)
    query_vec = model.encode([query_text], normalize_embeddings=True)[0]
    query_blob = pack_vector(query_vec)

    top_k = 10
    chunk_results = conn.execute(
        "SELECT rowid, distance FROM chunks_vec WHERE vector MATCH ? AND k = ? AND ef_search = ?",
        (query_blob, top_k, HNSW_EF_SEARCH),
    ).fetchall()

    seed_chunks = []
    for rowid, distance in chunk_results:
        text_row = conn.execute("SELECT text FROM chunks WHERE chunk_id = ?", (rowid,)).fetchone()
        seed_chunks.append({
            "chunk_id": rowid,
            "distance": round(distance, 4),
            "text_preview": text_row[0][:200] if text_row else "",
        })

    result["stages"]["1_hnsw_chunks"] = {
        "count": len(seed_chunks),
        "chunks": seed_chunks[:5],  # Top 5 for output
    }
    log.info("[1] HNSW chunk search: %d results", len(seed_chunks))

    # ── [2] Lookup seed entities from matching chunks ─────────────
    seed_chunk_ids = [c["chunk_id"] for c in seed_chunks]
    seed_entities = set()

    if seed_chunk_ids:
        placeholders = ",".join("?" * len(seed_chunk_ids))
        entity_rows = conn.execute(f"""
            SELECT DISTINCT ec.canonical
            FROM entities e
            JOIN entity_clusters ec ON e.name = ec.name
            WHERE e.chunk_id IN ({placeholders})
        """, seed_chunk_ids).fetchall()
        seed_entities = {r[0] for r in entity_rows}

    result["stages"]["2_seed_entities"] = {
        "count": len(seed_entities),
        "entities": sorted(seed_entities)[:20],
    }
    log.info("[2] Seed entities from chunks: %d", len(seed_entities))

    # ── [3] Graph TVF: BFS 2-hop expansion ────────────────────────
    expanded_entities = set(seed_entities)

    for seed in list(seed_entities)[:10]:  # Limit to top 10 seeds for performance
        try:
            bfs_rows = conn.execute("""
                SELECT node, depth FROM graph_bfs
                WHERE edge_table = 'edges'
                  AND src_col = 'src'
                  AND dst_col = 'dst'
                  AND start_node = ?
                  AND max_depth = 2
            """, (seed,)).fetchall()
            for node, depth in bfs_rows:
                expanded_entities.add(node)
        except sqlite3.OperationalError as e:
            log.debug("BFS from %r failed: %s", seed, e)

    newly_discovered = expanded_entities - seed_entities
    result["stages"]["3_bfs_expansion"] = {
        "seed_count": len(seed_entities),
        "expanded_count": len(expanded_entities),
        "newly_discovered": sorted(newly_discovered)[:20],
    }
    log.info("[3] BFS 2-hop: %d -> %d entities (+%d)", len(seed_entities), len(expanded_entities), len(newly_discovered))

    # ── [4] Centrality ranking on edges ───────────────────────────
    centrality_scores = {}

    try:
        # Betweenness centrality
        betweenness = conn.execute("""
            SELECT node, centrality FROM graph_betweenness
            WHERE edge_table = 'edges'
              AND src_col = 'src'
              AND dst_col = 'dst'
              AND direction = 'both'
        """).fetchall()
        for node, score in betweenness:
            if node in expanded_entities:
                centrality_scores.setdefault(node, {})["betweenness"] = round(score, 6)
    except sqlite3.OperationalError as e:
        log.warning("Betweenness centrality failed: %s", e)

    try:
        # Degree centrality
        degree = conn.execute("""
            SELECT node, centrality FROM graph_degree
            WHERE edge_table = 'edges'
              AND src_col = 'src'
              AND dst_col = 'dst'
        """).fetchall()
        for node, score in degree:
            if node in expanded_entities:
                centrality_scores.setdefault(node, {})["degree"] = round(score, 6)
    except sqlite3.OperationalError as e:
        log.warning("Degree centrality failed: %s", e)

    try:
        # Closeness centrality
        closeness = conn.execute("""
            SELECT node, centrality FROM graph_closeness
            WHERE edge_table = 'edges'
              AND src_col = 'src'
              AND dst_col = 'dst'
              AND direction = 'both'
        """).fetchall()
        for node, score in closeness:
            if node in expanded_entities:
                centrality_scores.setdefault(node, {})["closeness"] = round(score, 6)
    except sqlite3.OperationalError as e:
        log.warning("Closeness centrality failed: %s", e)

    # Rank by combined centrality
    ranked = []
    for node, scores in centrality_scores.items():
        combined = sum(scores.values()) / max(len(scores), 1)
        ranked.append({"node": node, "combined_score": round(combined, 6), **scores})
    ranked.sort(key=lambda x: -x["combined_score"])

    result["stages"]["4_centrality"] = {
        "count": len(ranked),
        "top_bridges": ranked[:10],
    }
    log.info("[4] Centrality: ranked %d entities, top bridge: %s",
             len(ranked), ranked[0]["node"] if ranked else "none")

    # ── [5] Leiden communities containing seed entities ────────────
    community_entities = set()
    seed_communities = set()

    try:
        communities = conn.execute("""
            SELECT node, community_id FROM graph_leiden
            WHERE edge_table = 'edges'
              AND src_col = 'src'
              AND dst_col = 'dst'
              AND weight_col = 'weight'
        """).fetchall()

        # Find which communities contain seed entities
        node_community = {node: comm for node, comm in communities}
        seed_communities = {node_community[s] for s in seed_entities if s in node_community}

        for node, comm in communities:
            if comm in seed_communities:
                community_entities.add(node)
    except sqlite3.OperationalError as e:
        log.warning("Leiden community detection failed: %s", e)

    community_new = community_entities - expanded_entities
    result["stages"]["5_leiden_communities"] = {
        "seed_communities": len(seed_communities),
        "community_entities": len(community_entities),
        "newly_added": sorted(community_new)[:20],
    }
    log.info("[5] Leiden: %d community members (%d new)", len(community_entities), len(community_new))

    # ── [6] Node2Vec structural similarity ────────────────────────
    n2v_similar = set()

    if n2v_rowid_map:
        for seed in list(seed_entities)[:5]:
            seed_rowid = n2v_rowid_map.get(seed)
            if seed_rowid is None:
                continue
            vec_row = conn.execute("SELECT vector FROM node2vec_emb WHERE rowid = ?", (seed_rowid,)).fetchone()
            if vec_row is None:
                continue
            try:
                knn = conn.execute(
                    "SELECT rowid, distance FROM node2vec_emb WHERE vector MATCH ? AND k = ? AND ef_search = ?",
                    (vec_row[0], 10, HNSW_EF_SEARCH),
                ).fetchall()
                # Reverse map rowids to names
                rowid_to_name = {v: k for k, v in n2v_rowid_map.items()}
                for rid, dist in knn:
                    name = rowid_to_name.get(rid)
                    if name and name != seed:
                        n2v_similar.add(name)
            except sqlite3.OperationalError as e:
                log.debug("Node2Vec KNN from %r failed: %s", seed, e)

    n2v_new = n2v_similar - expanded_entities - community_entities
    result["stages"]["6_node2vec"] = {
        "structurally_similar": len(n2v_similar),
        "newly_added": sorted(n2v_new)[:20],
    }
    log.info("[6] Node2Vec: %d structurally similar (%d new)", len(n2v_similar), len(n2v_new))

    # ── [7] Assembly: merge, rank, collect passages ───────────────
    all_entities = expanded_entities | community_entities | n2v_similar

    # Collect relevant passages: chunks containing any of the final entities
    final_chunks = set()
    for entity in sorted(all_entities)[:50]:  # Limit for performance
        rows = conn.execute("""
            SELECT DISTINCT e.chunk_id
            FROM entities e
            JOIN entity_clusters ec ON e.name = ec.name
            WHERE ec.canonical = ? AND e.chunk_id IS NOT NULL
        """, (entity,)).fetchall()
        for r in rows:
            final_chunks.add(r[0])

    # Rank chunks by number of relevant entities they contain
    chunk_scores = collections.Counter()
    for entity in all_entities:
        rows = conn.execute("""
            SELECT e.chunk_id
            FROM entities e
            JOIN entity_clusters ec ON e.name = ec.name
            WHERE ec.canonical = ? AND e.chunk_id IS NOT NULL
        """, (entity,)).fetchall()
        for r in rows:
            # Weight by centrality if available
            weight = centrality_scores.get(entity, {}).get("betweenness", 0.01)
            chunk_scores[r[0]] += weight

    # Get top passages
    top_passages = []
    for chunk_id, score in chunk_scores.most_common(10):
        text_row = conn.execute("SELECT text FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        if text_row:
            top_passages.append({
                "chunk_id": chunk_id,
                "score": round(score, 6),
                "text_preview": text_row[0][:300],
            })

    result["stages"]["7_assembly"] = {
        "total_entities": len(all_entities),
        "total_chunks": len(final_chunks),
        "top_passages": top_passages[:5],
    }
    log.info("[7] Assembly: %d entities, %d chunks, %d ranked passages",
             len(all_entities), len(final_chunks), len(top_passages))

    # Summary
    result["summary"] = {
        "total_entities_found": len(all_entities),
        "total_passages": len(final_chunks),
        "subsystems_exercised": [
            "hnsw (chunk + entity search)",
            "graph_bfs (2-hop expansion)",
            "graph_betweenness + graph_closeness + graph_degree (centrality)",
            "graph_leiden (community detection)",
            "node2vec_train + node2vec_emb KNN (structural similarity)",
        ],
    }

    return result


# ── Main pipeline ─────────────────────────────────────────────────────


def coalesce_book(conn, book_id, blocking_threshold=0.4):
    """Run the full coalescing pipeline on a book's KG database."""
    log.info("Coalescing KG for book #%d (threshold=%.2f)", book_id, blocking_threshold)

    # Stage 1: HNSW blocking
    candidates = hnsw_blocking(conn, threshold=blocking_threshold)

    # Stage 2: Matching cascade
    matches = matching_cascade(candidates)

    # Stage 3: Leiden clustering
    entity_to_canonical = leiden_clustering(conn, matches)

    # Save cluster mappings
    save_clusters(conn, entity_to_canonical)

    # Build clean graph
    node_count, edge_count = build_clean_graph(conn)

    # Train Node2Vec
    n2v_rowid_map = {}
    if edge_count > 0:
        n2v_rowid_map = run_node2vec(conn)

    # Summary
    log.info("=" * 60)
    log.info("Coalescing complete for book #%d", book_id)
    log.info("  Canonical entities: %d", node_count)
    log.info("  Clean edges: %d", edge_count)
    log.info("  Node2Vec embeddings: %d", len(n2v_rowid_map))
    log.info("=" * 60)

    return n2v_rowid_map


def process_book(book_id, query_text=None, blocking_threshold=0.4, embedding_model=DEFAULT_EMBEDDING_MODEL):
    """Process a single book: coalesce + optional GraphRAG query."""
    db_path = KG_DIR / f"{book_id}.db"
    if not db_path.exists():
        log.error("KG database not found: %s", db_path)
        log.error("Run: python scripts/kg_extract.py --book-id %d", book_id)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    conn.load_extension(MUNINN_PATH)

    n2v_rowid_map = coalesce_book(conn, book_id, blocking_threshold)

    if query_text:
        result = graphrag_query(conn, query_text, embedding_model, n2v_rowid_map)
        result["book_id"] = book_id

        # Write to JSONL
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_path = RESULTS_DIR / "kg_graphrag.jsonl"
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        log.info("GraphRAG results appended to %s", results_path)

        # Print summary
        summary = result.get("summary", {})
        log.info("GraphRAG summary: %d entities, %d passages, %d subsystems",
                 summary.get("total_entities_found", 0),
                 summary.get("total_passages", 0),
                 len(summary.get("subsystems_exercised", [])))

    conn.close()


# ── CLI ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="KG Entity Resolution + GraphRAG Demo")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--book-id", type=int, help="Process a specific book")
    group.add_argument("--all", action="store_true", help="Process all KG databases")
    parser.add_argument("--query", type=str, help="GraphRAG query to run after coalescing")
    parser.add_argument("--blocking-threshold", type=float, default=0.4,
                        help="Cosine distance threshold for HNSW blocking (default: 0.4)")
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL,
                        help=f"Embedding model for queries (default: {DEFAULT_EMBEDDING_MODEL})")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    if args.book_id:
        process_book(args.book_id, args.query, args.blocking_threshold, args.embedding_model)
    elif args.all:
        KG_DIR.mkdir(parents=True, exist_ok=True)
        db_files = sorted(KG_DIR.glob("*.db"))
        if not db_files:
            log.warning("No KG databases found in %s", KG_DIR)
            sys.exit(1)
        for db_path in db_files:
            book_id = int(db_path.stem)
            log.info("Processing book #%d...", book_id)
            process_book(book_id, args.query, args.blocking_threshold, args.embedding_model)


if __name__ == "__main__":
    main()
