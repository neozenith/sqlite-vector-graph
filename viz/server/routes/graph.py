"""Graph exploration endpoints using muninn TVFs."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from server.services.db import db_session, discover_edge_tables
from server.services.validation import validate_identifier

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["graph"])


@router.get("/graphs")
def list_graphs(conn=Depends(db_session)) -> list[dict[str, Any]]:
    """List all discovered edge tables."""
    return discover_edge_tables(conn)


def _validate_edge_table(conn, edge_table: str) -> dict[str, Any]:
    """Validate and find an edge table or raise 400/404."""
    try:
        validate_identifier(edge_table)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    tables = discover_edge_tables(conn)
    match = next((t for t in tables if t["table_name"] == edge_table), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"Edge table not found: {edge_table}")
    return match


@router.get("/graph/{edge_table}/subgraph")
def get_subgraph(
    edge_table: str,
    limit: int = Query(default=500, ge=1, le=5000),
    conn=Depends(db_session),
) -> dict[str, Any]:
    """Get a subgraph of nodes and edges for visualization."""
    info = _validate_edge_table(conn, edge_table)
    src_col = info["src_col"]
    dst_col = info["dst_col"]
    weight_col = info["weight_col"]

    # Check for optional rel_type column
    columns = conn.execute(f"PRAGMA table_info([{edge_table}])").fetchall()
    col_names = {row["name"] for row in columns}
    has_rel_type = "rel_type" in col_names

    # Get edges with optional weight and rel_type
    weight_expr = f", [{weight_col}]" if weight_col else ""
    rel_type_expr = ", [rel_type]" if has_rel_type else ""
    rows = conn.execute(
        f"SELECT [{src_col}], [{dst_col}]{weight_expr}{rel_type_expr} FROM [{edge_table}] LIMIT ?",
        (limit,),
    ).fetchall()

    nodes_set: set[str] = set()
    edges = []
    for row in rows:
        src, dst = row[0], row[1]
        col_idx = 2
        weight = float(row[col_idx]) if weight_col else 1.0
        if weight_col:
            col_idx += 1
        edge: dict[str, Any] = {"source": str(src), "target": str(dst), "weight": weight}
        if has_rel_type:
            edge["rel_type"] = row[col_idx]
        nodes_set.add(str(src))
        nodes_set.add(str(dst))
        edges.append(edge)

    # Build node list with optional metadata from nodes table
    nodes = []
    for node_name in sorted(nodes_set):
        node: dict[str, Any] = {"id": node_name, "label": node_name}
        # Try to get metadata from a nodes table
        try:
            meta_row = conn.execute(
                "SELECT mention_count, entity_type FROM nodes WHERE name = ?", (node_name,)
            ).fetchone()
            if meta_row:
                node["mention_count"] = meta_row["mention_count"]
                node["entity_type"] = meta_row["entity_type"]
        except Exception:
            pass
        nodes.append(node)

    return {
        "edge_table": edge_table,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


@router.get("/graph/{edge_table}/bfs")
def bfs_traversal(
    edge_table: str,
    start: str = Query(...),
    max_depth: int = Query(default=3, ge=1, le=10),
    direction: str = Query(default="both", pattern="^(out|in|both)$"),
    conn=Depends(db_session),
) -> dict[str, Any]:
    """BFS traversal from a start node using muninn's graph_bfs TVF."""
    info = _validate_edge_table(conn, edge_table)

    try:
        rows = conn.execute("""
            SELECT node, depth FROM graph_bfs
            WHERE edge_table = ?
              AND src_col = ?
              AND dst_col = ?
              AND start_node = ?
              AND max_depth = ?
              AND direction = ?
        """, (edge_table, info["src_col"], info["dst_col"], start, max_depth, direction)).fetchall()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"BFS failed: {e}") from e

    nodes = [{"node": row["node"], "depth": row["depth"]} for row in rows]

    return {
        "edge_table": edge_table,
        "start_node": start,
        "max_depth": max_depth,
        "direction": direction,
        "count": len(nodes),
        "nodes": nodes,
    }


@router.get("/graph/{edge_table}/communities")
def detect_communities(
    edge_table: str,
    resolution: float = Query(default=1.0, ge=0.1, le=10.0),
    conn=Depends(db_session),
) -> dict[str, Any]:
    """Community detection via muninn's graph_leiden TVF."""
    info = _validate_edge_table(conn, edge_table)

    weight_clause = f"AND weight_col = '{info['weight_col']}'" if info["weight_col"] else ""

    try:
        rows = conn.execute(f"""
            SELECT node, community_id FROM graph_leiden
            WHERE edge_table = ?
              AND src_col = ?
              AND dst_col = ?
              {weight_clause}
              AND resolution = ?
        """, (edge_table, info["src_col"], info["dst_col"], resolution)).fetchall()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Leiden failed: {e}") from e

    communities: dict[int, list[str]] = {}
    node_community: dict[str, int] = {}
    for row in rows:
        node, comm = row["node"], row["community_id"]
        communities.setdefault(comm, []).append(node)
        node_community[node] = comm

    return {
        "edge_table": edge_table,
        "resolution": resolution,
        "community_count": len(communities),
        "node_count": len(node_community),
        "communities": communities,
        "node_community": node_community,
    }


@router.get("/graph/{edge_table}/centrality")
def compute_centrality(
    edge_table: str,
    measure: str = Query(default="degree", pattern="^(degree|betweenness|closeness)$"),
    direction: str = Query(default="both", pattern="^(out|in|both)$"),
    conn=Depends(db_session),
) -> dict[str, Any]:
    """Compute centrality measures via muninn's centrality TVFs."""
    info = _validate_edge_table(conn, edge_table)

    tvf_name = f"graph_{measure}"

    # Degree doesn't support direction parameter
    if measure == "degree":
        try:
            rows = conn.execute(f"""
                SELECT node, centrality FROM [{tvf_name}]
                WHERE edge_table = ?
                  AND src_col = ?
                  AND dst_col = ?
            """, (edge_table, info["src_col"], info["dst_col"])).fetchall()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Centrality failed: {e}") from e
    else:
        try:
            rows = conn.execute(f"""
                SELECT node, centrality FROM [{tvf_name}]
                WHERE edge_table = ?
                  AND src_col = ?
                  AND dst_col = ?
                  AND direction = ?
            """, (edge_table, info["src_col"], info["dst_col"], direction)).fetchall()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Centrality failed: {e}") from e

    scores = [{"node": row["node"], "centrality": round(row["centrality"], 6)} for row in rows]
    scores.sort(key=lambda x: -x["centrality"])

    return {
        "edge_table": edge_table,
        "measure": measure,
        "direction": direction,
        "count": len(scores),
        "scores": scores,
    }
