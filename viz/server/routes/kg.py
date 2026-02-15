"""Knowledge Graph pipeline endpoints."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from server.services.db import db_session
from server.services.kg import get_pipeline_summary, get_stage_detail

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/kg", tags=["kg"])


@router.get("/pipeline")
def pipeline_summary(conn=Depends(db_session)) -> dict[str, Any]:
    """Get summary of all KG pipeline stages."""
    return get_pipeline_summary(conn)


@router.get("/stage/{stage_num}")
def stage_detail(stage_num: int, conn=Depends(db_session)) -> dict[str, Any]:
    """Get detailed data for a specific pipeline stage (1-7)."""
    if stage_num < 1 or stage_num > 7:
        raise HTTPException(status_code=400, detail="Stage number must be between 1 and 7")

    return get_stage_detail(conn, stage_num)


# Stage → (table_name, columns, filter_column)
_STAGE_TABLE_MAP: dict[int, tuple[str, list[str], str]] = {
    1: ("chunks", ["chunk_id", "text"], "text"),
    3: ("entities", ["name", "entity_type", "chunk_id"], "name"),
    4: ("relations", ["src", "dst", "rel_type", "weight"], "src"),
    5: ("entity_clusters", ["canonical", "name"], "name"),
}


@router.get("/stage/{stage_num}/items")
def stage_items(
    stage_num: int,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    q: str = Query(default=""),
    conn=Depends(db_session),
) -> dict[str, Any]:
    """Get paginated rows from a pipeline stage's table."""
    if stage_num not in _STAGE_TABLE_MAP:
        raise HTTPException(status_code=400, detail=f"Stage {stage_num} does not support item listing")

    table, columns, filter_col = _STAGE_TABLE_MAP[stage_num]

    # Check table exists
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if not exists:
        return {"stage": stage_num, "items": [], "total": 0, "page": page, "page_size": page_size}

    # Discover which columns actually exist (graceful if schema differs)
    pragma_rows = conn.execute(f"PRAGMA table_info([{table}])").fetchall()
    actual_cols = {row["name"] for row in pragma_rows}
    select_cols = [c for c in columns if c in actual_cols]
    if not select_cols:
        select_cols = list(actual_cols)[:4]  # Fallback to first 4 columns

    col_expr = ", ".join(f"[{c}]" for c in select_cols)

    # Filter
    where = ""
    params: list[Any] = []
    if q and filter_col in actual_cols:
        where = f"WHERE [{filter_col}] LIKE ?"
        params.append(f"%{q}%")

    # Total count
    total = conn.execute(f"SELECT count(*) FROM [{table}] {where}", params).fetchone()[0]

    # Paginated query
    offset = (page - 1) * page_size
    rows = conn.execute(
        f"SELECT {col_expr} FROM [{table}] {where} LIMIT ? OFFSET ?",
        params + [page_size, offset],
    ).fetchall()

    items = [dict(row) for row in rows]

    return {
        "stage": stage_num,
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


class GraphRAGRequest(BaseModel):
    """Request body for GraphRAG query."""

    query: str
    k: int = 10
    max_depth: int = 2


@router.post("/query")
def graphrag_query(request: GraphRAGRequest, conn=Depends(db_session)) -> dict[str, Any]:
    """Execute a GraphRAG query (stages 2-7, using pre-embedded chunks)."""

    try:
        from server.services.kg import run_graphrag_query

        return run_graphrag_query(conn, request.query, k=request.k, max_depth=request.max_depth)
    except Exception as e:
        log.error("GraphRAG query failed: %s", e)
        raise HTTPException(status_code=500, detail=f"GraphRAG query failed: {e}") from e
