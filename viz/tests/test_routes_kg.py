"""Tests for KG pipeline endpoints.

These tests use the test_db which has minimal data -- no KG pipeline tables.
The endpoints should return gracefully with available=False for missing stages.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient


def test_pipeline_summary(client: TestClient) -> None:
    """GET /api/kg/pipeline returns all 7 stages."""
    resp = client.get("/api/kg/pipeline")
    assert resp.status_code == 200
    data = resp.json()
    assert "stages" in data
    assert len(data["stages"]) == 7
    # Each stage should have required fields
    for stage in data["stages"]:
        assert "stage" in stage
        assert "name" in stage
        assert "count" in stage
        assert "available" in stage


def test_pipeline_stages_numbered(client: TestClient) -> None:
    """Pipeline stages are numbered 1-7."""
    resp = client.get("/api/kg/pipeline")
    data = resp.json()
    stage_nums = [s["stage"] for s in data["stages"]]
    assert stage_nums == [1, 2, 3, 4, 5, 6, 7]


def test_stage_detail_valid(client: TestClient) -> None:
    """GET /api/kg/stage/{n} returns stage detail for valid numbers."""
    for n in range(1, 8):
        resp = client.get(f"/api/kg/stage/{n}")
        assert resp.status_code == 200


def test_stage_detail_invalid_number(client: TestClient) -> None:
    """GET /api/kg/stage/{n} returns 400 for out-of-range numbers."""
    resp = client.get("/api/kg/stage/0")
    assert resp.status_code == 400
    resp = client.get("/api/kg/stage/8")
    assert resp.status_code == 400


def test_stage_unavailable_returns_gracefully(client: TestClient) -> None:
    """Stages with missing tables return available=False or count=0."""
    # The test_db doesn't have chunks, entities, etc.
    # Stage 2 (embeddings) should be unavailable -- no chunks_vec_config
    resp = client.get("/api/kg/stage/2")
    data = resp.json()
    assert data.get("available") is False


def test_graphrag_query(client: TestClient) -> None:
    """POST /api/kg/query returns a result dict with stages."""
    resp = client.post("/api/kg/query", json={"query": "test query"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "test query"
    assert "stages" in data


def test_graphrag_query_error_handling(client: TestClient) -> None:
    """POST /api/kg/query returns 500 when graphrag execution fails."""
    with patch("server.services.kg.run_graphrag_query", side_effect=RuntimeError("forced error")):
        resp = client.post("/api/kg/query", json={"query": "test query"})
        assert resp.status_code == 500
        assert "GraphRAG query failed" in resp.json()["detail"]


def test_stage_items_invalid_stage(client: TestClient) -> None:
    """GET /api/kg/stage/{n}/items returns 400 for unsupported stages."""
    resp = client.get("/api/kg/stage/2/items")
    assert resp.status_code == 400


def test_stage_items_missing_table(client: TestClient) -> None:
    """GET /api/kg/stage/{n}/items returns empty for missing tables."""
    # The test_db doesn't have a chunks table, but stage 1 is valid
    resp = client.get("/api/kg/stage/1/items")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["items"] == []
    assert data["page"] == 1


def test_stage_items_pagination(tmp_path) -> None:
    """GET /api/kg/stage/{n}/items returns paginated data."""
    import pathlib

    try:
        import pysqlite3 as sqlite3
    except ImportError:
        import sqlite3

    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
    EXTENSION_PATH = str(PROJECT_ROOT / "build" / "muninn")

    db_path = str(tmp_path / "pagination.db")
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(EXTENSION_PATH)

    # Create entities table with data
    conn.execute("CREATE TABLE entities (name TEXT, entity_type TEXT, chunk_id INTEGER)")
    for i in range(50):
        conn.execute("INSERT INTO entities VALUES (?, ?, ?)", (f"entity_{i}", "PERSON", i))
    conn.commit()
    conn.close()

    from server import config
    from server.services import db

    original_db_path = config.DB_PATH
    config.DB_PATH = db_path
    db.close_connection()
    db.reset_connection()

    from server.main import app

    test_client = TestClient(app)
    try:
        # Page 1
        resp = test_client.get("/api/kg/stage/3/items?page=1&page_size=10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 50
        assert len(data["items"]) == 10
        assert data["page"] == 1

        # Page 2
        resp = test_client.get("/api/kg/stage/3/items?page=2&page_size=10")
        data = resp.json()
        assert len(data["items"]) == 10
        assert data["page"] == 2

        # With filter
        resp = test_client.get("/api/kg/stage/3/items?q=entity_1")
        data = resp.json()
        assert data["total"] >= 1
        assert all("entity_1" in item["name"] for item in data["items"])
    finally:
        config.DB_PATH = original_db_path
        db.close_connection()
        db.reset_connection()
