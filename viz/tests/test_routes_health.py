"""Tests for the health endpoint."""

import pathlib

from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    """Health endpoint returns status ok."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["db_exists"] is True
    assert data["extension_loaded"] is True


def test_health_reports_index_count(client: TestClient) -> None:
    """Health endpoint reports HNSW index count."""
    resp = client.get("/api/health")
    data = resp.json()
    assert "hnsw_index_count" in data
    assert data["hnsw_index_count"] >= 1  # test_vec


def test_health_reports_edge_table_count(client: TestClient) -> None:
    """Health endpoint reports edge table count."""
    resp = client.get("/api/health")
    data = resp.json()
    assert "edge_table_count" in data
    assert data["edge_table_count"] >= 1  # test_edges


def test_health_error_branch_when_db_unavailable(tmp_path: pathlib.Path) -> None:
    """Health endpoint catches exceptions and sets extension_loaded=False."""
    from server import config
    from server.services import db

    original_db_path = config.DB_PATH
    config.DB_PATH = str(tmp_path / "nonexistent.db")
    db.close_connection()
    db.reset_connection()

    from server.main import app

    test_client = TestClient(app)
    try:
        resp = test_client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        # The error branch is exercised: extension_loaded=False and error present
        assert data["extension_loaded"] is False
        assert "error" in data
        assert "Database not found" in data["error"]
    finally:
        config.DB_PATH = original_db_path
        db.close_connection()
        db.reset_connection()
