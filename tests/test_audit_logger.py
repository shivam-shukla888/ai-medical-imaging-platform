"""
Tests for append-only audit logger and FastAPI audit endpoints.
"""
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.services.audit_logger import AuditLogger
from app.api.main import app

client = TestClient(app)


def test_audit_logger_write_and_read(tmp_path):
    """Verify that AuditLogger appends records safely and retrieves recent items."""
    log_file = tmp_path / "test_audit.jsonl"
    logger = AuditLogger(log_path=log_file)

    logger.log_event("TEST_EVENT_1", {"sample_id": 1, "score": 0.95})
    logger.log_event("TEST_EVENT_2", {"sample_id": 2, "error": "none"})

    records = logger.get_recent_events(limit=10)
    assert len(records) == 2
    assert records[0]["event_type"] == "TEST_EVENT_1"
    assert records[0]["details"]["sample_id"] == 1
    assert records[1]["event_type"] == "TEST_EVENT_2"
    assert "timestamp" in records[0]


def test_api_audit_logs_endpoint():
    """Verify /audit-logs endpoint returns recorded events."""
    # Trigger a sample prediction to produce an audit record
    client.post("/predict/sample/0")

    response = client.get("/audit-logs?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "events" in data
    assert isinstance(data["events"], list)
    assert data["count"] >= 1


def test_api_orthanc_status_endpoint():
    """Verify /orthanc/status endpoint returns valid status schema."""
    response = client.get("/orthanc/status")
    assert response.status_code == 200
    data = response.json()
    assert "online" in data


def test_api_monai_3d_demo_endpoint():
    """Verify /monai/demo-3d endpoint runs 3D segmentation demonstration."""
    response = client.get("/monai/demo-3d")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["spatial_dimensions"] == 3
