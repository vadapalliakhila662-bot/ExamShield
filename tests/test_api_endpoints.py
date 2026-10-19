"""
Integration Tests for ExamShield FastAPI Endpoints
"""

from fastapi.testclient import TestClient
from examshield.web.api import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["opencv_version"].startswith("5.")
    assert data["opencv_major_requirement_met"] is True

def test_verify_opencv5_endpoint():
    response = client.get("/api/verify-opencv5")
    assert response.status_code == 200
    data = response.json()
    assert data["verified"] is True
    assert data["modules"]["cv2.solvePnP"] is True
    assert data["modules"]["cv2.createCLAHE"] is True

def test_evaluate_frame_and_trace_endpoint():
    session_id = "test-endpoint-session-42"
    payload = {
        "session_id": session_id,
        "student_id": "test-student-01",
        "duration_anomaly_active_seconds": 10.0,
        "accommodation": {
            "student_id": "test-student-01",
            "has_adhd": False,
            "has_strabismus": False
        }
    }
    response = client.post("/api/evaluate-frame", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert "status" in res_data
    assert "confidence" in res_data

    # Now verify trace retrieval
    trace_res = client.get(f"/api/trace/{session_id}")
    assert trace_res.status_code == 200
    trace_data = trace_res.json()
    assert trace_data["session_id"] == session_id
    assert trace_data["total_steps"] >= 1

def test_appeal_endpoint():
    payload = {
        "session_id": "test-endpoint-session-42",
        "student_id": "test-student-01",
        "student_claim": "My pen dropped on the ground so I bent down to retrieve it."
    }
    response = client.post("/api/appeal", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "consistency_verdict" in data
    assert "recommended_action" in data
