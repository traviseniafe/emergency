from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_endpoint_reports_ok():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_frontend_page_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "Emergency Detection" in response.text


def test_frontend_script_is_served():
    response = client.get("/app.js")
    assert response.status_code == 200
    assert "/api/health" in response.text