"""FastAPI endpoint tests."""


def test_health():
    from fastapi.testclient import TestClient

    from backend.app import app

    r = TestClient(app).get("/health")
    assert r.status_code == 200 and r.json()["status"] == "alive"


def test_ready():
    from fastapi.testclient import TestClient

    from backend.app import app

    r = TestClient(app).get("/ready")
    assert r.status_code in (200, 503)


def test_version():
    from fastapi.testclient import TestClient

    from backend.app import app

    r = TestClient(app).get("/version")
    assert r.status_code == 200 and "api_version" in r.json()


def test_metrics():
    from fastapi.testclient import TestClient

    from backend.app import app

    r = TestClient(app).get("/metrics")
    assert r.status_code == 200 and "tidal_requests_total" in r.text
