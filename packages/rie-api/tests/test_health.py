"""`/healthz` is the cheap liveness probe — no I/O required."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_healthz_returns_ok(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["components"]["api"] == "ok"


def test_request_id_echoed_in_response(client: TestClient) -> None:
    response = client.get("/healthz", headers={"x-request-id": "deadbeef"})
    assert response.status_code == 200
    assert response.headers.get("x-request-id") == "deadbeef"
