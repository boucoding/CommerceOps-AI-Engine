import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from tests.conftest import make_settings


@pytest.fixture
def client(build):
    engine = build()
    with TestClient(create_app(engine.settings, engine)) as c:
        yield c


def test_health_and_readiness(client):
    assert client.get("/healthz").json()["status"] == "ok"
    ready = client.get("/readyz")
    assert ready.status_code == 200 and ready.json() == {"status": "ready", "storage": "memory", "llm_mode": "offline"}


def test_process_email_and_fetch_it_back(client):
    resp = client.post("/v1/emails", json={"email_id": "W-1", "subject": "Order", "body": "Please send me 2 SWL2345."})
    assert resp.status_code == 200
    body = resp.json()
    assert body["category"] == "order request" and body["status"] == "processed"
    assert body["lines"][0] == {
        "product_id": "SWL2345",
        "product_name": "Sleek Wallet",
        "requested_quantity": 2,
        "fulfilled_quantity": 2,
        "status": "created",
        "remaining_stock": 3,
    }
    assert resp.headers["X-Request-ID"]

    again = client.post("/v1/emails", json={"email_id": "W-1", "body": "Please send me 2 SWL2345."}).json()
    assert again["duplicate"] is True
    assert client.get("/v1/products/swl2345").json()["stock"] == 3
    assert client.get("/v1/emails/W-1").json()["lines"][0]["remaining_stock"] == 3
    assert client.get("/v1/emails/unknown").status_code == 404


def test_generated_id_and_validation(client):
    resp = client.post("/v1/emails", json={"body": "What material is the Leather Tote?"})
    assert resp.status_code == 200 and resp.json()["email_id"].startswith("API-")
    assert client.post("/v1/emails", json={"body": ""}).status_code == 422
    assert client.post("/v1/emails", json={"email_id": "bad id!", "body": "x"}).status_code == 422


def test_batch(client):
    resp = client.post(
        "/v1/emails/batch",
        json={
            "emails": [
                {"email_id": "B1", "body": "I would like to order 1 RSG8901."},
                {"email_id": "B2", "body": "I would like to order 1 RSG8901."},
            ]
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert [o["lines"][0]["status"] for o in data["outcomes"]] == ["created", "out of stock"]
    assert data["summary"]["order_lines"] == 2
    dup = client.post(
        "/v1/emails/batch", json={"emails": [{"email_id": "X", "body": "a"}, {"email_id": "X", "body": "b"}]}
    )
    assert dup.status_code == 422


def test_product_search(client):
    data = client.get("/v1/products", params={"query": "DHN0987 or a laptop bag", "limit": 2}).json()
    assert [p["product_id"] for p in data["mentioned"]] == ["CHN0987"]
    assert len(data["related"]) == 2
    assert client.get("/v1/products/NOPE0000").status_code == 404


def test_api_key_is_enforced_when_configured(build):
    engine = build()
    settings = make_settings(API_KEY="s3cret")
    with TestClient(create_app(settings, engine)) as c:
        assert c.get("/healthz").status_code == 200
        assert c.post("/v1/emails", json={"body": "hi"}).status_code == 401
        ok = c.post("/v1/emails", json={"body": "What is the Leather Tote?"}, headers={"X-API-Key": "s3cret"})
        assert ok.status_code == 200
