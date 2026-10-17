import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.models.domain import ORDER_REQUEST
from tests.conftest import ScriptedLLM, make_settings, plan

REPLY = "Hello,\n\nYour Sleek Wallet (SWL2345) x2 is confirmed.\n\nWarm regards,"


def wallet_order():
    return [
        ("search_catalog", {"query": "SWL2345"}),
        ("reserve_stock", {"items": [{"product_id": "SWL2345", "quantity": 2, "all_available": False}]}),
        ("submit_reply", {"reply": REPLY}),
    ]


@pytest.fixture
def api(make_engine):
    llm = ScriptedLLM(
        plans={"W-1": plan(ORDER_REQUEST, "search_catalog", "reserve_stock", "submit_reply")},
        steps={"W-1": wallet_order()},
    )
    engine = make_engine(llm)
    with TestClient(create_app(engine.settings, engine)) as client:
        yield client


def test_health_and_readiness(api):
    assert api.get("/healthz").json()["status"] == "ok"
    ready = api.get("/readyz")
    assert ready.status_code == 200 and ready.json()["status"] == "ready"


def test_process_email_returns_plan_trace_and_lines(api):
    resp = api.post("/v1/emails", json={"email_id": "W-1", "body": "Please send me 2 SWL2345."})
    assert resp.status_code == 200 and resp.headers["X-Request-ID"]
    body = resp.json()
    assert body["status"] == "processed" and body["category"] == "order request"
    assert body["lines"][0]["status"] == "created" and body["lines"][0]["remaining_stock"] == 3
    assert [s["tool"] for s in body["plan"]["steps"]] == ["search_catalog", "reserve_stock", "submit_reply"]
    assert [e["tool"] for e in body["trace"] if e["type"] == "tool"] == [
        "search_catalog",
        "reserve_stock",
        "submit_reply",
    ]
    assert api.post("/v1/emails", json={"email_id": "W-1", "body": "Please send me 2 SWL2345."}).json()["duplicate"]
    assert api.get("/v1/emails/W-1").json()["lines"][0]["remaining_stock"] == 3
    assert api.get("/v1/emails/unknown").status_code == 404


def test_validation_and_batch_rules(api):
    assert api.post("/v1/emails", json={"body": ""}).status_code == 422
    assert api.post("/v1/emails", json={"email_id": "bad id!", "body": "x"}).status_code == 422
    dup = api.post(
        "/v1/emails/batch", json={"emails": [{"email_id": "X", "body": "a"}, {"email_id": "X", "body": "b"}]}
    )
    assert dup.status_code == 422


def test_model_failure_is_reported_not_raised(api):
    body = api.post("/v1/emails", json={"email_id": "NOSCRIPT", "body": "Hello?"}).json()
    assert body["status"] == "failed" and body["response"] is None and "planning failed" in body["error"]


def test_product_endpoints(api):
    assert api.get("/v1/products/swl2345").json()["stock"] == 5
    assert api.get("/v1/products/NOPE0000").status_code == 404
    hits = api.get("/v1/products", params={"query": "DHN0987", "limit": 3}).json()["results"]
    assert hits[0]["product_id"] == "CHN0987" and hits[0]["match"] == "similar_id"


def test_api_key_is_enforced_when_configured(make_engine):
    engine = make_engine()
    with TestClient(create_app(make_settings(API_KEY="s3cret"), engine)) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/v1/products/SWL2345").status_code == 401
        assert client.get("/v1/products/SWL2345", headers={"X-API-Key": "s3cret"}).status_code == 200
