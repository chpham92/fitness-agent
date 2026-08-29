"""
HTTP-level tests for app/main.py: confirm both /chat and /chat/stream turn
an Anthropic API failure into a clean, typed response — never a raw
exception or stack trace reaching the client.

Uses the real FastAPI lifespan (so app.state.engine/anthropic_client get
created normally), then swaps app.state.anthropic_client for a client that
fails deterministically before making requests. The lifespan's seed() call
writes to the default fitness_agent.db in the cwd; it's idempotent, and the
fixture removes the file afterward.
"""
import json
import os

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.fakes import FailingClient, make_rate_limit_error, make_timeout_error

DB_FILE = "fitness_agent.db"


@pytest.fixture
def api_client():
    with TestClient(app) as client:
        yield client
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)


def test_health(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_surfaces_rate_limit_as_clean_error(api_client):
    app.state.anthropic_client = FailingClient(make_rate_limit_error())

    resp = api_client.post("/chat", json={"user_id": "chris", "message": "hi"})

    assert resp.status_code == 429
    body = resp.json()
    assert "detail" in body
    assert "Traceback" not in json.dumps(body)


def test_chat_stream_emits_error_event_on_rate_limit(api_client):
    app.state.anthropic_client = FailingClient(make_rate_limit_error())

    with api_client.stream(
        "POST", "/chat/stream", json={"user_id": "chris", "message": "hi"}
    ) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())

    assert "event: error" in body
    assert "Traceback" not in body


def test_chat_surfaces_timeout_as_clean_error(api_client):
    app.state.anthropic_client = FailingClient(make_timeout_error())

    resp = api_client.post("/chat", json={"user_id": "chris", "message": "hi"})

    assert resp.status_code == 504
    body = resp.json()
    assert "detail" in body
    assert "Traceback" not in json.dumps(body)


# ---------------------------------------------------------------------------
# Empty/whitespace-only messages are rejected by ChatRequest's own
# validation, before the route ever runs — so no Anthropic call happens
# and no fake client is needed here. Covers both endpoints since each has
# its own ChatRequest body.
# ---------------------------------------------------------------------------


def test_chat_rejects_empty_message(api_client):
    resp = api_client.post("/chat", json={"user_id": "chris", "message": ""})
    assert resp.status_code == 422


def test_chat_rejects_whitespace_only_message(api_client):
    resp = api_client.post("/chat", json={"user_id": "chris", "message": "   \n\t  "})
    assert resp.status_code == 422


def test_chat_stream_rejects_empty_message(api_client):
    resp = api_client.post("/chat/stream", json={"user_id": "chris", "message": ""})
    assert resp.status_code == 422
