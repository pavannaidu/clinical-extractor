"""Keep demo telemetry private, optional, and independent of page responses."""

import hashlib
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import main
from api import database
from dbdemos_tracker.tracker import _InFlightPoster


@pytest.fixture
def tracking_environment(monkeypatch):
    events = []
    workspace = Mock()
    workspace.get_workspace_id.return_value = "synthetic-workspace"
    workspace_factory = Mock(return_value=workspace)
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example.org")
    monkeypatch.setattr(main, "WorkspaceClient", workspace_factory)
    monkeypatch.setattr(database, "is_configured", lambda: False)
    monkeypatch.setattr(
        main.Tracker,
        "_poster",
        SimpleNamespace(post=lambda payload, _headers: events.append(payload)),
    )
    return events, workspace_factory


@pytest.mark.parametrize("path", ["/", "/index.html", "/info/", "/info/index.html"])
def test_page_views_use_demo_id_and_hashed_internal_email(tracking_environment, path):
    events, workspace_factory = tracking_environment
    with TestClient(main.app) as client:
        response = client.get(path, headers={"x-forwarded-email": " Demo@Databricks.com "})

    assert response.status_code == 200
    assert len(events) == 1
    event = events[0]
    assert event["demo_name"] == "hls-clinical-extractor"
    assert event["category"] == "demo-app"
    assert event["event"] == "APP-VIEW"
    assert event["version"] == 1
    assert event["org_id"] == "synthetic-workspace"
    assert event["app_path"] == path
    assert event["user_hash"] == hashlib.sha256(b"demo@databricks.com").hexdigest()
    assert "@" not in json.dumps(event)
    workspace_factory.assert_called_once()
    workspace_factory.return_value.get_workspace_id.assert_called_once()


@pytest.mark.parametrize("email", [None, "", "demo@example.org", "demo@databricks.com.example.org"])
def test_other_visitors_do_not_emit_events(tracking_environment, email):
    events, _ = tracking_environment
    headers = {"x-forwarded-email": email} if email is not None else {}
    with TestClient(main.app) as client:
        response = client.get("/", headers=headers)

    assert response.status_code == 200
    assert events == []


def test_only_page_views_are_tracked_without_queries(tracking_environment):
    events, _ = tracking_environment
    headers = {"x-forwarded-email": "demo@databricks.com"}
    with TestClient(main.app) as client:
        assert client.get("/?note=synthetic-note", headers=headers).status_code == 200
        assert client.get("/info/deck.css", headers=headers).status_code == 200
        assert client.get("/openapi.json", headers=headers).status_code == 200

    assert len(events) == 1
    assert events[0]["app_path"] == "/"
    assert "synthetic-note" not in json.dumps(events)


def test_info_redirect_counts_one_view(tracking_environment):
    events, _ = tracking_environment
    with TestClient(main.app) as client:
        response = client.get("/info", headers={"x-forwarded-email": "demo@databricks.com"})

    assert response.status_code == 200
    assert len(events) == 1
    assert events[0]["app_path"] == "/info/"


def test_tracking_errors_do_not_break_the_page(tracking_environment, monkeypatch):
    events, _ = tracking_environment
    monkeypatch.setattr(main.Tracker, "track_app_view", Mock(side_effect=RuntimeError("unavailable")))
    with TestClient(main.app) as client:
        response = client.get("/", headers={"x-forwarded-email": "demo@databricks.com"})

    assert response.status_code == 200
    assert events == []


def test_tracker_initialization_errors_do_not_break_startup(tracking_environment):
    events, workspace_factory = tracking_environment
    workspace_factory.side_effect = RuntimeError("workspace unavailable")
    with TestClient(main.app) as client:
        response = client.get("/", headers={"x-forwarded-email": "demo@databricks.com"})

    assert response.status_code == 200
    assert events == []


def test_local_startup_without_workspace_skips_tracking(tracking_environment, monkeypatch):
    events, workspace_factory = tracking_environment
    monkeypatch.delenv("DATABRICKS_HOST")
    with TestClient(main.app) as client:
        response = client.get("/", headers={"x-forwarded-email": "demo@databricks.com"})

    assert response.status_code == 200
    assert events == []
    workspace_factory.assert_not_called()


def test_slow_analytics_delivery_does_not_hold_the_page(tracking_environment, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def slow_post(*_args, **_kwargs):
        started.set()
        try:
            assert release.wait(timeout=5)
        finally:
            finished.set()

    session = Mock()
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)
    session.post.side_effect = slow_post
    monkeypatch.setattr("dbdemos_tracker.tracker.requests.Session", lambda: session)
    monkeypatch.setattr(main.Tracker, "_poster", _InFlightPoster("https://analytics.example.org"))
    try:
        with TestClient(main.app) as client:
            response = client.get("/", headers={"x-forwarded-email": "demo@databricks.com"})

        assert response.status_code == 200
        assert started.wait(timeout=1)
        assert not finished.is_set()
    finally:
        release.set()
        assert finished.wait(timeout=1)
