"""Camera ownership stays visible across modes, release failures, and HTTP polls."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from interface import server


HEADERS = {"host": "127.0.0.1:7432", "x-intuition-camera": "1"}


def resource(state):
    return {"state": state, "in_use": False if state == "released" else None if state == "opening" else True,
            "release_confirmed": state == "released", "cleanup_pending": state in ("releasing", "release_failed")}


@pytest.fixture
def cameras(monkeypatch):
    values = {"gestures": resource("released"), "preview_camera": resource("released")}
    owners = {name: SimpleNamespace(status=lambda key=name: {"camera_resource": values[key]}) for name in values}
    monkeypatch.setattr(server, "_state", owners)
    monkeypatch.setattr(server, "_camera_privacy_previous", None)
    monkeypatch.setattr(server, "_camera_privacy_revision", 0)
    return values


@pytest.mark.parametrize("owner", ["gestures", "preview_camera"])
@pytest.mark.parametrize("native,state,in_use,released", [
    ("released", "off", False, True), ("opening", "starting", None, False),
    ("open", "on", True, False), ("releasing", "stopping", True, False),
    ("release_failed", "unknown", True, False),
])
def test_either_native_capture_controls_indicator(cameras, owner, native, state, in_use, released):
    cameras[owner] = resource(native)
    info = server._camera_privacy_info()
    assert (info["state"], info["in_use"], info["release_confirmed"]) == (state, in_use, released)
    assert info["owners"] == ([] if released else ["hand" if owner == "gestures" else "eeg"])


def test_stopped_hand_mode_cannot_hide_active_eeg_camera(cameras):
    cameras["preview_camera"] = resource("open")
    assert server._gesture_info()["camera_privacy"]["in_use"] is True
    cameras["gestures"] = resource("releasing")
    assert server._camera_privacy_info()["state"] == "on"
    cameras["gestures"] = resource("released")
    assert server._camera_privacy_info()["owners"] == ["eeg"]


def test_revision_changes_only_for_new_ownership_and_failed_release_never_means_off(cameras):
    initial = server._camera_privacy_info()
    assert server._camera_privacy_info() == initial
    cameras["preview_camera"] = resource("open")
    opened = server._camera_privacy_info()
    cameras["preview_camera"] = resource("release_failed")
    failed = server._camera_privacy_info()
    cameras["preview_camera"] = resource("released")
    stopped = server._camera_privacy_info()
    assert initial["revision"] < opened["revision"] < failed["revision"] < stopped["revision"]
    assert failed["in_use"] is True and stopped["in_use"] is False
    assert opened["instance_id"] == failed["instance_id"] == stopped["instance_id"]


@pytest.mark.parametrize("failure", ["missing", "exception"])
def test_unknown_native_lifetime_cannot_claim_off(cameras, failure):
    def status():
        if failure == "exception":
            raise RuntimeError("Native status unavailable")
        return {"state": "off", "running": False}
    server._state["preview_camera"] = SimpleNamespace(status=status)
    info = server._camera_privacy_info()
    assert info["state"] == "unknown" and info["release_confirmed"] is False and info["in_use"] is None


def test_poll_is_metadata_only_and_never_starts_camera(cameras):
    cameras["preview_camera"] = resource("open")
    response = TestClient(server.app).get("/camera/privacy", headers=HEADERS)
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    assert response.json()["camera_privacy"]["owners"] == ["eeg"]
    assert "image" not in response.text and "waveform" not in response.text


@pytest.mark.parametrize("headers", [{}, {**HEADERS, "origin": "null"},
    {**HEADERS, "origin": "https://example.org"}, {**HEADERS, "host": "example.org"},
    {**HEADERS, "x-intuition-camera": "0"}])
def test_poll_rejects_browser_and_nonlocal_requests(cameras, headers):
    response = TestClient(server.app).get("/camera/privacy", headers=headers)
    assert response.status_code == 403 and response.headers["cache-control"] == "no-store"


def test_unconstructed_services_have_no_camera_owner(monkeypatch):
    monkeypatch.setattr(server, "_state", {})
    info = server._camera_privacy_info()
    assert info["state"] == "off" and info["release_confirmed"] is True and info["owners"] == []


@pytest.mark.parametrize("invalid", [
    {"state": [], "in_use": False, "release_confirmed": True},
    {"state": "released", "in_use": True, "release_confirmed": True},
    {"state": "released", "in_use": False, "release_confirmed": True, "cleanup_pending": True},
    {"state": "open", "in_use": False, "release_confirmed": False},
    {"state": "released", "in_use": False, "release_confirmed": "yes"},
])
def test_malformed_or_contradictory_metadata_becomes_unknown(cameras, invalid):
    cameras["preview_camera"] = invalid
    info = server._camera_privacy_info()
    assert info["state"] == "unknown" and info["release_confirmed"] is False
