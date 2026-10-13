"""Private preview API, camera exclusion, and responsive cancellation; no hardware."""
import asyncio
import json
import threading

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from interface import server


HEADERS = {"host": "127.0.0.1:7432", "x-intuition-multimodal": "1"}
GUARD_TOKEN = "bb215fc4-78fd-4ee5-b4f1-e35ca96f9416"


class Preview:
    """Record protocol calls without accessing hardware or injecting input."""
    def __init__(self):
        self.calls = []
        self.state = "stopped"
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.request_epoch = 0

    def status(self):
        return {"state": self.state, "running": self.state == "running",
                "eeg_control": {"method": "cnn_blink_fast", "target": "mouse", "armed": False}}

    def snapshot(self):
        return {**self.status(), "camera": {"image": "private-frame"},
                "eeg": {"waveform": [{"samples": [0.01]}]}}

    def camera_request_epoch(self):
        self.request_epoch += 1
        return self.request_epoch

    def camera_request_is_current(self, epoch):
        return epoch == self.request_epoch

    def start(self, request_epoch=None):
        assert request_epoch == self.request_epoch
        self.calls.append(("start", threading.get_ident()))
        self.state = "starting"
        if self.block:
            self.entered.set()
            assert self.release.wait(3), "Stop was queued behind Start"
        else:
            self.state = "running"
        return self.status()

    def stop(self):
        self.camera_request_epoch()
        self.calls.append(("stop", threading.get_ident()))
        self.state = "stopped"
        self.release.set()
        return self.status()

    def eeg_trial(self, label, phase):
        self.calls.append(("eeg_trial", label, phase))
        return self.status()

    def eeg_train(self):
        self.calls.append(("eeg_train",))
        return self.status()

    def eeg_reset(self):
        self.calls.append(("eeg_reset",))
        return self.status()

    def eeg_arm(self, enabled, guard_token=None):
        self.calls.append(("eeg_arm", enabled, guard_token))
        return self.status()

    def eeg_guard(self, token):
        self.calls.append(("eeg_guard", token))
        return self.status()

    def eeg_export_cnn_model(self):
        self.calls.append(("eeg_export_cnn_model",))
        return {"format": "test-only-fast-cnn"}

    def eeg_use_saved_cnn_model(self, decoder, *, source):
        self.calls.append(("eeg_use_saved_cnn_model", decoder, source))
        return {**self.status(), "eeg_control": {**self.status()["eeg_control"], "saved_model_source": source}}


@pytest.fixture
def api(monkeypatch):
    preview = Preview()
    monkeypatch.setattr(server, "_state", {"multimodal": preview, "gesture_lock": asyncio.Lock()})
    monkeypatch.setattr(server, "_clients", set())
    def unexpected(*args, **kwargs):
        pytest.fail("Preview/raw data must not enter action journal or broadcasts")
    monkeypatch.setattr(server.actions, "dispatch", unexpected)
    monkeypatch.setattr(server, "_broadcast", unexpected)
    return preview, TestClient(server.app)


@pytest.mark.parametrize("headers", [{}, {**HEADERS, "origin": "null"},
    {**HEADERS, "origin": "https://example.org"}, {**HEADERS, "host": "example.org"},
    {**HEADERS, "x-intuition-multimodal": "0"}])
def test_preview_and_controls_reject_browser_or_nonlocal_requests(api, headers):
    preview, client = api
    for path in ("status", "preview"):
        assert client.get(f"/multimodal/{path}", headers=headers).status_code == 403
    assert client.post("/multimodal/start", headers=headers, json={}).status_code == 403
    assert preview.calls == []


def test_raw_data_only_on_explicit_private_preview_get_and_no_auto_start(api):
    preview, client = api
    meta = client.get("/multimodal/status", headers=HEADERS)
    raw = client.get("/multimodal/preview", headers=HEADERS)
    assert meta.status_code == raw.status_code == 200
    assert "private-frame" not in meta.text and "waveform" not in meta.text
    assert raw.json()["camera"]["image"] == "private-frame"
    assert meta.headers["cache-control"] == raw.headers["cache-control"] == "no-store"
    assert preview.calls == []


def test_browser_cannot_renew_escape_lease_or_arm(api):
    preview, client = api
    for operation, payload in (("eeg_guard", {"token": GUARD_TOKEN}),
                               ("eeg_arm", {"enabled": True, "guard_token": GUARD_TOKEN})):
        assert client.post(f"/multimodal/{operation}", headers={**HEADERS, "origin": "null"},
                           json=payload).status_code == 403
    assert preview.calls == []


def request(path="/multimodal/start", payload=None):
    async def receive():
        return {"type": "http.request", "body": json.dumps(payload or {}).encode(), "more_body": False}
    return Request({"type": "http", "method": "POST", "scheme": "http", "path": path,
                    "query_string": b"", "headers": [(k.encode(), v.encode()) for k, v in HEADERS.items()]}, receive)


def test_stop_cached_status_and_health_remain_responsive_during_start(api):
    preview, _ = api
    preview.block = True
    async def exercise():
        start = asyncio.create_task(server.multimodal_operation("start", request()))
        assert await asyncio.to_thread(preview.entered.wait, 1)
        try:
            response = await asyncio.wait_for(server.multimodal_status(request("/multimodal/status")), 0.2)
            assert json.loads(response.body)["state"] == "starting"
            assert (await asyncio.wait_for(server.health(), 0.2))["ok"]
            stopped = await asyncio.wait_for(server.multimodal_operation("stop", request()), 1)
            assert json.loads(stopped.body)["state"] == "stopped"
            await asyncio.wait_for(start, 1)
            assert all(tid != threading.get_ident() for _, tid in preview.calls)
        finally:
            preview.release.set()
            await start
    asyncio.run(exercise())


@pytest.mark.parametrize("state", ["starting", "running", "stopping"])
def test_normal_camera_cannot_start_while_preview_owns_devices(api, state):
    preview, _ = api
    preview.state = state
    result = asyncio.run(server._change_gestures(True))
    assert result["status_code"] == 409
    assert preview.calls == []


def test_unavailable_service_is_reported_without_acquisition(api):
    _, client = api
    server._state.pop("multimodal")
    assert client.get("/multimodal/status", headers=HEADERS).json()["state"] == "unavailable"
    assert client.post("/multimodal/start", headers=HEADERS, json={}).status_code == 503


def test_connected_headset_alone_does_not_block_normal_camera(api):
    from types import SimpleNamespace
    preview, _ = api
    started = []
    server._state["brainbit"] = SimpleNamespace(status=lambda: {"state": "connected"})
    server._state["gestures"] = SimpleNamespace(start=lambda: started.append(True) or {"ok": True})
    result = asyncio.run(server._change_gestures(True))
    assert result["status_code"] == 200
    assert started == [True]
    assert preview.calls == []


@pytest.mark.parametrize("operation,payload", [
    ("eeg_trial", {"label": "rest", "phase": "train"}),
    ("eeg_trial", {"label": "select", "phase": "validate"}),
    ("eeg_train", {}), ("eeg_reset", {}),
    ("eeg_guard", {"token": GUARD_TOKEN}),
    ("eeg_arm", {"enabled": True, "guard_token": GUARD_TOKEN}),
    ("eeg_arm", {"enabled": False}),
])
def test_retained_operations_are_explicit_private_metadata_only(api, operation, payload):
    preview, client = api
    response = client.post(f"/multimodal/{operation}", headers=HEADERS, json=payload)
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    arguments = {"eeg_train": (), "eeg_reset": (), "eeg_guard": (GUARD_TOKEN,)}.get(operation)
    if operation == "eeg_trial":
        arguments = (payload["label"], payload["phase"])
    if operation == "eeg_arm":
        arguments = (payload["enabled"], payload.get("guard_token"))
    assert preview.calls == [(operation, *arguments)]
    assert GUARD_TOKEN not in response.text and "waveform" not in response.text


@pytest.mark.parametrize("operation,payload", [
    ("start", {"record": True}), ("stop", []), ("eeg_trial", {"label": "rest"}),
    ("eeg_trial", {"label": "left", "phase": "train"}),
    ("eeg_trial", {"label": "right", "phase": "validate"}),
    ("eeg_trial", {"label": {}, "phase": "train"}),
    ("eeg_trial", {"label": "select", "phase": "test"}),
    ("eeg_trial", {"label": "select", "phase": "train", "click": True}),
    ("eeg_train", {"refit_validation": True}), ("eeg_reset", {"save": True}),
    ("eeg_guard", {}), ("eeg_guard", {"token": "bad"}), ("eeg_guard", {"token": []}),
    ("eeg_guard", {"token": GUARD_TOKEN, "enabled": True}),
    ("eeg_arm", {"enabled": True}), ("eeg_arm", {"enabled": 1}),
    ("eeg_arm", {"enabled": True, "guard_token": "bad"}),
    ("eeg_arm", {"enabled": True, "guard_token": GUARD_TOKEN, "force": True}),
    ("eeg_arm", {"enabled": False, "guard_token": GUARD_TOKEN}),
])
def test_invalid_payloads_have_no_side_effects(api, operation, payload):
    preview, client = api
    response = client.post(f"/multimodal/{operation}", headers=HEADERS, json=payload)
    assert response.status_code == 400 and preview.calls == []


def test_service_refusal_is_preserved_without_dispatch(api, monkeypatch):
    preview, client = api
    monkeypatch.setattr(preview, "eeg_train", lambda: {"error": "More training examples required"})
    response = client.post("/multimodal/eeg_train", headers=HEADERS, json={})
    assert response.status_code == 409 and response.json()["error"] == "More training examples required"
    assert preview.calls == []


def test_unreleased_eeg_camera_blocks_standalone_start_even_after_preview_error(api):
    from types import SimpleNamespace
    preview, _ = api
    preview.state = "error"
    server._state["preview_camera"] = SimpleNamespace(is_running=lambda: False, status=lambda: {
        "state": "error", "camera_resource": {"state": "release_failed", "in_use": True,
                                                "release_confirmed": False, "cleanup_pending": True}})
    result = asyncio.run(server._change_gestures(True))
    assert result["status_code"] == 409 and preview.calls == []
