"""Private preview API, camera exclusion, and responsive cancellation; no hardware."""
import asyncio
import json
import threading

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from interface import server


HEADERS = {"host": "127.0.0.1:7432", "x-intuition-multimodal": "1"}


class Preview:
    def __init__(self):
        self.calls = []
        self.state = "stopped"
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = False

    def status(self):
        return {"state": self.state, "running": self.state == "running", "armed": False}

    def snapshot(self):
        return {**self.status(), "camera": {"image": "private-frame"},
                "eeg": {"waveform": [{"samples": [0.01]}]}}

    def start(self):
        self.calls.append(("start", threading.get_ident()))
        self.state = "starting"
        if self.block:
            self.entered.set()
            assert self.release.wait(3), "Stop was queued behind Start"
        else:
            self.state = "running"
        return self.status()

    def stop(self):
        self.calls.append(("stop", threading.get_ident()))
        self.state = "stopped"
        self.release.set()
        return self.status()

    def check_contact(self):
        self.calls.append(("contact", threading.get_ident()))
        return self.status()

    def arm(self, enabled):
        self.calls.append(("arm", enabled))
        return {**self.status(), "armed": enabled}

    def mark_trial(self, label):
        self.calls.append(("calibrate", label))
        return self.status()

    def reset_calibration(self):
        self.calls.append(("reset",))
        return self.status()


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


@pytest.mark.parametrize("operation,payload", [("start", {"record": True}), ("stop", []),
    ("contact", None), ("arm", {}), ("arm", {"enabled": 1}),
    ("arm", {"enabled": True, "force": True}), ("calibrate", {"label": "up"}),
    ("calibrate", {}), ("reset_calibration", {"save": True})])
def test_invalid_operations_cannot_touch_sensors(api, operation, payload):
    preview, client = api
    response = client.post(f"/multimodal/{operation}", headers=HEADERS, json=payload)
    assert response.status_code == 400
    assert preview.calls == []


def test_explicit_actions_and_unknown_actions(api):
    preview, client = api
    for name, payload in (("start", {}), ("arm", {"enabled": True}),
                          ("calibrate", {"label": "left"}), ("arm", {"enabled": False}),
                          ("stop", {}), ("contact", {}), ("reset_calibration", {})):
        assert client.post(f"/multimodal/{name}", headers=HEADERS, json=payload).status_code == 200
    assert [c[0] for c in preview.calls] == ["start", "arm", "calibrate", "arm", "stop", "contact", "reset"]
    assert client.post("/multimodal/record", headers=HEADERS, json={}).status_code == 404


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


@pytest.mark.parametrize("state", ["starting", "running", "contact", "stopping"])
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
