"""CNN camera API reuses stopped tracker settings without acquiring input twice."""

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from interface import server


HEADERS = {"host": "127.0.0.1:7432", "x-intuition-multimodal": "1"}


class Camera:
    def __init__(self, *, ordinary=False):
        self.state = "off"
        self.tracker_cleanup_pending = False
        self.navigation_cleanup_pending = False
        self.camera_resource = {"state": "released", "in_use": False, "release_confirmed": True, "cleanup_pending": False}
        self.tracker_backend = "wilor" if ordinary else "mediapipe"
        self.model_complexity = 0 if ordinary else 1
        self.camera_index = 3 if ordinary else 0
        self.input_mode = "mouse" if ordinary else "desktop"
        self.bend_click = ordinary
        self.bindings = {"pinch": "click"} if ordinary else {}
        self.on_motion = (lambda *_: None) if ordinary else None
        self.calls = []
        self.configure_error = None

    def status(self):
        return {"state": self.state, "tracker_cleanup_pending": self.tracker_cleanup_pending,
                "camera_resource": self.camera_resource,
                "navigation_cleanup_pending": self.navigation_cleanup_pending,
                "tracker_backend": self.tracker_backend, "model_complexity": self.model_complexity,
                "input_mode": self.input_mode, "bend_click": self.bend_click}

    def is_running(self):
        return self.state == "running"

    def configure(self, **settings):
        assert self.state == "off", "Changing a live tracker is forbidden"
        self.calls.append(("configure", settings, threading.get_ident()))
        if self.configure_error:
            return {"error": self.configure_error}
        for key, value in settings.items():
            setattr(self, key, value)
        return self.status()

    def start(self):
        self.calls.append(("start",))
        self.state = "running"
        return self.status()

    def stop(self):
        self.calls.append(("stop",))
        self.state = "off"
        return self.status()


class Pointer:
    def __init__(self):
        self.bend_click = False
        self.calls = []
        self.error = None
        self.navigation_error = None
        self.navigation_settings = None

    def set_bend_click(self, enabled):
        self.calls.append(("set_bend_click", enabled, threading.get_ident()))
        if self.error:
            return {"error": self.error}
        self.bend_click = enabled
        return {"ok": True}

    def configure_navigation(self, *, mode, travel_palms):
        self.calls.append(("configure_navigation", mode, travel_palms, threading.get_ident()))
        if self.navigation_error:
            return {"error": self.navigation_error}
        self.navigation_settings = {"mode": mode, "travel_palms": travel_palms}
        return {"ok": True}


class Preview:
    def __init__(self):
        self.camera = Camera()
        self.pointer_controller = Pointer()
        self.state = "running"
        self.method = "cnn_blink_fast"
        self.camera_enabled = False
        self.armed = True
        self.calls = []
        self.request_epoch = 0

    def status(self):
        return {"state": self.state, "running": self.state == "running",
                "eeg": {"session_id": "unchanged-session"},
                "eeg_control": {"method": self.method, "armed": self.armed,
                    "camera_mouse_enabled": self.camera_enabled,
                    "saved_model_source": "fast-cnn-blink-existing.json", "passed_validation": True,
                    "train_counts": {"rest": 8, "select": 8}}}

    def camera_request_epoch(self):
        self.request_epoch += 1
        return self.request_epoch

    def camera_request_is_current(self, epoch):
        return epoch == self.request_epoch

    def eeg_camera(self, enabled, request_epoch=None):
        if request_epoch is not None and not self.camera_request_is_current(request_epoch):
            return {**self.status(), "error": "Superseded camera request"}
        self.calls.append(("eeg_camera", enabled, threading.get_ident()))
        self.armed = False
        if self.camera_enabled != enabled and self.state == "running":
            self.camera.start() if enabled else self.camera.stop()
        self.camera_enabled = enabled
        return self.status()

    def stop(self):
        self.camera_request_epoch()
        self.calls.append(("stop", threading.get_ident()))
        self.armed = False
        self.state = "stopped"
        self.camera.stop()
        return self.status()

    def start(self, request_epoch=None):
        if request_epoch is not None and not self.camera_request_is_current(request_epoch):
            return {**self.status(), "error": "Superseded camera request"}
        self.calls.append(("start", threading.get_ident()))
        if self.state != "running":
            self.state = "running"
            self.armed = False
            if self.camera_enabled:
                self.camera.start()
        return self.status()


@pytest.fixture
def api(monkeypatch):
    preview, ordinary = Preview(), Camera(ordinary=True)
    desktop = {"cleanup_pending": False}
    monkeypatch.setattr(server, "_state", {
        "multimodal": preview, "preview_camera": preview.camera, "gestures": ordinary,
        "hand_controls": SimpleNamespace(status=lambda: {"desktop": desktop}),
        "gesture_settings": {"desktop_mode": "shortcut", "travel_palms": 2.2},
        "gesture_lock": asyncio.Lock(),
    })
    monkeypatch.setattr(server.actions, "dispatch", lambda *_a, **_kw: pytest.fail("Duplicate action dispatch"))
    monkeypatch.setattr(server, "_broadcast", lambda *_a, **_kw: pytest.fail("Private operation broadcast"))
    return preview, ordinary, desktop, TestClient(server.app)


@pytest.mark.parametrize("operation", ["eeg_camera", "start"])
def test_camera_enable_uses_current_standalone_settings_and_existing_actuator(api, operation):
    preview, ordinary, _, client = api
    camera, pointer = preview.camera, preview.pointer_controller
    before = preview.status()
    if operation == "start":
        preview.state, preview.camera_enabled = "stopped", True
    # These may have changed since the preview camera was constructed.
    ordinary.tracker_backend, ordinary.camera_index = "rtmpose", 2
    response = client.post("/multimodal/" + operation, headers=HEADERS,
                           json={"enabled": True} if operation == "eeg_camera" else {})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    result = response.json()
    assert result["eeg"] == before["eeg"]
    for key in ("saved_model_source", "passed_validation", "train_counts"):
        assert result["eeg_control"][key] == before["eeg_control"][key]
    assert result["eeg_control"]["camera_mouse_enabled"] is True
    assert result["eeg_control"]["armed"] is False
    assert preview.camera is camera and preview.pointer_controller is pointer
    assert camera.tracker_backend == "rtmpose" and camera.model_complexity == 0
    assert camera.camera_index == 2 and camera.input_mode == "desktop"
    assert camera.bindings == {} and camera.on_motion is None
    assert pointer.bend_click is True and len(pointer.calls) == 2
    assert pointer.navigation_settings == {"mode": "shortcut", "travel_palms": 2.2}
    assert [call[0] for call in camera.calls] == ["configure", "start"]
    assert ordinary.calls == [] and ordinary.input_mode == "mouse"


def test_eeg_only_start_leaves_camera_and_pointer_settings_untouched(api):
    preview, ordinary, _, client = api
    preview.state = "stopped"
    response = client.post("/multimodal/start", headers=HEADERS, json={})
    assert response.status_code == 200
    assert not preview.camera.calls and not preview.pointer_controller.calls and not ordinary.calls


def test_live_idempotent_enable_preserves_tracker_even_if_standalone_settings_changed(api):
    preview, _, _, client = api
    preview.camera_enabled = True
    preview.camera.state = "running"
    response = client.post("/multimodal/eeg_camera", headers=HEADERS, json={"enabled": True})
    assert response.status_code == 200 and response.json()["eeg_control"]["armed"] is False
    assert [call[:2] for call in preview.calls] == [("eeg_camera", True)]
    assert not preview.camera.calls and not preview.pointer_controller.calls


@pytest.mark.parametrize("payload", [None, [], {}, {"enabled": 1}, {"enabled": "true"},
                                      {"enabled": True, "camera_index": 4}])
def test_camera_payload_is_exact_boolean(api, payload):
    preview, ordinary, _, client = api
    response = client.post("/multimodal/eeg_camera", headers=HEADERS, content=json.dumps(payload))
    assert response.status_code == 400
    assert not preview.calls and not ordinary.calls and not preview.camera.calls


@pytest.mark.parametrize("headers", [{}, {**HEADERS, "origin": "null"},
                                      {**HEADERS, "host": "example.org"}])
def test_camera_route_requires_local_private_protocol(api, headers):
    preview, _, _, client = api
    response = client.post("/multimodal/eeg_camera", headers=headers, json={"enabled": True})
    assert response.status_code == 403 and not preview.calls and not preview.camera.calls


@pytest.mark.parametrize("owner,field,value", [
    ("ordinary", "state", "running"), ("ordinary", "state", "starting"),
    ("ordinary", "state", "stopping"), ("ordinary", "tracker_cleanup_pending", True),
    ("ordinary", "navigation_cleanup_pending", True), ("desktop", "cleanup_pending", True),
    ("camera", "state", "running"), ("camera", "state", "stopping"),
    ("camera", "tracker_cleanup_pending", True), ("camera", "navigation_cleanup_pending", True),
    ("ordinary", "camera_resource", {"state": "release_failed", "in_use": True, "release_confirmed": False, "cleanup_pending": True}),
    ("camera", "camera_resource", {"state": "release_failed", "in_use": True, "release_confirmed": False, "cleanup_pending": True}),
    ("camera", "bindings", {"pinch": "click"}), ("camera", "on_click", lambda *_: None),
])
def test_enable_refuses_busy_or_nonisolated_input_before_configuration(api, owner, field, value):
    preview, ordinary, desktop, client = api
    target = {"ordinary": ordinary, "camera": preview.camera, "desktop": desktop}[owner]
    if isinstance(target, dict):
        target[field] = value
    else:
        setattr(target, field, value)
    response = client.post("/multimodal/eeg_camera", headers=HEADERS, json={"enabled": True})
    assert response.status_code == 409 and response.json()["error"]
    assert not preview.calls and not preview.camera.calls and not preview.pointer_controller.calls


@pytest.mark.parametrize("target,error", [("camera", "Tracker cleanup is unconfirmed"),
                                         ("pointer", "Button release is unconfirmed"),
                                         ("navigation", "Desktop input release is unconfirmed")])
def test_configuration_failure_does_not_start_camera_or_replace_eeg_model(api, target, error):
    preview, _, _, client = api
    if target == "camera":
        preview.camera.configure_error = error
    elif target == "navigation":
        preview.pointer_controller.navigation_error = error
    else:
        preview.pointer_controller.error = error
    before = preview.status()
    response = client.post("/multimodal/eeg_camera", headers=HEADERS, json={"enabled": True})
    assert response.status_code == 409 and response.json()["error"] == error
    assert not preview.calls and preview.status() == before
    assert not any(call[0] == "start" for call in preview.camera.calls)


@pytest.mark.parametrize("settings", [
    {"desktop_mode": "invalid", "travel_palms": 1.2},
    {"desktop_mode": "auto", "travel_palms": True},
    {"desktop_mode": "auto", "travel_palms": float("nan")},
    {"desktop_mode": "auto", "travel_palms": 0.2},
    {"desktop_mode": "auto", "travel_palms": 10},
])
def test_invalid_standalone_navigation_settings_cannot_start_combined_camera(api, settings):
    preview, _, _, client = api
    server._state["gesture_settings"] = settings
    response = client.post("/multimodal/eeg_camera", headers=HEADERS, json={"enabled": True})
    assert response.status_code == 409
    assert "desktop swipe configuration" in response.json()["error"]
    assert not preview.calls and not preview.camera.calls and not preview.pointer_controller.calls


def test_missing_navigation_controller_requires_restart_before_enabling(api, monkeypatch):
    preview, _, _, client = api
    monkeypatch.setattr(preview.pointer_controller, "configure_navigation", None)
    response = client.post("/multimodal/eeg_camera", headers=HEADERS, json={"enabled": True})
    assert response.status_code == 409 and "restart" in response.json()["error"]
    assert not preview.calls and not preview.camera.calls




def _request(payload):
    async def receive():
        return {"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}
    return Request({"type": "http", "method": "POST", "path": "/multimodal/eeg_camera",
                    "headers": [(key.encode(), value.encode()) for key, value in HEADERS.items()]}, receive)


@pytest.mark.parametrize("cancel", ["disable", "stop"])
def test_queued_enable_is_cancelled_by_responsive_disable_or_stop(api, cancel):
    preview, _, _, _ = api

    async def exercise():
        main_thread = threading.get_ident()
        lock = server._state["gesture_lock"] = asyncio.Lock()
        await lock.acquire()
        enable = asyncio.create_task(server.multimodal_operation("eeg_camera", _request({"enabled": True})))
        await asyncio.sleep(0)
        assert not enable.done() and not preview.calls
        # Stop-like disable remains responsive while another operation owns this lock.
        disabled = await asyncio.wait_for(server.multimodal_operation(
            "eeg_camera" if cancel == "disable" else "stop",
            _request({"enabled": False} if cancel == "disable" else {})), timeout=1)
        assert disabled.status_code == 200
        lock.release()
        enabled = await asyncio.wait_for(enable, timeout=1)
        assert enabled.status_code == 409 and "superseded" in json.loads(enabled.body)["error"]
        assert preview.camera_enabled is False and preview.camera.state == "off"
        assert not any(call[0] in ("configure", "start") for call in preview.camera.calls)
        assert all(call[-1] != main_thread for call in preview.calls)

    asyncio.run(exercise())


def test_camera_configuration_and_enable_run_off_the_event_loop(api):
    preview, _, _, _ = api

    async def exercise():
        main_thread = threading.get_ident()
        response = await server.multimodal_operation("eeg_camera", _request({"enabled": True}))
        assert response.status_code == 200
        assert preview.calls[0][-1] != main_thread
        assert preview.camera.calls[0][-1] != main_thread
        assert preview.pointer_controller.calls[0][-1] != main_thread

    asyncio.run(exercise())


def test_disable_during_settings_sync_prevents_late_enable(api, monkeypatch):
    preview, _, _, _ = api
    entered, release = threading.Event(), threading.Event()
    configure = preview.camera.configure

    def blocked_configure(**settings):
        entered.set()
        assert release.wait(2), "Camera disable could not finish during settings synchronization"
        return configure(**settings)

    monkeypatch.setattr(preview.camera, "configure", blocked_configure)

    async def exercise():
        enable = asyncio.create_task(server.multimodal_operation("eeg_camera", _request({"enabled": True})))
        assert await asyncio.to_thread(entered.wait, 1)
        try:
            disabled = await asyncio.wait_for(
                server.multimodal_operation("eeg_camera", _request({"enabled": False})), timeout=1)
            assert disabled.status_code == 200
        finally:
            release.set()
        enabled = await asyncio.wait_for(enable, timeout=1)
        assert enabled.status_code == 409 and "superseded" in json.loads(enabled.body)["error"]
        assert not preview.camera_enabled and preview.camera.state == "off"
        assert not any(call[0] == "start" for call in preview.camera.calls)
        assert [call[:2] for call in preview.calls] == [("eeg_camera", False)]

    asyncio.run(exercise())


@pytest.mark.parametrize("camera_enabled", [False, True])
def test_queued_start_cannot_restart_eeg_or_camera_after_stop(api, camera_enabled):
    preview, _, _, _ = api
    preview.state, preview.camera_enabled = "stopped", camera_enabled

    async def exercise():
        lock = server._state["gesture_lock"] = asyncio.Lock()
        await lock.acquire()
        start = asyncio.create_task(server.multimodal_operation("start", _request({})))
        await asyncio.sleep(0)
        assert not start.done() and not preview.calls
        stopped = await asyncio.wait_for(server.multimodal_operation("stop", _request({})), timeout=1)
        assert stopped.status_code == 200
        lock.release()
        response = await asyncio.wait_for(start, timeout=1)
        assert response.status_code == 409 and "superseded" in json.loads(response.body)["error"]
        assert preview.state == "stopped" and preview.camera.state == "off"
        assert not any(call[0] == "start" for call in preview.calls + preview.camera.calls)

    asyncio.run(exercise())
