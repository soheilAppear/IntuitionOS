"""Hand-control API tests without app startup, camera access or native input."""

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from interface import server


class JsonRequest:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


def preview_request(headers=None):
    if headers is None:
        headers = {"host": "127.0.0.1:7432", "x-intuition-preview": "1"}
    return Request({"type": "http", "method": "GET", "path": "/gestures/preview",
                    "headers": [(key.lower().encode(), value.encode())
                                for key, value in headers.items()],
                    "query_string": b"", "scheme": "http"})


@pytest.fixture
def hand_api(monkeypatch):
    calls = []
    settings = {"desktop_mode": "auto", "travel_palms": 1.8, "model_complexity": 1}

    class Gestures:
        state = "off"

        def status(self):
            return {"state": self.state, "running": self.state == "running",
                    "available": True, "text": self.state, "bindings": {}}

        def configure(self, **kwargs):
            calls.append(("travel", kwargs))
            return {"ok": True}

        def preview(self):
            calls.append(("preview",))
            return {"running": self.is_running(), "image": None, "tracked": False,
                    "landmarks": [], "pose": "none", "fps": 0, "hint": "Show your hand"}

        def is_running(self):
            return self.state == "running"

        def action_for(self, name):
            return ("os_window_state", {"state": "restore"})

        def stop(self):
            calls.append(("camera_stop",))
            self.state = "off"
            return {"ok": True}

    class Desktop:
        def configure(self, **kwargs):
            calls.append(("mode", kwargs))
            return {"ok": True}

    class Controls:
        desktop = Desktop()

        def status(self):
            return {"close": {"pending": False}, "desktop": {"active": False}}

        def handle(self, event, binding):
            calls.append(("action", event.name, binding, threading.get_ident()))
            return ("os_window_state", {"state": "restore"}, {"ok": True})

        def stop(self):
            calls.append(("controls_stop",))

    gestures, controls = Gestures(), Controls()
    monkeypatch.setattr(server, "_state", {
        "gestures": gestures, "hand_controls": controls,
        "gesture_lock": asyncio.Lock(), "gesture_settings": settings,
    })
    monkeypatch.setattr(server, "_clients", set())
    return SimpleNamespace(gestures=gestures, controls=controls, calls=calls, settings=settings)


@pytest.mark.parametrize("payload", [
    None, [], {}, {"desktop_mode": "auto"},
    {"desktop_mode": "native", "travel_palms": 1.8},
    {"desktop_mode": "auto", "travel_palms": True},
    {"desktop_mode": "auto", "travel_palms": "1.8"},
    {"desktop_mode": "auto", "travel_palms": 0.79},
    {"desktop_mode": "auto", "travel_palms": 3.01},
    {"desktop_mode": "auto", "travel_palms": float("nan")},
    {"desktop_mode": "auto", "travel_palms": 1.8, "unexpected": 1},
])
def test_hand_settings_reject_invalid_payload_before_mutating(hand_api, payload):
    response = asyncio.run(server.gesture_settings(JsonRequest(payload)))
    assert response.status_code == 400
    assert "error" in json.loads(response.body)
    assert hand_api.calls == []
    assert server._state["gesture_settings"] == hand_api.settings


@pytest.mark.parametrize("state", ["starting", "running", "stopping"])
def test_hand_settings_cannot_reset_a_live_camera_session(hand_api, state):
    hand_api.gestures.state = state
    response = asyncio.run(server.gesture_settings(JsonRequest({
        "desktop_mode": "shortcut", "travel_palms": 2.2,
    })))
    assert response.status_code == 409
    assert hand_api.calls == []


def test_hand_settings_apply_to_idle_controls_and_return_shared_status(hand_api):
    payload = {"desktop_mode": "shortcut", "travel_palms": 2.2}
    response = asyncio.run(server.gesture_settings(JsonRequest(payload)))
    assert hand_api.calls == [("travel", {"travel_palms": 2.2}), ("mode", {"mode": "shortcut"})]
    assert response["gestures"]["settings"] == {**payload, "model_complexity": 1}
    assert response["gestures"]["state"] == "off"


@pytest.mark.parametrize("complexity", [0, 1])
def test_optional_model_setting_reaches_recognizer_and_shared_status(hand_api, complexity):
    payload = {"desktop_mode": "auto", "travel_palms": 1.2, "model_complexity": complexity}
    response = asyncio.run(server.gesture_settings(JsonRequest(payload)))
    assert hand_api.calls == [("travel", {"travel_palms": 1.2, "model_complexity": complexity}),
                              ("mode", {"mode": "auto"})]
    assert response["gestures"]["settings"] == payload


@pytest.mark.parametrize("complexity", [True, False, 0.0, 1.0, "0", "1", None, -1, 2, [], {}])
def test_model_setting_requires_integer_zero_or_one_before_mutation(hand_api, complexity):
    response = asyncio.run(server.gesture_settings(JsonRequest({
        "desktop_mode": "auto", "travel_palms": 1.8, "model_complexity": complexity,
    })))
    assert response.status_code == 400
    assert hand_api.calls == []
    assert server._state["gesture_settings"] == hand_api.settings


def test_older_two_key_payload_preserves_previously_selected_light_model(hand_api):
    server._state["gesture_settings"]["model_complexity"] = 0
    response = asyncio.run(server.gesture_settings(JsonRequest({
        "desktop_mode": "shortcut", "travel_palms": 2.0,
    })))
    assert hand_api.calls[0] == ("travel", {"travel_palms": 2.0})
    assert response["gestures"]["settings"]["model_complexity"] == 0


@pytest.mark.parametrize("backend", ["mediapipe", "rtmpose", "wilor"])
def test_tracker_backend_reaches_recognizer_without_changing_legacy_model(hand_api, backend):
    payload = {"desktop_mode": "auto", "travel_palms": 1.8, "tracker_backend": backend}
    response = asyncio.run(server.gesture_settings(JsonRequest(payload)))
    assert hand_api.calls == [("travel", {"travel_palms": 1.8, "tracker_backend": backend}),
                              ("mode", {"mode": "auto"})]
    assert response["gestures"]["settings"] == {**payload, "model_complexity": 1}
    assert response["gestures"]["running"] is False


@pytest.mark.parametrize("backend", [None, True, False, 0, 1, "", "RTMPose", "WiLoR", [], {}])
def test_invalid_tracker_backend_cannot_mutate_controls(hand_api, backend):
    response = asyncio.run(server.gesture_settings(JsonRequest({
        "desktop_mode": "auto", "travel_palms": 1.8, "tracker_backend": backend,
    })))
    assert response.status_code == 400
    assert hand_api.calls == []
    assert server._state["gesture_settings"] == hand_api.settings


@pytest.mark.parametrize("state", ["starting", "running", "stopping"])
@pytest.mark.parametrize("backend", ["rtmpose", "wilor"])
def test_tracker_backend_cannot_change_while_camera_session_exists(hand_api, state, backend):
    hand_api.gestures.state = state
    response = asyncio.run(server.gesture_settings(JsonRequest({
        "desktop_mode": "auto", "travel_palms": 1.8, "tracker_backend": backend,
    })))
    assert response.status_code == 409
    assert hand_api.calls == []


@pytest.mark.parametrize("backend", ["rtmpose", "wilor"])
def test_legacy_settings_preserve_previously_selected_tracker_backend(hand_api, backend):
    server._state["gesture_settings"]["tracker_backend"] = backend
    response = asyncio.run(server.gesture_settings(JsonRequest({
        "desktop_mode": "shortcut", "travel_palms": 2.0, "model_complexity": 0,
    })))
    assert response["gestures"]["settings"]["tracker_backend"] == backend
    assert hand_api.calls[0] == ("travel", {"travel_palms": 2.0, "model_complexity": 0})


@pytest.mark.parametrize("mode", ["mouse", "desktop"])
def test_input_mode_changes_only_between_camera_sessions(hand_api, mode):
    response = asyncio.run(server.gesture_settings(JsonRequest({
        "desktop_mode": "auto", "travel_palms": 1.8, "input_mode": mode,
    })))
    assert response["gestures"]["settings"]["input_mode"] == mode
    assert hand_api.calls == [("controls_stop",),
                              ("travel", {"travel_palms": 1.8, "input_mode": mode}),
                              ("mode", {"mode": "auto"})]


@pytest.mark.parametrize("mode", [None, True, 0, "pointer", "", [], {}])
def test_invalid_input_mode_cannot_mutate_controls(hand_api, mode):
    response = asyncio.run(server.gesture_settings(JsonRequest({
        "desktop_mode": "auto", "travel_palms": 1.8, "input_mode": mode,
    })))
    assert response.status_code == 400
    assert hand_api.calls == []


def test_live_camera_cannot_switch_to_mouse_mode(hand_api):
    hand_api.gestures.state = "running"
    response = asyncio.run(server.gesture_settings(JsonRequest({
        "desktop_mode": "auto", "travel_palms": 1.8, "input_mode": "mouse",
    })))
    assert response.status_code == 409
    assert hand_api.calls == []


def test_ordinary_settings_preserve_selected_mouse_mode(hand_api):
    server._state["gesture_settings"]["input_mode"] = "mouse"
    response = asyncio.run(server.gesture_settings(JsonRequest({
        "desktop_mode": "auto", "travel_palms": 1.8,
    })))
    assert response["gestures"]["settings"]["input_mode"] == "mouse"
    assert hand_api.calls[0] == ("travel", {"travel_palms": 1.8})


@pytest.mark.parametrize("enabled", [False, True])
def test_bend_click_setting_reaches_recognizer_and_shared_status(hand_api, enabled):
    response = asyncio.run(server.gesture_settings(JsonRequest({
        "desktop_mode": "auto", "travel_palms": 1.8, "bend_click": enabled,
    })))
    assert response["gestures"]["settings"]["bend_click"] is enabled
    assert hand_api.calls == [("travel", {"travel_palms": 1.8, "bend_click": enabled}),
                              ("mode", {"mode": "auto"})]


@pytest.mark.parametrize("enabled", [None, 0, 1, "true", "false", [], {}])
def test_bend_click_setting_rejects_non_boolean_without_mutating_controls(hand_api, enabled):
    response = asyncio.run(server.gesture_settings(JsonRequest({
        "desktop_mode": "auto", "travel_palms": 1.8, "bend_click": enabled,
    })))
    assert response.status_code == 400
    assert hand_api.calls == []
    assert "bend_click" not in hand_api.settings


def test_omitted_bend_click_preserves_preference_and_live_changes_are_blocked(hand_api):
    hand_api.settings["bend_click"] = True
    response = asyncio.run(server.gesture_settings(JsonRequest({
        "desktop_mode": "auto", "travel_palms": 1.8,
    })))
    assert response["gestures"]["settings"]["bend_click"] is True
    assert hand_api.calls[0] == ("travel", {"travel_palms": 1.8})
    hand_api.calls.clear()
    hand_api.gestures.state = "running"
    response = asyncio.run(server.gesture_settings(JsonRequest({
        "desktop_mode": "auto", "travel_palms": 1.8, "bend_click": False,
    })))
    assert response.status_code == 409
    assert hand_api.calls == []
    assert server._state["gesture_settings"]["bend_click"] is True


def test_mouse_mode_cannot_dispatch_late_desktop_or_window_gestures(hand_api):
    hand_api.gestures.state = "running"
    hand_api.gestures.input_mode = "mouse"
    assert server._handle_gesture(SimpleNamespace(name="restore_window", at=1.0)) is None
    assert server._handle_gesture(SimpleNamespace(name="close_request", at=2.0)) is None
    assert hand_api.calls == []


def test_successful_mouse_click_gets_sound_and_discrete_feedback(hand_api, monkeypatch):
    hand_api.gestures.state = "running"
    hand_api.gestures.input_mode = "mouse"
    played, queued = [], []
    server._state["hand_click_sound"] = SimpleNamespace(play=played.append)
    monkeypatch.setattr(server.time, "monotonic", lambda: 100)
    monkeypatch.setattr(server, "_queue_hand_feedback", lambda loop, event: queued.append(event))
    event = {"type": "gesture_click", "id": "session:1", "source": "bend", "at": 100}
    server._queue_mouse_click(None, event)
    assert played == ["session:1"]
    assert queued == [event]
    assert hand_api.calls == []


@pytest.mark.parametrize("running,mode,at,source", [
    (False, "mouse", 100, "bend"), (True, "desktop", 100, "bend"),
    (True, "mouse", 98, "bend"), (True, "mouse", 101, "bend"),
    (True, "mouse", 100, "drag"),
])
def test_stale_or_inactive_mouse_click_feedback_is_not_played(hand_api, monkeypatch, running, mode, at, source):
    hand_api.gestures.state = "running" if running else "off"
    hand_api.gestures.input_mode = mode
    played = []
    server._state["hand_click_sound"] = SimpleNamespace(play=played.append)
    monkeypatch.setattr(server.time, "monotonic", lambda: 100)
    monkeypatch.setattr(server, "_queue_hand_feedback", lambda *args: pytest.fail("stale click feedback"))
    server._queue_mouse_click(None, {"id": "session:1", "source": source, "at": at})
    assert played == []


@pytest.mark.parametrize("payload", [None, {}, {"enabled": 1}, {"enabled": "false"},
                                     {"enabled": False, "extra": True}])
def test_sound_endpoint_rejects_invalid_settings(hand_api, payload):
    response = asyncio.run(server.gesture_sound(JsonRequest(payload)))
    assert response.status_code == 400
    assert hand_api.calls == []


def test_sound_can_be_muted_during_live_tracking_without_stopping_camera(hand_api):
    from core.hand_feedback import HandClickSound
    hand_api.gestures.state = "running"
    server._state["hand_click_sound"] = HandClickSound(player=lambda _: pytest.fail("settings played sound"))
    result = asyncio.run(server.gesture_sound(JsonRequest({"enabled": False})))
    assert result["click_sound"]["enabled"] is False
    assert hand_api.gestures.is_running()
    assert hand_api.calls == []
    assert server._gesture_info()["click_sound"]["enabled"] is False


def test_model_change_does_not_reconfigure_live_capture(hand_api):
    hand_api.gestures.state = "running"
    response = asyncio.run(server.gesture_settings(JsonRequest({
        "desktop_mode": "auto", "travel_palms": 1.8, "model_complexity": 0,
    })))
    assert response.status_code == 409
    assert hand_api.calls == []


def test_rejected_recognizer_configuration_preserves_shared_settings(hand_api, monkeypatch):
    monkeypatch.setattr(hand_api.gestures, "configure", lambda **kwargs:
                        {"error": "Camera is still stopping."})
    response = asyncio.run(server.gesture_settings(JsonRequest({
        "desktop_mode": "auto", "travel_palms": 1.8, "model_complexity": 0,
    })))
    assert response.status_code == 409
    assert hand_api.calls == []
    assert server._state["gesture_settings"]["model_complexity"] == 1


@pytest.mark.parametrize("state", ["off", "running"])
def test_preview_only_reads_snapshot_and_never_starts_camera(hand_api, state, monkeypatch):
    hand_api.gestures.state = state
    monkeypatch.setattr(server, "_executor", SimpleNamespace(
        submit=lambda *args, **kwargs: pytest.fail("Preview must not use model workers")))
    response = asyncio.run(server.gesture_preview(preview_request()))
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert json.loads(response.body)["running"] is (state == "running")
    assert hand_api.calls == [("preview",)]
    assert hand_api.gestures.state == state


def test_preview_preserves_matching_frame_and_landmark_metadata(hand_api, monkeypatch):
    snapshot = {"running": True, "image": "data:image/jpeg;base64,c25hcHNob3Q=",
                "landmarks": [[0.1, 0.2, 0.0]] * 21, "pose": "open_palm",
                "tracked": True, "fps": 24.5, "sequence": 7,
                "hint": "Move left for next desktop", "model_complexity": 1}
    monkeypatch.setattr(hand_api.gestures, "preview", lambda: snapshot)
    response = asyncio.run(server.gesture_preview(preview_request()))
    assert json.loads(response.body) == snapshot
    assert response.headers["cache-control"] == "no-store"


def test_preview_without_recognizer_is_unavailable_and_uncacheable(monkeypatch):
    monkeypatch.setattr(server, "_state", {})
    response = asyncio.run(server.gesture_preview(preview_request()))
    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"
    assert json.loads(response.body)["image"] is None


@pytest.mark.parametrize("headers", [
    {"host": "127.0.0.1:7432"},
    {"host": "127.0.0.1:7432", "x-intuition-preview": "0"},
    {"host": "127.0.0.1:7432", "x-intuition-preview": "1", "origin": "https://example.com"},
    {"host": "127.0.0.1:7432", "x-intuition-preview": "1", "origin": "null"},
    {"host": "127.0.0.1:7432", "x-intuition-preview": "1", "origin": ""},
    {"host": "attacker.example:7432", "x-intuition-preview": "1"},
    {"host": "127.0.0.1:7433", "x-intuition-preview": "1"},
    {"x-intuition-preview": "1"},
])
def test_preview_denies_browser_and_rebinding_requests_before_reading_frame(hand_api, headers):
    hand_api.gestures.state = "running"
    response = asyncio.run(server.gesture_preview(preview_request(headers)))
    assert response.status_code == 403
    assert response.headers["cache-control"] == "no-store"
    assert "image" not in json.loads(response.body)
    assert hand_api.calls == []


def test_preview_allows_native_localhost_request(hand_api):
    response = asyncio.run(server.gesture_preview(preview_request({
        "host": "localhost:7432", "x-intuition-preview": "1",
    })))
    assert response.status_code == 200
    assert hand_api.calls == [("preview",)]


def test_cors_preflight_cannot_authorize_website_camera_preview(hand_api):
    # No lifespan context: the API fixture supplies a fake recognizer, and no
    # camera or other backend services start during this ASGI integration test.
    client = TestClient(server.app, base_url="http://127.0.0.1:7432")
    try:
        preflight = client.options("/gestures/preview", headers={
            "Origin": "https://example.com", "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "X-Intuition-Preview",
        })
        assert preflight.status_code == 200
        response = client.get("/gestures/preview", headers={
            "Origin": "https://example.com", "X-Intuition-Preview": "1",
        })
        assert response.status_code == 403
        assert response.headers["cache-control"] == "no-store"
        assert hand_api.calls == []
        native = client.get("/gestures/preview", headers={"X-Intuition-Preview": "1"})
        assert native.status_code == 200
        assert native.headers["cache-control"] == "no-store"
        assert hand_api.calls == [("preview",)]
    finally:
        client.close()


def test_capture_callback_handles_target_on_capture_thread_before_feedback_queue(hand_api, monkeypatch):
    hand_api.gestures.state = "running"
    queued = []
    monkeypatch.setattr(server, "_queue_hand_feedback", lambda loop, message: queued.append(message))
    loop = SimpleNamespace(is_closed=lambda: False)
    event = SimpleNamespace(name="restore_window", at=1.0)
    server._queue_gesture(loop, event)
    assert hand_api.calls == [("action", "restore_window",
                               ("os_window_state", {"state": "restore"}), threading.get_ident())]
    assert queued[0]["type"] == "gesture"
    assert queued[0]["ok"] is True


def test_stopped_capture_cannot_dispatch_a_late_callback(hand_api, monkeypatch):
    queued = []
    monkeypatch.setattr(server, "_queue_hand_feedback", lambda loop, message: queued.append(message))
    server._queue_gesture(SimpleNamespace(is_closed=lambda: False), SimpleNamespace(name="restore_window"))
    assert hand_api.calls == []
    assert queued == []


def test_http_stop_and_hand_control_cleanup_do_not_use_the_model_executor(hand_api, monkeypatch):
    class ForbiddenExecutor:
        def submit(self, *args, **kwargs):
            raise AssertionError("Camera control must not wait for the model worker pool")

    monkeypatch.setattr(server, "_executor", ForbiddenExecutor())
    hand_api.gestures.state = "running"
    response = asyncio.run(server.set_gestures(JsonRequest({"enabled": False})))
    assert response.status_code == 200
    assert json.loads(response.body)["gestures"]["state"] == "off"
    assert hand_api.calls == [("camera_stop",), ("controls_stop",)]


def test_failed_native_cleanup_is_reported_after_camera_release(hand_api, monkeypatch):
    monkeypatch.setattr(hand_api.controls, "stop", lambda: {"error": "Native input is still releasing."})
    monkeypatch.setattr(hand_api.controls, "status", lambda: {"desktop": {"cleanup_pending": True}})
    hand_api.gestures.state = "running"
    response = asyncio.run(server.set_gestures(JsonRequest({"enabled": False})))
    payload = json.loads(response.body)
    assert response.status_code == 503
    assert "still releasing" in payload["error"]
    assert payload["gestures"]["state"] == "off"
    assert payload["gestures"]["desktop"]["cleanup_pending"]


def test_pending_native_cleanup_blocks_restart_and_settings_before_mutation(hand_api, monkeypatch):
    monkeypatch.setattr(hand_api.controls, "status", lambda: {"desktop": {"cleanup_pending": True}})
    response = asyncio.run(server.set_gestures(JsonRequest({"enabled": True})))
    assert response.status_code == 503
    response = asyncio.run(server.gesture_settings(JsonRequest({
        "desktop_mode": "auto", "travel_palms": 2.2, "input_mode": "mouse",
    })))
    assert response.status_code == 409
    assert hand_api.calls == []


def test_mode_change_preserves_settings_if_native_close_fails(hand_api, monkeypatch):
    monkeypatch.setattr(hand_api.controls, "stop", lambda: {"error": "Native input is still releasing."})
    response = asyncio.run(server.gesture_settings(JsonRequest({
        "desktop_mode": "shortcut", "travel_palms": 2.2, "input_mode": "mouse",
    })))
    assert response.status_code == 409
    assert hand_api.calls == []
    assert server._state["gesture_settings"] == hand_api.settings


def test_close_error_feedback_remains_a_failure(hand_api, monkeypatch):
    hand_api.gestures.state = "running"
    monkeypatch.setattr(hand_api.controls, "handle", lambda event, binding: (
        "os_close_window", {}, {"error": "The active window changed."}))
    message = server._handle_gesture(SimpleNamespace(name="close_confirm"))
    assert message["ok"] is False
    assert "active window changed" in message["text"]
    assert "closed" not in message["text"].lower()
