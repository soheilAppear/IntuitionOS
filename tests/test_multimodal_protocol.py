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
    def __init__(self):
        self.calls = []
        self.state = "stopped"
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.method = "eyes_closed"  # Keep archive selection explicit while preserving legacy route defaults.

    def status(self):
        return {"state": self.state, "running": self.state == "running", "armed": False,
                "eeg_control": {"method": self.method, "armed": False}}

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

    def set_control_mode(self, mode):
        self.calls.append(("control_mode", mode))
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

    def eeg_target(self, target):
        self.calls.append(("eeg_target", target))  # Record destination changes without touching a real pointer.
        return self.status()  # Target changes return metadata only and do not arm output.

    def eeg_method(self, method):
        self.calls.append(("eeg_method", method))  # Method changes do not select an archive or touch hardware.
        self.method = method
        return self.status()

    def eeg_export_blink_model(self):
        self.calls.append(("eeg_export_blink_model",))
        return {"format": "test-only-blink-state"}  # Protocol tests substitute storage; schema validation has its own real-decoder tests.

    def eeg_export_blink_calibration(self):
        self.calls.append(("eeg_export_blink_calibration",))
        return {"format": "test-only-blink-calibration"}

    def eeg_use_saved_blink_calibration(self, decoder, *, source):
        self.calls.append(("eeg_use_saved_blink_calibration", decoder, source))
        return {**self.status(), "eeg_control": {"method": self.method, "armed": False, "saved_calibration_source": source}}

    def eeg_use_saved_blink_model(self, decoder, *, source):
        self.calls.append(("eeg_use_saved_blink_model", decoder, source))  # The signature intentionally admits no validation-bypass argument.
        return {**self.status(), "eeg_control": {"method": self.method, "armed": False, "saved_model_source": source}}

    def eeg_use_saved_model(self, decoder, *, use_unvalidated, source):
        self.calls.append(("eeg_use_saved_model", decoder, use_unvalidated, source))
        return {**self.status(), "eeg_control": {"armed": False, "saved_model_source": source}}

    def eeg_export_model(self):
        self.calls.append(("eeg_export_model",))
        return {"format": "test-only-archive"}


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


def test_saved_model_load_uses_fixed_directory_and_preserves_explicit_disarmed_state(api, monkeypatch, tmp_path):
    preview, client = api
    folder = tmp_path / "eeg-models"
    restored = object()
    reads = []
    monkeypatch.setattr(server, "_EEG_MODEL_DIRECTORY", folder)
    monkeypatch.setattr(server, "load_latest_decoder", lambda path: (reads.append(path) or restored, folder / "model-saved.json"))
    response = client.post("/multimodal/eeg_load_saved", headers=HEADERS, json={})
    assert response.status_code == 200
    assert reads == [folder]
    assert preview.calls == [("eeg_use_saved_model", restored, True, "model-saved.json")]
    assert response.json()["eeg_control"]["armed"] is False
    assert response.json()["eeg_control"]["saved_model_source"] == "model-saved.json"
    assert str(folder) not in response.text


def test_save_model_exports_only_to_fixed_directory_and_reports_basename(api, monkeypatch, tmp_path):
    preview, client = api
    folder = tmp_path / "eeg-models"
    writes = []
    monkeypatch.setattr(server, "_EEG_MODEL_DIRECTORY", folder)
    monkeypatch.setattr(server, "save_archive", lambda archive, path: writes.append((archive, path)) or path)
    original_status = preview.status
    monkeypatch.setattr(preview, "status", lambda: {**original_status(), "eeg_control": {"armed": False}})
    response = client.post("/multimodal/eeg_save", headers=HEADERS, json={})
    assert response.status_code == 200
    assert preview.calls == [("eeg_export_model",)]
    assert writes[0][0] == {"format": "test-only-archive"}
    assert writes[0][1].parent == folder
    assert writes[0][1].name.startswith("model-") and writes[0][1].suffix == ".json"
    assert response.json()["eeg_control"]["saved_model_file"] == writes[0][1].name
    assert str(folder) not in response.text
    assert client.get("/multimodal/status", headers=HEADERS).json()["eeg_control"]["saved_model_file"] == writes[0][1].name


def test_blink_load_reads_only_separate_archive_without_validation_bypass_or_arm(api, monkeypatch, tmp_path):
    preview, client = api
    preview.method = "double_blink"
    folder = tmp_path / "eeg-blink-models"
    restored, reads = object(), []
    monkeypatch.setattr(server, "_EEG_BLINK_MODEL_DIRECTORY", folder)
    monkeypatch.setattr(server, "load_latest_blink_decoder", lambda path: (reads.append(path) or restored, folder / "double-blink-saved.json"))
    monkeypatch.setattr(server, "load_latest_decoder", lambda *_: pytest.fail("Blink loading touched the original model archive"))
    server._state["eeg_saved_blink_model_file"] = "double-blink-older.json"

    response = client.post("/multimodal/eeg_load_saved", headers=HEADERS, json={})

    assert response.status_code == 200 and reads == [folder]
    assert preview.calls == [("eeg_use_saved_blink_model", restored, "double-blink-saved.json")]
    assert response.json()["eeg_control"] == {"method": "double_blink", "armed": False,
                                               "saved_model_source": "double-blink-saved.json"}
    assert str(folder) not in response.text
    assert "eeg_saved_blink_model_file" not in server._state


def test_blink_save_uses_distinct_folder_and_acknowledgment_follows_active_method(api, monkeypatch, tmp_path):
    preview, client = api
    preview.method = "double_blink"
    folder, writes = tmp_path / "eeg-blink-models", []
    monkeypatch.setattr(server, "_EEG_BLINK_MODEL_DIRECTORY", folder)
    monkeypatch.setattr(server, "save_blink_archive", lambda state, path: writes.append((state, path)) or path)
    monkeypatch.setattr(server, "save_archive", lambda *_: pytest.fail("Blink saving touched the original model archive"))
    server._state["eeg_saved_model_file"] = "model-original.json"

    response = client.post("/multimodal/eeg_save", headers=HEADERS, json={})

    assert response.status_code == 200 and preview.calls == [("eeg_export_blink_model",)]
    state, path = writes[0]
    assert state == {"format": "test-only-blink-state"}
    assert path.parent == folder and path.name.startswith("double-blink-") and path.suffix == ".json"
    assert response.json()["eeg_control"]["saved_model_file"] == path.name
    assert response.json()["eeg_control"]["method"] == "double_blink"
    assert str(folder) not in response.text
    assert client.post("/multimodal/eeg_method", headers=HEADERS, json={"method": "eyes_closed"}).status_code == 200
    assert client.get("/multimodal/status", headers=HEADERS).json()["eeg_control"]["saved_model_file"] == "model-original.json"
    assert client.post("/multimodal/eeg_method", headers=HEADERS, json={"method": "double_blink"}).status_code == 200
    assert client.get("/multimodal/preview", headers=HEADERS).json()["eeg_control"]["saved_model_file"] == path.name
    assert server._state["eeg_saved_model_file"] == "model-original.json"  # Blink work does not rewrite the user's existing saved-model acknowledgment.


def test_progress_save_reports_only_fixed_folder_basename_and_active_method(api, monkeypatch, tmp_path):
    preview, client = api
    preview.method = "double_blink"
    folder, writes = tmp_path / "eeg-blink-drafts", []
    monkeypatch.setattr(server, "_EEG_BLINK_CALIBRATION_DIRECTORY", folder)
    monkeypatch.setattr(server, "save_calibration_archive", lambda state, path: writes.append((state, path)) or path)
    monkeypatch.setattr(server, "save_blink_archive", lambda *_: pytest.fail("Saving progress wrote a trained model archive"))
    response = client.post("/multimodal/eeg_save_progress", headers=HEADERS, json={})
    assert response.status_code == 200 and preview.calls == [("eeg_export_blink_calibration",)]
    state, path = writes[0]
    assert state == {"format": "test-only-blink-calibration"}
    assert path.parent == folder and path.name.startswith("calibration-") and path.suffix == ".json"
    assert response.json()["eeg_control"]["saved_calibration_file"] == path.name
    assert str(folder) not in response.text and "test-only-blink-calibration" not in response.text
    assert "saved_model_file" not in response.json()["eeg_control"]
    preview.method = "eyes_closed"
    assert "saved_calibration_file" not in client.get("/multimodal/status", headers=HEADERS).json()["eeg_control"]


def test_progress_load_uses_only_canonical_folder_and_explicit_resume_method(api, monkeypatch, tmp_path):
    preview, client = api
    preview.method = "double_blink"
    folder, reads, restored = tmp_path / "eeg-blink-drafts", [], object()
    monkeypatch.setattr(server, "_EEG_BLINK_CALIBRATION_DIRECTORY", folder)
    monkeypatch.setattr(server, "load_latest_calibration", lambda path: (reads.append(path) or restored, folder / "calibration-saved.json"))
    monkeypatch.setattr(server, "load_latest_blink_decoder", lambda *_: pytest.fail("Loading progress selected a frozen model"))
    server._state["eeg_saved_calibration_file"] = "calibration-older.json"
    response = client.post("/multimodal/eeg_load_progress", headers=HEADERS, json={})
    assert response.status_code == 200 and reads == [folder]
    assert preview.calls == [("eeg_use_saved_blink_calibration", restored, "calibration-saved.json")]
    assert response.json()["eeg_control"] == {"method": "double_blink", "armed": False,
                                               "saved_calibration_source": "calibration-saved.json"}
    assert str(folder) not in response.text and "eeg_saved_calibration_file" not in server._state


@pytest.mark.parametrize("operation", ["eeg_save_progress", "eeg_load_progress"])
def test_progress_requests_reject_wrong_method_and_visible_storage_failure(api, monkeypatch, operation):
    preview, client = api
    assert client.post(f"/multimodal/{operation}", headers=HEADERS, json={}).status_code == 409
    assert preview.calls == []
    preview.method = "double_blink"

    def failure(*_args):
        raise ValueError("No usable calibration progress")

    monkeypatch.setattr(server, "load_latest_calibration", failure)
    monkeypatch.setattr(preview, "eeg_export_blink_calibration", failure)
    response = client.post(f"/multimodal/{operation}", headers=HEADERS, json={})
    assert response.status_code == 409 and "No usable calibration progress" in response.json()["error"]
    assert preview.calls == [] and response.json()["eeg_control"]["armed"] is False


@pytest.mark.parametrize("operation", ["eeg_save_progress", "eeg_load_progress"])
@pytest.mark.parametrize("payload", [{"path": "../outside.json"}, {"enabled": True}, {"force": True}, [], None])
def test_progress_routes_accept_only_empty_json_and_never_paths_or_arm_flags(api, operation, payload):
    preview, client = api
    preview.method = "double_blink"
    response = client.post(f"/multimodal/{operation}", headers=HEADERS, json=payload)
    assert response.status_code == 400 and preview.calls == []


@pytest.mark.parametrize("operation", ["eeg_save_progress", "eeg_load_progress"])
def test_progress_routes_reject_browser_requests_before_disk_access(api, monkeypatch, operation):
    preview, client = api
    preview.method = "double_blink"
    monkeypatch.setattr(server, "load_latest_calibration", lambda *_: pytest.fail("Browser request accessed calibration"))
    monkeypatch.setattr(server, "save_calibration_archive", lambda *_: pytest.fail("Browser request wrote calibration"))
    response = client.post(f"/multimodal/{operation}", headers={**HEADERS, "origin": "null"}, json={})
    assert response.status_code == 403 and preview.calls == []


@pytest.mark.parametrize("operation", ["eeg_load_saved", "eeg_save"])
def test_blink_storage_failure_is_visible_without_falling_back_to_original_model(api, monkeypatch, operation):
    preview, client = api
    preview.method = "double_blink"

    def failure(*_args):
        raise ValueError("Double-blink archive is invalid")

    monkeypatch.setattr(server, "load_latest_blink_decoder", failure)
    monkeypatch.setattr(preview, "eeg_export_blink_model", failure)
    monkeypatch.setattr(server, "load_latest_decoder", lambda *_: pytest.fail("Blink failure fell back to the original decoder"))
    response = client.post(f"/multimodal/{operation}", headers=HEADERS, json={})
    assert response.status_code == 409 and "Double-blink archive is invalid" in response.json()["error"]
    assert response.json()["eeg_control"]["armed"] is False and preview.calls == []


@pytest.mark.parametrize("operation,payload", [("eeg_method", {"method": "double_blink"}),
                                                ("eeg_load_saved", {}), ("eeg_save", {})])
def test_blink_controls_reject_browser_requests_before_decoder_or_disk_access(api, monkeypatch, operation, payload):
    preview, client = api
    preview.method = "double_blink"
    monkeypatch.setattr(server, "load_latest_blink_decoder", lambda *_: pytest.fail("Browser request reached the blink archive"))
    monkeypatch.setattr(server, "save_blink_archive", lambda *_: pytest.fail("Browser request wrote the blink archive"))
    response = client.post(f"/multimodal/{operation}", headers={**HEADERS, "origin": "null"}, json=payload)
    assert response.status_code == 403 and preview.calls == []


@pytest.mark.parametrize("operation", ["eeg_load_saved", "eeg_save"])
def test_model_storage_failures_are_visible_and_never_arm(api, monkeypatch, operation):
    preview, client = api
    def failure(*args):
        raise ValueError("No usable saved EEG model")
    monkeypatch.setattr(server, "load_latest_decoder", failure)
    monkeypatch.setattr(preview, "eeg_export_model", failure)
    response = client.post(f"/multimodal/{operation}", headers=HEADERS, json={})
    assert response.status_code == 409
    assert "No usable saved EEG model" in response.json()["error"]
    assert response.json()["armed"] is False
    assert preview.calls == []


@pytest.mark.parametrize("operation", ["eeg_load_saved", "eeg_save"])
def test_model_storage_routes_reject_browser_requests_before_accessing_disk(api, monkeypatch, operation):
    preview, client = api
    def unexpected(*args):
        pytest.fail("Untrusted model request reached disk")
    monkeypatch.setattr(server, "load_latest_decoder", unexpected)
    monkeypatch.setattr(server, "save_archive", unexpected)
    response = client.post(f"/multimodal/{operation}", headers={**HEADERS, "origin": "null"}, json={})
    assert response.status_code == 403
    assert preview.calls == []


@pytest.mark.parametrize("operation,payload", [("start", {"record": True}), ("stop", []),
    ("contact", None), ("arm", {}), ("arm", {"enabled": 1}),
    ("arm", {"enabled": True, "force": True}), ("calibrate", {"label": "up"}),
    ("calibrate", {}), ("reset_calibration", {"save": True}),
    ("eeg_load_saved", {"path": "C:/outside/model.json"}),
    ("eeg_load_saved", {"enabled": True}), ("eeg_save", {"path": "../model.json"})])
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


@pytest.mark.parametrize("operation,payload", [
    ("control_mode", {"mode": "eeg"}),
    ("control_mode", {"mode": "webcam"}),  # Preserve explicit access to the established camera workflow.
    ("control_mode", {"mode": "eeg_click"}),  # Allow the new binary rest/select mode through the private API.
    ("eeg_target", {"target": "practice"}),  # A practice selection changes no OS input.
    ("eeg_target", {"target": "mouse"}),  # Mouse output still needs separate service-level arming.
    ("eeg_method", {"method": "eyes_closed"}),  # Existing saved calibration remains independently selectable.
    ("eeg_method", {"method": "double_blink"}),  # Blink calibration never implicitly arms desktop input.
    ("eeg_trial", {"label": "rest", "phase": "train"}),
    ("eeg_trial", {"label": "left", "phase": "validate"}),
    ("eeg_trial", {"label": "right", "phase": "train"}),  # Legacy directional trials remain routable.
    ("eeg_trial", {"label": "select", "phase": "train"}),  # Binary training labels reach the active decoder's checks.
    ("eeg_trial", {"label": "select", "phase": "validate"}),  # Validation remains a separately explicit phase.
    ("eeg_train", {}), ("eeg_reset", {}),
    ("eeg_guard", {"token": GUARD_TOKEN}),
    ("eeg_arm", {"enabled": True, "guard_token": GUARD_TOKEN}),
    ("eeg_arm", {"enabled": False}),
])
def test_eeg_actions_are_explicit_private_metadata_only(api, operation, payload):
    preview, client = api
    response = client.post(f"/multimodal/{operation}", headers=HEADERS, json=payload)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    expected = {"eeg_train": (), "eeg_reset": (),  # Mode and target arguments are checked against their exact payloads below.
                "eeg_guard": (GUARD_TOKEN,)}.get(operation)
    if operation == "control_mode":  # Every admitted mode must be forwarded without rewriting it.
        expected = (payload["mode"],)  # Verify the explicit mode selected by this test case.
    if operation == "eeg_target":  # The target setter receives exactly one constrained destination.
        expected = (payload["target"],)  # No mouse coordinates or action options may appear.
    if operation == "eeg_method":
        expected = (payload["method"],)  # Forward only the exact profile, with no implicit model loading or validation permission.
    if operation == "eeg_trial":
        expected = (payload["label"], payload["phase"])
    if operation == "eeg_arm":
        expected = (payload["enabled"], payload.get("guard_token"))
    assert preview.calls == [(operation, *expected)]
    assert GUARD_TOKEN not in response.text
    assert "private-frame" not in response.text and "waveform" not in response.text


@pytest.mark.parametrize("operation,payload", [
    ("control_mode", {"mode": "thought"}), ("control_mode", {"mode": ["eeg"]}),
    ("eeg_method", {}), ("eeg_method", {"method": "blink"}),
    ("eeg_method", {"method": None}), ("eeg_method", {"method": True}),
    ("eeg_method", {"method": ["double_blink"]}), ("eeg_method", {"method": {}}),
    ("eeg_method", {"method": "double_blink", "enabled": True}),
    ("eeg_method", {"method": "double_blink", "use_unvalidated": True}),
    ("control_mode", {"mode": "eeg_click", "enabled": True}),  # Choosing the new mode must not also arm it.
    ("control_mode", {"mode": "eeg_mouse"}),  # Do not accept undeclared click-mode aliases.
    ("eeg_target", {}),  # Destination selection requires an explicit target.
    ("eeg_target", {"target": "desktop"}),  # Reject arbitrary destinations or semantic OS actions.
    ("eeg_target", {"target": "click"}),  # This endpoint selects a target; it cannot directly request a click.
    ("eeg_target", {"target": ["mouse"]}),  # Reject structured values in place of a target string.
    ("eeg_target", {"target": {}}),  # Reject object payloads without reaching the service.
    ("eeg_target", {"target": None}),  # Null does not mean a default mouse target.
    ("eeg_target", {"target": True}),  # Booleans cannot implicitly enable mouse output.
    ("eeg_target", {"target": "mouse", "x": 1}),  # A click may not be redirected to supplied coordinates.
    ("eeg_target", {"target": "mouse", "button": "right"}),  # The actuator remains single-left-click only.
    ("eeg_target", {"target": "practice", "enabled": True}),  # Selecting practice cannot bypass explicit arming.
    ("eeg_trial", {"label": "rest"}), ("eeg_trial", {"label": "up", "phase": "train"}),
    ("eeg_trial", {"label": "select", "phase": "test"}),  # Binary labels do not create an unvalidated third phase.
    ("eeg_trial", {"label": "select", "phase": "train", "click": True}),  # Trials cannot carry an actuator request.
    ("eeg_trial", {"label": "rest", "phase": "test"}),
    ("eeg_trial", {"label": {}, "phase": "train"}),
    ("eeg_train", {"refit_validation": True}), ("eeg_reset", {"save": True}),
    ("eeg_guard", {}), ("eeg_guard", {"token": "bad"}),
    ("eeg_guard", {"token": GUARD_TOKEN, "enabled": True}),
    ("eeg_guard", {"token": []}),
    ("eeg_arm", {"enabled": True}), ("eeg_arm", {"enabled": 1}),
    ("eeg_arm", {"enabled": True, "guard_token": "bad"}),
    ("eeg_arm", {"enabled": True, "guard_token": GUARD_TOKEN, "force": True}),
    ("eeg_arm", {"enabled": False, "guard_token": GUARD_TOKEN}),
])
def test_invalid_eeg_actions_have_no_side_effects(api, operation, payload):
    preview, client = api
    assert client.post(f"/multimodal/{operation}", headers=HEADERS, json=payload).status_code == 400
    assert preview.calls == []


def test_browser_cannot_renew_escape_lease_or_arm(api):
    preview, client = api
    for operation, payload in (("eeg_guard", {"token": GUARD_TOKEN}),
                               ("eeg_arm", {"enabled": True, "guard_token": GUARD_TOKEN})):
        assert client.post(f"/multimodal/{operation}", headers={**HEADERS, "origin": "null"},
                           json=payload).status_code == 403
    assert preview.calls == []


@pytest.mark.parametrize("headers", [{}, {**HEADERS, "origin": "null"},
    {**HEADERS, "origin": "https://example.org"}, {**HEADERS, "host": "example.org"}])
def test_browser_cannot_select_click_mode_target_or_trials(api, headers):
    preview, client = api  # Use the fake service so blocked requests cannot reach any hardware.
    for operation, payload in (("control_mode", {"mode": "eeg_click"}),
                               ("eeg_target", {"target": "mouse"}),
                               ("eeg_target", {"target": "practice"}),
                               ("eeg_trial", {"label": "select", "phase": "train"})):  # Protect every new route with the existing origin guard.
        response = client.post(f"/multimodal/{operation}", headers=headers, json=payload)  # Submit a well-shaped but untrusted request.
        assert response.status_code == 403  # Authorization must run before target, mode, or trial mutation.
        assert response.headers["cache-control"] == "no-store"  # Errors must not expose cached private control data.
    assert preview.calls == []  # No denied request reached the service or armed mouse output.


def test_click_target_service_refusal_is_returned_without_dispatch(api, monkeypatch):
    preview, client = api  # Route validation alone must not override the service's lifecycle checks.
    monkeypatch.setattr(preview, "eeg_target", lambda target: {"error": "Choose EEG click mode first"})  # Simulate a core-mode refusal.
    response = client.post("/multimodal/eeg_target", headers=HEADERS, json={"target": "mouse"})  # Use a syntactically valid target request.
    assert response.status_code == 409  # Keep service refusal distinct from successful destination selection.
    assert response.json()["error"] == "Choose EEG click mode first"  # Preserve the reason needed for the HUD to recover.
    assert preview.calls == []  # An error cannot fall through to an immediate click or another operation.


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
