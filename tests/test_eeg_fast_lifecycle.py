"""Acquisition ownership, input cleanup and packet recovery after legacy removal."""
import threading

import pytest

from core.multimodal import MultimodalPreview
from fast_blink_fakes import Camera, Clock, EEG, FakeClick, FakePointer
from test_eeg_fast_blink_control import fast, fast_hand


def test_fixed_contract_defaults_camera_when_pointer_available_but_never_starts_or_arms():
    camera, eeg = Camera(), EEG()
    service = MultimodalPreview(camera, eeg, click_controller=FakeClick(), pointer_controller=FakePointer())
    control = service.status()["eeg_control"]
    assert service.status()["control_mode"] == "eeg_click"
    assert control["method"] == "cnn_blink_fast" and control["target"] == "mouse"
    assert control["camera_mouse_enabled"] and not control["armed"] and not control["decoder"]["trained"]
    assert "selection_count" not in control and "calibration" not in service.status() and "motion" not in service.status()
    for removed in ("arm", "set_control_mode", "eeg_method", "eeg_target", "check_contact", "mark_trial",
                    "reset_calibration", "eeg_export_model", "eeg_export_blink_model", "eeg_use_saved_blink_model"):
        assert not hasattr(service, removed)
    service.snapshot()
    assert camera.starts == camera.stops == 0 and eeg.starts == [] and eeg.stops == 0


def test_ordinary_hand_owner_and_external_eeg_session_are_never_stopped():
    camera, eeg = Camera(), EEG()
    service = MultimodalPreview(camera, eeg, lambda: True, pointer_controller=FakePointer())
    assert service.start().get("error")
    service.stop()
    assert camera.starts == camera.stops == eeg.stops == 0
    service.gesture_busy = lambda: False
    eeg.acq.update(state="running", mode="signal")
    assert "existing" in service.start()["error"]
    service.stop()
    assert eeg.stops == 0


@pytest.mark.parametrize("failed_device", ["eeg", "camera"])
def test_failed_start_cleans_only_owned_resources(failed_device, monkeypatch):
    camera, eeg = Camera(), EEG()
    service = MultimodalPreview(camera, eeg, pointer_controller=FakePointer())
    if failed_device == "eeg":
        monkeypatch.setattr(eeg, "start_acquisition", lambda **_: {"error": "EEG startup failed"})
    else:
        monkeypatch.setattr(camera, "start", lambda: {"error": "Camera startup failed"})
    assert service.start().get("error")
    assert eeg.stops == 1
    assert camera.stops == (failed_device == "camera")
    assert not service.status()["eeg_control"]["armed"]


@pytest.mark.parametrize("pending", [{"stopping": True}, {"camera_cleanup_pending": True},
                                      {"camera_resource": {"cleanup_pending": True, "in_use": True}}])
def test_unconfirmed_camera_cleanup_keeps_ownership_visible_and_blocks_restart(fast_hand, monkeypatch, pending):
    h = fast_hand
    h.enable()
    original = h.camera.stop
    monkeypatch.setattr(h.camera, "stop", lambda: {"ok": True, **pending})
    result = h.service.stop()
    assert result.get("error") and result["cleanup_pending"] and result["camera"]["owned"]
    assert h.service.start().get("error")
    monkeypatch.setattr(h.camera, "stop", original)
    assert h.service.stop().get("ok")
    assert not h.service.status()["camera"]["owned"] and not h.service.status()["cleanup_pending"]


def test_old_worker_failure_cannot_stop_a_restarted_session(fast):
    h = fast
    generation = h.service._generation
    h.service.stop()
    h.service.start()
    stops = h.eeg.stops
    h.service._fail("old worker", generation)
    assert h.service.status()["state"] == "starting" and h.eeg.stops == stops


def test_failure_rechecks_generation_after_waiting_for_lifecycle_lock(fast):
    h = fast
    h.service._operation_lock.acquire()
    thread = threading.Thread(target=h.service._fail, args=("old failure", h.service._generation))
    thread.start()
    try:
        assert h.service._halt.wait(2)
        with h.service._lock:
            h.service._generation += 1
            h.service._state = "starting"
            h.service._halt.clear()
    finally:
        h.service._operation_lock.release()
        thread.join(2)
    assert not thread.is_alive() and h.eeg.stops == 0 and h.service.status()["state"] == "starting"


def test_status_remains_responsive_while_native_camera_release_is_pending(fast_hand, monkeypatch):
    h = fast_hand
    h.enable()
    entered, release = threading.Event(), threading.Event()
    original = h.camera.stop

    def stop():
        entered.set()
        assert release.wait(3)
        return original()

    monkeypatch.setattr(h.camera, "stop", stop)
    thread = threading.Thread(target=h.service.stop)
    thread.start()
    try:
        assert entered.wait(2)
        status = h.service.status()
        assert status["state"] == "stopping" and status["camera"]["owned"]
        assert not status["eeg_control"]["armed"]
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive() and not h.service.status()["camera"]["owned"]


def test_camera_startup_allowance_starts_after_eeg_native_ack(monkeypatch):
    camera, eeg, clock = Camera(), EEG(), Clock()
    original = eeg.start_acquisition

    def delayed(mode):
        clock.now += 20
        return original(mode)

    monkeypatch.setattr(eeg, "start_acquisition", delayed)
    service = MultimodalPreview(camera, eeg, clock=clock, pointer_controller=FakePointer())
    service._launch_worker = lambda: None
    assert service.start().get("ok")
    assert service._started_at == service._camera_started_at == 120
    assert camera.starts == 1 and not service.status()["eeg_control"]["armed"]
    service.stop()


@pytest.mark.parametrize("state", ["error", "unavailable", "off", "stopping"])
def test_terminal_camera_failure_releases_devices_and_reports_original_reason(fast_hand, monkeypatch, state):
    h = fast_hand
    h.enable()
    monkeypatch.setattr(h.camera, "preview", lambda: {"running": False, "state": state, "hint": "capture failed"})
    h.tick()
    assert h.service.status()["state"] == "error" and "capture failed" in h.service.status()["error"]
    assert h.camera.stops == h.eeg.stops == 1 and not h.control()["armed"]


def test_camera_first_frame_timeout_and_stale_eeg_have_independent_budgets(fast_hand, monkeypatch):
    h = fast_hand
    h.enable()
    h.service._camera_ready = False
    h.service._camera_started_at = h.clock.now
    h.service.CAMERA_STARTUP_SECONDS = .6
    monkeypatch.setattr(h.camera, "preview", lambda: {"running": True, "state": "starting", "age_ms": 10000})
    h.tick(4)
    assert h.service.status()["state"] == "running" and h.eeg.stops == 0
    h.tick()
    assert "Camera startup timed out" in h.service.status()["error"]
    assert h.camera.stops == h.eeg.stops == 1


def publish(h, counter, *, gap=1, invalid=False):
    row = {"counter": counter, "estimated_monotonic": h.clock.now,
           "host_received_monotonic": h.clock.now, "samples": [1e-6, -2e-6, 3e-6, -4e-6]}
    if invalid:
        row["samples"][0] = None
    h.eeg.rows.append(row)
    h.tick(add_rows=False, gap=gap)


def test_counter_reset_reanchors_to_current_tail_and_requires_new_progress(fast):
    h = fast
    h.tick(2)
    assert h.service._last_valid_counter > 100
    publish(h, 0)
    assert h.service._last_valid_counter == 0 and not h.service._eeg_signal_healthy
    h.tick(8, add_rows=False, gap=1)
    assert not h.service._eeg_signal_healthy and h.service.snapshot()["eeg"]["waveform"] == []
    publish(h, 1)
    assert h.service._eeg_signal_healthy and h.service.snapshot()["eeg"]["waveform"][-1]["counter"] == 1


def test_malformed_reset_tail_does_not_use_retained_historical_rows(fast):
    h = fast
    publish(h, 0, invalid=True)
    assert h.service._eeg_counter_reanchor and not h.service._eeg_signal_healthy
    h.tick(8, add_rows=False, gap=1)
    assert h.service._eeg_counter_reanchor
    publish(h, 1)
    assert not h.service._eeg_counter_reanchor and not h.service._eeg_signal_healthy
    publish(h, 2)
    assert h.service._eeg_signal_healthy


def test_normal_uint32_wrap_advances_without_integrity_reset(fast):
    h = fast
    h.service._last_valid_counter = 2**32 - 2
    publish(h, 2**32 - 1, gap=0)
    assert h.service._eeg_signal_healthy
    publish(h, 0, gap=0)
    assert h.service._eeg_signal_healthy and h.service._last_valid_counter == 0


def test_backward_counter_without_reported_gap_cannot_refresh_stale_cache(fast):
    h = fast
    previous = h.service._last_valid_counter
    publish(h, 0, gap=0)
    h.tick(8, add_rows=False, gap=0)
    assert h.service._last_valid_counter == previous and not h.service._eeg_signal_healthy
    assert h.service.snapshot()["eeg"]["waveform"] == []
