"""Fast blink optional camera aiming, input ownership and lifecycle races."""
import threading
import pytest
from fast_blink_fakes import CameraHarness

@pytest.fixture
def hand(monkeypatch):
    return CameraHarness(monkeypatch)

def test_live_enable_disable_preserves_exact_model_eeg_session_and_source(hand):
    h = hand
    h.service._model_source = "cnn-blink-original.json"
    decoder, session = h.decoder, h.eeg.acq["session_id"]
    h.arm()
    old_guard = h.guard_token
    h.enable()
    assert not h.control()["armed"]
    assert h.service.eeg_guard(old_guard).get("error")
    assert h.control()["camera_mouse_enabled"] and h.control()["camera_required"]
    assert h.control()["hand_pointer"]["enabled"] and not h.control()["hand_pointer"]["required"]
    assert h.camera.starts == 1 and h.eeg.starts == ["signal"] and h.eeg.stops == 0
    assert h.service._decoder is decoder and h.eeg.acq["session_id"] == session
    assert h.control()["saved_model_source"] == "cnn-blink-original.json"
    assert h.service.eeg_camera(False).get("ok")
    h.tick()
    assert not h.control()["camera_mouse_enabled"] and not h.control()["camera_required"]
    assert h.camera.stops == 1 and h.eeg.stops == 0
    assert h.service._decoder is decoder and decoder.trained and decoder.passed
    assert decoder.train_calls == decoder.reset_calls == 0 and h.mouse.clicks == 0


def test_cnn_reuses_one_full_signal_snapshot_and_bounds_only_the_display(hand, monkeypatch):
    h = hand
    h.tick(44)
    calls = {"full": 0, "display": 0}
    original_full, original_display = h.eeg.acquisition_signal_window, h.eeg.acquisition_snapshot

    def full():
        calls["full"] += 1
        return original_full()

    def display():
        calls["display"] += 1
        return original_display()

    monkeypatch.setattr(h.eeg, "acquisition_signal_window", full)
    monkeypatch.setattr(h.eeg, "acquisition_snapshot", display)
    h.tick()
    assert calls == {"full": 1, "display": 0}
    assert len(h.decoder.updates[-1]["samples"]) == 1250
    waveform = h.service.snapshot()["eeg"]["waveform"]
    assert 1 <= len(waveform) <= 250
    assert waveform[-1]["counter"] == h.eeg.rows[-1]["counter"]
    assert h.service._last_valid_counter == h.eeg.rows[-1]["counter"]
    assert h.service.status()["eeg"]["stats"] == h.eeg.acq["stats"]


def test_missing_full_signal_still_reads_acquisition_status_and_disarms(hand, monkeypatch):
    h = hand
    h.arm()
    calls = []
    original = h.eeg.acquisition_snapshot
    monkeypatch.setattr(h.eeg, "acquisition_signal_window", lambda: None)
    monkeypatch.setattr(h.eeg, "acquisition_snapshot", lambda: (calls.append(True), original())[1])
    h.tick()
    assert calls == [True]
    assert not h.control()["armed"] and h.mouse.clicks == 0


def test_pointer_uses_camera_frame_published_during_eeg_snapshot_copy(hand, monkeypatch):
    h = hand
    h.enable()
    h.arm()
    original = h.eeg.acquisition_signal_window
    newer_points = [[.6, .5, 0] for _ in range(21)]

    def full():
        signal = original()
        h.camera.frame = {**h.camera.frame, "sequence": 99999, "landmarks": newer_points}
        return signal

    monkeypatch.setattr(h.eeg, "acquisition_signal_window", full)
    h.tick()
    assert h.pointer.updates[-1]["points"] == newer_points
    assert h.service._pointer_sequence == 99999


def test_camera_timing_is_finite_and_aligned_with_tracking_or_display_source(hand, monkeypatch):
    h = hand
    h.enable()
    original = h.camera.preview

    def preview():
        return {**original(), "timing": {"capture_read_ms": 3.2, "frame_wait_ms": 1,
                                         "inference_ms": 17.6, "frame_age_ms": 21.8,
                                         "dropped_frames": 4, "unknown": "not public"}}

    def tracking():
        return {**original(), "timing": {"capture_read_ms": 7, "frame_wait_ms": -1,
                                         "inference_ms": float("nan"), "frame_age_ms": float("inf"),
                                         "dropped_frames": 3.5, "unknown": "not public"}}

    monkeypatch.setattr(h.camera, "preview", preview)
    monkeypatch.setattr(h.camera, "tracking_snapshot", tracking, raising=False)
    h.tick()
    assert h.service.status()["camera"]["timing"] == {"capture_read_ms": 7}
    assert h.service.snapshot()["camera"]["timing"] == {"capture_read_ms": 3.2, "frame_wait_ms": 1,
                                                        "inference_ms": 17.6, "frame_age_ms": 21.8,
                                                        "dropped_frames": 4}


@pytest.mark.parametrize("value", [None, [], "bad", {"capture_read_ms": True, "dropped_frames": False},
                                   {"inference_ms": 10 ** 1000, "dropped_frames": -1}])
def test_invalid_camera_timing_cannot_break_status_or_leak_unvalidated_values(hand, value):
    h = hand
    h.service._camera["timing"] = value
    assert h.service.status()["camera"]["timing"] == {}


def test_same_state_toggle_disarms_without_restarting_either_device(hand):
    h = hand
    h.enable()
    h.arm()
    token = h.guard_token
    assert h.service.eeg_camera(True).get("ok")
    assert not h.control()["armed"] and h.service.eeg_guard(token).get("error")
    assert h.camera.starts == 1 and h.camera.stops == 0 and h.eeg.starts == ["signal"]
    assert h.pointer.closes >= 2 and h.mouse.clicks == 0


def test_same_controller_moves_pointer_then_accepts_camera_independent_cnn_click(hand):
    h = hand
    h.enable()
    assert not h.pointer.moves
    h.arm()
    h.tick(2)
    assert h.pointer.moves and h.control()["hand_pointer"]["ready"]
    h.emit("aimed-pair")
    assert h.mouse.clicks == 1
    assert h.control()["armed"]


def test_successful_cnn_click_keeps_full_stream_and_consumes_event(hand):
    h = hand
    h.enable()
    h.arm()
    resets, cancellations = h.decoder.stream_resets, h.decoder.cancel_calls
    h.emit("first-fast-cnn-click")
    h.tick(20)
    assert h.mouse.clicks == 1
    assert h.decoder.stream_resets == resets and h.decoder.cancel_calls == cancellations
    h.emit("first-fast-cnn-click")
    assert h.mouse.clicks == 1
    h.emit("new-cnn-click")
    assert h.mouse.clicks == 2 and h.decoder.stream_resets == resets


def test_cnn_finger_ownership_cancels_only_at_edges_while_context_keeps_updating(hand):
    h = hand
    h.enable()
    h.arm()
    resets, updates, cancellations = h.decoder.stream_resets, len(h.decoder.updates), h.decoder.cancel_calls
    h.pointer.button_down = True
    h.tick(8)
    assert h.decoder.cancel_calls == cancellations + 1 and h.decoder.stream_resets == resets
    h.emit("event-while-dragging")
    assert h.mouse.clicks == 0
    h.pointer.button_down = False
    h.tick()
    assert h.decoder.cancel_calls == cancellations + 2
    h.emit("event-during-cooldown")
    h.tick(20)
    assert h.decoder.cancel_calls == cancellations + 2 and h.decoder.stream_resets == resets
    assert len(h.decoder.updates) >= updates + 30
    h.emit("event-during-cooldown")
    assert h.mouse.clicks == 0  # An event consumed while blocked cannot become a queued click.
    h.emit("fresh-after-finger-control")
    assert h.mouse.clicks == 1


def test_pinch_approach_does_not_block_or_cancel_cnn_but_real_action_does(hand, monkeypatch):
    h = hand
    h.enable()
    h.arm()
    actual_action = False
    original = h.pointer.update

    def update(*args, **kwargs):
        return {**original(*args, **kwargs), "finger_active": True, "finger_action_active": actual_action}

    monkeypatch.setattr(h.pointer, "update", update)
    resets, cancellations = h.decoder.stream_resets, h.decoder.cancel_calls
    h.tick(20)
    assert h.control()["can_eeg_click"]
    assert h.decoder.stream_resets == resets and h.decoder.cancel_calls == cancellations
    h.emit("cnn-during-pinch-approach")
    assert h.mouse.clicks == 1
    actual_action = True
    h.emit("cnn-during-real-pinch")
    assert h.mouse.clicks == 1 and not h.control()["can_eeg_click"]
    assert h.decoder.cancel_calls == cancellations + 1


def test_owned_button_always_blocks_cnn_even_when_action_flag_is_false(hand, monkeypatch):
    h = hand
    h.enable()
    h.arm()
    original = h.pointer.update

    def update(*args, **kwargs):
        return {**original(*args, **kwargs), "finger_active": False, "finger_action_active": False, "button_down": True}

    monkeypatch.setattr(h.pointer, "update", update)
    h.emit("owned-button-wins")
    assert h.mouse.clicks == 0 and not h.control()["can_eeg_click"]
    assert h.decoder.cancel_calls == 1


def test_cnn_recovery_remains_armed_but_is_not_click_ready(hand):
    h = hand
    h.enable()
    h.arm()
    h.emit("suppressed-old-evidence", eligible_for_event=False, ready_for_blink=False,
           suppressed_evidence_remaining_seconds=3.0, reason="Waiting for fresh evidence after finger control.")
    assert h.control()["armed"] and not h.control()["can_eeg_click"]
    assert h.control()["prediction"]["suppressed_evidence_remaining_seconds"] == 3
    assert h.mouse.clicks == 0
    h.tick(20)
    h.emit("suppressed-old-evidence", eligible_for_event=True)
    assert h.mouse.clicks == 0
    h.emit("fresh-eligible-evidence", eligible_for_event=True)
    assert h.mouse.clicks == 1 and not h.control()["can_eeg_click"]


def test_real_near_pinch_does_not_cancel_cnn_history(hand):
    from core.hand_mouse import HandMouseController
    from test_hand_mouse import Device, hand as hand_pose

    h = hand
    device = Device()
    h.service.pointer_controller = HandMouseController(device_factory=lambda: device, eeg_compatible=True)
    points = hand_pose()
    original = h.camera.preview

    def preview():
        frame = original()
        frame["landmarks"] = points
        return frame

    h.camera.preview = preview
    h.enable()
    h.arm()
    h.tick(4)
    resets, cancellations = h.decoder.stream_resets, h.decoder.cancel_calls
    points = hand_pose("pinch")
    points[4] = [points[8][0] + .45 * .2, points[8][1], 0]
    h.tick(5)
    assert h.service._pointer_status["finger_active"]
    assert not h.service._pointer_status["finger_action_active"]
    assert h.decoder.stream_resets == resets and h.decoder.cancel_calls == cancellations
    h.emit("real-pinch-approach")
    assert h.mouse.clicks == 1


@pytest.mark.parametrize("no_target", ["hand_absent", "pointing_dwell"])
def test_missing_hand_or_pointing_pose_never_gates_cnn_click_or_resets_history(hand, no_target):
    h = hand
    h.enable()
    h.arm()
    h.tick()
    serial, resets = h.service._blink_reset_serial, h.decoder.stream_resets
    if no_target == "hand_absent":
        h.show_hand = False
    else:
        h.pointer.ready = False
    h.tick(3)
    assert h.control()["armed"] and h.control()["can_eeg_click"]
    assert h.service._blink_reset_serial == serial and h.decoder.stream_resets == resets
    h.emit("no-hand-needed")
    assert h.mouse.clicks == 1


def test_cnn_can_arm_without_visible_hand_with_camera_enabled(hand):
    h = hand
    h.show_hand = False
    h.enable()
    h.arm()
    h.emit("no-hand-at-arm")
    assert h.mouse.clicks == 1 and not h.pointer.moves


@pytest.mark.parametrize("invalid", ["trained", "passed", "eligible"])
def test_camera_aiming_cannot_bypass_cnn_validation(hand, invalid):
    h = hand
    h.enable()
    setattr(h.decoder, invalid, False)
    h.tick()
    assert not h.control()["can_arm"]
    assert h.service.eeg_guard("not-trained").get("ok")
    assert h.service.eeg_arm(True, guard_token="not-trained").get("error")
    h.tick()
    assert h.pointer.moves == [] and h.mouse.clicks == 0


@pytest.mark.parametrize("ownership", ["finger_active", "button_down", "click_id"])
def test_finger_input_consumes_cnn_event_and_requires_new_event_after_release(hand, ownership):
    h = hand
    h.enable()
    h.arm()
    setattr(h.pointer, ownership, 1 if ownership == "click_id" else True)
    h.emit("finger-owned")
    assert h.mouse.clicks == 0 and not h.control()["can_eeg_click"]
    h.pointer.finger_active = h.pointer.button_down = False
    h.tick(22)
    h.emit("new-after-release")
    assert h.mouse.clicks == 1


def test_tracking_loss_releases_finger_drag_and_cancels_pending_eeg_history(hand):
    h = hand
    h.enable()
    h.arm()
    h.pointer.button_down = True
    h.tick()
    serial = h.service._blink_reset_serial
    h.show_hand = False
    h.tick()
    assert not h.pointer.button_down and h.service._blink_reset_serial > serial
    assert h.control()["armed"] and not h.control()["hand_pointer"]["ready"]


def test_tracking_cleanup_failure_disarms_and_blocks_native_input(hand, monkeypatch):
    h = hand
    h.enable()
    h.arm()
    monkeypatch.setattr(h.pointer, "reset", lambda *args: {"error": "release failed"})
    h.show_hand = False
    h.tick()
    assert not h.control()["armed"] and h.service.status()["cleanup_pending"]
    h.emit("cleanup-failure")
    assert h.mouse.clicks == 0


def test_toggle_refuses_pending_trial_or_input_cleanup_without_camera_start(hand):
    h = hand
    h.service._decoder_status["state"] = "trained"
    assert h.service.eeg_trial("rest", phase="validate").get("ok")
    assert h.service.eeg_camera(True).get("error")
    h.service._eeg_pending = None
    h.service._pointer_cleanup_pending = True
    assert h.service.eeg_camera(True).get("error")
    assert h.camera.starts == 0 and h.eeg.stops == 0


def test_toggle_refuses_conflicting_controller_without_stopping_eeg(hand):
    h = hand
    h.service.gesture_busy = lambda: True
    assert h.service.eeg_camera(True).get("error")
    assert h.camera.starts == h.eeg.stops == 0
    assert not h.control()["armed"] and not h.control()["camera_mouse_enabled"]


def test_camera_choice_while_stopped_acquires_camera_only_on_explicit_start(hand):
    h = hand
    assert h.service.stop().get("ok")
    assert h.service.eeg_camera(True).get("ok")
    assert h.camera.starts == 0
    assert h.service.start().get("ok")
    h.tick()
    assert h.camera.starts == 1 and h.eeg.starts == ["signal", "signal"]


def test_stop_and_expired_guard_release_shared_pointer_ownership(hand):
    h = hand
    h.enable()
    h.arm()
    h.pointer.button_down = True
    h.tick()
    h.guard_active = False
    for _ in range(28):
        h.tick()
    assert not h.control()["armed"] and not h.pointer.button_down
    assert h.service.stop().get("ok")
    assert h.camera.stops == h.eeg.stops == 1 and h.mouse.clicks == 0


def test_camera_start_in_progress_cannot_arm_switch_method_or_fail_worker(hand, monkeypatch):
    h = hand
    entered, release = threading.Event(), threading.Event()
    original = h.camera.start

    def start():
        entered.set()
        assert release.wait(3)
        return original()

    monkeypatch.setattr(h.camera, "start", start)
    result = []
    thread = threading.Thread(target=lambda: result.append(h.service.eeg_camera(True)))
    thread.start()
    try:
        assert entered.wait(2)
        h.tick(3)
        assert h.service.status()["state"] == "running" and h.eeg.stops == 0
        assert h.service.eeg_guard("during-camera-start").get("ok")
        assert h.service.eeg_arm(True, guard_token="during-camera-start").get("error")
        assert not hasattr(h.service, "eeg_method")
        assert h.service.eeg_trial("rest", phase="validate").get("error")
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive() and result[0].get("ok")
    h.tick()
    assert h.service.status()["state"] == "running" and not h.control()["armed"]


def test_stop_during_camera_start_cleans_camera_and_cannot_rearm(hand, monkeypatch):
    h = hand
    entered, release = threading.Event(), threading.Event()
    original = h.camera.start

    def start():
        entered.set()
        assert release.wait(3)
        return original()

    monkeypatch.setattr(h.camera, "start", start)
    enable = threading.Thread(target=lambda: h.service.eeg_camera(True))
    stop = threading.Thread(target=h.service.stop)
    enable.start()
    try:
        assert entered.wait(2)
        stop.start()
        assert h.service._halt.wait(2)
    finally:
        release.set()
        enable.join(3)
        stop.join(3)
    assert not enable.is_alive() and not stop.is_alive()
    assert h.service.status()["state"] == "stopped"
    assert not h.service._owns_camera and not h.camera.running
    assert h.camera.stops == h.eeg.stops == 1 and h.mouse.clicks == 0


def test_camera_failure_disarms_and_reports_owned_cleanup(hand, monkeypatch):
    h = hand
    monkeypatch.setattr(h.camera, "start", lambda: {"error": "camera unavailable"})
    result = h.service.eeg_camera(True)
    assert result.get("error") and "camera unavailable" in result["error"]
    assert h.service.status()["state"] == "error" and not h.control()["armed"]
    assert h.camera.stops == h.eeg.stops == 1 and h.service._decoder is h.decoder


@pytest.mark.parametrize("cancel", ["disable", "stop"])
def test_admitted_but_delayed_enable_cannot_survive_later_revocation(hand, cancel):
    h = hand
    epoch = h.service.camera_request_epoch()
    assert h.service.camera_request_is_current(epoch)
    if cancel == "disable":
        assert h.service.eeg_camera(False).get("ok")
    elif cancel == "stop":
        assert h.service.stop().get("ok")
    assert not h.service.camera_request_is_current(epoch)
    assert h.service.eeg_camera(True, request_epoch=epoch).get("error")
    assert h.camera.starts == 0 and not h.service._owns_camera


def test_delayed_configured_start_cannot_reverse_stop(hand):
    h = hand
    assert h.service.stop().get("ok")
    assert h.service.eeg_camera(True).get("ok")
    epoch = h.service.camera_request_epoch()
    assert h.service.stop().get("ok")
    assert h.service.start(request_epoch=epoch).get("error")
    assert h.camera.starts == 0 and h.eeg.starts == ["signal"]


def test_configured_start_rechecks_admission_after_native_eeg_ack(hand, monkeypatch):
    h = hand
    assert h.service.stop().get("ok")
    assert h.service.eeg_camera(True).get("ok")
    epoch = h.service.camera_request_epoch()
    original = h.eeg.start_acquisition

    def start_acquisition(mode="signal"):
        result = original(mode)
        assert h.service.eeg_camera(False).get("error")  # Startup is busy, but the newer request still cancels admission.
        return result

    monkeypatch.setattr(h.eeg, "start_acquisition", start_acquisition)
    assert h.service.start(request_epoch=epoch).get("ok")
    assert h.service.status()["state"] == "stopped"
    assert h.camera.starts == 0 and h.eeg.stops == 2


def test_disable_during_native_start_cancels_camera_without_restarting_eeg(hand, monkeypatch):
    h = hand
    entered, release = threading.Event(), threading.Event()
    original = h.camera.start

    def start():
        entered.set()
        assert release.wait(3)
        return original()

    monkeypatch.setattr(h.camera, "start", start)
    result = []
    thread = threading.Thread(target=lambda: result.append(h.service.eeg_camera(True)))
    thread.start()
    try:
        assert entered.wait(2)
        assert h.service.eeg_camera(False).get("error")  # The running operation acknowledges cancellation after native return.
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive() and result[0].get("error")
    assert not h.camera.running and not h.control()["camera_mouse_enabled"]
    assert h.camera.starts == h.camera.stops == 1
    assert h.eeg.starts == ["signal"] and h.eeg.stops == 0 and h.service._decoder is h.decoder


def test_real_hand_and_eeg_actuators_share_aiming_blink_and_finger_priority(hand):
    from core.eeg_click import EegClickController
    from core.hand_mouse import HandMouseController
    from test_hand_mouse import Device, buttons, hand as hand_pose

    h = hand
    device = Device()
    pointer = HandMouseController(device_factory=lambda: device, eeg_compatible=True)
    h.service.pointer_controller = pointer
    h.service.click_controller = EegClickController(device_factory=lambda: device)
    pose, dx = "point", 0.0
    original_preview = h.camera.preview

    def preview():
        frame = original_preview()
        if frame.get("tracked"):
            frame["landmarks"] = hand_pose(pose, dx=dx)
        return frame

    h.camera.preview = preview
    h.enable()
    assert not device.events
    h.arm()
    h.tick(4)
    assert h.control()["hand_pointer"]["ready"]
    dx = .06
    h.tick(3)
    assert any(event[0] == "move" for event in device.events)
    h.emit("cnn-at-aimed-cursor")
    assert buttons(device) == ["down", "up"] and h.control()["click_count"] == 1
    pose = "pinch"
    h.tick(4)
    assert buttons(device) == ["down", "up", "down"]
    h.emit("cnn-during-drag")
    assert buttons(device) == ["down", "up", "down"]
    assert not h.control()["can_eeg_click"]
    pose = "point"
    h.tick()
    assert buttons(device) == ["down", "up", "down", "up"]
    h.tick(23)
    h.show_hand = False
    h.tick()
    h.emit("cnn-without-hand")
    assert buttons(device) == ["down", "up", "down", "up", "down", "up"]
    assert h.control()["click_count"] == 2
    assert h.service.stop().get("ok")
    assert len(buttons(device)) == 6
