"""Recoverable amplitude artifacts pause EEG, never camera aiming or safety gates."""

import pytest

from test_eeg_fast_blink import trained_decoder
from test_eeg_fast_blink_control import fast, fast_hand


AMPLITUDE = {"valid": False, "ready": False, "ready_for_blink": False, "eligible_for_event": False,
             "fault_category": "signal_quality", "fault_code": "signal_amplitude", "recoverable": True,
             "reason": "O2: 10.29 mV peak-to-peak exceeds the 10 mV raw signal guard."}


def artifact(h, event_id="artifact-evidence", **changes):
    h.emit(event_id, **{**AMPLITUDE, **changes})


def ready_rest(h):
    h.emit(None, label="rest", ready=True, ready_for_blink=True, eligible_for_event=True,
           reason="Fresh CNN context and rest recognized.")


def test_amplitude_pause_keeps_camera_arm_guard_and_model_while_suppressing_all_eeg(fast_hand):
    h = fast_hand
    h.enable()
    h.arm()
    h.tick(2)
    token, decoder, session = h.guard_token, h.decoder, h.eeg.acq["session_id"]
    moves, resets, serial = len(h.pointer.moves), h.decoder.stream_resets, h.service._blink_reset_serial
    artifact(h)
    control = h.control()
    assert control["armed"] and control["guard_ready"] and control["signal_quality_paused"]
    assert not control["can_eeg_click"] and control["signal_quality_reason"] == AMPLITUDE["reason"]
    assert control["last_disarm_reason"] is None and "paused" in control["reason"].lower()
    assert h.service._blink_reset_serial > serial and h.service._blink_dispatch_id is None
    for _ in range(12):
        artifact(h)
    assert len(h.pointer.moves) > moves and h.control()["armed"]
    assert h.service.eeg_guard(token).get("ok") and not h.service.status()["cleanup_pending"]
    assert h.decoder.stream_resets == resets  # The detector handles its fault; service adds no second reset/wait.
    assert h.service._decoder is decoder and h.eeg.acq["session_id"] == session
    assert h.camera.stops == h.eeg.stops == 0 and h.mouse.clicks == 0


def test_pause_survives_refill_and_select_until_fresh_recognized_rest_then_accepts_only_new_event(fast):
    h = fast
    h.arm()
    artifact(h)
    h.emit(None, label="rest", ready=False, ready_for_blink=False, eligible_for_event=False,
           reason="Collecting fresh CNN context: 0.5 / 1.0 seconds.")
    assert h.control()["armed"] and h.control()["signal_quality_paused"]
    assert not h.control()["can_eeg_click"] and "0.5 / 1.0" in h.control()["reason"]
    h.emit("queued-during-recovery", ready=True, ready_for_blink=False, eligible_for_event=True)
    assert h.mouse.clicks == 0 and h.control()["signal_quality_paused"]
    ready_rest(h)
    assert not h.control()["signal_quality_paused"] and h.control()["signal_quality_reason"] is None
    assert h.control()["can_eeg_click"] and h.control()["armed"]
    h.emit("queued-during-recovery")
    assert h.mouse.clicks == 0
    h.emit("new-after-recovery")
    assert h.mouse.clicks == 1 and h.control()["last_disarm_reason"] is None


def test_existing_drag_and_new_finger_clicks_remain_authorized_during_amplitude_pause(fast_hand):
    h = fast_hand
    h.enable()
    h.arm()
    h.pointer.button_down = h.pointer.finger_active = True
    h.tick()
    artifact(h)
    assert h.pointer.button_down and h.control()["armed"]
    assert not h.service.status()["cleanup_pending"]
    h.pointer.button_down = h.pointer.finger_active = False
    h.pointer.click_id += 1
    h.tick()
    assert h.control()["finger_click_count"] == 1
    assert h.control()["signal_quality_paused"] and h.mouse.clicks == 0


def test_temporary_hand_loss_during_artifact_does_not_revoke_eeg_authorization(fast_hand):
    h = fast_hand
    h.enable()
    h.arm()
    artifact(h)
    h.show_hand = False
    h.tick(3)
    assert h.control()["armed"] and h.control()["signal_quality_paused"]
    ready_rest(h)
    h.emit("recovered-no-hand")
    assert h.mouse.clicks == 1 and not h.control()["hand_pointer"]["ready"]


@pytest.mark.parametrize("changes", [{"recoverable": False}, {"recoverable": 1},
                                      {"fault_code": "signal_flat"}, {"fault_code": "signal_clipping"},
                                      {"fault_category": "transport"}, {"fault_category": None},
                                      {"valid": None}, {"fault_code": None}])
def test_only_exact_checked_amplitude_fault_may_retain_fast_arm(fast, changes):
    h = fast
    h.arm()
    token = h.guard_token
    artifact(h, **changes)
    assert not h.control()["armed"] and not h.control()["signal_quality_paused"]
    assert h.service.eeg_guard(token).get("error")
    assert h.control()["last_disarm_reason"] == AMPLITUDE["reason"] and h.mouse.clicks == 0



@pytest.mark.parametrize("fault", ["stale", "gap", "session", "missing_full_signal", "processing_pause", "guard", "stop"])
def test_transport_guard_and_lifecycle_faults_still_hard_disarm_during_quality_pause(fast_hand, monkeypatch, fault):
    h = fast_hand
    h.enable()
    h.arm()
    token = h.guard_token
    artifact(h)
    if fault == "stale":
        h.tick(age=1)
    elif fault == "gap":
        h.tick(gap=1)
    elif fault == "session":
        h.eeg.acq["session_id"] = "different-acquisition"
        h.tick()
    elif fault == "missing_full_signal":
        monkeypatch.setattr(h.eeg, "acquisition_signal_window", lambda: None)
        h.tick()
    elif fault == "processing_pause":
        h.tick(step=.5)
    elif fault == "guard":
        h.guard_active = False
        h.tick(25)
    else:
        assert h.service.stop().get("ok")
    assert not h.control()["armed"] and not h.control()["signal_quality_paused"]
    assert h.control()["last_disarm_reason"] and h.service.eeg_guard(token).get("error")
    assert h.mouse.clicks == 0


def test_hard_disarm_reason_survives_warmup_and_clears_only_after_successful_explicit_arm(fast):
    h = fast
    h.arm()
    h.emit("fault", valid=False, reason="T3 is flat; check signal contact.")
    reason = h.control()["last_disarm_reason"]
    assert reason == "T3 is flat; check signal contact."
    h.tick()  # Consume the fake detector's post-disarm prime before supplying its warmup result.
    h.emit(None, label="rest", ready=False, reason="Stream primed. Keep eyes open for fresh context.")
    assert h.control()["last_disarm_reason"] == reason and not h.control()["armed"]
    assert h.service.eeg_guard("still-unready").get("ok")
    assert h.service.eeg_arm(True, guard_token="still-unready").get("error")
    assert h.control()["last_disarm_reason"] == reason
    h.tick(2)
    h.arm()
    assert h.control()["last_disarm_reason"] is None and h.control()["armed"]


def test_escape_during_recoverable_decoder_update_cannot_be_overwritten_by_pause(fast):
    h = fast
    h.arm()
    h.decoder.on_update = lambda: h.service.eeg_arm(False)
    artifact(h)
    assert not h.control()["armed"] and not h.control()["signal_quality_paused"]
    assert h.control()["last_disarm_reason"] == "EEG control disarmed." and h.mouse.clicks == 0


def test_real_fast_decoder_artifact_recovery_and_service_pause_complete_without_rearming(fast_hand, trained_decoder, monkeypatch):
    import copy
    import importlib

    import numpy as np

    from test_eeg_fast_blink import add, clone

    h = fast_hand
    decoder = clone(trained_decoder)
    for index in range(16, 32):
        add(decoder, index, "select" if index % 2 else "rest", phase="validate")
    assert decoder.status()["passed_validation"]
    archive = decoder.export_state()
    h.eeg.acq["channels"] = copy.deepcopy(decoder.status()["context"]["channels"])
    decoder.bind_prediction_context({"session_id": h.eeg.acq["session_id"], "channels": h.eeg.acq["channels"],
                                     "nominal_hz": 250, "units": "V", "feature_profile": decoder.feature_profile})
    decoder.clock = h.clock
    # The real quality, transport, waveform refill, counter dwell and event
    # state run unchanged. Controlled scores isolate the service contract from
    # classifier accuracy, which has separate held-out/continuous tests.
    score = {"value": .1}
    monkeypatch.setattr(importlib.import_module("core.eeg_fast_blink"), "_scores", lambda *_: np.array([score["value"]]))
    h.decoder = h.service._decoder = decoder
    h.service._decoder_status = decoder.status()
    h.enable()
    h.tick(12)
    h.arm()
    h.tick(10)
    assert h.control()["can_eeg_click"]
    moves = len(h.pointer.moves)
    h.eeg.rows[-1]["samples"][0] += .02
    h.tick()
    assert h.control()["prediction"]["fault_code"] == "signal_amplitude"
    assert h.control()["armed"] and h.control()["signal_quality_paused"]
    score["value"] = .99
    h.tick(20)
    assert h.control()["prediction"]["ready"] and not h.control()["prediction"]["ready_for_blink"]
    assert h.control()["signal_quality_paused"] and h.mouse.clicks == 0
    score["value"] = .1
    h.tick(3)
    assert h.control()["prediction"]["ready_for_blink"] and not h.control()["signal_quality_paused"]
    score["value"] = .99
    h.tick(3)
    assert h.mouse.clicks == 1 and h.control()["armed"]
    h.tick(10)
    assert h.mouse.clicks == 1 and len(h.pointer.moves) > moves
    assert decoder.export_state() == archive and h.control()["last_disarm_reason"] is None
