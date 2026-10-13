"""High amplitudes reach the CNN; transport, input ownership and Escape still gate clicks."""

import copy
import importlib

import numpy as np
import pytest

from test_eeg_fast_blink import add, clone, trained_decoder
from test_eeg_fast_blink_control import fast, fast_hand


@pytest.fixture
def high_amplitude_hand(fast_hand, trained_decoder, monkeypatch):
    """Run real signal checks and detector state with a synthetic 40 mV waveform."""
    h = fast_hand
    decoder = clone(trained_decoder)
    for index in range(16, 32):
        add(decoder, index, "select" if index % 2 else "rest", phase="validate")
    assert decoder.status()["passed_validation"]
    h.eeg.acq["channels"] = copy.deepcopy(decoder.status()["context"]["channels"])
    decoder.bind_prediction_context({"session_id": h.eeg.acq["session_id"], "channels": h.eeg.acq["channels"],
                                     "nominal_hz": 250, "units": "V", "feature_profile": decoder.feature_profile})
    decoder.clock = h.clock
    # Fix only classifier scores: these tests prove input acceptance and output
    # arbitration, not the model's accuracy on unfamiliar high-amplitude data.
    h.score = {"value": .1}
    monkeypatch.setattr(importlib.import_module("core.eeg_fast_blink"), "_scores",
                        lambda *_: np.array([h.score["value"]]))
    raw_signal = h.eeg.acquisition_signal_window

    def scaled_signal():
        signal = raw_signal()
        for row in signal["samples"]:
            row["samples"] = [value * 20_000 for value in row["samples"]]
        return signal

    monkeypatch.setattr(h.eeg, "acquisition_signal_window", scaled_signal)
    h.decoder = h.service._decoder = decoder
    h.service._decoder_status = decoder.status()
    h.enable()
    h.tick(12)
    h.arm()
    h.tick(10)
    assert h.control()["can_eeg_click"]
    values = np.array([row["samples"] for row in scaled_signal()["samples"][-250:]])
    assert np.all(np.ptp(values, axis=0) > .01)  # All channels exceed the removed raw-voltage cutoff.
    return h


@pytest.mark.parametrize("show_hand", [True, False])
def test_high_amplitude_reaches_real_decoder_without_pausing_clicks(high_amplitude_hand, show_hand):
    h = high_amplitude_hand
    archive, moves = h.decoder.export_state(), len(h.pointer.moves)
    h.show_hand = show_hand
    h.score["value"] = .99
    h.tick(3)
    control = h.control()
    assert control["prediction"]["valid"] and control["prediction"]["fault_code"] is None
    assert control["armed"] and control["guard_ready"] and control["last_disarm_reason"] is None
    assert not control["signal_quality_paused"] and control["signal_quality_reason"] is None
    assert h.mouse.clicks == 1
    h.tick(10)
    assert h.mouse.clicks == 1  # Sustained CNN evidence cannot repeat an already consumed event.
    assert (len(h.pointer.moves) > moves) == show_hand
    assert h.decoder.export_state() == archive
    assert h.camera.stops == h.eeg.stops == 0


def test_high_amplitude_keeps_pinch_drag_and_finger_priority(high_amplitude_hand):
    h = high_amplitude_hand
    h.pointer.button_down = h.pointer.finger_active = True
    h.tick()
    h.score["value"] = .99
    h.tick(3)
    assert h.pointer.button_down and h.control()["armed"] and h.mouse.clicks == 0
    assert not h.service.status()["cleanup_pending"]
    h.pointer.button_down = h.pointer.finger_active = False
    h.pointer.click_id += 1
    h.tick()
    assert h.control()["finger_click_count"] == 1
    h.tick(12)
    assert h.mouse.clicks == 0  # A blink overlapping finger ownership is never replayed after release.
    h.score["value"] = .1
    h.tick(3)
    h.score["value"] = .99
    h.tick(3)
    assert h.mouse.clicks == 1 and h.control()["armed"]


@pytest.mark.parametrize("fault", ["stale", "gap", "session", "missing_full_signal", "processing_pause", "guard", "stop"])
def test_high_amplitude_does_not_bypass_transport_guard_or_lifecycle(high_amplitude_hand, monkeypatch, fault):
    h = high_amplitude_hand
    token = h.guard_token
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


@pytest.mark.parametrize("recoverable", [False, True, 1, None])
def test_invalid_prediction_always_disarms_even_if_marked_recoverable(fast, recoverable):
    h = fast
    h.arm()
    token = h.guard_token
    h.emit("invalid-evidence", valid=False, ready=False, fault_category="signal_quality",
           fault_code="signal_flat", recoverable=recoverable, reason="T3 is flat; check signal contact.")
    assert not h.control()["armed"] and not h.control()["signal_quality_paused"]
    assert h.service.eeg_guard(token).get("error")
    assert h.control()["last_disarm_reason"] == "T3 is flat; check signal contact."
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


def test_escape_during_high_amplitude_inference_cannot_be_overwritten(high_amplitude_hand, monkeypatch):
    h = high_amplitude_hand
    update = h.decoder.update

    def escape_during_update(signal):
        prediction = update(signal)
        h.service.eeg_arm(False)
        return prediction

    monkeypatch.setattr(h.decoder, "update", escape_during_update)
    h.score["value"] = .99
    h.tick()
    assert not h.control()["armed"] and not h.control()["signal_quality_paused"]
    assert h.control()["last_disarm_reason"] == "EEG control disarmed." and h.mouse.clicks == 0
