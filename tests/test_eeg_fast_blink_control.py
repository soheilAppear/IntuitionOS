"""One-second CNN lifecycle and cue collection, without hardware or native input."""
import pytest
from core import multimodal
from fast_blink_fakes import FastHarness, CameraHarness, FakeFastCnnBlinkDecoder

@pytest.fixture
def fast(monkeypatch):
    return FastHarness(monkeypatch)

@pytest.fixture
def fast_hand(monkeypatch):
    return CameraHarness(monkeypatch)

def collecting(h, phase="train"):
    h.decoder.trained = phase == "validate"
    h.decoder.passed = h.decoder.eligible = False
    h.tick()


def test_fast_start_arm_event_stop_never_require_a_camera(fast):
    h = fast
    control = h.control()
    assert control["method"] == "cnn_blink_fast"
    assert control["window_seconds"] == control["trial_seconds"] == control["click_cooldown_seconds"] == 1
    assert control["decoder"]["validation_scope"] == "held_out_one_second_trials"
    assert control["target"] == "mouse" and not control["camera_required"]
    assert not control["hand_pointer"]["required"]
    h.arm()
    h.emit("fast-command")
    assert h.mouse.clicks == 1
    assert h.service.stop().get("ok")
    assert h.camera.calls == [] and h.eeg.stops == 1


@pytest.mark.parametrize("missing", ["trained", "passed", "eligible"])
def test_fast_has_no_training_or_validation_bypass(fast, missing):
    h = fast
    setattr(h.decoder, missing, False)
    h.tick()
    assert not h.control()["can_arm"]
    assert h.service.eeg_guard("invalid-fast").get("ok")
    assert h.service.eeg_arm(True, guard_token="invalid-fast").get("error")
    h.emit("not-validated")
    assert h.mouse.clicks == 0


@pytest.mark.parametrize("label", ["rest", "select"])
@pytest.mark.parametrize("phase", ["train", "validate"])
def test_fast_server_preparation_excludes_all_earlier_packets_from_exact_one_second_trial(fast, label, phase):
    h = fast
    collecting(h, phase)
    cue_at = h.clock.now
    assert h.service.eeg_trial(label, phase).get("ok")
    pending = h.control()["pending"]
    assert pending["stage"] == "prepare" and pending["prepare_remaining_seconds"] == 3
    assert pending["remaining_seconds"] == pending["capture_seconds"] == 1
    assert pending["total_remaining_seconds"] == 4 and pending["samples_required"] == 250
    assert "now" not in pending["cue"].lower()
    h.tick(23)
    assert h.control()["pending"]["stage"] == "prepare"
    assert h.control()["pending"]["samples_collected"] == 0 and h.decoder.trials == []
    h.tick()
    pending = h.control()["pending"]
    watermark = h.eeg.rows[-1]["counter"]
    assert pending["stage"] == "record" and pending["prepare_remaining_seconds"] == 0
    assert pending["samples_collected"] == 0 and pending["remaining_seconds"] == 1
    if label == "select":
        assert "now" in pending["cue"].lower()
    h.tick(7)
    assert len(h.decoder.trials) == 0
    h.tick()
    assert h.control()["pending"] is None
    assert len(h.decoder.trials) == 1
    trial = h.decoder.trials[0]
    rows = trial["signal"]["samples"]
    assert trial["start"] == cue_at + 3 and trial["end"] == cue_at + 4
    assert trial["end"] - trial["start"] == 1 and len(rows) == 250
    assert rows[0]["counter"] == watermark + 1 and rows[-1]["counter"] == watermark + 250
    assert rows[0]["estimated_monotonic"] == trial["start"]
    assert rows[-1]["estimated_monotonic"] == pytest.approx(trial["end"] - 1 / 250)
    assert h.camera.calls == [] and h.mouse.clicks == 0


@pytest.mark.parametrize("phase", ["train", "validate"])
def test_fast_negatives_cycle_only_actions_that_fit_a_one_second_clip(fast, phase):
    h = fast
    collecting(h, phase)
    for expected in ("quiet", "single_blink", "movement", "quiet"):
        assert h.service.eeg_trial("rest", phase).get("ok")
        assert h.control()["pending"]["cue_subtype"] == expected
        h.tick(32)
    assert len(h.decoder.trials) == 4
    assert all(len(trial["signal"]["samples"]) == 250 for trial in h.decoder.trials)


@pytest.mark.parametrize("stage", ["prepare", "record"])
@pytest.mark.parametrize("fault", ["gap", "stale", "stop"])
def test_fast_interrupted_cue_never_becomes_training_evidence(fast, stage, fault):
    h = fast
    collecting(h)
    assert h.service.eeg_trial("select").get("ok")
    h.tick(25 if stage == "record" else 2)
    assert h.control()["pending"]["stage"] == stage
    if fault == "stop":
        assert h.service.stop().get("ok")
    else:
        h.tick(**({"gap": 1} if fault == "gap" else {"age": 1.0}))
    assert h.control()["pending"] is None
    assert not h.control()["armed"] and h.decoder.trials == [] and h.mouse.clicks == 0


def test_fast_pending_preparation_cannot_change_model_or_camera(fast_hand):
    h = fast_hand
    collecting(h)
    assert h.service.eeg_trial("rest").get("ok")
    assert h.service.eeg_camera(True).get("error")
    assert not hasattr(h.service, "eeg_method")
    assert h.service._decoder is h.decoder and h.camera.starts == 0


@pytest.mark.parametrize("rows", [[], [{}], [{"counter": None}], [{"counter": True}], [{"counter": -1}]])
def test_fast_recording_boundary_discards_malformed_or_empty_snapshot_without_preview_failure(fast, monkeypatch, rows):
    h = fast
    collecting(h)
    assert h.service.eeg_trial("select").get("ok")
    signal = h.eeg.acquisition_signal_window()
    signal["samples"] = rows
    h.clock.now += 3
    monkeypatch.setattr(h.decoder, "update", lambda _: {**h.decoder.rest(), "valid": False})
    # Normally _tick also rejects a malformed tail. This exercises the independent
    # collection boundary against a changed snapshot before decoder validation.
    h.service._process_blink(signal, h.clock.now, h.service._generation)
    assert h.service.status()["state"] == "running"
    assert h.control()["pending"] is None and h.decoder.trials == []
    assert "discarded" in h.control()["last_result"].lower()
    assert h.mouse.clicks == 0


def test_fast_click_cooldown_is_one_second_without_history_reset_or_queued_click(fast):
    h = fast
    h.arm()
    resets = h.decoder.stream_resets
    h.emit("first")
    h.tick(3)
    h.emit("during-cooldown")
    assert h.mouse.clicks == 1
    h.tick(3)
    h.emit("during-cooldown")
    assert h.mouse.clicks == 1  # Expiry never replays an event consumed during the cooldown.
    h.emit("new-after-one-second")
    assert h.mouse.clicks == 2 and h.decoder.stream_resets == resets


def test_fast_camera_aiming_and_no_hand_clicks_preserve_model_and_eeg_session(fast_hand):
    h = fast_hand
    decoder, session = h.decoder, h.eeg.acq["session_id"]
    h.service._model_source = "fast-cnn-blink-validated.json"
    h.enable()
    h.arm()
    h.tick(2)
    assert h.pointer.moves
    h.show_hand = False
    h.tick(2)
    h.emit("without-visible-hand")
    assert h.mouse.clicks == 1 and h.control()["armed"]
    assert h.service._decoder is decoder and h.eeg.acq["session_id"] == session
    assert h.control()["saved_model_source"] == "fast-cnn-blink-validated.json"
    assert h.service.eeg_camera(False).get("ok")
    assert h.eeg.starts == ["signal"] and h.eeg.stops == 0 and not h.control()["armed"]


@pytest.mark.parametrize("ownership", ["finger_active", "button_down", "click_id"])
def test_fast_pinch_and_drag_consume_blocked_events_and_soft_cancel_on_edges(fast_hand, ownership):
    h = fast_hand
    h.enable()
    h.arm()
    resets = h.decoder.stream_resets
    setattr(h.pointer, ownership, 1 if ownership == "click_id" else True)
    h.emit("finger-owned")
    assert h.mouse.clicks == 0 and not h.control()["can_eeg_click"]
    h.tick(3)
    assert h.decoder.stream_resets == resets
    h.pointer.finger_active = h.pointer.button_down = False
    h.tick(10)
    h.emit("finger-owned")
    assert h.mouse.clicks == 0
    h.emit("new-after-release")
    assert h.mouse.clicks == 1 and h.decoder.cancel_calls >= 1


def test_saved_fast_profile_binds_current_session_but_never_arms_or_retrains(fast):
    h = fast
    restored = FakeFastCnnBlinkDecoder()
    h.arm()
    assert h.service.eeg_use_saved_cnn_model(restored, source="fast-cnn-blink-saved.json").get("ok")
    assert h.service._decoder is restored and not h.control()["armed"]
    assert restored.bindings[-1]["feature_profile"] == FakeFastCnnBlinkDecoder.feature_profile
    assert restored.bindings[-1]["session_id"] == h.eeg.acq["session_id"]
    assert restored.train_calls == 0 and restored.passed
    assert h.service.eeg_export_cnn_model()["format"] == "intuitionos.eeg-fast-cnn-blink-model"
