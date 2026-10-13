"""Preserved input safety cases from the superseded model controllers."""
import math
import pytest
from fast_blink_fakes import FastHarness, FakeFastCnnBlinkDecoder, FakePointer

@pytest.fixture
def cnn(monkeypatch):
    return FastHarness(monkeypatch)

def test_cnn_start_arm_event_and_stop_require_no_camera_or_hand(cnn):
    h = cnn
    assert h.service.status()["state"] == "running"
    assert h.control()["method"] == "cnn_blink_fast"
    assert h.control()["target"] == "mouse"
    assert not h.control()["hand_pointer"]["required"]
    assert not h.control()["camera_required"]
    assert h.control()["can_arm"] and not h.control()["armed"]
    h.service.CAMERA_STARTUP_SECONDS = h.service.CAMERA_LOSS_SECONDS = .5
    h.tick(8)
    h.arm()
    h.emit("cnn-pair")
    assert h.mouse.clicks == h.control()["click_count"] == 1
    assert h.service.stop().get("ok")
    assert h.camera.calls == [] and h.eeg.stops == 1


@pytest.mark.parametrize("missing", ["trained", "passed", "eligible"])
def test_cnn_cannot_arm_without_training_and_successful_held_out_validation(cnn, missing):
    h = cnn
    setattr(h.decoder, missing, False)
    h.tick()
    assert not h.control()["can_arm"]
    assert h.service.eeg_guard("unvalidated-cnn-guard").get("ok")
    assert h.service.eeg_arm(True, guard_token="unvalidated-cnn-guard").get("error")
    h.emit("unvalidated-pair")
    assert h.mouse.clicks == 0 and not h.control()["armed"]


def test_cnn_freezing_disarms_and_still_requires_separate_validation(cnn):
    h = cnn
    h.arm()
    old_guard = h.guard_token
    assert h.service.eeg_train().get("ok")
    assert h.decoder.train_calls == 1
    assert not h.control()["armed"] and not h.control()["can_arm"]
    assert h.service.eeg_guard(old_guard).get("error")
    assert h.mouse.clicks == 0 and h.camera.calls == []


def test_cnn_consumes_disarmed_and_buffered_events_without_replay(cnn):
    h = cnn
    h.emit("old-pair")
    assert h.mouse.clicks == 0
    h.decoder.event("buffered-before-arm", h.clock.now)
    h.arm()
    assert h.mouse.clicks == 0
    h.emit("old-pair")
    assert h.mouse.clicks == 0
    h.emit("fresh-pair")
    assert h.mouse.clicks == 1
    h.tick(20)
    h.emit("fresh-pair")
    assert h.mouse.clicks == 1


@pytest.mark.parametrize("extra", [{"valid": False}, {"ready": False}, {"label": "rest"},
                                   {"label": "left"}, {"event_id": None}])
def test_cnn_rejects_invalid_unready_or_noncommand_predictions(cnn, extra):
    h = cnn
    h.arm()
    options = dict(extra)
    event_id = options.pop("event_id", "invalid-pair")
    h.emit(event_id, **options)
    assert h.mouse.clicks == 0
    if options.get("valid") is False:
        assert not h.control()["armed"]
    if options.get("ready") is False:
        assert h.control()["armed"]  # Valid post-reset warmup may retain explicit authorization.


@pytest.mark.parametrize("interruption", ["stale", "gap", "session", "processing_pause", "missing_full_signal"])
def test_cnn_stream_interruptions_disarm_but_retain_trained_model(cnn, monkeypatch, interruption):
    h = cnn
    h.arm()
    old_guard = h.guard_token
    h.decoder.event("interrupted-pair", h.clock.now)
    if interruption == "session":
        h.eeg.acq["session_id"] = "replacement-session"
        h.tick()
    elif interruption == "processing_pause":
        h.tick(step=.5)
    elif interruption == "missing_full_signal":
        monkeypatch.setattr(h.eeg, "acquisition_signal_window", lambda: None)
        h.tick()
    else:
        h.tick(**({"age": 1.0} if interruption == "stale" else {"gap": 1}))
    assert not h.control()["armed"]
    assert h.service.eeg_guard(old_guard).get("error")
    assert h.decoder.trained and h.decoder.passed and h.decoder.eligible
    assert h.decoder.reset_calls == 0  # Only pending stream evidence is discarded.
    assert h.mouse.clicks == 0 and h.camera.calls == []


@pytest.mark.parametrize("change", ["disarm", "stop"])
def test_cnn_lifecycle_change_during_prediction_cancels_inflight_event(cnn, change):
    h = cnn
    h.arm()
    h.decoder.on_update = {
        "disarm": lambda: h.service.eeg_arm(False),
        "stop": h.service.stop,
    }[change]
    h.emit("in-flight-pair")
    assert h.mouse.clicks == h.control()["click_count"] == 0
    assert not h.control()["armed"]


def test_cnn_escape_guard_expiry_disarms(cnn):
    h = cnn
    h.arm()
    h.guard_active = False
    h.tick(28)
    h.emit("expired-pair")
    assert not h.control()["armed"] and h.mouse.clicks == 0
    assert h.service.eeg_guard(h.guard_token).get("error")


def test_cnn_saved_model_selection_binds_current_context_without_arming(cnn):
    h = cnn
    h.arm()
    old_guard = h.guard_token
    restored = FakeFastCnnBlinkDecoder()
    result = h.service.eeg_use_saved_cnn_model(restored, source="cnn-blink-saved.json")
    assert result.get("ok")
    assert h.service._decoder is restored
    assert h.control()["saved_model_source"] == "cnn-blink-saved.json"
    assert not h.control()["armed"]
    assert h.service.eeg_guard(old_guard).get("error")
    assert restored.bindings[-1]["session_id"] == h.eeg.acq["session_id"]
    assert restored.bindings[-1]["units"] == "V"
    assert h.mouse.clicks == 0 and h.camera.calls == []


@pytest.mark.parametrize("offset", [-1.0, 1.0, math.inf, -math.inf, math.nan])
def test_stale_future_and_nonfinite_event_times_never_click(cnn, offset):
    h = cnn
    h.arm()
    h.emit(at=h.clock.now + offset)
    assert h.mouse.clicks == 0


def test_expired_escape_lease_disarms_without_camera(cnn):
    h = cnn
    h.arm()
    h.guard_active = False
    h.tick(28)
    h.emit()
    assert not h.control()["armed"]
    assert h.mouse.clicks == 0
    assert h.service.eeg_guard(h.guard_token).get("error")


def test_stop_and_restart_discards_old_pairs_and_requires_explicit_new_arm(cnn):
    h = cnn
    h.arm()
    old_guard = h.guard_token
    h.decoder.event("before-stop", h.clock.now)
    assert h.service.stop().get("ok")
    assert h.service.eeg_guard(old_guard).get("error")
    assert h.service.start().get("ok")
    h.tick(3)
    assert not h.control()["armed"] and h.mouse.clicks == 0
    h.arm()
    h.emit("after-restart")
    assert h.mouse.clicks == 1
    assert h.camera.calls == []


def test_unowned_hand_controller_cannot_move_or_block_auto_clicks(cnn):
    h = cnn
    pointer = FakePointer()
    h.service.pointer_controller = pointer
    h.arm()
    pointer.finger_active = pointer.button_down = True
    pointer.click_id = 99  # Stale feedback from an inactive hand path is not an input source.
    h.emit()
    assert pointer.updates == [] and pointer.moves == []
    assert h.control()["finger_click_count"] == 0
    assert h.mouse.clicks == 1


def test_failed_click_disarms_and_unconfirmed_cleanup_blocks_rearm_and_restart(cnn):
    h = cnn
    h.arm()
    h.mouse.fail = h.mouse.cleanup_fail = True
    h.emit()
    assert h.mouse.clicks == 1 and h.control()["click_count"] == 0
    assert not h.control()["armed"]
    assert h.service.status()["cleanup_pending"]
    assert h.service.eeg_guard("cleanup-pending-guard").get("ok")
    assert h.service.eeg_arm(True, guard_token="cleanup-pending-guard").get("error")
    assert h.service.stop().get("error")
    assert h.service.start().get("error")
    h.mouse.fail = h.mouse.cleanup_fail = False
    assert h.service.stop().get("ok")
    assert not h.service.status()["cleanup_pending"]
