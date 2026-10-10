"""EEG control safety contracts with fake sensors, decoder, and desktop actions."""

import copy
import json
import math

import pytest

from core import multimodal


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class Camera:
    def __init__(self):
        self.running = False
        self.starts = 0
        self.stops = 0
        self.reads = 0
        self.frame = {"running": False, "state": "off", "tracked": False}

    def is_running(self):
        return self.running

    def start(self):
        self.starts += 1
        self.running = True
        return {"ok": True}

    def stop(self):
        self.stops += 1
        self.running = False
        self.frame = {"running": False, "state": "off", "tracked": False}
        return {"ok": True}

    def preview(self):
        self.reads += 1
        return copy.deepcopy(self.frame)


class EEG:
    """Its display snapshot contains only one row; its signal window is full-rate."""

    def __init__(self):
        self.starts = []
        self.stops = 0
        self.full_reads = 0
        self.rows = []
        self.counter = 0
        self.acq = {"state": "stopped", "mode": None, "session_id": "fake-session",
                    "channels": [{"num": 0, "name": "T3"}, {"num": 1, "name": "T4"}],
                    "nominal_hz": 250, "units": "V", "stats": {}}

    def acquisition_snapshot(self):
        return {**copy.deepcopy(self.acq), "samples": copy.deepcopy(self.rows[-1:])}

    def acquisition_signal_window(self):
        self.full_reads += 1
        return {**copy.deepcopy(self.acq), "samples": copy.deepcopy(self.rows)}

    def start_acquisition(self, mode="signal"):
        self.starts.append(mode)
        self.acq.update(state="running", mode=mode)
        return self.acquisition_snapshot()

    def stop_acquisition(self):
        self.stops += 1
        self.acq.update(state="stopped", mode=None)
        self.rows = []
        return {"state": "stopped"}


class Decoder:
    def __init__(self):
        self.eligible = True
        self.trained = True
        self.passed = True
        self.label = "rest"
        self.confidence = 0.95
        self.margin = 0.9
        self.valid = True
        self.ood = False
        self.fixed_window_id = None
        self.predictions = []
        self.trials = []
        self.train_calls = 0
        self.reset_calls = 0

    def status(self):
        return {"trained": self.trained, "arm_eligible": self.eligible,
                "passed_validation": self.passed,
                "state": "validated" if self.passed else "insufficient_validation",
                "counts": {label: sum(trial["label"] == label for trial in self.trials)
                           for label in ("left", "right", "rest")}}

    def predict(self, extracted):
        self.predictions.append(copy.deepcopy(extracted))
        if not extracted.get("ok"):
            return {"label": None, "confidence": 0.0, "margin": 0.0, "valid": False,
                    "ood": False, "window_id": None, "reason": extracted.get("error")}
        return {"label": self.label, "confidence": self.confidence, "margin": self.margin,
                "valid": self.valid, "ood": self.ood,
                "window_id": self.fixed_window_id or extracted["window"]["window_id"]}

    def add_trial(self, label, extracted, phase="train", trial_id=None, session_context=None):
        self.trials.append({"label": label, "phase": phase, "trial_id": trial_id,
                            "features": list(extracted["features"]),
                            "context": copy.deepcopy(extracted["context"]),
                            "window": copy.deepcopy(extracted["window"])})
        return {"ok": True}

    def train(self):
        self.train_calls += 1
        return {"ok": True, **self.status()}

    def reset(self):
        self.reset_calls += 1
        self.trials.clear()
        self.trained = self.eligible = self.passed = False
        return {"ok": True}


class Harness:
    def __init__(self, monkeypatch, mode="eeg"):
        self.clock, self.camera, self.eeg = Clock(), Camera(), EEG()
        self.actions = []
        self.extractions = []
        self.extract_ok = True
        self.guard_active = False
        self.guard_sequence = 0
        self.guard_token = "test-emergency-stop-guard-0"
        self.service = multimodal.MultimodalPreview(
            self.camera, self.eeg, lambda: False,
            lambda *args, **kwargs: self.actions.append((args, kwargs)), self.clock)
        self.service._launch_worker = lambda: None
        self.decoder = Decoder()
        self.service._decoder = self.decoder
        monkeypatch.setattr(multimodal, "extract_features", self.extract)
        assert self.service.set_control_mode(mode).get("ok")
        assert self.service.start().get("ok")
        self.started_at = self.clock.now

    def extract(self, rows, channels, nominal_hz, *, session_context=None):
        self.extractions.append(copy.deepcopy(rows))
        if not self.extract_ok or len(rows) != 500:
            return {"ok": False, "error": "Invalid or discontinuous EEG window."}
        first, last = rows[0], rows[-1]
        context = dict(session_context or {})
        context.update(channels=copy.deepcopy(channels), nominal_hz=nominal_hz)
        return {"ok": True, "features": [1.0, 2.0], "context": context,
                "window": {"start": first["estimated_monotonic"],
                           "end": last["estimated_monotonic"],
                           "counter_start": first["counter"], "counter_end": last["counter"],
                           "window_id": str(first["counter"]) + ":" + str(last["counter"])}}

    def tick(self, count=1, *, x=0.5, tracked=True, gap=0, eeg_age=0.0, add_rows=True):
        for _ in range(count):
            self.clock.now += 0.125
            if add_rows:
                target = int(round((self.clock.now - self.started_at) * 250))
                while self.eeg.counter < target:
                    counter = self.eeg.counter
                    at = self.started_at + counter / 250
                    self.eeg.rows.append({"counter": counter, "estimated_monotonic": at,
                                          "host_received_monotonic": self.clock.now,
                                          "samples": [math.sin(counter / 10) * 1e-6,
                                                      math.cos(counter / 8) * 1e-6]})
                    self.eeg.counter += 1
                self.eeg.rows = self.eeg.rows[-1250:]
            self.eeg.acq["stats"] = {"age_seconds": eeg_age, "gaps": gap, "queue_drops": 0,
                                     "duplicates": 0, "nonfinite": 0, "channel_mismatches": 0,
                                     "received_rate_hz": 250}
            self.camera.frame = {
                "running": self.camera.running, "state": "running" if self.camera.running else "off",
                "tracked": tracked and self.camera.running, "age_ms": 0,
                "sequence": self.camera.frame.get("sequence", 0) + 1,
                "landmarks": [[x, 0.5, 0] for _ in range(21)] if tracked else [],
                "image": "data:image/jpeg;base64,ZmFrZQ==", "width": 480, "height": 360}
            if self.guard_active:
                if self.service.eeg_guard(self.guard_token).get("error"):
                    self.guard_active = False
            self.service._tick()

    def warm(self):
        self.tick(18)

    def arm_request(self):
        self.guard_sequence += 1
        self.guard_token = "test-emergency-stop-guard-" + str(self.guard_sequence)
        self.guard_active = True
        assert self.service.eeg_guard(self.guard_token).get("ok")
        return self.service.eeg_arm(True, guard_token=self.guard_token)

    def arm(self):
        result = self.arm_request()
        assert not result.get("error"), result
        assert self.control()["armed"]

    def neutral(self):
        self.decoder.label = "rest"
        self.tick(12)
        assert self.control()["neutral_ready"]

    def control(self):
        return self.service.status()["eeg_control"]

    def movement(self, direction):
        self.tick(4)
        sign = 1 if direction == "right" else -1
        for amount in (0.075, 0.15, 0.225, 0.3):
            self.tick(x=0.5 + sign * amount)


@pytest.fixture
def setup(monkeypatch):
    return Harness(monkeypatch)


def test_eeg_start_and_runtime_need_no_camera(setup):
    h = setup
    h.warm()
    assert h.service.status()["state"] == "running"
    assert h.camera.starts == h.camera.stops == 0
    assert h.eeg.starts == ["signal"]
    h.arm()
    h.neutral()
    h.decoder.label = "right"
    h.tick(6, tracked=False)
    assert h.actions == [(("os_switch_desktop", {"direction": "right"}), {"actor": "gesture"})]
    assert h.control()["armed"]


def test_runtime_extracts_contiguous_two_seconds_from_full_rate_reader(setup):
    h = setup
    h.warm()
    assert h.eeg.full_reads > 0
    assert len(h.eeg.acquisition_snapshot()["samples"]) == 1
    assert h.extractions
    assert max(map(len, h.extractions)) == 500
    for rows in [rows for rows in h.extractions if len(rows) == 500]:
        assert len(rows) == 500
        assert rows[-1]["estimated_monotonic"] - rows[0]["estimated_monotonic"] == pytest.approx(1.996)
        assert all(b["counter"] == a["counter"] + 1 for a, b in zip(rows, rows[1:]))


def test_incomplete_two_second_window_never_enables_arming(setup):
    h = setup
    h.tick(15)
    assert not h.control()["can_arm"]
    assert h.service.eeg_arm(True).get("error")
    assert h.actions == []


def test_predictions_remain_preview_only_until_explicit_eeg_arm(setup):
    h = setup
    h.warm()
    h.tick(12)
    h.decoder.label = "left"
    h.tick(12)
    assert h.control()["prediction"]["label"] == "left"
    assert not h.control()["armed"]
    assert h.actions == []


@pytest.mark.parametrize("field", ["trained", "eligible", "passed"])
def test_each_training_and_validation_gate_is_required_for_arm(setup, field):
    h = setup
    setattr(h.decoder, field, False)
    h.warm()
    assert not h.control()["can_arm"]
    assert h.arm_request().get("error")
    assert not h.control()["armed"]
    h.decoder.label = "left"
    h.tick(8)
    assert h.actions == []


@pytest.mark.parametrize("changes", [
    {"label": "rest"}, {"label": None}, {"valid": False}, {"ood": True},
    {"confidence": 0.79}, {"margin": 0.24},
])
def test_rest_uncertain_invalid_and_ood_predictions_never_dispatch(setup, changes):
    h = setup
    h.warm()
    h.arm()
    h.neutral()
    h.decoder.label = "left"
    for field, value in changes.items():
        setattr(h.decoder, field, value)
    h.tick(12)
    assert h.actions == []


@pytest.mark.parametrize("changes", [
    {"label": None}, {"label": None, "valid": False, "ood": True},
    {"confidence": 0.79}, {"margin": 0.24},
])
def test_brief_model_uncertainty_preserves_confirmed_rest_but_restarts_direction_streak(setup, changes):
    h = setup
    h.warm()
    h.arm()
    h.neutral()
    h.decoder.label = "left"
    h.tick(4)  # Two direction windows are not enough to dispatch.
    assert h.actions == []
    for field, value in changes.items():
        setattr(h.decoder, field, value)
    h.tick(6)
    assert h.control()["armed"]
    assert not h.control()["can_arm"]
    assert h.control()["neutral_ready"]
    assert h.actions == []
    h.decoder.label, h.decoder.valid, h.decoder.ood = "left", True, False
    h.decoder.confidence, h.decoder.margin = 0.95, 0.9
    h.tick(4)
    assert h.control()["armed"]
    assert h.actions == []  # Old pre-transition direction windows cannot count.
    h.tick(2)
    assert len(h.actions) == 1
    assert h.actions[0][0][1] == {"direction": "left"}


@pytest.mark.parametrize("ood", [False, True])
def test_three_seconds_of_uncertainty_disarms_and_does_not_rearm_after_recovery(setup, ood):
    h = setup
    h.warm()
    h.arm()
    h.neutral()
    h.decoder.label, h.decoder.ood = None, ood
    h.decoder.valid = not ood
    h.tick(22)
    assert h.control()["armed"]
    assert h.control()["neutral_ready"]
    assert not h.control()["can_arm"]
    assert h.actions == []
    h.tick(4)
    assert not h.control()["armed"]
    assert not h.control()["neutral_ready"]
    h.decoder.label, h.decoder.valid, h.decoder.ood = "left", True, False
    h.tick(8)
    assert not h.control()["armed"]
    assert h.actions == []


def test_confident_recovery_at_uncertainty_deadline_cannot_preserve_arming(setup):
    h = setup
    h.warm()
    h.arm()
    h.neutral()
    h.decoder.label, h.decoder.valid, h.decoder.ood = None, False, True
    h.tick(24)
    assert h.control()["armed"]  # First uncertain prediction was one tick later.
    h.decoder.label, h.decoder.valid, h.decoder.ood = "left", True, False
    h.tick()
    assert not h.control()["armed"]
    assert not h.control()["neutral_ready"]
    h.tick(6)
    assert not h.control()["armed"]
    assert h.actions == []


@pytest.mark.parametrize("failure", ["invalid_features", "gap", "missing_reader"])
def test_raw_signal_failure_during_transition_grace_still_disarms_immediately(setup, monkeypatch, failure):
    h = setup
    h.warm()
    h.arm()
    h.neutral()
    h.decoder.label, h.decoder.valid, h.decoder.ood = None, False, True
    h.tick(4)
    assert h.control()["armed"]
    if failure == "invalid_features":
        h.extract_ok = False
    elif failure == "missing_reader":
        monkeypatch.setattr(h.eeg, "acquisition_signal_window", None)
    h.tick(2, gap=int(failure == "gap"))
    assert not h.control()["armed"]
    assert not h.control()["neutral_ready"]
    assert h.actions == []


def test_bad_feature_window_disarms_without_dispatch(setup):
    h = setup
    h.warm()
    h.arm()
    h.neutral()
    h.extract_ok = False
    h.decoder.label = "left"
    h.tick(4)
    assert not h.control()["armed"]
    assert h.actions == []


@pytest.mark.parametrize("failure", ["missing_reader", "wrong_session"])
def test_full_rate_reader_must_supply_the_current_acquisition_session(setup, monkeypatch, failure):
    h = setup
    h.warm()
    h.arm()
    h.neutral()
    if failure == "missing_reader":
        monkeypatch.setattr(h.eeg, "acquisition_signal_window", None)
    else:
        read = h.eeg.acquisition_signal_window

        def wrong_session():
            return {**read(), "session_id": "different-acquisition"}

        monkeypatch.setattr(h.eeg, "acquisition_signal_window", wrong_session)
    h.decoder.label = "left"
    h.tick(6)
    assert not h.control()["armed"]
    assert not h.control()["prediction"]
    assert h.actions == []


def test_eeg_direction_ignores_opposite_webcam_motion(monkeypatch):
    h = Harness(monkeypatch, mode="webcam")
    h.warm()
    assert h.service.set_control_mode("eeg").get("ok")
    h.tick(4)
    h.arm()
    h.neutral()
    h.decoder.label = "left"
    for amount in (0.075, 0.15, 0.225, 0.3):
        h.tick(x=0.5 + amount)
    h.tick(2, x=0.8)
    assert len(h.actions) == 1
    assert h.actions[0][0][1] == {"direction": "left"}


@pytest.mark.parametrize("change", ["webcam", "reset", "disconnect", "gap"])
def test_mode_reset_disconnect_or_gap_disarms_and_never_auto_rearms(setup, change):
    h = setup
    h.warm()
    h.arm()
    if change == "webcam":
        result = h.service.set_control_mode("webcam")
        # A camera-free session needs a stop/restart to acquire the camera;
        # the attempted mode transition still has to disarm existing controls.
        assert result.get("error")
    elif change == "reset":
        assert h.service.eeg_reset().get("ok")
    elif change == "disconnect":
        h.eeg.acq.update(state="error", error="Disconnected")
        h.tick()
    else:
        h.tick(gap=1)
    assert not h.control()["armed"]
    h.decoder.label = "left"
    h.tick(8, gap=int(change == "gap"))
    assert not h.control()["armed"]
    assert h.actions == []


def test_replayed_packets_with_misleading_fresh_age_cannot_keep_eeg_armed(setup):
    h = setup
    h.warm()
    h.arm()
    h.neutral()
    h.decoder.label = "left"
    h.tick(8, add_rows=False, eeg_age=0)
    assert not h.control()["armed"]
    assert h.actions == []


def test_one_fresh_poll_without_a_new_batch_preserves_explicit_arming(setup):
    h = setup
    h.warm()
    h.arm()
    h.neutral()
    h.tick(add_rows=False, eeg_age=0.125)
    assert h.control()["armed"]
    assert h.control()["neutral_ready"]
    assert h.actions == []
    h.decoder.label = "left"
    h.tick(6)
    assert len(h.actions) == 1


def test_one_fresh_poll_without_a_new_batch_preserves_guided_trial(monkeypatch):
    h = Harness(monkeypatch, mode="webcam")
    h.decoder.trained = h.decoder.passed = h.decoder.eligible = False
    h.warm()
    assert h.service.eeg_trial("rest", "train").get("ok")
    h.tick(add_rows=False, eeg_age=0.125)
    assert h.control()["pending"] is not None
    h.tick(23)
    assert len(h.decoder.trials) == 1
    assert h.actions == []


def test_elapsed_cached_prediction_and_sensor_age_blocks_arming(setup):
    h = setup
    h.warm()
    assert h.control()["can_arm"]
    h.clock.now += 0.8
    assert not h.control()["can_arm"]
    assert h.service.eeg_arm(True).get("error")
    assert h.actions == []


def test_three_distinct_predictions_and_one_second_rest_are_required(setup):
    h = setup
    h.warm()
    h.arm()
    h.decoder.label = "left"
    h.tick(8)
    assert h.actions == []
    h.decoder.label = "rest"
    h.tick(6)
    assert not h.control()["neutral_ready"]
    h.tick(6)
    assert h.control()["neutral_ready"]
    h.decoder.label = "left"
    h.tick(4)
    assert h.actions == []
    h.tick(2)
    assert len(h.actions) == 1


def test_repeated_prediction_window_cannot_satisfy_direction_debounce(setup):
    h = setup
    h.warm()
    h.arm()
    h.neutral()
    h.decoder.label = "left"
    h.decoder.fixed_window_id = "duplicate-window"
    h.tick(10)
    assert h.actions == []


def test_dispatch_requires_new_rest_and_two_second_cooldown(setup):
    h = setup
    h.warm()
    h.arm()
    h.neutral()
    h.decoder.label = "right"
    h.tick(6)
    assert len(h.actions) == 1
    first_at = h.clock.now
    h.tick(20)
    assert len(h.actions) == 1  # Holding a prediction cannot repeat an action.
    h.decoder.label = "rest"
    h.tick(10)
    h.decoder.label = "left"
    h.tick(6)
    assert len(h.actions) == 2
    assert h.clock.now - first_at >= 2.0


def test_explicit_rearm_cannot_bypass_two_second_action_cooldown(setup):
    h = setup
    h.warm()
    h.arm()
    h.neutral()
    h.decoder.label = "right"
    h.tick(6)
    assert len(h.actions) == 1
    assert h.service.eeg_arm(False).get("ok")
    h.arm()
    h.decoder.label = "rest"
    h.tick(10)
    h.decoder.label = "left"
    h.tick(4)
    assert len(h.actions) == 1
    h.tick(4)
    assert len(h.actions) == 2


def test_direction_predictions_must_be_consecutive(setup):
    h = setup
    h.warm()
    h.arm()
    h.neutral()
    for label, ticks in (("left", 2), ("right", 2), ("left", 4)):
        h.decoder.label = label
        h.tick(ticks)
    assert h.actions == []
    h.tick(2)
    assert len(h.actions) == 1
    assert h.actions[0][0][1] == {"direction": "left"}


def capture_ready_dispatch(h, monkeypatch):
    h.warm()
    h.arm()
    h.neutral()
    captured = []
    original = h.service._dispatch_eeg_direction
    monkeypatch.setattr(h.service, "_dispatch_eeg_direction", lambda *args: captured.append(args))
    h.decoder.label = "left"
    h.tick(6)
    assert len(captured) == 1
    assert h.actions == []
    monkeypatch.setattr(h.service, "_dispatch_eeg_direction", original)
    return original, captured[0]


@pytest.mark.parametrize("race", ["arm_epoch", "generation", "prediction_id", "prediction_age"])
def test_actuator_boundary_rechecks_epoch_session_and_current_prediction(setup, monkeypatch, race):
    h = setup
    dispatch, event = capture_ready_dispatch(h, monkeypatch)
    # Preserve every other ready-to-dispatch condition, so each race exercises
    # its own final guard rather than merely relying on a disarmed boolean.
    if race == "arm_epoch":
        h.service._arm_epoch += 1
    elif race == "generation":
        h.service._generation += 1
    elif race == "prediction_id":
        h.service._eeg_prediction["window_id"] = "newer-prediction"
    else:
        h.service._eeg_prediction_at = h.clock.now - 0.501
    dispatch(*event)
    assert h.actions == []


def test_final_actuator_boundary_enforces_exact_two_second_cooldown(setup, monkeypatch):
    h = setup
    dispatch, event = capture_ready_dispatch(h, monkeypatch)
    h.service._last_dispatch_at = h.clock.now - 1.999
    dispatch(*event)
    assert h.actions == []
    h.service._last_dispatch_at = h.clock.now - 2.0
    dispatch(*event)
    assert len(h.actions) == 1


def test_invalid_mode_and_non_boolean_arm_are_rejected(setup):
    h = setup
    h.warm()
    assert h.service.set_control_mode("automatic").get("error")
    assert h.service.eeg_arm("true").get("error")
    assert not h.control()["armed"]
    assert h.actions == []


def test_eeg_arm_requires_matching_emergency_stop_guard(setup):
    h = setup
    h.warm()
    assert h.control()["can_arm"]  # Model/data eligibility alone is insufficient.
    assert h.service.eeg_arm(True).get("error")
    assert not h.control()["armed"]
    assert h.service.eeg_guard(h.guard_token).get("ok")
    assert h.control()["guard_ready"]
    assert h.service.eeg_arm(True, guard_token="wrong-guard").get("error")
    assert not h.control()["armed"]
    assert h.service.eeg_arm(True, guard_token=h.guard_token).get("ok")
    assert h.control()["armed"]


def test_guard_renewal_alone_never_arms_eeg(setup):
    h = setup
    h.warm()
    h.guard_active = True
    h.tick(32)
    assert h.control()["guard_ready"]
    assert not h.control()["armed"]
    assert h.actions == []


def test_guard_expiry_disarms_and_renewal_does_not_restore_arming(setup):
    h = setup
    h.warm()
    h.arm()
    h.guard_active = False
    h.tick(26)
    assert not h.control()["guard_ready"]
    assert not h.control()["armed"]
    assert h.service.eeg_guard(h.guard_token).get("error")
    h.guard_token = "new-emergency-stop-guard-after-expiry"
    assert h.service.eeg_guard(h.guard_token).get("ok")
    assert h.control()["guard_ready"]
    assert not h.control()["armed"]
    h.decoder.label = "left"
    h.guard_active = True
    h.tick(10)
    assert h.actions == []


def test_new_guard_token_disarms_existing_eeg_control(setup):
    h = setup
    h.warm()
    h.arm()
    assert h.service.eeg_guard("replacement-emergency-stop-guard").get("ok")
    assert not h.control()["armed"]
    assert h.service.eeg_arm(True, guard_token=h.guard_token).get("error")
    assert h.service.eeg_guard(h.guard_token).get("error")
    assert h.control()["guard_ready"]
    assert h.service.eeg_arm(True, guard_token="replacement-emergency-stop-guard").get("ok")
    assert h.control()["armed"]
    assert h.service.eeg_guard(h.guard_token).get("error")
    assert h.control()["armed"]  # An old heartbeat cannot replace the new lease.
    assert h.actions == []


@pytest.mark.parametrize("operation", ["disarm", "stop", "mode", "reset", "train", "legacy_reset"])
def test_explicit_lifecycle_actions_retire_guard_against_delayed_messages(monkeypatch, operation):
    h = Harness(monkeypatch, mode="webcam")
    h.warm()
    assert h.service.set_control_mode("eeg").get("ok")
    h.tick(4)
    h.arm()
    token = h.guard_token
    h.guard_active = False
    if operation == "disarm":
        result = h.service.eeg_arm(False)
    elif operation == "stop":
        result = h.service.stop()
    elif operation == "mode":
        result = h.service.set_control_mode("webcam")
    elif operation == "reset":
        result = h.service.eeg_reset()
    elif operation == "train":
        result = h.service.eeg_train()
    else:
        result = h.service.reset_calibration()
    assert result.get("ok"), result
    assert not h.control()["armed"]
    assert not h.control()["guard_ready"]
    assert h.service.eeg_guard(token).get("error")
    assert not h.control()["guard_ready"]
    assert h.service.eeg_arm(True, guard_token=token).get("error")
    assert not h.control()["armed"]
    assert h.actions == []


def test_retired_guard_cannot_rearm_but_a_new_explicit_guard_can(setup):
    h = setup
    h.warm()
    h.arm()
    retired_token = h.guard_token
    assert h.service.eeg_arm(False).get("ok")
    assert h.service.eeg_guard(retired_token).get("error")
    assert h.service.eeg_arm(True, guard_token=retired_token).get("error")
    assert not h.control()["armed"]
    h.arm()
    assert h.guard_token != retired_token
    assert h.control()["armed"]
    assert h.actions == []


@pytest.mark.parametrize("failure", ["gap", "sustained_uncertainty"])
def test_automatic_safety_disarm_retires_guard_even_after_inputs_recover(setup, failure):
    h = setup
    h.warm()
    h.arm()
    token = h.guard_token
    if failure == "gap":
        h.tick(gap=1)
        h.tick(4, gap=1)
    else:
        h.decoder.label = None
        h.tick(26)
        h.decoder.label = "rest"
        h.tick(4)
    assert h.control()["can_arm"]
    assert not h.control()["armed"]
    assert not h.control()["guard_ready"]
    assert h.service.eeg_guard(token).get("error")
    assert h.service.eeg_arm(True, guard_token=token).get("error")
    assert not h.control()["armed"]
    h.arm()
    assert h.guard_token != token
    assert h.control()["armed"]
    assert h.actions == []


def test_retirement_capacity_fails_closed_without_evicting_old_tokens(setup):
    h = setup
    h.warm()
    for i in range(256):
        assert h.service.eeg_guard("bounded-guard-" + str(i)).get("ok")
    assert h.service.eeg_arm(False).get("ok")  # Retire the final active token.
    assert not h.control()["guard_ready"]
    assert h.service.eeg_guard("bounded-guard-0").get("error")
    assert h.service.eeg_guard("overflow-new-guard").get("error")
    assert h.service.eeg_arm(True, guard_token="bounded-guard-0").get("error")
    assert not h.control()["armed"]
    # Starting acquisition again is not a backend restart and must not clear
    # the retired-token history or turn an exhausted guard store back on.
    assert h.service.stop().get("ok")
    assert h.service.start().get("ok")
    assert h.service.eeg_guard("after-acquisition-restart").get("error")
    assert h.actions == []


def test_expired_guard_is_rechecked_at_final_actuator_boundary(setup, monkeypatch):
    h = setup
    dispatch, event = capture_ready_dispatch(h, monkeypatch)
    h.guard_active = False
    h.clock.now += 3.001
    # Keep the cached sensor/model ages fresh to isolate the lease boundary.
    h.service._observed_at = h.clock.now
    h.service._last_valid_eeg_at = h.clock.now
    h.service._eeg_prediction_at = h.clock.now
    dispatch(*event)
    assert h.actions == []


def test_eeg_guided_trial_requires_a_live_tracked_camera(setup):
    h = setup
    h.warm()
    assert h.service.eeg_trial("left", "train").get("error")
    assert h.camera.starts == 0
    assert not h.decoder.trials


@pytest.mark.parametrize("label", ["left", "right", "rest"])
@pytest.mark.parametrize("phase", ["train", "validate"])
def test_guided_three_second_trials_retain_features_only(monkeypatch, label, phase):
    h = Harness(monkeypatch, mode="webcam")
    h.decoder.trained = phase == "validate"
    h.decoder.passed = h.decoder.eligible = False
    h.warm()
    result = h.service.eeg_trial(label, phase)
    assert not result.get("error"), result
    assert not h.control()["armed"]
    started_at = h.clock.now
    if label == "rest":
        h.tick(23)
    else:
        h.movement(label)
        h.tick(15, x=0.8 if label == "right" else 0.2)
    assert h.decoder.trials == []
    h.tick(x=0.8 if label == "right" else 0.2 if label == "left" else 0.5)
    assert h.clock.now - started_at == pytest.approx(3.0)
    assert len(h.decoder.trials) == 1
    trial = h.decoder.trials[0]
    assert trial["label"] == label and trial["phase"] == phase
    assert trial["features"] == [1.0, 2.0]
    assert "samples" not in trial
    assert trial["window"]["counter_end"] - trial["window"]["counter_start"] == 499
    assert h.actions == []


@pytest.mark.parametrize("direction", ["left", "right"])
def test_guided_movement_after_selected_feature_window_is_rejected(monkeypatch, direction):
    h = Harness(monkeypatch, mode="webcam")
    h.decoder.trained = h.decoder.passed = h.decoder.eligible = False
    h.warm()
    assert h.service.eeg_trial(direction, "train").get("ok")
    h.tick(18)
    sign = 1 if direction == "right" else -1
    for amount in (0.075, 0.15, 0.225, 0.3):
        h.tick(x=0.5 + sign * amount)
    # The matching event occurs 2.75 seconds into a 3-second trial, after
    # the selected feature window (approximately 0.5 through 2.5 seconds).
    h.tick(2, x=0.5 + sign * 0.3)
    assert not h.control()["pending"]
    assert not h.decoder.trials
    assert h.actions == []


def test_guided_trial_selects_500_packets_despite_boundary_timestamp_jitter(monkeypatch):
    h = Harness(monkeypatch, mode="webcam")
    h.decoder.trained = h.decoder.passed = h.decoder.eligible = False
    h.warm()
    assert h.service.eeg_trial("rest", "train").get("ok")
    trial_start = h.clock.now
    read = h.eeg.acquisition_signal_window
    shifted = []

    def jittered_window():
        result = read()
        for row in result["samples"]:
            if trial_start + 2.496 <= row["estimated_monotonic"] < trial_start + 2.5:
                row["estimated_monotonic"] += 0.008
                shifted.append(row["counter"])
        return result

    monkeypatch.setattr(h.eeg, "acquisition_signal_window", jittered_window)
    h.tick(24)
    assert shifted
    assert len(h.decoder.trials) == 1
    window = h.decoder.trials[0]["window"]
    assert window["counter_end"] - window["counter_start"] == 499
    assert window["end"] > trial_start + 2.5
    assert h.actions == []


@pytest.mark.parametrize("failure", ["wrong_direction", "camera_dropout", "moving_rest", "packet_gap"])
def test_guided_trials_reject_wrong_motion_dropouts_and_nonstationary_rest(monkeypatch, failure):
    h = Harness(monkeypatch, mode="webcam")
    h.decoder.trained = h.decoder.passed = h.decoder.eligible = False
    h.warm()
    label = "rest" if failure == "moving_rest" else "left"
    assert h.service.eeg_trial(label, "train").get("ok")
    if failure in ("wrong_direction", "moving_rest"):
        h.movement("right")
        h.tick(16, x=0.8)
    elif failure == "camera_dropout":
        h.tick(tracked=False)
        h.tick(23)
    else:
        h.tick(24, gap=1)
    assert not h.decoder.trials
    assert not h.control()["pending"]
    assert h.actions == []


def test_trial_entry_disarms_and_blocks_arm_and_overlapping_trials(monkeypatch):
    h = Harness(monkeypatch, mode="webcam")
    h.decoder.trained = h.decoder.passed = h.decoder.eligible = False
    h.warm()
    assert h.service.arm(True)["armed"]
    assert h.service.eeg_trial("rest", "train").get("ok")
    assert not h.control()["armed"]
    assert not h.service.status()["armed"]
    assert h.service.eeg_arm(True).get("error")
    assert h.service.eeg_trial("left", "train").get("error")
    h.tick(24)
    assert h.actions == []


@pytest.mark.parametrize("label,phase", [("up", "train"), ("left", "test"), (None, "train")])
def test_guided_trial_rejects_invalid_label_or_phase(monkeypatch, label, phase):
    h = Harness(monkeypatch, mode="webcam")
    h.warm()
    assert h.service.eeg_trial(label, phase).get("error")
    assert not h.control()["pending"]
    assert not h.decoder.trials


def test_training_disarms_and_reset_removes_retained_features(setup):
    h = setup
    h.warm()
    h.arm()
    assert h.service.eeg_train().get("ok")
    assert h.decoder.train_calls == 1
    assert not h.control()["armed"]
    h.decoder.trials.append({"label": "left", "features": [1.0, 2.0]})
    assert h.service.eeg_reset().get("ok")
    assert h.decoder.reset_calls == 1
    assert not h.decoder.trials
    assert h.service.eeg_arm(True).get("error")


def test_stop_discards_live_raw_and_predictions_but_preserves_feature_trials(setup):
    h = setup
    h.warm()
    h.arm()
    h.decoder.trials.append({"label": "left", "features": [1.0, 2.0]})
    assert h.service.stop().get("ok")
    assert not h.control()["armed"]
    assert not h.control()["guard_ready"]
    assert not h.control()["prediction"]
    assert h.decoder.trials == [{"label": "left", "features": [1.0, 2.0]}]
    assert not h.service._raw
    assert not h.service._eeg_full_rows
    payload = json.dumps(h.service.snapshot())
    assert "ZmFrZQ==" not in payload
    assert '"samples"' not in payload
    assert '"features"' not in json.dumps(h.service.status())
