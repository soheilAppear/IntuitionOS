"""Preview safety and calibration tests: all devices and actions are fakes."""

import copy
import json
import math
import threading
import time

import pytest

from core.multimodal import MirroredMotionTracker, MultimodalPreview


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
        self.frame = {}
        self.start_error = None
        self.stop_result = {"ok": True}

    def is_running(self):
        return self.running

    def start(self):
        self.starts += 1
        self.running = not bool(self.start_error)
        return {"error": self.start_error} if self.start_error else {"ok": True}

    def stop(self):
        self.stops += 1
        self.running = False
        return self.stop_result

    def preview(self):
        return copy.deepcopy(self.frame)


class EEG:
    def __init__(self):
        self.starts = []
        self.stops = 0
        self.start_error = None
        self.stop_error = None
        self.acq = {"state": "stopped", "mode": None, "samples": [], "stats": {}}

    def acquisition_snapshot(self):
        return copy.deepcopy(self.acq)

    def start_acquisition(self, mode="signal"):
        self.starts.append(mode)
        if self.start_error:
            return {"error": self.start_error}
        self.acq.update(state="running", mode=mode, units="V" if mode == "signal" else "ohm",
                        nominal_hz=250, session_id="fake", channels=[{"num": 0, "name": "T3"}, {"num": 1, "name": "T4"}])
        return self.acquisition_snapshot()

    def stop_acquisition(self):
        self.stops += 1
        self.acq.update(state="error" if self.stop_error else "stopped", mode=None, samples=[])
        return {"error": self.stop_error, "state": "error"} if self.stop_error else {"state": "stopped"}


@pytest.fixture
def setup():
    clock, camera, eeg, actions = Clock(), Camera(), EEG(), []
    service = MultimodalPreview(camera, eeg, lambda: False,
                               lambda *args, **kwargs: actions.append((args, kwargs)), clock)
    service._launch_worker = lambda: None
    assert service.start()["ok"]
    return service, clock, camera, eeg, actions


def tick(setup, x=0.4, *, tracked=True, camera_age=0, eeg_age=0, gap=0, increment=True):
    service, clock, camera, eeg, _ = setup
    previous = clock.now
    clock.now += 0.125
    sequence = camera.frame.get("sequence", 0) + int(increment)
    camera.frame = {"running": camera.running, "tracked": tracked,
                    "landmarks": [[x, 0.5, 0] for _ in range(21)] if tracked else [],
                    "sequence": sequence, "age_ms": camera_age,
                    "image": "data:image/jpeg;base64,ZmFrZQ==", "width": 480, "height": 360, "fps": 8}
    eeg.acq["stats"] = {"age_seconds": eeg_age, "gaps": gap, "queue_drops": 0,
                         "channel_mismatches": 0, "received_rate_hz": 250}
    eeg.acq["samples"] = [{"counter": int((previous + i * 0.004) * 250),
                           "estimated_monotonic": previous + i * 0.004,
                           "host_received_monotonic": clock.now,
                           "samples": [(1 if i % 2 else -1) * 1e-6, math.sin(i) * 2e-6]}
                          for i in range(31)]
    service._tick()


def movement(setup, direction="right"):
    for _ in range(4):
        tick(setup)
    sign = 1 if direction == "right" else -1
    for amount in (0.075, 0.15, 0.225, 0.30):
        tick(setup, 0.4 + sign * amount)


@pytest.mark.parametrize("direction,sign", [("right", 1), ("left", -1)])
def test_mirrored_sign_debounce_cooldown_and_neutral(direction, sign):
    tracker = MirroredMotionTracker()
    for now in (0, 0.125, 0.25, 0.375):
        assert tracker.update(0.5, 0.5, now) is None
    for now, amount in ((0.5, 0.075), (0.625, 0.15), (0.75, 0.225)):
        assert tracker.update(0.5 + sign * amount, 0.5, now) is None
    assert tracker.update(0.5 + sign * 0.30, 0.5, 0.875) == direction
    for i in range(1, 12):
        assert tracker.update(0.5 + sign * 0.30, 0.5, 0.875 + i * 0.125) is None
    assert not tracker.status(2.25)["ready"]
    tracker.update(0.5 + sign * 0.30, 0.5, 2.5)
    assert tracker.status(2.5)["ready"]


def test_jitter_vertical_motion_and_single_frame_jump_do_not_fire():
    tracker = MirroredMotionTracker()
    for i in range(40):
        assert tracker.update(0.5 + (0.01 if i % 2 else -0.01), 0.5, i * 0.125) is None
    assert tracker.update(0.8, 0.5, 5.1) is None
    assert tracker.update(0.5, 0.5, 5.2) is None
    assert tracker.update(0.8, 0.9, 5.3) is None


def test_preview_only_arm_required_and_no_replay(setup):
    service, _, _, _, actions = setup
    movement(setup)
    assert actions == []
    assert service.arm(True)["armed"]
    assert actions == []
    movement(setup, "left")
    assert actions == [(("os_switch_desktop", {"direction": "left"}), {"actor": "gesture"})]
    service.arm(False)
    movement(setup, "right")
    assert len(actions) == 1


def test_explicit_rearm_does_not_bypass_desktop_action_cooldown(setup):
    service, _, _, _, actions = setup
    tick(setup)
    service.arm(True)
    movement(setup)
    assert len(actions) == 1
    service.arm(False)
    service.arm(True)
    movement(setup, "left")
    assert len(actions) == 1


@pytest.mark.parametrize("dropout", [{"tracked": False}, {"camera_age": 501}, {"eeg_age": 0.8}, {"gap": 1}])
def test_dropout_disarms_and_never_rearms_automatically(setup, dropout):
    service, _, _, _, actions = setup
    tick(setup)
    assert service.arm(True)["armed"]
    tick(setup, **dropout)
    assert not service.status()["armed"]
    movement(setup)
    assert not service.status()["armed"]
    assert actions == []


def test_repeated_camera_sequence_is_not_fresh(setup):
    service = setup[0]
    tick(setup)
    service.arm(True)
    for _ in range(5):
        tick(setup, increment=False)
    assert not service.status()["armed"]


@pytest.mark.parametrize("ages", [{"camera_age": 490}, {"eeg_age": 0.74}])
def test_cached_sensor_age_advances_before_arming_or_dispatch(setup, ages):
    service, clock, _, _, actions = setup
    tick(setup, **ages)
    assert service.arm(True)["armed"]
    generation, epoch = service._generation, service._arm_epoch
    clock.now += 0.1
    assert not service.status()["can_arm"]
    service._dispatch_direction("right", generation, epoch)
    assert actions == []
    assert service.arm(True)["error"]


@pytest.mark.parametrize("invalid", ["duplicate", "nulls", "nonfinite", "empty"])
def test_invalid_or_duplicate_eeg_cannot_keep_control_armed(setup, invalid):
    service, clock, camera, eeg, actions = setup
    tick(setup)
    service.arm(True)
    before = service.status()["revision"]
    clock.now += 0.125
    camera.frame["sequence"] += 1
    if invalid == "nulls":
        for row in eeg.acq["samples"]:
            row["samples"] = [None, None]
    elif invalid == "nonfinite":
        for row in eeg.acq["samples"]:
            row["samples"] = [float("nan"), 1e-6]
        eeg.acq["stats"]["nonfinite"] = 31
    elif invalid == "empty":
        eeg.acq["samples"] = []
    # In all cases the simulated callback reports a misleading fresh arrival.
    eeg.acq["stats"]["age_seconds"] = 0
    for row in eeg.acq["samples"]:
        row["host_received_monotonic"] = clock.now
        row["estimated_monotonic"] = clock.now
    service._tick()
    result = service.status()
    assert result["revision"] > before
    assert not result["armed"] and not result["can_arm"]
    assert service.arm(True)["error"]
    assert actions == []


def test_revision_changes_for_arm_disarm_tick_and_stop_but_not_status_reads(setup):
    service = setup[0]
    first = service.status()["revision"]
    assert service.status()["revision"] == first
    tick(setup)
    second = service.status()["revision"]
    third = service.arm(True)["revision"]
    fourth = service.arm(False)["revision"]
    fifth = service.stop()["revision"]
    assert first < second < third < fourth < fifth


def test_old_failure_cannot_stop_restarted_session(setup):
    service, _, camera, eeg, _ = setup
    generation = service._generation
    service.stop()
    service.start()
    camera_stops, eeg_stops = camera.stops, eeg.stops
    service._fail("delayed old worker failure", generation)
    assert service.status()["state"] == "starting"
    assert camera.stops == camera_stops and eeg.stops == eeg_stops


def test_failure_rechecks_generation_after_waiting_for_lifecycle_lock(setup):
    service, _, camera, eeg, _ = setup
    generation = service._generation
    service._operation_lock.acquire()
    thread = threading.Thread(target=service._fail, args=("old failure", generation))
    thread.start()
    deadline = time.monotonic() + 1
    while not service._halt.is_set() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert service._halt.is_set()
    # Represents a completed restart while the old failure waits for ownership.
    with service._lock:
        service._generation += 1
        service._state = "starting"
        service._halt.clear()
    service._operation_lock.release()
    thread.join(1)
    assert not thread.is_alive()
    assert camera.stops == 0 and eeg.stops == 0
    assert service.status()["state"] == "starting"


def test_reads_are_cached_and_never_start_hardware():
    camera, eeg = Camera(), EEG()
    service = MultimodalPreview(camera, eeg)
    service.status()
    service.snapshot()
    assert camera.starts == 0 and eeg.starts == []
    assert service.arm(True)["error"]
    assert service.stop()["ok"]
    assert camera.stops == 0 and eeg.stops == 0


def test_normal_camera_owner_and_existing_acquisition_are_never_stopped():
    camera, eeg = Camera(), EEG()
    service = MultimodalPreview(camera, eeg, lambda: True)
    assert "Turn off" in service.start()["error"]
    service.stop()
    assert camera.starts == camera.stops == 0 and eeg.stops == 0
    service.gesture_busy = lambda: False
    eeg.acq.update(state="running", mode="signal")
    assert "existing" in service.start()["error"]
    service.stop()
    assert eeg.stops == 0


def test_camera_start_failure_cleans_up_owned_acquisition():
    camera, eeg = Camera(), EEG()
    camera.start_error = "missing model"
    service = MultimodalPreview(camera, eeg)
    result = service.start()
    assert "missing model" in result["error"]
    assert camera.stops == 1 and eeg.stops == 1
    assert not result["armed"]


def test_eeg_start_failure_does_not_start_or_stop_unowned_camera():
    camera, eeg = Camera(), EEG()
    eeg.start_error = "connect first"
    service = MultimodalPreview(camera, eeg)
    assert "connect first" in service.start()["error"]
    assert camera.starts == 0 and camera.stops == 0


def test_stop_unconfirmed_is_reported_and_retried(setup):
    service, _, camera, eeg, _ = setup
    camera.stop_result = {"ok": True, "stopping": True}
    eeg.stop_error = "native stop failed"
    result = service.stop()
    assert result["state"] == "error"
    assert result["cleanup_pending"]
    assert "Camera stop unconfirmed" in result["error"] and "EEG stop unconfirmed" in result["error"]
    assert "cleanup" in service.start()["error"]
    camera.stop_result, eeg.stop_error = {"ok": True}, None
    assert service.stop()["state"] == "stopped"
    assert not service.status()["cleanup_pending"]


def test_stale_device_stops_both_owned_resources_after_startup_grace(setup):
    service, clock, camera, eeg, _ = setup
    clock.now += 6
    tick(setup, eeg_age=2)
    result = service.status()
    assert result["state"] == "error"
    assert camera.stops == 1 and eeg.stops == 1
    assert not result["armed"]


def test_stop_discards_images_raw_and_pending_trial_but_not_feature_counts(setup):
    service = setup[0]
    tick(setup)
    assert service.mark_trial("left")["ok"]
    assert service._raw
    assert service.stop()["ok"]
    assert not service._raw
    snapshot = service.snapshot()
    assert "image" not in snapshot["camera"]
    assert "waveform" not in snapshot["eeg"]
    assert snapshot["calibration"]["pending"] is None


def test_status_has_no_images_samples_or_trial_features(setup):
    service = setup[0]
    tick(setup)
    encoded = json.dumps(service.status())
    assert '"image"' not in encoded and '"samples"' not in encoded and '"features"' not in encoded
    snapshot = service.snapshot()
    assert snapshot["camera"]["image"].startswith("data:image/jpeg;base64,")
    assert len(snapshot["eeg"]["waveform"]) <= 250


def run_trial(setup, target, observed=None):
    service = setup[0]
    tick(setup)
    assert service.mark_trial(target)["ok"]
    movement(setup, observed or target)
    last_x = 0.7 if (observed or target) == "right" else 0.1
    for _ in range(16):
        tick(setup, last_x)


def test_trials_require_observed_target_and_distinct_nonoverlapping_windows(setup):
    service = setup[0]
    run_trial(setup, "right", "left")
    assert service.status()["calibration"]["total"] == 0
    run_trial(setup, "left")
    assert service.status()["calibration"]["counts"]["left"] == 1
    run_trial(setup, "right")
    assert service.status()["calibration"]["total"] == 2
    a, b = service._trials
    assert a["end"] <= b["start"]
    assert a["end"] - a["start"] == 3
    assert len(a["features"]) == 2 and "samples" not in a
    assert service.status()["calibration"]["evaluation"]["state"] == "insufficient_evidence"
    service.stop()
    assert service.status()["calibration"]["total"] == 2
    service.reset_calibration()
    assert service.status()["calibration"]["total"] == 0


def test_mark_trial_requires_live_inputs_disarms_and_never_dispatches(setup):
    service, _, _, _, actions = setup
    assert service.mark_trial("left")["error"]
    tick(setup)
    service.arm(True)
    assert service.mark_trial("left")["ok"]
    assert not service.status()["armed"]
    assert service.arm(True)["error"]
    assert service.mark_trial("right")["error"]
    movement(setup, "left")
    assert actions == []


def test_chronological_holdout_sizes_baselines_and_training_only_scaling(monkeypatch):
    trials = [{"id": i, "label": "left" if i % 2 == 0 else "right",
               "features": [0 if i % 2 == 0 else 5, 1 if i % 2 == 0 else 4]}
              for i in range(20)]
    from core import multimodal
    original = multimodal.statistics.pstdev
    scaler_inputs = []

    def record(values):
        values = list(values)
        scaler_inputs.append(values)
        return original(values)

    monkeypatch.setattr(multimodal.statistics, "pstdev", record)
    result = MultimodalPreview._evaluate(trials)
    assert result["n_train"] == 14 and result["n_test"] == 6
    assert result["train_counts"] == {"left": 7, "right": 7}
    assert result["test_counts"] == {"left": 3, "right": 3}
    assert result["accuracy"] == result["balanced_accuracy"] == 1
    assert result["baseline_accuracy"] == result["balanced_baseline"] == 0.5
    assert scaler_inputs == [[t["features"][i] for t in trials[:14]] for i in range(2)]
    assert "not intent decoding" in result["disclaimer"]
    for trial in trials[14:]:
        trial["label"] = "left"
    assert MultimodalPreview._evaluate(trials)["accuracy"] is None


def test_contact_check_is_signal_exclusive_and_stops_without_camera(setup):
    service, clock, camera, eeg, _ = setup
    assert "Stop the preview" in service.check_contact()["error"]
    service.stop()
    service.CONTACT_SECONDS = 0
    eeg.acq["contact_precheck"] = {"values": [1234, None], "host_received_monotonic": clock.now, "units": "ohm"}
    result = service.check_contact()
    assert result["state"] == "stopped"
    assert eeg.starts == ["signal", "contact"]
    assert camera.starts == 1
    assert result["eeg"]["contact_precheck"]["values"] == [1234, None]


def test_stop_cancels_pending_start_before_camera_ownership():
    camera, eeg = Camera(), EEG()
    entered, cancelled = threading.Event(), threading.Event()

    def start(mode):
        eeg.acq["state"] = "starting"
        entered.set()
        assert cancelled.wait(2)
        return {"state": "stopped"}

    def stop():
        eeg.acq["state"] = "stopped"
        cancelled.set()
        return {"state": "stopped"}

    eeg.start_acquisition, eeg.stop_acquisition = start, stop
    service = MultimodalPreview(camera, eeg)
    thread = threading.Thread(target=service.start)
    thread.start()
    assert entered.wait(1)
    assert service.stop()["state"] == "stopped"
    thread.join(1)
    assert not thread.is_alive() and camera.starts == 0


def test_status_remains_responsive_during_device_cleanup(setup):
    service, _, camera, _, _ = setup
    entered, release = threading.Event(), threading.Event()

    def slow_stop():
        entered.set()
        assert release.wait(2)
        return {"ok": True}

    camera.stop = slow_stop
    thread = threading.Thread(target=service.stop)
    thread.start()
    assert entered.wait(1)
    started = time.monotonic()
    result = service.status()
    assert time.monotonic() - started < 0.2
    assert result["state"] == "stopping" and not result["armed"]
    release.set()
    thread.join(1)
    assert not thread.is_alive()
