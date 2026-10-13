"""Shared fast-blink sensor/decoder/native-input fakes; no device I/O."""
import copy
import math

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


class FakeClick:
    def __init__(self):
        self.clicks = 0
        self.resets = 0
        self.closes = 0
        self.fail = False
        self.cleanup_fail = False

    def click(self):
        self.clicks += 1
        return {"error": "fake injection failure"} if self.fail else {"ok": True}

    def reset(self):
        self.resets += 1
        return {"error": "fake release failure"} if self.cleanup_fail else {"ok": True}

    def close(self):
        self.closes += 1
        return {"error": "fake release failure"} if self.cleanup_fail else {"ok": True}


class FakePointer:
    """Record hand-pointer updates without creating a native mouse device."""

    eeg_compatible = True  # The controller reports finger/button ownership so EEG clicks can arbitrate safely with hand input.

    def __init__(self):
        self.updates = []  # Keep observed frames and hold flags for control-boundary assertions.
        self.moves = []  # Only unheld ready updates represent possible pointer movement.
        self.resets = []  # Reset can pause input but must never create movement.
        self.closes = 0  # Track cleanup independently from actual motion.
        self.ready = True  # Tests can simulate a pointer that is still waiting for its dwell.
        self.finger_active = False  # A deliberate finger gesture suppresses EEG selection without revoking joint authorization.
        self.button_down = False  # Owned drag input must exclude an EEG click until the finger releases it.
        self.click_id = 0  # Completed finger clicks have a monotonic identity rather than a repeated boolean event.
        self.click_source = None  # Identify the hand gesture separately from the EEG actuator's completion count.

    def update(self, points, now, hold=False):
        update = {"points": copy.deepcopy(points), "now": now, "hold": hold}  # Snapshot one distinct camera observation.
        self.updates.append(update)  # Recording a held update does not imply moving the pointer.
        if self.ready and not hold:  # The service must request a hold throughout selection and uncertainty.
            self.moves.append(update)  # A test-only list replaces every native pointer movement.
        return {"state": "mouse_pointer" if self.ready else "mouse_dwell",
                "pointer_ready": self.ready, "finger_active": self.finger_active, "button_down": self.button_down,
                "click_id": self.click_id, "click_source": self.click_source,
                "hint": "Fake hand pointer"}  # Follow the combined hand/EEG controller's explicit ownership contract.

    def reset(self, reason="Pointer paused"):
        self.resets.append(reason)  # Preserve the pause reason without emitting any input.
        self.finger_active = self.button_down = False  # Cleanup releases the fake owned button and discards an unfinished finger gesture.
        return {"state": "mouse_paused", "pointer_ready": False, "finger_active": False,
                "button_down": False, "click_id": self.click_id, "click_source": self.click_source,
                "hint": reason}  # Dwell must be re-established while completed click identities remain monotonic.

    def close(self):
        self.closes += 1  # A cleanup attempt only changes this in-memory counter.
        self.finger_active = self.button_down = False  # Disarm must release a finger drag as well as pausing pointer movement.
        return {"state": "mouse_paused", "pointer_ready": False, "finger_active": False,
                "button_down": False, "click_id": self.click_id, "click_source": self.click_source,
                "hint": "Pointer closed"}  # Confirm that no hand gesture or pressed input remains owned.


class CameraForbidden:
    """Any camera dependency breaks this explicitly camera-free method."""

    def __init__(self):
        self.calls = []

    def _called(self, name):
        self.calls.append(name)
        raise AssertionError("Automatic EEG blinking must not call camera." + name)

    def is_running(self):
        return self._called("is_running")

    def start(self):
        return self._called("start")

    def stop(self):
        return self._called("stop")

    def preview(self):
        return self._called("preview")

    def tracking_snapshot(self):
        return self._called("tracking_snapshot")


class FakeDecoder:
    """Expose causal detector events without classifying or injecting real input."""

    labels = ("rest", "select")
    feature_profile = "cnn_blink_fast_endpoint_v1"

    def __init__(self):
        self.trained = self.passed = self.eligible = True
        self.trials = []
        self.updates = []
        self.stream_resets = 0
        self.reset_calls = 0
        self.train_calls = 0
        self.bindings = []
        self.next_prediction = None
        self.prime = True
        self.watermarks = []
        self.on_update = None

    def status(self):
        return {"labels": list(self.labels), "feature_profile": self.feature_profile,
                "trained": self.trained, "passed_validation": self.passed,
                "arm_eligible": self.eligible, "state": "validated" if self.passed else
                "collecting_validation" if self.trained else "collecting_training",
                "train_counts": {label: sum(t["label"] == label and t["phase"] == "train" for t in self.trials)
                                 for label in self.labels},
                "validation_counts": {label: sum(t["label"] == label and t["phase"] == "validate" for t in self.trials)
                                      for label in self.labels},
                "evaluation": {"passed": self.passed}}

    @staticmethod
    def rest():
        return {"valid": True, "label": "rest", "confidence": 0.0, "ready": True,
                "event_id": None, "event_at": None, "reason": "Ready for a deliberate pair."}

    def event(self, event_id, event_at, **extra):
        # Similarity below 0.8 can still be accepted by the blink model's own learned threshold.
        self.next_prediction = {**self.rest(), "label": "select", "confidence": 0.64,
                                "event_id": event_id, "event_at": event_at, **extra}

    def update(self, signal):
        self.updates.append(copy.deepcopy(signal))
        result = self.next_prediction or self.rest()
        self.next_prediction = None
        if self.prime:
            self.prime = False
            self.watermarks.append(signal["samples"][-1]["counter"])
            result = self.rest()  # The first update after reset consumes buffered history without emitting an event.
        callback, self.on_update = self.on_update, None
        if callback:
            callback()  # Exercise a state change occurring while a prediction is in flight.
        return copy.deepcopy(result)

    def reset_stream(self):
        self.stream_resets += 1
        self.prime = True  # Preserve training while requiring a new stream watermark.

    def add_trial(self, label, signal, *, phase, trial_id, start, end):
        self.trials.append({"label": label, "phase": phase, "trial_id": trial_id,
                            "start": start, "end": end, "signal": copy.deepcopy(signal)})
        return {"ok": True}

    def train(self):
        self.train_calls += 1
        self.trained = True
        self.passed = self.eligible = False  # Freezing training never grants held-out validation.
        return {"ok": True}

    def reset(self):
        self.reset_calls += 1
        self.trials.clear()
        self.trained = self.passed = self.eligible = False
        self.reset_stream()
        return {"ok": True}

    def bind_prediction_context(self, context):
        self.bindings.append(copy.deepcopy(context))
        self.reset_stream()

    def export_state(self):
        return {"fake_blink_archive": True, "trials": copy.deepcopy(self.trials)}


class FakeFastCnnBlinkDecoder(FakeDecoder):
    feature_profile = "cnn_blink_fast_endpoint_v1"
    window_seconds = trial_seconds = cooldown_seconds = 1.0

    def __init__(self, *, clock=None):
        super().__init__()
        self.cancel_calls = 0

    def cancel_pending(self):
        self.cancel_calls += 1  # Service tests verify ownership edges separately from full stream resets.

    def status(self):
        return {**super().status(), "window_seconds": 1.0, "trial_seconds": 1.0,
                "validation_scope": "held_out_one_second_trials"}

    def add_trial(self, label, signal, *, phase, trial_id, start, end, **extra):
        result = super().add_trial(label, signal, phase=phase, trial_id=trial_id, start=start, end=end)
        self.trials[-1].update(extra)
        return result

    def export_state(self):
        return {"format": "intuitionos.eeg-fast-cnn-blink-model", "fake": True}


class FastHarness:
    def __init__(self, monkeypatch):
        self.clock, self.camera, self.eeg = Clock(), CameraForbidden(), EEG()
        self.eeg.acq["channels"] = [{"num": i, "name": name}
                                    for i, name in enumerate(("O2", "T3", "T4", "O1"))]
        self.mouse = FakeClick()
        self.guard_active = False
        self.guard_sequence = 0
        self.guard_token = None
        self.started_at = self.clock.now
        monkeypatch.setattr(multimodal, "FastCnnBlinkDecoder", FakeFastCnnBlinkDecoder)
        self.service = multimodal.MultimodalPreview(
            self.camera, self.eeg, lambda: False, clock=self.clock,
            click_controller=self.mouse, pointer_controller=None, camera_mouse_enabled=False)
        self.service._launch_worker = lambda: None
        self.decoder = self.service._decoder
        assert isinstance(self.decoder, FakeFastCnnBlinkDecoder)
        assert self.service.start().get("ok")
        self.tick(3)

    def control(self):
        return self.service.status()["eeg_control"]

    def tick(self, count=1, *, age=0.0, gap=0, step=0.125, add_rows=True):
        for _ in range(count):
            self.clock.now += step
            if add_rows:
                target = round((self.clock.now - self.started_at) * 250)
                while self.eeg.counter < target:
                    n = self.eeg.counter
                    self.eeg.rows.append({"counter": n,
                                          "estimated_monotonic": self.started_at + n / 250,
                                          "host_received_monotonic": self.clock.now,
                                          "samples": [math.sin(n / 10 + i) * 1e-6 for i in range(4)]})
                    self.eeg.counter += 1
                self.eeg.rows = self.eeg.rows[-1250:]
            self.eeg.acq["stats"] = {"age_seconds": age, "gaps": gap, "queue_drops": 0,
                                      "duplicates": 0, "nonfinite": 0,
                                      "channel_mismatches": 0, "received_rate_hz": 250}
            if self.guard_active and self.service.eeg_guard(self.guard_token).get("error"):
                self.guard_active = False
            self.service._tick()

    def arm(self):
        self.guard_sequence += 1
        self.guard_token = "automatic-eeg-test-guard-" + str(self.guard_sequence)
        assert self.service.eeg_guard(self.guard_token).get("ok")
        self.guard_active = True
        outcome = self.service.eeg_arm(True, guard_token=self.guard_token)
        assert not outcome.get("error"), outcome
        assert self.control()["armed"]
        self.tick()  # Consume buffered history at a new post-arm watermark.

    def emit(self, event_id="new-pair", *, at=None, **extra):
        self.decoder.event(event_id, self.clock.now if at is None else at, **extra)
        self.tick()


class CameraHarness(FastHarness):
    def __init__(self, monkeypatch):
        self.show_hand = True
        self.frame_sequence = 0
        super().__init__(monkeypatch)
        self.camera = Camera()
        self.service.camera = self.camera
        self.pointer = FakePointer()
        self.service.pointer_controller = self.pointer

    def tick(self, count=1, **kwargs):
        for _ in range(count):
            if getattr(self.camera, "running", False):
                self.frame_sequence += 1
                self.camera.frame = {"running": True, "state": "running", "tracked": self.show_hand,
                                     "sequence": self.frame_sequence, "age_ms": 0, "fps": 30,
                                     "landmarks": [[.5, .5, 0] for _ in range(21)] if self.show_hand else []}
            super().tick(1, **kwargs)

    def enable(self):
        assert self.service.eeg_camera(True).get("ok")
        self.tick(2)
