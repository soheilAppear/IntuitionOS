"""Real click calibration and inference using synthetic EEG and fake input only.

The deliberately separated synthetic frequencies test software integration;
they do not establish that eye closure will reliably control a real headset.
"""

import math  # Generate known finite waveforms instead of touching an EEG device.

import numpy as np  # Add repeatable noise to independent calibration and validation samples.

from core import multimodal  # Restore production feature extraction in the shared fake-sensor harness.
from core.eeg_decoder import CLICK_FEATURE_PROFILE, EegDecoder, extract_features  # Exercise the real processing contract.
from test_eeg_click_control import FakePointer  # Reuse an EEG-compatible hand-input recorder that never initializes desktop input.
from test_eeg_control import Harness  # Reuse fake time, cached sensors, and a disabled background worker.


class FakeClickActuator:
    """Record completed clicks without constructing a native mouse adapter."""

    def __init__(self):
        self.clicks = 0  # Count only calls that the service explicitly dispatches.
        self.resets = 0  # Show when mouse-target arming prepares the fake actuator.

    def click(self):
        self.clicks += 1  # This test counter is the only effect of a simulated mouse click.
        return {"ok": True, "cleanup_pending": False}  # Emulate a confirmed button press and release.

    def reset(self):
        self.resets += 1  # No physical input is emitted while preparing this fake actuator.
        return {"ok": True, "cleanup_pending": False}  # Report that no button is held.

    def close(self):
        return {"ok": True, "cleanup_pending": False}  # Cleanup also has no hardware or input effects.


class RealClickHarness(Harness):
    """Drive real binary features/classification from bounded synthetic packets."""

    def __init__(self, monkeypatch):
        super().__init__(monkeypatch, mode="eeg_click")  # Start the fake camera and EEG together for hand aiming plus selection.
        monkeypatch.setattr(multimodal, "extract_features", extract_features)  # Replace the parent harness's fake extractor.
        self.decoder = self.service._decoder  # Keep the real binary decoder created by set_control_mode.
        assert isinstance(self.decoder, EegDecoder)  # Fail if this integration accidentally substitutes the boundary-test fake.
        self.eeg.acq["channels"] = [  # Match the four logical channels used by the BrainBit2 adapter.
            {"num": index, "name": name} for index, name in enumerate(("O2", "T3", "T4", "O1"))
        ]
        self.mouse = FakeClickActuator()  # Supply a strictly in-memory replacement for desktop injection.
        self.service.click_controller = self.mouse  # Real click authorization still goes through the production service.
        self.pointer = FakePointer()  # The pointer recorder validates orchestration without issuing any native movement.
        self.service.pointer_controller = self.pointer  # Inject the required hand/finger arbitration contract before explicit arming.
        self.signal_label = "rest"  # Begin with a stable synthetic neutral state before any cue.
        self.raw_valid = True  # Tests can introduce a raw flatline independently of classifier uncertainty.
        self.rng = np.random.default_rng(731)  # Reproducible noise keeps failures deterministic.

    def tick(self, count=1, *, add_rows=True, **kwargs):
        for _ in range(count):  # Keep the production 125 ms observation cadence without wall-clock sleeps.
            if add_rows:  # Allow the same cached-sample behavior as the existing service harness.
                target = round((self.clock.now + 0.125 - self.started_at) * 250)  # Generate the next full-rate batch.
                while self.eeg.counter < target:  # Preserve every original counter instead of interpolating a display waveform.
                    counter = self.eeg.counter  # One packet contains all four simultaneous channel samples.
                    frequency = 20 if self.signal_label == "rest" else 10  # Known synthetic bands distinguish the two test cues.
                    angle = 2 * math.pi * frequency * counter / 250  # Maintain sampling at the nominal 250 Hz.
                    line = 120e-6 * math.sin(2 * math.pi * 60 * counter / 250)  # Include line interference without using it as a label.
                    values = [  # Give each channel its own amplitude, phase, and independent finite noise.
                        amplitude * math.sin(angle + phase) + line + self.rng.normal(0, 0.5e-6)
                        for amplitude, phase in ((20e-6, 0), (12e-6, 0.6), (14e-6, 1.2), (16e-6, 1.8))
                    ]
                    self.eeg.rows.append({  # Keep the same packet schema as the production cached acquisition API.
                        "counter": counter,
                        "estimated_monotonic": self.started_at + counter / 250,
                        "host_received_monotonic": self.clock.now + 0.125,
                        "samples": values if self.raw_valid else [0.0] * 4,
                    })
                    self.eeg.counter += 1  # Every generated packet gets a unique chronological sample identity.
                self.eeg.rows = self.eeg.rows[-1250:]  # Enforce the real five-second cached raw-window limit.
            super().tick(add_rows=False, **kwargs)  # Update freshness and run the real service tick without replacing samples.

    def capture(self, label, phase):
        self.signal_label = label  # Follow the visible cue using a controlled synthetic spectral state.
        key = "train_counts" if phase == "train" else "validation_counts"  # Check the phase's public counter only.
        before = self.control()["decoder"][key][label]  # Retain the count to prove one complete trial is accepted.
        response = self.service.eeg_trial(label, phase)  # Calibrate through the real cue-labelled EEG service entry point.
        assert response.get("ok"), response  # Reject failures at trial initiation instead of silently skipping evidence.
        self.tick(32, tracked=False)  # Keep the requested state through the four-second closed-eye cue period.
        self.signal_label = "rest"  # Reopening at four seconds must not contaminate the earlier measured window.
        self.tick(8, tracked=False)  # Finish the fifth second, allowing the service to extract and retain the trial.
        control = self.control()  # Read the actual public outcome after feature extraction and classifier submission.
        assert control["pending"] is None  # No partial trial may remain pending after the full cue interval.
        assert control["decoder"][key][label] == before + 1, control["last_result"]  # Prove the real decoder accepted this window.
        assert self.mouse.clicks == 0  # Training and validation must never dispatch real or fake desktop input.
        assert self.pointer.updates == []  # A running camera does not authorize hand-pointer movement during calibration.

    def calibrate(self):
        self.warm()  # Supply the first complete two-second raw window before requesting a trial.
        for _ in range(8):  # Collect the unchanged minimum number of training trials for each class.
            for label in self.decoder.labels:  # Alternate neutral and selection cues instead of creating chronological blocks.
                self.capture(label, "train")  # Use service timing, raw extraction, and model storage end to end.
        assert self.service.eeg_train().get("ok")  # Freeze the real binary model before collecting any validation data.
        assert not self.control()["can_arm"]  # Training alone must not authorize a practice selection or mouse click.
        frozen = {key: value.copy() for key, value in self.decoder._model.items()}  # Preserve every fitted parameter for a holdout check.
        for _ in range(8):  # The validation set contains exactly eight new complete trials per class.
            for label in self.decoder.labels:  # Continue the same cue order without reusing sample windows.
                self.capture(label, "validate")  # Validation uses the already frozen feature transform and classifier.
        status = self.control()["decoder"]  # Read production metrics instead of forcing eligibility in a fake decoder.
        assert status["passed_validation"], status["evaluation"]  # Known synthetic classes must pass the unchanged engineering gates.
        assert status["train_counts"] == status["validation_counts"] == {"rest": 8, "select": 8}  # Verify the exact fixed split.
        assert status["evaluation"]["rest_false_activations"] == 0  # No held-out rest trial may have predicted a selection.
        assert status["feature_profile"] == CLICK_FEATURE_PROFILE  # Every accepted trial belongs to the click processing version.
        for key, expected in frozen.items():  # Validation must not fit normalization, centroids, variance, or rejection radii.
            np.testing.assert_array_equal(self.decoder._model[key], expected)  # Compare fitted parameters exactly after holdout.
        assert not self.service.eeg_trial("rest", "validate").get("ok")  # A completed holdout cannot be extended to cherry-pick outcomes.
        self.tick(24, tracked=True)  # Settle EEG and provide a fresh tracked hand before any explicit arm request.

    def selection_cycle(self):
        self.signal_label = "select"  # Introduce a genuine mixed raw feature window at the rest-to-select transition.
        saw_uncertain = False  # Confirm that the test exercises transition handling instead of swapping perfect labels.
        for _ in range(28):  # Hold selection long enough for a full window and three distinct confident predictions.
            self.tick(tracked=True)  # Fresh hand observations keep the intended pointer target valid during the EEG transition.
            prediction = self.control().get("prediction") or {}  # Read the real inference result for this spectral mixture.
            saw_uncertain |= prediction.get("label") is None  # Count abstention or OOD as a suppressed transition.
            assert self.control()["armed"], self.control()["reason"]  # Brief valid-raw transitions must preserve deliberate authorization.
        assert self.control()["selection_pending"]  # Sustained select only prepares an action; it cannot click yet.
        self.signal_label = "rest"  # Complete the voluntary protocol by returning to the calibrated neutral state.
        for _ in range(28):  # Allow the reopened/rest spectrum and the one-second rest debounce to complete.
            self.tick(tracked=True)  # Continue real feature extraction while maintaining fresh tracked hand frames.
            prediction = self.control().get("prediction") or {}  # Observe the return transition without changing the classifier.
            saw_uncertain |= prediction.get("label") is None  # Ensure at least one mixed-window abstention is traversed.
            assert self.control()["armed"], self.control()["reason"]  # A normal completed cycle must remain ready for later deliberate selections.
        assert saw_uncertain  # Prevent this integration test from accidentally reducing to mocked perfect predictions.
        assert not self.control()["selection_pending"]  # Exactly one completed cycle consumes the pending selection.


def test_real_cued_calibration_validation_hand_aim_practice_then_explicit_mouse(monkeypatch):
    h = RealClickHarness(monkeypatch)  # All sensors, clocks, and desktop effects stay in memory.
    h.calibrate()  # Exercise service-level cue timing, real extraction, freeze, and fixed held-out validation.
    assert h.camera.starts == 1 and h.camera.reads > 0  # Click mode owns live camera preview throughout calibration and validation.
    assert h.pointer.updates == []  # Viewing calibration frames alone must not cause pointer input.
    assert h.control()["target"] == "practice"  # Completed validation cannot silently enable mouse output.
    h.arm()  # Explicitly prepare the fake Escape guard and arm the practice destination.
    h.tick(12, tracked=True)  # Establish confident rest and a current hand-pointer target before attempting selection.
    assert h.control()["neutral_ready"]  # Arming alone does not replace the initial rest requirement.
    assert h.pointer.moves and h.control()["hand_pointer"]["ready"]  # The same EEG arm explicitly enables hand aiming.
    h.selection_cycle()  # Drive the real classifier through rest, select, and reopened/rest windows.
    assert h.control()["selection_count"] == 1  # One complete cycle increments the local practice counter once.
    assert h.mouse.clicks == h.control()["click_count"] == 0  # Practice cannot touch the fake native actuator.
    h.tick(24, tracked=True)  # Holding rest and tracking the hand after completion must not repeat the action.
    assert h.control()["selection_count"] == 1  # Confirm exactly-once practice behavior.
    assert h.service.eeg_target("mouse").get("ok")  # The user must separately choose desktop output.
    assert not h.control()["armed"]  # Changing output invalidates the old practice authorization and lease.
    updates_before = len(h.pointer.updates)  # Revoking the practice authorization also pauses hand aiming.
    h.signal_label = "select"  # A deliberate signal pattern while disarmed still must not cause a click.
    h.tick(28, tracked=True)  # Run real inference with visible hands through the disarmed selection transition.
    h.signal_label = "rest"  # Return to rest without granting new authorization.
    h.tick(28, tracked=True)  # Complete the signal cycle while the mouse destination remains disarmed.
    assert h.mouse.clicks == 0  # Explicit target selection alone must never activate the mouse.
    assert len(h.pointer.updates) == updates_before  # A fresh hand cannot restore movement without the new explicit arm.
    h.arm()  # A new Escape lease and explicit arm request authorize the selected mouse target.
    assert h.mouse.resets == 1  # Only real mouse arming prepares the fake actuator.
    h.tick(12, tracked=True)  # Require a fresh rest interval and pointer target after the new authorization.
    h.selection_cycle()  # The same real binary model can now complete one guarded mouse action.
    assert h.mouse.clicks == h.control()["click_count"] == 1  # Count only a confirmed complete fake mouse click.
    assert h.control()["selection_count"] == 1  # Mouse output does not falsely increment practice feedback.
    assert h.actions == []  # EEG selection must never call the old desktop-swipe dispatcher.
    assert h.camera.starts == 1 and h.camera.reads > 0  # The same owned camera supplies hand aiming without a second camera session.


def test_real_click_trials_require_fresh_camera_but_not_tracked_hand_labels(monkeypatch):
    h = RealClickHarness(monkeypatch)  # The production decoder still owns all feature and profile checks.
    h.warm()  # Supply valid EEG and fresh preview frames before starting the calibration sequence.
    h.tick(tracked=False)  # Remove the hand while the fake camera continues delivering fresh frames.
    h.capture("rest", "train")  # A complete neutral trial uses its EEG cue, not a verified hand gesture.
    h.capture("select", "train")  # A complete selection trial also relies on its explicit cue rather than camera labels.
    assert h.service.status()["state"] == "running"  # Absent hands do not cancel a disarmed EEG calibration session.
    assert h.control()["decoder"]["train_counts"] == {"rest": 1, "select": 1}  # Both real feature windows reached the binary decoder.
    assert h.camera.starts == 1 and h.camera.reads > 0  # Fresh preview frames remain part of the combined-mode contract.
    assert h.pointer.updates == []  # Neither fresh frames nor completed calibration cues authorize hand aiming.
    assert h.mouse.clicks == 0 and h.actions == []  # Calibration remains free of mouse input and desktop switching.


def test_real_click_calibration_stops_when_its_owned_camera_fails(monkeypatch):
    h = RealClickHarness(monkeypatch)  # Use production feature processing with fake sensors and pointer input only.
    h.warm()  # The preview first reaches its ordinary camera-plus-EEG running state.
    assert h.service.eeg_trial("rest").get("ok")  # Begin a real cue interval before introducing the camera failure.
    monkeypatch.setattr(h.camera, "preview", lambda: {"running": False, "state": "error", "hint": "Synthetic camera failure"})  # Model a fatal capture result.
    h.tick()  # The combined mode must stop and discard the partial cue rather than accept stale camera state.
    assert not h.service.status()["running"]  # Owned-camera failure cannot silently degrade into camera-free clicking.
    assert h.control()["pending"] is None  # Interrupted calibration must not remain available for later completion.
    assert h.control()["decoder"]["train_counts"] == {"rest": 0, "select": 0}  # No partial trial reaches the real decoder.
    assert h.pointer.updates == [] and h.mouse.clicks == 0  # Failure and cleanup must never start pointer or click input.
