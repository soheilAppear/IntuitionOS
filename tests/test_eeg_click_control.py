"""Exercise EEG selection cycles with fake sensors, time, and mouse input only."""

import copy  # Preserve a fixed camera frame when testing replayed-frame protection.
import threading

import pytest

from core.eeg_decoder import CLICK_FEATURE_PROFILE, EegDecoder
from test_eeg_control import Harness


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


class ClickHarness(Harness):
    def __init__(self, monkeypatch):
        super().__init__(monkeypatch, mode="eeg_click")  # Click mode owns both camera and EEG from the outset.
        self.mouse = FakeClick()
        self.service.click_controller = self.mouse
        self.pointer = FakePointer()  # A fake pointer permits arming without invoking native input.
        self.service.pointer_controller = self.pointer  # Inject before any armed hand frames can be processed.
        self.real_decoder = self.service._decoder
        self.service._decoder = self.decoder
        self.service._decoder_status = self.decoder.status()
        self.profiles = []
        original_extract = self.extract

        def extract(*args, feature_profile=None, **kwargs):
            self.profiles.append(feature_profile)
            result = original_extract(*args, **kwargs)
            if result.get("ok") and feature_profile:
                result["context"]["feature_profile"] = feature_profile
            return result

        monkeypatch.setattr("core.multimodal.extract_features", extract)
        self.warm()

    def select(self):
        self.decoder.label = "select"
        self.tick(8, tracked=True)  # Fresh tracked hands remain required while an EEG selection is armed.

    def release(self):
        self.decoder.label = "rest"
        self.tick(12, tracked=True)  # Reopened/rest completion also requires a current hand-pointer target.


@pytest.fixture
def click(monkeypatch):
    return ClickHarness(monkeypatch)


def test_click_mode_owns_camera_and_binary_profile_without_unarmed_input(click):
    assert click.real_decoder.labels == ("rest", "select")
    assert click.real_decoder.feature_profile == CLICK_FEATURE_PROFILE
    assert click.service.status()["state"] == "running"
    assert click.camera.starts == 1 and click.camera.reads > 0  # Camera preview runs so a hand can aim after explicit arming.
    assert set(click.profiles) == {CLICK_FEATURE_PROFILE}
    assert click.control()["target"] == "practice"
    assert click.mouse.clicks == 0
    assert click.pointer.updates == []  # Starting preview must not move the pointer before EEG arming.
    assert click.control()["hand_pointer"]["required"]  # Status makes the hand-target dependency explicit.
    assert not click.control()["hand_pointer"]["ready"]  # Preview alone cannot satisfy armed pointer dwell.


def test_practice_needs_rest_select_rest_and_never_calls_mouse(click):
    click.arm()
    click.select()
    assert not click.control()["selection_pending"]
    click.release()
    click.select()
    assert click.control()["selection_pending"]
    assert click.control()["selection_count"] == 0
    click.tick(16, tracked=True)  # Holding select with a visible hand must still produce no early click.
    assert click.control()["selection_count"] == 0
    click.release()
    assert click.control()["selection_count"] == 1
    click.tick(24, tracked=True)  # Continued rest and hand tracking cannot repeat a completed selection.
    assert click.control()["selection_count"] == 1
    assert click.mouse.clicks == 0
    assert click.actions == []


def test_mouse_requires_explicit_target_and_rearm_and_clicks_only_after_release(click):
    click.arm()
    click.neutral()
    click.select()
    assert click.service.eeg_target("mouse").get("ok")
    assert not click.control()["armed"]
    assert not click.control()["selection_pending"]
    click.release()
    assert click.mouse.clicks == 0
    click.arm()
    assert click.mouse.resets == 1
    click.neutral()
    click.select()
    assert click.mouse.clicks == 0
    click.release()
    assert click.mouse.clicks == click.control()["click_count"] == 1
    assert click.control()["selection_count"] == 0
    assert click.actions == []
    click.tick(24)
    assert click.mouse.clicks == 1


@pytest.mark.parametrize("method", ["stop", "disarm", "target", "reset", "stale", "gap", "session", "guard", "hand", "camera"])
def test_pending_selection_cannot_survive_revocation(click, method):
    click.service.eeg_target("mouse")
    click.arm()
    click.neutral()
    click.select()
    assert click.control()["selection_pending"]
    if method == "stop":
        click.service.stop()
    elif method == "disarm":
        click.service.eeg_arm(False)
    elif method == "target":
        click.service.eeg_target("practice")
    elif method == "reset":
        click.service.eeg_reset()
    elif method == "stale":
        click.tick(eeg_age=1)
    elif method == "gap":
        click.tick(gap=1)
    elif method == "session":
        click.eeg.acq["session_id"] = "replacement-session"
        click.tick()
    elif method == "hand":
        click.tick(tracked=False)  # Losing the hand immediately revokes a previously prepared EEG selection.
    elif method == "camera":
        click.camera.preview = lambda: {"running": False, "state": "error", "hint": "fake camera failure"}  # Fail capture without invoking hardware.
        click.tick()  # The combined mode must stop rather than retaining a stale pointer target.
    else:
        click.guard_active = False
        click.tick(25)
    pointer_updates_at_revocation = len(click.pointer.updates)  # Revoked EEG authorization must also stop hand aiming.
    click.release()
    assert click.mouse.clicks == 0
    assert click.control()["armed"] is (method in ("stale", "gap", "session", "hand"))  # Sensor interruptions pause only EEG/hand targeting; explicit revocation still ends the shared arm.
    assert not click.control()["selection_pending"]
    if method in ("stale", "gap", "session", "hand"):
        assert len(click.pointer.updates) > pointer_updates_at_revocation  # Fresh hands can resume aiming without repeating the shared arm.
    else:
        assert len(click.pointer.updates) == pointer_updates_at_revocation  # Stop, Escape, and explicit changes still require a new arm.


def test_replayed_window_cannot_sustain_a_selection(click):
    click.arm()
    click.neutral()
    click.decoder.label = "select"
    click.decoder.fixed_window_id = "same-window"
    click.tick(16)
    assert not click.control()["selection_pending"]
    assert click.control()["selection_count"] == 0


def test_selection_expires_instead_of_clicking_after_long_eye_closure(click):
    click.arm()
    click.neutral()
    click.select()
    click.tick(72)
    click.release()
    assert click.control()["armed"]  # Expired EEG selection leaves independent hand controls available.
    assert click.control()["selection_count"] == 0


def test_click_trial_is_five_seconds_and_does_not_require_a_tracked_hand(click):
    click.decoder.trained = click.decoder.eligible = click.decoder.passed = False
    click.service._decoder_status = click.decoder.status()
    start = click.clock.now
    result = click.service.eeg_trial("select")
    assert not result.get("error"), result
    assert result["eeg_control"]["pending"]["remaining_seconds"] == 5
    assert "close your eyes" in result["eeg_control"]["pending"]["cue"]
    click.tick(32, tracked=False)
    assert "Open your eyes" in click.control()["pending"]["cue"]
    click.tick(8, tracked=False)
    assert len(click.decoder.trials) == 1
    trial = click.decoder.trials[0]
    assert trial["label"] == "select"
    assert trial["window"]["start"] >= start + 1
    assert trial["window"]["end"] < start + 4
    assert trial["context"]["feature_profile"] == CLICK_FEATURE_PROFILE
    assert click.camera.starts == 1 and click.camera.reads > 0  # A fresh camera may run without seeing a hand during calibration.
    assert click.mouse.clicks == 0
    assert click.pointer.updates == []  # Cue labels never authorize pointer movement during a trial.


def test_click_trial_dropout_does_not_add_partial_training(click):
    click.decoder.trained = False
    click.service._decoder_status = click.decoder.status()
    assert click.service.eeg_trial("rest").get("ok")
    click.tick(4)
    click.tick(gap=1)
    click.tick(40)
    assert click.decoder.trials == []
    assert click.control()["pending"] is None


@pytest.mark.parametrize("label", ["left", "right", "click", None])
def test_click_mode_rejects_swipe_or_unknown_trial_labels(click, label):
    assert click.service.eeg_trial(label).get("error")
    assert click.control()["pending"] is None


def test_mode_change_discards_model_evidence_and_returns_to_practice(click):
    click.service.eeg_target("mouse")
    assert click.service.set_control_mode("eeg").get("ok")
    assert isinstance(click.service._decoder, EegDecoder)
    assert click.service._decoder.labels == ("left", "right", "rest")
    assert click.service.set_control_mode("eeg_click").get("ok")
    assert click.service._decoder.labels == ("rest", "select")
    assert not click.control()["decoder"]["trained"]
    assert click.control()["target"] == "practice"


def test_failed_click_disarms_and_unconfirmed_release_blocks_restart(click):
    click.service.eeg_target("mouse")
    click.arm()
    click.neutral()
    click.select()
    click.mouse.fail = click.mouse.cleanup_fail = True
    click.release()
    assert click.mouse.clicks == 1
    assert click.control()["click_count"] == 0
    assert not click.control()["armed"]
    assert click.service.status()["cleanup_pending"]
    assert click.service.stop().get("error")
    assert click.service.start().get("error")
    click.mouse.cleanup_fail = False
    assert click.service.stop().get("ok")
    assert not click.service.status()["cleanup_pending"]


@pytest.mark.parametrize("change", ["target", "mode", "disarm"])
def test_revocation_while_click_waits_at_final_boundary_prevents_input(click, change):
    click.service.eeg_target("mouse")
    click.arm()
    click.neutral()
    click.select()
    click.decoder.label = "rest"
    click.tick(8)
    entered = threading.Event()
    proceed = threading.Event()  # Let hand-pointer updates finish before blocking the later click authorization boundary.
    revoked = threading.Event()
    completed = threading.Event()
    original_dispatch = click.service._dispatch_eeg_click
    original_disarm = click.service._disarm_eeg_locked

    def delayed(*args):
        entered.set()
        assert proceed.wait(3)  # Pause immediately before the click's action lock, after pointer processing has completed.
        return original_dispatch(*args)

    def disarm(*args, **kwargs):
        result = original_disarm(*args, **kwargs)
        revoked.set()
        return result

    def change_target():
        if change == "target":
            click.service.eeg_target("practice")
        elif change == "mode":
            click.service.set_control_mode("eeg")
        else:
            click.service.eeg_arm(False)
        completed.set()

    click.service._dispatch_eeg_click = delayed
    click.service._disarm_eeg_locked = disarm
    worker = threading.Thread(target=lambda: click.tick(8))  # The final hand frame must be allowed to reach its own action boundary first.
    worker.start()  # Advance the return-to-rest cycle until it reaches the delayed click dispatch.
    assert entered.wait(2)  # The wrapper proves that pointer processing and EEG selection are both complete.
    try:
        with click.service._action_lock:  # Queue the prepared click at the real final input boundary.
            changer = threading.Thread(target=change_target)  # Revoke state from another thread while input is unavailable.
            changer.start()  # Target, mode, and disarm changes must revoke before waiting for the action barrier.
            assert revoked.wait(2)  # Verify that cancellation happened before the pending click can acquire its lock.
            assert not completed.is_set()  # The user-facing change must still wait for the input barrier.
            proceed.set()  # The prepared click can now queue for a lock with already-revoked authorization.
    finally:
        proceed.set()  # A failed assertion must not leave the synthetic worker blocked indefinitely.
    worker.join(2)
    changer.join(2)
    assert not worker.is_alive() and not changer.is_alive()
    assert completed.is_set()
    assert click.mouse.clicks == 0
    assert click.control()["selection_count"] == 0


@pytest.mark.parametrize("change", ["target", "mode", "disarm"])
def test_inflight_click_finishes_before_change_returns_without_stale_status(click, change):
    click.service.eeg_target("mouse")
    click.arm()
    click.neutral()
    click.select()
    entered, finish, revoked, completed = (threading.Event() for _ in range(4))
    original_disarm = click.service._disarm_eeg_locked

    def slow_click():
        entered.set()
        assert finish.wait(3)
        return {"ok": True}

    def disarm(*args, **kwargs):
        result = original_disarm(*args, **kwargs)
        revoked.set()
        return result

    def change_target():
        if change == "target":
            click.service.eeg_target("practice")
        elif change == "mode":
            click.service.set_control_mode("eeg")
        else:
            click.service.eeg_arm(False)
        completed.set()

    click.mouse.click = slow_click
    click.service._disarm_eeg_locked = disarm
    worker = threading.Thread(target=click.release)
    worker.start()
    assert entered.wait(2)
    changer = threading.Thread(target=change_target)
    changer.start()
    try:
        assert revoked.wait(2)
        assert not completed.is_set()
    finally:
        finish.set()
    worker.join(2)
    changer.join(2)
    assert not worker.is_alive() and not changer.is_alive()
    assert completed.is_set()
    assert not click.control()["armed"]
    assert click.control()["click_count"] == 0
    assert "One left click completed" not in click.control()["reason"]


@pytest.mark.parametrize("operation", ["reset", "close"])
def test_unexpected_cleanup_exception_is_reported_and_keeps_input_disarmed(click, operation):
    click.service.eeg_target("mouse")

    def fail_cleanup():
        raise RuntimeError("adapter cleanup exception")

    setattr(click.mouse, operation, fail_cleanup)
    if operation == "reset":
        result = click.arm_request()
    else:
        result = click.service.eeg_arm(False)
    assert "adapter cleanup exception" in result["error"]
    assert not click.control()["armed"]
    assert click.service.status()["cleanup_pending"]


def test_guard_lease_without_explicit_arm_cannot_move_or_click(click):
    assert click.service.eeg_guard("lease-without-arm").get("ok")  # Preparing Escape alone is not input authorization.
    click.tick(8, tracked=True)  # Feed fresh visible hands and valid EEG while controls remain disarmed.
    assert click.pointer.updates == []  # The service must not invoke even a fake pointer before arming.
    assert click.mouse.clicks == 0  # A guard heartbeat cannot trigger desktop input.
    assert not click.control()["armed"]  # Status must not imply that a valid guard armed the system.


@pytest.mark.parametrize("fault", ["missing", "incompatible", "untracked"])
def test_arming_requires_compatible_hand_controller_and_fresh_tracked_hand(click, fault):
    if fault == "missing":
        click.service.pointer_controller = None  # A practice counter still requires a usable hand-aiming path.
    elif fault == "incompatible":
        click.pointer.eeg_compatible = False  # Hand input must expose arbitration metadata before it can share EEG authorization.
    else:
        click.tick(tracked=False)  # A running camera alone cannot establish a current hand target.
    result = click.arm_request()  # Exercise the public guard and arm entry points.
    assert result.get("error"), result  # Invalid targeting prerequisites must fail closed before native input.
    assert not click.control()["armed"]  # A failed attempt must not leave partial input authorization behind.
    assert click.pointer.updates == [] and click.mouse.clicks == 0  # No movement or click may occur during failed arming.


def test_pointer_dwell_is_required_after_arming_and_before_selection(click):
    click.pointer.ready = False  # Simulate a hand that has not yet completed the pointer controller's dwell.
    click.arm()  # Arming may precede dwell when fresh hand tracking and the correct controller are present.
    click.tick(12, tracked=True)  # Supply valid EEG rest without pretending the pointer target is ready.
    assert not click.control()["hand_pointer"]["ready"]  # Publish the targeting wait independently of EEG readiness.
    click.select()  # A strong EEG selection cannot substitute for a ready pointer target.
    click.release()  # Completing the cue must not count an action against an unconfirmed target.
    assert click.mouse.clicks == click.control()["selection_count"] == 0  # Both practice and real output remain suppressed.
    assert click.pointer.moves == []  # The dwell simulation must also remain free of pointer movement.


def test_pointer_moves_for_armed_rest_and_holds_during_pending_selection(click):
    click.arm()  # One explicit EEG authorization enables both hand aiming and selection.
    click.neutral()  # Give the fake pointer distinct tracked frames while the classifier recognizes rest.
    assert click.pointer.moves  # Armed neutral hand updates may aim the normal pointer.
    assert click.control()["hand_pointer"]["ready"]  # A ready pointer is advertised only after armed updates.
    click.select()  # Confirm the EEG selection while preserving the intended pointer target.
    assert click.control()["selection_pending"]  # The pointer must stay fixed while this cycle awaits reopening/rest.
    updates_before = len(click.pointer.updates)  # Isolate the subsequent pending interval from earlier aim frames.
    moves_before = len(click.pointer.moves)  # Preserve the exact movement count at the start of that interval.
    click.tick(8, tracked=True, x=0.8)  # Deliberately move the fake hand while EEG remains selected.
    assert len(click.pointer.updates) > updates_before  # Held frames still validate hand pose and freshness.
    assert all(update["hold"] for update in click.pointer.updates[updates_before:])  # Every pending-frame update must hold the target.
    assert len(click.pointer.moves) == moves_before  # Hand motion cannot change the pointer while a click is pending.
    click.decoder.label = None  # Introduce a brief valid-raw classifier transition rather than a raw data error.
    click.tick(4, tracked=True)  # Transition grace may preserve authorization while suppressing pointer motion.
    assert len(click.pointer.moves) == moves_before  # An already confirmed target stays fixed through brief valid-raw transition windows.
    assert click.mouse.clicks == click.control()["selection_count"] == 0  # Neither select nor uncertainty can complete the action.


def test_repeated_camera_frame_never_replays_pointer_motion_and_eventually_pauses(click):
    click.arm()  # Obtain deliberate authorization before checking frame identity protection.
    click.neutral()  # Establish a ready pointer and valid EEG rest from distinct camera frames.
    frozen_frame = copy.deepcopy(click.camera.frame)  # Preserve the most recently processed image and sequence number.
    click.camera.preview = lambda: copy.deepcopy(frozen_frame)  # Repeated cached previews contain no new hand observation.
    updates_before = len(click.pointer.updates)  # Only truly new sequences may reach the pointer controller.
    click.tick(2, tracked=True)  # Two replayed frames are still within the camera freshness interval.
    assert len(click.pointer.updates) == updates_before  # Cached images cannot repeatedly move the desktop pointer.
    click.tick(4, tracked=True)  # The unchanged sequence must eventually become stale despite fresh EEG.
    assert click.control()["armed"]  # Brief stale targeting pauses input while retaining the explicit arm.
    assert not click.control()["hand_pointer"]["ready"]  # Retained authorization never makes the stale frame actionable.
    click.tick(16)  # A sustained camera-frame failure still stops the service.
    assert not click.control()["armed"]
    assert not click.control()["selection_pending"]  # A later new frame cannot complete an old selection.
    assert click.mouse.clicks == 0  # Replayed frames never produce a mouse click.


def test_full_rate_tracking_moves_pointer_while_jpeg_and_overlay_remain_cached(click):
    click.arm()
    click.neutral()
    display = copy.deepcopy(click.camera.frame)
    click.camera.preview = lambda: copy.deepcopy(display)  # Simulate a cached JPEG with its correctly aligned older skeleton.
    click.camera.tracking_snapshot = lambda: copy.deepcopy(click.camera.frame)  # Inference continues at the hand-control rate.
    updates_before = len(click.pointer.updates)

    click.tick(30, x=0.7, step=1 / 30)  # One second contains thirty distinct hand frames despite the unchanged display frame.

    assert len(click.pointer.updates) - updates_before == 30
    assert all(update["points"][0][0] == 0.7 for update in click.pointer.updates[updates_before:])
    assert click.control()["armed"] and click.control()["hand_pointer"]["ready"]
    assert click.service.status()["camera"]["sequence"] == click.camera.frame["sequence"]
    snapshot = click.service.snapshot()
    assert snapshot["camera"]["sequence"] == display["sequence"]
    assert snapshot["camera"]["landmarks"] == display["landmarks"]
    assert snapshot["camera"]["image"] == display["image"]  # The HUD must never overlay the newest hand onto an older image.
    click.clock.now += 0.05
    assert click.service.snapshot()["camera"]["age_ms"] == 50  # Snapshot reads preserve the display frame's own age.
    snapshot["camera"]["landmarks"][0][0] = 99
    assert click.service.snapshot()["camera"]["landmarks"][0][0] != 99
    click.service.stop()
    assert not click.service._camera and not click.service._camera_display  # Stop retires both independently owned snapshots.


@pytest.mark.parametrize("failure", ["lost", "stale", "replayed"])
def test_fresh_jpeg_cannot_hide_tracking_loss_or_complete_pending_eeg_click(click, failure):
    click.arm()
    click.neutral()
    click.select()
    assert click.control()["selection_pending"]
    frozen_tracking = copy.deepcopy(click.camera.frame)

    def tracking():
        if failure == "replayed":
            return copy.deepcopy(frozen_tracking)  # A dishonest age of zero cannot refresh a repeated inference sequence.
        result = copy.deepcopy(click.camera.frame)
        if failure == "lost":
            result.update(tracked=False, landmarks=[])  # Report inference loss while the independently cached image still contains a hand.
        else:
            result["age_ms"] = 501  # A newly delivered sequence may still be too old to control input.
        return result

    click.camera.tracking_snapshot = tracking
    updates_before = len(click.pointer.updates)
    resets_before = len(click.pointer.resets)
    click.pointer.button_down = True  # Owned fake input must be released when current tracking becomes unusable.
    click.tick(16 if failure == "replayed" else 1, tracked=True, step=1 / 30)

    assert click.service.snapshot()["camera"]["tracked"]  # Image visibility is deliberately insufficient to authorize control.
    assert len(click.pointer.updates) == updates_before
    assert len(click.pointer.resets) == resets_before + 1
    assert not click.pointer.button_down
    assert click.control()["armed"] and not click.control()["hand_pointer"]["ready"]
    assert not click.control()["selection_pending"]
    assert click.mouse.clicks == click.control()["selection_count"] == 0


def test_thirty_hz_hand_updates_keep_eeg_prediction_hop_and_complete_only_full_cycle(click):
    click.arm()
    click.neutral()
    click.camera.tracking_snapshot = lambda: copy.deepcopy(click.camera.frame)
    click.decoder.predictions.clear()
    click.tick(30, step=1 / 30)

    predictions = click.decoder.predictions
    assert 3 <= len(predictions) <= 4  # Thirty hand frames still yield only the established quarter-second EEG windows.
    ends = [prediction["window"]["end"] for prediction in predictions]
    assert all(b - a >= 0.25 for a, b in zip(ends, ends[1:]))
    click.decoder.label = "select"
    click.tick(3, step=1 / 30)
    assert not click.control()["selection_pending"]  # Three camera polls cannot substitute for three distinct EEG predictions.
    click.tick(27, step=1 / 30)
    assert click.control()["selection_pending"] and click.control()["selection_count"] == 0
    click.decoder.label = "rest"
    click.tick(40, step=1 / 30)
    assert click.control()["selection_count"] == 1  # Rest/select/rest timing remains usable and completes only once.
    assert click.mouse.clicks == 0


def test_fast_hand_polls_retain_fresh_eeg_waveform_between_batches_but_clear_stale_data(click):
    waveform = click.service.snapshot()["eeg"]["waveform"]
    assert waveform
    click.tick(6, add_rows=False, step=1 / 30)  # Hand updates may outpace the arrival of a Bluetooth EEG batch.
    assert click.service.snapshot()["eeg"]["waveform"] == waveform
    click.tick(step=1 / 30)  # A new batch replaces the displayed data normally.
    assert click.service.snapshot()["eeg"]["waveform"] != waveform
    click.tick(25, add_rows=False, step=1 / 30)  # Repeated counters expire even if the adapter reports a misleading zero age.
    assert click.service.snapshot()["eeg"]["waveform"] == []


def test_calibration_from_armed_state_stops_pointer_updates(click):
    click.arm()  # Start with legitimately armed hand aiming to exercise revocation at calibration entry.
    click.neutral()  # Show that pointer movement was possible before requesting a new experiment.
    updates_before = len(click.pointer.updates)  # Preserve the last authorized update count.
    click.decoder.trained = click.decoder.eligible = click.decoder.passed = False  # Prepare the fake decoder for a new training trial.
    click.service._decoder_status = click.decoder.status()  # Publish the corresponding untrained calibration state.
    assert click.service.eeg_trial("rest").get("ok")  # Starting a labelled cue explicitly disarms input.
    click.tick(40, tracked=True)  # Fresh visible hands must not reactivate aiming during the entire cue.
    assert len(click.pointer.updates) == updates_before  # Calibration never forwards hand frames to native input.
    assert not click.control()["armed"] and click.mouse.clicks == 0  # Cue completion also leaves all input disarmed.


@pytest.mark.parametrize("change", ["target", "mode", "disarm"])
def test_queued_pointer_update_rechecks_revocation_before_moving(click, change):
    click.arm()  # Enable hand aiming through the same explicit EEG authorization as the production flow.
    click.neutral()  # Establish a ready target before queuing one additional camera frame.
    before = len(click.pointer.updates)  # The queued frame must never reach the fake native controller after revocation.
    entered, revoked, completed = (threading.Event() for _ in range(3))  # Synchronize the actual pointer action boundary.
    original_update = click.service._update_hand_pointer  # Keep production freshness, epoch, and authorization checks intact.
    original_disarm = click.service._disarm_eeg_locked  # Observe cancellation without replacing its state changes.

    def queued_update(*args):
        entered.set()  # Signal that the fresh frame is about to wait for the action lock.
        return original_update(*args)  # Recheck all authorization after that lock becomes available.

    def disarm(*args, **kwargs):
        result = original_disarm(*args, **kwargs)  # Apply the real cancellation before notifying the test.
        revoked.set()  # The pointer must see this revoked state when it eventually enters its boundary.
        return result  # Preserve the existing method's return contract.

    def revoke():
        if change == "target":
            click.service.eeg_target("mouse")  # Destination changes must revoke practice movement as well as pending clicks.
        elif change == "mode":
            click.service.set_control_mode("eeg")  # Leaving the hand/EEG mode must cancel queued camera input.
        else:
            click.service.eeg_arm(False)  # Explicit disarm must cancel both hand aiming and EEG selection.
        completed.set()  # Record when the user-facing action has crossed its input cleanup barrier.

    click.service._update_hand_pointer = queued_update  # Observe the production boundary without faking its decision.
    click.service._disarm_eeg_locked = disarm  # Capture the ordering of authorization revocation.
    with click.service._action_lock:  # A preceding action delays this next pointer frame.
        worker = threading.Thread(target=lambda: click.tick(x=0.8))  # Supply one new tracked frame that would ordinarily move the pointer.
        worker.start()  # Advance acquisition and queue the frame on the action lock.
        assert entered.wait(2)  # The frame is now waiting before its final authorization check.
        changer = threading.Thread(target=revoke)  # Cancellation runs independently of the blocked capture worker.
        changer.start()  # Revoke immediately, then wait for the same action barrier.
        assert revoked.wait(2)  # Authorization must be gone before releasing the queued pointer frame.
        assert not completed.is_set()  # No change may claim completed cleanup while the action barrier is still occupied.
    worker.join(2)  # The queued update can finish only by observing the newer revoked state.
    changer.join(2)  # Cleanup should complete promptly after the input barrier becomes available.
    assert not worker.is_alive() and not changer.is_alive()  # Detect lock-order deadlocks rather than silently leaving threads behind.
    assert completed.is_set()  # The public change finished after pending input was resolved.
    assert len(click.pointer.updates) == before  # The revoked camera frame never reaches native pointer input.
    assert not click.control()["hand_pointer"]["ready"]  # Old hand engagement cannot survive target or mode changes.
    assert click.mouse.clicks == 0  # Pointer cancellation must not accidentally create EEG button input.


@pytest.mark.parametrize("change", ["target", "mode", "disarm"])
def test_inflight_pointer_move_finishes_before_revocation_returns_without_stale_readiness(click, change):
    click.arm()  # A pointer operation may start only after explicit input authorization.
    click.neutral()  # Establish a valid ready hand target before exercising in-flight cleanup.
    before = len(click.pointer.updates)  # One already-dispatched move may finish, but no later update may follow it.
    entered, finish, revoked, completed = (threading.Event() for _ in range(4))  # Coordinate native completion with state revocation.
    original_update = click.pointer.update  # The fake native controller remains an in-memory input recorder.
    original_disarm = click.service._disarm_eeg_locked  # Preserve all real cancellation side effects.

    def slow_update(*args, **kwargs):
        entered.set()  # The service has already acquired its action lock and authorized this specific move.
        assert finish.wait(3)  # Simulate a bounded native operation that has not yet confirmed completion.
        return original_update(*args, **kwargs)  # Report the old operation's success only after the newer state is revoked.

    def disarm(*args, **kwargs):
        result = original_disarm(*args, **kwargs)  # Cancel new input immediately even while the old move is finishing.
        revoked.set()  # Let the test verify that state changes precede the cleanup barrier.
        return result  # Keep the production cancellation return value unchanged.

    def revoke():
        if change == "target":
            click.service.eeg_target("mouse")  # A new output destination cannot inherit in-flight hand readiness.
        elif change == "mode":
            click.service.set_control_mode("eeg")  # Leaving click mode must finish hand input before returning.
        else:
            click.service.eeg_arm(False)  # Disarm must wait for already-dispatched input to finish safely.
        completed.set()  # The operation may report completion only after the native action barrier clears.

    click.pointer.update = slow_update  # Delay the fake native adapter, not the service's authorization checks.
    click.service._disarm_eeg_locked = disarm  # Observe the real epoch change caused by revocation.
    worker = threading.Thread(target=lambda: click.tick(x=0.8))  # Submit exactly one newly tracked hand frame.
    worker.start()  # Enter the already-authorized native pointer call.
    assert entered.wait(2)  # Ensure the slow operation owns the action lock before requesting cancellation.
    changer = threading.Thread(target=revoke)  # Revoke from a separate request just as UI controls do.
    changer.start()  # Publish disarmed state immediately and then wait for input completion.
    try:
        assert revoked.wait(2)  # State revocation must not wait behind the native pointer call.
        assert not completed.is_set()  # The user-facing request cannot return with input cleanup still outstanding.
    finally:
        finish.set()  # Always allow the fake native operation to complete, including failed assertions.
    worker.join(2)  # The old completion must observe the changed epoch and avoid restoring ready state.
    changer.join(2)  # Cleanup may now finish after the action barrier is released.
    assert not worker.is_alive() and not changer.is_alive()  # Catch regressions in action/state lock ordering.
    assert completed.is_set()  # The state-change request completed after the old operation finished.
    assert len(click.pointer.updates) == before + 1  # Only the single already-dispatched move is allowed to finish.
    assert not click.control()["armed"] and not click.control()["hand_pointer"]["ready"]  # Late success cannot reauthorize hand targeting.
    assert click.mouse.clicks == 0  # A finishing pointer move must never synthesize a mouse click.


@pytest.mark.parametrize("interaction", ["pinch", "drag", "completed_click"])
def test_finger_interaction_cancels_pending_eeg_without_disarming_and_requires_new_cycle(click, interaction):
    click.arm()  # Joint authorization enables the hand controller and the EEG practice path.
    click.neutral()  # Establish a ready hand target before preparing an EEG selection.
    click.select()  # A complete selection is now waiting for the user's return to rest.
    assert click.control()["selection_pending"]  # Verify that the finger interaction has something real to cancel.
    if interaction == "completed_click":
        click.pointer.click_id += 1  # A fast finger click may complete entirely inside one camera update.
        click.pointer.click_source = "pinch"  # Keep the manual gesture distinct from an EEG actuator completion.
    else:
        click.pointer.finger_active = True  # An in-progress pinch or drag owns the interaction before its click completes.
        click.pointer.button_down = interaction == "drag"  # A drag additionally owns a pressed mouse button.
        click.pointer.ready = False  # A legitimate finger pose need not look like the ordinary pointing pose.
    click.tick(tracked=True)  # Process hand ownership on the current camera frame before evaluating EEG completion.
    assert click.control()["armed"]  # Legitimate finger input must not revoke the shared authorization.
    assert not click.control()["selection_pending"]  # Finger activity immediately consumes the previously queued EEG cycle.
    assert not click.control()["neutral_ready"]  # A finger interaction also requires a new quiet EEG rest interval.
    assert click.control()["selection_count"] == click.mouse.clicks == 0  # Manual input cannot accidentally complete an EEG action.
    click.tick(4, tracked=True)  # Continue the same finger interaction or replay the same completed-click identity.
    assert click.control()["armed"]  # A stable finger phase is not a tracking-loss or raw-signal failure.
    if interaction != "completed_click":
        click.pointer.finger_active = click.pointer.button_down = False  # Release the deliberate finger interaction before returning to aim.
        click.pointer.ready = True  # Restore a valid pointing pose instead of hiding a real target loss.
        click.pointer.click_id += 1  # Report one completed finger gesture, not one click per held frame.
        click.pointer.click_source = "drag" if interaction == "drag" else "pinch"  # Preserve the source in the hand feedback.
        click.tick(tracked=True)  # Publish the release and completion identity before the fresh EEG rest interval.
    assert click.control()["finger_click_count"] == 1  # The manual completion has a separate exactly-once public counter.
    click.release()  # Returning to rest after manual input cannot resurrect the cancelled EEG selection.
    assert click.control()["selection_count"] == click.mouse.clicks == 0  # Rest alone never duplicates the manual click.
    click.select()  # A new deliberate EEG selection is permitted after fresh rest and hand readiness.
    click.release()  # Complete this new cycle with a distinct reopened/rest interval.
    assert click.control()["selection_count"] == 1  # Exactly one new EEG practice action should be counted.
    assert click.control()["finger_click_count"] == 1  # The unchanged manual click ID must not inflate finger feedback.
    assert click.mouse.clicks == 0  # EEG practice remains local even though finger gestures are independently available.


def test_repeated_finger_click_id_does_not_repeat_count_or_restart_rest_cooldown(click):
    click.arm()  # Authorize hand interactions and EEG practice through the joint guard.
    click.neutral()  # Begin with a valid pointer target and completed EEG rest interval.
    click.pointer.click_id = 7  # A controller may preserve its monotonic click identifier from earlier use.
    click.pointer.click_source = "index_bend"  # Source feedback must remain separate from the EEG trigger.
    click.tick()  # Consume this one newly observed completion identity.
    observed_count = click.control()["finger_click_count"]  # The aggregate counts newly observed events, not the numeric ID itself.
    assert observed_count == 1  # A jump from zero to seven must not invent six unobserved user clicks.
    manual_at = click.service._last_dispatch_at  # Manual input starts a fresh shared cooldown once.
    click.release()  # Repeated identical metadata accompanies ordinary fresh rest frames.
    assert click.control()["finger_click_count"] == observed_count  # Replaying the same click identity is not another event.
    assert click.service._last_dispatch_at == manual_at  # Duplicate metadata must not indefinitely extend the action cooldown.
    assert click.control()["neutral_ready"]  # The next EEG cycle can prepare after genuine fresh rest.
    assert click.mouse.clicks == click.control()["selection_count"] == 0  # Manual completion alone never causes an EEG action.


@pytest.mark.parametrize("ownership_field", ["finger_active", "button_down"])
def test_final_eeg_click_boundary_rechecks_current_finger_ownership(click, ownership_field):
    click.service.eeg_target("mouse")  # This test targets the real-click branch while keeping the actuator entirely fake.
    click.arm()  # Grant a fresh guard and joint authorization for this destination.
    click.neutral()  # Obtain a current ready hand target before the EEG selection begins.
    click.select()  # Prepare an EEG action that should complete only after rest and ownership checks.
    original_dispatch = click.service._dispatch_eeg_click  # Retain the production action-lock and final authorization checks.

    def ownership_changed(*args):
        with click.service._lock:  # Simulate newly published current-frame hand ownership at the last possible boundary.
            click.service._pointer_status[ownership_field] = True  # Either finger activity or a held button must independently exclude EEG input.
        return original_dispatch(*args)  # The real final check must see the changed ownership and suppress the click.

    click.service._dispatch_eeg_click = ownership_changed  # Alter only feedback timing, not the authorization logic under test.
    click.release()  # Let the EEG rest/select/rest cycle reach its ordinary final dispatch point.
    assert click.mouse.clicks == 0  # A current finger interaction must win arbitration even after the EEG cycle is complete.
    assert click.control()["click_count"] == 0  # Suppressed input cannot appear as a successful EEG click.


@pytest.mark.parametrize("failure", ["hand_loss", "packet_gap", "raw_fault", "guard_expiry"])
def test_automatic_disarm_releases_owned_finger_drag_before_tick_returns(click, failure):
    click.arm()  # The fake hand controller may own a drag only after explicit joint authorization.
    click.neutral()  # Establish the pointer engagement that distinguishes a legitimate finger phase from target loss.
    click.pointer.finger_active = click.pointer.button_down = True  # Simulate a deliberate finger drag with a held mouse button.
    click.pointer.ready = False  # Dragging may not report the ordinary pointing pose as ready.
    click.tick()  # Let production arbitration observe the ongoing manual interaction.
    assert click.control()["armed"] and click.pointer.button_down  # Ordinary finger dragging must remain usable while armed.
    closes_before = click.pointer.closes  # Native cleanup must happen after the upcoming automatic revocation.
    if failure == "hand_loss":
        click.tick(tracked=False)  # A lost hand cannot leave its previously owned button held.
    elif failure == "packet_gap":
        click.tick(gap=1)  # Invalid EEG cancels the joint input authorization as before.
    elif failure == "raw_fault":
        click.extract_ok = False  # Force the real service's raw-quality failure path, bypassing statistical transition grace.
        click.tick(2)  # Reach the next scheduled feature extraction and its immediate disarm decision.
    else:
        click.guard_active = False  # Stop renewing the guard without sending an explicit user disarm request.
        click.tick(25)  # Allow the existing three-second lease to expire while the fake drag is held.
    assert click.control()["armed"] is (failure != "guard_expiry")  # Only guard expiry revokes shared hand authorization.
    if failure in ("hand_loss", "guard_expiry"):
        assert not click.pointer.finger_active and not click.pointer.button_down  # A missing hand or expired guard releases the owned drag immediately.
        assert click.pointer.closes > closes_before or click.pointer.resets  # Suspension/cleanup ran outside the service state lock.
    else:
        assert click.pointer.finger_active and click.pointer.button_down  # EEG faults must not interrupt a valid finger drag.
    assert not click.control()["hand_pointer"]["ready"]  # Dragging or paused tracking cannot authorize an EEG click.
    assert click.mouse.clicks == 0  # Releasing a failed drag must not create an EEG click.
