"""Double-blink service contracts using fake signals, a fake detector, and fake input."""

import copy
import math

import pytest

from core import multimodal
from test_eeg_click_control import ClickHarness


class FakeBlinkDecoder:
    """Expose causal detector events without classifying or injecting real input."""

    labels = ("rest", "select")
    feature_profile = "double_blink_waveform_v1"

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


@pytest.fixture
def blink(monkeypatch):
    h = ClickHarness(monkeypatch)
    h.eyes_closed = h.decoder
    monkeypatch.setattr(multimodal, "BlinkDecoder", FakeBlinkDecoder)
    assert not h.service.eeg_method("double_blink").get("error")
    h.blink = h.service._decoder
    h.tick()  # Prime display inference without arming any fake input.
    return h


def arm_ready(h, target="mouse"):
    assert not h.service.eeg_target(target).get("error")
    h.arm()
    h.tick()  # Complete the post-arm stream watermark and consume a fresh hand frame.
    assert h.control()["hand_pointer"]["ready"]


def emit(h, event_id="event-1", *, at=None, **extra):
    h.blink.event(event_id, h.clock.now if at is None else at, **extra)
    h.tick()


def test_method_switch_preserves_each_decoder_evidence_and_archive_source(blink):
    h = blink
    h.eyes_closed.trials.append({"label": "rest", "phase": "validate", "marker": "original"})
    assert not h.service.eeg_method("eyes_closed").get("error")
    h.service._model_source = "model-original.json"
    h.service._experimental_model = True
    h.tick()
    h.arm()
    old_guard = h.guard_token
    assert not h.service.eeg_method("double_blink").get("error")
    assert not h.control()["armed"]
    assert h.service.eeg_guard(old_guard).get("error"), "a method switch must retire the prior Escape lease"
    assert h.service._decoder is h.blink
    h.blink.trials.append({"label": "select", "phase": "train", "marker": "blink-only"})
    h.service._model_source = "double-blink-original.json"
    assert not h.service.eeg_method("eyes_closed").get("error")
    assert h.service._decoder is h.eyes_closed
    assert h.eyes_closed.trials[-1]["marker"] == "original"
    assert h.control()["saved_model_source"] == "model-original.json"
    assert h.control()["experimental_model"] is True
    assert not h.service.eeg_method("double_blink").get("error")
    assert h.service._decoder is h.blink
    assert h.blink.trials[-1]["marker"] == "blink-only"
    assert h.control()["saved_model_source"] == "double-blink-original.json"
    assert not h.control()["experimental_model"]
    assert h.mouse.clicks == 0


def test_completed_pair_clicks_once_immediately_without_spectral_rest_delay(blink):
    h = blink
    arm_ready(h)
    before = h.clock.now
    assert h.service._eeg_neutral_since is None
    emit(h)
    assert h.clock.now - before < h.service.EEG_REST_SECONDS
    assert h.mouse.clicks == h.control()["click_count"] == 1
    assert not h.control()["selection_pending"]
    assert h.control()["selection_count"] == 0
    assert h.actions == []
    assert h.control()["prediction"]["confidence"] == 0.64, "blink acceptance must not inherit the spectral score threshold"


def test_practice_pair_counts_once_without_mouse_input_and_leaves_fingers_enabled(blink):
    h = blink
    arm_ready(h, "practice")
    emit(h)
    assert h.control()["selection_count"] == 1
    assert h.mouse.clicks == 0
    assert h.pointer.moves
    h.pointer.click_id = 1
    h.pointer.click_source = "pinch"
    h.tick()
    assert h.control()["finger_click_count"] == 1
    assert h.control()["armed"]


@pytest.mark.parametrize("trained,passed,eligible", [(False, False, False), (True, False, False),
                                                       (True, False, True), (True, True, False)])
def test_blink_validation_cannot_use_eyes_closed_experimental_override(blink, trained, passed, eligible):
    h = blink
    h.blink.trained, h.blink.passed, h.blink.eligible = trained, passed, eligible
    h.service._experimental_model = True  # Even an accidentally retained old flag cannot authorize the separate blink model.
    arm_ready(h)
    emit(h)
    assert h.control()["armed"], "hand control remains available independently"
    assert h.pointer.moves
    assert not h.control()["can_eeg_click"]
    assert h.mouse.clicks == 0


def test_replayed_event_id_cannot_click_again_after_cooldown_or_cached_packets(blink):
    h = blink
    arm_ready(h)
    emit(h, "stable-event")
    assert h.mouse.clicks == 1
    h.tick(18)  # Satisfy cooldown and the post-click stream watermark before replaying the identity.
    h.blink.event("stable-event", h.clock.now)
    h.tick(add_rows=False)  # Repeated display buffers cannot authorize another delivery of an old event.
    assert h.mouse.clicks == 1
    emit(h, "new-event")
    assert h.mouse.clicks == 2


@pytest.mark.parametrize("offset", [-1.0, 1.0, math.inf, -math.inf, math.nan])
def test_stale_future_and_nonfinite_event_times_never_click(blink, offset):
    h = blink
    arm_ready(h)
    emit(h, at=h.clock.now + offset)
    assert h.mouse.clicks == 0
    assert h.control()["armed"]


@pytest.mark.parametrize("extra", [{"valid": False}, {"ready": False}, {"label": "rest"}, {"label": "left"}, {"event_id": None}])
def test_only_valid_completed_pair_events_can_click(blink, extra):
    h = blink
    arm_ready(h)
    fields = dict(extra)  # Keep parametrized case dictionaries immutable across test runs.
    event_id = fields.pop("event_id", "event-invalid")
    emit(h, event_id, **fields)
    assert h.mouse.clicks == 0


def test_arming_resets_and_watermarks_a_pair_observed_in_the_old_buffer(blink):
    h = blink
    h.blink.event("before-arm", h.clock.now)
    resets = h.blink.stream_resets
    arm_ready(h)
    assert h.blink.stream_resets > resets
    assert h.blink.watermarks[-1] == h.eeg.rows[-1]["counter"]
    assert h.mouse.clicks == 0
    emit(h, "after-arm")
    assert h.mouse.clicks == 1


@pytest.mark.parametrize("interruption", ["hand", "eeg_stale", "gap"])
def test_recovery_watermarks_and_requires_a_new_pair_without_rearming_hand(blink, interruption):
    h = blink
    arm_ready(h)
    resets = h.blink.stream_resets
    h.blink.event("blocked-pair", h.clock.now)
    h.tick(**({"tracked": False} if interruption == "hand" else {"eeg_age": 1.0} if interruption == "eeg_stale" else {"gap": 1}))
    assert h.control()["armed"]
    assert h.mouse.clicks == 0
    h.blink.event("old-buffer-pair", h.clock.now)
    h.tick()
    assert h.blink.stream_resets > resets
    assert h.mouse.clicks == 0, "reacquisition must not finish a pair begun before losing the signal or target"
    emit(h, "fresh-recovery-pair")
    assert h.mouse.clicks == 1


def test_expired_guard_blocks_pair_and_requires_a_new_explicit_arm(blink):
    h = blink
    arm_ready(h)
    h.guard_active = False
    h.tick(25)
    emit(h)
    assert not h.control()["armed"]
    assert h.mouse.clicks == 0
    assert h.service.eeg_guard(h.guard_token).get("error")


@pytest.mark.parametrize("gesture", ["pinch", "drag", "completed_click"])
def test_finger_priority_cancels_pair_and_requires_new_stream_evidence(blink, gesture):
    h = blink
    arm_ready(h)
    if gesture == "completed_click":
        h.pointer.click_id = 1
        h.pointer.click_source = "pinch"
    else:
        h.pointer.finger_active = True
        h.pointer.button_down = gesture == "drag"
    resets = h.blink.stream_resets
    emit(h, "finger-collision")
    assert h.mouse.clicks == 0
    assert h.control()["armed"]
    assert h.blink.stream_resets > resets
    h.pointer.finger_active = h.pointer.button_down = False
    h.tick(18)
    emit(h, "after-finger")
    assert h.mouse.clicks == 1


def test_method_switch_during_prediction_discards_that_event_and_keeps_old_model(blink):
    h = blink
    arm_ready(h)
    h.blink.on_update = lambda: h.service.eeg_method("eyes_closed")
    emit(h, "in-flight-pair")
    assert h.mouse.clicks == 0
    assert h.service._decoder is h.eyes_closed
    assert not h.control()["armed"]


def test_guided_blink_trials_preserve_the_complete_five_second_signal_and_active_phase(blink):
    h = blink
    h.blink.reset()
    h.service._decoder_status = h.blink.status()
    start = h.clock.now
    assert not h.service.eeg_trial("select", "train").get("error")
    assert h.service.eeg_method("eyes_closed").get("error"), "the active cue cannot change detectors midway"
    h.tick(41)
    assert len(h.blink.trials) == 1
    trial = h.blink.trials[0]
    assert (trial["label"], trial["phase"], trial["start"], trial["end"]) == ("select", "train", start, start + 5)
    rows = trial["signal"]["samples"]
    assert len(rows) == 1250
    assert len({row["counter"] for row in rows}) == 1250, "overlapping acquisition buffers must not duplicate evidence"
    assert start <= rows[0]["estimated_monotonic"] < start + 1 / 250
    assert rows[-1]["estimated_monotonic"] < start + 5
    assert not h.service.eeg_train().get("error")
    assert h.blink.train_calls == 1
    assert not h.service.eeg_trial("rest", "validate").get("error")
    h.tick(41)
    assert [trial["phase"] for trial in h.blink.trials] == ["train", "validate"]
    assert h.mouse.clicks == 0


def test_reset_affects_only_the_current_blink_model(blink):
    h = blink
    h.eyes_closed.trials.append({"label": "rest", "phase": "validate", "retained": True})
    h.blink.trials.append({"label": "select", "phase": "train"})
    assert not h.service.eeg_reset().get("error")
    assert h.blink.trials == [] and h.blink.reset_calls == 1
    assert not h.service.eeg_method("eyes_closed").get("error")
    assert h.service._decoder is h.eyes_closed
    assert h.eyes_closed.trials[-1]["retained"]
    assert h.eyes_closed.reset_calls == 0


def test_returning_to_eyes_closed_keeps_its_original_rest_select_rest_sequence(blink):
    h = blink
    assert not h.service.eeg_method("eyes_closed").get("error")
    h.tick(4)
    h.service.eeg_target("mouse")
    h.arm()
    h.neutral()
    h.select()
    assert h.mouse.clicks == 0
    assert h.control()["selection_pending"]
    h.release()
    assert h.mouse.clicks == 1


def feed_real_signal(h, stream, count):
    """Deliver real synthetic four-channel packets through the normal service tick."""
    h.eeg.acq.update(session_id=stream["session_id"], channels=copy.deepcopy(stream["channels"]), nominal_hz=250)
    h.eeg.rows = copy.deepcopy(stream["samples"][max(0, count - 1250):count])
    h.clock.now = h.eeg.rows[-1]["estimated_monotonic"] + 1 / 250
    h.tick(add_rows=False, step=0)  # The fake camera and Escape heartbeat are refreshed at the packet delivery time.


@pytest.mark.parametrize("kind,expected", [(None, 1), ("single", 0)])
def test_real_blink_decoder_playback_reaches_mouse_once_for_pair_only(monkeypatch, kind, expected):
    from test_eeg_blink import signal, trained_decoder

    h = ClickHarness(monkeypatch)
    assert not h.service.eeg_method("double_blink").get("error")
    decoder = trained_decoder(validate=True)
    archived = decoder.export_state()
    stream = signal(40, "select", session="fresh-real-playback", kind=kind)
    feed_real_signal(h, stream, 125)  # Publish the compatible current stream before explicitly selecting the saved model.
    result = h.service.eeg_use_saved_blink_model(decoder, source="double-blink-real-fixture.json")
    assert not result.get("error"), result
    assert not h.control()["armed"]
    assert h.control()["decoder"]["train_counts"] == {"rest": 8, "select": 8}
    assert h.control()["decoder"]["validation_counts"] == {"rest": 8, "select": 8}
    assert h.control()["decoder"]["passed_validation"]
    assert h.control()["decoder"]["validation_scope"] == "saved_acquisition"
    h.service.eeg_target("mouse")
    h.arm()
    clicked_at = []
    click = h.mouse.click
    def record_click():
        clicked_at.append(h.clock.now)
        return click()
    h.mouse.click = record_click
    for count in range(150, 1251, 25):
        feed_real_signal(h, stream, count)
    assert h.mouse.clicks == h.control()["click_count"] == expected
    assert h.control()["armed"]
    assert h.pointer.moves, "real blink inference must not prevent hand aiming"
    if expected:
        elapsed = clicked_at[0] - stream["samples"][0]["estimated_monotonic"]
        assert 2.05 < elapsed < 3.0, "a completed pair must not inherit the one-second eyes-closed rest release"
    h.tick(add_rows=False)
    assert h.mouse.clicks == expected, "replaying the last display buffer cannot repeat a click"
    assert decoder.export_state() == archived, "current-session inference must not rewrite the saved calibration"


def test_real_blink_training_accepts_five_second_trial_collected_by_service(monkeypatch):
    from test_eeg_blink import signal

    h = ClickHarness(monkeypatch)
    assert not h.service.eeg_method("double_blink").get("error")
    decoder = h.service._decoder
    previous = signal(39, "rest")
    for row in previous["samples"]:
        row["estimated_monotonic"] += 1.0  # End this prefix immediately before the next cue rather than leaving the fixture's one-second gap.
        row["counter"] += 250  # Close the same one-second gap in sample identity, so the cue starts at the next actual packet.
    feed_real_signal(h, previous, 1250)
    stream = signal(40, "select")
    h.clock.now = stream["samples"][0]["estimated_monotonic"]
    h.tick(add_rows=False, step=0)  # Fresh camera metadata permits the explicitly timed cue; no input is armed.
    result = h.service.eeg_trial("select", "train")
    assert not result.get("error"), result
    assert "eyes open" in h.control()["pending"]["cue"].lower()
    for count in range(25, 1251, 25):
        feed_real_signal(h, stream, count)
        if count == 500:
            assert "blink" in h.control()["pending"]["cue"].lower()
    next_packet = copy.deepcopy(stream["samples"][-1])
    next_packet["counter"] += 1
    next_packet["estimated_monotonic"] += 1 / 250
    stream["samples"].append(next_packet)
    feed_real_signal(h, stream, 1251)  # The first packet after the cue confirms that all five seconds have arrived.
    assert h.control()["pending"] is None
    assert decoder.status()["train_counts"] == {"rest": 0, "select": 1}, h.control()["last_result"]
    assert h.control()["last_result"] == "Accepted train double-blink trial."
    assert h.mouse.clicks == 0
    assert not h.control()["armed"]
