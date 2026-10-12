"""The standalone practice observer uses fake snapshots and an in-memory device."""

import copy
import io
import json

import pytest

from test_hand_mouse import hand
from tools.eeg_model_practice import PracticeController, fetch_snapshot, metrics_text


class PracticeHarness:
    def __init__(self):
        self.now, self.frame, self.window = 100.0, 0, 0
        self.label, self.pose, self.sequence = "rest", "point", 1
        self.snapshot = {
            "state": "running",
            "camera": {"running": True, "tracked": True, "sequence": 1, "age_ms": 0,
                       "landmarks": hand()},
            "eeg": {"state": "running", "mode": "signal", "session_id": "same-live-session",
                    "stats": {"age_seconds": 0, "gaps": 0}},
            "eeg_control": {
                "decoder": {"trained": True, "labels": ["rest", "select"], "state": "validation_failed",
                            "passed_validation": False, "arm_eligible": False,
                            "train_counts": {"rest": 8, "select": 8}, "validation_counts": {"rest": 8, "select": 8},
                            "evaluation": {"balanced_accuracy": 0.625, "rest_false_activation_rate": 0.25,
                                           "confusion_matrix": {"rest": {"rest": 6, "select": 2},
                                                                "select": {"rest": 4, "select": 4}}}},
                "prediction": {"label": "rest", "valid": True, "ood": False, "confidence": 0.95,
                               "margin": 0.7, "window_id": "0:499:500"}}}
        self.controller = PracticeController()
        assert self.controller.start(self.snapshot, self.now)

    def tick(self, count=1, *, camera=True, prediction=True, transform=None):
        for _ in range(count):
            self.now += 0.1
            self.frame += 1
            if camera:
                self.sequence += 1
            self.snapshot["camera"].update(sequence=self.sequence, landmarks=hand(self.pose))
            if prediction and self.frame % 3 == 0:
                self.window += 1
                self.snapshot["eeg_control"]["prediction"].update(
                    label=self.label, window_id=f"{self.window * 75}:{499 + self.window * 75}:500")
            if transform:
                transform(self.snapshot)
            self.controller.update(self.snapshot, self.now)

    def ready(self):
        self.tick(18)
        assert self.controller.rest_ready and self.controller.engaged

    def select(self):
        self.label = "select"
        self.tick(12)
        assert self.controller.pending_at is not None


@pytest.fixture
def practice():
    return PracticeHarness()


def test_failed_original_validation_is_preserved_and_only_local_effects_occur(practice):
    original = copy.deepcopy(practice.snapshot["eeg_control"]["decoder"])
    practice.ready()
    practice.select()
    assert practice.controller.counts == {"eeg": 0, "finger": 0}
    practice.label = "rest"
    practice.tick(18)
    assert practice.controller.counts == {"eeg": 1, "finger": 0}
    assert practice.controller.hits[0] == 1
    assert practice.snapshot["eeg_control"]["decoder"] == original
    assert practice.controller.original_decoder == original
    text = metrics_text(original)
    assert "validation_failed" in text and "62.5%" in text and "25.0%" in text
    assert "Results are unchanged" in text


def test_pause_and_start_never_inherit_an_unfinished_cycle(practice):
    practice.ready()
    practice.select()
    practice.controller.pause()
    assert practice.controller.pending_at is None and not practice.controller.device.down
    cursor = practice.controller.device.cursor
    practice.label, practice.pose = "rest", "pinch"
    practice.tick(20)
    assert practice.controller.device.cursor == cursor
    assert practice.controller.counts == {"eeg": 0, "finger": 0}
    assert practice.controller.start(practice.snapshot, practice.now)
    practice.pose = "point"
    practice.tick(18)
    assert practice.controller.counts["eeg"] == 0


def test_holding_select_cannot_repeat_and_expiration_prevents_late_click(practice):
    practice.ready()
    practice.select()
    practice.tick(90)
    assert practice.controller.pending_at is None
    assert practice.controller.counts["eeg"] == 0
    practice.label = "rest"
    practice.tick(18)
    assert practice.controller.counts["eeg"] == 0


def test_finger_release_activates_canvas_and_cancels_pending_eeg(practice):
    practice.ready()
    practice.select()
    practice.pose = "pinch"
    practice.tick(2)
    practice.pose = "point"
    practice.tick(1)
    assert practice.controller.counts == {"eeg": 0, "finger": 1}
    assert practice.controller.hits[0] == 1
    assert practice.controller.pending_at is None
    practice.label = "rest"
    practice.tick(18)
    assert practice.controller.counts == {"eeg": 0, "finger": 1}


def test_drag_release_never_counts_as_a_canvas_click(practice):
    practice.ready()
    practice.pose = "pinch"
    practice.tick(6)
    assert practice.controller.device.down
    practice.pose = "point"
    practice.tick(1)
    assert not practice.controller.device.down
    assert practice.controller.counts == {"eeg": 0, "finger": 0}


@pytest.mark.parametrize("fault", ["camera_stale", "eeg_stale", "missing_hand", "session", "model", "gap", "malformed", "geometry", "fist"])
def test_invalid_stream_or_hand_cancels_pending_selection(practice, fault):
    practice.ready()
    practice.select()
    def invalidate(snapshot):
        if fault == "camera_stale":
            snapshot["camera"]["age_ms"] = 600
        elif fault == "eeg_stale":
            snapshot["eeg"]["stats"]["age_seconds"] = 1.0
        elif fault == "missing_hand":
            snapshot["camera"]["tracked"] = False
        elif fault == "session":
            snapshot["eeg"]["session_id"] = "new-session"
        elif fault == "model":
            snapshot["eeg_control"]["decoder"]["train_counts"]["rest"] += 1
        elif fault == "gap":
            snapshot["eeg"]["stats"]["gaps"] += 1
        elif fault == "malformed":
            snapshot["camera"]["landmarks"][0][0] = float("nan")
        elif fault == "geometry":
            snapshot["camera"]["landmarks"][0][0] = 5.0
        else:
            snapshot["camera"]["landmarks"] = hand("fist")
    practice.tick(transform=invalidate)
    assert not practice.controller.active
    assert practice.controller.pending_at is None
    assert practice.controller.counts["eeg"] == 0


@pytest.mark.parametrize("signal", ["camera", "prediction"])
def test_repeated_snapshot_identity_cannot_extend_freshness_or_finish_cycle(practice, signal):
    practice.ready()
    practice.select()
    practice.label = "rest"
    practice.tick(12, camera=signal != "camera", prediction=signal != "prediction")
    assert not practice.controller.active
    assert practice.controller.pending_at is None
    assert practice.controller.counts["eeg"] == 0


@pytest.mark.parametrize("change", ["invalid", "ood", "weak", "margin", "label"])
def test_unconfident_prediction_never_completes_cycle_and_keeps_finger_practice(practice, change):
    practice.ready()
    practice.select()
    def alter(snapshot):
        prediction = snapshot["eeg_control"]["prediction"]
        prediction.update({"invalid": {"valid": False}, "ood": {"ood": True},
                           "weak": {"confidence": 0.79}, "margin": {"margin": 0.24},
                           "label": {"label": "left"}}[change])
    practice.tick(transform=alter)
    assert practice.controller.active
    assert (practice.controller.pending_at is None) is (change == "invalid")
    assert practice.controller.counts["eeg"] == 0
    practice.pose = "pinch"
    practice.tick(2, transform=alter)
    practice.pose = "point"
    practice.tick(1, transform=alter)
    assert practice.controller.counts["finger"] == 1


def test_brief_valid_raw_transitions_preserve_rest_and_pending_selection(practice):
    practice.ready()
    def uncertain(snapshot):
        snapshot["eeg_control"]["prediction"].update(label=None, confidence=0.4, margin=0.05, valid=True, ood=False)
    def confident(snapshot):
        snapshot["eeg_control"]["prediction"].update(confidence=0.95, margin=0.7, valid=True, ood=False)
    practice.tick(9, transform=uncertain)
    assert practice.controller.rest_ready and practice.controller.counts["eeg"] == 0
    practice.label = "select"
    practice.tick(12, transform=confident)
    assert practice.controller.pending_at is not None
    practice.tick(9, transform=uncertain)
    assert practice.controller.pending_at is not None and practice.controller.counts["eeg"] == 0
    practice.label = "rest"
    practice.tick(18, transform=confident)
    assert practice.controller.active
    assert practice.controller.counts["eeg"] == 1


def test_prolonged_valid_raw_uncertainty_cancels_pending_selection(practice):
    practice.ready()
    practice.select()
    practice.tick(35, transform=lambda snapshot: snapshot["eeg_control"]["prediction"].update(
        label=None, confidence=0.4, margin=0.05, valid=False, ood=True))
    assert not practice.controller.active
    assert practice.controller.pending_at is None and practice.controller.counts["eeg"] == 0
    assert "three seconds" in practice.controller.message


def test_http_delay_and_replayed_old_window_cancel_practice(practice):
    practice.ready()
    snapshot = copy.deepcopy(practice.snapshot)
    practice.controller.update(snapshot, practice.now + 0.8, practice.now)
    assert not practice.controller.active
    assert practice.controller.start(practice.snapshot, practice.now + 1.0)
    practice.now += 1.0
    practice.tick(6)
    old_window = practice.snapshot["eeg_control"]["prediction"]["window_id"]
    practice.tick(6)
    practice.tick(transform=lambda snapshot: snapshot["eeg_control"]["prediction"].update(window_id=old_window))
    assert not practice.controller.active
    assert "replayed" in practice.controller.message


def test_two_select_windows_are_insufficient_even_with_many_duplicate_polls(practice):
    practice.ready()
    practice.label = "select"
    practice.tick(6)
    assert practice.controller.select_count == 2
    practice.tick(4, prediction=False)
    assert practice.controller.pending_at is None
    assert practice.controller.counts["eeg"] == 0


def test_readonly_transport_uses_fixed_get_and_discards_raw_preview_data():
    class Response(io.BytesIO):
        pass
    class Opener:
        def open(self, request, timeout):
            assert request.full_url == "http://127.0.0.1:7432/multimodal/preview"
            assert request.method == "GET" and request.data is None
            assert request.get_header("X-intuition-multimodal") == "1"
            assert timeout == 2.0
            return Response(json.dumps({"camera": {"image": "raw-camera", "landmarks": []},
                                        "eeg": {"waveform": ["raw-eeg"], "session_id": "same"}}).encode())
    snapshot = fetch_snapshot(Opener())
    assert "image" not in snapshot["camera"] and "waveform" not in snapshot["eeg"]
    assert snapshot["eeg"]["session_id"] == "same"
