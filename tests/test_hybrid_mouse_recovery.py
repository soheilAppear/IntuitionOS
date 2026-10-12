"""Camera mouse recovery and explicitly restored EEG models; all native input is fake."""

import copy

import pytest

from core.eeg_decoder import CLICK_FEATURE_PROFILE, EegDecoder
from core.eeg_model_store import export_decoder
from test_eeg_click_control import ClickHarness
from test_eeg_decoder import click_extracted
from test_hand_eeg_pointer import HandEegHarness
from test_hand_mouse import buttons


def failed_saved_model():
    decoder = EegDecoder(labels=("rest", "select"), feature_profile=CLICK_FEATURE_PROFILE)
    for index in range(32):
        if index == 16:
            assert decoder.train()["ok"]
        label = decoder.labels[index % 2]
        trial = click_extracted("select" if index == 16 else label, index, session="saved-session")
        trial["features"] *= 2
        trial["context"]["channels"] = [{"num": 0, "name": "T3"}, {"num": 1, "name": "T4"}]
        assert decoder.add_trial(label, trial, phase="train" if index < 16 else "validate", trial_id=index)["ok"]
    assert decoder.status()["state"] == "validation_failed"
    return decoder


def use_real_prediction(monkeypatch, h):
    def extract(rows, channels, nominal_hz, *, session_context=None, feature_profile=None):
        trial = click_extracted(h.decoder.label, rows[0]["counter"], session=session_context["session_id"])
        trial["features"] *= 2
        trial["context"].update(channels=copy.deepcopy(channels), nominal_hz=nominal_hz, feature_profile=feature_profile)
        trial["window"].update(start=rows[0]["estimated_monotonic"], end=rows[-1]["estimated_monotonic"],
                               counter_start=rows[0]["counter"], counter_end=rows[-1]["counter"], n_samples=len(rows))
        return trial
    monkeypatch.setattr("core.multimodal.extract_features", extract)


@pytest.mark.parametrize("fault", ["untrained", "uncertain", "raw_invalid", "stale", "unavailable"])
def test_hand_mouse_arms_and_moves_without_ready_eeg(monkeypatch, fault):
    h = HandEegHarness(monkeypatch)
    if fault == "untrained":
        h.decoder.trained = h.decoder.eligible = h.decoder.passed = False
    elif fault == "uncertain":
        h.decoder.label = None
    elif fault == "raw_invalid":
        h.extract_ok = False
    elif fault == "unavailable":
        h.eeg.acq.update(state="error", error="fake disconnected EEG")
    h.tick(2, eeg_age=1 if fault == "stale" else 0)
    h.arm()
    h.tick(28, eeg_age=1 if fault == "stale" else 0)
    h.dx = 0.05
    h.tick(3, eeg_age=1 if fault == "stale" else 0)
    assert h.control()["armed"]
    assert any(event[0] == "move" for event in h.device.events)
    assert buttons(h.device) == []
    assert h.control()["click_count"] == 0


def test_missing_hand_releases_drag_and_recovers_to_actual_cursor_without_jump(monkeypatch):
    h = HandEegHarness(monkeypatch)
    h.arm()
    h.engage()
    h.pose = "pinch"
    h.tick(4)
    assert buttons(h.device) == ["down"]
    h.tick(tracked=False)
    assert buttons(h.device) == ["down", "up"]
    assert h.control()["armed"] and not h.control()["hand_pointer"]["ready"]
    before = list(h.device.events)
    h.tick(12, tracked=False)
    assert h.device.events == before
    h.device.position = lambda: (0.8, 0.3)
    h.pose, h.dx = "point", 0.04
    for _ in range(8):
        h.tick()
        if h.control()["hand_pointer"]["ready"]:
            break
    assert h.control()["hand_pointer"]["ready"]
    assert h.device.events == before  # Completing recovery only reanchors; it never moves or replays the drag.
    h.tick()
    moves = [event for event in h.device.events[len(before):] if event[0] == "move"]
    assert moves and moves[0][1:] == pytest.approx((0.8, 0.3))
    assert h.control()["finger_click_count"] == 0


@pytest.mark.parametrize("allow", [False, True])
def test_saved_failed_model_needs_explicit_optin_for_eeg_but_preserves_original_evidence(monkeypatch, allow):
    h = ClickHarness(monkeypatch)
    saved = failed_saved_model()
    original = export_decoder(saved)
    evaluation = saved.status()["evaluation"]
    use_real_prediction(monkeypatch, h)
    result = h.service.eeg_use_saved_model(saved, use_unvalidated=allow, source="saved.json")
    assert result.get("ok"), result
    assert not h.control()["armed"]
    assert h.control()["experimental_model"] is allow
    assert h.control()["saved_model_source"] == "saved.json"
    assert not h.control()["decoder"]["passed_validation"]
    assert not h.control()["decoder"]["arm_eligible"]
    assert h.control()["decoder"]["evaluation"] == evaluation
    assert h.service.eeg_export_model()["decoder"] == original["decoder"]
    h.service.eeg_target("mouse")
    h.tick(2)
    h.arm()
    h.release()  # Failed validation without explicit opt-in must not establish EEG selection readiness.
    assert h.control()["neutral_ready"] is allow
    h.select()
    h.release()
    assert h.mouse.clicks == (1 if allow else 0)
    assert h.control()["armed"]
    assert not h.control()["decoder"]["passed_validation"]
    assert h.service.eeg_export_model()["decoder"] == original["decoder"]


def test_model_session_change_pauses_eeg_but_preserves_hand_authorization(monkeypatch):
    h = ClickHarness(monkeypatch)
    saved = failed_saved_model()
    use_real_prediction(monkeypatch, h)
    assert h.service.eeg_use_saved_model(saved, use_unvalidated=True).get("ok")
    h.service.eeg_target("mouse")
    h.tick(2)
    h.arm()
    h.neutral()
    h.select()
    assert h.control()["selection_pending"]
    h.eeg.acq["session_id"] = "new-session-after-restore"
    h.tick(4)
    assert h.control()["armed"]
    assert not h.control()["selection_pending"]
    assert not h.control()["can_eeg_click"]
    assert not h.control()["prediction"]["valid"]
    assert h.mouse.clicks == 0


def test_incompatible_saved_model_does_not_replace_live_decoder(monkeypatch):
    h = ClickHarness(monkeypatch)
    saved = failed_saved_model()
    h.service._eeg["nominal_hz"] = 500
    before = h.service._decoder
    result = h.service.eeg_use_saved_model(saved, use_unvalidated=True)
    assert result.get("error")
    assert h.service._decoder is before
    assert not h.control()["experimental_model"]


def test_bare_select_or_uncertain_eeg_does_not_hold_hand_aiming(monkeypatch):
    h = ClickHarness(monkeypatch)
    h.decoder.label = "select"
    h.tick(2)
    h.arm()
    h.tick(8)
    assert h.pointer.moves and not any(update["hold"] for update in h.pointer.updates)
    h.decoder.label = None
    before = len(h.pointer.moves)
    h.tick(32)
    assert h.control()["armed"] and len(h.pointer.moves) > before
    assert not h.control()["selection_pending"]


def test_brief_processing_gap_releases_drag_and_recovers_without_new_arm(monkeypatch):
    h = HandEegHarness(monkeypatch)
    h.arm()
    h.engage()
    h.pose = "pinch"
    h.tick(4)
    assert buttons(h.device) == ["down"]
    h.clock.now += 0.6  # A brief backend scheduling pause is shorter than the emergency lease.
    h.tick()
    assert h.control()["armed"]
    assert buttons(h.device) == ["down", "up"]
    assert not h.control()["hand_pointer"]["ready"]
    h.pose = "point"
    h.tick(12)
    assert h.control()["armed"] and h.control()["hand_pointer"]["ready"]
    assert buttons(h.device) == ["down", "up"]  # Resuming never replays the drag as a click.
