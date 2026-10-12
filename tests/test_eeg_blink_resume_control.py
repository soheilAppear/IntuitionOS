"""Resume accepted draft evidence through the service without arming real input."""

import copy

import pytest

from core.eeg_blink import BlinkDecoder, VERSION
from test_eeg_blink import add, signal
from test_eeg_blink_control import feed_real_signal
from test_eeg_click_control import ClickHarness


def draft(version=1):
    decoder = BlinkDecoder()
    decoder._version = version  # Exercise actual historical extraction when preparing an older accepted draft.
    for index in range(8):
        assert add(decoder, index, "rest")["ok"]
        decoder._training[-1]["id"] = "eeg-trial-" + str(90 + index)
    return decoder.export_draft()


def session(monkeypatch):
    h = ClickHarness(monkeypatch)
    assert h.service.eeg_method("double_blink").get("ok")
    feed_real_signal(h, signal(40, "rest", session="resumed-acquisition"), 1250)
    return h


@pytest.mark.parametrize("version", [1, 2])
def test_resume_preserves_examples_original_model_and_acquisition_provenance(monkeypatch, version):
    h = session(monkeypatch)
    saved = draft(version)
    original_profiles = copy.copy(h.service._click_profiles)
    decoder = BlinkDecoder.from_calibration_state(saved, upgrade=True)
    result = h.service.eeg_use_saved_blink_calibration(decoder, source="calibration-test.json")
    assert result.get("ok"), result.get("error")
    assert h.control()["decoder"]["train_counts"] == {"rest": 8, "select": 0}
    assert h.control()["decoder"]["detector_version"] == VERSION
    assert not h.control()["armed"] and not h.control()["can_eeg_click"]
    assert h.control()["saved_calibration_source"] == "calibration-test.json"
    assert decoder._training == saved["training"]
    assert decoder._context == saved["context"]
    assert decoder._calibration_context["session_id"] == "resumed-acquisition"
    assert h.service._click_profiles["eyes_closed"] == original_profiles["eyes_closed"]
    assert h.service._eeg_trial_sequence == 97
    exported = h.service.eeg_export_blink_calibration()
    assert exported["training"] == saved["training"]
    assert exported["calibration_context"]["session_id"] == "resumed-acquisition"
    assert h.service.eeg_trial("select", "train").get("ok")
    assert h.service._eeg_pending["id"] == "eeg-trial-98"
    assert h.mouse.clicks == 0


@pytest.mark.parametrize("version", [1, 2])
def test_complete_untrained_archive_keeps_all_sixteen_trials_after_upgrade_resume_and_train(monkeypatch, version):
    h = session(monkeypatch)
    original = BlinkDecoder()
    original._version = version  # Match a complete historical calibration without inventing a trained or validated model.
    for index in range(16):
        assert add(original, index, "rest" if index < 8 else "select")["ok"]
    saved = original.export_draft()
    decoder = BlinkDecoder.from_calibration_state(saved, upgrade=True)
    assert h.service.eeg_use_saved_blink_calibration(decoder, source="calibration-complete.json").get("ok")
    resumed = h.service.eeg_export_blink_calibration()
    assert resumed["version"] == VERSION and resumed["training"] == saved["training"]
    assert resumed["context"] == saved["context"] and resumed["validation"] == [] and resumed["model"] is None
    assert h.control()["decoder"]["train_counts"] == {"rest": 8, "select": 8}
    assert not h.control()["armed"] and not h.control()["can_eeg_click"]
    assert h.service.eeg_train().get("ok")  # Training remains an explicit action after restoring all retained evidence.
    frozen = h.service.eeg_export_blink_model()
    assert frozen["training"] == saved["training"] and frozen["context"] == saved["context"]
    assert frozen["validation"] == [] and not h.control()["decoder"]["passed_validation"]
    assert not h.control()["armed"] and not h.control()["can_eeg_click"] and h.mouse.clicks == 0
    assert original.export_draft() == saved  # Neither migration nor freezing mutates the caller's original draft.


def test_resume_cannot_replace_newer_current_examples(monkeypatch):
    h = session(monkeypatch)
    saved = draft()
    active = BlinkDecoder.from_calibration_state(saved, upgrade=True)
    assert h.service.eeg_use_saved_blink_calibration(active).get("ok")
    extra = signal(50, "select", session="resumed-acquisition", spacing=.3)
    assert active.add_trial("select", extra, trial_id="newest-example", start=400, end=405)["ok"]
    before = copy.deepcopy(active._training)
    old_copy = BlinkDecoder.from_calibration_state(saved, upgrade=True)
    result = h.service.eeg_use_saved_blink_calibration(old_copy)
    assert "newer calibration" in result["error"]
    assert h.service._decoder is active and active._training == before


def test_incompatible_resume_leaves_current_decoder_untouched(monkeypatch):
    h = session(monkeypatch)
    saved = draft()
    saved["context"]["channels"][0]["name"] = "different-contact"
    decoder = BlinkDecoder.from_calibration_state(saved, upgrade=True)
    previous = h.service._decoder
    result = h.service.eeg_use_saved_blink_calibration(decoder)
    assert "layout" in result["error"]
    assert h.service._decoder is previous
    assert h.mouse.clicks == 0


def test_progress_export_requires_accepted_draft_and_completed_cue(monkeypatch):
    h = session(monkeypatch)
    with pytest.raises(ValueError):
        h.service.eeg_export_blink_calibration()
    decoder = BlinkDecoder.from_calibration_state(draft(), upgrade=True)
    assert h.service.eeg_use_saved_blink_calibration(decoder).get("ok")
    assert h.service.eeg_trial("select", "train").get("ok")
    with pytest.raises(ValueError, match="finish"):
        h.service.eeg_export_blink_calibration()
    assert h.mouse.clicks == 0
