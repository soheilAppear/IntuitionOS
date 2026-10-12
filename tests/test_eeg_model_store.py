"""Model persistence tests use synthetic features and never access hardware."""

import copy
import json
import os

import numpy as np
import pytest

from core.eeg_model_store import (
    MAX_BYTES, decoder_from_archive, export_decoder, load_decoder, load_latest_decoder, save_archive, save_decoder,
)
from test_eeg_decoder import extracted, trained_decoder, validate


@pytest.mark.parametrize("failed", [False, True])
def test_backup_retains_parameters_evidence_and_original_validation(tmp_path, failed):
    decoder = trained_decoder()
    validate(decoder, rest_as="left" if failed else None)
    original = decoder.status()
    path = save_decoder(decoder, tmp_path / "calibration.json")
    restored = load_decoder(path)
    assert restored.status() == original
    assert restored.status()["passed_validation"] is not failed
    for key in decoder._model:
        np.testing.assert_array_equal(restored._model[key], decoder._model[key])
    for original_trial, saved_trial in zip(decoder._training, restored._training):
        np.testing.assert_array_equal(original_trial["features"], saved_trial["features"])
    assert restored._validation == decoder._validation
    assert restored._trial_ids == decoder._trial_ids
    assert restored._windows == decoder._windows
    assert restored.predict(extracted("left", 49)) == decoder.predict(extracted("left", 49))


def test_loading_never_rebinds_an_old_model_to_a_new_acquisition(tmp_path):
    decoder = trained_decoder()
    validate(decoder)
    restored = load_decoder(save_decoder(decoder, tmp_path / "model.json"))
    prediction = restored.predict(extracted("left", 50, session="new-session"))
    assert not prediction["valid"]
    assert "session" in prediction["reason"].lower()
    assert restored._context == decoder._context


def test_export_does_not_share_mutable_arrays_or_records_with_live_model():
    decoder = trained_decoder()
    archive = export_decoder(decoder)
    original = float(decoder._training[0]["features"][0])
    archive["decoder"]["_training"][0]["features"][0] += 5
    archive["decoder"]["_model"]["means"][0] += 5
    assert decoder._training[0]["features"][0] == original
    assert export_decoder(decoder) != archive


@pytest.mark.parametrize("change", [
    lambda a: a.update(version=99),
    lambda a: a.update(version=True),
    lambda a: a["decoder"].pop("_model"),
    lambda a: a["decoder"]["_model"]["means"].__setitem__(0, float("nan")),
    lambda a: a["decoder"]["_model"]["means"].__setitem__(0, 42.0),
    lambda a: a["decoder"]["_model"]["scales"].__setitem__(0, 0.0),
    lambda a: a["decoder"]["_model"].update(centroids=[[1.0]]),
    lambda a: a["decoder"]["_context"].update(session_id=""),
    lambda a: a["decoder"]["_context"].update(feature_profile="unknown"),
    lambda a: a["decoder"]["_training"][0].update(features=[0.0]),
    lambda a: a["decoder"]["_training"][0].update(id=a["decoder"]["_training"][1]["id"]),
    lambda a: a["decoder"]["_trial_ids"].append("missing-trial"),
    lambda a: a["decoder"]["_windows"][0].__setitem__(1, 1),
    lambda a: a["decoder"].update(_last_end=float("inf")),
])
def test_corrupt_archives_cannot_replace_an_existing_backup(tmp_path, change):
    decoder = trained_decoder()
    path = save_decoder(decoder, tmp_path / "retained.json")
    before = path.read_bytes()
    corrupt = export_decoder(decoder)
    change(corrupt)
    with pytest.raises(ValueError):
        save_archive(corrupt, path)
    assert path.read_bytes() == before
    assert load_decoder(path).status() == decoder.status()
    assert not list(tmp_path.glob("*.tmp"))


def test_forged_passed_flag_does_not_change_failed_validation():
    decoder = trained_decoder()
    validate(decoder, rest_as="left")
    archive = export_decoder(decoder)
    archive["passed_validation"] = True
    restored = decoder_from_archive(archive)
    assert not restored.status()["passed_validation"]
    assert not restored.status()["arm_eligible"]


def test_validation_records_are_preserved_independently():
    decoder = trained_decoder()
    validate(decoder)
    archive = export_decoder(decoder)
    restored = decoder_from_archive(archive)
    archive["decoder"]["_validation"][0]["prediction"]["label"] = "right"
    assert restored._validation == decoder._validation


def test_oversized_and_invalid_json_files_fail_without_code_execution(tmp_path):
    path = tmp_path / "bad.json"
    path.write_bytes(b" " * (MAX_BYTES + 1))
    with pytest.raises(ValueError, match="too large"):
        load_decoder(path)
    path.write_text("__import__('os').system('anything')", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_decoder(path)


def test_untrained_decoder_is_not_reported_as_a_saved_model():
    decoder = trained_decoder()
    decoder.reset()
    with pytest.raises(ValueError, match="trained"):
        export_decoder(decoder)


def test_explicit_inference_binding_preserves_trials_parameters_and_failed_evaluation(tmp_path):
    decoder = trained_decoder()
    validate(decoder, rest_as="left")
    original_archive = export_decoder(decoder)["decoder"]
    original_evaluation = copy.deepcopy(decoder.status()["evaluation"])
    original_context = copy.deepcopy(decoder._context)
    current_context = {**original_context, "session_id": "explicit-new-acquisition"}
    assert decoder.bind_prediction_context(current_context) == {"ok": True}
    current_context["session_id"] = "caller-mutation"  # Binding must own its copy of the session identity.
    prediction = decoder.predict(extracted("left", 50, session="explicit-new-acquisition"))
    assert prediction["valid"] and prediction["label"] == "left"
    assert not decoder.predict(extracted("left", 50, session="another-session"))["valid"]
    assert not decoder.predict(extracted("left", 50))["valid"]
    assert decoder._context == original_context
    assert decoder.status()["evaluation"] == original_evaluation
    assert decoder.status()["validation_scope"] == "saved_acquisition"
    assert not decoder.status()["passed_validation"] and not decoder.status()["arm_eligible"]
    assert export_decoder(decoder)["decoder"] == original_archive
    assert not decoder.add_trial("left", extracted("left", 51), phase="validate", trial_id=51)["ok"]
    restored = load_decoder(save_decoder(decoder, tmp_path / "preserved.json"))
    assert restored._prediction_context is None  # A saved file cannot carry permission for another live stream.
    assert restored._context == original_context
    assert restored.status()["evaluation"] == original_evaluation


@pytest.mark.parametrize("change", [
    lambda context: context.update(session_id=""),
    lambda context: context.update(nominal_hz=500),
    lambda context: context.update(feature_profile="changed"),
    lambda context: context["channels"][0].update(name="different-electrode"),
])
def test_incompatible_inference_binding_does_not_modify_existing_model(change):
    decoder = trained_decoder()
    before = export_decoder(decoder)["decoder"]
    context = copy.deepcopy(decoder._context)
    change(context)
    with pytest.raises(ValueError):
        decoder.bind_prediction_context(context)
    assert decoder._prediction_context is None
    assert export_decoder(decoder)["decoder"] == before


def test_latest_model_loader_ignores_reports_and_returns_newest_archive(tmp_path):
    decoder = trained_decoder()
    first = save_decoder(decoder, tmp_path / "model-first.json")
    validate(decoder, rest_as="left")
    latest = save_decoder(decoder, tmp_path / "model-second.json")
    os.utime(first, ns=(1, 1))
    (tmp_path / "validation-report.json").write_text("{}", encoding="utf-8")
    restored, path = load_latest_decoder(tmp_path)
    assert path == latest.resolve()
    assert restored.status() == decoder.status()


def test_corrupt_latest_archive_does_not_silently_select_an_older_model(tmp_path):
    old = save_decoder(trained_decoder(), tmp_path / "model-old.json")
    os.utime(old, ns=(1, 1))
    (tmp_path / "model-new.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        load_latest_decoder(tmp_path)


def test_missing_archive_folder_has_clear_error(tmp_path):
    with pytest.raises(ValueError, match="No saved EEG model"):
        load_latest_decoder(tmp_path / "missing")
