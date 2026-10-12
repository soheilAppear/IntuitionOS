"""Separate blink archives use synthetic calibration and never read real EEG or emit input."""

import json
import copy
import os
from pathlib import Path

import pytest

from core import eeg_blink_store as store


@pytest.fixture
def decoder():
    from test_eeg_blink import trained_decoder  # Share the model tests' synthetic evidence, never an actual user's saved calibration.

    return trained_decoder()


@pytest.fixture
def calibration():
    from core.eeg_blink import BlinkDecoder
    from test_eeg_blink import add

    decoder = BlinkDecoder()
    for index, label in enumerate(("rest", "select", "rest")):
        assert add(decoder, index, label)["ok"]  # Accepted synthetic evidence remains intentionally incomplete and unfrozen.
    return decoder.export_draft()


def test_blink_archive_round_trip_preserves_evidence_parameters_and_status(tmp_path, decoder):
    state = decoder.export_state()
    path = store.save_archive(state, tmp_path / "double-blink-saved.json")
    restored = store.load_decoder(path)
    assert restored.export_state() == state
    assert restored.status() == decoder.status()
    assert json.loads(path.read_text(encoding="utf-8")) == state  # Save the decoder's inert versioned schema without rewriting its evidence.
    state["context"]["session_id"] = "caller-mutation"
    assert restored.export_state()["context"]["session_id"] != "caller-mutation"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("failed", [False, True])
def test_restoring_preserves_original_passed_or_failed_validation_without_bypass(tmp_path, decoder, failed):
    from test_eeg_blink import signal

    for index in range(16, 32):
        label = "rest" if index % 2 == 0 else "select"
        stream = signal(index, "rest" if failed else label)  # Missing commanded blinks produce a real held-out failure.
        assert decoder.add_trial(label, stream, phase="validate", trial_id=index,
                                 start=100 + index * 6, end=105 + index * 6)["ok"]
    before = decoder.status()
    assert before["evaluation"]["complete"] and before["passed_validation"] is not failed
    path = store.save_archive(decoder.export_state(), tmp_path / "double-blink-evaluated.json")
    restored = store.load_decoder(path)
    assert restored.status() == before
    assert restored.status()["arm_eligible"] is not failed
    assert '"samples":' not in path.read_text(encoding="utf-8")  # Calibration archives retain derived evidence, never the raw EEG packets.


def test_archive_never_carries_live_acquisition_binding(tmp_path, decoder):
    original = decoder.export_state()
    context = {**original["context"], "session_id": "explicitly-bound-new-session"}
    decoder.bind_prediction_context(context)
    assert decoder.status()["inference_context_bound"]
    path = store.save_archive(decoder.export_state(), tmp_path / "double-blink-bound.json")
    restored = store.load_decoder(path)
    assert restored.export_state() == original  # Export keeps historical calibration context even after live inference was rebound.
    assert not restored.status()["inference_context_bound"]  # Loading data does not confer permission for the currently connected headset.
    assert restored.status()["context"] == original["context"]


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("failed_validation", [False, True])
def test_historical_frozen_archives_keep_version_model_validation_and_blink_behavior(tmp_path, version, failed_validation):
    from core.eeg_blink import BlinkDecoder
    from test_eeg_blink import add, events, playback, signal

    historical = BlinkDecoder()
    historical._version = version  # Collect through the historical extractor instead of relabeling newer derived features.
    for index in range(16):
        assert add(historical, index, "rest" if index % 2 == 0 else "select")["ok"]
    assert historical.train()["ok"]
    for index in range(16, 32):
        label = "rest" if index % 2 == 0 else "select"
        stream = signal(index, "rest" if failed_validation else label)
        assert historical.add_trial(label, stream, phase="validate", trial_id=index,
                                    start=100 + index * 6, end=105 + index * 6)["ok"]
    state = historical.export_state()
    path = store.save_archive(state, tmp_path / f"double-blink-v{version}.json")
    encoded = path.read_bytes()
    restored = store.load_decoder(path)
    assert restored.export_state() == state  # Frozen archives never receive the draft loader's classifier upgrade.
    assert restored.status() == historical.status()
    assert restored.status()["detector_version"] == version
    assert restored.status()["passed_validation"] is not failed_validation
    assert ("template" if version == 1 else "templates") in restored.export_state()["model"]
    for index, spacing, expected in ((40, .55, 1), (41, .3, 0)):
        restored.reset_stream()
        assert len(events(playback(restored, signal(index, "select", spacing=spacing)))) == expected  # These older models trained only on wide pairs; loading must not add new timing tolerance.
    assert path.read_bytes() == encoded  # Loading and prediction cannot rewrite the historical file or its evaluation.


@pytest.mark.parametrize("change", [
    lambda state: state.update(format="intuitionos.eeg-model"),  # Eyes-closed archives are never interchangeable with blink evidence.
    lambda state: state.update(version=True),
    lambda state: state.update(version=999),
    lambda state: state.pop("model"),
    lambda state: state.update(model=None),
    lambda state: state.update(training=[]),
    lambda state: state["context"].update(session_id=""),
    lambda state: state.update(passed_validation=True),  # Caller-supplied eligibility flags cannot bypass the decoder's validation.
])
def test_bad_schema_or_inconsistent_evidence_cannot_replace_existing_blink_model(tmp_path, decoder, change):
    path = store.save_archive(decoder.export_state(), tmp_path / "double-blink-retained.json")
    before = path.read_bytes()
    corrupt = decoder.export_state()
    change(corrupt)
    with pytest.raises(ValueError):
        store.save_archive(corrupt, path)
    assert path.read_bytes() == before
    assert store.load_decoder(path).export_state() == decoder.export_state()
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("change", [
    lambda state: state["model"].update(threshold=str(state["model"]["threshold"])),
    lambda state: state["model"].update(amplitude_min=str(state["model"]["amplitude_min"])),
    lambda state: state["model"]["noise"].__setitem__(0, str(state["model"]["noise"][0])),
    lambda state: state["model"]["templates"][0].__setitem__(0, str(state["model"]["templates"][0][0])),
    lambda state: state["model"]["template_trial_indices"].__setitem__(0, str(state["model"]["template_trial_indices"][0])),
    lambda state: state["model"]["template_trial_indices"].__setitem__(0, state["model"]["template_trial_indices"][0] + 1e-13),
])
def test_numerically_matching_wrong_model_types_never_replace_or_load_a_saved_archive(tmp_path, decoder, change):
    path = store.save_archive(decoder.export_state(), tmp_path / "double-blink-retained.json")
    retained = path.read_bytes()
    malformed = decoder.export_state()
    change(malformed)  # Float conversion alone would make each corrupt value appear to match its valid fitted parameter.
    with pytest.raises(ValueError):
        store.save_archive(malformed, path)
    assert path.read_bytes() == retained  # Failed schema checks must finish before atomic replacement can touch valid evidence.
    corrupt_path = tmp_path / "double-blink-malformed.json"
    corrupt_path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValueError):
        store.load_decoder(corrupt_path)  # The same schema is required when malformed data arrives through an existing file.
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("payload", [b'__import__("os").system("anything")', b'{"model":NaN}',
                                       b'{"model":Infinity}', b'{"version":1,"version":2}', b"\xff",
                                       b"[" * 2000 + b"]" * 2000])  # Excessive nesting must become a visible validation error.
def test_invalid_json_and_duplicate_fields_fail_before_model_restore(tmp_path, monkeypatch, payload):
    path = tmp_path / "double-blink-invalid.json"
    path.write_bytes(payload)
    monkeypatch.setattr(store, "_validated_decoder", lambda _state: pytest.fail("Malformed JSON reached the decoder"))
    with pytest.raises(ValueError):
        store.load_decoder(path)


def test_oversized_reads_and_writes_are_rejected_before_decoding_or_creating_files(tmp_path, monkeypatch):
    path = tmp_path / "double-blink-large.json"
    path.write_bytes(b" " * (store.MAX_BYTES + 1))
    monkeypatch.setattr(store, "_validated_decoder", lambda _state: pytest.fail("Oversized JSON reached the decoder"))
    with pytest.raises(ValueError, match="too large"):
        store.load_decoder(path)
    destination = tmp_path / "double-blink-new.json"
    with pytest.raises(ValueError, match="too large"):
        store.save_archive({"padding": " " * store.MAX_BYTES}, destination)
    assert not destination.exists() and not list(tmp_path.glob("*.tmp"))


def test_nonfinite_and_python_objects_are_never_written(tmp_path):
    path = tmp_path / "double-blink-invalid.json"
    for value in (float("nan"), float("inf"), object(), {1, 2}):
        with pytest.raises(ValueError):
            store.save_archive({"model": value}, path)
        assert not path.exists()


def test_failed_atomic_replace_preserves_previous_archive_and_cleans_temporary_file(tmp_path, monkeypatch, decoder):
    path = store.save_archive(decoder.export_state(), tmp_path / "double-blink-retained.json")
    before = path.read_bytes()

    def failure(*_args):
        raise OSError("simulated local write failure")

    monkeypatch.setattr(store.os, "replace", failure)
    with pytest.raises(OSError, match="simulated"):
        store.save_archive(decoder.export_state(), path)
    assert path.read_bytes() == before and not list(tmp_path.glob("*.tmp"))


def test_latest_loader_ignores_legacy_archives_and_reports(tmp_path, decoder):
    state = decoder.export_state()
    first = store.save_archive(state, tmp_path / "double-blink-first.json")
    latest = store.save_archive(state, tmp_path / "double-blink-second.json")
    os.utime(first, ns=(1, 1))
    (tmp_path / "model-original.json").write_text("{}", encoding="utf-8")
    (tmp_path / "validation-report.json").write_text("{}", encoding="utf-8")
    restored, selected = store.load_latest_decoder(tmp_path)
    assert selected == latest.resolve() and restored.export_state() == state
    assert (tmp_path / "model-original.json").read_text(encoding="utf-8") == "{}"


def test_corrupt_newest_blink_archive_does_not_silently_load_an_older_model(tmp_path, decoder):
    old = store.save_archive(decoder.export_state(), tmp_path / "double-blink-old.json")
    os.utime(old, ns=(1, 1))
    (tmp_path / "double-blink-new.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        store.load_latest_decoder(tmp_path)


@pytest.mark.parametrize("existing", [False, True])
def test_missing_blink_model_does_not_consider_an_eyes_closed_archive(tmp_path, existing):
    folder = tmp_path / "eeg-blink-models"
    if existing:
        folder.mkdir()
        (folder / "model-original.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="No saved double-blink model"):
        store.load_latest_decoder(folder)


def test_symbolic_archive_is_rejected_before_read_or_replacement(tmp_path, monkeypatch, decoder):
    link = tmp_path / "double-blink-link.json"
    link.write_text("sentinel", encoding="utf-8")
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == link or original(path))  # Simulate the filesystem flag without requiring Windows symlink privileges.
    with pytest.raises(ValueError, match="symbolic"):
        store.load_decoder(link)
    with pytest.raises(ValueError, match="symbolic"):
        store.save_archive(decoder.export_state(), link)
    with pytest.raises(ValueError, match="No saved double-blink model"):
        store.load_latest_decoder(tmp_path)
    assert link.read_text(encoding="utf-8") == "sentinel"


def test_symbolic_archive_folder_cannot_redirect_storage(tmp_path, monkeypatch, decoder):
    folder = tmp_path / "redirected"
    folder.mkdir()
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == folder or original(path))
    with pytest.raises(ValueError, match="symbolic"):
        store.load_latest_decoder(folder)
    with pytest.raises(ValueError, match="symbolic"):
        store.save_archive(decoder.export_state(), folder / "double-blink-saved.json")
    assert not list(folder.iterdir())


def test_untrained_state_cannot_be_advertised_as_a_saved_blink_model(tmp_path, monkeypatch):
    from core.eeg_blink import BlinkDecoder

    untrained = BlinkDecoder()
    monkeypatch.setattr(BlinkDecoder, "from_state", classmethod(lambda _cls, _state: untrained))
    with pytest.raises(ValueError, match="trained"):
        store.save_archive({}, tmp_path / "double-blink-empty.json")


def test_partial_calibration_round_trip_preserves_features_without_model_or_input_permission(tmp_path, calibration):
    path = store.save_calibration_archive(calibration, tmp_path / "calibration-partial.json")
    restored = store.load_calibration(path)
    assert restored.export_draft() == calibration
    status = restored.status()
    assert status["train_counts"] == {"rest": 2, "select": 1}
    assert not status["trained"] and not status["passed_validation"] and not status["arm_eligible"]
    assert not status["inference_context_bound"]
    assert '"samples":' not in path.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        store.load_decoder(path)  # A draft can never silently satisfy the trained-model endpoint.


@pytest.mark.parametrize("version", [1, 2])
def test_legacy_partial_calibration_upgrade_retains_exact_accepted_evidence(tmp_path, version):
    from core.eeg_blink import BlinkDecoder, VERSION
    from test_eeg_blink import add

    historical = BlinkDecoder()
    historical._version = version  # Keep real historical features in the fixture rather than changing a newer archive's version tag.
    for index, label in enumerate(("rest", "select", "rest")):
        assert add(historical, index, label)["ok"]
    calibration = historical.export_draft()
    path = store.save_calibration_archive(calibration, tmp_path / "calibration-legacy.json")
    encoded = path.read_bytes()
    restored = store.load_calibration(path).export_draft()
    assert restored == {**calibration, "version": VERSION}  # Only the explicitly requested future classifier changes; every accepted feature stays exact.
    assert path.read_bytes() == encoded and json.loads(encoded)["version"] == version  # Loading never rewrites the original archived provenance.


@pytest.mark.parametrize("change", [
    lambda state: state.update(format="intuitionos.eeg-blink-model"),
    lambda state: state.update(training=[]),
    lambda state: state.update(model={}),
    lambda state: state.update(validation=[{}]),
    lambda state: state["context"].update(session_id=""),
    lambda state: state["training"].append(copy.deepcopy(state["training"][0])),
])
def test_invalid_progress_cannot_replace_saved_partial_calibration(tmp_path, calibration, change):
    path = store.save_calibration_archive(calibration, tmp_path / "calibration-retained.json")
    before = path.read_bytes()
    corrupt = copy.deepcopy(calibration)
    change(corrupt)
    with pytest.raises(ValueError):
        store.save_calibration_archive(corrupt, path)
    assert path.read_bytes() == before and not list(tmp_path.glob("*.tmp"))


def test_progress_loader_ignores_maintenance_wrappers_and_frozen_models(tmp_path, calibration, decoder):
    path = store.save_calibration_archive(calibration, tmp_path / "calibration-progress.json")
    os.utime(path, ns=(1, 1))
    (tmp_path / "blink-draft-maintenance.json").write_text(json.dumps({"format": "intuitionos.eeg-blink-draft", "decoder": calibration}), encoding="utf-8")
    store.save_archive(decoder.export_state(), tmp_path / "double-blink-frozen.json")
    restored, selected = store.load_latest_calibration(tmp_path)
    assert selected == path.resolve() and restored.export_draft() == calibration
    (tmp_path / "calibration-new-corrupt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        store.load_latest_calibration(tmp_path)  # A corrupt newest canonical draft cannot silently select older evidence.


def test_progress_loader_rejects_missing_drafts_and_renamed_maintenance_wrapper(tmp_path, calibration):
    wrapper = {"format": "intuitionos.eeg-blink-draft", "decoder": calibration}
    (tmp_path / "blink-draft-maintenance.json").write_text(json.dumps(wrapper), encoding="utf-8")
    with pytest.raises(ValueError, match="No saved double-blink calibration progress"):
        store.load_latest_calibration(tmp_path)
    path = tmp_path / "calibration-renamed-wrapper.json"
    path.write_text(json.dumps(wrapper), encoding="utf-8")
    with pytest.raises(ValueError):
        store.load_latest_calibration(tmp_path)  # Renaming a maintenance wrapper does not convert its schema.


def test_progress_round_trip_preserves_explicit_new_session_provenance(tmp_path, calibration):
    from core.eeg_blink import BlinkDecoder
    from test_eeg_blink import signal

    decoder = BlinkDecoder.from_calibration_state(calibration, upgrade=True)
    context = {**calibration["context"], "session_id": "explicit-resumed-acquisition"}
    assert decoder.resume_calibration(context)["ok"]
    stream = signal(10, "select", session=context["session_id"])
    assert decoder.add_trial("select", stream, trial_id="resumed-trial", start=160, end=165)["ok"]
    state = decoder.export_draft()
    path = store.save_calibration_archive(state, tmp_path / "calibration-resumed.json")
    restored = store.load_calibration(path).export_draft()
    assert restored == state
    assert restored["context"] == calibration["context"] and restored["calibration_context"] == context
    assert restored["training"][:3] == calibration["training"]  # Historical trial records are never relabeled as new-session measurements.
    assert restored["training"][-1]["session_id"] == context["session_id"]
