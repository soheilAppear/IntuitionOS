"""Fast CNN archives cannot mix with the existing five-second calibration."""

import copy
import json
import os
from pathlib import Path

import pytest

from core import eeg_fast_blink_store as store
from test_eeg_fast_blink import trained_decoder


@pytest.fixture
def state(monkeypatch):
    from core import eeg_fast_blink

    class ArchiveDecoder:
        @classmethod
        def from_state(cls, value):
            if (set(value) != {"format", "trained", "passed_validation"}
                    or value["format"] != eeg_fast_blink.FORMAT):
                raise ValueError("Wrong model family")
            result = cls()
            result.state = copy.deepcopy(value)
            return result

        def status(self):
            return copy.deepcopy(self.state)

        def export_state(self):
            return copy.deepcopy(self.state)

    monkeypatch.setattr(eeg_fast_blink, "FastCnnBlinkDecoder", ArchiveDecoder)
    return {"format": eeg_fast_blink.FORMAT, "trained": True, "passed_validation": False}


@pytest.mark.parametrize("passed", [False, True])
def test_round_trip_keeps_original_validation_and_does_not_mutate(tmp_path, state, passed):
    state["passed_validation"] = passed
    path = store.save_archive(state, tmp_path / "fast-cnn-blink-test.json")
    restored = store.load_decoder(path)
    assert restored.export_state() == state == json.loads(path.read_bytes())
    state["passed_validation"] = not passed
    assert restored.status()["passed_validation"] is passed
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("name", ["cnn-blink-old.json", "double-blink-old.json", "model-old.json", "fast-cnn-blink-model.pkl"])
def test_another_family_is_never_read_or_replaced(tmp_path, state, name):
    path = tmp_path / name
    path.write_text("retained", encoding="utf-8")
    with pytest.raises(ValueError, match="filename"):
        store.load_decoder(path)
    with pytest.raises(ValueError, match="filename"):
        store.save_archive(state, path)
    assert path.read_text(encoding="utf-8") == "retained"


@pytest.mark.parametrize("change", [
    {"format": "intuitionos.eeg-cnn-blink-model"}, {"trained": False},
])
def test_invalid_model_cannot_replace_archive(tmp_path, state, change):
    path = store.save_archive(state, tmp_path / "fast-cnn-blink-retained.json")
    original = path.read_bytes()
    with pytest.raises(ValueError):
        store.save_archive({**state, **change}, path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("payload", [
    b'__import__("os").system("invalid")', b'{"weight":NaN}',
    b'{"format":1,"format":2}', b"\xff", b"[" * 100 + b"]" * 100, b"[]",
])
def test_bad_json_never_reaches_decoder(tmp_path, monkeypatch, payload):
    path = tmp_path / "fast-cnn-blink-invalid.json"
    path.write_bytes(payload)
    monkeypatch.setattr(store, "_validated_decoder", lambda *_: pytest.fail("Invalid JSON reached model loading"))
    with pytest.raises(ValueError):
        store.load_decoder(path)


def test_size_limit_applies_before_read_restore_or_write(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "MAX_BYTES", 256)
    monkeypatch.setattr(store, "_validated_decoder", lambda *_: pytest.fail("Oversized input reached decoder"))
    path = tmp_path / "fast-cnn-blink-big.json"
    path.write_bytes(b" " * 257)
    with pytest.raises(ValueError, match="too large"):
        store.load_decoder(path)
    with pytest.raises(ValueError, match="too large"):
        store.save_archive({"value": "x" * 257}, path)
    assert path.read_bytes() == b" " * 257


def test_atomic_failure_keeps_existing_archive(tmp_path, monkeypatch, state):
    path = store.save_archive(state, tmp_path / "fast-cnn-blink-retained.json")
    original = path.read_bytes()

    def fail_replace(*_):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(store.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        store.save_archive(state, path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_latest_is_profile_specific_and_does_not_hide_corrupt_newest(tmp_path, state):
    old = store.save_archive(state, tmp_path / "fast-cnn-blink-old.json")
    os.utime(old, ns=(1, 1))
    (tmp_path / "cnn-blink-existing.json").write_text("legacy untouched", encoding="utf-8")
    _, selected = store.load_latest_decoder(tmp_path)
    assert selected == old.resolve()
    (tmp_path / "fast-cnn-blink-new.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        store.load_latest_decoder(tmp_path)
    assert (tmp_path / "cnn-blink-existing.json").read_text(encoding="utf-8") == "legacy untouched"


def test_missing_fast_archive_does_not_load_old_model(tmp_path):
    (tmp_path / "cnn-blink-existing.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="No saved one-second"):
        store.load_latest_decoder(tmp_path)


@pytest.mark.parametrize("link_folder", [False, True])
def test_symbolic_paths_cannot_redirect_fast_storage(tmp_path, state, monkeypatch, link_folder):
    path = tmp_path / "fast-cnn-blink-link.json"
    path.write_text("retained", encoding="utf-8")
    original = Path.is_symlink
    link = tmp_path if link_folder else path
    monkeypatch.setattr(Path, "is_symlink", lambda value: value == link or original(value))
    with pytest.raises(ValueError, match="symbolic"):
        store.load_decoder(path)
    with pytest.raises(ValueError, match="symbolic"):
        store.save_archive(state, path)
    with pytest.raises(ValueError):
        store.load_latest_decoder(tmp_path)
    assert path.read_text(encoding="utf-8") == "retained"


@pytest.mark.parametrize("failed_validation", [False, True])
def test_actual_fast_weights_and_evidence_restore_without_training_or_new_binding(tmp_path, monkeypatch, trained_decoder, failed_validation):
    import numpy as np

    from core import eeg_fast_blink as cnn
    from test_eeg_fast_blink import signal

    decoder = cnn.FastCnnBlinkDecoder.from_state(trained_decoder.export_state())
    for index in range(16, 32):
        label = "select" if index % 2 else "rest"
        result = decoder.add_trial(label, signal(index, "select" if failed_validation else label),
                                   phase="validate", trial_id=index,
                                   start=100 + index * 2, end=101 + index * 2)
        assert result["ok"], result.get("error")
    assert decoder.status()["passed_validation"] is not failed_validation
    original = decoder.export_state()
    decoder.bind_prediction_context({**original["context"], "session_id": "new-live-session"})
    assert decoder.status()["inference_context_bound"]

    def forbidden(*_args, **_kwargs):
        pytest.fail("Loading archives cannot train a model or construct a Torch network")

    monkeypatch.setattr(cnn.FastCnnBlinkDecoder, "train", forbidden)
    monkeypatch.setattr(cnn, "_torch_network", forbidden)
    path = store.save_archive(decoder.export_state(), tmp_path / "fast-cnn-blink-actual.json")
    restored = store.load_decoder(path)
    assert restored.export_state() == original
    assert restored.status()["passed_validation"] is not failed_validation
    assert not restored.status()["inference_context_bound"]
    waveform = cnn.preprocess_waveform([row["samples"] for row in signal(42, "select")["samples"]])
    np.testing.assert_array_equal(cnn._scores(waveform, restored._weights), cnn._scores(waveform, decoder._weights))
    legacy_path = tmp_path / "cnn-blink-wrong-profile.json"
    legacy_path.write_bytes(path.read_bytes())
    with pytest.raises(ValueError):
        store.load_decoder(legacy_path)
    assert not list(tmp_path.glob("*.tmp"))
