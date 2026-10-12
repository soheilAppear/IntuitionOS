"""Local JSON backups of EEG calibration; loading never arms desktop controls.

The original acquisition context and failed validation results are preserved.
This module does not grant a saved model permission to control a new session.
"""

import copy
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from core.eeg_decoder import EegDecoder


FORMAT = "intuitionos.eeg-model"
VERSION = 1
MAX_BYTES = 1024 * 1024  # A small calibration must not require an unbounded file read.
STATE_KEYS = {"_labels", "_feature_profile", "_training", "_validation", "_trial_ids",
              "_windows", "_last_end", "_context", "_model"}
MODEL_KEYS = {"means", "scales", "centroids", "diagonal", "radii"}


def _plain(value):
    """Convert only inert calibration values, never arbitrary Python objects."""
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())  # JSON round-trips finite float64 values exactly.
    if isinstance(value, np.generic):
        return _plain(value.item())  # Normalize NumPy scalars before the JSON encoder.
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Calibration dictionaries require string keys.")
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, set):
        return sorted(_plain(item) for item in value)  # Trial IDs have deterministic ordering.
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("Calibration contains an unsupported or nonfinite value.")


def export_decoder(decoder):
    """Call while holding the service decoder lock if acquisition is concurrent."""
    if not isinstance(decoder, EegDecoder) or decoder._model is None:
        raise ValueError("Only a trained EEG decoder can be backed up.")
    # Capture just the calibration, excluding devices, locks, frames and input ownership.
    state = {key: _plain(getattr(decoder, key)) for key in sorted(STATE_KEYS)}
    return {"format": FORMAT, "version": VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(), "decoder": state}


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def decoder_from_archive(archive):
    """Validate inert JSON, restore exact parameters, and retain historical context."""
    if not isinstance(archive, dict) or archive.get("format") != FORMAT or type(archive.get("version")) is not int or archive["version"] != VERSION:
        raise ValueError("Unsupported EEG model archive.")
    # Plain JSON conversion also rejects nested nonfinite values and object payloads.
    archive = _plain(archive)
    state = archive.get("decoder")
    if not isinstance(state, dict) or set(state) != STATE_KEYS:
        raise ValueError("Incomplete EEG decoder state.")
    decoder = EegDecoder(labels=state["_labels"], feature_profile=state["_feature_profile"])
    context = state["_context"]
    if not isinstance(context, dict) or not isinstance(context.get("session_id"), str) or not 1 <= len(context["session_id"]) <= 128:
        raise ValueError("The original acquisition session is missing.")
    channels = context.get("channels")
    if (not isinstance(channels, list) or not 1 <= len(channels) <= 16
            or not _number(context.get("nominal_hz")) or not 1 <= context["nominal_hz"] <= 10000
            or context.get("feature_profile") != decoder.feature_profile):
        raise ValueError("Invalid saved acquisition layout or feature profile.")
    if any(not isinstance(channel, dict) or type(channel.get("num")) is not int
           or not isinstance(channel.get("name"), str) or not 1 <= len(channel["name"]) <= 128
           for channel in channels) or len({channel["num"] for channel in channels}) != len(channels):
        raise ValueError("Invalid saved channel identities.")
    width = len(channels) * 4  # Every supported transform emits four features per channel.
    training, validation = state["_training"], state["_validation"]
    if (not isinstance(training, list) or not isinstance(validation, list)
            or len(training) + len(validation) > decoder.MAX_TRIALS):
        raise ValueError("Invalid calibration trial counts.")
    ids = []
    for trial in training + validation:
        if (not isinstance(trial, dict) or trial.get("label") not in decoder.labels
                or type(trial.get("id")) not in (str, int) or not 1 <= len(str(trial["id"])) <= 256):
            raise ValueError("Invalid calibration trial identity.")
        ids.append(str(trial["id"]))
    if len(set(ids)) != len(ids) or not isinstance(state["_trial_ids"], list) or sorted(state["_trial_ids"]) != sorted(ids):
        raise ValueError("Calibration trial IDs do not match the stored records.")
    restored_training = []
    for trial in training:
        vector = trial.get("features")
        if not isinstance(vector, list) or len(vector) != width or not all(_number(value) and abs(value) <= 100 for value in vector):
            raise ValueError("Invalid saved training features.")
        restored_training.append({**trial, "features": np.asarray(vector, dtype=np.float64)})
    for trial in validation:
        prediction = trial.get("prediction")
        if not isinstance(prediction, dict) or prediction.get("label") not in (*decoder.labels, None):
            raise ValueError("Invalid saved validation prediction.")
    if any(sum(trial["label"] == label for trial in validation) > decoder.MIN_PER_CLASS for label in decoder.labels):
        raise ValueError("Saved validation exceeds the fixed trial quota.")
    windows = state["_windows"]
    if (not isinstance(windows, list) or len(windows) != len(ids)
            or any(not isinstance(window, list) or len(window) != 2
                   or type(window[0]) is not int or not 0 <= window[0] < 2**32
                   or type(window[1]) is not int or window[1] != round(2 * context["nominal_hz"])
                   for window in windows) or not _number(state["_last_end"])):
        raise ValueError("Invalid saved trial window identities.")
    parameters = state["_model"]
    if not isinstance(parameters, dict) or set(parameters) != MODEL_KEYS:
        raise ValueError("Incomplete saved model parameters.")
    shapes = {"means": (width,), "scales": (width,), "diagonal": (width,),
              "centroids": (len(decoder.labels), width), "radii": (len(decoder.labels),)}
    arrays = {}
    for key, shape in shapes.items():
        try:
            array = np.asarray(parameters[key], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid saved model array.") from exc
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError("Saved model array dimensions or values are invalid.")
        arrays[key] = array
    if any(np.any(arrays[key] <= 0) for key in ("scales", "diagonal", "radii")):
        raise ValueError("Saved model divisors and rejection radii must be positive.")
    decoder._training = restored_training
    decoder._context = copy.deepcopy(context)
    result = decoder.train()  # Verify the parameters against the retained training evidence.
    if not result["ok"] or any(not np.allclose(decoder._model[key], arrays[key], rtol=1e-12, atol=1e-12) for key in MODEL_KEYS):
        raise ValueError("Saved parameters do not match the saved training features.")
    decoder._model = arrays  # Retain the original float64 parameters after consistency checking.
    decoder._validation = copy.deepcopy(validation)  # Failed validation stays failed after loading.
    decoder._trial_ids = set(ids)
    decoder._windows = [tuple(window) for window in windows]
    decoder._last_end = state["_last_end"]
    return decoder  # No acquisition, model rebinding, keyboard or mouse action occurs here.


def save_archive(archive, path):
    """Validate then atomically write a local model file; no pickle or code loading."""
    decoder_from_archive(archive)  # Do not write a purported backup that cannot be restored.
    encoded = json.dumps(_plain(archive), indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_BYTES:
        raise ValueError("EEG model archive is too large.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())  # Persist the complete file before replacing its destination.
        os.replace(temporary, destination)  # Readers see the old complete file or the new complete file.
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def save_decoder(decoder, path):
    return save_archive(export_decoder(decoder), path)


def load_decoder(path):
    with Path(path).open("rb") as handle:
        encoded = handle.read(MAX_BYTES + 1)  # Reject oversized files before parsing their contents.
    if len(encoded) > MAX_BYTES:
        raise ValueError("EEG model archive is too large.")
    return decoder_from_archive(json.loads(encoded.decode("utf-8-sig")))


def load_latest_decoder(directory):
    """Load the newest regular model archive from the application's fixed folder.

    Reports and arbitrary user paths are never considered. A corrupt newest file
    raises an error instead of silently substituting a different person's model.
    """
    folder = Path(directory).resolve()
    if not folder.is_dir():
        raise ValueError("No saved EEG model is available. Train and save a model first.")
    candidates = []
    for path in folder.glob("model-*.json"):
        # Resolve symlinks before any read so an archive cannot redirect outside this local folder.
        resolved = path.resolve()
        if path.is_symlink() or resolved.parent != folder or not resolved.is_file():
            continue
        candidates.append(resolved)
    if not candidates:
        raise ValueError("No saved EEG model is available. Train and save a model first.")
    latest = max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))
    return load_decoder(latest), latest
