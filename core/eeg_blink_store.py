"""Separate local JSON backups for deliberate double-blink calibration.

Files contain inert decoder evidence and parameters. Restoring a file neither
binds it to a live acquisition nor enables hand or EEG mouse input.
"""

import json
import os
import tempfile
from pathlib import Path


MAX_BYTES = 1024 * 1024  # Bound both file reads and writes before accepting a local calibration archive.


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON value is not allowed: {value}")  # Python's permissive NaN/Infinity parser must not accept invalid evidence.


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON fields are not allowed in a blink model.")  # A second field cannot silently replace the first schema value.
        result[key] = value
    return result


def _decode(encoded):
    if len(encoded) > MAX_BYTES:
        raise ValueError("Double-blink model archive is too large.")
    try:
        return json.loads(encoded.decode("utf-8-sig"), parse_constant=_reject_constant,
                          object_pairs_hook=_unique_object)  # Only data is parsed; executable model formats are never loaded.
    except RecursionError as exc:
        raise ValueError("Double-blink model nesting is invalid.") from exc  # Deep malformed JSON remains a normal storage error.


def _validated_decoder(state):
    from core.eeg_blink import BlinkDecoder  # Keep detector imports deferred until this separate archive type is actually requested.

    decoder = BlinkDecoder.from_state(state)  # The decoder owns schema, evidence, and deterministic parameter verification.
    if decoder.status().get("trained") is not True:
        raise ValueError("Only a trained double-blink decoder can be saved or restored.")
    return decoder  # Validation failure remains intact; this module never grants permission to bypass it.


def save_archive(state, path):
    """Validate a detached decoder state, then atomically save the inert JSON."""
    return _save_validated(state, path, _validated_decoder)  # Frozen models retain their stricter trained-state restore path.


def _validated_calibration(state):
    from core.eeg_blink import BlinkDecoder

    decoder = BlinkDecoder.from_calibration_state(state, upgrade=True)  # Validate partial evidence before applying the explicitly supported extractor upgrade.
    status = decoder.status()
    counts = status.get("train_counts") or {}
    if status.get("trained") is not False or sum(value for value in counts.values() if type(value) is int) <= 0:
        raise ValueError("Calibration progress requires accepted training trials and no frozen model.")
    return decoder  # Restoring progress grants neither an acquisition binding nor permission to click.


def save_calibration_archive(state, path):
    """Save accepted, untrained blink evidence in its separate draft schema."""
    return _save_validated(state, path, _validated_calibration)


def _save_validated(state, path, validator):
    try:
        encoded = json.dumps(state, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, RecursionError) as exc:
        raise ValueError("Double-blink model contains unsupported JSON data.") from exc
    validator(_decode(encoded))  # Validate the exact bytes being saved, without retaining references to a live model.
    destination = Path(path)
    if destination.is_symlink() or destination.parent.is_symlink():
        raise ValueError("Double-blink model paths cannot be symbolic links.")  # Never replace or follow an alternate archive target.
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())  # Complete the file before publishing its final filename.
        os.replace(temporary, destination)  # Readers see either the preceding archive or a complete replacement.
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)  # A failed write cannot leave partial calibration data masquerading as a saved model.
    return destination


def load_decoder(path):
    """Restore a bounded JSON file without rebinding or arming its decoder."""
    return _load_validated(path, _validated_decoder)


def load_calibration(path):
    """Restore canonical untrained progress; maintenance wrappers are not this schema."""
    return _load_validated(path, _validated_calibration)


def _load_validated(path, validator):
    source = Path(path)
    if source.is_symlink() or source.parent.is_symlink():
        raise ValueError("Double-blink model paths cannot be symbolic links.")
    with source.open("rb") as handle:
        encoded = handle.read(MAX_BYTES + 1)  # Read at most one byte beyond the permitted archive size.
    return validator(_decode(encoded))


def load_latest_decoder(directory):
    """Load only the newest double-blink archive from the fixed local directory."""
    latest = _latest_path(directory, "double-blink-*.json", "model")
    return load_decoder(latest), latest  # A corrupt newest archive fails visibly instead of silently selecting another model.


def load_latest_calibration(directory):
    """Read only canonical calibration files, never maintenance or frozen-model backups."""
    latest = _latest_path(directory, "calibration-*.json", "calibration progress")
    return load_calibration(latest), latest  # Explicit progress loading cannot fall back to a trained or unrelated archive.


def _latest_path(directory, pattern, label):
    requested = Path(directory)
    if requested.is_symlink():
        raise ValueError("Double-blink model folders cannot be symbolic links.")
    folder = requested.resolve()
    if not folder.is_dir():
        raise ValueError(f"No saved double-blink {label} is available. Save accepted calibration first.")
    candidates = []
    for path in folder.glob(pattern):  # Each caller admits only its own fixed archive prefix.
        if path.is_symlink():
            continue  # Reject links before resolving them, including malformed links that point back to themselves.
        resolved = path.resolve()
        if resolved.parent != folder or not resolved.is_file():
            continue  # An archive name cannot redirect a read outside this specific local folder.
        candidates.append(resolved)
    if not candidates:
        raise ValueError(f"No saved double-blink {label} is available. Save accepted calibration first.")
    return max(candidates, key=lambda item: (item.stat().st_mtime_ns, item.name))
