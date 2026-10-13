"""Bounded, inert JSON archives for the one-second EEG blink CNN.

Reading a model does not train it, bind a live acquisition, or arm mouse input.
The original fast-model schema stays stable so existing calibration still loads.
"""

import json
import math
import os
import tempfile
from pathlib import Path

MAX_BYTES = 16 * 1024 * 1024  # Limit the file before parsing or constructing arrays.
MAX_DEPTH = 16  # Reject deeply nested input before the JSON parser allocates it.
MAX_VALUES = 1_000_000  # Bound the total containers and scalar values together.
PREFIX = "fast-cnn-blink-"  # Keep the established saved-model filenames compatible.


def _plain(value, depth=0, budget=None):
    """Permit only finite JSON values within a shared traversal budget."""
    budget = [MAX_VALUES] if budget is None else budget  # Siblings share one limit.
    budget[0] -= 1  # Count containers as well as their scalar contents.
    if depth > MAX_DEPTH or budget[0] < 0:
        raise ValueError("Fast CNN archive nesting or value count is too large.")
    if value is None or type(value) in (str, bool, int):
        return  # These scalars have an unambiguous JSON representation.
    if type(value) is float and math.isfinite(value):
        return  # NaN and infinities are invalid weights and invalid portable JSON.
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _plain(item, depth + 1, budget)  # Inspect every nested mapping value.
        return
    if type(value) is list:
        for item in value:
            _plain(item, depth + 1, budget)  # Reject Python objects hidden in arrays.
        return
    raise ValueError("Fast CNN archive contains unsupported or nonfinite values.")


def _reject_constant(value):
    raise ValueError("Fast CNN archive contains a nonfinite JSON number: " + value)


def _unique_object(pairs):
    result = {}  # Preserve only mappings with one value for each field name.
    for key, value in pairs:
        if key in result:
            raise ValueError("Fast CNN archive contains duplicate JSON fields.")
        result[key] = value  # A repeated field must never silently replace a weight.
    return result


def _decode(encoded):
    """Apply structural bounds before validating the decoder's numeric schema."""
    if len(encoded) > MAX_BYTES:
        raise ValueError("Fast CNN archive is too large.")
    depth = 0  # Scan structural bytes before entering the platform JSON parser.
    quoted = escaped = False  # Brackets inside JSON strings do not increase depth.
    for byte in encoded:
        if quoted:
            if escaped:
                escaped = False  # An escaped quote cannot terminate the string.
            elif byte == 92:
                escaped = True  # Remember a backslash until the following byte.
            elif byte == 34:
                quoted = False  # An unescaped quote closes the current string.
        elif byte == 34:
            quoted = True
        elif byte in (91, 123):
            depth += 1  # Arrays and objects consume the same nesting budget.
            if depth > MAX_DEPTH:
                raise ValueError("Fast CNN archive nesting is too deep.")
        elif byte in (93, 125):
            depth -= 1  # The JSON parser will reject unbalanced closing brackets.
    try:
        state = json.loads(encoded.decode("utf-8-sig"), parse_constant=_reject_constant,
                           object_pairs_hook=_unique_object)  # Reject ambiguous or nonfinite JSON.
        if type(state) is not dict:
            raise ValueError("Fast CNN archive must be a JSON object.")
        _plain(state)  # Apply the value-count bound before allocating CNN tensors.
        return state
    except RecursionError as exc:
        raise ValueError("Fast CNN archive nesting is too deep.") from exc


def _validated_decoder(state):
    from core.eeg_fast_blink import FastCnnBlinkDecoder

    decoder = FastCnnBlinkDecoder.from_state(state)
    if decoder.status().get("trained") is not True:
        raise ValueError("Only a trained one-second CNN can be saved or restored.")
    return decoder


def _path(path):
    candidate = Path(path)
    if candidate.is_symlink() or candidate.parent.is_symlink():
        raise ValueError("One-second CNN archive paths cannot be symbolic links.")
    if not candidate.name.startswith(PREFIX) or candidate.suffix != ".json":
        raise ValueError("One-second CNN archives require a fast-cnn-blink-*.json filename.")
    return candidate


def save_archive(state, path):
    """Validate the short profile before atomically replacing its archive."""
    destination = _path(path)  # Reject another model family or a symbolic redirect.
    _plain(state)  # Validate the Python structure before serialization can recurse.
    try:
        encoded = json.dumps(state, ensure_ascii=False, allow_nan=False,
                             separators=(",", ":")).encode("utf-8")
    except (TypeError, RecursionError) as exc:
        raise ValueError("One-second CNN archive contains unsupported JSON data.") from exc
    _validated_decoder(_decode(encoded))  # Verify weights and their evidence before replacing anything.
    destination.parent.mkdir(parents=True, exist_ok=True)  # Create only the selected model folder.
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)  # Same volume enables atomic replacement.
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)  # Write a complete temporary archive first.
            handle.flush()  # Flush Python's buffer before requesting filesystem persistence.
            os.fsync(handle.fileno())  # Do not publish a partially buffered model.
        os.replace(temporary, destination)  # Readers see the old archive or the complete new one.
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)  # Failed writes leave an existing archive untouched.
    return destination


def load_decoder(path):
    """Restore weights and evidence without training or granting input control."""
    with _path(path).open("rb") as handle:
        encoded = handle.read(MAX_BYTES + 1)  # Read one extra byte to detect oversize without loading the whole file.
    return _validated_decoder(_decode(encoded))  # Restoring never grants native input permission.


def load_latest_decoder(directory):
    """Restore only this profile's newest archive; a corrupt newest fails visibly."""
    requested = Path(directory)
    if requested.is_symlink():
        raise ValueError("One-second CNN archive folders cannot be symbolic links.")
    folder = requested.resolve()  # Compare resolved candidates against this fixed directory.
    candidates = []
    if folder.is_dir():
        for path in folder.glob(PREFIX + "*.json"):
            if path.is_symlink():
                continue  # A matching filename cannot authorize reading an outside file.
            resolved = path.resolve()
            if resolved.parent == folder and resolved.is_file():
                candidates.append(resolved)
    if not candidates:
        raise ValueError("No saved one-second CNN model is available. Collect short examples and train first.")
    latest = max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))  # Break equal-time ties deterministically.
    return load_decoder(latest), latest
