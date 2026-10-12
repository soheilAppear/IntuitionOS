"""Personalized, causal double-blink events; no devices, files or desktop input.

This detects a learned ocular waveform in EEG, not thoughts. A pulse by itself
never requests a click. Training and saved state are separate from bandpower EEG.
"""

from __future__ import annotations

import copy
import math
from collections import Counter, deque

import numpy as np


FEATURE_PROFILE = "double_blink_waveform_v1"
FORMAT = "intuitionos.eeg-blink-model"
DRAFT_FORMAT = "intuitionos.eeg-blink-calibration"
VERSION = 3  # Frozen v1/v2 archives retain their original extractors and fitting/matching rules.
LABELS = ("rest", "select")
MIN_PER_CLASS = 8
HZ = 250
SHAPE_BINS = 24
PAIR_MIN = 0.18  # Peak-to-peak interval, not callback arrival time.
PAIR_MAX = 0.90
REFRACTORY = 0.80  # A completed pair cannot also be half of another command.


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False  # Malformed JSON integers must fail validation, not escape as server errors.


def _context(signal):
    """Canonicalize channel ordering without silently accepting a new montage."""
    if not isinstance(signal, dict) or not isinstance(signal.get("session_id"), str) or not 1 <= len(signal["session_id"]) <= 128:
        raise ValueError("A current EEG acquisition session is required.")
    if not _finite(signal.get("nominal_hz")) or signal["nominal_hz"] != HZ:
        raise ValueError("Double-blink detection requires the nominal 250 Hz stream.")
    if signal.get("feature_profile", FEATURE_PROFILE) != FEATURE_PROFILE:
        raise ValueError("EEG feature profile does not match the double-blink decoder.")
    channels = signal.get("channels")
    if not isinstance(channels, list) or len(channels) != 4:
        raise ValueError("Double-blink detection requires four EEG channels.")
    if any(not isinstance(c, dict) or type(c.get("num")) is not int or not isinstance(c.get("name"), str)
           or not 1 <= len(c["name"]) <= 64 for c in channels):
        raise ValueError("Invalid EEG channel identity.")
    channels = sorted(({"num": c["num"], "name": c["name"]} for c in channels), key=lambda c: c["num"])
    if [c["num"] for c in channels] != list(range(4)):
        raise ValueError("EEG channel indices must uniquely cover all four samples.")
    return {"session_id": signal["session_id"], "channels": channels, "nominal_hz": HZ,
            "feature_profile": FEATURE_PROFILE}


def _rows(rows):
    """Validate sample identity before filtering; timestamps never replace counters."""
    if not isinstance(rows, (list, tuple)) or not 1 <= len(rows) <= 2500:
        raise ValueError("A bounded full-rate EEG sample buffer is required.")
    previous = None
    for row in rows:
        if (not isinstance(row, dict) or type(row.get("counter")) is not int or not 0 <= row["counter"] < 2**32
                or not _finite(row.get("estimated_monotonic")) or not isinstance(row.get("samples"), (list, tuple))
                or len(row["samples"]) != 4 or not all(_finite(v) for v in row["samples"])):
            raise ValueError("Missing, nonfinite or malformed EEG packet.")
        if previous is not None:
            if (row["counter"] - previous["counter"]) % 2**32 != 1:
                raise ValueError("EEG packet gap, duplicate or counter reset; blink candidates cancelled.")
            if row["estimated_monotonic"] - previous["estimated_monotonic"] < -0.25:
                raise ValueError("EEG timing moved backwards; blink candidates cancelled.")
        previous = row
    if len(rows) >= HZ // 2:
        expected = (len(rows) - 1) / HZ
        elapsed = rows[-1]["estimated_monotonic"] - rows[0]["estimated_monotonic"]
        if not 0.75 * expected <= elapsed <= 1.25 * expected:
            raise ValueError("EEG timing does not match the nominal sample rate.")
    return np.asarray([row["samples"] for row in rows], dtype=np.float64)


def _quality(data):
    """Retain raw engineering guards; filtering must not hide clipping or spikes."""
    if len(data) < HZ // 2:
        raise ValueError("Waiting for at least half a second of valid EEG.")
    excursion = np.ptp(data, axis=0)
    if np.any(excursion > 0.010):
        raise ValueError("EEG peak-to-peak amplitude exceeds the conservative 10 mV engineering limit.")
    if np.any(excursion < 1e-9):
        raise ValueError("EEG channel is flat; double-blink detection paused.")
    for column in data.T:
        run = 1
        low, high = column.min(), column.max()
        for before, value in zip(column, column[1:]):
            run = run + 1 if value == before else 1
            if run >= 25:
                raise ValueError("EEG has a sustained constant segment (possible flatline).")
            if run >= 5 and value in (low, high):
                raise ValueError("EEG has a repeated extreme plateau (possible clipping).")


def _noise(data):
    # Robust first differences largely exclude slow baseline shifts and rare blinks.
    difference = np.diff(data, axis=0)
    centered = difference - np.median(difference, axis=0)
    return np.maximum(np.median(np.abs(centered), axis=0) / (0.67449 * math.sqrt(2)), 1e-7)


class _PulseStream:
    """One-pass filter and completed-pulse extraction, shared by training and use."""

    def __init__(self, noise, *, split=True):
        self.noise = np.asarray(noise, dtype=float)
        self.baseline = None
        self.filtered = np.zeros(4)
        self.history = deque(maxlen=20)  # Eighty milliseconds of pre-onset context.
        self.active = None
        self.quiet = 0
        self.count = 0
        self.inhibit = False  # A long artifact must return to baseline before a new pulse.
        self.split = split  # Retain the original isolated-pulse path for compatibility checks.
        self.axis = None
        self.peak_projection = 0.0
        self.peak_index = 0
        self.valley_index = None
        self.valley_projection = math.inf
        self.rise_count = 0

    def _start(self, entries, entry):
        self.active = entries
        scaled = entry[2] / self.noise
        self.axis = scaled / max(float(np.linalg.norm(scaled)), 1e-12)
        self.peak_projection = float(np.dot(scaled, self.axis))
        self.peak_index = max(0, len(entries) - 1)
        self.valley_index = None
        self.valley_projection = math.inf
        self.rise_count = 0

    @staticmethod
    def _finish(entries):
        if len(entries) < round(0.10 * HZ):
            return None
        values = np.asarray([e[2] for e in entries])
        peak = int(np.argmax([e[3] for e in entries]))
        amplitude = np.max(np.abs(values), axis=0)
        scale = max(float(np.linalg.norm(amplitude)), 1e-12)
        positions = np.linspace(0, len(entries) - 1, SHAPE_BINS)
        shape = np.column_stack([np.interp(positions, np.arange(len(entries)), values[:, i]) for i in range(4)])
        shape = (shape / scale).ravel()
        norm = float(np.linalg.norm(shape))
        if norm > 0:
            return {"shape": (shape / norm).tolist(), "amplitude": amplitude.tolist(),
                    "duration": len(entries) / HZ, "peak_counter": entries[peak][0],
                    "peak_at": entries[peak][1], "end_counter": entries[-1][0], "end": entries[-1][1]}
        return None

    def push(self, row):
        value = np.asarray(row["samples"], dtype=float)
        if self.baseline is None:
            self.baseline = value.copy()
        # Both filters use only the current/past samples; no filtfilt or lookahead.
        self.baseline += (value - self.baseline) * (1 - math.exp(-1 / (HZ * 0.08)))
        self.filtered += (value - self.baseline - self.filtered) * (1 - math.exp(-2 * math.pi * 15 / HZ))
        score = float(np.max(np.abs(self.filtered) / self.noise))
        entry = (row["counter"], row["estimated_monotonic"], self.filtered.copy(), score)
        self.count += 1
        self.quiet = self.quiet + 1 if score < 2.5 else 0
        if self.inhibit and self.quiet >= 15:
            self.inhibit = False
        if self.active is None and self.count >= HZ // 2 and not self.inhibit and score >= 6:
            self._start(list(self.history), entry)  # Start once, then await a completed return toward baseline.
        if self.active is not None:
            self.active.append(entry)
            # A second same-polarity rise can begin before the first filter tail is quiet.
            # Require a deep valley, a substantial new rise and several confirming samples;
            # the opposite-polarity rebound of one ordinary blink is never a second pulse.
            projection = float(np.dot(self.filtered / self.noise, self.axis))
            current = len(self.active) - 1
            if self.valley_index is None and projection > self.peak_projection:
                self.peak_projection, self.peak_index = projection, current
            if (current - self.peak_index >= round(.06 * HZ)
                    and projection < .15 * self.peak_projection and projection < self.valley_projection):
                self.valley_projection, self.valley_index = projection, current
            rising = (self.split and self.valley_index is not None
                      and current - self.peak_index >= round(.12 * HZ)
                      and projection >= max(6.0, .45 * self.peak_projection))
            self.rise_count = self.rise_count + 1 if rising else 0
            if self.rise_count >= 3:
                boundary = self.valley_index
                first = self._finish(self.active[:boundary + 1])
                # Keep at most the normal pre-onset context for the second pulse;
                # a long intervening tail must not consume its own duration budget.
                remainder = self.active[max(boundary, current - 20):]
                self._start(remainder, entry)
                self.history.clear()
                if first is not None:
                    return first
            if len(self.active) > round(0.75 * HZ):
                self.active = None
                self.inhibit = True  # Sustained drift/muscle activity is not a blink.
            elif self.quiet >= 10:  # Forty milliseconds below release threshold completes a pulse causally.
                entries, self.active = self.active, None
                self.history.clear()
                return self._finish(entries)
        self.history.append(entry)
        return None


def _pulse_features(rows, noise, *, split=True):
    stream = _PulseStream(noise, split=split)
    pulses = []
    for row in rows:
        pulse = stream.push(row)
        if pulse is not None:
            pulses.append(pulse)
    return pulses


def _interval(first, second):
    return ((second["peak_counter"] - first["peak_counter"]) % 2**32) / HZ


def _pairs(pulses):
    """Nonoverlapping, bounded pairs; a single or a slow sequence never commands."""
    previous = None
    cooldown_end = -math.inf
    result = []
    for pulse in pulses:
        if pulse["end"] < cooldown_end:
            continue
        if previous is not None and PAIR_MIN <= _interval(previous, pulse) <= PAIR_MAX:
            result.append((previous, pulse))
            cooldown_end = pulse["end"] + REFRACTORY
            previous = None
        else:
            previous = pulse
    return result


def _aligned_similarity(shapes, templates):
    """Match within two waveform bins without flipping polarity or stretching time."""
    a = np.asarray(shapes, dtype=float).reshape(-1, SHAPE_BINS, 4)  # Restore each candidate's 24 time bins and four simultaneous channels.
    b = np.asarray(templates, dtype=float).reshape(-1, SHAPE_BINS, 4)  # Keep the identical channel order in every learned reference waveform.
    scores = a.reshape(len(a), -1) @ b.reshape(len(b), -1).T  # Unit-length features make each dot product an unshifted similarity score.
    for shift in (1, 2):  # These fixed offsets tolerate small onset differences; fitting cannot expand the allowed timing range.
        # Zero padding, rather than renormalizing a cropped overlap, penalizes any
        # discarded edge energy. A tiny matching fragment cannot earn a high score.
        later = a[:, shift:, :].reshape(len(a), -1) @ b[:, :-shift, :].reshape(len(b), -1).T  # Move all candidate channels together later in time.
        earlier = a[:, :-shift, :].reshape(len(a), -1) @ b[:, shift:, :].reshape(len(b), -1).T  # Also try the same bounded offset earlier in time.
        scores = np.maximum(scores, np.maximum(later, earlier))  # Retain the best allowed alignment for each candidate/reference pair.
    return scores  # Signed scores preserve polarity: an inverted artifact is not automatically a matching blink.


class BlinkDecoder:
    """Learn pulse shapes, then emit only fresh completed double-blink commands."""

    labels = LABELS
    feature_profile = FEATURE_PROFILE
    MIN_PER_CLASS = MIN_PER_CLASS

    def __init__(self):
        self.reset()

    def reset(self):
        self._version = VERSION  # A new calibration uses the current explicitly versioned classifier.
        self._training = []
        self._validation = []
        self._context = None
        self._calibration_context = None  # An explicit draft resume never rewrites the original acquisition.
        self._prediction_context = None
        self._model = None
        self.reset_stream()
        return self._result()

    def reset_stream(self):
        """Discard pending events, never calibration; first update only watermarks."""
        self._stream = None
        self._counter = None
        self._last_at = None
        self._first_pulse = None
        self._cooldown_until = -math.inf
        self._last_event = None

    def _result(self, error=None):
        result = {**self.status(), "ok": error is None}
        if error:
            result["error"] = str(error)
        return result

    def _expected(self):
        return self._prediction_context or self._calibration_context or self._context

    def resume_calibration(self, context):
        """Continue an unfrozen current draft once in a matching, explicitly chosen acquisition."""
        candidate = _context(context)
        if self._version < 2:
            raise ValueError("Explicitly upgrade the calibration draft before resuming a new acquisition.")
        if self._model is not None or self._validation or self._prediction_context is not None:
            raise ValueError("Only an untrained draft can resume calibration; frozen validation cannot be reset.")
        if self._context is None or not self._training:
            raise ValueError("Accepted training trials are required to resume a calibration draft.")
        if {k: v for k, v in candidate.items() if k != "session_id"} != {k: v for k, v in self._context.items() if k != "session_id"}:
            raise ValueError("Calibration channel layout or sample rate does not match the retained draft.")
        if self._calibration_context is not None and candidate != self._calibration_context:
            raise ValueError("This draft already resumed once; another acquisition cannot replace its provenance.")
        if candidate != self._context:
            self._calibration_context = candidate  # Keep both source identities; all old trials remain byte-for-byte unchanged.
        self.reset_stream()  # Recovery carries no pending gesture or inference authorization.
        return {"ok": True}

    def bind_prediction_context(self, context):
        candidate = _context(context)
        if self._model is None:
            raise ValueError("Train or load the double-blink model before binding a session.")
        if {k: v for k, v in candidate.items() if k != "session_id"} != {k: v for k, v in self._context.items() if k != "session_id"}:
            raise ValueError("Saved double-blink channel layout or sample rate does not match.")
        self._prediction_context = candidate
        self.reset_stream()  # A new acquisition cannot inherit either half of an old double blink.
        return {"ok": True}

    def add_trial(self, label, signal, phase="train", trial_id=None, start=None, end=None):
        """Retain derived pulse features/noise only, never the supplied EEG packets."""
        try:
            if self._prediction_context is not None:
                raise ValueError("Saved-model inference preserves calibration; reset only to start a separate calibration.")
            if label not in LABELS or phase not in ("train", "validate"):
                raise ValueError("A trial needs rest/select and train/validate.")
            if type(trial_id) not in (str, int) or not 1 <= len(str(trial_id)) <= 128:
                raise ValueError("A unique bounded trial ID is required.")
            trials = self._training + self._validation
            if str(trial_id) in {t["id"] for t in trials}:
                raise ValueError("This trial ID was already used.")
            if len(trials) >= 96:
                raise ValueError("Double-blink calibration is full.")
            if phase == "train" and len(self._training) >= 80:
                raise ValueError("Double-blink training is full at 80 accepted trials; saved progress is unchanged.")
            if phase == "train" and self._model is not None or phase == "validate" and self._model is None:
                raise ValueError("Train first, then collect separate held-out validation trials.")
            if phase == "validate" and sum(t["label"] == label for t in self._validation) >= MIN_PER_CLASS:
                raise ValueError("Held-out validation already has eight trials for this label.")
            context = _context(signal)
            if self._context is not None and context != (self._calibration_context or self._context):
                raise ValueError("Calibration acquisition or channel layout changed.")
            if not _finite(start) or not _finite(end) or not 4.75 <= end - start <= 5.25:
                raise ValueError("A complete five-second cue interval is required.")
            rows = [r for r in signal.get("samples", []) if isinstance(r, dict)
                    and _finite(r.get("estimated_monotonic")) and start <= r["estimated_monotonic"] < end]
            data = _rows(rows)
            if len(rows) < round(4.7 * HZ) or len(rows) > round(5.25 * HZ):
                raise ValueError("Insufficient contiguous samples for the five-second trial.")
            for t in trials:
                # Device counters and host monotonic clocks may both restart with a new acquisition.
                if t.get("session_id", self._context["session_id"]) != context["session_id"]:
                    continue
                if max(start, t["start"]) < min(end, t["end"]):
                    raise ValueError("Calibration trials must not overlap.")
                offset = (rows[0]["counter"] - t["counter_start"]) % 2**32
                reverse = (t["counter_start"] - rows[0]["counter"]) % 2**32
                if offset < t["n_samples"] or reverse < len(rows):
                    raise ValueError("Calibration packet windows must not overlap.")
            _quality(data)
            noise = _noise(data)
            pulses = _pulse_features(rows, self._model["noise"] if self._model is not None else noise,
                                     split=self._version >= 2)
            pairs = _pairs(pulses)
            if phase == "train" and label == "select" and len(pairs) != 1:
                raise ValueError("Cue needs one clear double blink: blink twice promptly, then keep eyes open.")
            trial = {"id": str(trial_id), "label": label, "start": start, "end": end,
                     "counter_start": rows[0]["counter"], "n_samples": len(rows), "noise": noise.tolist(), "pulses": pulses}
            if self._calibration_context is not None:
                trial["session_id"] = context["session_id"]  # Only newly accepted records need explicit resumed-session provenance.
            if phase == "train":
                self._training.append(trial)
            else:
                accepted = [p for p in pulses if self._score(p)[0]]
                commands = _pairs(accepted)
                # Preserve command-level mistakes; single ordinary blinks remain a correct no-command trial.
                trial["predicted"] = "select" if commands else "rest"
                trial["event_count"] = len(commands)
                self._validation.append(trial)
            if self._context is None:
                self._context = context
            return self._result()
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            return self._result(str(exc))

    def train(self):
        if self._model is not None:
            return self._result("The double-blink model is already frozen.")
        counts = Counter(t["label"] for t in self._training)
        if any(counts[label] < MIN_PER_CLASS for label in LABELS):
            return self._result("Collect eight no-command and eight deliberate-double-blink training trials first.")
        positive = [pulse for t in self._training if t["label"] == "select" for pair in _pairs(t["pulses"]) for pulse in pair]  # Fit only the two pulses belonging to each accepted command pair.
        trial_ids = [t["id"] for t in self._training if t["label"] == "select" for pair in _pairs(t["pulses"]) for _ in pair]  # Repeat the source ID for both pulses so they cannot validate one another.
        shapes = np.asarray([p["shape"] for p in positive])  # These saved normalized features suffice; fitting never needs the original EEG packets.
        template = shapes.mean(axis=0)
        norm = float(np.linalg.norm(template))
        if self._version == 1 and norm < 0.2:
            return self._result("Blink shapes are inconsistent; collect more consistent double-blink examples.")
        if self._version == 1:
            # Archived v1 parameters must still reproduce the original single-template fit exactly.
            template /= norm
            similarities = shapes @ template
            if np.quantile(similarities, 0.10) < 0.55:
                return self._result("Blink shapes are inconsistent across training trials.")
            threshold = max(0.60, float(np.quantile(similarities, 0.10)) - 0.12)
            classifier = {"template": template.tolist()}
        elif self._version == 2:
            # Preserve the original v2 all-trial gate and unshifted scoring when loading a frozen v2 model.
            # First pulses shortened by a following blink and fully completed second
            # pulses have different durations. Keep their learned prototypes rather
            # than averaging incompatible time alignments into an unusable template.
            similarities = shapes @ shapes.T
            for i, source in enumerate(trial_ids):
                similarities[i, [j for j, other in enumerate(trial_ids) if other == source]] = -1
            neighbors = similarities.max(axis=1)  # A pulse must resemble evidence from a DIFFERENT trial.
            if np.quantile(neighbors, 0.10) < 0.65:
                return self._result("Blink shapes do not repeat consistently across separate training trials.")
            threshold = max(0.70, float(np.quantile(neighbors, 0.10)) - 0.10)
            classifier = {"templates": shapes.tolist()}
        else:
            # An atypical first attempt must not veto a repeatable command pattern.
            # Retain all accepted evidence, but fit only whole trials whose BOTH
            # pulses resemble another trial, never just each other or themselves.
            similarities = _aligned_similarity(shapes, shapes)  # Compare every pulse with every reference using the same bounded rule as live prediction.
            sources = np.asarray(trial_ids)  # This parallel array identifies the complete command trial behind each matrix row/column.
            for i, source in enumerate(trial_ids):
                similarities[i, sources == source] = -1  # Remove self-matches and same-trial matches before measuring repeatability.
            supported = np.ones(len(positive), dtype=bool)  # Start by considering all accepted commands; exclusion affects fitting, never stored evidence.
            while np.any(supported):  # Stop safely if all candidates lose support, avoiding a maximum over an empty reference set.
                neighbors = similarities[:, supported].max(axis=1)  # Find each pulse's closest reference among the trials still under consideration.
                next_supported = np.asarray([all(neighbors[sources == source] >= 0.65) for source in trial_ids]) & supported  # Keep both pulses only when both pass the unchanged consistency floor.
                if np.array_equal(next_supported, supported):  # A stable set no longer depends on a trial that was just excluded.
                    break
                supported = next_supported  # Remove an entire inconsistent command, then verify the remaining mutual support again.
            source_count = len(set(trial_ids))  # Count command trials, not pulses; two pulses still represent one calibration attempt.
            supported_count = len(set(sources[supported]))  # Every supported trial contributes both of its pulses or neither.
            if supported_count < max(6, math.ceil(source_count * 0.75)):  # Require six of the first eight, and at least 75% if more examples are added.
                return self._result("Fewer than three quarters of the double-blink trials repeat consistently; add clearer examples without clearing accepted progress.")
            neighbors = similarities[np.ix_(supported, supported)].max(axis=1)  # Learn sensitivity only from mutually supported examples.
            threshold = max(0.70, float(np.quantile(neighbors, 0.10)) - 0.10)  # Allow the existing similarity margin while retaining the absolute detection floor.
            # Store indices, not copies or rewritten trial records, so the fit is
            # reproducible and every original trial remains available for review.
            training_indices = {trial["id"]: i for i, trial in enumerate(self._training)}  # Archive indices identify which unchanged records supplied the model.
            classifier = {"templates": shapes[supported].tolist(),
                          "template_trial_indices": [training_indices[source] for source in sources[supported]]}
            positive = [pulse for pulse, keep in zip(positive, supported) if keep]  # Excluded attempts also cannot widen the amplitude or duration envelope.
        amplitudes = np.asarray([np.linalg.norm(p["amplitude"]) for p in positive])  # Combine all four channel amplitudes without changing their voltage units.
        durations = np.asarray([p["duration"] for p in positive])  # Pulse duration remains separate from the allowed interval between the two blinks.
        self._model = {**classifier, "threshold": threshold,
                       "amplitude_min": float(amplitudes.min() * 0.35), "amplitude_max": float(amplitudes.max() * 2.5),
                       "duration_min": max(0.10, float(durations.min() * 0.55)), "duration_max": min(0.75, float(durations.max() * 1.6)),
                       "noise": np.median([t["noise"] for t in self._training if t["label"] == "rest"], axis=0).tolist()}
        # Negative trials can contain real single blinks. Reject command collisions rather than relabeling those pulses.
        collisions = sum(bool(_pairs([p for p in t["pulses"] if self._score(p)[0]])) for t in self._training if t["label"] == "rest")
        if collisions:
            self._model = None
            return self._result("No-command training contains a matching double-blink command; use a more distinct deliberate gesture.")
        self.reset_stream()
        return self._result()

    def _score(self, pulse):
        model = self._model
        # Dispatch by the stored model version, so a newer application cannot silently broaden an older model's matching rule.
        score = (float(_aligned_similarity([pulse["shape"]], model["templates"]).max()) if self._version >= 3
                 else float(np.max(np.asarray(model["templates"]) @ pulse["shape"])) if self._version == 2
                 else float(np.dot(pulse["shape"], model["template"])))
        amplitude = float(np.linalg.norm(pulse["amplitude"]))  # Compare the same four-channel amplitude measure used while fitting.
        accepted = (score >= model["threshold"] and model["amplitude_min"] <= amplitude <= model["amplitude_max"]
                    and model["duration_min"] <= pulse["duration"] <= model["duration_max"])
        return accepted, max(0.0, min(1.0, score))  # Bound the display score; only the separate accepted flag authorizes a pulse candidate.

    def _prediction(self, valid=False, reason="Waiting for a trained double-blink model.", **extra):
        return {"valid": valid, "reason": reason, "label": "rest" if valid else None,
                "confidence": 0.0, "event_id": None, "event_at": None,
                "ready": valid and self._stream is not None and self._stream.count >= HZ // 2,
                "first_blink_pending": self._first_pulse is not None, **extra}

    def update(self, signal):
        """Consume only new contiguous samples and return at most one fresh command."""
        if self._model is None:
            return self._prediction()
        try:
            context = _context(signal)
            if context != self._expected():
                raise ValueError("EEG acquisition or layout changed. Use the saved double-blink model again before arming.")
            rows = signal.get("samples")
            if not isinstance(rows, list) or not rows:
                raise ValueError("Waiting for current full-rate EEG samples.")
            # A reset/load/arm only establishes the newest watermark; it cannot replay a completed blink.
            if self._counter is None:
                data = _rows(rows[-HZ:])
                _quality(data)
                self._counter, self._last_at = rows[-1]["counter"], rows[-1]["estimated_monotonic"]
                self._stream = _PulseStream(self._model["noise"], split=self._version >= 2)
                return self._prediction(True, "Stream primed; keep eyes open briefly, then double blink.")
            fresh = [r for r in rows if isinstance(r, dict) and type(r.get("counter")) is int
                     and 0 < (r["counter"] - self._counter) % 2**32 < 2**31]
            if not fresh:
                return self._prediction(True, "Waiting for new EEG packets; cached data cannot trigger another blink.")
            if (fresh[0]["counter"] - self._counter) % 2**32 != 1:
                raise ValueError("EEG packet gap; double-blink candidate cancelled.")
            data = _rows(fresh)
            if len(fresh) > round(0.5 * HZ):
                raise ValueError("EEG backlog is too old for a live blink command; candidates cancelled.")
            # Inspect a short raw buffer before filtering, including samples immediately before this batch.
            _quality(_rows(rows[-HZ:]))
            if fresh[0]["estimated_monotonic"] - self._last_at > 0.5:
                raise ValueError("EEG timing paused; double-blink candidate cancelled.")
            result = None
            for row in fresh:
                self._counter, self._last_at = row["counter"], row["estimated_monotonic"]
                pulse = self._stream.push(row)
                if self._first_pulse is not None and (row["counter"] - self._first_pulse["peak_counter"]) % 2**32 / HZ > PAIR_MAX + 0.75:
                    self._first_pulse = None
                if pulse is None:
                    continue
                accepted, score = self._score(pulse)
                if not accepted:
                    self._first_pulse = None  # A conflicting waveform breaks, rather than bridges, a possible pair.
                    continue
                if pulse["end"] < self._cooldown_until:
                    continue
                if self._first_pulse is not None and PAIR_MIN <= _interval(self._first_pulse, pulse) <= PAIR_MAX:
                    first = self._first_pulse
                    self._first_pulse = None
                    self._cooldown_until = pulse["end"] + REFRACTORY
                    event_id = f"{context['session_id']}:{first['peak_counter']}:{pulse['end_counter']}"
                    self._last_event = event_id
                    result = self._prediction(True, "Completed learned double blink.", label="select", confidence=score,
                                              event_id=event_id, event_at=pulse["end"], ready=True, first_blink_pending=False)
                else:
                    self._first_pulse = pulse
            # Deliver only a command completed near the newest sample, never old commands from a delayed backlog.
            if result is not None and rows[-1]["estimated_monotonic"] - result["event_at"] <= 0.25:
                return result
            return self._prediction(True, "One blink recognized; blink again promptly." if self._first_pulse else "Ready for a deliberate double blink.")
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            self.reset_stream()  # Raw faults discard filter history and both halves of the gesture.
            return self._prediction(False, str(exc))

    def status(self):
        counts = {phase: {label: sum(t["label"] == label for t in trials) for label in LABELS}
                  for phase, trials in (("train", self._training), ("validate", self._validation))}
        matrix = {label: {p: 0 for p in LABELS} for label in LABELS}
        for trial in self._validation:
            matrix[trial["label"]][trial["predicted"]] += 1
        complete = all(counts["validate"][label] >= MIN_PER_CLASS for label in LABELS)
        recalls = [matrix[label][label] / counts["validate"][label] if counts["validate"][label] else 0 for label in LABELS]
        false_rate = matrix["rest"]["select"] / counts["validate"]["rest"] if counts["validate"]["rest"] else None
        passed = complete and sum(recalls) / 2 >= 0.80 and false_rate <= 0.10 and all(t["event_count"] == 1 for t in self._validation if t["label"] == "select")
        per_class = {label: {"support": counts["validate"][label],
                             "recall": recalls[i],
                             "precision": matrix[label][label] / max(1, sum(matrix[actual][label] for actual in LABELS))}
                     for i, label in enumerate(LABELS)}
        state = ("validated" if passed else "validation_failed" if complete else "validating" if self._model is not None else "collecting")
        return {"state": state, "labels": list(LABELS), "feature_profile": FEATURE_PROFILE,
                "detector_version": self._version, "trained": self._model is not None,
                "total": len(self._training) + len(self._validation),
                "training_counts": counts["train"], "train_counts": dict(counts["train"]),
                "validation_counts": counts["validate"], "min_per_class": MIN_PER_CLASS,
                "can_train": self._model is None and all(counts["train"][l] >= MIN_PER_CLASS for l in LABELS),
                "passed_validation": bool(passed), "arm_eligible": bool(passed and self._model is not None),
                "inference_context_bound": self._prediction_context is not None,
                "context": copy.deepcopy(self._context),
                "calibration_context": copy.deepcopy(self._calibration_context),
                "validation_scope": "saved_acquisition" if self._prediction_context is not None else "current_acquisition",
                "evaluation": {"complete": complete, "passed": bool(passed), "confusion": matrix, "per_class": per_class,
                               "balanced_accuracy": sum(recalls) / 2 if self._validation else None,
                               "rest_false_activation_rate": false_rate},
                "training_fit": ({"supported_select_trials": len(set(self._model["template_trial_indices"])),
                                  "total_select_trials": counts["train"]["select"],
                                  "excluded_select_trials": counts["train"]["select"] - len(set(self._model["template_trial_indices"]))}
                                 if self._version >= 3 and self._model is not None else None),
                "feature_processing": "Causal baseline removal and 15 Hz lowpass; learned normalized pulse waveform, amplitude/duration envelope and bounded double-blink timing.",
                "disclaimer": "Personalized ocular-artifact command detection; scores are waveform similarity, not probabilities. Validation is limited to recorded trials."}

    def export_state(self):
        if self._model is None:
            raise ValueError("Only a trained double-blink model can be saved.")
        state = {"format": FORMAT, "version": self._version, "feature_profile": FEATURE_PROFILE,  # Save the model's actual fitting version, not the running application's default.
                 "context": self._context, "training": self._training,
                 "validation": self._validation, "model": self._model}
        if self._calibration_context is not None:
            state["calibration_context"] = self._calibration_context
        return copy.deepcopy(state)  # Callers cannot mutate the live calibration through an exported object.

    def export_draft(self):
        """Back up accepted, unfrozen trials without claiming that a model exists."""
        if self._model is not None or self._validation or not self._training or self._context is None:
            raise ValueError("A calibration draft requires accepted training trials and no frozen model.")
        state = {"format": DRAFT_FORMAT, "version": self._version,
                 "feature_profile": FEATURE_PROFILE, "context": self._context,
                 "training": self._training, "validation": [], "model": None}
        if self._calibration_context is not None:
            state["calibration_context"] = self._calibration_context
        return copy.deepcopy(state)

    @classmethod
    def from_calibration_state(cls, state, *, upgrade=False):
        """Validate a draft; explicitly choose current fitting without rewriting evidence."""
        if type(upgrade) is not bool or not isinstance(state, dict) or state.get("format") != DRAFT_FORMAT:
            raise ValueError("A recognized double-blink calibration draft is required.")
        if state.get("model") is not None or state.get("validation") != [] or not state.get("training"):
            raise ValueError("An untrained draft cannot carry model parameters or held-out validation.")
        archive = {**state, "format": FORMAT}  # Reuse the same strict inert feature/window validator.
        decoder = cls.from_state(archive, _allow_draft=True)  # Validate the entire draft before changing its future fitting version.
        if upgrade:
            decoder._version = VERSION  # Explicit caller choice; every accepted feature and its provenance remains intact.
        return decoder

    @classmethod
    def from_state(cls, state, *, _allow_draft=False):
        """Restore bounded inert values and verify parameters against retained features."""
        def plain(value, depth=0):
            if depth > 12:
                raise ValueError("Saved blink state is nested too deeply.")
            if value is None or type(value) in (str, bool, int):
                if isinstance(value, str) and len(value) > 256:
                    raise ValueError("Saved blink text is too long.")
                return
            if type(value) is float and math.isfinite(value):
                return
            if isinstance(value, list) and len(value) <= 2500:
                for item in value:
                    plain(item, depth + 1)
                return
            if isinstance(value, dict) and len(value) <= 32 and all(type(k) is str for k in value):
                for item in value.values():
                    plain(item, depth + 1)
                return
            raise ValueError("Saved blink state contains unsupported or nonfinite values.")

        plain(state)
        required = {"format", "version", "feature_profile", "context", "training", "validation", "model"}  # Reject missing or unexpected fields instead of guessing how a future archive should behave.
        if (not isinstance(state, dict) or not required <= set(state) or set(state) - required - {"calibration_context"}
                or state["format"] != FORMAT or type(state["version"]) is not int or state["version"] not in (1, 2, VERSION)
                or state["feature_profile"] != FEATURE_PROFILE
                or state["version"] == 1 and "calibration_context" in state):
            raise ValueError("Unsupported double-blink archive.")
        decoder = cls()  # Build a separate object; import failure must never partially replace the active calibration.
        decoder._version = state["version"]  # Never silently reinterpret an archived classifier as the new algorithm.
        decoder._context = _context(state["context"])
        sessions = {decoder._context["session_id"]}  # Older records implicitly belong to this original, immutable acquisition identity.
        if "calibration_context" in state:
            resumed = _context(state["calibration_context"])
            if (resumed["session_id"] in sessions
                    or {k: v for k, v in resumed.items() if k != "session_id"}
                    != {k: v for k, v in decoder._context.items() if k != "session_id"}):
                raise ValueError("Invalid resumed calibration acquisition or channel layout.")
            decoder._calibration_context = resumed  # A resumed draft keeps its source acquisition as well as the explicitly selected new one.
            sessions.add(resumed["session_id"])
        ids, intervals, counter_intervals = set(), [], []
        for name in ("training", "validation"):
            trials = state[name]
            if not isinstance(trials, list) or len(trials) > (80 if name == "training" else 16):
                raise ValueError("Invalid saved blink trial count.")
            for trial in trials:
                required = {"id", "label", "start", "end", "counter_start", "n_samples", "noise", "pulses"}
                if name == "validation":
                    required |= {"predicted", "event_count"}
                if (not isinstance(trial, dict) or not required <= set(trial) or set(trial) - required - {"session_id"}
                        or trial["label"] not in LABELS or decoder._version == 1 and "session_id" in trial):
                    raise ValueError("Invalid saved blink trial.")
                session_id = trial.get("session_id", decoder._context["session_id"])
                if not isinstance(session_id, str) or session_id not in sessions:
                    raise ValueError("Saved blink trial has an unknown acquisition identity.")
                if not isinstance(trial["id"], str) or not 1 <= len(trial["id"]) <= 128 or trial["id"] in ids:
                    raise ValueError("Invalid or duplicate saved blink trial ID.")
                ids.add(trial["id"])
                if (not _finite(trial["start"]) or not _finite(trial["end"]) or not 4.75 <= trial["end"] - trial["start"] <= 5.25
                        or type(trial["counter_start"]) is not int or not 0 <= trial["counter_start"] < 2**32
                        or type(trial["n_samples"]) is not int or not round(4.7 * HZ) <= trial["n_samples"] <= round(5.25 * HZ)
                        or any(session_id == session and max(trial["start"], a) < min(trial["end"], b)
                               for session, a, b in intervals)):
                    raise ValueError("Invalid or overlapping saved blink window.")
                intervals.append((session_id, trial["start"], trial["end"]))
                for session, counter_start, sample_count in counter_intervals:
                    if session == session_id and ((trial["counter_start"] - counter_start) % 2**32 < sample_count
                            or (counter_start - trial["counter_start"]) % 2**32 < trial["n_samples"]):
                        raise ValueError("Saved blink packet windows overlap.")
                counter_intervals.append((session_id, trial["counter_start"], trial["n_samples"]))
                if not isinstance(trial["noise"], list) or len(trial["noise"]) != 4 or not all(_finite(v) and 0 < v < 0.010 for v in trial["noise"]):
                    raise ValueError("Invalid saved blink noise scale.")
                if not isinstance(trial["pulses"], list) or len(trial["pulses"]) > 32:
                    raise ValueError("Invalid saved blink pulse count.")
                for pulse in trial["pulses"]:
                    if not isinstance(pulse, dict) or set(pulse) != {"shape", "amplitude", "duration", "peak_counter", "peak_at", "end_counter", "end"}:
                        raise ValueError("Invalid saved blink pulse.")
                    if (not isinstance(pulse["shape"], list) or len(pulse["shape"]) != SHAPE_BINS * 4
                            or not all(_finite(v) and abs(v) <= 1 for v in pulse["shape"])
                            or not math.isclose(float(np.linalg.norm(pulse["shape"])), 1, abs_tol=1e-8)
                            or not isinstance(pulse["amplitude"], list) or len(pulse["amplitude"]) != 4
                            or not all(_finite(v) and 0 <= v <= 0.020 for v in pulse["amplitude"])
                            or not _finite(pulse["duration"]) or not 0.1 <= pulse["duration"] <= 0.75
                            or any(type(pulse[k]) is not int or not 0 <= pulse[k] < 2**32 for k in ("peak_counter", "end_counter"))
                            or any((pulse[k] - trial["counter_start"]) % 2**32 >= trial["n_samples"] for k in ("peak_counter", "end_counter"))
                            or not _finite(pulse["peak_at"]) or not _finite(pulse["end"])
                            or not trial["start"] <= pulse["peak_at"] <= pulse["end"] < trial["end"]):
                        raise ValueError("Invalid saved blink waveform features.")
                if name == "training" and trial["label"] == "select" and len(_pairs(trial["pulses"])) != 1:
                    raise ValueError("Saved positive trial does not contain one double blink.")
                if name == "validation" and (trial["predicted"] not in LABELS or type(trial["event_count"]) is not int or not 0 <= trial["event_count"] <= 16):
                    raise ValueError("Invalid saved blink validation result.")
            setattr(decoder, "_" + name, copy.deepcopy(trials))
        model = state["model"]
        if model is None and _allow_draft and decoder._training and not decoder._validation:
            decoder.reset_stream()  # Progress recovery cannot restore an armed gesture or replay an earlier blink.
            return decoder
        saved_validation, decoder._validation = decoder._validation, []  # Hold validation aside; fitting must never learn from held-out outcomes.
        result = decoder.train()  # Refit under the archived version and verify that the retained training evidence reproduces every parameter.
        if not result["ok"] or not isinstance(model, dict) or set(model) != set(decoder._model):
            raise ValueError("Saved blink model cannot be reproduced from training features.")
        def numeric_matches(value, expected):
            # Validate the original JSON types before comparison: coercing "0.8"
            # to a float would accept an archive that later crashes live scoring.
            if isinstance(expected, list):
                return (isinstance(value, list) and len(value) == len(expected)
                        and all(numeric_matches(item, reference) for item, reference in zip(value, expected)))
            return _finite(value) and math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)  # Booleans, strings and nonfinite numbers are not model parameters.

        matches = True
        for key, reference in decoder._model.items():
            value = model[key]
            if key == "template_trial_indices":
                # These are record identities, not approximate measurements. Even
                # a tiny numeric change could invent another supported trial.
                matches = isinstance(value, list) and all(type(index) is int for index in value) and value == reference
            else:
                matches = numeric_matches(value, reference)  # Preserve tolerances only for finite numeric waveform/model values.
            if not matches:
                break
        if not matches:
            raise ValueError("Saved blink parameters do not match the retained training features.")
        decoder._model = copy.deepcopy(model)  # After verification, retain the exact saved values instead of substituting newly calculated approximations.
        for trial in saved_validation:
            commands = _pairs([p for p in trial["pulses"] if decoder._score(p)[0]])
            if trial["event_count"] != len(commands) or trial["predicted"] != ("select" if commands else "rest"):
                raise ValueError("Saved blink validation does not match the frozen model.")
        if any(sum(t["label"] == label for t in saved_validation) > MIN_PER_CLASS for label in LABELS):
            raise ValueError("Saved blink validation exceeds its fixed quota.")
        decoder._validation = saved_validation  # Restore the original outcomes only after independently checking them against the frozen model.
        decoder.reset_stream()  # Archives cannot carry live event history, acquisition permission or pending clicks.
        return decoder
