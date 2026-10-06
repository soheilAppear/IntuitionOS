"""Ephemeral experimental EEG-only features and a frozen three-class decoder.

This module has no camera, actuator, device, file, or network access. Movement
labels can correlate with muscle/electrode artifacts; this is not thought
reading or a medical measurement. Engineering eligibility is not a reliability
guarantee. Callers still own freshness, debounce, arming, and emergency stop.

Each feature vector describes exactly two seconds of contiguous packet counters.
Per channel: remove its least-squares straight line, compute log population
variance, and compute log integrated one-sided Hann-periodogram power in
[4, 8), [8, 13), and [13, 30) Hz. Values are volts; logarithms are natural.
The FFT uses numpy.fft.rfft with its default unscaled forward transform.
"""

from __future__ import annotations

import copy
import math
from collections import Counter

import numpy as np


LABELS = ("left", "right", "rest")
WINDOW_SECONDS = 2.0
MIN_PER_CLASS = 8
MAX_TRIALS = 120
SCORE_THRESHOLD = 0.80
MARGIN_THRESHOLD = 0.25
DISCLAIMER = (
    "Experimental movement/artifact classification, not thought reading or a medical result. "
    "Scores are uncalibrated model scores, not probabilities. Engineering validation gates "
    "do not guarantee reliable control. Model and validation apply only to this acquisition "
    "session and channel layout; reset and recalibrate after restarting acquisition."
)
FEATURE_DESCRIPTION = (
    "Two-second contiguous full-rate EEG window; per-channel linear detrend, natural log "
    "variance and Hann-periodogram integrated power in [4,8), [8,13), [13,30) Hz. "
    "Training-only standardization, diagonal LDA with 20% variance shrinkage, equal "
    "class priors, and training-only distance rejection. No camera input."
)


def _number(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


def _reject(reason):
    return {"ok": False, "reason": reason, "error": reason}


def extract_features(rows, channels, nominal_hz, session_context=None):
    """Return features from the newest complete two-second EEG packet window.

    ``rows`` are SDK packet dictionaries with counter, samples (volts), and
    estimated_monotonic. Channel ``num`` identifies the samples index. The
    approximate host-derived timing is checked loosely; contiguous counters,
    not packet-batch timestamp spacing, define spectral sample intervals.
    Rejection thresholds below are conservative engineering checks, not device
    ADC specifications or diagnoses of contact/signal validity.
    """
    if not _number(nominal_hz) or not 64 <= nominal_hz <= 1000:
        return _reject("Unsupported or missing nominal EEG sample rate.")
    hz = float(nominal_hz)
    size = round(WINDOW_SECONDS * hz)
    if not isinstance(rows, (list, tuple)) or len(rows) < size:
        return _reject("Need a complete two-second full-rate EEG window.")
    if not isinstance(channels, (list, tuple)) or not 1 <= len(channels) <= 16:
        return _reject("Missing or unsupported EEG channel layout.")
    signature = []
    for channel in channels:
        if not isinstance(channel, dict):
            return _reject("Invalid EEG channel identity.")
        num, name = channel.get("num"), channel.get("name")
        if type(num) is not int or not isinstance(name, str) or not name or len(name) > 64:
            return _reject("Invalid EEG channel identity.")
        signature.append({"num": num, "name": name})
    signature.sort(key=lambda channel: channel["num"])
    if [channel["num"] for channel in signature] != list(range(len(signature))):
        return _reject("EEG channel indices must uniquely cover the sample vector.")
    context = session_context or {}
    if not isinstance(context, dict):
        return _reject("Invalid acquisition session context.")
    session_id = context.get("session_id")
    if session_id is not None and (not isinstance(session_id, str) or not session_id or len(session_id) > 128):
        return _reject("Invalid acquisition session identity.")
    selected = rows[-size:]
    counters, timestamps, values = [], [], []
    for row in selected:
        if not isinstance(row, dict):
            return _reject("Invalid EEG packet.")
        counter, at, samples = row.get("counter"), row.get("estimated_monotonic"), row.get("samples")
        if type(counter) is not int or not 0 <= counter < 2**32 or not _number(at):
            return _reject("Missing EEG packet counter or approximate timestamp.")
        if not isinstance(samples, (list, tuple)) or len(samples) != len(signature) or not all(_number(value) for value in samples):
            return _reject("Nonfinite EEG values or changed channel count.")
        if counters and (counter - counters[-1]) % 2**32 != 1:
            return _reject("EEG packet gap, duplicate, or counter reset inside the window.")
        counters.append(counter)
        timestamps.append(float(at))
        values.append(samples)
    expected_span = (size - 1) / hz
    span = timestamps[-1] - timestamps[0]
    if not expected_span * 0.75 <= span <= expected_span * 1.25:
        return _reject("Approximate EEG timing does not match the nominal full sample rate.")
    # Small batch-timing reversals are expected from host receipts. A large
    # reversal is inconsistent with this bounded two-second sample window.
    if any(after - before < -0.25 for before, after in zip(timestamps, timestamps[1:])):
        return _reject("Approximate EEG timing moved backwards excessively.")
    data = np.asarray(values, dtype=np.float64)
    if np.any(np.abs(data) > 0.010) or np.any(np.ptp(data, axis=0) > 0.010):
        return _reject("EEG amplitude exceeds the conservative 10 mV engineering limit.")
    if np.any(np.ptp(data, axis=0) < 1e-9):
        return _reject("A channel is flat or below the numerical variation limit.")
    # Sustained exact extremum plateaus are suspicious for clipping. This is
    # not an assertion that these values equal the device's hardware rails.
    plateau = max(4, round(hz * 0.020))
    flat_run_limit = max(8, round(hz * 0.100))
    for column in data.T:
        extrema = (column == column.min()) | (column == column.max())
        run = 0
        constant_run = 0
        previous = None
        for value, extreme in zip(column, extrema):
            run = run + 1 if extreme and value == previous else int(extreme)
            constant_run = constant_run + 1 if value == previous else 1
            previous = value
            if constant_run >= flat_run_limit:
                return _reject("A channel contains a sustained exactly constant segment (possible flatline).")
            if run >= plateau:
                return _reject("A channel has a repeated extreme plateau (possible clipping).")
    t = np.arange(size, dtype=np.float64)
    t -= t.mean()
    centered = data - data.mean(axis=0)
    detrended = centered - t[:, None] * (t @ centered / (t @ t))[None, :]
    variance = np.mean(detrended**2, axis=0)
    if np.any(variance < 1e-18):
        return _reject("A channel is flat after linear detrending.")
    taper = np.hanning(size)
    spectrum = np.fft.rfft(detrended * taper[:, None], axis=0)
    density = np.abs(spectrum)**2 / (hz * np.sum(taper**2))
    if size % 2 == 0:
        density[1:-1] *= 2
    else:
        density[1:] *= 2
    frequencies = np.fft.rfftfreq(size, 1.0 / hz)
    band_power = [np.sum(density[(frequencies >= low) & (frequencies < high)], axis=0) * hz / size
                  for low, high in ((4, 8), (8, 13), (13, 30))]
    features = np.log(np.maximum(np.column_stack([variance, *band_power]), 1e-24)).ravel()
    window = {"start": timestamps[0], "end": timestamps[-1] + 1 / hz,
              "counter_start": counters[0], "counter_end": counters[-1], "n_samples": size,
              "window_id": f"{counters[0]}:{counters[-1]}:{size}"}
    return {"ok": True, "features": features.tolist(),
            "context": {"session_id": session_id, "channels": signature, "nominal_hz": hz},
            "window": window,
            "diagnostics": {"n_samples": size, "duration_seconds": size / hz,
                            "approximate_span_seconds": span,
                            "peak_to_peak_v": np.ptp(data, axis=0).tolist(),
                            "rms_v": np.sqrt(variance).tolist()}}


class EegDecoder:
    """Bounded in-memory trials; explicit train freezes all model parameters.

    Validation consists of exactly the first eight accepted new trials per
    class after freezing. It cannot be extended until a favorable score appears.
    Reusing IDs, overlapping packet windows, or changing acquisition identity
    is rejected. Holdout features never update scaling, centroids, rejection
    radii, thresholds, or the model. Callers serialize calls with their lock.
    """

    MIN_PER_CLASS = MIN_PER_CLASS
    MAX_TRIALS = MAX_TRIALS
    SCORE_THRESHOLD = SCORE_THRESHOLD
    MARGIN_THRESHOLD = MARGIN_THRESHOLD

    def __init__(self):
        self.reset()

    def reset(self):
        self._training = []
        self._validation = []
        self._trial_ids = set()
        self._windows = []
        self._last_end = -math.inf
        self._context = None
        self._model = None
        return self._result()

    def _result(self, error=None):
        result = self.status()
        result["ok"] = not bool(error)
        if error:
            result["error"] = str(error)
        return result

    @staticmethod
    def _counts(trials):
        counts = Counter(trial["label"] for trial in trials)
        return {label: counts[label] for label in LABELS}

    def _check(self, extracted):
        if not isinstance(extracted, dict) or extracted.get("ok") is not True:
            return None, (extracted.get("reason") if isinstance(extracted, dict) else None) or "Rejected EEG window."
        features, context, window = extracted.get("features"), extracted.get("context"), extracted.get("window")
        if not isinstance(context, dict) or not context.get("session_id"):
            return None, "A current acquisition session identity is required."
        if self._context is not None and context != self._context:
            return None, "Acquisition session, channels, or sample rate changed. Reset and recalibrate."
        channels = context.get("channels")
        if (not isinstance(channels, list) or not 1 <= len(channels) <= 16
                or not _number(context.get("nominal_hz"))
                or not isinstance(features, (list, tuple)) or len(features) != len(channels) * 4
                or not all(_number(value) and abs(value) <= 100 for value in features)):
            return None, "Invalid EEG feature vector or context."
        if (not isinstance(window, dict) or not _number(window.get("start")) or not _number(window.get("end"))
                or window["end"] <= window["start"]
                or any(type(window.get(key)) is not int for key in ("counter_start", "counter_end", "n_samples"))
                or not 0 <= window["counter_start"] < 2**32 or not 0 <= window["counter_end"] < 2**32
                or window["n_samples"] != round(2 * context["nominal_hz"])
                or (window["counter_end"] - window["counter_start"]) % 2**32 != window["n_samples"] - 1):
            return None, "Invalid contiguous EEG window identity."
        return np.asarray(features, dtype=np.float64), None

    def add_trial(self, label, features, phase="train", trial_id=None, session_context=None):
        """Add one extracted complete trial, retaining no raw samples.

        ``features`` is the entire result of :func:`extract_features`, including
        its window/context metadata; bare vectors are deliberately insufficient.
        ``session_context`` optionally provides a second explicit session check.
        """
        if label not in LABELS or phase not in ("train", "validate"):
            return self._result("Trial needs a left, right, or rest label and train/validate phase.")
        if not isinstance(trial_id, (str, int)) or isinstance(trial_id, bool) or str(trial_id) == "":
            return self._result("A unique trial identity is required.")
        if str(trial_id) in self._trial_ids:
            return self._result("Trial identity was already used.")
        if len(self._training) + len(self._validation) >= MAX_TRIALS:
            return self._result("Calibration is full. Reset to start again.")
        if phase == "train" and self._model is not None:
            return self._result("The model is frozen. Reset before adding training trials.")
        if phase == "validate" and self._model is None:
            return self._result("Train and freeze the model before collecting new validation trials.")
        if phase == "validate" and self._counts(self._validation)[label] >= MIN_PER_CLASS:
            return self._result("This class already has its eight fixed validation trials. Reset to run a new experiment.")
        if phase == "train" and len(self._training) >= MAX_TRIALS - len(LABELS) * MIN_PER_CLASS:
            return self._result("Training storage is full; remaining capacity is reserved for validation.")
        vector, error = self._check(features)
        if error:
            return self._result(error)
        if session_context is not None and (not isinstance(session_context, dict)
                or session_context.get("session_id") != features["context"].get("session_id")):
            return self._result("Trial session does not match the current acquisition.")
        window = features["window"]
        if window["start"] < self._last_end - 1e-9:
            return self._result("Trial windows must be chronological and must not overlap.")
        start, size = window["counter_start"], window["n_samples"]
        if any((start - old_start) % 2**32 < old_size or (old_start - start) % 2**32 < size
               for old_start, old_size in self._windows):
            return self._result("EEG sample counters overlap a previous training or validation trial.")
        if self._context is None:
            self._context = copy.deepcopy(features["context"])
        record = {"label": label, "id": str(trial_id)}
        if phase == "train":
            record["features"] = vector.copy()
            self._training.append(record)
        else:
            record["prediction"] = self.predict(features)
            self._validation.append(record)
        self._trial_ids.add(str(trial_id))
        self._windows.append((start, size))
        self._last_end = window["end"]
        return self._result()

    def train(self):
        if self._model is not None:
            return self._result("Model is already frozen. Reset before retraining.")
        if min(self._counts(self._training).values()) < MIN_PER_CLASS:
            return self._result("Need at least eight training trials each for left, right, and rest.")
        matrix = np.stack([trial["features"] for trial in self._training])
        means = matrix.mean(axis=0)
        # A floor in log-feature units prevents constant features from causing
        # explosive scaling; it is fixed before looking at validation trials.
        scales = np.maximum(matrix.std(axis=0), 0.1)
        standardized = (matrix - means) / scales
        centroids = np.stack([standardized[[trial["label"] == label for trial in self._training]].mean(axis=0)
                              for label in LABELS])
        residuals = np.stack([row - centroids[LABELS.index(trial["label"])]
                              for row, trial in zip(standardized, self._training)])
        pooled = np.sum(residuals**2, axis=0) / max(1, len(matrix) - len(LABELS))
        diagonal = np.maximum(0.8 * pooled + 0.2 * pooled.mean(), 0.05)
        radii = []
        for index, label in enumerate(LABELS):
            rows = standardized[[trial["label"] == label for trial in self._training]]
            distances = np.sqrt(np.mean((rows - centroids[index])**2 / diagonal, axis=1))
            radii.append(max(3.0, float(distances.max()) * 1.5))
        self._model = {"means": means, "scales": scales, "centroids": centroids,
                       "diagonal": diagonal, "radii": np.asarray(radii)}
        return self._result()

    def predict(self, features):
        result = {"label": None, "confidence": 0.0, "margin": 0.0,
                  "scores": {label: 0.0 for label in LABELS}, "valid": False,
                  "ood": False, "reason": "Train and freeze a model first.", "window_id": None,
                  "score_note": "Uncalibrated normalized model scores, not probabilities."}
        if isinstance(features, dict):
            window = features.get("window")
            result["window_id"] = window.get("window_id") if isinstance(window, dict) else None
        if self._model is None:
            return result
        vector, error = self._check(features)
        if error:
            result["reason"] = error
            return result
        model = self._model
        standardized = (vector - model["means"]) / model["scales"]
        squared = np.sum((standardized[None, :] - model["centroids"])**2 / model["diagonal"], axis=1)
        logits = -0.5 * squared
        weights = np.exp(logits - logits.max())
        scores = weights / weights.sum()
        order = np.argsort(scores)
        winner = int(order[-1])
        result.update(confidence=float(scores[winner]), margin=float(scores[order[-1]] - scores[order[-2]]),
                      scores={label: float(scores[index]) for index, label in enumerate(LABELS)})
        distance = math.sqrt(float(squared[winner]) / len(vector))
        if distance > model["radii"][winner]:
            result.update(ood=True, reason="EEG features lie outside the training-distance limit.")
            return result
        result["valid"] = True
        if result["confidence"] < SCORE_THRESHOLD or result["margin"] < MARGIN_THRESHOLD:
            result["reason"] = "Uncertain: score or score margin is below the fixed threshold."
            return result
        result.update(label=LABELS[winner], reason="Accepted EEG-only prediction; controls require separate arming.")
        return result

    def _evaluation(self):
        counts = self._counts(self._validation)
        matrix = {label: {predicted: 0 for predicted in (*LABELS, "uncertain")} for label in LABELS}
        for trial in self._validation:
            matrix[trial["label"]][trial["prediction"]["label"] or "uncertain"] += 1
        per_class = {}
        for label in LABELS:
            predicted = sum(matrix[actual][label] for actual in LABELS)
            per_class[label] = {"support": counts[label], "predicted": predicted,
                                "precision": matrix[label][label] / predicted if predicted else 0.0,
                                "recall": matrix[label][label] / counts[label] if counts[label] else 0.0}
        total = len(self._validation)
        complete = min(counts.values()) >= MIN_PER_CLASS
        balanced = sum(per_class[label]["recall"] for label in LABELS) / len(LABELS) if total else None
        rest_activations = matrix["rest"]["left"] + matrix["rest"]["right"]
        rest_rate = rest_activations / counts["rest"] if counts["rest"] else None
        gates = {"enough_trials": complete,
                 "balanced_accuracy": balanced is not None and balanced >= 0.75,
                 "each_recall": all(per_class[label]["recall"] >= 0.70 for label in LABELS),
                 "each_precision": all(per_class[label]["precision"] >= 0.70 for label in LABELS),
                 "rest_false_activation_rate": rest_rate is not None and rest_rate <= 0.10}
        passed = complete and all(gates.values())
        return {"state": "passed" if passed else "failed" if complete else "incomplete",
                "passed": passed, "complete": complete, "n_train": len(self._training), "n_test": total,
                "accuracy": sum(matrix[label][label] for label in LABELS) / total if total else None,
                "balanced_accuracy": balanced, "baseline_accuracy": max(counts.values()) / total if total else None,
                "per_class": per_class, "confusion_matrix": matrix,
                "rest_false_activations": rest_activations, "rest_trials": counts["rest"],
                "rest_false_activation_rate": rest_rate,
                "rest_false_activation_unit": "Accepted left/right predictions per held-out rest trial, before action debounce; not activations per hour.",
                "abstentions": sum(matrix[label]["uncertain"] for label in LABELS),
                "gates": gates, "thresholds": {"balanced_accuracy": 0.75, "each_recall": 0.70,
                                                  "each_precision": 0.70, "rest_false_activation_rate": 0.10,
                                                  "score": SCORE_THRESHOLD, "margin": MARGIN_THRESHOLD},
                "reason": ("Engineering gates passed on these held-out trials; reliability is not guaranteed." if passed
                           else "Engineering gates failed. Controls remain unavailable; reset for a new experiment." if complete
                           else "Freeze training, then collect eight new validation trials for each class."),
                "split": "Separate complete trials collected after freezing; no shared counters or overlapping windows. Validation cannot refit or extend a completed class.",
                "disclaimer": DISCLAIMER}

    def status(self):
        evaluation = self._evaluation()
        trained = self._model is not None
        state = ("validated" if evaluation["passed"] else "validation_failed" if evaluation["complete"]
                 else "collecting_validation" if trained else "collecting_training")
        return {"state": state, "trained": trained, "arm_eligible": trained and evaluation["passed"],
                "passed_validation": evaluation["passed"], "train_counts": self._counts(self._training),
                "validation_counts": self._counts(self._validation), "min_per_class": MIN_PER_CLASS,
                "max_trials": MAX_TRIALS, "total": len(self._training) + len(self._validation),
                "can_train": not trained and min(self._counts(self._training).values()) >= MIN_PER_CLASS,
                "context": {"bound_to_acquisition": self._context is not None,
                            "nominal_hz": self._context.get("nominal_hz") if self._context else None,
                            "channels": copy.deepcopy(self._context.get("channels", [])) if self._context else []},
                "evaluation": evaluation, "feature_processing": FEATURE_DESCRIPTION,
                "score_threshold": SCORE_THRESHOLD, "margin_threshold": MARGIN_THRESHOLD,
                "disclaimer": DISCLAIMER}
