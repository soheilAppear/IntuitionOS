"""Ephemeral experimental EEG-only features and a frozen two/three-class decoder.

This module has no camera, actuator, device, file, or network access. Movement
labels can correlate with muscle/electrode artifacts; this is not thought
reading or a medical measurement. Engineering eligibility is not a reliability
guarantee. Callers still own freshness, debounce, arming, and emergency stop.

Each feature vector describes exactly two seconds of contiguous packet counters.
Per channel: remove its least-squares straight line, compute log population
variance, and compute log integrated one-sided Hann-periodogram power in
[4, 8), [8, 13), and [13, 30) Hz. Values are volts; logarithms are natural.
The opt-in click profile replaces variance with integrated [4, 30) Hz power;
it retains the original raw integrity checks and unfiltered RMS diagnostics.
The FFT uses numpy.fft.rfft with its default unscaled forward transform.
"""

from __future__ import annotations

import copy
import math
from collections import Counter

import numpy as np


LABELS = ("left", "right", "rest")
# Keep the existing swipe labels as the default; selection is an explicit opt-in.
ALLOWED_LABELS = (*LABELS, "select")
# Version the click transform so incompatible training and prediction cannot mix.
CLICK_FEATURE_PROFILE = "click_4_30hz_v1"
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
# Explain the click transform separately without relabeling the legacy features.
CLICK_FEATURE_DESCRIPTION = (
    "Two-second contiguous full-rate EEG window; per-channel linear detrend, natural log "
    "Hann-periodogram integrated power in [4,30), [4,8), [8,13), [13,30) Hz. "
    "Raw amplitude, flatline, clipping, and packet checks run before band selection. "
    "Training-only standardization, diagonal LDA with 20% variance shrinkage, equal "
    "class priors, and training-only distance rejection. No camera input. "
    "Restricting feature power does not certify artifact-free EEG."
)


def _number(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


def _reject(reason):
    return {"ok": False, "reason": reason, "error": reason}


def extract_features(rows, channels, nominal_hz, session_context=None, feature_profile=None):
    """Return features from the newest complete two-second EEG packet window.

    ``rows`` are SDK packet dictionaries with counter, samples (volts), and
    estimated_monotonic. Channel ``num`` identifies the samples index. The
    approximate host-derived timing is checked loosely; contiguous counters,
    not packet-batch timestamp spacing, define spectral sample intervals.
    Rejection thresholds below are conservative engineering checks, not device
    ADC specifications or diagnoses of contact/signal validity.

    ``feature_profile=None`` preserves the existing variance-based features and
    context. ``click_4_30hz_v1`` uses integrated 4--30 Hz power as the first
    feature, keeping 60 Hz power out of that feature without hiding raw faults.
    """
    # Reject unknown transforms instead of silently training on a different signal.
    if feature_profile is not None and (not isinstance(feature_profile, str) or feature_profile != CLICK_FEATURE_PROFILE):
        # Return the normal extraction error shape, leaving callers fail-closed.
        return _reject("Unsupported EEG feature profile.")
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
    # A constant baseline is removed by the existing feature preprocessing.
    # Check raw excursions before detrending so spikes and large drift cannot
    # disappear into that preprocessing. This is not a device ADC-rail limit.
    peak_to_peak = np.ptp(data, axis=0)
    excessive = np.flatnonzero(peak_to_peak > 0.010)
    if len(excessive):
        summary = ", ".join(f"{signature[i]['name']} [{signature[i]['num']}]: {peak_to_peak[i] * 1000:.2f} mV"
                            for i in excessive)
        return _reject("EEG peak-to-peak amplitude exceeds the conservative 10 mV engineering limit: " + summary + ".")
    flat = np.flatnonzero(peak_to_peak < 1e-9)
    if len(flat):
        names = ", ".join(f"{signature[i]['name']} [{signature[i]['num']}]" for i in flat)
        return _reject("EEG channels are flat or below the numerical variation limit: " + names + ".")
    # Sustained exact extremum plateaus are suspicious for clipping. This is
    # not an assertion that these values equal the device's hardware rails.
    plateau = max(4, round(hz * 0.020))
    flat_run_limit = max(8, round(hz * 0.100))
    for channel_index, column in enumerate(data.T):
        extrema = (column == column.min()) | (column == column.max())
        run = 0
        constant_run = 0
        previous = None
        for value, extreme in zip(column, extrema):
            run = run + 1 if extreme and value == previous else int(extreme)
            constant_run = constant_run + 1 if value == previous else 1
            previous = value
            if constant_run >= flat_run_limit:
                return _reject(f"EEG channel {signature[channel_index]['name']} [{signature[channel_index]['num']}] contains a sustained exactly constant segment (possible flatline).")
            if run >= plateau:
                return _reject(f"EEG channel {signature[channel_index]['name']} [{signature[channel_index]['num']}] has a repeated extreme plateau (possible clipping).")
    t = np.arange(size, dtype=np.float64)
    t -= t.mean()
    centered = data - data.mean(axis=0)
    detrended = centered - t[:, None] * (t @ centered / (t @ t))[None, :]
    variance = np.mean(detrended**2, axis=0)
    flat = np.flatnonzero(variance < 1e-18)
    if len(flat):
        names = ", ".join(f"{signature[i]['name']} [{signature[i]['num']}]" for i in flat)
        return _reject("EEG channels are flat after linear detrending: " + names + ".")
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
    # The three adjacent bands exactly partition [4,30), excluding the 60 Hz bin.
    feature_power = np.sum(band_power, axis=0) if feature_profile == CLICK_FEATURE_PROFILE else variance
    # Preserve feature order and width so the same bounded classifier handles both profiles.
    features = np.log(np.maximum(np.column_stack([feature_power, *band_power]), 1e-24)).ravel()
    window = {"start": timestamps[0], "end": timestamps[-1] + 1 / hz,
              "counter_start": counters[0], "counter_end": counters[-1], "n_samples": size,
              "window_id": f"{counters[0]}:{counters[-1]}:{size}"}
    # Leave the legacy context byte-for-byte compatible when no profile is requested.
    feature_context = {"session_id": session_id, "channels": signature, "nominal_hz": hz}
    # Tag only the opt-in transform so its model cannot accept legacy feature vectors.
    if feature_profile is not None:
        # Bind the transform version together with session, channels, and sample rate.
        feature_context["feature_profile"] = feature_profile
    return {"ok": True, "features": features.tolist(),
            "context": feature_context,
            "window": window,
            "diagnostics": {"n_samples": size, "duration_seconds": size / hz,
                            "approximate_span_seconds": span,
                            "peak_to_peak_v": peak_to_peak.tolist(),
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

    def __init__(self, labels=LABELS, feature_profile=None):
        # Reject strings/generators: a finite explicit label sequence is required.
        if not isinstance(labels, (list, tuple)):
            # Configuration errors must surface before any calibration can be retained.
            raise ValueError("EEG labels must be a list or tuple of two or three allowed labels including rest.")
        # Check element types before constructing a set, avoiding unhashable input failures.
        if (not 2 <= len(labels) <= 3 or any(not isinstance(label, str) or label not in ALLOWED_LABELS for label in labels)
                or len(set(labels)) != len(labels) or "rest" not in labels):
            # Every model needs a distinct neutral class and at least one valid action class.
            raise ValueError("EEG labels must be unique allowed labels (left, right, rest, select), with two or three including rest.")
        # Feature transforms are fixed before training, never inferred from trial data.
        if feature_profile is not None and (not isinstance(feature_profile, str) or feature_profile != CLICK_FEATURE_PROFILE):
            # A typo must not silently fall back to the old full-spectrum variance feature.
            raise ValueError("Unsupported EEG feature profile.")
        # Copy caller-owned lists into a tuple so external mutation cannot relabel a model.
        self._labels = tuple(labels)
        # A read-only property exposes this binding while reset preserves it.
        self._feature_profile = feature_profile
        self.reset()

    @property
    def labels(self):
        """Return the immutable class order used by scores and model centroids."""
        # Returning the tuple does not expose mutable calibration configuration.
        return self._labels

    @property
    def feature_profile(self):
        """Return the transform this decoder requires for every accepted window."""
        # Assignment has no setter, preventing profile changes on a frozen model.
        return self._feature_profile

    def reset(self):
        self._training = []
        self._validation = []
        self._trial_ids = set()
        self._windows = []
        self._last_end = -math.inf
        self._context = None
        self._prediction_context = None  # Explicit saved-model use may bind inference without rewriting trial provenance.
        self._model = None
        return self._result()

    def _result(self, error=None):
        result = self.status()
        result["ok"] = not bool(error)
        if error:
            result["error"] = str(error)
        return result

    def _counts(self, trials):
        counts = Counter(trial["label"] for trial in trials)
        # Include absent configured classes so incomplete binary training cannot pass.
        return {label: counts[label] for label in self.labels}

    def bind_prediction_context(self, context):
        """Explicitly reuse a frozen model in a compatible acquisition for inference.

        Original training, validation and their session identity stay unchanged.
        This never grants input authorization or marks historical validation passed.
        """
        if self._model is None or self._context is None:
            raise ValueError("Train or load a frozen model before selecting an inference session.")
        if (not isinstance(context, dict) or not isinstance(context.get("session_id"), str)
                or not 1 <= len(context["session_id"]) <= 128):
            raise ValueError("A current acquisition identity is required for saved-model inference.")
        candidate = copy.deepcopy(context)  # Caller mutations must not silently change the accepted session.
        original_layout = {key: value for key, value in self._context.items() if key != "session_id"}
        candidate_layout = {key: value for key, value in candidate.items() if key != "session_id"}
        if candidate_layout != original_layout:
            raise ValueError("Saved model channel layout, sample rate and feature profile must match the current EEG stream.")
        self._prediction_context = candidate  # Only predict() consults this override; saved evidence is immutable.
        return {"ok": True}

    def _check(self, extracted, *, prediction=False):
        if not isinstance(extracted, dict) or extracted.get("ok") is not True:
            return None, (extracted.get("reason") if isinstance(extracted, dict) else None) or "Rejected EEG window."
        features, context, window = extracted.get("features"), extracted.get("context"), extracted.get("window")
        if not isinstance(context, dict) or not context.get("session_id"):
            return None, "A current acquisition session identity is required."
        # Enforce the configured transform even before the first trial binds a session.
        if context.get("feature_profile") != self.feature_profile:
            # Missing, changed, or unknown click profiles must never reuse a model.
            return None, "EEG feature profile does not match this decoder. Reset and recalibrate with matching processing."
        expected_context = self._prediction_context if prediction and self._prediction_context is not None else self._context
        if expected_context is not None and context != expected_context:
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
        if self._prediction_context is not None:
            return self._result("Saved-model inference preserves the original trials. Reset explicitly before starting a new calibration.")
        # Accept only this model's classes; a click model cannot consume old swipe trials.
        if label not in self.labels or phase not in ("train", "validate"):
            # State the actual configured labels rather than advertising inactive classes.
            return self._result("Trial needs one of " + ", ".join(self.labels) + " and a train/validate phase.")
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
        # Reserve eight held-out trials for every configured class, including binary models.
        if phase == "train" and len(self._training) >= MAX_TRIALS - len(self.labels) * MIN_PER_CLASS:
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
            # Keep the eight-per-class requirement while explaining the active classes.
            return self._result("Need at least eight training trials each for " + ", ".join(self.labels) + ".")
        matrix = np.stack([trial["features"] for trial in self._training])
        means = matrix.mean(axis=0)
        # A floor in log-feature units prevents constant features from causing
        # explosive scaling; it is fixed before looking at validation trials.
        scales = np.maximum(matrix.std(axis=0), 0.1)
        standardized = (matrix - means) / scales
        # Class order is immutable and shared by centroids, score names, and predictions.
        centroids = np.stack([standardized[[trial["label"] == label for trial in self._training]].mean(axis=0)
                              for label in self.labels])
        # Measure each training vector relative to its configured class centroid.
        residuals = np.stack([row - centroids[self.labels.index(trial["label"])]
                              for row, trial in zip(standardized, self._training)])
        # Preserve the pooled-variance degrees of freedom for either two or three classes.
        pooled = np.sum(residuals**2, axis=0) / max(1, len(matrix) - len(self.labels))
        diagonal = np.maximum(0.8 * pooled + 0.2 * pooled.mean(), 0.05)
        radii = []
        # Fit each class's OOD radius using training data only, as before.
        for index, label in enumerate(self.labels):
            rows = standardized[[trial["label"] == label for trial in self._training]]
            distances = np.sqrt(np.mean((rows - centroids[index])**2 / diagonal, axis=1))
            radii.append(max(3.0, float(distances.max()) * 1.5))
        self._model = {"means": means, "scales": scales, "centroids": centroids,
                       "diagonal": diagonal, "radii": np.asarray(radii)}
        return self._result()

    def predict(self, features):
        # Score keys expose only configured classes; binary models never invent directions.
        result = {"label": None, "confidence": 0.0, "margin": 0.0,
                  "scores": {label: 0.0 for label in self.labels}, "valid": False,
                  "ood": False, "reason": "Train and freeze a model first.", "window_id": None,
                  "score_note": "Uncalibrated normalized model scores, not probabilities."}
        if isinstance(features, dict):
            window = features.get("window")
            result["window_id"] = window.get("window_id") if isinstance(window, dict) else None
        if self._model is None:
            return result
        vector, error = self._check(features, prediction=True)  # Explicit saved-model use can target a compatible new stream.
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
        # Name scores with the same immutable order used to construct the classifier.
        result.update(confidence=float(scores[winner]), margin=float(scores[order[-1]] - scores[order[-2]]),
                      scores={label: float(scores[index]) for index, label in enumerate(self.labels)})
        distance = math.sqrt(float(squared[winner]) / len(vector))
        if distance > model["radii"][winner]:
            result.update(ood=True, reason="EEG features lie outside the training-distance limit.")
            return result
        result["valid"] = True
        if result["confidence"] < SCORE_THRESHOLD or result["margin"] < MARGIN_THRESHOLD:
            result["reason"] = "Uncertain: score or score margin is below the fixed threshold."
            return result
        # Prediction is still only a label; separate guarded code owns every desktop action.
        result.update(label=self.labels[winner], reason="Accepted EEG-only prediction; controls require separate arming.")
        return result

    def _evaluation(self):
        counts = self._counts(self._validation)
        # Include abstentions alongside the active classes in the held-out confusion matrix.
        matrix = {label: {predicted: 0 for predicted in (*self.labels, "uncertain")} for label in self.labels}
        for trial in self._validation:
            matrix[trial["label"]][trial["prediction"]["label"] or "uncertain"] += 1
        per_class = {}
        # Evaluate every configured class independently; binary mode keeps identical gates.
        for label in self.labels:
            # Precision counts predictions from all real classes, not only the matching row.
            predicted = sum(matrix[actual][label] for actual in self.labels)
            per_class[label] = {"support": counts[label], "predicted": predicted,
                                "precision": matrix[label][label] / predicted if predicted else 0.0,
                                "recall": matrix[label][label] / counts[label] if counts[label] else 0.0}
        total = len(self._validation)
        complete = min(counts.values()) >= MIN_PER_CLASS
        # Average recall over this model's classes rather than dividing binary results by three.
        balanced = sum(per_class[label]["recall"] for label in self.labels) / len(self.labels) if total else None
        # Any non-rest prediction during rest is a false action, including a select prediction.
        rest_activations = sum(matrix["rest"][label] for label in self.labels if label != "rest")
        rest_rate = rest_activations / counts["rest"] if counts["rest"] else None
        gates = {"enough_trials": complete,
                 "balanced_accuracy": balanced is not None and balanced >= 0.75,
                 # Do not relax the original recall/precision requirements for click mode.
                 "each_recall": all(per_class[label]["recall"] >= 0.70 for label in self.labels),
                 "each_precision": all(per_class[label]["precision"] >= 0.70 for label in self.labels),
                 "rest_false_activation_rate": rest_rate is not None and rest_rate <= 0.10}
        passed = complete and all(gates.values())
        return {"state": "passed" if passed else "failed" if complete else "incomplete",
                "passed": passed, "complete": complete, "n_train": len(self._training), "n_test": total,
                # Correct predictions are the active confusion-matrix diagonal only.
                "accuracy": sum(matrix[label][label] for label in self.labels) / total if total else None,
                "balanced_accuracy": balanced, "baseline_accuracy": max(counts.values()) / total if total else None,
                "per_class": per_class, "confusion_matrix": matrix,
                "rest_false_activations": rest_activations, "rest_trials": counts["rest"],
                "rest_false_activation_rate": rest_rate,
                # State the active action labels and retain the distinction from a time-based rate.
                "rest_false_activation_unit": "Accepted " + "/".join(label for label in self.labels if label != "rest") + " predictions per held-out rest trial, before action debounce; not activations per hour.",
                # Abstentions count as misses for every configured class, including select.
                "abstentions": sum(matrix[label]["uncertain"] for label in self.labels),
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
                # Return a new list so UI consumers cannot mutate the model's class order.
                "labels": list(self.labels), "feature_profile": self.feature_profile,
                "passed_validation": evaluation["passed"], "train_counts": self._counts(self._training),
                "validation_counts": self._counts(self._validation), "min_per_class": MIN_PER_CLASS,
                "max_trials": MAX_TRIALS, "total": len(self._training) + len(self._validation),
                "can_train": not trained and min(self._counts(self._training).values()) >= MIN_PER_CLASS,
                "context": {"bound_to_acquisition": self._context is not None,
                            "nominal_hz": self._context.get("nominal_hz") if self._context else None,
                            "channels": copy.deepcopy(self._context.get("channels", [])) if self._context else []},
                "inference_context_bound": self._prediction_context is not None,
                "validation_scope": "saved_acquisition" if self._prediction_context is not None and self._prediction_context != self._context else "current_acquisition",
                # Describe the fixed transform that training, validation, and inference share.
                "evaluation": evaluation, "feature_processing": CLICK_FEATURE_DESCRIPTION if self.feature_profile == CLICK_FEATURE_PROFILE else FEATURE_DESCRIPTION,
                "score_threshold": SCORE_THRESHOLD, "margin_threshold": MARGIN_THRESHOLD,
                "disclaimer": DISCLAIMER}
