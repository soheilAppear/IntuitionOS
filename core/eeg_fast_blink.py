"""A one-second causal endpoint CNN with inert archives and NumPy inference.

Torch is imported only by train(). One-second trials keep the supplied waveform
and label together; no handcrafted blink gate discards positive examples.
Held-out results describe whole-trial classification, not continuous click rates.
"""

from __future__ import annotations

import copy
import math
import time
from collections import deque

import numpy as np


FORMAT = "intuitionos.eeg-fast-cnn-blink-model"  # Stable archive identifier; existing saved models keep loading.
VERSION = 1  # Increment only for an incompatible archive schema change.
FEATURE_PROFILE = "cnn_blink_fast_endpoint_v1"  # Bind preprocessing and weights to the same one-second representation.
ARCHITECTURE = "tiny_causal_endpoint_cnn_v1"  # Reject weights from a different convolutional architecture.
LABELS = ("rest", "select")  # No command versus one complete deliberate double-blink command.
MIN_PER_CLASS = 8  # Require independent examples of both classes in each phase.
HZ = 250  # BrainBit full-rate sampling contract.
WINDOW_SAMPLES = 250  # Exactly one second of full-rate EEG per trial and live window.
INPUT_SAMPLES = 125  # Pair averaging halves the network input rate without changing duration.
MAX_TRIALS = 64  # Reserve 16 held-out trials alongside at most 48 training trials.
SEED = 1729  # Reproduce augmentation and training without changing validation.
EPOCHS = 140  # Fixed training budget; held-out scores never tune this value.
HOP_SAMPLES = 10  # Score the newest one-second context every 40 ms.
SCORE_DWELL_SAMPLES = 20  # Require 80 ms of confident rest and command evidence.
WEIGHT_SHAPES = {
    "conv1.weight": (8, 4, 9), "conv1.bias": (8,),
    "conv2.weight": (12, 8, 9), "conv2.bias": (12,),
    "conv3.weight": (16, 12, 9), "conv3.bias": (16,),
    "conv4.weight": (16, 16, 9), "conv4.bias": (16,),
    "output.weight": (2, 16), "output.bias": (2,),
}
PARAMETER_COUNT = sum(math.prod(shape) for shape in WEIGHT_SHAPES.values())  # Report the actual number of numeric network parameters.


class _InputFault(ValueError):
    """Typed live-input failures; callers must not infer recovery from prose."""

    def __init__(self, message, code, category, **details):
        super().__init__(message)  # Keep a human explanation separate from machine-readable fault codes.
        self.code, self.category, self.details = code, category, details  # Recovery decisions use these explicit typed fields.

    def prediction_fields(self):
        return {"fault_code": self.code, "fault_category": self.category,
                "recoverable": False,
                "fault_details": self.details}


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)  # Reject booleans, array scalars, NaN, and infinity.
    except OverflowError:
        return False


def _context(signal):
    if (not isinstance(signal, dict) or not isinstance(signal.get("session_id"), str)
            or not 1 <= len(signal["session_id"]) <= 128 or signal.get("nominal_hz") != HZ
            or signal.get("units") != "V"):
        raise _InputFault("CNN blink input requires a current 250 Hz EEG session with units V.",
                          "context_invalid", "context")
    channels = signal.get("channels")
    if (not isinstance(channels, list) or len(channels) != 4
            or any(not isinstance(c, dict) or type(c.get("num")) is not int or c["num"] != i
                   or not isinstance(c.get("name"), str) or not 1 <= len(c["name"]) <= 64
                   for i, c in enumerate(channels))):
        raise _InputFault("Four EEG channels in sample order 0, 1, 2, 3 are required.",
                          "context_invalid", "context")
    if signal.get("feature_profile", FEATURE_PROFILE) != FEATURE_PROFILE:
        raise _InputFault("CNN waveform feature profile does not match.", "context_invalid", "context")
    return {"session_id": signal["session_id"], "nominal_hz": HZ, "units": "V",
            "channels": [{"num": c["num"], "name": c["name"]} for c in channels]}


def _rows(rows):
    if not isinstance(rows, (list, tuple)) or not 1 <= len(rows) <= WINDOW_SAMPLES:
        raise _InputFault("A bounded full-rate EEG packet buffer is required.", "buffer_invalid", "transport")
    previous = None
    for row in rows:
        if (not isinstance(row, dict) or type(row.get("counter")) is not int
                or not 0 <= row["counter"] < 2 ** 32 or not _finite(row.get("estimated_monotonic"))
                or not isinstance(row.get("samples"), (list, tuple)) or len(row["samples"]) != 4
                or not all(_finite(v) and abs(v) <= 1e100 for v in row["samples"])
                or "host_received_monotonic" in row and not _finite(row["host_received_monotonic"])):
            raise _InputFault("Malformed or nonfinite EEG packet.", "packet_invalid", "transport")
        if previous is not None and (row["counter"] - previous) % 2 ** 32 != 1:
            raise _InputFault("EEG packet gap, duplicate, or counter reset; CNN history cleared.",
                              "packet_discontinuity", "transport")
        previous = row["counter"]  # Track continuity modulo the SDK counter width.
    return np.asarray([row["samples"] for row in rows], dtype=np.float64)  # Validate packets before creating a numeric matrix.


def _quality(data, channels):
    """Reject unusable samples; waveform magnitude alone is not a CNN gate."""
    if len(data) < 125:
        raise _InputFault("Waiting for at least half a second of current EEG.", "buffer_short", "transport")
    for index, column in enumerate(data.T):
        name = channels[index]["name"]
        # Positive gain is handled by the existing per-channel normalization.
        # Keep the lower bound: a flat trace contains no useful waveform shape.
        excursion = float(np.ptp(column))
        if excursion < 1e-9:
            raise _InputFault(f"{name} is flat; check signal contact.", "signal_flatline", "signal_quality", channel=name)
        run = 1
        low, high = np.min(column), np.max(column)
        for before, after in zip(column[:-1], column[1:]):
            run = run + 1 if before == after else 1
            if run >= 25:
                raise _InputFault(f"{name} has 100 ms of identical samples.", "signal_identical", "signal_quality", channel=name)
            if run >= 5 and after in (low, high):
                raise _InputFault(f"{name} has a repeated extreme plateau (possible clipping).",
                                  "signal_plateau", "signal_quality", channel=name)


def preprocess_waveform(data):
    """Preserve waveform shape while removing per-window DC and linear drift."""
    values = np.asarray(data, dtype=np.float64)
    if values.shape != (WINDOW_SAMPLES, 4) or not np.all(np.isfinite(values)):
        raise ValueError("CNN preprocessing needs exactly 250 finite four-channel samples.")
    # Pair averaging reduces 250 Hz data to 125 Hz without picking every other
    # noisy sample. Input remains a waveform, not bandpower or peak summaries.
    values = values.reshape(INPUT_SAMPLES, 2, 4).mean(axis=1)  # Average adjacent samples separately for all four channels.
    t = np.linspace(-1.0, 1.0, INPUT_SAMPLES)  # Use centered time coordinates for the drift estimate.
    values = values - values.mean(axis=0)  # Remove each channel baseline independently.
    values -= t[:, None] * (t @ values / (t @ t))[None, :]  # Subtract the least-squares linear trend.
    values -= np.median(values, axis=0)  # Center remaining waveform shape robustly.
    scale = np.maximum(np.median(np.abs(values), axis=0) / .67448975, 1e-6)  # A scale floor prevents amplifying almost-flat noise.
    return np.clip(values / scale, -12, 12).T.astype(np.float32)  # Produce bounded channel-first input in the frozen model format.


def _conv(values, weight, bias, *, dilation=1):
    extent = (weight.shape[-1] - 1) * dilation + 1  # Dilated kernels span a wider history without future samples.
    # Left padding only: the endpoint never depends on invented future EEG.
    padded = np.pad(values, ((0, 0), (0, 0), (extent - 1, 0)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, extent, axis=-1)[..., ::dilation]  # Expose each causal receptive field as a view.
    return np.maximum(0, np.einsum("bctk,ock->bot", windows, weight, optimize=True) + bias[None, :, None])


def _logits(waveforms, weights):
    values = np.asarray(waveforms, dtype=np.float32)
    if values.ndim == 2:
        values = values[None, :, :]  # Accept one waveform through the same batched implementation.
    if values.ndim != 3 or values.shape[1:] != (4, INPUT_SAMPLES):
        raise ValueError("CNN input shape must be [batch, 4, 125].")
    for name, dilation in (("conv1", 1), ("conv2", 2), ("conv3", 4), ("conv4", 8)):
        values = _conv(values, weights[name + ".weight"], weights[name + ".bias"], dilation=dilation)
    # The newest endpoint sees 121/125 samples (968 ms), with no global pooling
    # that could keep classifying an earlier region after the gesture has left.
    return values[:, :, -1] @ weights["output.weight"].T + weights["output.bias"]


def _scores(waveforms, weights):
    logits = _logits(waveforms, weights).astype(np.float64)
    logits -= logits.max(axis=1, keepdims=True)  # Stabilize the exponentials without changing the softmax result.
    probabilities = np.exp(logits)  # Scores are relative network outputs, not calibrated probabilities.
    return probabilities[:, 1] / probabilities.sum(axis=1)  # Return only the double-blink class score.


def _torch_network(torch):
    """Construct only after an explicit Train request; never import on startup."""
    class TinyCnn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv1d(4, 8, 9)
            self.conv2 = torch.nn.Conv1d(8, 12, 9, dilation=2)
            self.conv3 = torch.nn.Conv1d(12, 16, 9, dilation=4)
            self.conv4 = torch.nn.Conv1d(16, 16, 9, dilation=8)
            self.output = torch.nn.Linear(16, 2)

        def forward(self, values):
            for layer in (self.conv1, self.conv2, self.conv3, self.conv4):
                values = torch.relu(layer(torch.nn.functional.pad(values, (8 * layer.dilation[0], 0))))
            return self.output(values[:, :, -1])
    return TinyCnn()


def _augmentation(waveform, rng, *, command):
    if command:
        # Preserve the whole positive gesture. Cropping could turn its label
        # into a single blink, so command examples only compress slightly.
        length = int(rng.integers(112, INPUT_SAMPLES + 1))
        offset = int(rng.integers(0, INPUT_SAMPLES - length + 1))
        shortened = np.stack([np.interp(np.linspace(0, INPUT_SAMPLES - 1, length), np.arange(INPUT_SAMPLES), channel)
                              for channel in waveform])
        result = np.pad(shortened, ((0, 0), (offset, INPUT_SAMPLES - length - offset)), mode="edge")
    else:
        # A no-command recording remains a control when seen partially at the
        # newest/oldest edge. Cover those live phases explicitly. Do not compress
        # it: that could move two intentionally separated blinks closer together.
        offset = int(rng.integers(-INPUT_SAMPLES + 1, INPUT_SAMPLES))
        result = np.pad(waveform, ((0, 0), (INPUT_SAMPLES, INPUT_SAMPLES)), mode="edge")
        result = result[:, INPUT_SAMPLES - offset:2 * INPUT_SAMPLES - offset].copy()
        t = np.linspace(-1.0, 1.0, INPUT_SAMPLES)  # Use centered time coordinates for the drift estimate.
        result -= result.mean(axis=1, keepdims=True)
        result -= (result @ t / (t @ t))[:, None] * t
        result -= np.median(result, axis=1, keepdims=True)
        result /= np.maximum(np.median(np.abs(result), axis=1, keepdims=True) / .67448975, .05)
    result *= rng.uniform(.85, 1.15, (4, 1))
    result += rng.normal(0, .025, result.shape)
    return np.clip(result, -12, 12).astype(np.float32)


def _threshold(scores, labels):
    negatives = scores[np.asarray(labels) == 0]  # Use only training no-command examples to choose the threshold.
    positives = scores[np.asarray(labels) == 1]  # Validation cannot influence command or release thresholds.
    threshold = max(.65, float(np.max(negatives)) + .05)  # Require margin above every recorded training control.
    if threshold > .99 or np.mean(positives >= threshold) < .75:
        raise ValueError("CNN training did not separate the recorded controls from double blinks. Add clearer, varied examples before freezing.")
    return threshold, max(.25, min(.60, threshold - .20))


class FastCnnBlinkDecoder:
    labels = LABELS
    feature_profile = FEATURE_PROFILE
    MIN_PER_CLASS = MIN_PER_CLASS
    window_seconds = trial_seconds = 1.0
    cooldown_seconds = 1.0

    def __init__(self, *, clock=time.monotonic):
        self.clock = clock
        self.reset()

    def reset(self):
        self._training, self._validation = [], []  # Reset explicitly discards both phases of this in-memory experiment.
        self._context = self._prediction_context = None  # Historical calibration and current inference binding are separate.
        self._model = self._weights = None  # Clear both inert archive data and runtime numeric arrays.
        self.reset_stream()
        return self._result()

    def reset_stream(self):
        self._counter = self._last_at = self._last_receipt = None  # Discard the prior stream boundary after a lifecycle change.
        self._live_rows = deque(maxlen=WINDOW_SAMPLES)  # Keep at most one second of inference evidence.
        self._last_predict_counter = None  # The next complete window starts a new scoring schedule.
        self._rest_count = self._select_count = 0
        self._rest_since_counter = self._select_since_counter = None
        self._rest_ready = False
        self._last_event_counter = None
        self._event_watermark = None
        self._cancel_pending_watermark = False
        self._last_score = None
        self._ready = False

    def cancel_pending(self):
        """Retain rolling context, but exclude all evidence present at cancellation.

        Used once on hand input ownership entry/release, never on a successful CNN
        event. A whole-window classifier cannot locate a physical blink inside
        old data, so score transitions cannot reauthorize that buffered gesture.
        Lifecycle changes and stream faults still require reset_stream().
        """
        self._rest_count = self._select_count = 0
        self._rest_since_counter = self._select_since_counter = None
        self._rest_ready = False
        self._event_watermark = self._counter  # Suppress any waveform that includes pre-cancellation evidence.
        # Service cancellation can occur before this tick's full-rate update.
        # Anchor to that update's newest validated packet, not the previous
        # poll's counter, so a newly delivered old gesture cannot slip through.
        self._cancel_pending_watermark = True

    def _cancelled_evidence_remaining(self, counter):
        if self._cancel_pending_watermark:
            return WINDOW_SAMPLES
        if self._event_watermark is None or counter is None:
            return 0
        return max(0, WINDOW_SAMPLES - (counter - self._event_watermark) % 2 ** 32)

    def _result(self, error=None):
        return {**self.status(), "ok": error is None, **({"error": str(error)} if error else {})}

    def _expected(self):
        return self._prediction_context or self._context

    def bind_prediction_context(self, context):
        candidate = _context(context)
        if self._model is None or self._context is None:
            raise ValueError("Train or load a CNN model before binding its inference session.")
        if {k: v for k, v in candidate.items() if k != "session_id"} != {k: v for k, v in self._context.items() if k != "session_id"}:
            raise ValueError("Saved CNN channel layout, units, or sample rate does not match.")
        self._prediction_context = candidate  # Bind only a compatible new acquisition; retain training provenance.
        self.reset_stream()
        return {"ok": True}

    def add_trial(self, label, signal, phase="train", trial_id=None, start=None, end=None):
        try:
            if self._prediction_context is not None:
                raise ValueError("Saved-model inference preserves its original calibration; reset to collect a new model.")
            if label not in LABELS or phase not in ("train", "validate"):
                raise ValueError("CNN trials require rest/select and train/validate.")
            if type(trial_id) not in (str, int) or not 1 <= len(str(trial_id)) <= 128:
                raise ValueError("A bounded unique trial ID is required.")
            trials = self._training + self._validation
            if str(trial_id) in {trial["id"] for trial in trials}:
                raise ValueError("This trial ID has already been recorded.")
            if len(trials) >= MAX_TRIALS or phase == "train" and len(self._training) >= MAX_TRIALS - 16:
                raise ValueError("CNN calibration capacity is full; retained data is unchanged.")
            if phase == "train" and self._model is not None or phase == "validate" and self._model is None:
                raise ValueError("Collect training first, freeze the CNN, then collect separate held-out validation.")
            if phase == "validate" and sum(t["label"] == label for t in self._validation) >= MIN_PER_CLASS:
                raise ValueError("Eight held-out trials for this class have already been recorded.")
            context = _context(signal)
            if self._context is not None and context != self._context:
                raise ValueError("EEG acquisition or channel layout changed during calibration.")
            if not _finite(start) or not _finite(end) or abs(end - start - 1) > .02:
                raise ValueError("A complete one-second cue is required.")
            supplied = signal.get("samples")
            if not isinstance(supplied, list) or len(supplied) > WINDOW_SAMPLES:
                raise ValueError("Supply exactly the bounded one-second trial packet buffer.")
            rows = [row for row in supplied if isinstance(row, dict) and _finite(row.get("estimated_monotonic"))
                    and start - 1e-8 <= row["estimated_monotonic"] < end - 1e-8]
            data = _rows(rows)
            if len(rows) != WINDOW_SAMPLES:
                raise ValueError("CNN trial needs all 250 contiguous samples; repeat the interrupted cue.")
            for trial in trials:
                if max(start, trial["start"]) < min(end, trial["end"]):
                    raise ValueError("Whole training and validation trials must not overlap.")
                forward = (rows[0]["counter"] - trial["counter_start"]) % 2 ** 32
                backward = (trial["counter_start"] - rows[0]["counter"]) % 2 ** 32
                if forward < trial["n_samples"] or backward < len(rows):
                    raise ValueError("Training and validation cannot share EEG packets.")
            _quality(data, context["channels"])
            waveform = preprocess_waveform(data)  # Use exactly the same preprocessing for training and live inference.
            trial = {"id": str(trial_id), "label": label, "start": float(start), "end": float(end),
                     "counter_start": rows[0]["counter"], "n_samples": len(rows), "waveform": waveform.tolist()}
            if phase == "train":
                self._training.append(trial)  # Store accepted labelled examples before any model is frozen.
            else:
                score = float(_scores(waveform, self._weights)[0])
                trial.update(score=score, predicted="select" if score >= self._model["threshold"] else "rest")
                self._validation.append(trial)  # Retain a frozen-model judgment without updating weights.
            self._context = context
            self.reset_stream()
            return self._result()
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            return self._result(str(exc))

    def train(self):
        if self._model is not None:
            return self._result("The CNN is already frozen; held-out trials never retrain it.")
        if any(sum(t["label"] == label for t in self._training) < MIN_PER_CLASS for label in LABELS):
            return self._result("Collect at least eight no-command and eight double-blink training trials.")
        try:
            import torch
        except (ImportError, OSError) as exc:
            return self._result("CNN training requires the optional CPU PyTorch package. Install with: python -m pip install torch --index-url https://download.pytorch.org/whl/cpu. " + str(exc))
        old_threads = torch.get_num_threads()  # Restore the process-wide CPU setting after explicit training.
        try:
            rng = np.random.default_rng(SEED)  # Keep augmentation independent from ambient random state.
            originals = np.asarray([t["waveform"] for t in self._training], dtype=np.float32)
            labels = np.asarray([int(t["label"] == "select") for t in self._training], dtype=np.int64)
            augmented, targets = [], []
            for waveform, label in zip(originals, labels):
                augmented.append(waveform)
                targets.append(label)
                for _ in range(12):
                    augmented.append(_augmentation(waveform, rng, command=bool(label)))
                    targets.append(label)
            x = torch.from_numpy(np.asarray(augmented, dtype=np.float32))
            y = torch.from_numpy(np.asarray(targets, dtype=np.int64))
            torch.set_num_threads(min(2, old_threads))  # Bound CPU contention with camera and EEG acquisition.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(SEED)  # The surrounding fork restores the caller random-generator state.
                network = _torch_network(torch).cpu()  # Training does not reserve the camera or Ollama GPU.
                optimizer = torch.optim.AdamW(network.parameters(), lr=.002, weight_decay=.001)
                network.train()
                for _ in range(EPOCHS):
                    indices = torch.randperm(len(x))
                    for batch in indices.split(16):
                        optimizer.zero_grad(set_to_none=True)  # Each batch accumulates only its own gradients.
                        loss = torch.nn.functional.cross_entropy(network(x[batch]), y[batch])
                        loss.backward()  # Differentiate training examples only.
                        torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)  # Bound unstable gradient updates.
                        optimizer.step()  # Update weights before computing the frozen training threshold.
                network.eval()  # Freeze inference behavior before extracting plain numeric weights.
                weights = {name: tensor.detach().cpu().numpy().copy() for name, tensor in network.state_dict().items()}
            scores = _scores(originals, weights)
            threshold, release = _threshold(scores, labels)  # Fail training if the controls and commands remain inseparable.
            self._model = {"architecture": ARCHITECTURE, "weights": {name: value.tolist() for name, value in weights.items()},
                           "threshold": threshold, "release_threshold": release, "seed": SEED, "epochs": EPOCHS,
                           "training_loss": float(loss.detach().cpu().item()), "torch_version": str(torch.__version__)}
            self._weights = weights
            self.reset_stream()
            return self._result()
        except (ValueError, RuntimeError, TypeError, OverflowError) as exc:
            return self._result("CNN training failed: " + str(exc))
        finally:
            torch.set_num_threads(old_threads)  # Restore the original setting on success and every handled failure.

    def _prediction(self, valid=False, reason="Collect and train the CNN first.", **extra):
        remaining = self._cancelled_evidence_remaining(self._counter) / HZ
        return {"valid": valid, "ready": bool(valid and self._ready), "label": "rest" if valid else None,
                "reason": reason, "confidence": self._last_score or 0.0, "event_id": None, "event_at": None,
                "fault_code": None, "fault_category": None, "recoverable": False, "fault_details": {},
                "experimental": True,
                "ready_for_blink": bool(valid and self._ready and self._rest_ready and not remaining),
                # The emitted event consumes the rest latch, but remains
                # eligible for its one immediate final dispatch check.
                "eligible_for_event": bool(valid and self._ready and not remaining
                                            and (self._rest_ready or extra.get("event_id"))),
                "suppressed_evidence_remaining_seconds": remaining, **extra}

    def _observe_score(self, score, counter, event_at, session_id):
        self._last_score = score  # Expose the latest score without treating it as a click by itself.
        if self._cancelled_evidence_remaining(counter):
            self._rest_count = self._select_count = 0
            self._rest_since_counter = self._select_since_counter = None
            self._rest_ready = False
            return None
        self._event_watermark = None
        if score <= self._model["release_threshold"]:
            self._select_count = 0
            self._select_since_counter = None
            if self._rest_since_counter is None:
                self._rest_since_counter = counter  # Measure dwell using newly received packets, not repeated polls.
            self._rest_count += 1
            if self._rest_count >= 2 and (counter - self._rest_since_counter) % 2 ** 32 >= SCORE_DWELL_SAMPLES:
                self._rest_ready = True
        elif score >= self._model["threshold"]:
            self._rest_count = 0
            self._rest_since_counter = None
            if self._select_since_counter is None:
                self._select_since_counter = counter  # A command needs persistent new evidence after recognized rest.
            self._select_count += 1
            outside_refractory = self._last_event_counter is None or (counter - self._last_event_counter) % 2 ** 32 >= HZ
            if (self._rest_ready and self._select_count >= 2 and outside_refractory
                    and (counter - self._select_since_counter) % 2 ** 32 >= SCORE_DWELL_SAMPLES):
                self._rest_ready = False
                self._select_count = 0
                self._select_since_counter = None
                self._last_event_counter = counter  # Enforce one second of sample-time cooldown between events.
                return self._prediction(True, "CNN recognized a double-blink waveform; one selection completed.",
                                        label="select", event_id=f"cnn-fast:{session_id}:{counter}", event_at=float(event_at))
        else:
            self._rest_count = self._select_count = 0
            self._rest_since_counter = self._select_since_counter = None
        return None

    def update(self, signal, now=None):
        if self._model is None:
            return self._prediction()
        now = self.clock() if now is None else now
        try:
            if not _finite(now):
                raise _InputFault("A finite current monotonic time is required.", "clock_invalid", "timing")
            context = _context(signal)
            if context != self._expected():
                raise _InputFault("CNN acquisition or montage changed; explicitly load/bind the model to this session.",
                                  "context_changed", "context")
            supplied = signal.get("samples")
            if not isinstance(supplied, (list, tuple)) or not 1 <= len(supplied) <= 1250:
                raise _InputFault("A bounded current full-rate EEG buffer is required.", "buffer_invalid", "transport")
            # The adapter shares its existing five-second cache; this model's
            # input and quality decisions concern only the newest one second.
            rows = supplied[-WINDOW_SAMPLES:]
            data = _rows(rows)
            latest = rows[-1]
            receipt = latest.get("host_received_monotonic", latest["estimated_monotonic"])
            if not _finite(receipt) or not -.05 <= now - receipt <= .75 or not -.05 <= now - latest["estimated_monotonic"] <= 1:
                raise _InputFault("CNN EEG buffer is stale or incorrectly timed.", "buffer_stale", "timing")
            fresh = None
            if self._counter is not None:
                prior_counter, prior_at, prior_receipt = self._counter, self._last_at, self._last_receipt
                fresh = [row for row in rows if 0 < (row["counter"] - prior_counter) % 2 ** 32 < 2 ** 31]
                if not fresh:
                    if latest["counter"] != prior_counter:
                        raise _InputFault("EEG counter moved backwards; CNN history cleared.", "counter_backwards", "transport")
                else:
                    if len(fresh) > 125 or (fresh[0]["counter"] - prior_counter) % 2 ** 32 != 1:
                        raise _InputFault("EEG packet gap or delayed backlog; CNN history cleared.",
                                          "packet_discontinuity", "transport")
                    if (fresh[0]["estimated_monotonic"] - prior_at > .5
                            or fresh[0]["estimated_monotonic"] - prior_at < -.25 or receipt - prior_receipt > .75):
                        raise _InputFault("EEG delivery paused; CNN history cleared.", "delivery_paused", "timing")
            # Large waveforms still need valid transport and usable samples.
            # Normalization cannot repair gaps, flatlines, or clipped plateaus.
            _quality(data, context["channels"])
            if self._cancel_pending_watermark:
                self._event_watermark = latest["counter"]
                self._cancel_pending_watermark = False
            if self._counter is None:
                self._counter, self._last_at, self._last_receipt = latest["counter"], latest["estimated_monotonic"], receipt
                return self._prediction(True, "Stream primed. Keep eyes open for one second of fresh CNN context.")
            if not fresh:
                return self._prediction(True, "Waiting for new EEG packets; buffered windows cannot repeat a click.")
            self._live_rows.extend(copy.deepcopy(fresh))
            self._counter, self._last_at, self._last_receipt = latest["counter"], latest["estimated_monotonic"], receipt
            if len(self._live_rows) < WINDOW_SAMPLES:
                return self._prediction(True, f"Collecting fresh CNN context: {len(self._live_rows) / HZ:.1f} / 1.0 seconds.")
            self._ready = True
            if self._last_predict_counter is not None and (self._counter - self._last_predict_counter) % 2 ** 32 < HOP_SAMPLES:
                return self._prediction(True, "CNN context ready; waiting for its next fresh window.")
            self._last_predict_counter = self._counter
            score = float(_scores(preprocess_waveform(_rows(list(self._live_rows))), self._weights)[0])
            result = self._observe_score(score, self._counter, self._last_at, context["session_id"])
            if self._cancelled_evidence_remaining(self._counter):
                return self._prediction(True, "Hand input ended; waiting for a CNN window containing only newer EEG.")
            return result or self._prediction(True, "CNN ready: double blink deliberately." if self._rest_ready
                                              else "Keep eyes open until the CNN recognizes no command.")
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            self.reset_stream()
            fields = (exc.prediction_fields() if isinstance(exc, _InputFault) else
                      {"fault_code": "input_invalid", "fault_category": "input", "recoverable": False})
            return self._prediction(False, str(exc), **fields)

    def status(self):
        counts = {phase: {label: sum(t["label"] == label for t in trials) for label in LABELS}
                  for phase, trials in (("train", self._training), ("validate", self._validation))}
        matrix = {label: {predicted: 0 for predicted in (*LABELS, "uncertain")} for label in LABELS}
        for trial in self._validation:
            matrix[trial["label"]][trial["predicted"]] += 1
        complete = all(counts["validate"][label] >= MIN_PER_CLASS for label in LABELS)
        recalls = [matrix[label][label] / counts["validate"][label] if counts["validate"][label] else 0 for label in LABELS]
        per_class = {label: {"support": counts["validate"][label], "recall": recalls[index],
                             "precision": matrix[label][label] / max(1, sum(matrix[actual][label] for actual in LABELS))}
                     for index, label in enumerate(LABELS)}
        false_rate = matrix["rest"]["select"] / counts["validate"]["rest"] if counts["validate"]["rest"] else None
        passed = bool(complete and sum(recalls) / 2 >= .75 and false_rate <= .10
                      and all(row["precision"] >= .70 and row["recall"] >= .70 for row in per_class.values()))
        return {"state": "validated" if passed else "validation_failed" if complete else "validating" if self._model is not None else "collecting",
                "labels": list(LABELS), "feature_profile": FEATURE_PROFILE, "detector_version": VERSION,
                "trained": self._model is not None, "passed_validation": passed,
                "arm_eligible": bool(passed and self._model is not None), "experimental": True,
                "can_train": self._model is None and all(counts["train"][label] >= MIN_PER_CLASS for label in LABELS),
                "training_counts": counts["train"], "train_counts": dict(counts["train"]),
                "validation_counts": counts["validate"], "min_per_class": MIN_PER_CLASS,
                "total": len(self._training) + len(self._validation), "max_training_trials": MAX_TRIALS - 16,
                "context": copy.deepcopy(self._context),
                "inference_context_bound": self._prediction_context is not None,
                "validation_scope": "held_out_one_second_trials", "parameter_count": PARAMETER_COUNT,
                "window_seconds": self.window_seconds, "trial_seconds": self.trial_seconds,
                "cooldown_seconds": self.cooldown_seconds,
                "fresh_context_seconds": len(self._live_rows) / HZ,
                "prediction_interval_seconds": HOP_SAMPLES / HZ,
                "ready_for_blink": bool(self._ready and self._rest_ready and not self._cancelled_evidence_remaining(self._counter)),
                "suppressed_evidence_remaining_seconds": self._cancelled_evidence_remaining(self._counter) / HZ,
                "evaluation": {"complete": complete, "passed": passed, "confusion": matrix,
                               "accuracy": sum(matrix[label][label] for label in LABELS) / len(self._validation) if self._validation else None,
                               "per_class": per_class, "abstentions": 0,
                               "balanced_accuracy": sum(recalls) / 2 if self._validation else None,
                               "rest_false_activations": matrix["rest"]["select"], "rest_trials": counts["validate"]["rest"],
                               "rest_false_activation_rate": false_rate},
                "feature_processing": "One-second four-channel waveform, pair-averaged to 125 Hz; DC/linear-drift removal and robust scaling; causal dilated endpoint CNN. Requires fresh calibration for this profile.",
                "disclaimer": "Held-out results classify one-second trials, not measured physical blink-completion latency. Continuous false-click rate, overlapping-window behavior, and latency require live assessment. CNN scores are not calibrated probabilities."}

    def export_state(self):
        if self._model is None:
            raise ValueError("Train the CNN before saving a frozen model.")
        return copy.deepcopy({"format": FORMAT, "version": VERSION, "feature_profile": FEATURE_PROFILE,
                              "context": self._context, "model": self._model,
                              "training": self._training, "validation": self._validation})

    @classmethod
    def from_state(cls, state):
        if (not isinstance(state, dict) or set(state) != {"format", "version", "feature_profile", "context", "model", "training", "validation"}
                or state.get("format") != FORMAT or type(state.get("version")) is not int or state["version"] != VERSION
                or state.get("feature_profile") != FEATURE_PROFILE):
            raise ValueError("Unsupported CNN model archive.")
        context = _context(state["context"])  # Validate saved montage and sampling metadata before restoring weights.
        model = state["model"]
        expected_model = {"architecture", "weights", "threshold", "release_threshold", "seed", "epochs", "training_loss", "torch_version"}
        if (not isinstance(model, dict) or set(model) != expected_model or model["architecture"] != ARCHITECTURE
                or not isinstance(model["weights"], dict) or set(model["weights"]) != set(WEIGHT_SHAPES)
                or not _finite(model["threshold"]) or not .65 <= model["threshold"] <= .99
                or not _finite(model["release_threshold"]) or not .25 <= model["release_threshold"] <= .60
                or type(model["seed"]) is not int or model["seed"] != SEED
                or type(model["epochs"]) is not int or model["epochs"] != EPOCHS
                or not _finite(model["training_loss"]) or not 0 <= model["training_loss"] <= 100
                or not isinstance(model["torch_version"], str) or not 1 <= len(model["torch_version"]) <= 64):
            raise ValueError("Malformed CNN architecture or training metadata.")
        weights = {}
        for name, shape in WEIGHT_SHAPES.items():
            values = model["weights"][name]
            _check_tensor(values, shape, limit=100)  # Check every tensor element and dimension before NumPy allocation.
            weights[name] = np.asarray(values, dtype=np.float32)  # Recreate only the fixed numeric inference arrays.
        if (not isinstance(state["training"], list) or not isinstance(state["validation"], list)
                or not 16 <= len(state["training"]) <= 48 or len(state["validation"]) > 16
                or len(state["training"]) + len(state["validation"]) > MAX_TRIALS):
            raise ValueError("CNN archive has invalid trial counts.")
        decoder = cls()  # A restored decoder starts with empty live history and no armed state.
        decoder._context = context
        decoder._model, decoder._weights = copy.deepcopy(model), weights
        seen = []
        for phase in ("training", "validation"):
            for record in state[phase]:
                keys = {"id", "label", "start", "end", "counter_start", "n_samples", "waveform"}
                if phase == "validation":
                    keys |= {"score", "predicted"}
                if (not isinstance(record, dict) or set(record) != keys or not isinstance(record["id"], str)
                        or not 1 <= len(record["id"]) <= 128 or record["label"] not in LABELS
                        or not _finite(record["start"]) or not _finite(record["end"]) or abs(record["end"] - record["start"] - 1) > .02
                        or type(record["counter_start"]) is not int or not 0 <= record["counter_start"] < 2 ** 32
                        or type(record["n_samples"]) is not int or record["n_samples"] != WINDOW_SAMPLES):
                    raise ValueError("Malformed CNN trial record.")
                for previous in seen:
                    if record["id"] == previous["id"] or max(record["start"], previous["start"]) < min(record["end"], previous["end"]):
                        raise ValueError("Duplicate or overlapping CNN trial archive.")
                    if ((record["counter_start"] - previous["counter_start"]) % 2 ** 32 < WINDOW_SAMPLES
                            or (previous["counter_start"] - record["counter_start"]) % 2 ** 32 < WINDOW_SAMPLES):
                        raise ValueError("CNN archive reuses training/validation packets.")
                _check_tensor(record["waveform"], (4, INPUT_SAMPLES), limit=12)
                trial = copy.deepcopy(record)
                if phase == "validation":
                    score = float(_scores(np.asarray(trial["waveform"], dtype=np.float32), weights)[0])
                    predicted = "select" if score >= model["threshold"] else "rest"
                    if not _finite(trial["score"]) or abs(trial["score"] - score) > 1e-5 or trial["predicted"] != predicted:
                        raise ValueError("CNN validation evidence does not match its frozen weights.")
                    trial.update(score=score, predicted=predicted)  # Retain recomputed evidence from the unchanged saved weights.
                getattr(decoder, "_" + phase).append(trial)
                seen.append(trial)  # Later records must not reuse this ID, time interval, or packet range.
        if any(sum(t["label"] == label for t in decoder._training) < MIN_PER_CLASS for label in LABELS):
            raise ValueError("CNN archive needs eight training trials of each class.")
        if any(sum(t["label"] == label for t in decoder._validation) > MIN_PER_CLASS for label in LABELS):
            raise ValueError("CNN archive exceeds held-out trial quotas.")
        scores = _scores(np.asarray([t["waveform"] for t in decoder._training], dtype=np.float32), weights)
        threshold, release = _threshold(scores, [int(t["label"] == "select") for t in decoder._training])
        if abs(threshold - model["threshold"]) > 1e-5 or abs(release - model["release_threshold"]) > 1e-5:
            raise ValueError("CNN threshold does not match its retained training evidence.")
        return decoder  # Loading restores evidence; the caller must separately bind and explicitly arm.


def _check_tensor(value, shape, *, limit):
    if not shape:
        if not _finite(value) or abs(value) > limit:
            raise ValueError("CNN archive contains a nonfinite or out-of-bounds tensor value.")
        return
    if not isinstance(value, list) or len(value) != shape[0]:
        raise ValueError("CNN tensor shape does not match the fixed architecture.")
    for item in value:
        _check_tensor(item, shape[1:], limit=limit)
