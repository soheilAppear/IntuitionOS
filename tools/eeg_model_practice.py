"""GET-only practice with the running EEG model; all input stays inside a canvas.

Run from the repository: .\.venv\Scripts\python.exe tools\eeg_model_practice.py
This observer never trains, arms, resets, saves, or stops the backend model.
"""

from __future__ import annotations

import copy
from collections import deque
import json
import math
from pathlib import Path
import queue
import sys
import threading
import time
import urllib.request

# Direct script execution otherwise places only tools/ on the module search path.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.hand_mouse import HandMouseController


PREVIEW_URL = "http://127.0.0.1:7432/multimodal/preview"
TARGETS = ((0.08, 0.12, 0.46, 0.44), (0.54, 0.12, 0.92, 0.44),
           (0.08, 0.56, 0.46, 0.88), (0.54, 0.56, 0.92, 0.88))


def finite_number(value):
    """Reject booleans, NaN, and infinity at the snapshot boundary."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class CanvasDevice:
    """In-memory hand-controller device: no operating-system input API exists here."""

    def __init__(self):
        self.cursor = (0.5, 0.5)  # Start the practice dot at the canvas center.
        self.down = False  # A pinch drag owns only this in-memory flag.

    def position(self):
        return self.cursor  # Read the practice dot, never the desktop cursor.

    def blocked(self, *, own_left=False):
        return None  # Physical keyboard/mouse state cannot affect this isolated canvas device.

    def move(self, x, y):
        self.cursor = (x, y)  # The GUI paints this position on its next refresh.

    def left_down(self):
        self.down = True  # This deliberately does not call SendInput or another OS API.

    def left_up(self):
        self.down = False  # Completed clicks are read from hand feedback, so drag release is not a click.


class PracticeController:
    """Pure snapshot/time state machine; effects are counters and canvas coordinates."""

    def __init__(self):
        self.device = CanvasDevice()  # Supply the in-memory device explicitly to prevent native construction.
        self.hand = HandMouseController(eeg_compatible=True, device_factory=lambda: self.device)
        self.active = False  # Opening the observer never starts practice automatically.
        self.message = "Start practice when the current camera and EEG preview are running."
        self.counts = {"eeg": 0, "finger": 0}  # These are local attempts, including attempts between targets.
        self.hits = [0] * len(TARGETS)  # Each target reports successful inside-window selections.
        self.last_event = "No practice selections yet."
        self.original_decoder = None  # Preserve the original metrics of the model chosen on Start.
        self.prediction = {}  # Live display only; no feature vectors or raw samples are retained.
        self.session = None  # Bind each explicit practice run to one acquisition session.
        self.model_signature = None  # Refuse a changed decoder instead of silently practising another model.
        self.last_dispatch = -math.inf  # Share the two-second spacing rule between manual and EEG attempts.
        self.last_click_id = 0  # Hand click IDs survive controller reset, preventing replay after Pause.
        self._reset_cycle()

    def _reset_cycle(self):
        self.uncertain_since = None  # No transition allowance survives a cancelled practice cycle.
        self.rest_since = None  # Require one continuous second of distinct confident rest predictions.
        self.rest_ready = False  # Arming or pausing while selected cannot create an immediate selection.
        self.select_count = 0  # Every select streak needs three distinct prediction windows.
        self.pending_at = None  # No queued EEG action survives a cancellation or manual gesture.

    @staticmethod
    def _signature(decoder):
        # Status contains no model weights; frozen counts/context/metrics still detect observable model changes.
        return json.dumps(decoder, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @staticmethod
    def _stream_error(snapshot, now, received_at):
        # Account for HTTP transit and GUI delays rather than trusting a cached age of zero.
        delay = now - received_at
        if not finite_number(delay) or delay < 0 or delay > 0.75:
            return "Preview response is stale; reconnect and start practice again."
        if not isinstance(snapshot, dict) or snapshot.get("state") != "running":
            return "The existing camera + EEG preview must remain running."
        camera, eeg = snapshot.get("camera") or {}, snapshot.get("eeg") or {}
        camera_age, eeg_age = camera.get("age_ms"), (eeg.get("stats") or {}).get("age_seconds")
        if camera.get("running") is not True or camera.get("tracked") is not True:
            return "Hand tracking is unavailable; show your hand and start practice again."
        if not finite_number(camera_age) or not 0 <= camera_age / 1000 + delay <= 0.5:
            return "Camera frame is stale; start again after fresh hand tracking returns."
        if eeg.get("state") != "running" or eeg.get("mode") != "signal" or not eeg.get("session_id"):
            return "Current-session EEG signal acquisition is required."
        if not finite_number(eeg_age) or not 0 <= eeg_age + delay <= 0.75:
            return "EEG samples are stale; start again after fresh samples return."
        points = camera.get("landmarks")
        if (not isinstance(points, list) or len(points) != 21
                or any(not isinstance(point, (list, tuple)) or len(point) < 2
                       or any(not finite_number(value) for value in point[:2]) for point in points)):
            return "A complete, finite 21-landmark hand is required."
        sequence = camera.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            return "Camera frame identity is unavailable."
        return None

    @staticmethod
    def _gaps(snapshot):
        stats = (snapshot.get("eeg") or {}).get("stats") or {}
        # Invalid diagnostic values also fail closed instead of being treated as healthy zeroes.
        values = [stats.get(key, 0) for key in ("gaps", "queue_drops", "channel_mismatches", "duplicates", "nonfinite")]
        return sum(values) if all(finite_number(value) and value >= 0 for value in values) else math.inf

    def start(self, snapshot, now, received_at=None):
        received_at = now if received_at is None else received_at  # Tests can provide explicit fake receipt times.
        self.pause("Preparing local practice.")  # Clear only local state; this sends no backend request.
        error = self._stream_error(snapshot, now, received_at)
        decoder = ((snapshot or {}).get("eeg_control") or {}).get("decoder") or {}
        if error:
            self.message = error
            return False
        if decoder.get("trained") is not True or set(decoder.get("labels") or []) != {"rest", "select"}:
            self.message = "The existing frozen model must have rest and select labels."
            return False
        try:
            self.model_signature = self._signature(decoder)  # Freeze the observable identity without changing validation.
        except (TypeError, ValueError):
            self.message = "The current model status is invalid."
            return False
        self.original_decoder = copy.deepcopy(decoder)  # Failed validation remains visible and unchanged.
        self.session = snapshot["eeg"]["session_id"]  # A reconnect must require another explicit Start.
        self.gap_count = self._gaps(snapshot)  # Reject new packet errors during this run.
        if not math.isfinite(self.gap_count):
            self.message = "EEG packet diagnostics are invalid."
            return False
        self.camera_sequence, self.camera_at = None, now  # Fresh distinct frames must establish the pointing dwell.
        self.window_id, self.prediction_at = None, now  # Repeated predictions cannot advance EEG timing.
        self.seen_windows = set()  # Reject earlier windows replayed within this practice run.
        self.window_order = deque(maxlen=4096)  # Bound replay bookkeeping to roughly seventeen minutes of four-Hz predictions.
        self.engaged = False  # No hand readiness is inherited from a prior run.
        self.active = True
        self.message = "Point your index finger briefly, then move your hand to aim inside this window."
        return True

    def pause(self, reason="Practice paused. The running model is unchanged."):
        self.active = False  # Disable the pure state machine before releasing its in-memory button.
        self.hand.close()  # This controller uses CanvasDevice exclusively; no OS button exists.
        self._reset_cycle()  # Never retain a partially completed EEG selection after Pause.
        self.message = reason

    def _activate(self, source, now):
        self.counts[source] += 1  # Count one completed local selection attempt.
        x, y = self.device.cursor  # The target belongs to the canvas, not the desktop pointer.
        target = next((index for index, (left, top, right, bottom) in enumerate(TARGETS)
                       if left <= x <= right and top <= y <= bottom), None)
        if target is not None:
            self.hits[target] += 1  # Only targets containing the held practice dot receive a hit.
        name = chr(65 + target) if target is not None else "between targets"
        self.last_event = f"{source.upper()} selection: {name}."
        self.last_dispatch = now  # Manual input also delays the next EEG completion.
        self._reset_cycle()  # One completion consumes the entire rest/select/rest cycle.

    def update(self, snapshot, now, received_at=None):
        received_at = now if received_at is None else received_at
        self.prediction = copy.deepcopy(((snapshot or {}).get("eeg_control") or {}).get("prediction") or {})
        if not self.active:
            return  # Paused polling updates the display but cannot move the practice dot.
        error = self._stream_error(snapshot, now, received_at)
        if error:
            self.pause(error)
            return
        decoder = (snapshot.get("eeg_control") or {}).get("decoder") or {}
        try:
            same_model = self._signature(decoder) == self.model_signature
        except (TypeError, ValueError):
            same_model = False
        if snapshot["eeg"]["session_id"] != self.session or not same_model:
            self.pause("The acquisition session or frozen model changed; start a new local practice run.")
            return
        if self._gaps(snapshot) != self.gap_count:
            self.pause("EEG packet diagnostics changed; the unfinished selection was cancelled.")
            return
        camera, prediction = snapshot["camera"], self.prediction
        sequence = camera["sequence"]
        if self.camera_sequence is not None and sequence < self.camera_sequence:
            self.pause("Camera sequence moved backwards; start again with fresh tracking.")
            return
        new_frame = sequence != self.camera_sequence
        if new_frame:
            self.camera_sequence, self.camera_at = sequence, now
        if now - self.camera_at > 0.5:
            self.pause("Camera stopped producing distinct frames; selection cancelled.")
            return
        window_id = prediction.get("window_id")
        new_window = isinstance(window_id, str) and bool(window_id) and window_id != self.window_id
        if new_window:
            if window_id in self.seen_windows:
                self.pause("An old EEG prediction window was replayed; selection cancelled.")
                return
            if len(self.window_order) == self.window_order.maxlen:
                self.seen_windows.discard(self.window_order.popleft())  # Retire the oldest identity together with its set entry.
            self.window_order.append(window_id)  # Keep eviction order separate from fast replay lookup.
            self.seen_windows.add(window_id)  # This is a short string identity, never an EEG recording.
            self.window_id, self.prediction_at = window_id, now
        if now - self.prediction_at > 0.75:
            self.pause("EEG model predictions stopped updating; selection cancelled.")
            return
        if self.pending_at is not None and now - self.pending_at > 8.0:
            self._reset_cycle()  # An expired selection cannot fire when rest eventually returns.
            self.message = "Selection expired. Rest again before another attempt."
        confident = (prediction.get("valid") is True and prediction.get("ood") is False
                     and prediction.get("label") in ("rest", "select")
                     and finite_number(prediction.get("confidence")) and prediction["confidence"] >= 0.8
                     and finite_number(prediction.get("margin")) and prediction["margin"] >= 0.25)
        if new_frame:
            # Freeze ordinary aiming during selection/uncertainty; finger input still takes priority.
            hold = self.pending_at is not None or not confident or prediction.get("label") != "rest"
            self.hand_feedback = self.hand.update(camera["landmarks"], now, hold=hold)
        feedback = getattr(self, "hand_feedback", {})
        if feedback.get("error") or self.engaged and feedback.get("state") in ("mouse_paused", "mouse_recovering", "mouse_error"):
            self.pause("Hand tracking became unusable; point and start again.")
            return
        if new_frame and feedback.get("finger_active"):
            self._reset_cycle()  # Any manual intent cancels the competing EEG cycle before completion.
            self.last_dispatch = now
            click_id = feedback.get("click_id")
            if isinstance(click_id, int) and click_id > self.last_click_id:
                self.last_click_id = click_id  # Count completed pinch releases only once.
                self._activate("finger", now)
            self.message = "Finger input has priority. Rest again before a new EEG selection."
            return
        if feedback.get("pointer_ready") is not True or feedback.get("finger_active") or feedback.get("button_down"):
            self._reset_cycle()
            self.message = feedback.get("hint") or "Point your index finger briefly to engage."
            return
        self.engaged = True  # Future loss of this established target cancels the whole run.
        if not confident:
            valid_raw_transition = (isinstance(window_id, str) and bool(window_id)
                                    and (prediction.get("valid") is True or prediction.get("ood") is True))
            if valid_raw_transition:
                if self.uncertain_since is None:
                    self.uncertain_since = now  # Allow the two-second analysis window to cross between eyes-open/closed states.
                self.select_count = 0  # Uncertainty cannot contribute to a three-window select streak.
                self.rest_since = None  # Rest must become continuously confident again before completing a click.
                if now - self.uncertain_since >= 3.0:
                    self.pause("EEG stayed uncertain for three seconds; start practice again after recovery.")
                else:
                    # Retain a prepared rest interval or pending selection, while holding aim and producing no action.
                    self.message = "EEG transition: target held; waiting up to three seconds for confident rest/select."
            else:
                self._reset_cycle()  # Invalid raw/model input cancels the cycle immediately, not just its streak.
                self.message = "EEG prediction is invalid; finger practice remains available."
            return
        self.uncertain_since = None  # A confident current window ends the bounded transition allowance.
        if not new_window:
            return  # Cached predictions cannot add streaks or complete a rest interval.
        if prediction["label"] == "rest":
            self.select_count = 0
            if self.rest_since is None:
                self.rest_since = now  # Start the fresh rest interval on a distinct confident window.
            if now - self.rest_since >= 1.0:
                self.rest_ready = True
                if self.pending_at is not None and now - self.last_dispatch >= 2.0:
                    self._activate("eeg", now)
                    self.message = "EEG selection completed inside this window. Rest before selecting again."
                else:
                    self.message = "Rest ready. Use your calibrated select pattern, then return to rest."
            else:
                self.message = "Keep resting until one second of confident rest is complete."
        else:
            self.rest_since = None  # Select interrupts the continuous rest timer.
            if self.pending_at is not None:
                self.message = "Selection detected. Return to rest to complete it."
            elif self.rest_ready:
                self.select_count += 1
                if self.select_count >= 3:
                    self.pending_at, self.rest_ready = now, False
                    self.message = "Selection detected. Return to rest to activate the held target."
            else:
                self.message = "Begin with one second of confident rest before selecting."


def metrics_text(decoder):
    """Describe original validation honestly, including a failed or incomplete result."""
    decoder, evaluation = decoder or {}, (decoder or {}).get("evaluation") or {}
    percent = lambda value: f"{value:.1%}" if finite_number(value) else "not available"
    return (f"Original model: {decoder.get('state', 'unavailable')} | "
            f"training {decoder.get('train_counts', {})} | validation {decoder.get('validation_counts', {})}\n"
            f"Held-out balanced accuracy: {percent(evaluation.get('balanced_accuracy'))} | "
            f"Rest false selections: {percent(evaluation.get('rest_false_activation_rate'))}\n"
            f"Confusion matrix: {evaluation.get('confusion_matrix', {})}\n"
            "Results are unchanged. Practice is allowed here even if validation failed; every effect stays in this window.")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None  # A localhost observer must never follow a redirect to another service.


def fetch_snapshot(opener):
    # Only this fixed GET endpoint is available; there is no backend mutation transport.
    request = urllib.request.Request(PREVIEW_URL, headers={"X-Intuition-Multimodal": "1"}, method="GET")
    with opener.open(request, timeout=2.0) as response:
        payload = response.read(4_000_001)  # Bound camera-image payload memory before JSON parsing.
    if len(payload) > 4_000_000:
        raise ValueError("Preview response exceeded the size limit.")
    snapshot = json.loads(payload)
    if not isinstance(snapshot, dict):
        raise ValueError("Preview returned an invalid snapshot.")
    # Discard image and waveform bytes immediately; practice keeps only existing predictions/landmarks.
    (snapshot.get("camera") or {}).pop("image", None)
    (snapshot.get("eeg") or {}).pop("waveform", None)
    return snapshot


def run_window():
    import tkinter as tk  # Import lazily so pure controller tests need neither Tk nor a display.

    root = tk.Tk()
    root.title("EEG practice — current live model")
    root.geometry("1000x760")
    root.minsize(760, 650)
    controller, inbox, halted = PracticeController(), queue.Queue(maxsize=1), threading.Event()
    latest = {"snapshot": None, "received_at": -math.inf, "error": None}
    root.configure(bg="#111827")
    tk.Label(root, text="EEG + hand practice", font=("Segoe UI", 20, "bold"),
             bg="#111827", fg="#f9fafb").pack(pady=(16, 4))
    tk.Label(root, text="Same live model. No retraining or restart. Selections affect these four targets only.",
             bg="#111827", fg="#d1d5db").pack()
    metrics, status, prediction_line, counts = (tk.StringVar() for _ in range(4))
    tk.Label(root, textvariable=metrics, justify="left", anchor="w", wraplength=960,
             bg="#1f2937", fg="#fbbf24", padx=12, pady=10).pack(fill="x", padx=16, pady=12)
    bar = tk.Frame(root, bg="#111827")
    bar.pack(fill="x", padx=16)

    def toggle():
        if controller.active:
            controller.pause()
        elif latest["snapshot"] is not None:
            controller.start(latest["snapshot"], time.monotonic(), latest["received_at"])
        else:
            controller.message = latest["error"] or "Waiting for the existing backend preview."

    button = tk.Button(bar, text="Start practice", command=toggle, font=("Segoe UI", 12, "bold"), padx=16)
    button.pack(side="left")
    tk.Label(bar, text="Point briefly to aim • Pinch/release to select • EEG rest → select → rest • Esc pauses",
             bg="#111827", fg="#d1d5db", wraplength=720, justify="left").pack(side="left", padx=14)
    canvas = tk.Canvas(root, bg="#0b1220", highlightthickness=0)
    canvas.pack(fill="both", expand=True, padx=16, pady=12)
    for variable in (status, prediction_line, counts):
        tk.Label(root, textvariable=variable, wraplength=950, bg="#111827", fg="#e5e7eb",
                 anchor="w", justify="left").pack(fill="x", padx=18, pady=3)
    tk.Label(root, text="This window does not arm desktop controls or change the backend. Keep the existing preview running.",
             bg="#111827", fg="#9ca3af").pack(pady=(4, 14))

    def poll():
        # Disable proxy inheritance so even this localhost GET cannot be sent through an external proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        while not halted.is_set():
            started = time.monotonic()  # Freshness includes time spent waiting for the HTTP response.
            try:
                item = (fetch_snapshot(opener), started, None)
            except Exception as exc:
                item = (None, started, str(exc))
            try:
                inbox.get_nowait()  # Keep only the newest response; raw previews never accumulate in a backlog.
            except queue.Empty:
                pass
            inbox.put_nowait(item)
            halted.wait(0.1)  # Waiting on an event lets window close stop polling promptly.

    def refresh():
        try:
            snapshot, received_at, error = inbox.get_nowait()
            latest.update(snapshot=snapshot, received_at=received_at, error=error)
        except queue.Empty:
            pass
        if latest["snapshot"] is not None:
            controller.update(latest["snapshot"], time.monotonic(), latest["received_at"])
            current_decoder = (latest["snapshot"].get("eeg_control") or {}).get("decoder")
            metrics.set(metrics_text(controller.original_decoder or current_decoder))
        elif controller.active:
            controller.pause("Preview connection failed; the unfinished selection was cancelled.")
        if latest["error"] and not controller.active:
            status.set("Connection: " + latest["error"])
        else:
            status.set(controller.message)
        button.configure(text="Pause practice" if controller.active else "Start practice")
        prediction = controller.prediction
        confidence, margin = prediction.get("confidence"), prediction.get("margin")
        score_text = lambda value: f"{value:.2f}" if finite_number(value) else "—"
        prediction_line.set(f"Live EEG: {prediction.get('label') or 'uncertain'} | score {score_text(confidence)} | "
                            f"margin {score_text(margin)} | select streak {controller.select_count}/3")
        counts.set(f"EEG selections: {controller.counts['eeg']} | Finger selections: {controller.counts['finger']} | {controller.last_event}")
        width, height = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        canvas.delete("all")  # Repaint only simple local shapes; no camera imagery is displayed or saved.
        x, y = controller.device.cursor
        for index, (left, top, right, bottom) in enumerate(TARGETS):
            hovered = controller.active and left <= x <= right and top <= y <= bottom
            canvas.create_rectangle(left * width, top * height, right * width, bottom * height,
                                    fill="#155e75" if hovered else "#1f2937", outline="#67e8f9" if hovered else "#374151", width=3)
            canvas.create_text((left + right) * width / 2, (top + bottom) * height / 2,
                               text=f"Target {chr(65 + index)}\n{controller.hits[index]} selections",
                               fill="#f9fafb", font=("Segoe UI", 20, "bold"), justify="center")
        color = "#fbbf24" if controller.pending_at is not None else "#34d399" if controller.active else "#9ca3af"
        canvas.create_oval(x * width - 10, y * height - 10, x * width + 10, y * height + 10, fill=color, outline="white", width=2)
        root.after(50, refresh)  # UI rendering never waits for network access.

    def close():
        controller.pause()  # Release only the local virtual button and discard the local cycle.
        halted.set()  # No stop/reset request is sent to the running backend.
        root.destroy()

    root.bind("<Escape>", lambda event: controller.pause())  # Escape affects this focused practice window only.
    root.protocol("WM_DELETE_WINDOW", close)
    threading.Thread(target=poll, name="eeg-practice-readonly", daemon=True).start()
    refresh()
    root.mainloop()


if __name__ == "__main__":
    run_window()
