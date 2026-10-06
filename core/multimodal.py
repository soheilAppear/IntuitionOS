"""Explicit, ephemeral webcam/EEG preview and separately armed control modes.

The camera already mirrors frames before tracking; increasing landmark x is
right in that mirrored view. EEG timing is reconstructed from host arrival, not
hardware synchronization. Calibration measures movement/artifact correlation.
No file, network, journal, or sensor discovery operations belong in this module.
"""

from __future__ import annotations

import copy
import math
import statistics
import threading
import time
from collections import deque

from .hand_tracking import FIRST_FRAME_TIMEOUT, STARTUP_TIMEOUT
from .eeg_decoder import EegDecoder, extract_features


DISCLAIMER = ("Exploratory movement/artifact correlations, not intent decoding. "
              "Webcam mode uses camera direction; EEG mode uses only EEG features. "
              "Camera/EEG alignment uses approximate host timing, not synchronized hardware timestamps.")


class MirroredMotionTracker:
    """Horizontal travel with neutral rearming, frame debounce, and cooldown."""

    settings = {"travel": 0.18, "window_seconds": 0.7, "deadband": 0.035,
                "neutral_seconds": 0.35, "cooldown_seconds": 1.5}

    def __init__(self):
        self.reset()

    def reset(self):
        self._points = deque(maxlen=16)
        self._neutral = deque(maxlen=16)
        self._ready = False
        self._candidate = None
        self._candidate_count = 0
        self._last_event = None
        self._cooldown_until = -math.inf
        self._progress = 0.0

    def update(self, x, y, now):
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (x, y, now)):
            self.reset()
            return None
        if not self._ready:
            self._neutral.append((now, x, y))
            while self._neutral and now - self._neutral[0][0] > 0.55:
                self._neutral.popleft()
            if self._neutral:
                xs = [p[1] for p in self._neutral]
                ys = [p[2] for p in self._neutral]
                if max(xs) - min(xs) > self.settings["deadband"] or max(ys) - min(ys) > self.settings["deadband"]:
                    self._neutral.clear()
                    self._neutral.append((now, x, y))
                elif (now >= self._cooldown_until and len(self._neutral) >= 3
                      and now - self._neutral[0][0] >= self.settings["neutral_seconds"]):
                    self._ready = True
                    self._points.clear()
                    self._points.append((now, x, y))
            return None
        self._points.append((now, x, y))
        while self._points and now - self._points[0][0] > self.settings["window_seconds"]:
            self._points.popleft()
        first = self._points[0]
        dx, dy = x - first[1], y - first[2]
        self._progress = min(1.0, abs(dx) / self.settings["travel"])
        direction = None
        if len(self._points) >= 3 and now - first[0] >= 0.15:
            if abs(dx) >= self.settings["travel"] and abs(dx) > abs(dy) * 1.5:
                direction = "right" if dx > 0 else "left"
        if direction is None:
            self._candidate = None
            self._candidate_count = 0
            return None
        self._candidate_count = self._candidate_count + 1 if self._candidate == direction else 1
        self._candidate = direction
        if self._candidate_count < 2:
            return None
        self._last_event = {"direction": direction, "at": now}
        self._cooldown_until = now + self.settings["cooldown_seconds"]
        self._ready = False
        self._neutral.clear()
        self._points.clear()
        self._candidate = None
        self._candidate_count = 0
        self._progress = 0.0
        return direction

    def status(self, now):
        event = self._last_event
        return {"direction": event["direction"] if event and now - event["at"] < 0.8 else "neutral",
                "ready": self._ready, "progress": round(self._progress, 3),
                "last_event": copy.deepcopy(event), "settings": dict(self.settings)}


class MultimodalPreview:
    """One owned camera, one explicit EEG session, and a lightweight reader worker.

    Call start/stop/check_contact on an executor. status/snapshot never touch
    hardware. The supplied camera must have empty bindings and no callbacks.
    """

    TICK_SECONDS = 0.125
    CAMERA_FRESH_SECONDS = 0.5
    EEG_FRESH_SECONDS = 0.75
    STARTUP_GRACE_SECONDS = 5.0
    # The camera starts asynchronously and is not running until first inference.
    # Allow its existing GPU startup/first-frame budgets plus capture setup time.
    CAMERA_STARTUP_SECONDS = STARTUP_TIMEOUT + FIRST_FRAME_TIMEOUT + STARTUP_GRACE_SECONDS
    CAMERA_LOSS_SECONDS = 2.0
    EEG_LOSS_SECONDS = 2.0
    CONTACT_SECONDS = 5.0
    TRIAL_SECONDS = 3.0
    MAX_TRIALS = 64
    EEG_WINDOW_SECONDS = 2.0
    EEG_PREDICTION_HOP = 0.25
    EEG_CONFIDENCE = 0.8
    EEG_MARGIN = 0.25
    EEG_REST_SECONDS = 1.0
    EEG_COOLDOWN_SECONDS = 2.0
    EEG_GUARD_SECONDS = 3.0
    EEG_TRANSITION_SECONDS = 3.0
    MAX_RETIRED_GUARDS = 256

    def __init__(self, camera, brainbit, gesture_busy=lambda: False,
                 dispatch=None, clock=time.monotonic):
        self.camera = camera
        self.brainbit = brainbit
        self.gesture_busy = gesture_busy
        self.dispatch = dispatch
        self.clock = clock
        self._lock = threading.RLock()
        self._operation_lock = threading.Lock()
        self._action_lock = threading.Lock()
        self._halt = threading.Event()
        self._thread = None
        self._generation = 0
        self._arm_epoch = 0
        self._revision = 0
        self._last_dispatch_at = -math.inf
        self._state = "stopped"
        self._reason = "Preview is off. Start explicitly to acquire camera and EEG."
        self._error = None
        self._armed = False
        self._healthy = False
        self._owns_camera = False
        self._owns_eeg = False
        self._started_at = 0.0
        self._camera_started_at = 0.0
        self._camera_ready = False
        self._last_camera_fresh_at = -math.inf
        self._last_eeg_live_at = -math.inf
        self._observed_at = -math.inf
        self._camera_sequence = None
        self._camera_changed_at = -math.inf
        self._camera = {}
        self._eeg = {}
        self._waveform = []
        self._raw = deque(maxlen=1500)
        self._seen_samples = set()
        self._seen_order = deque(maxlen=2000)
        self._gap_count = 0
        self._last_valid_counter = None
        self._last_valid_eeg_at = -math.inf
        self._motion = MirroredMotionTracker()
        self._pending = None
        self._trials = []
        self._trial_id = 0
        self._last_trial_end = -math.inf
        self._last_trial_result = None
        self._evaluation = self._evaluate([])
        self._control_mode = "webcam"
        self._decoder = EegDecoder()
        self._decoder_lock = threading.RLock()
        self._decoder_status = self._decoder.status()
        self._decoder_epoch = 0
        self._eeg_armed = False
        self._eeg_signal_healthy = False
        self._eeg_full_rows = []
        self._eeg_prediction = None
        self._eeg_prediction_at = -math.inf
        self._eeg_prediction_session = None
        self._eeg_uncertain_since = None
        self._eeg_raw_window_valid = False
        self._eeg_last_predict_at = -math.inf
        self._eeg_last_prediction_id = None
        self._eeg_pending = None
        self._eeg_last_trial_end = -math.inf
        self._eeg_trial_sequence = 0
        self._eeg_last_result = None
        self._eeg_reason = "Collect labelled training trials, freeze the model, then collect held-out validation trials."
        self._eeg_guard_token = None
        self._eeg_guard_until = -math.inf
        self._eeg_retired_guard_tokens = set()
        self._reset_eeg_debounce()

    @staticmethod
    def _error_of(result):
        if isinstance(result, dict):
            return result.get("error")
        return "Device returned an invalid result."

    def _result(self, error=None):
        with self._lock:
            self._revision += 1
            result = self.status()
        if error:
            result["error"] = str(error)
        else:
            result["ok"] = True
        return result

    def start(self):
        with self._operation_lock:
            with self._lock:
                if self._state in ("starting", "running"):
                    return self._result()
                if self._state == "contact":
                    return self._result("Wait for the contact check to finish.")
                if self._owns_camera or self._owns_eeg:
                    return self._result("Stop again to confirm device cleanup before restarting.")
            if self.gesture_busy() or self.camera.is_running():
                return self._result("Turn off the normal camera and Hand Mouse controls before starting this preview.")
            existing = self.brainbit.acquisition_snapshot()
            if existing.get("state") in ("starting", "running", "stopping"):
                return self._result("Stop the existing EEG/contact acquisition before starting this preview.")
            with self._lock:
                self._generation += 1
                self._halt.clear()
                self._clear_live()
                self._state = "starting"
                self._reason = "Starting camera and EEG; controls remain disarmed."
                self._error = None
                self._started_at = self.clock()
                self._owns_eeg = True
            try:
                result = self.brainbit.start_acquisition(mode="signal")
                if self._error_of(result):
                    raise RuntimeError(self._error_of(result))
                with self._lock:
                    cancelled = self._halt.is_set()
                    if not cancelled:
                        # Neither stream's first-data budget includes time spent
                        # waiting for the native start command acknowledgement.
                        self._started_at = self.clock()
                        self._owns_camera = self._control_mode == "webcam"
                        self._camera_started_at = self._started_at
                if cancelled:
                    return self._stop_owned("Preview startup cancelled.")
                if self._owns_camera:
                    result = self.camera.start()
                    if self._error_of(result):
                        raise RuntimeError(self._error_of(result))
                self._launch_worker()
            except Exception as exc:
                return self._stop_owned("Start failed: " + str(exc), failed=True)
            return self._result()

    def _launch_worker(self):
        self._thread = threading.Thread(target=self._run, args=(self._generation,), name="multimodal-preview", daemon=True)
        self._thread.start()

    def _run(self, generation):
        while not self._halt.is_set() and generation == self._generation:
            try:
                self._tick()
            except Exception as exc:
                if generation == self._generation:
                    self._fail("Preview failed: " + str(exc), generation)
                break
            self._halt.wait(self.TICK_SECONDS)

    def stop(self):
        # Fail closed even if a contact/start operation currently owns the lock.
        with self._lock:
            self._armed = False
            self._retire_eeg_guard_locked()
            self._disarm_eeg_locked("Preview stopped.")
            self._healthy = False
            self._halt.set()
            self._arm_epoch += 1
            self._revision += 1
            owns_eeg = self._owns_eeg
        # The adapter explicitly supports cancelling a pending native operation.
        # Do this before waiting for the lifecycle lock held by start/contact.
        if owns_eeg:
            try:
                if self.brainbit.acquisition_snapshot().get("state") == "starting":
                    self.brainbit.stop_acquisition()
            except Exception:
                pass  # Owned cleanup below reports any unconfirmed stop.
        with self._action_lock:
            pass  # Finish any action already dispatched before stop was requested.
        with self._operation_lock:
            return self._stop_owned("Preview stopped; live samples and images discarded.")

    def _stop_owned(self, reason, failed=False):
        with self._lock:
            self._state = "stopping"
            self._armed = False
            self._healthy = False
            self._halt.set()
            self._generation += 1
            owns_camera, owns_eeg = self._owns_camera, self._owns_eeg
            self._clear_live()
        errors = []
        if owns_camera:
            try:
                result = self.camera.stop()
                error = self._error_of(result)
                if error or result.get("stopping") or result.get("tracker_cleanup_pending"):
                    errors.append("Camera stop unconfirmed: " + str(error or "cleanup still pending"))
                else:
                    with self._lock:
                        self._owns_camera = False
            except Exception as exc:
                errors.append("Camera stop unconfirmed: " + str(exc))
        if owns_eeg:
            try:
                result = self.brainbit.stop_acquisition()
                error = self._error_of(result)
                if error or result.get("state") in ("running", "starting", "stopping", "error"):
                    errors.append("EEG stop unconfirmed: " + str(error or result.get("state")))
                else:
                    with self._lock:
                        self._owns_eeg = False
            except Exception as exc:
                errors.append("EEG stop unconfirmed: " + str(exc))
        with self._lock:
            self._state = "error" if errors or failed else "stopped"
            self._error = "; ".join(errors) if errors else (reason if failed else None)
            self._reason = reason + (" " + "; ".join(errors) if errors else "")
        return self._result(self._error)

    def _clear_live(self):
        self._revision += 1
        self._armed = False
        self._arm_epoch += 1
        self._healthy = False
        self._camera = {}
        self._eeg = {}
        self._waveform = []
        self._raw.clear()
        self._seen_samples.clear()
        self._seen_order.clear()
        self._camera_sequence = None
        self._camera_changed_at = -math.inf
        self._camera_ready = False
        self._last_camera_fresh_at = -math.inf
        self._last_eeg_live_at = -math.inf
        self._observed_at = -math.inf
        self._gap_count = 0
        self._last_valid_counter = None
        self._last_valid_eeg_at = -math.inf
        self._motion.reset()
        if self._pending:
            self._last_trial_result = "Trial cancelled: preview stopped."
        self._pending = None
        self._retire_eeg_guard_locked()
        self._disarm_eeg_locked("Live samples cleared. A new acquisition session requires new training and validation.")
        self._eeg_signal_healthy = False
        self._eeg_full_rows = []
        self._eeg_prediction = None
        self._eeg_prediction_at = -math.inf
        self._eeg_prediction_session = None
        self._eeg_raw_window_valid = False
        self._eeg_last_predict_at = -math.inf
        self._eeg_last_prediction_id = None
        if self._eeg_pending:
            self._eeg_last_result = "EEG trial cancelled: preview stopped."
        self._eeg_pending = None
        self._eeg_guard_token = None
        self._eeg_guard_until = -math.inf

    def _fail(self, reason, generation):
        with self._lock:
            if self._halt.is_set() or generation != self._generation:
                return
            self._armed = False
            self._disarm_eeg_locked(reason)
            self._healthy = False
            self._revision += 1
            self._halt.set()
        with self._operation_lock:
            with self._lock:
                if generation != self._generation or self._state not in ("starting", "running"):
                    return
            self._stop_owned(reason, failed=True)

    def check_contact(self):
        """Bounded resistance-only precheck; never concurrent with signal."""
        with self._operation_lock:
            with self._lock:
                if self._state in ("starting", "running", "contact", "stopping") or self._owns_eeg or self._owns_camera:
                    return self._result("Stop the preview before checking electrode contact.")
            if self.gesture_busy():
                return self._result("Turn off the normal camera and Hand Mouse controls before the contact check.")
            existing = self.brainbit.acquisition_snapshot()
            if existing.get("state") in ("running", "starting", "stopping"):
                return self._result("Stop the existing acquisition before checking contact.")
            with self._lock:
                self._halt.clear()
                self._armed = False
                self._state = "contact"
                self._reason = "Checking contact for up to five seconds; EEG signal is off."
                self._error = None
                self._owns_eeg = True
            precheck = None
            failure = None
            try:
                result = self.brainbit.start_acquisition(mode="contact")
                if self._error_of(result):
                    raise RuntimeError(self._error_of(result))
                deadline = time.monotonic() + self.CONTACT_SECONDS
                while not self._halt.is_set():
                    acquisition = self.brainbit.acquisition_snapshot()
                    if acquisition.get("state") == "error" or acquisition.get("error"):
                        raise RuntimeError(acquisition.get("error") or "Contact acquisition failed.")
                    precheck = copy.deepcopy(acquisition.get("contact_precheck")) or precheck
                    with self._lock:
                        self._eeg = self._eeg_metadata(acquisition)
                        self._observed_at = self.clock()
                        self._revision += 1
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._halt.wait(min(self.TICK_SECONDS, remaining))
            except Exception as exc:
                failure = "Contact check failed: " + str(exc)
            result = self._stop_owned(failure or "Contact precheck complete; signal remains off.", failed=bool(failure))
            with self._lock:
                if precheck:
                    self._eeg = {"state": "stopped", "contact_precheck": precheck}
                    self._observed_at = self.clock()
                elif not failure and not result.get("error"):
                    self._reason = "Contact check finished without a resistance reading."
            return self._result(result.get("error"))

    @staticmethod
    def _eeg_metadata(acquisition):
        return copy.deepcopy({key: acquisition.get(key) for key in
                              ("state", "mode", "session_id", "channels", "nominal_hz", "units", "stats", "contact_precheck")})

    @staticmethod
    def _age(value):
        return float(value) if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 else math.inf

    @staticmethod
    def _valid_row(row, width):
        values = row.get("samples") or []
        counter = row.get("counter")
        return (width > 0 and len(values) == width and isinstance(counter, int)
                and not isinstance(counter, bool)
                and all(isinstance(value, (int, float)) and math.isfinite(value) for value in values))

    def _tick(self):
        """Read cached device outputs once. Does not start capture or acquisition."""
        with self._lock:
            if self._state not in ("starting", "running") or self._halt.is_set():
                return
            generation = self._generation
            owns_camera = self._owns_camera
            mode = self._control_mode
        try:
            camera = self.camera.preview() if owns_camera else {}
        except Exception:
            with self._lock:
                if self._control_mode != "eeg":
                    raise
            camera = {}
        acquisition = self.brainbit.acquisition_snapshot()
        signal_reader = getattr(self.brainbit, "acquisition_signal_window", None)
        signal = signal_reader() if signal_reader else None
        now = self.clock()
        failure = None
        dispatch_direction = None
        arm_epoch = None
        direction = None
        with self._lock:
            if self._halt.is_set() or generation != self._generation or self._state not in ("starting", "running"):
                return
            self._revision += 1
            if self._eeg_armed and not self._guard_ready(now):
                self._disarm_eeg_locked("Global Escape guard expired; prepare the guard and explicitly rearm EEG control.")
            if (self._eeg_armed and self._eeg_uncertain_since is not None
                    and now - self._eeg_uncertain_since >= self.EEG_TRANSITION_SECONDS):
                self._disarm_eeg_locked("Prediction stayed uncertain for three seconds; explicitly rearm after recovery.")
            if self._eeg.get("session_id") and self._eeg.get("session_id") != acquisition.get("session_id"):
                self._armed = False
                self._arm_epoch += 1
                self._disarm_eeg_locked("Acquisition session changed; reset, retrain, and validate before EEG arming.")
                self._eeg_prediction = None
                self._eeg_prediction_session = None
                self._eeg_pending = None
                self._last_valid_counter = None
            if now - self._observed_at > self.TICK_SECONDS * 3 and (self._armed or self._eeg_armed):
                self._armed = False
                self._disarm_eeg_locked("Preview processing paused; explicitly rearm after fresh predictions.")
                self._arm_epoch += 1
                self._motion.reset()
            self._camera = copy.deepcopy(camera)
            self._eeg = self._eeg_metadata(acquisition)
            self._observed_at = now
            sequence = camera.get("sequence")
            new_frame = sequence is not None and sequence != self._camera_sequence
            if new_frame:
                self._camera_sequence = sequence
                self._camera_changed_at = now
            camera_age = max(self._age(camera.get("age_ms")) / 1000, now - self._camera_changed_at)
            camera_fresh = bool(camera.get("running")) and camera_age <= self.CAMERA_FRESH_SECONDS
            if camera_fresh:
                # Seeing a hand is not required to finish camera startup.
                self._camera_ready = True
                self._last_camera_fresh_at = now - camera_age
            points = camera.get("landmarks") or []
            tracked = camera_fresh and bool(camera.get("tracked")) and len(points) >= 21
            stats = acquisition.get("stats") or {}
            eeg_age = self._age(stats.get("age_seconds"))
            rows = acquisition.get("samples") or []
            width = len(acquisition.get("channels") or [])
            advanced = False
            for row in rows:
                if not self._valid_row(row, width):
                    continue
                counter = row["counter"]
                delta = ((counter - self._last_valid_counter) & 0xFFFFFFFF) if self._last_valid_counter is not None else 1
                if 0 < delta < 0x80000000:
                    self._last_valid_counter = counter
                    self._last_valid_eeg_at = now
                    advanced = True
            valid_eeg_age = max(eeg_age, now - self._last_valid_eeg_at)
            eeg_live = (acquisition.get("state") == "running" and acquisition.get("mode") == "signal"
                        and valid_eeg_age <= self.EEG_FRESH_SECONDS and bool(rows)
                        and self._valid_row(rows[-1], width))
            if eeg_live:
                self._last_eeg_live_at = now - valid_eeg_age
            # A poll without a new batch must disarm controls, but is not itself
            # evidence that acquisition stopped while the last batch is fresh.
            eeg_fresh = eeg_live and advanced
            gaps = sum(max(0, int(stats.get(key) or 0)) for key in
                       ("gaps", "queue_drops", "channel_mismatches", "duplicates", "nonfinite"))
            gap = gaps > self._gap_count
            self._gap_count = gaps
            self._healthy = bool(tracked and eeg_fresh and not gap)
            self._eeg_signal_healthy = bool(eeg_live and not gap)
            if not self._eeg_signal_healthy:
                self._disarm_eeg_locked("Fresh finite EEG without packet gaps is required.")
            elif self._control_mode == "eeg":
                self._state = "running"
            self._waveform = copy.deepcopy(rows[-250:]) if eeg_fresh else []
            for row in rows:
                timestamp = row.get("estimated_monotonic")
                if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
                    continue
                key = (acquisition.get("session_id"), row.get("counter"))
                if key in self._seen_samples:
                    continue
                if len(self._seen_order) == self._seen_order.maxlen:
                    self._seen_samples.discard(self._seen_order.popleft())
                self._seen_order.append(key)
                self._seen_samples.add(key)
                self._raw.append((timestamp, list(row.get("samples") or [])))
            while self._raw and now - self._raw[0][0] > 5.0:
                self._raw.popleft()
            observation_healthy = self._healthy or (tracked and self._eeg_signal_healthy
                                                    and (self._eeg_pending or self._control_mode == "eeg"))
            if not observation_healthy:
                webcam_was_armed = self._armed
                self._armed = False
                if webcam_was_armed or self._control_mode == "webcam":
                    self._arm_epoch += 1
                self._motion.reset()
                if self._pending:
                    self._last_trial_result = "Trial discarded: camera/EEG dropout or packet gap."
                    self._pending = None
                if gap:
                    self._raw.clear()
                self._reason = ("Packet gap detected; controls disarmed." if gap else
                                "Waiting for the camera's first fresh frame; controls disarmed."
                                if not self._camera_ready else
                                "Waiting for a fresh tracked hand and EEG; controls disarmed.")
            else:
                self._state = "running"
                self._reason = "Armed: mirrored camera direction, with EEG freshness gate." if self._armed else "Preview only; controls are disarmed."
                if new_frame:
                    # Palm landmarks avoid fingertip flexion looking like travel.
                    palm = [points[i] for i in (0, 5, 9, 13, 17)]
                    x = sum(float(p[0]) for p in palm) / len(palm)
                    y = sum(float(p[1]) for p in palm) / len(palm)
                    direction = self._motion.update(x, y, now)
                if direction and self._pending:
                    if direction == self._pending["label"]:
                        self._pending["motion_seen"] += 1
                    else:
                        self._pending["wrong_motion"] = True
                if direction and self._control_mode == "webcam" and self._armed and self.dispatch is not None:
                    dispatch_direction, arm_epoch = direction, self._arm_epoch
                self._finish_trial(now)
            self._observe_eeg_trial_locked(points, direction, new_frame, tracked, now)
            if self._control_mode == "eeg":
                self._reason = ("EEG control armed; camera movement does not supply directions." if self._eeg_armed
                                else "EEG prediction only; desktop actions are disarmed.")
            elapsed = now - self._started_at
            if acquisition.get("state") == "error" or acquisition.get("error"):
                failure = "EEG acquisition failed: " + str(acquisition.get("error") or "device error")
            elif self._control_mode == "webcam" and not camera.get("running") and camera.get("state") in ("error", "unavailable", "off", "stopping"):
                # preview() already exposes the capture worker's cached reason;
                # do not replace model/startup failures with a generic timeout.
                failure = "Camera stopped: " + str(camera.get("hint") or camera.get("error")
                                                   or camera.get("state"))
            elif not eeg_live and (
                    (not math.isfinite(self._last_eeg_live_at) and elapsed >= self.STARTUP_GRACE_SECONDS)
                    or (math.isfinite(self._last_eeg_live_at) and now - self._last_eeg_live_at >= self.EEG_LOSS_SECONDS)):
                failure = "EEG stopped or samples stayed stale; preview stopped."
            elif self._control_mode == "webcam" and not self._camera_ready and now - self._camera_started_at >= self.CAMERA_STARTUP_SECONDS:
                failure = "Camera startup timed out before the first fresh frame; preview stopped."
            elif self._control_mode == "webcam" and self._camera_ready and not camera_fresh and now - self._last_camera_fresh_at >= self.CAMERA_LOSS_SECONDS:
                failure = "Camera frames stayed stale; preview stopped."
        if failure:
            self._fail(failure, generation)
        else:
            if dispatch_direction:
                self._dispatch_direction(dispatch_direction, generation, arm_epoch)
            self._process_eeg(signal, now, generation)

    def _dispatch_direction(self, direction, generation, arm_epoch):
        # Never hold the status lock across an OS action. Epoch checks reject an
        # event whose ownership/arming changed before it reached this boundary.
        with self._action_lock:
            with self._lock:
                if (not self._armed or not self._can_arm(self.clock())
                        or self._control_mode != "webcam"
                        or generation != self._generation or arm_epoch != self._arm_epoch
                        or self.clock() - self._last_dispatch_at < self._motion.settings["cooldown_seconds"]):
                    return
                self._last_dispatch_at = self.clock()
            try:
                result = self.dispatch("os_switch_desktop", {"direction": direction}, actor="gesture")
                if isinstance(result, dict) and (result.get("error") or result.get("needs_confirmation")):
                    raise RuntimeError(result.get("error") or "Action requires confirmation")
            except Exception as exc:
                with self._lock:
                    self._armed = False
                    self._arm_epoch += 1
                    self._reason = "Action failed; controls disarmed: " + str(exc)

    def arm(self, enabled):
        with self._lock:
            if not isinstance(enabled, bool):
                return self._result("Arm requires a boolean enabled value.")
            self._armed = False
            self._retire_eeg_guard_locked()
            self._disarm_eeg_locked("Webcam arming changed; EEG control is disarmed.")
            self._arm_epoch += 1
            self._motion.reset()
            if enabled:
                if self._control_mode != "webcam":
                    return self._result("Select Webcam mode before arming webcam control.")
                if self.dispatch is None:
                    return self._result("Desktop dispatch is unavailable.")
                if not self._can_arm(self.clock()):
                    return self._result("Arming requires fresh tracked camera and EEG, with no trial in progress.")
                self._armed = True
                self._reason = "Armed. Hold still briefly, then move left or right in the mirrored preview."
            else:
                self._reason = "Controls disarmed."
            return self._result()

    def _can_arm(self, now):
        elapsed = max(0.0, now - self._observed_at)
        camera_age = max(self._age(self._camera.get("age_ms")) / 1000 + elapsed,
                         now - self._camera_changed_at)
        eeg_age = self._age((self._eeg.get("stats") or {}).get("age_seconds")) + elapsed
        return (self._control_mode == "webcam" and self._state == "running" and self._healthy and not self._pending and not self._eeg_pending
                and not self._halt.is_set() and elapsed <= self.TICK_SECONDS * 3
                and camera_age <= self.CAMERA_FRESH_SECONDS and eeg_age <= self.EEG_FRESH_SECONDS
                and now - self._last_valid_eeg_at <= self.EEG_FRESH_SECONDS)

    def _reset_eeg_debounce(self):
        self._eeg_neutral_since = None
        self._eeg_neutral_ready = False
        self._eeg_candidate = None
        self._eeg_candidate_count = 0

    def _disarm_eeg_locked(self, reason, retire_guard=True):
        if self._eeg_armed:
            self._arm_epoch += 1
        if retire_guard:
            self._retire_eeg_guard_locked()
        self._eeg_armed = False
        self._eeg_reason = reason
        self._eeg_uncertain_since = None
        self._reset_eeg_debounce()

    def set_control_mode(self, mode):
        with self._lock:
            if mode not in ("webcam", "eeg"):
                return self._result("Control mode must be webcam or eeg.")
            self._armed = False
            self._retire_eeg_guard_locked()
            self._arm_epoch += 1
            self._decoder_epoch += 1
            self._disarm_eeg_locked("Mode changed; explicitly arm the selected control mode.")
            self._motion.reset()
            if mode == "webcam" and self._state in ("starting", "running") and not self._owns_camera:
                return self._result("Stop the EEG-only preview before selecting Webcam mode and restarting for camera-guided trials.")
            self._control_mode = mode
            self._pending = None
            self._eeg_pending = None
            self._reason = "Selected " + mode + " mode. Both controls are disarmed."
            return self._result()

    def _eeg_inputs_fresh(self, now):
        elapsed = max(0.0, now - self._observed_at)
        eeg_age = self._age((self._eeg.get("stats") or {}).get("age_seconds")) + elapsed
        return (self._state == "running" and self._eeg_signal_healthy and not self._halt.is_set()
                and elapsed <= self.TICK_SECONDS * 3 and eeg_age <= self.EEG_FRESH_SECONDS
                and now - self._last_valid_eeg_at <= self.EEG_FRESH_SECONDS)

    def _guard_ready(self, now):
        return bool(self._eeg_guard_token and now < self._eeg_guard_until)

    def _retire_eeg_guard_locked(self):
        # Never evict: a delayed request must not resurrect an old shortcut
        # lease. At the bounded limit, new EEG arming fails closed until restart.
        if self._eeg_guard_token and len(self._eeg_retired_guard_tokens) < self.MAX_RETIRED_GUARDS:
            self._eeg_retired_guard_tokens.add(self._eeg_guard_token)
        self._eeg_guard_token = None
        self._eeg_guard_until = -math.inf

    def eeg_guard(self, token):
        """Renew a short main-process Escape lease; this never arms or starts anything."""
        with self._lock:
            if not isinstance(token, str) or not token or len(token) > 128:
                return self._result("A valid Escape guard token is required.")
            if token in self._eeg_retired_guard_tokens:
                return self._result("This Escape guard token was revoked. Prepare a new global Escape guard.")
            if len(self._eeg_retired_guard_tokens) >= self.MAX_RETIRED_GUARDS:
                return self._result("Escape guard retirement capacity reached; restart the backend before preparing another guard.")
            now = self.clock()
            if self._eeg_guard_token and (token != self._eeg_guard_token or not self._guard_ready(now)):
                self._disarm_eeg_locked("Escape guard changed or expired; explicitly rearm EEG control.")
                if token in self._eeg_retired_guard_tokens:
                    return self._result("This Escape guard token expired and was revoked. Prepare a new global Escape guard.")
                if len(self._eeg_retired_guard_tokens) >= self.MAX_RETIRED_GUARDS:
                    return self._result("Escape guard retirement capacity reached; restart the backend before preparing another guard.")
            self._eeg_guard_token = token
            self._eeg_guard_until = now + self.EEG_GUARD_SECONDS
            return self._result()

    def _can_eeg_arm(self, now, require_guard=False):
        prediction = self._eeg_prediction or {}
        return bool(self._control_mode == "eeg" and self.dispatch is not None
                    and (not require_guard or self._guard_ready(now))
                    and not self._eeg_pending and not self._pending and self._eeg_inputs_fresh(now)
                    and self._decoder_status.get("trained") and self._decoder_status.get("arm_eligible")
                    and self._decoder_status.get("passed_validation")
                    and self._eeg_prediction_session == self._eeg.get("session_id")
                    and now - self._eeg_prediction_at <= self.EEG_PREDICTION_HOP * 2
                    and self._confident_eeg_prediction(prediction))

    def _eeg_control_authorized(self, now):
        """Show retained authorization during a brief valid-raw model transition."""
        if not self._eeg_armed:
            return False
        if self._can_eeg_arm(now, require_guard=True):
            return True
        return bool(self._control_mode == "eeg" and self._guard_ready(now) and self._eeg_inputs_fresh(now)
                    and not self._eeg_pending and not self._pending and self._eeg_raw_window_valid
                    and self._decoder_status.get("trained") and self._decoder_status.get("arm_eligible")
                    and self._decoder_status.get("passed_validation")
                    and self._eeg_prediction_session == self._eeg.get("session_id")
                    and now - self._eeg_prediction_at <= self.EEG_PREDICTION_HOP * 2
                    and self._eeg_uncertain_since is not None
                    and now - self._eeg_uncertain_since < self.EEG_TRANSITION_SECONDS)

    @classmethod
    def _confident_eeg_prediction(cls, prediction):
        confidence, margin = prediction.get("confidence"), prediction.get("margin")
        return (prediction.get("valid") is True and not prediction.get("ood")
                and prediction.get("label") in ("left", "right", "rest")
                and isinstance(confidence, (int, float)) and math.isfinite(confidence) and confidence >= cls.EEG_CONFIDENCE
                and isinstance(margin, (int, float)) and math.isfinite(margin) and margin >= cls.EEG_MARGIN)

    def eeg_arm(self, enabled, guard_token=None):
        with self._lock:
            if not isinstance(enabled, bool):
                return self._result("EEG arm requires a boolean enabled value.")
            self._armed = False
            self._arm_epoch += 1
            if not enabled:
                self._retire_eeg_guard_locked()
            self._disarm_eeg_locked("EEG control disarmed.", retire_guard=not enabled)
            self._motion.reset()
            if enabled:
                if guard_token != self._eeg_guard_token or not self._guard_ready(self.clock()):
                    return self._result("A live global Escape guard is required before arming EEG control.")
                if not self._can_eeg_arm(self.clock(), require_guard=True):
                    self._disarm_eeg_locked("EEG arming prerequisites changed; prepare a new Escape guard after recovery.")
                    return self._result("EEG arming requires EEG mode, a frozen model that passed separate validation, and a fresh confident prediction from valid EEG in the same acquisition session.")
                self._eeg_armed = True
                self._eeg_reason = "Armed. Rest for at least one second, then sustain a confident EEG direction; camera direction is ignored."
            return self._result()

    def eeg_trial(self, label, phase="train"):
        with self._lock:
            if label not in ("left", "right", "rest") or phase not in ("train", "validate"):
                return self._result("EEG trial requires left, right, or rest and train or validate phase.")
            now = self.clock()
            if self._pending or self._eeg_pending or now < self._eeg_last_trial_end:
                return self._result("Finish the current trial window before recording another trial.")
            camera_age = max(self._age(self._camera.get("age_ms")) / 1000 + max(0, now - self._observed_at),
                             now - self._camera_changed_at)
            if (not self._eeg_inputs_fresh(now) or not self._camera.get("tracked") or not self._owns_camera
                    or camera_age > self.CAMERA_FRESH_SECONDS):
                return self._result("Guided trials require fresh tracked camera and EEG. Start in Webcam mode to collect trials.")
            if phase == "train" and self._decoder_status.get("trained"):
                return self._result("The model is frozen. Collect validation trials or reset before training again.")
            if phase == "validate" and not self._decoder_status.get("trained"):
                return self._result("Freeze a trained model before collecting separate validation trials.")
            if phase == "validate" and self._decoder_status.get("state") in ("validated", "validation_failed"):
                return self._result("Held-out validation is complete and frozen. Reset to begin a new experiment.")
            self._armed = False
            self._arm_epoch += 1
            self._retire_eeg_guard_locked()
            self._disarm_eeg_locked("Guided EEG trial; all desktop actions are disarmed.")
            self._motion.reset()
            self._eeg_trial_sequence += 1
            self._eeg_pending = {"label": label, "phase": phase, "start": now, "end": now + self.TRIAL_SECONDS,
                                 "id": "eeg-trial-" + str(self._eeg_trial_sequence), "motion_seen": 0,
                                 "motion_at": None,
                                 "wrong_motion": False, "min_x": math.inf, "max_x": -math.inf,
                                 "min_y": math.inf, "max_y": -math.inf, "frames": 0,
                                 "generation": self._generation, "decoder_epoch": self._decoder_epoch}
            self._eeg_last_trial_end = now + self.TRIAL_SECONDS
            self._eeg_last_result = ("Hold the hand still for three seconds." if label == "rest" else
                                     "Hold still for half a second, then move " + label + " once before the final half-second of this three-second trial.")
            return self._result()

    def _observe_eeg_trial_locked(self, points, direction, new_frame, tracked, now):
        trial = self._eeg_pending
        if not trial:
            return
        if not tracked or not self._eeg_signal_healthy:
            self._eeg_pending = None
            self._eeg_last_result = "EEG trial discarded: camera or EEG became invalid."
            return
        if new_frame:
            palm = [points[i] for i in (0, 5, 9, 13, 17)]
            x, y = (sum(float(p[axis]) for p in palm) / len(palm) for axis in (0, 1))
            if not math.isfinite(x) or not math.isfinite(y):
                self._eeg_pending = None
                self._eeg_last_result = "EEG trial discarded: invalid camera landmarks."
                return
            trial["min_x"], trial["max_x"] = min(trial["min_x"], x), max(trial["max_x"], x)
            trial["min_y"], trial["max_y"] = min(trial["min_y"], y), max(trial["max_y"], y)
            trial["frames"] += 1
        if direction:
            if direction == trial["label"]:
                trial["motion_seen"] += 1
                trial["motion_at"] = now
            else:
                trial["wrong_motion"] = True

    def eeg_train(self):
        with self._lock:
            self._armed = False
            self._arm_epoch += 1
            self._decoder_epoch += 1
            self._retire_eeg_guard_locked()
            self._disarm_eeg_locked("Training freezes the EEG model; separate validation is required before arming.")
            if self._pending or self._eeg_pending:
                return self._result("Finish the current trial before training.")
        with self._decoder_lock:
            result = self._decoder.train()
            decoder_status = self._decoder.status()
            with self._lock:
                self._decoder_status = decoder_status
                self._eeg_last_result = result.get("error") or "Model frozen. Collect eight fresh held-out validation trials per class."
                self._eeg_prediction = None
                self._eeg_prediction_at = -math.inf
                return self._result(result.get("error"))

    def eeg_reset(self):
        with self._lock:
            self._armed = False
            self._arm_epoch += 1
            self._decoder_epoch += 1
            self._retire_eeg_guard_locked()
            self._disarm_eeg_locked("EEG training and validation reset.")
            self._eeg_pending = None
            self._eeg_prediction = None
            self._eeg_prediction_at = -math.inf
            self._eeg_full_rows = []
            self._eeg_last_trial_end = -math.inf
            self._eeg_last_result = "EEG model, training features, and validation cleared."
        with self._decoder_lock:
            self._decoder.reset()
            with self._lock:
                self._decoder_status = self._decoder.status()
                return self._result()

    def _process_eeg(self, signal, now, generation):
        """Only full-rate EEG enters the decoder. Camera is used solely to label trials."""
        with self._lock:
            if generation != self._generation or self._halt.is_set():
                return
            if (not signal or signal.get("state") != "running" or signal.get("mode") != "signal"
                    or signal.get("session_id") != self._eeg.get("session_id")):
                self._eeg_full_rows = []
                self._eeg_prediction = None
                self._disarm_eeg_locked("Full-rate EEG from the current acquisition session is unavailable.")
                if self._eeg_pending:
                    self._eeg_pending = None
                    self._eeg_last_result = "EEG trial discarded: full-rate current-session samples unavailable."
                return
            self._eeg_full_rows = copy.deepcopy((signal.get("samples") or [])[-1250:])
            if not self._eeg_signal_healthy:
                self._eeg_full_rows = []
                self._eeg_prediction = None
                return
            trial = copy.deepcopy(self._eeg_pending) if self._eeg_pending and now >= self._eeg_pending["end"] else None
            if trial:
                self._eeg_pending = None
            predict_due = now - self._eeg_last_predict_at >= self.EEG_PREDICTION_HOP
            if predict_due:
                self._eeg_last_predict_at = now
            rows = self._eeg_full_rows
            channels = copy.deepcopy(signal.get("channels") or [])
            hz = signal.get("nominal_hz")
            context = {"session_id": signal.get("session_id")}
            decoder_epoch = self._decoder_epoch
        if trial:
            still = (trial["frames"] >= 3 and trial["max_x"] - trial["min_x"] <= self._motion.settings["deadband"]
                     and trial["max_y"] - trial["min_y"] <= self._motion.settings["deadband"])
            verified = (still and not trial["wrong_motion"] if trial["label"] == "rest" else
                        trial["motion_seen"] == 1 and not trial["wrong_motion"])
            if not verified:
                with self._lock:
                    self._eeg_last_result = "EEG trial discarded: camera did not verify the requested " + trial["label"] + " movement/rest."
            else:
                # Counters define the 2s window. Host batch jitter must not
                # turn 500 valid packets into 499 via an exact timestamp cutoff.
                size = round(hz * self.EEG_WINDOW_SECONDS) if isinstance(hz, (int, float)) and math.isfinite(hz) else 0
                trial_rows = [r for r in rows if trial["start"] + 0.5 <= r.get("estimated_monotonic", -math.inf) < trial["end"]][:size]
                extracted = extract_features(trial_rows, channels, hz, session_context=context)
                window = extracted.get("window") or {}
                if extracted.get("ok") and (window.get("end", math.inf) > trial["end"]
                        or (trial["label"] != "rest" and not window.get("start", math.inf) <= trial["motion_at"] < window.get("end", -math.inf))):
                    extracted = {"ok": False, "reason": "Observed movement was outside the selected two-second EEG trial window; move after the initial still period and before the final half-second."}
                with self._decoder_lock:
                    with self._lock:
                        current = generation == self._generation and decoder_epoch == self._decoder_epoch and not self._halt.is_set()
                    if current:
                        result = (self._decoder.add_trial(trial["label"], extracted, phase=trial["phase"],
                                                          trial_id=trial["id"], session_context=context)
                                  if extracted.get("ok") else {"error": extracted.get("reason") or extracted.get("error") or "Invalid EEG feature window."})
                        with self._lock:
                            self._decoder_status = self._decoder.status()
                            self._eeg_last_result = result.get("error") or ("Accepted " + trial["phase"] + " " + trial["label"] + " trial; features remain in memory only.")
                            self._revision += 1
        if not predict_due:
            return
        sample_count = round(hz * self.EEG_WINDOW_SECONDS) if isinstance(hz, (int, float)) and math.isfinite(hz) and hz > 0 else 0
        extracted = (extract_features(rows[-sample_count:], channels, hz, session_context=context)
                     if sample_count and len(rows) >= sample_count else
                     {"ok": False, "reason": "Need a complete two-second full-rate EEG window."})
        with self._decoder_lock:
            prediction = (self._decoder.predict(extracted) if extracted.get("ok") else
                          {"label": None, "valid": False, "confidence": 0.0, "margin": 0.0,
                           "ood": False, "window_id": None, "reason": extracted.get("reason") or extracted.get("error")})
            decoder_status = self._decoder.status()
        dispatch_direction = None
        with self._lock:
            if generation != self._generation or decoder_epoch != self._decoder_epoch or self._halt.is_set():
                return
            self._decoder_status = decoder_status
            self._eeg_prediction = copy.deepcopy(prediction)
            self._eeg_prediction_at = now
            self._eeg_prediction_session = context["session_id"]
            self._eeg_raw_window_valid = bool(extracted.get("ok"))
            self._revision += 1
            window_id = prediction.get("window_id")
            if not self._confident_eeg_prediction(prediction):
                transition = self._eeg_raw_window_valid and (prediction.get("valid") is True or prediction.get("ood") is True)
                if transition and self._eeg_armed:
                    if self._eeg_uncertain_since is None:
                        self._eeg_uncertain_since = now
                    self._eeg_candidate = None
                    self._eeg_candidate_count = 0
                    self._eeg_neutral_since = None
                    if now - self._eeg_uncertain_since >= self.EEG_TRANSITION_SECONDS:
                        self._disarm_eeg_locked("Prediction stayed uncertain for three seconds; explicitly rearm after recovery.")
                    else:
                        self._eeg_reason = "Model transition: actions suppressed until three new confident direction predictions; authorization expires after three seconds of uncertainty."
                else:
                    self._disarm_eeg_locked(prediction.get("reason") or "Invalid EEG prediction; explicitly rearm after recovery.")
                return
            self._eeg_uncertain_since = None
            if not window_id or window_id == self._eeg_last_prediction_id:
                return
            self._eeg_last_prediction_id = window_id
            if not self._eeg_armed or self._control_mode != "eeg":
                self._reset_eeg_debounce()
                self._eeg_reason = "Prediction only. Explicit EEG arming is required for desktop actions."
                return
            if not self._can_eeg_arm(now, require_guard=True):
                self._disarm_eeg_locked("Validation, current-session prediction, or signal freshness no longer permits EEG control.")
                return
            label = prediction["label"]
            if label == "rest":
                self._eeg_candidate = None
                self._eeg_candidate_count = 0
                if self._eeg_neutral_since is None:
                    self._eeg_neutral_since = now
                if now - self._eeg_neutral_since >= self.EEG_REST_SECONDS:
                    self._eeg_neutral_ready = True
                self._eeg_reason = "Rest confirmed; ready for a sustained EEG direction." if self._eeg_neutral_ready else "Keep resting until one second of confident rest predictions is complete."
            else:
                self._eeg_neutral_since = None
                if not self._eeg_neutral_ready:
                    self._eeg_candidate = None
                    self._eeg_candidate_count = 0
                    self._eeg_reason = "Rest for at least one second before another EEG direction."
                    return
                self._eeg_candidate_count = self._eeg_candidate_count + 1 if self._eeg_candidate == label else 1
                self._eeg_candidate = label
                if self._eeg_candidate_count >= 3 and now - self._last_dispatch_at >= self.EEG_COOLDOWN_SECONDS:
                    dispatch_direction = label
            arm_epoch = self._arm_epoch
        if dispatch_direction:
            self._dispatch_eeg_direction(dispatch_direction, generation, arm_epoch, window_id)

    def _dispatch_eeg_direction(self, direction, generation, arm_epoch, prediction_id):
        """The only EEG actuator boundary; no camera input is consulted here."""
        with self._action_lock:
            with self._lock:
                now = self.clock()
                prediction = self._eeg_prediction or {}
                if (direction not in ("left", "right") or not self._eeg_armed or not self._can_eeg_arm(now, require_guard=True)
                        or generation != self._generation or arm_epoch != self._arm_epoch
                        or prediction_id != prediction.get("window_id") or prediction.get("label") != direction
                        or not self._eeg_neutral_ready or self._eeg_candidate_count < 3
                        or self._eeg_candidate != direction or now - self._last_dispatch_at < self.EEG_COOLDOWN_SECONDS):
                    return
                self._last_dispatch_at = now
                self._reset_eeg_debounce()
                self._eeg_reason = "EEG direction dispatched. Rest for at least one second before another direction."
                self._revision += 1
            try:
                result = self.dispatch("os_switch_desktop", {"direction": direction}, actor="gesture")
                if isinstance(result, dict) and (result.get("error") or result.get("needs_confirmation")):
                    raise RuntimeError(result.get("error") or "Action requires confirmation")
            except Exception as exc:
                with self._lock:
                    self._disarm_eeg_locked("EEG action failed: " + str(exc))
                    self._revision += 1

    def mark_trial(self, label):
        with self._lock:
            if label not in ("left", "right"):
                return self._result("Trial label must be left or right.")
            if self._pending:
                return self._result("Finish the current three-second trial first.")
            if len(self._trials) >= self.MAX_TRIALS:
                return self._result("Calibration is full (64 trials). Reset to begin again.")
            now = self.clock()
            if not self._can_arm(now):
                return self._result("A trial requires fresh tracked camera and EEG.")
            if now < self._last_trial_end:
                return self._result("Wait for the previous trial window to end.")
            self._armed = False
            self._arm_epoch += 1
            self._motion.reset()
            self._pending = {"label": label, "start": now, "end": now + self.TRIAL_SECONDS,
                             "motion_seen": 0, "wrong_motion": False}
            self._last_trial_end = now + self.TRIAL_SECONDS
            self._last_trial_result = "Hold still briefly, then move " + label + " once during this three-second trial."
            return self._result()

    def _finish_trial(self, now):
        trial = self._pending
        if not trial or now < trial["end"]:
            return
        self._pending = None
        if trial["motion_seen"] != 1 or trial["wrong_motion"]:
            self._last_trial_result = "Trial discarded: expected one validated " + trial["label"] + " movement."
            return
        rows = [(at, values) for at, values in self._raw if trial["start"] <= at < trial["end"]]
        if len(rows) < 40 or rows[-1][0] - rows[0][0] < 1.0:
            self._last_trial_result = "Trial discarded: insufficient distinct EEG samples covering at least one second."
            return
        width = len(rows[0][1])
        if not width or any(len(values) != width or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values) for _, values in rows):
            self._last_trial_result = "Trial discarded: missing/nonfinite EEG channels."
            return
        features = [math.log(max(statistics.pvariance(values[i] for _, values in rows), 1e-24)) for i in range(width)]
        self._trial_id += 1
        self._trials.append({"id": self._trial_id, "label": trial["label"], "features": features,
                             "channel_signature": [(c.get("num"), c.get("name")) for c in self._eeg.get("channels") or []],
                             "start": trial["start"], "end": trial["end"], "n_samples": len(rows)})
        self._evaluation = self._evaluate(self._trials)
        self._last_trial_result = "Accepted " + trial["label"] + " trial; feature vector retained in memory only."

    @staticmethod
    def _evaluate(trials):
        split = int(len(trials) * 0.7)
        train, test = trials[:split], trials[split:]
        train_counts = {label: sum(t["label"] == label for t in train) for label in ("left", "right")}
        test_counts = {label: sum(t["label"] == label for t in test) for label in ("left", "right")}
        report = {"state": "insufficient_evidence", "reason": "Need at least six training and three held-out trials per direction in chronological 70/30 split.",
                  "n_train": len(train), "n_test": len(test), "train_counts": train_counts, "test_counts": test_counts,
                  "accuracy": None, "balanced_accuracy": None, "baseline_accuracy": None,
                  "balanced_baseline": 0.5, "disclaimer": DISCLAIMER,
                  "split": "First 70% of complete trials train; last 30% are held out. No overlapping windows or holdout tuning.",
                  "model": "Per-channel log variance; training-only standardization and nearest centroid."}
        if min(train_counts.values()) < 6 or min(test_counts.values()) < 3:
            return report
        width = len(train[0]["features"])
        if any(len(t["features"]) != width or t.get("channel_signature") != train[0].get("channel_signature") for t in trials):
            report["reason"] = "Channel layout changed. Reset calibration before comparing trials."
            return report
        means = [statistics.mean(t["features"][i] for t in train) for i in range(width)]
        scales = [max(statistics.pstdev(t["features"][i] for t in train), 1e-12) for i in range(width)]
        normalized = lambda t: [(t["features"][i] - means[i]) / scales[i] for i in range(width)]
        centroids = {label: [statistics.mean(normalized(t)[i] for t in train if t["label"] == label) for i in range(width)] for label in ("left", "right")}
        correct = {"left": 0, "right": 0}
        for trial in test:
            feature = normalized(trial)
            prediction = min(("left", "right"), key=lambda label: sum((a - b) ** 2 for a, b in zip(feature, centroids[label])))
            correct[trial["label"]] += int(prediction == trial["label"])
        majority = max(("left", "right"), key=lambda label: train_counts[label])
        report.update(state="evaluated", reason="Held-out trial result; exploratory movement correlations only.",
                      accuracy=sum(correct.values()) / len(test),
                      balanced_accuracy=sum(correct[label] / test_counts[label] for label in correct) / 2,
                      baseline_accuracy=test_counts[majority] / len(test))
        return report

    def reset_calibration(self):
        with self._lock:
            self._armed = False
            self._retire_eeg_guard_locked()
            self._disarm_eeg_locked("Calibration reset; EEG control is disarmed.")
            self._arm_epoch += 1
            self._pending = None
            self._trials.clear()
            self._raw.clear()
            self._last_trial_end = -math.inf
            self._last_trial_result = "Calibration reset."
            self._evaluation = self._evaluate([])
            self._motion.reset()
            return self._result()

    def status(self):
        with self._lock:
            now = self.clock()
            elapsed = max(0.0, now - self._observed_at)
            camera = {key: copy.deepcopy(self._camera.get(key)) for key in ("running", "tracked", "sequence", "age_ms", "fps")}
            if isinstance(camera.get("age_ms"), (int, float)):
                camera["age_ms"] += round(elapsed * 1000)
            eeg = copy.deepcopy(self._eeg)
            stats = eeg.get("stats")
            if isinstance(stats, dict) and isinstance(stats.get("age_seconds"), (int, float)):
                stats["age_seconds"] += elapsed
            precheck = eeg.get("contact_precheck")
            if isinstance(precheck, dict):
                received = precheck.get("host_received_monotonic")
                if isinstance(received, (int, float)):
                    precheck["age_seconds"] = max(0, now - received)
            pending = ({"label": self._pending["label"], "remaining_seconds": round(max(0, self._pending["end"] - now), 1)} if self._pending else None)
            eeg_pending = ({"label": self._eeg_pending["label"], "phase": self._eeg_pending["phase"],
                            "remaining_seconds": round(max(0, self._eeg_pending["end"] - now), 1)} if self._eeg_pending else None)
            return {"revision": self._revision, "state": self._state, "running": self._state in ("starting", "running"),
                    "cleanup_pending": (self._owns_camera or self._owns_eeg) and self._state in ("error", "stopped", "stopping"),
                    "armed": self._armed and self._can_arm(now), "can_arm": self._can_arm(now),
                    "reason": self._reason, "error": self._error,
                    "control_mode": self._control_mode,
                    "eeg_control": {"armed": self._eeg_control_authorized(now),
                                    "can_arm": self._can_eeg_arm(now), "reason": self._eeg_reason,
                                    "guard_ready": self._guard_ready(now),
                                    "prediction": copy.deepcopy(self._eeg_prediction),
                                    "decoder": copy.deepcopy(self._decoder_status), "pending": eeg_pending,
                                    "neutral_ready": self._eeg_neutral_ready,
                                    "last_result": self._eeg_last_result},
                    "camera": camera, "eeg": eeg, "motion": self._motion.status(now),
                    "calibration": {"total": len(self._trials),
                                    "counts": {label: sum(t["label"] == label for t in self._trials) for label in ("left", "right")},
                                    "pending": pending, "last_result": self._last_trial_result,
                                    "evaluation": copy.deepcopy(self._evaluation)},
                    "timing_note": DISCLAIMER}

    def snapshot(self):
        with self._lock:
            result = self.status()
            if self._state in ("starting", "running"):
                for key in ("image", "landmarks", "width", "height"):
                    result["camera"][key] = copy.deepcopy(self._camera.get(key))
                result["eeg"]["waveform"] = copy.deepcopy(self._waveform)
            return result


PreviewSession = MultimodalPreview
