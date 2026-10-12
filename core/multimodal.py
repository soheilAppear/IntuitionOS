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
from .eeg_decoder import CLICK_FEATURE_PROFILE, EegDecoder, extract_features  # Share the decoder's fixed click transform identifier.
from .eeg_blink import BlinkDecoder  # The short waveform model stays separate from the saved eyes-closed bandpower model.


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
    HAND_TICK_SECONDS = 1.0 / 30.0  # Poll hand input at 30 Hz without changing legacy freshness or EEG window timing.
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
    CLICK_FEATURE_PROFILE = CLICK_FEATURE_PROFILE  # Keep click calibration and inference on the decoder's versioned frequency features.
    CLICK_TRIAL_SECONDS = 5.0  # Allow one settling second, two measured seconds, and time to reopen the eyes.
    CLICK_PENDING_SECONDS = 8.0  # A selection must return to rest promptly rather than becoming an indefinitely queued click.
    BLINK_COLLECTION_GRACE = 1.5  # Allow bounded Bluetooth delivery delay after the five-second cue without accepting a partial sample window.

    def __init__(self, camera, brainbit, gesture_busy=lambda: False,
                 dispatch=None, clock=time.monotonic, click_controller=None, pointer_controller=None):
        self.camera = camera
        self.brainbit = brainbit
        self.gesture_busy = gesture_busy
        self.dispatch = dispatch
        self.clock = clock
        self.click_controller = click_controller  # This narrow actuator can only click at the existing pointer position.
        if pointer_controller is not None and getattr(pointer_controller, "eeg_compatible", False) is not True:
            raise ValueError("EEG aiming requires an EEG-compatible hand controller.")  # Require gesture ownership feedback before combining two click sources.
        self.pointer_controller = pointer_controller  # Hand clicks remain available alongside the separate EEG click actuator.
        self._pointer_status = {"state": "mouse_paused", "pointer_ready": False, "hint": "Arm hand aiming explicitly."}  # Preview starts without input.
        self._pointer_at = -math.inf  # No cached pointing gesture can authorize a click before a fresh update.
        self._pointer_sequence = None  # Consume each camera frame at most once when aiming.
        self._pointer_cleanup_pending = False  # Block restart if the controller cannot confirm it stopped.
        self._pointer_needs_cleanup = False  # Automatic disarm schedules owned-button release outside the state lock.
        self._pointer_engaged = False  # Retain tracking-loss protection while a legitimate finger gesture temporarily owns input.
        self._last_hand_click_id = 0  # Controller click IDs increase across resets; count each successful finger click once.
        self._finger_click_count = 0  # Keep hand clicks separate from EEG actions and EEG practice selections.
        self._click_target = "practice"  # New sessions begin with a counter that cannot affect another application.
        self._selection_count = 0  # Count completed practice selections separately from real mouse clicks.
        self._click_count = 0  # Report only mouse clicks whose press and release both completed successfully.
        self._click_cleanup_pending = False  # An unconfirmed button release must block another acquisition start.
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
        self._camera_display = {}  # JPEGs retain their own aligned landmarks while control uses newer inference results.
        self._eeg = {}
        self._waveform = []
        self._raw = deque(maxlen=1500)
        self._seen_samples = set()
        self._seen_order = deque(maxlen=2000)
        self._gap_count = 0
        self._last_valid_counter = None
        self._eeg_counter_reanchor = False  # A reported discontinuity needs a new watermark and then a later advancing packet.
        self._last_valid_eeg_at = -math.inf
        self._motion = MirroredMotionTracker()
        self._pending = None
        self._trials = []
        self._trial_id = 0
        self._last_trial_end = -math.inf
        self._last_trial_result = None
        self._evaluation = self._evaluate([])
        self._control_mode = "webcam"
        self._experimental_model = False  # Only explicit saved-model selection may permit EEG output despite its original failed validation.
        self._model_source = None  # Describe the selected archive without changing its retained validation evidence.
        self._hand_tracking_suspended = False  # Missing hands pause input once, then resume through a fresh pointing dwell.
        self._decoder = EegDecoder()
        self._eeg_method = "eyes_closed"  # Existing installations keep their original calibration and click timing.
        self._blink_calibration_source = None  # A resumed draft is evidence in progress, never an eligible trained model.
        self._click_profiles = {}  # Switching methods retains each decoder, archive source, and validation evidence in memory.
        self._blink_reset_serial = 0  # State changes invalidate a blink pair without taking the decoder lock in reverse order.
        self._blink_reset_pending = True  # The next stream update establishes a current-sample watermark before detecting events.
        self._blink_last_event_id = None  # Consume each event once, including events observed while disarmed or blocked.
        self._blink_dispatch_id = None  # Only an event approved by the current service update can reach the mouse boundary.
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
                if self._owns_camera or self._owns_eeg or self._click_cleanup_pending or self._pointer_cleanup_pending or self._pointer_needs_cleanup:
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
                self._reason = ("Starting camera and EEG; controls remain disarmed." if self._control_mode in ("webcam", "eeg_click")
                                else "Starting EEG; controls remain disarmed.")  # Camera-free modes acquire only EEG.
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
                        self._owns_camera = self._control_mode in ("webcam", "eeg_click")  # EEG clicks use this camera to aim with the hand.
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
            tick_started = self.clock()  # Include processing time in the faster hand-control polling interval.
            try:
                self._tick()
            except Exception as exc:
                if generation == self._generation:
                    self._fail("Preview failed: " + str(exc), generation)
                break
            delay = self.TICK_SECONDS  # Preserve the original worker cadence for webcam and advanced EEG modes.
            if self._control_mode == "eeg_click":  # Combined pointer input must not inherit the JPEG preview's eight-FPS cap.
                delay = max(0.0, self.HAND_TICK_SECONDS - max(0.0, self.clock() - tick_started))
            self._halt.wait(delay)  # Stop remains interruptible, including when a slow tick used the whole interval.

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
        with self._action_lock:  # Finish hand movement before acknowledging that the preview stopped.
            pointer_error = self._cleanup_hand_pointer()  # Only cleanup is allowed here; no camera frame is processed.
            if pointer_error:
                errors.append("Hand pointer cleanup unconfirmed: " + pointer_error)  # Keep Stop available for an explicit retry.
        if self.click_controller is not None:  # Always retry release cleanup, even after leaving click mode.
            with self._action_lock:  # Do not race a mouse release against an in-flight click.
                try:
                    cleanup = self.click_controller.close()  # Release only input owned by this controller.
                    cleanup_error = self._error_of(cleanup)  # Preserve any failure instead of claiming a clean stop.
                except Exception as exc:
                    cleanup_error = str(exc)  # Surface native failures through the existing stop response.
                self._click_cleanup_pending = bool(cleanup_error)  # Require a confirmed release before restarting.
                if cleanup_error:
                    errors.append("EEG click cleanup unconfirmed: " + cleanup_error)  # Keep Stop available for a retry.
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
        self._camera_display = {}  # Discard image-aligned frames together with the control snapshot on stop or restart.
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
        self._eeg_counter_reanchor = False
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
        """Read cached outputs and update freshness; never start either device here.

        The preview uses a small display buffer, while calibration and prediction
        use the separate full-rate signal window. Device sample counters establish
        progress; the arrival of another HTTP poll does not make old samples new.
        """
        with self._lock:
            if self._state not in ("starting", "running") or self._halt.is_set():
                return
            generation = self._generation
            owns_camera = self._owns_camera
            mode = self._control_mode
        try:
            camera_display = self.camera.preview() if owns_camera else {}  # Keep the HUD image and its overlay in one frame.
            tracking_reader = getattr(self.camera, "tracking_snapshot", None)  # Older adapters may expose only a preview.
            camera = (tracking_reader() if owns_camera and mode == "eeg_click" and callable(tracking_reader)
                      else camera_display)  # Read completed inference directly; cached JPEG sequences cannot throttle the pointer.
        except Exception:
            with self._lock:
                if self._control_mode != "eeg":  # Hand aiming requires a working camera; only advanced EEG swipes may ignore it.
                    raise
            camera = camera_display = {}
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
            if (self._control_mode != "eeg_click" and self._eeg_armed and self._eeg_uncertain_since is not None
                    and now - self._eeg_uncertain_since >= self.EEG_TRANSITION_SECONDS):
                self._disarm_eeg_locked("Prediction stayed uncertain for three seconds; explicitly rearm after recovery.")
            if self._eeg.get("session_id") and self._eeg.get("session_id") != acquisition.get("session_id"):
                self._armed = False
                self._arm_epoch += 1
                if self._control_mode == "eeg_click":
                    self._pause_eeg_locked("EEG session changed; select the saved model again before EEG clicking. Hand controls remain available.")  # Session identity still blocks the old model's predictions.
                else:
                    self._disarm_eeg_locked("Acquisition session changed; reset, retrain, and validate before EEG arming.")
                self._eeg_prediction = None
                self._eeg_prediction_session = None
                self._eeg_pending = None
                self._last_valid_counter = None
                self._eeg_counter_reanchor = False
            if now - self._observed_at > self.TICK_SECONDS * 3 and (self._armed or self._eeg_armed):
                self._armed = False
                if self._control_mode == "eeg_click":
                    self._pause_eeg_locked("Preview processing paused; point briefly to resume hand aiming.")  # The hand controller's shorter frame-gap limit releases and resets its dwell on this fresh update.
                else:
                    self._disarm_eeg_locked("Preview processing paused; explicitly rearm after fresh predictions.")
                self._arm_epoch += 1
                self._motion.reset()
            self._camera = copy.deepcopy(camera)
            self._camera_display = copy.deepcopy(camera_display)  # UI pixels never become the freshness source for hand input.
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
            gaps = sum(max(0, int(stats.get(key) or 0)) for key in  # Combine cumulative integrity counters into one discontinuity signal.
                       ("gaps", "queue_drops", "channel_mismatches", "duplicates", "nonfinite"))
            gap = gaps > self._gap_count  # An already reported fault must not restart recovery on every display refresh.
            self._gap_count = gaps  # Remember which faults this preview has already handled.
            if gap:
                self._eeg_counter_reanchor = True  # The adapter may have accepted a backward device-counter reset.
                self._last_valid_eeg_at = -math.inf  # Old valid samples cannot make the interrupted stream look fresh.
            rows = acquisition.get("samples") or []
            width = len(acquisition.get("channels") or [])
            advanced = False
            if rows and self._valid_row(rows[-1], width):
                # Only the newest packet describes current acquisition progress. A retained buffer
                # can still contain high pre-reset counters before the new low-valued sequence.
                counter = rows[-1]["counter"]
                delta = ((counter - self._last_valid_counter) & 0xFFFFFFFF) if self._last_valid_counter is not None else 1
                if self._eeg_counter_reanchor:
                    self._last_valid_counter = counter  # Establish the new starting point without calling this packet fresh yet.
                    self._eeg_counter_reanchor = False  # A malformed tail leaves this pending until an actual finite newest packet arrives.
                elif 0 < delta < 0x80000000:
                    self._last_valid_counter = counter  # Forward modulo-32-bit movement also accepts a normal counter wrap.
                    self._last_valid_eeg_at = now  # Update this clock only after observing actual packet progress.
                    advanced = True  # Legacy swipe decisions require a newly observed batch, not just a still-fresh cached batch.
            valid_eeg_age = max(eeg_age, now - self._last_valid_eeg_at)  # Either an old receipt or a stalled counter can make the stream stale.
            eeg_live = (acquisition.get("state") == "running" and acquisition.get("mode") == "signal"
                        and valid_eeg_age <= self.EEG_FRESH_SECONDS and bool(rows)
                        and self._valid_row(rows[-1], width))
            if eeg_live:
                self._last_eeg_live_at = now - valid_eeg_age
            # A poll without a new batch must disarm controls, but is not itself
            # evidence that acquisition stopped while the last batch is fresh.
            eeg_fresh = eeg_live and advanced
            self._healthy = bool(tracked and eeg_fresh and not gap)
            self._eeg_signal_healthy = bool(eeg_live and not gap)
            if not self._eeg_signal_healthy:
                if self._control_mode == "eeg_click":
                    self._pause_eeg_locked("EEG clicks paused until fresh finite samples return; hand controls remain available.")  # Finger input does not depend on EEG quality.
                else:
                    self._disarm_eeg_locked("Fresh finite EEG without packet gaps is required.")
            if self._control_mode == "eeg" and eeg_live or self._control_mode == "eeg_click" and camera_fresh:
                self._state = "running"
            if self._control_mode == "eeg_click" and self._eeg_armed and not self._hand_inputs_fresh(now):
                self._pause_eeg_locked("Hand tracking paused; show your hand and point briefly to resume.")  # Release owned finger input below without revoking the explicit arm.
            waveform_live = eeg_live if self._control_mode == "eeg_click" else eeg_fresh  # Faster hand polls may arrive between EEG batches; keep a still-fresh display stable.
            self._waveform = copy.deepcopy(rows[-250:]) if waveform_live else []  # Stale, stopped, missing, or invalid EEG still clears the display immediately.
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
            if self._control_mode in ("eeg", "eeg_click"):
                self._reason = ("EEG control armed; use the calibrated selection pattern." if self._eeg_armed
                                else "EEG prediction only; desktop actions are disarmed.")
            elapsed = now - self._started_at
            if self._control_mode != "eeg_click" and (acquisition.get("state") == "error" or acquisition.get("error")):
                failure = "EEG acquisition failed: " + str(acquisition.get("error") or "device error")
            elif self._control_mode in ("webcam", "eeg_click") and not camera.get("running") and camera.get("state") in ("error", "unavailable", "off", "stopping"):
                # preview() already exposes the capture worker's cached reason;
                # do not replace model/startup failures with a generic timeout.
                failure = "Camera stopped: " + str(camera.get("hint") or camera.get("error")
                                                   or camera.get("state"))
            elif self._control_mode != "eeg_click" and not eeg_live and (
                    (not math.isfinite(self._last_eeg_live_at) and elapsed >= self.STARTUP_GRACE_SECONDS)
                    or (math.isfinite(self._last_eeg_live_at) and now - self._last_eeg_live_at >= self.EEG_LOSS_SECONDS)):
                failure = "EEG stopped or samples stayed stale; preview stopped."
            elif self._control_mode in ("webcam", "eeg_click") and not self._camera_ready and now - self._camera_started_at >= self.CAMERA_STARTUP_SECONDS:
                failure = "Camera startup timed out before the first fresh frame; preview stopped."
            elif self._control_mode in ("webcam", "eeg_click") and self._camera_ready and not camera_fresh and now - self._last_camera_fresh_at >= self.CAMERA_LOSS_SECONDS:
                failure = "Camera frames stayed stale; preview stopped."
        if failure:
            self._fail(failure, generation)
        else:
            if dispatch_direction:
                self._dispatch_direction(dispatch_direction, generation, arm_epoch)
            self._update_hand_pointer(generation)  # Validate this frame and aim before EEG may complete a click on its target.
            self._process_eeg(signal, now, generation)
        self._finish_disarmed_hand_input()  # Every automatic revocation must release a finger drag before this tick completes.

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
        self._click_pending_at = None  # Clearing or disarming a gesture also discards its unfinished click.
        self._blink_reset_serial += 1  # A blink begun before a pause, finger action, or arm change cannot finish afterward.
        self._blink_reset_pending = True  # Apply the reset under the decoder lock on the next full-rate update.
        self._blink_dispatch_id = None  # Cancel any event waiting for final input authorization.

    def _pause_eeg_locked(self, reason):
        """Cancel EEG evidence without revoking independently authorized hand input."""
        self._reset_eeg_debounce()  # Recovery must establish a whole new rest/select/rest cycle.
        self._eeg_uncertain_since = None  # Paused EEG has no remaining three-second transition authorization.
        self._eeg_reason = reason  # Publish why EEG is idle without claiming finger control stopped.

    def _disarm_eeg_locked(self, reason, retire_guard=True):
        if self._eeg_armed:
            self._arm_epoch += 1
            self._pointer_needs_cleanup = self.pointer_controller is not None  # Native release waits until the caller has dropped the state lock.
        if retire_guard:
            self._retire_eeg_guard_locked()
        self._eeg_armed = False
        self._eeg_reason = reason
        self._eeg_uncertain_since = None
        self._reset_eeg_debounce()
        self._pointer_at = -math.inf  # Revoking EEG authorization revokes hand aiming at the same instant.
        self._pointer_sequence = None  # A new explicit arm must start from current camera observations.
        self._pointer_engaged = False  # Fresh pointing is required after any revocation.
        self._hand_tracking_suspended = False  # The next explicit arm begins a new hand tracking lifecycle.
        self._pointer_status = {"state": "mouse_paused", "pointer_ready": False, "finger_active": False,
                                "button_down": False, "hint": reason}  # Cleanup flags separately retain any unresolved native ownership.

    def set_control_mode(self, mode):
        error = None  # A rejected camera transition still needs to finish revoking any in-flight action.
        with self._decoder_lock, self._lock:  # Use the decoder's existing lock order when replacing its label set.
            if mode not in ("webcam", "eeg", "eeg_click"):
                return self._result("Control mode must be webcam, eeg, or eeg_click.")
            self._armed = False
            self._retire_eeg_guard_locked()
            self._arm_epoch += 1
            self._decoder_epoch += 1
            self._disarm_eeg_locked("Mode changed; explicitly arm the selected control mode.")
            self._motion.reset()
            if mode in ("webcam", "eeg_click") and self._state in ("starting", "running") and not self._owns_camera:
                error = "Stop the EEG-only preview before selecting a camera mode and restarting for hand control."
            elif (mode == "eeg_click") != (self._control_mode == "eeg_click"):  # Swipe evidence cannot authorize a binary click model.
                self._eeg_method = "eyes_closed"  # Legacy mode changes retain their original default; method selection is explicit.
                self._click_profiles = {}  # Method preservation applies within the combined click mode, not across swipe calibration.
                self._experimental_model = False  # Model overrides belong only to the explicitly selected binary model.
                self._model_source = None  # Leaving its mode ends any archive-specific authorization.
                self._decoder = (EegDecoder(labels=("rest", "select"), feature_profile=self.CLICK_FEATURE_PROFILE)
                                 if mode == "eeg_click" else EegDecoder())  # Start fresh when labels or features change.
                self._decoder_status = self._decoder.status()  # Publish the new two-state counts immediately.
                self._eeg_prediction = None  # Reject predictions produced by the previous decoder.
                self._eeg_prediction_at = -math.inf  # Wait for a new current-session feature window.
                self._eeg_last_prediction_id = None  # Let the replacement model evaluate the next window.
                self._eeg_last_trial_end = -math.inf  # No old calibration timing belongs to the new model.
                self._click_target = "practice"  # Returning to click mode never silently restores mouse output.
                self._selection_count = self._click_count = 0  # Counters describe only this click experiment.
                self._finger_click_count = 0  # The hand counter also starts with the new combined-mode experiment.
            if error is None:  # Keep the active mode unchanged when its required camera cannot be started here.
                self._control_mode = mode
                self._pending = None
                self._eeg_pending = None
                self._reason = "Selected " + mode + " mode. Both controls are disarmed."
        with self._action_lock:  # Revoke first, then finish any already dispatched input before acknowledging the change.
            pointer_error = self._cleanup_hand_pointer()  # Do not carry an engaged pointing gesture between control sources.
            return self._result(error or pointer_error)  # Waiting actions now see the revoked epoch and cannot issue input.

    def eeg_method(self, method):
        """Select a separate EEG click model without clearing either method's calibration."""
        with self._decoder_lock, self._lock:  # Keep model replacement serialized with training and streaming inference.
            if self._control_mode != "eeg_click" or method not in ("eyes_closed", "double_blink"):
                return self._result("Choose eyes_closed or double_blink in hand + EEG click mode.")
            if self._eeg_pending or self._pending:
                return self._result("Finish the current trial before changing EEG methods.")
            self._arm_epoch += 1  # Revoke queued mouse input before changing the active model.
            self._decoder_epoch += 1  # Predictions from the previous profile cannot be published after the switch.
            self._disarm_eeg_locked("EEG method changed; explicitly arm again after calibration and validation.")
            self._click_profiles[self._eeg_method] = (self._decoder, self._experimental_model,
                                                       self._model_source, self._eeg_last_trial_end,
                                                       self._eeg_last_result)  # Retain the original model and all of its evidence.
            if method not in self._click_profiles:
                decoder = (BlinkDecoder() if method == "double_blink" else
                           EegDecoder(labels=("rest", "select"), feature_profile=self.CLICK_FEATURE_PROFILE))
                self._click_profiles[method] = (decoder, False, None, -math.inf, None)
            (self._decoder, self._experimental_model, self._model_source,
             self._eeg_last_trial_end, self._eeg_last_result) = self._click_profiles[method]
            self._eeg_method = method  # Both profiles keep the same hand and finger-click controller.
            self._decoder_status = self._decoder.status()
            self._eeg_prediction = None  # Never reuse a prediction from a different classifier.
            self._eeg_prediction_at = self._eeg_last_predict_at = -math.inf
            self._eeg_prediction_session = self._eeg_last_prediction_id = None
            self._eeg_raw_window_valid = False
            self._blink_last_event_id = None
        with self._action_lock:  # Finish any already executing hand input before acknowledging the method switch.
            return self._result(self._cleanup_hand_pointer())

    def eeg_target(self, target):
        """Choose a practice counter or one mouse click; changing the destination always disarms."""
        with self._lock:  # Serialize destination changes with the final click authorization check.
            if self._control_mode != "eeg_click" or target not in ("practice", "mouse"):
                return self._result("Choose practice or mouse in EEG selection / click mode.")  # Reject arbitrary actuator names.
            if self._eeg_pending:
                return self._result("Finish the current calibration trial before changing the click target.")  # Keep trial instructions stable.
            self._arm_epoch += 1  # Invalidate any selection that was prepared for the old destination.
            self._disarm_eeg_locked("Click target changed; explicitly arm the selected target.")  # Revoke the old Escape lease too.
            self._click_target = target  # Remember only the destination explicitly chosen in the HUD.
        with self._action_lock:  # Wait for previously dispatched input after revoking queued selections immediately.
            return self._result(self._cleanup_hand_pointer())  # Changing the target also clears the old hand engagement.

    def _eeg_inputs_fresh(self, now):
        elapsed = max(0.0, now - self._observed_at)
        eeg_age = self._age((self._eeg.get("stats") or {}).get("age_seconds")) + elapsed
        return (self._state == "running" and self._eeg_signal_healthy and not self._halt.is_set()
                and elapsed <= self.TICK_SECONDS * 3 and eeg_age <= self.EEG_FRESH_SECONDS
                and now - self._last_valid_eeg_at <= self.EEG_FRESH_SECONDS)

    def _hand_inputs_fresh(self, now, require_hand=True):
        """Check the owned camera independently from EEG and pointer engagement."""
        elapsed = max(0.0, now - self._observed_at)  # Cached frame ages keep increasing between worker ticks.
        age = max(self._age(self._camera.get("age_ms")) / 1000 + elapsed,
                  now - self._camera_changed_at)  # Replayed frame sequences cannot refresh a stationary snapshot indefinitely.
        points = self._camera.get("landmarks") or []  # Missing landmarks never count as a tracked hand.
        valid_hand = (self._camera.get("tracked") is True and isinstance(points, (list, tuple)) and len(points) == 21
                      and all(isinstance(point, (list, tuple)) and len(point) >= 2
                              and all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                                      for value in point[:2]) for point in points))  # Reject malformed coordinates before the native adapter sees them.
        return bool(self._owns_camera and self._camera.get("running") and not self._halt.is_set()
                    and elapsed <= self.TICK_SECONDS * 3 and age <= self.CAMERA_FRESH_SECONDS
                    and (valid_hand or not require_hand))  # EEG cue collection needs video freshness but not a pointing pose.

    def _hand_pointer_ready(self, now):
        """Only a recent, validated pointing pose can authorize an EEG selection."""
        return bool(self._hand_inputs_fresh(now) and self._pointer_status.get("pointer_ready") is True
                    and self._pointer_status.get("finger_active") is False and self._pointer_status.get("button_down") is False
                    and self._pointer_status.get("state") == "mouse_pointer" and not self._pointer_status.get("error")
                    and self._pointer_sequence == self._camera_sequence
                    and now - self._pointer_at <= self.CAMERA_FRESH_SECONDS)  # Never click using old engagement after a newer unprocessed frame.

    def _can_hand_arm(self, now, require_guard=False):
        """Camera mouse input does not require a trained or currently confident EEG model."""
        return bool(self._control_mode == "eeg_click" and self._state == "running" and not self._halt.is_set()
                    and self.pointer_controller is not None and getattr(self.pointer_controller, "eeg_compatible", False) is True
                    and self._hand_inputs_fresh(now) and not self._pending and not self._eeg_pending
                    and not self._click_cleanup_pending and not self._pointer_cleanup_pending and not self._pointer_needs_cleanup
                    and (not require_guard or self._guard_ready(now)))  # Initial arming still needs a current hand and confirmed cleanup.

    def _hand_control_authorized(self, now):
        """Retain the explicit hand arm through temporary tracking or EEG loss."""
        return bool(self._control_mode == "eeg_click" and self._eeg_armed and self._state == "running"
                    and self._owns_camera and not self._halt.is_set() and self._guard_ready(now)
                    and not self._pending and not self._eeg_pending and not self._click_cleanup_pending
                    and not self._pointer_cleanup_pending and not self._pointer_needs_cleanup)  # Fresh frames are rechecked before each actual input.

    def _suspend_hand_tracking(self, reason):
        """Release a drag and forget hand engagement while preserving the shared arm."""
        try:
            suspend = getattr(self.pointer_controller, "suspend_tracking", None)  # New adapters preserve faults during automatic recovery.
            result = suspend(reason) if suspend else self.pointer_controller.reset(reason)  # Legacy test adapters still provide equivalent clean reset feedback.
            error = (result.get("error") if isinstance(result, dict) else "Invalid hand suspension result.")
            if not isinstance(result, dict) or any(result.get(key) is not False for key in ("pointer_ready", "finger_active", "button_down")):
                error = error or "Hand input release was not confirmed."  # Never retain an arm while native ownership is ambiguous.
        except Exception as exc:
            result, error = {}, str(exc) or "Hand suspension failed."  # Preserve failure for explicit Stop cleanup.
        with self._lock:
            self._pause_eeg_locked(reason)  # A paused or reacquired target cannot release an old EEG click.
            self._pointer_status = copy.deepcopy(result)  # Publish the adapter's actual paused or fault state.
            self._pointer_at = -math.inf
            self._pointer_sequence = None
            self._pointer_engaged = False  # A new stable pointing dwell is mandatory before movement resumes.
            self._hand_tracking_suspended = True  # Repeated missing frames need no repeated reset/native release.
            if error:
                self._pointer_cleanup_pending = True  # Cleanup failures never recover automatically.
                self._disarm_eeg_locked("Hand input cleanup failed: " + error)
                self._pointer_needs_cleanup = False  # This attempt already failed; retain the fault for an explicit Stop retry.
            self._revision += 1

    def _cleanup_hand_pointer(self, reset=False):
        """Forget hand engagement without moving or clicking; caller owns the action lock."""
        if self.pointer_controller is None:
            return None  # Legacy modes may omit the hand pointer entirely.
        try:
            result = self.pointer_controller.reset() if reset else self.pointer_controller.close()  # Native input is never created by cleanup.
            error = ((result.get("error") or None) if isinstance(result, dict) and result.get("pointer_ready") is False
                     and result.get("button_down") is False and result.get("finger_active") is False
                     else "Hand pointer stop was not confirmed.")  # Require confirmation that the controller has no engaged target.
        except Exception as exc:
            error = str(exc) or "Hand pointer cleanup failed."  # Keep adapter failures visible and recoverable through Stop.
        with self._lock:
            self._pointer_cleanup_pending = bool(error)  # Restart and arming must wait for confirmed cleanup.
            self._pointer_needs_cleanup = False  # A failed attempt stays visible through cleanup_pending and requires an explicit retry.
            self._pointer_engaged = False  # Cleanup ends both aiming engagement and finger ownership.
            self._pointer_at = -math.inf  # Cleanup invalidates all previous pointing evidence.
            self._pointer_sequence = None  # Re-engagement must consume new camera frames.
            self._pointer_status = {"state": "mouse_error" if error else "mouse_paused", "pointer_ready": False,
                                    "finger_active": False, "button_down": False,
                                    "hint": error or "Point your index finger briefly, then move your hand to aim."}  # Explain the next deliberate step.
        return error  # Share the checked cleanup path across Stop, disarm, and target changes.

    def _finish_disarmed_hand_input(self):
        """Release a held finger drag after automatic revocation without reversing lock order."""
        with self._lock:
            if not self._pointer_needs_cleanup or self._eeg_armed:
                return  # Ordinary preview ticks and still-authorized finger gestures need no cleanup.
        with self._action_lock:  # Wait for any already dispatched press/move to finish before releasing its ownership.
            with self._lock:
                if not self._pointer_needs_cleanup or self._eeg_armed:
                    return  # A concurrent explicit reset may already have completed cleanup.
            self._cleanup_hand_pointer()  # This can only release owned input; it never repeats a press or click.

    def _update_hand_pointer(self, generation):
        """Run explicitly armed hand controls and give finger gestures priority over EEG clicks."""
        with self._lock:
            if self._control_mode != "eeg_click" or not self._eeg_armed:
                return  # Legacy directions and disarmed preview do not queue a pointer operation at all.
        with self._action_lock:  # Serialize pointer movement, EEG button input, and lifecycle barriers.
            with self._lock:
                now = self.clock()  # Freshness must be checked after waiting for any preceding action.
                if (generation != self._generation or self.pointer_controller is None or not self._hand_control_authorized(now)):
                    return  # Preview, calibration, revoked guards, and stale input cannot move the pointer.
                hand_fresh = self._hand_inputs_fresh(now)  # A brief missing hand suspends input without revoking the shared arm.
                should_suspend = not hand_fresh and not self._hand_tracking_suspended
            if not hand_fresh:
                if should_suspend:
                    self._suspend_hand_tracking("Hand tracking paused; show your hand and point briefly to resume.")  # The action lock serializes release with EEG clicks.
                return  # Never feed stale landmarks to the mouse controller.
            with self._lock:
                if generation != self._generation or not self._hand_control_authorized(self.clock()):
                    return  # Stop or a new arm may have revoked authorization between the two checks.
                self._hand_tracking_suspended = False  # Fresh tracking may begin a new stable pointing dwell.
                sequence = self._camera_sequence  # Bind this operation to exactly one camera frame.
                if sequence is None or sequence == self._pointer_sequence:
                    return  # A cached frame must not advance the pointing dwell or emit another move.
                arm_epoch = self._arm_epoch  # Reject completion after Stop, target changes, or another arm request.
                points = copy.deepcopy(self._camera.get("landmarks"))  # Pass a private snapshot to the pointer controller.
                hold = bool(self._click_pending_at is not None and now - self._click_pending_at <= self.CLICK_PENDING_SECONDS
                            and self._eeg_inputs_fresh(now) and self._eeg_raw_window_valid)  # A confirmed target stays fixed through bounded valid-raw transitions; bare select or unavailable EEG never freezes aiming.
                self._pointer_sequence = sequence  # Consume the frame before calling a potentially reentrant adapter.
            try:
                feedback = self.pointer_controller.update(points, now, hold=hold)  # Finger clicks and drags take priority even while EEG holds ordinary aiming.
                if not isinstance(feedback, dict) or any(not isinstance(feedback.get(key), bool)
                                                        for key in ("pointer_ready", "finger_active", "button_down")):
                    raise RuntimeError("Hand controller returned invalid input ownership.")  # Missing ownership feedback must never authorize another click source.
            except Exception as exc:
                feedback = {"state": "mouse_error", "pointer_ready": False, "error": str(exc), "hint": str(exc)}  # Fail closed on adapter errors.
            with self._lock:
                if generation != self._generation or arm_epoch != self._arm_epoch:
                    return  # An old move may finish, but it cannot overwrite a newer disarmed state.
                self._pointer_status = copy.deepcopy(feedback)  # Publish actual controller feedback instead of assuming tracked means ready.
                self._pointer_at = now  # Final click checks compare this update's age and camera identity.
                click_id = feedback.get("click_id")  # Only completed hand clicks carry the controller's monotonic identity.
                new_click = isinstance(click_id, int) and not isinstance(click_id, bool) and click_id > self._last_hand_click_id
                if new_click:
                    self._last_hand_click_id = click_id  # Replayed feedback cannot count the same finger click twice.
                    self._finger_click_count += 1  # EEG practice changes no aspect of real finger-click behavior.
                finger_active = feedback.get("finger_active") is True or feedback.get("button_down") is True or new_click
                if finger_active:
                    self._reset_eeg_debounce()  # A finger action consumes any pending EEG selection instead of generating a duplicate click.
                    self._last_dispatch_at = now  # EEG must wait through cooldown and a new rest/select/rest cycle after finger input.
                    self._eeg_reason = "Finger control takes priority; rest again before a new EEG selection."  # Explain why a queued EEG click was cancelled.
                if self._hand_pointer_ready(now):
                    self._pointer_engaged = True  # Keep this latch across pinch/bend/drag states that temporarily own the pointer.
                if feedback.get("error"):
                    self._pointer_cleanup_pending = bool(feedback.get("error"))  # Input errors require confirmed cleanup before reuse.
                    self._disarm_eeg_locked(feedback.get("error") or "Hand target lost; point again and explicitly rearm.")  # Never release a queued click after losing its target.
                elif (self._pointer_engaged or self._click_pending_at is not None) and not self._hand_pointer_ready(now) and not finger_active:
                    self._pause_eeg_locked("Hand target paused; point briefly to resume.")  # Pose uncertainty cancels EEG while the controller handles its own fresh dwell.
                self._revision += 1  # Deliver hand engagement and hold feedback to the HUD.

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
        actuator_ready = (self._click_target == "practice" or self.click_controller is not None) if self._control_mode == "eeg_click" else self.dispatch is not None
        hand_ready = (self._control_mode != "eeg_click" or self.pointer_controller is not None
                      and getattr(self.pointer_controller, "eeg_compatible", False) is True and self._hand_inputs_fresh(now))  # Combined feedback arbitrates finger and EEG input.
        return bool(self._control_mode in ("eeg", "eeg_click") and actuator_ready and hand_ready
                    and not self._click_cleanup_pending and not self._pointer_cleanup_pending and not self._pointer_needs_cleanup
                    and (not require_guard or self._guard_ready(now))
                    and not self._eeg_pending and not self._pending and self._eeg_inputs_fresh(now)
                    and self._decoder_status.get("trained")
                    and ((self._decoder_status.get("arm_eligible") and self._decoder_status.get("passed_validation"))
                         or self._control_mode == "eeg_click" and self._eeg_method == "eyes_closed" and self._experimental_model)  # The new blink model always requires passing held-out validation.
                    and self._eeg_prediction_session == self._eeg.get("session_id")
                    and now - self._eeg_prediction_at <= self.EEG_PREDICTION_HOP * 2
                    and self._confident_eeg_prediction(prediction))

    def _eeg_control_authorized(self, now):
        """Show retained authorization during a brief valid-raw model transition."""
        if not self._eeg_armed:
            return False
        if self._control_mode == "eeg_click":
            return self._hand_control_authorized(now)  # Shared hand authorization survives unavailable EEG and temporary hand loss.
        if self._can_eeg_arm(now, require_guard=True):
            return True
        return bool(self._control_mode in ("eeg", "eeg_click") and self._guard_ready(now) and self._eeg_inputs_fresh(now)
                    and (self._control_mode != "eeg_click" or self._hand_inputs_fresh(now))  # Model transitions cannot excuse a lost hand target.
                    and not self._click_cleanup_pending and not self._pointer_cleanup_pending and not self._pointer_needs_cleanup
                    and not self._eeg_pending and not self._pending and self._eeg_raw_window_valid
                    and self._decoder_status.get("trained") and self._decoder_status.get("arm_eligible")
                    and self._decoder_status.get("passed_validation")
                    and self._eeg_prediction_session == self._eeg.get("session_id")
                    and now - self._eeg_prediction_at <= self.EEG_PREDICTION_HOP * 2
                    and self._eeg_uncertain_since is not None
                    and now - self._eeg_uncertain_since < self.EEG_TRANSITION_SECONDS)

    def _confident_eeg_prediction(self, prediction):
        if self._control_mode == "eeg_click" and self._eeg_method == "double_blink":
            return (prediction.get("valid") is True and prediction.get("label") in ("rest", "select")
                    and prediction.get("ready") is True)  # The waveform model applies its own learned similarity threshold and finishes its causal warmup first.
        confidence, margin = prediction.get("confidence"), prediction.get("margin")
        labels = ("rest", "select") if self._control_mode == "eeg_click" else ("left", "right", "rest")  # Reject cross-mode labels.
        return (prediction.get("valid") is True and not prediction.get("ood")
                and prediction.get("label") in labels
                and isinstance(confidence, (int, float)) and math.isfinite(confidence) and confidence >= self.EEG_CONFIDENCE
                and isinstance(margin, (int, float)) and math.isfinite(margin) and margin >= self.EEG_MARGIN)

    def eeg_arm(self, enabled, guard_token=None):
        if enabled is False:  # Escape must revoke queued input before waiting for an already executing click.
            with self._lock:  # Publish revocation immediately, even while the actuator owns the action lock.
                self._armed = False  # Disable the legacy webcam path at the same boundary.
                self._arm_epoch += 1  # Invalidate any dispatch waiting to enter its final authorization check.
                self._disarm_eeg_locked("EEG control disarmed.")  # Retire the guard and discard a pending selection.
        with self._action_lock, self._lock:  # Finish any previous press/release before changing mouse authorization.
            if not isinstance(enabled, bool):
                return self._result("EEG arm requires a boolean enabled value.")
            self._armed = False
            self._arm_epoch += 1
            if not enabled:
                self._retire_eeg_guard_locked()
            self._disarm_eeg_locked("EEG control disarmed.", retire_guard=not enabled)
            self._motion.reset()
            pointer_error = self._cleanup_hand_pointer() if not enabled else None  # Disarm stops both hand aiming and EEG output.
            if not enabled and self.click_controller is not None:  # Disarm also retries a release after an earlier native failure.
                cleanup_error = self._cleanup_eeg_click()  # This cannot press a new button or move the pointer.
                if cleanup_error:
                    return self._result("; ".join(filter(None, (pointer_error, cleanup_error))))  # Report both cleanups without skipping either.
            if pointer_error:
                return self._result(pointer_error)  # Hand cleanup remains a blocker even if the click adapter stopped cleanly.
            if enabled:
                if guard_token != self._eeg_guard_token or not self._guard_ready(self.clock()):
                    return self._result("A live global Escape guard is required before arming EEG control.")
                if self._control_mode == "eeg_click":
                    pointer_error = self._cleanup_hand_pointer(reset=True)  # A fresh arm always requires a new deliberate pointing gesture.
                    if pointer_error:
                        return self._result(pointer_error)  # Never arm a hand controller whose cleanup is unconfirmed.
                if self._control_mode == "eeg_click" and self._click_target == "mouse" and self.click_controller is not None:
                    cleanup_error = self._cleanup_eeg_click(reset=True)  # Explicit mouse arming may clear a latched actuator failure.
                    if cleanup_error:
                        return self._result(cleanup_error)  # Require a successful release before granting new authorization.
                can_arm = (self._can_hand_arm(self.clock(), require_guard=True) if self._control_mode == "eeg_click"
                           else self._can_eeg_arm(self.clock(), require_guard=True))  # Ordinary camera mouse use is independent from EEG readiness.
                if not can_arm:
                    self._disarm_eeg_locked("EEG arming prerequisites changed; prepare a new Escape guard after recovery.")
                    return self._result("Hand arming requires fresh tracked camera frames and confirmed input cleanup." if self._control_mode == "eeg_click" else
                                        "EEG arming requires EEG mode, a frozen model that passed separate validation, and a fresh confident prediction from valid EEG in the same acquisition session.")
                self._eeg_armed = True
                self._eeg_reason = ("Armed. Point briefly to aim; a calibrated deliberate double blink requests one left click. Finger clicks remain available."
                                    if self._control_mode == "eeg_click" and self._eeg_method == "double_blink" else
                                    "Armed. Point briefly and move your hand to aim; rest, gently close your eyes, then reopen for one EEG selection."
                                    if self._control_mode == "eeg_click" else
                                    "Armed. Rest for at least one second, then sustain a confident EEG direction; camera direction is ignored.")
            return self._result()

    def eeg_use_saved_model(self, decoder, use_unvalidated=False, source=None):
        """Select a restored model explicitly; retain its original validation and require a separate arm."""
        if not isinstance(decoder, EegDecoder) or type(use_unvalidated) is not bool:
            return self._result("A restored EEG decoder and explicit boolean model permission are required.")
        with self._decoder_lock, self._lock:  # Match inference's decoder-before-state lock order.
            if self._control_mode != "eeg_click" or self._eeg_method != "eyes_closed" or self._state != "running" or self._halt.is_set() or self._eeg_pending or self._pending:
                return self._result("Start the hand + EEG preview and finish any trial before selecting a saved model.")
            if decoder.labels != ("rest", "select") or decoder.feature_profile != self.CLICK_FEATURE_PROFILE or not decoder.status().get("trained"):
                return self._result("The saved model must be a trained rest/select click decoder.")
            context = {"session_id": self._eeg.get("session_id"), "channels": copy.deepcopy(self._eeg.get("channels")),
                       "nominal_hz": self._eeg.get("nominal_hz"), "feature_profile": self.CLICK_FEATURE_PROFILE}  # Bind inference to this actual device layout/session.
            try:
                bound = decoder.bind_prediction_context(context)  # The decoder keeps archived training/validation context intact.
                if isinstance(bound, dict) and bound.get("error"):
                    return self._result(bound["error"])
            except (ValueError, TypeError, AttributeError) as exc:
                return self._result(str(exc))  # Reject an incompatible device without replacing the existing model.
            self._arm_epoch += 1
            self._decoder_epoch += 1
            self._disarm_eeg_locked("Saved model selected; explicitly arm hand controls and EEG output.")  # Model selection itself never sends desktop input.
            self._decoder = decoder
            self._decoder_status = decoder.status()
            self._experimental_model = use_unvalidated  # Permission is model-specific; no threshold or evaluation result is changed.
            self._model_source = str(source)[:512] if source is not None else None
            self._eeg_prediction = None
            self._eeg_prediction_at = -math.inf
            self._eeg_prediction_session = None
            self._eeg_last_prediction_id = None
            self._eeg_last_predict_at = -math.inf
            self._eeg_last_result = "Saved model restored with its original validation results. Fresh current-session predictions are required for EEG clicks."
        with self._action_lock:
            return self._result(self._cleanup_hand_pointer())  # Finish any earlier hand input before acknowledging the replacement.

    def eeg_export_model(self):
        """Return a restorable archive while protecting the decoder from concurrent trial updates."""
        from core.eeg_model_store import export_decoder  # Persistence is explicit and adds no device acquisition or input.
        with self._decoder_lock:
            if self._eeg_method != "eyes_closed":
                raise ValueError("Select eyes_closed before saving its model.")  # A concurrent method change cannot put waveform data into a spectral archive.
            return export_decoder(self._decoder)  # The server chooses a fixed local destination rather than accepting arbitrary paths.

    def eeg_export_blink_model(self):
        """Export only the currently selected blink profile to its separate local archive."""
        with self._decoder_lock:
            if self._control_mode != "eeg_click" or self._eeg_method != "double_blink":
                raise ValueError("Select double_blink before saving its model.")
            return self._decoder.export_state()  # The server owns bounded, explicit local persistence.

    def eeg_export_blink_calibration(self):
        """Save accepted unfinished evidence separately from frozen click models."""
        with self._decoder_lock, self._lock:
            if self._control_mode != "eeg_click" or self._eeg_method != "double_blink" or self._eeg_pending:
                raise ValueError("Select double_blink and finish the current cue before saving calibration progress.")
            return self._decoder.export_draft()  # The decoder validates that no model or held-out evaluation is being relabeled as a draft.

    def eeg_use_saved_blink_calibration(self, decoder, source=None):
        """Explicitly resume a verified draft with original acquisition provenance intact."""
        if not isinstance(decoder, BlinkDecoder):
            return self._result("A verified double-blink calibration draft is required.")
        with self._decoder_lock, self._lock:
            if (self._control_mode != "eeg_click" or self._eeg_method != "double_blink" or self._state != "running"
                    or self._halt.is_set() or self._eeg_pending or self._pending):
                return self._result("Start the double-blink preview and finish any cue before resuming saved calibration.")
            if self._decoder_status.get("trained"):
                return self._result("The current model is frozen; keep it saved before choosing a new calibration experiment.")
            restored_trials = {trial["id"]: trial for trial in decoder._training}
            if any(restored_trials.get(trial["id"]) != trial for trial in getattr(self._decoder, "_training", [])):
                return self._result("The current session has newer calibration examples. Save that progress first; loading this draft would discard them.")
            context = {"session_id": self._eeg.get("session_id"), "channels": copy.deepcopy(self._eeg.get("channels")),
                       "nominal_hz": self._eeg.get("nominal_hz"), "feature_profile": "double_blink_waveform_v1"}
            try:
                decoder.resume_calibration(context)  # This records the new acquisition separately; it does not rewrite the old trials.
            except (ValueError, TypeError, AttributeError) as exc:
                return self._result(str(exc))
            self._arm_epoch += 1
            self._decoder_epoch += 1
            self._disarm_eeg_locked("Saved training examples restored. Collect, train, then validate before arming.")
            self._decoder = decoder
            self._decoder_status = decoder.status()
            self._experimental_model = False  # An unfinished calibration cannot inherit any earlier EEG output permission.
            self._model_source = None
            self._blink_calibration_source = str(source)[:512] if source is not None else None
            self._eeg_prediction = None
            self._eeg_prediction_at = self._eeg_last_predict_at = -math.inf
            self._eeg_prediction_session = self._eeg_last_prediction_id = None
            self._blink_last_event_id = None
            self._eeg_last_trial_end = -math.inf  # Counter/time overlap is checked within each recorded acquisition by the decoder.
            numeric_ids = [int(t["id"].removeprefix("eeg-trial-")) for t in decoder._training
                           if t["id"].startswith("eeg-trial-") and t["id"].removeprefix("eeg-trial-").isdigit()]
            self._eeg_trial_sequence = max([self._eeg_trial_sequence, *numeric_ids])  # New cues never reuse an identifier in the restored draft.
            self._eeg_last_result = "Your saved training examples are ready. Finish any missing trials, then click Train & freeze model and collect validation."
        with self._action_lock:
            return self._result(self._cleanup_hand_pointer())

    def eeg_use_saved_blink_model(self, decoder, source=None):
        """Restore waveform calibration without allowing failed validation to authorize clicks."""
        if not isinstance(decoder, BlinkDecoder):
            return self._result("A restored double-blink decoder is required.")
        with self._decoder_lock, self._lock:
            if (self._control_mode != "eeg_click" or self._eeg_method != "double_blink" or self._state != "running"
                    or self._halt.is_set() or self._eeg_pending or self._pending):
                return self._result("Start the double-blink preview and finish any trial before loading its saved model.")
            context = {"session_id": self._eeg.get("session_id"), "channels": copy.deepcopy(self._eeg.get("channels")),
                       "nominal_hz": self._eeg.get("nominal_hz"), "feature_profile": "double_blink_waveform_v1"}
            try:
                bound = decoder.bind_prediction_context(context)  # Retain archived provenance while checking the current hardware layout.
                if isinstance(bound, dict) and bound.get("error"):
                    return self._result(bound["error"])
            except (ValueError, TypeError, AttributeError) as exc:
                return self._result(str(exc))
            self._arm_epoch += 1
            self._decoder_epoch += 1
            self._disarm_eeg_locked("Saved double-blink model selected; explicitly arm again after fresh samples arrive.")
            self._decoder = decoder
            self._decoder_status = decoder.status()
            self._experimental_model = False  # A previous spectral experiment never grants a bypass to this model.
            self._model_source = str(source)[:512] if source is not None else None
            self._blink_calibration_source = None  # A frozen model is reported through its separate model source.
            self._eeg_prediction = None
            self._eeg_prediction_at = self._eeg_last_predict_at = -math.inf
            self._eeg_prediction_session = self._eeg_last_prediction_id = None
            self._blink_last_event_id = None
            self._eeg_last_result = "Double-blink model restored with its original validation. Passing validation and fresh hand/EEG input are required."
        with self._action_lock:
            return self._result(self._cleanup_hand_pointer())

    def _cleanup_eeg_click(self, reset=False):
        """Confirm owned input was released; callers hold the action and state locks."""
        try:
            result = self.click_controller.reset() if reset else self.click_controller.close()  # Never creates a new click.
            error = (None if isinstance(result, dict) and result.get("ok") is True else
                     (result.get("error") or "Mouse release was not confirmed.") if isinstance(result, dict) else
                     "Invalid mouse cleanup result.")  # Require a positive release acknowledgement from the adapter.
        except Exception as exc:
            error = str(exc) or "Mouse cleanup failed."  # Adapter failures become visible, recoverable disarmed responses.
        self._click_cleanup_pending = bool(error)  # Block restart and rearming until a later cleanup succeeds.
        return error  # Reuse the same failure handling for explicit disarm and mouse rearming.

    def eeg_trial(self, label, phase="train"):
        with self._lock:
            click_mode = self._control_mode == "eeg_click"  # Binary trials use voluntary cues without camera labels.
            labels = ("rest", "select") if click_mode else ("left", "right", "rest")  # Never reinterpret a swipe label as a click.
            if label not in labels or phase not in ("train", "validate"):
                return self._result("EEG trial requires " + ", ".join(labels) + " and train or validate phase.")
            now = self.clock()
            if self._pending or self._eeg_pending or now < self._eeg_last_trial_end:
                return self._result("Finish the current trial window before recording another trial.")
            camera_age = max(self._age(self._camera.get("age_ms")) / 1000 + max(0, now - self._observed_at),
                             now - self._camera_changed_at)
            if click_mode and not self._eeg_inputs_fresh(now):  # All click trials still require fresh current-session EEG.
                return self._result("Start the EEG preview and wait for fresh samples before collecting a selection trial.")
            if click_mode and not self._hand_inputs_fresh(now, require_hand=False):  # The combined preview must have current camera frames.
                return self._result("Start the hand camera and wait for fresh frames before collecting a selection trial.")  # EEG cues do not require a pointing pose.
            if (not click_mode and (not self._eeg_inputs_fresh(now) or not self._camera.get("tracked") or not self._owns_camera
                    or camera_age > self.CAMERA_FRESH_SECONDS)):
                return self._result("Guided trials require fresh tracked camera and EEG. Start in Webcam mode to collect trials.")
            if phase == "train" and self._decoder_status.get("trained"):
                return self._result("The model is frozen. Collect validation trials or reset before training again.")
            if phase == "validate" and not self._decoder_status.get("trained"):
                return self._result("Freeze a trained model before collecting separate validation trials.")
            if phase == "validate" and self._decoder_status.get("state") in ("validated", "validation_failed"):
                return self._result("Held-out validation is complete and frozen. Reset to begin a new experiment.")
            if click_mode and self._eeg_method == "double_blink" and type(self._last_valid_counter) is not int:
                return self._result("Wait for a fresh EEG packet before starting the blink cue.")  # The sample watermark must come from actual acquired data.
            self._armed = False
            self._arm_epoch += 1
            self._retire_eeg_guard_locked()
            self._disarm_eeg_locked("Guided EEG trial; all desktop actions are disarmed.")
            self._motion.reset()
            self._eeg_trial_sequence += 1
            duration = self.CLICK_TRIAL_SECONDS if click_mode else self.TRIAL_SECONDS  # Click cues leave enough time to settle before measurement.
            self._eeg_pending = {"label": label, "phase": phase, "start": now, "end": now + duration,
                                 "click_mode": click_mode,  # Bind the trial to its cue protocol rather than the later UI selection.
                                 "method": self._eeg_method if click_mode else None,
                                 "blink_rows": [], "blink_counter": None,  # Preserve the complete five-second cue across the SDK's rolling window.
                                 "blink_start_counter": self._last_valid_counter,  # A stable packet identity defines the start instead of jittered callback timestamps.
                                 "blink_required": round(duration * 250), "blink_error": None,
                                 "id": "eeg-trial-" + str(self._eeg_trial_sequence), "motion_seen": 0,
                                 "motion_at": None,
                                 "wrong_motion": False, "min_x": math.inf, "max_x": -math.inf,
                                 "min_y": math.inf, "max_y": -math.inf, "frames": 0,
                                 "generation": self._generation, "decoder_epoch": self._decoder_epoch}
            self._eeg_last_trial_end = now + duration  # Never allow the next trial to overlap this complete cue interval.
            self._eeg_last_result = (("Keep your eyes open and relax without moving." if label == "rest" else
                                     "Gently close your eyes, count four seconds, then reopen; keep your head and jaw still.") if click_mode else
                                    "Hold the hand still for three seconds." if label == "rest" else
                                     "Hold still for half a second, then move " + label + " once before the final half-second of this three-second trial.")
            if click_mode and self._eeg_method == "double_blink":
                self._eeg_last_result = ("No command: look normally and allow ordinary single blinks; do not deliberately double blink."
                                        if label == "rest" else "Wait one second, blink twice deliberately near the middle of the cue, then reopen and rest.")
        with self._action_lock:  # Revocation above prevents queued moves; finish an already dispatched move before acknowledging the cue.
            return self._result(self._cleanup_hand_pointer())  # Calibration starts with the hand pointer disengaged.

    def _observe_eeg_trial_locked(self, points, direction, new_frame, tracked, now):
        trial = self._eeg_pending
        if not trial:
            return
        if trial.get("click_mode"):  # Self-labelled binary trials never depend on camera landmarks or hand motion.
            if not self._eeg_signal_healthy or not self._hand_inputs_fresh(now, require_hand=False):  # A sensor interruption invalidates this combined-preview trial.
                self._eeg_pending = None  # Do not turn a partial or stale window into training evidence.
                self._eeg_last_result = "Selection trial discarded: camera or EEG became unavailable."  # Explain why the count did not increase.
            return  # The decoder checks the complete, finite, contiguous signal window when the cue finishes.
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
        """Freeze the selected training data without collecting or passing validation.

        Fitting may ignore atypical blink examples, but the decoder retains their
        records. A failed fit leaves those records available for another attempt.
        """
        with self._decoder_lock:  # Method switching cannot replace the decoder between the precheck and fitting.
            with self._lock:
                self._armed = False
                self._arm_epoch += 1
                self._decoder_epoch += 1
                self._retire_eeg_guard_locked()
                self._disarm_eeg_locked("Training freezes the EEG model; separate validation is required before arming.")
                if self._pending or self._eeg_pending:
                    return self._result("Finish the current trial before training.")
            result = self._decoder.train()
            decoder_status = self._decoder.status()
            with self._lock:
                self._decoder_status = decoder_status
                self._eeg_last_result = result.get("error") or "Model frozen. Collect eight fresh held-out validation trials per class."
                self._eeg_prediction = None
                self._eeg_prediction_at = -math.inf
        with self._action_lock:  # State and decoder locks are released before waiting for any old pointer operation.
            pointer_error = self._cleanup_hand_pointer()  # Freezing a model cannot retain an engaged hand target.
            return self._result(result.get("error") or pointer_error)  # Acknowledge training only after input is quiescent.

    def eeg_reset(self):
        with self._decoder_lock:  # Clear exactly the profile selected at this boundary; other saved method evidence remains intact.
            with self._lock:
                self._armed = False
                self._arm_epoch += 1
                self._decoder_epoch += 1
                self._retire_eeg_guard_locked()
                self._disarm_eeg_locked("Active EEG method's training and validation reset.")
                self._eeg_pending = None
                self._eeg_prediction = None
                self._eeg_prediction_at = -math.inf
                self._eeg_full_rows = []
                self._eeg_last_trial_end = -math.inf
                self._eeg_last_result = "Active EEG model, training features, and validation cleared."
                self._experimental_model = False  # Reset cannot retain permission for a different future model.
                self._model_source = None
                if self._eeg_method == "double_blink":
                    self._blink_calibration_source = None  # Reset applies only to the explicitly selected calibration profile.
            self._decoder.reset()
            with self._lock:
                self._decoder_status = self._decoder.status()
        with self._action_lock:  # Immediate state revocation prevents new input while old operations finish.
            return self._result(self._cleanup_hand_pointer())  # Resetting calibration also resets pointing engagement.

    def _process_eeg(self, signal, now, generation):
        """Decode full-rate EEG; click trials use explicit cues and legacy swipe trials use camera labels."""
        with self._lock:
            blink_method = self._control_mode == "eeg_click" and self._eeg_method == "double_blink"
        if blink_method:
            return self._process_blink(signal, now, generation)  # Short pulses must not pass through the spectral two-second window or its 250ms hop.
        with self._lock:
            if generation != self._generation or self._halt.is_set():
                return
            if (not signal or signal.get("state") != "running" or signal.get("mode") != "signal"
                    or signal.get("session_id") != self._eeg.get("session_id")):
                self._eeg_full_rows = []
                self._eeg_prediction = None
                if self._control_mode == "eeg_click":
                    self._pause_eeg_locked("Full-rate EEG is unavailable; EEG clicks paused while hand controls remain available.")
                else:
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
            feature_options = ({"feature_profile": self.CLICK_FEATURE_PROFILE} if self._control_mode == "eeg_click" else {})  # Keep every stage on its model's feature profile.
        if trial:
            still = (trial["frames"] >= 3 and trial["max_x"] - trial["min_x"] <= self._motion.settings["deadband"]
                     and trial["max_y"] - trial["min_y"] <= self._motion.settings["deadband"])
            verified = (True if trial.get("click_mode") else  # A click trial's label comes from its explicit cue, not camera verification.
                        still and not trial["wrong_motion"] if trial["label"] == "rest" else
                        trial["motion_seen"] == 1 and not trial["wrong_motion"])
            if not verified:
                with self._lock:
                    self._eeg_last_result = "EEG trial discarded: camera did not verify the requested " + trial["label"] + " movement/rest."
            else:
                # Counters define the 2s window. Host batch jitter must not
                # turn 500 valid packets into 499 via an exact timestamp cutoff.
                size = round(hz * self.EEG_WINDOW_SECONDS) if isinstance(hz, (int, float)) and math.isfinite(hz) else 0
                settle = 1.0 if trial.get("click_mode") else 0.5  # Exclude the initial eye-closing transient from click calibration.
                sample_end = trial["end"] - 1.0 if trial.get("click_mode") else trial["end"]  # Finish click measurement before the cue to reopen.
                trial_rows = [r for r in rows if trial["start"] + settle <= r.get("estimated_monotonic", -math.inf) < sample_end][:size]
                extracted = extract_features(trial_rows, channels, hz, session_context=context, **feature_options)  # Use exactly the same processing as prediction.
                window = extracted.get("window") or {}
                if extracted.get("ok") and (window.get("end", math.inf) > sample_end
                        or (not trial.get("click_mode") and trial["label"] != "rest" and not window.get("start", math.inf) <= trial["motion_at"] < window.get("end", -math.inf))):
                    extracted = {"ok": False, "reason": ("Selection trial samples fell outside the settled two-second measurement window; repeat the five-second cue."
                                 if trial.get("click_mode") else
                                 "Observed movement was outside the selected two-second EEG trial window; move after the initial still period and before the final half-second.")}  # Give instructions for the active calibration method.
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
        extracted = (extract_features(rows[-sample_count:], channels, hz, session_context=context, **feature_options)
                     if sample_count and len(rows) >= sample_count else
                     {"ok": False, "reason": "Need a complete two-second full-rate EEG window."})
        with self._decoder_lock:
            with self._lock:
                if (generation != self._generation or decoder_epoch != self._decoder_epoch or self._halt.is_set()
                        or self._control_mode == "eeg_click" and self._eeg_method == "double_blink"):
                    return  # A method switch during feature extraction must not call predict on the new waveform decoder.
            prediction = (self._decoder.predict(extracted) if extracted.get("ok") else
                          {"label": None, "valid": False, "confidence": 0.0, "margin": 0.0,
                           "ood": False, "window_id": None, "reason": extracted.get("reason") or extracted.get("error")})
            decoder_status = self._decoder.status()
        dispatch_direction = None
        dispatch_click = False  # Practice and mouse selections share a separate return-to-rest dispatch path.
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
                if self._control_mode == "eeg_click":
                    transition = self._eeg_raw_window_valid and (prediction.get("valid") is True or prediction.get("ood") is True)
                    if transition and self._eeg_armed:
                        if self._eeg_uncertain_since is None:
                            self._eeg_uncertain_since = now  # Mixed sliding windows naturally occur while changing between rest and select.
                        self._eeg_candidate = None
                        self._eeg_candidate_count = 0
                        self._eeg_neutral_since = None  # A transition never counts as confident select or rest time.
                        if now - self._eeg_uncertain_since < self.EEG_TRANSITION_SECONDS:
                            self._eeg_reason = "EEG transition: waiting for confident rest/select. Hand controls remain available."  # Retain bounded prior-cycle evidence without permitting EEG input.
                            return
                    self._pause_eeg_locked("EEG clicks paused: " + (prediction.get("reason") or "prediction is uncertain") + ". Hand controls remain available.")  # Raw faults or prolonged uncertainty cancel EEG without freezing the mouse.
                    return
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
                        self._eeg_reason = ("Model transition: selections paused until confident rest/select returns; authorization expires after three seconds of uncertainty."
                                            if self._control_mode == "eeg_click" else
                                            "Model transition: actions suppressed until three new confident direction predictions; authorization expires after three seconds of uncertainty.")  # Describe the correct labels without implying that uncertainty can trigger input.
                else:
                    self._disarm_eeg_locked(prediction.get("reason") or "Invalid EEG prediction; explicitly rearm after recovery.")
                return
            self._eeg_uncertain_since = None
            if not window_id or window_id == self._eeg_last_prediction_id:
                return
            self._eeg_last_prediction_id = window_id
            if not self._eeg_armed or self._control_mode not in ("eeg", "eeg_click"):
                self._reset_eeg_debounce()
                self._eeg_reason = "Prediction only. Explicit EEG arming is required for desktop actions."
                return
            if not self._can_eeg_arm(now, require_guard=True):
                if self._control_mode == "eeg_click":
                    self._pause_eeg_locked("EEG clicks paused: select an eligible saved model and wait for fresh confident EEG. Hand controls remain available.")
                else:
                    self._disarm_eeg_locked("Validation, current-session prediction, or signal freshness no longer permits EEG control.")
                return
            label = prediction["label"]
            if self._control_mode == "eeg_click":  # Only the binary state machine can prepare a single mouse click.
                if not self._hand_pointer_ready(now):  # Tracking alone does not establish an engaged pointer target.
                    self._reset_eeg_debounce()  # Do not accumulate EEG selection evidence while still waiting for a pointing hand.
                    self._eeg_reason = self._pointer_status.get("hint") or "Point briefly to engage hand aiming before selecting."  # Show the actual missing prerequisite.
                    return  # Practice and real clicks both require a valid hand-controlled target.
                dispatch_click = self._advance_click_locked(label, now)  # Wait for reopened eyes/rest before dispatching.
            elif label == "rest":
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
        if dispatch_click:  # Recheck authorization immediately before counting a practice selection or touching the mouse.
            self._dispatch_eeg_click(generation, arm_epoch, window_id)

    def _process_blink(self, signal, now, generation):
        """Collect five-second cues and consume each completed blink pair once.

        Collection counts contiguous device packets instead of callback arrival
        timestamps. The short grace period waits for delivery only; it never adds
        extra samples to a cue. The decoder separately checks waveform quality,
        and the final dispatch path rechecks arming before any real mouse click.
        """
        dispatch_id = None  # No native input can occur until all model, stream, and hand checks succeed below.
        with self._decoder_lock:  # Reset and update the causal detector under one lock; never process the same stream concurrently.
            with self._lock:
                if (generation != self._generation or self._halt.is_set()
                        or self._control_mode != "eeg_click" or self._eeg_method != "double_blink"):
                    return
                if (not signal or signal.get("state") != "running" or signal.get("mode") != "signal"
                        or signal.get("session_id") != self._eeg.get("session_id") or not self._eeg_signal_healthy):
                    self._eeg_prediction = None
                    self._eeg_full_rows = []
                    self._pause_eeg_locked("Double-blink clicks paused until fresh, uninterrupted EEG returns.")
                    if self._eeg_pending:
                        self._eeg_pending = None
                        self._eeg_last_result = "Blink trial discarded: full-rate EEG was interrupted."
                    return
                rows = signal.get("samples") or []
                self._eeg_full_rows = copy.deepcopy(rows[-1250:])
                pending = self._eeg_pending
                if pending:
                    if now > pending["end"] + self.BLINK_COLLECTION_GRACE and len(pending["blink_rows"]) < pending["blink_required"]:
                        pending["blink_error"] = "Blink trial timed out: the remaining EEG samples arrived after the collection deadline. Accepted trials were kept."
                    # Copy exactly the next five seconds of contiguous packets; callback timestamps vary with BLE batching.
                    for row in rows:
                        if pending["blink_error"]:
                            break  # A late burst cannot turn an expired or discontinuous trial into accepted evidence.
                        at, counter = row.get("estimated_monotonic"), row.get("counter")
                        if type(counter) is not int or not 0 <= counter <= 0xFFFFFFFF:
                            pending["blink_error"] = "Blink trial discarded: an EEG packet had an invalid counter."
                            break
                        offset = (counter - pending["blink_start_counter"]) & 0xFFFFFFFF  # Measure this packet from the cue's last pre-start counter, including wraparound.
                        if offset == 0 or offset >= 0x80000000 or offset <= len(pending["blink_rows"]):
                            continue  # Overlapping device buffers never duplicate a training observation.
                        if len(pending["blink_rows"]) >= pending["blink_required"]:
                            break  # Later samples belong to a future cue, never this one.
                        if offset != len(pending["blink_rows"]) + 1:
                            pending["blink_error"] = "Blink trial discarded: EEG packet gap in the five-second window. Accepted trials were kept."
                            break  # Delivery jitter is acceptable; actual missing sample counters are not.
                        if not isinstance(at, (int, float)) or not math.isfinite(at):
                            pending["blink_error"] = "Blink trial discarded: invalid EEG packet timing."
                            break
                        trial_row = copy.deepcopy(row)  # Never change the live stream, its original timestamps, or raw voltage values.
                        trial_row["estimated_monotonic"] = pending["start"] + (offset - 1) / 250.0  # Reconstruct only the private calibration timeline at the nominal sample rate.
                        pending["blink_rows"].append(trial_row)  # A copied row belongs only to this pending trial; the shared stream stays unchanged.
                        pending["blink_counter"] = counter  # Retain the last accepted packet identity for diagnostics.
                    if now >= pending["end"] + self.BLINK_COLLECTION_GRACE and len(pending["blink_rows"]) < pending["blink_required"]:
                        pending["blink_error"] = ("Blink trial timed out after receiving " + str(len(pending["blink_rows"])) + "/" +
                                                  str(pending["blink_required"]) + " samples. Accepted trials were kept; wait for a steady stream and retry.")
                # Finish only after the complete cue and its sample count; a late callback cannot truncate calibration.
                due = bool(pending and now >= pending["end"] and
                           (len(pending["blink_rows"]) == pending["blink_required"] or pending["blink_error"]))
                trial = copy.deepcopy(pending) if due else None  # Detach completed evidence before releasing the service-state lock.
                if trial:
                    self._eeg_pending = None  # The same completed cue can be submitted to the decoder only once.
                decoder_epoch, reset_serial = self._decoder_epoch, self._blink_reset_serial
                reset_stream = self._blink_reset_pending
                self._blink_reset_pending = False  # Any later cancellation raises the serial and schedules another reset.
            if reset_stream:
                self._decoder.reset_stream()  # First update watermarks buffered samples, so arming cannot replay an earlier blink.
            if trial:
                trial_signal = {**signal, "samples": trial["blink_rows"]}  # Pass the complete cue to the same causal detector used live.
                result = ({"error": trial["blink_error"]} if trial["blink_error"] else
                          self._decoder.add_trial(trial["label"], trial_signal, phase=trial["phase"],
                                                  trial_id=trial["id"], start=trial["start"], end=trial["end"]))
                with self._lock:
                    self._eeg_last_result = result.get("error") or ("Accepted " + trial["phase"] + " " +
                                             ("no-command" if trial["label"] == "rest" else "double-blink") + " trial.")
            prediction = self._decoder.update(signal)  # The detector deduplicates counters and validates amplitude, clipping, gaps, and context.
            decoder_status = self._decoder.status()
            with self._lock:
                if (generation != self._generation or decoder_epoch != self._decoder_epoch
                        or reset_serial != self._blink_reset_serial or self._halt.is_set()):
                    return  # An Escape, hand pause, or profile switch that occurred during inference invalidates its result.
                self._decoder_status = decoder_status
                self._eeg_prediction = copy.deepcopy(prediction)
                self._eeg_prediction_at = now
                self._eeg_prediction_session = signal.get("session_id")
                self._eeg_raw_window_valid = prediction.get("valid") is True
                self._eeg_neutral_ready = prediction.get("ready") is True
                self._revision += 1
                event_id = prediction.get("event_id")
                fresh_event = isinstance(event_id, str) and bool(event_id) and event_id != self._blink_last_event_id
                if fresh_event:
                    self._blink_last_event_id = event_id  # Consume blocked/disarmed detections too; they cannot be queued for a later target.
                self._blink_dispatch_id = None
                if not self._eeg_armed:
                    self._eeg_reason = prediction.get("reason") or "Double-blink prediction only; explicitly arm hand controls before clicking."
                elif not self._hand_pointer_ready(now):
                    self._reset_eeg_debounce()  # Reprime on recovery; neither blink may be carried from an unavailable or finger-owned target.
                    self._eeg_reason = self._pointer_status.get("hint") or "Point briefly to engage hand aiming."
                elif not self._can_eeg_arm(now, require_guard=True):
                    self._eeg_reason = prediction.get("reason") or "Double-blink clicks require passing validation and fresh valid EEG. Finger controls remain available."
                elif fresh_event and prediction.get("label") == "select":
                    self._blink_dispatch_id = dispatch_id = event_id  # The final boundary checks age, hand ownership, guard, and cooldown again.
                else:
                    self._eeg_reason = prediction.get("reason") or "Aim with your hand; make one deliberate double blink to click."
                arm_epoch = self._arm_epoch
        if dispatch_id:
            self._dispatch_eeg_click(generation, arm_epoch, dispatch_id)  # Keep native input outside both decoder and state locks.

    def _advance_click_locked(self, label, now):
        """Recognize one rest -> sustained select -> rest cycle without issuing input."""
        if self._click_pending_at is not None and now - self._click_pending_at > self.CLICK_PENDING_SECONDS:
            self._pause_eeg_locked("EEG selection expired; return to rest before a new selection. Hand controls remain available.")  # Never retain an old click request or stop finger input because of it.
            return False  # An expired cycle must not become a late click when the user next relaxes.
        if label == "rest":  # A completed selection is released only after the model recognizes eyes-open rest again.
            self._eeg_candidate = None  # Rest ends any unfinished select streak.
            self._eeg_candidate_count = 0  # A later selection needs three new prediction windows.
            if self._eeg_neutral_since is None:
                self._eeg_neutral_since = now  # Start measuring continuous confident rest.
            if now - self._eeg_neutral_since < self.EEG_REST_SECONDS:
                self._eeg_reason = "Keep your eyes open until one second of confident rest is recognized."  # Expose the current waiting condition.
                return False  # Brief rest predictions cannot release or prepare a click.
            self._eeg_neutral_ready = True  # This rest interval can prepare exactly one later select cycle.
            if self._click_pending_at is not None:
                return now - self._last_dispatch_at >= self.EEG_COOLDOWN_SECONDS  # Preserve the existing spacing between actions.
            self._eeg_reason = "Rest ready. Gently close your eyes for select, then reopen for one selection."  # Give an actionable cue.
            return False  # Rest alone never sends a click.
        self._eeg_neutral_since = None  # A select prediction interrupts the continuous rest timer.
        if self._click_pending_at is not None:
            self._eeg_reason = "Selection detected. Open your eyes and return to rest to complete it."  # Holding select cannot repeat.
            return False  # Wait for the release phase instead of clicking while the eyes are closed.
        if not self._eeg_neutral_ready:
            self._eeg_reason = "Start with eyes-open rest before selecting."  # Arming while selected must never create an immediate action.
            return False  # A fresh rest interval is mandatory for the first action and after each click.
        self._eeg_candidate_count = self._eeg_candidate_count + 1 if self._eeg_candidate == label else 1  # Count distinct confident windows only.
        self._eeg_candidate = label  # Track the selection label whose streak is being confirmed.
        if self._eeg_candidate_count >= 3:
            self._click_pending_at = now  # Record when this confirmed selection began waiting for release.
            self._eeg_neutral_ready = False  # Consume the preceding rest interval so this cycle cannot repeat.
            self._eeg_reason = "Selection detected. Reopen your eyes; the action follows confirmed rest."  # Keep the click timing explicit.
        return False  # Selection is only prepared here; returning to rest is what completes it.

    def _dispatch_eeg_click(self, generation, arm_epoch, prediction_id):
        """Final authorization boundary for a practice count or one ordinary left click."""
        with self._action_lock:  # Serialize the complete press/release with Stop and explicit rearming.
            with self._lock:  # Compare every authorization fact against the latest service state.
                now = self.clock()  # Re-read time after waiting for any preceding action.
                prediction = self._eeg_prediction or {}  # Missing predictions always fail closed.
                blink_method = self._eeg_method == "double_blink"
                if blink_method:
                    event_at = prediction.get("event_at")
                    cycle_ready = (prediction_id == self._blink_dispatch_id == prediction.get("event_id")
                                   and prediction.get("label") == "select" and bool(prediction_id)
                                   and isinstance(event_at, (int, float)) and math.isfinite(event_at)
                                   and 0 <= now - event_at <= self.EEG_FRESH_SECONDS)  # Never click for a late BLE batch or a superseded blink.
                else:
                    cycle_ready = (prediction_id == prediction.get("window_id") and prediction.get("label") == "rest"
                                   and self._click_pending_at is not None and now - self._click_pending_at <= self.CLICK_PENDING_SECONDS
                                   and self._eeg_neutral_since is not None and now - self._eeg_neutral_since >= self.EEG_REST_SECONDS)
                if (self._control_mode != "eeg_click" or not self._eeg_armed or not self._can_eeg_arm(now, require_guard=True)
                        or not self._hand_pointer_ready(now)  # A lost or unprocessed hand frame cannot authorize a click.
                        or generation != self._generation or arm_epoch != self._arm_epoch
                        or not cycle_ready  # Blink events finish after the second reopened blink; eyes-closed selections keep their original rest confirmation.
                        or now - self._last_dispatch_at < self.EEG_COOLDOWN_SECONDS):
                    return  # A stale, disarmed, expired, or superseded cycle cannot affect the desktop.
                target = self._click_target  # Bind this action to the destination that was explicitly armed.
                self._last_dispatch_at = now  # Start the cooldown before leaving the final authorization boundary.
                self._reset_eeg_debounce()  # Consume the cycle before any callback can reenter the service.
                if target == "practice":
                    self._selection_count += 1  # Practice affects only an in-memory counter displayed in the HUD.
                    self._eeg_reason = "Practice selection completed. Rest again before another selection."  # Confirm success without a mouse event.
                    self._revision += 1  # Deliver the updated counter through the existing status polling.
                    return  # The practice branch never constructs or calls a native mouse device.
            try:
                result = self.click_controller.click()  # Send one left-button press/release at the normal pointer's current position.
                if not isinstance(result, dict) or result.get("ok") is not True:
                    raise RuntimeError(result.get("error", "Mouse click was not confirmed.") if isinstance(result, dict) else "Invalid mouse click result.")  # Require explicit completion.
                with self._lock:
                    if generation != self._generation or arm_epoch != self._arm_epoch:
                        return  # A completed old action must not overwrite the state or counters of a newer mode/target.
                    self._click_count += 1  # Count a real click only after the actuator confirms its release.
                    self._eeg_reason = "One left click completed. Move your hand to aim; rest before selecting again."  # The hand chooses the next target.
                    self._revision += 1  # Refresh the user-visible completion counter.
            except Exception as exc:
                try:
                    cleanup = self.click_controller.close()  # Retry release of owned input after a failed native operation.
                    cleanup_pending = bool(cleanup.get("error"))  # Preserve an unconfirmed release for Stop to retry.
                except Exception:
                    cleanup_pending = True  # Never assume input was released after a cleanup exception.
                with self._lock:
                    self._click_cleanup_pending = cleanup_pending  # Block subsequent starts/arming until cleanup succeeds.
                    self._disarm_eeg_locked("EEG click failed: " + str(exc))  # Any input failure revokes the entire gesture authorization.
                    self._revision += 1  # Publish the failure instead of incrementing the success counter.

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
            if eeg_pending and self._eeg_pending.get("click_mode"):  # The cue follows server time even when rendering is delayed.
                eeg_pending["cue"] = ("Keep your eyes open and relax without moving." if eeg_pending["label"] == "rest" else
                                      "Open your eyes and rest." if now >= self._eeg_pending["end"] - 1 else
                                      "Gently close your eyes; keep your head and jaw still.")  # These labels are voluntary instructions, not verified intention.
                if self._eeg_pending.get("method") == "double_blink":
                    eeg_pending["samples_collected"] = len(self._eeg_pending["blink_rows"])
                    eeg_pending["samples_required"] = self._eeg_pending["blink_required"]
                    eeg_pending["cue"] = ("Finishing signal collection; keep your eyes open." if now >= self._eeg_pending["end"] else
                                          "No command: look normally; ordinary single blinks are allowed."
                                          if eeg_pending["label"] == "rest" else
                                          "Keep your eyes open and settle for one second." if now < self._eeg_pending["start"] + 1 else
                                          "Reopen your eyes and rest." if now >= self._eeg_pending["end"] - 1 else
                                          "Blink twice deliberately, then reopen and rest.")  # One pair near the cue's middle leaves a baseline before and after it.
            return {"revision": self._revision, "state": self._state, "running": self._state in ("starting", "running"),
                    "cleanup_pending": self._click_cleanup_pending or self._pointer_cleanup_pending or self._pointer_needs_cleanup or (self._owns_camera or self._owns_eeg) and self._state in ("error", "stopped", "stopping"),
                    "armed": self._armed and self._can_arm(now), "can_arm": self._can_arm(now),
                    "reason": self._reason, "error": self._error,
                    "control_mode": self._control_mode,
                    "eeg_control": {"armed": self._eeg_control_authorized(now),
                                    "method": self._eeg_method,  # The HUD routes cues and saved-model controls to this explicit profile.
                                    "can_arm": self._can_hand_arm(now) if self._control_mode == "eeg_click" else self._can_eeg_arm(now), "reason": self._eeg_reason,
                                    "can_eeg_click": self._can_eeg_arm(now) and self._hand_pointer_ready(now) if self._control_mode == "eeg_click" else False,  # EEG readiness is independent from the shared hand arm.
                                    "experimental_model": self._experimental_model,  # Show the explicit permission without rewriting failed validation.
                                    "saved_model_source": self._model_source,
                                    "saved_calibration_source": self._blink_calibration_source if self._eeg_method == "double_blink" else None,
                                    "guard_ready": self._guard_ready(now),
                                    "prediction": copy.deepcopy(self._eeg_prediction),
                                    "decoder": copy.deepcopy(self._decoder_status), "pending": eeg_pending,
                                    "neutral_ready": self._eeg_neutral_ready,
                                    "target": self._click_target,  # Mouse output requires a separate explicit destination choice.
                                    "hand_pointer": {"required": self._control_mode == "eeg_click",  # This mode always uses the hand to aim.
                                                     "ready": self._eeg_armed and self._hand_pointer_ready(now),  # Stale engagement cannot look ready in the HUD.
                                                     "state": self._pointer_status.get("state"),  # Distinguish pointing dwell, active aiming, and pauses.
                                                     "hint": self._pointer_status.get("hint")},  # Explain how to engage or recover the hand target.
                                    "selection_count": self._selection_count,  # Practice selections never imply real mouse input.
                                    "click_count": self._click_count,  # Successful native clicks have their own completion count.
                                    "finger_click_count": self._finger_click_count,  # Finger clicks remain real input even when EEG uses its practice counter.
                                    "selection_pending": self._click_pending_at is not None,  # Show that reopening/rest will complete the current cycle.
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
                for key in ("running", "tracked", "sequence", "age_ms", "fps", "image", "landmarks", "width", "height"):
                    result["camera"][key] = copy.deepcopy(self._camera_display.get(key))  # Every overlay field refers to the same JPEG frame.
                if isinstance(result["camera"].get("age_ms"), (int, float)):
                    result["camera"]["age_ms"] += round(max(0.0, self.clock() - self._observed_at) * 1000)  # Preserve display-frame age between polls.
                result["eeg"]["waveform"] = copy.deepcopy(self._waveform)
            return result


PreviewSession = MultimodalPreview
