"""Explicit fast double-blink clicking with optional camera hand aiming.

This service owns acquisition and input lifetimes, not sensor discovery or files.
The fixed one-second model is trained and validated separately before arming.
Packet watermarks, a renewable Escape guard and native-input ownership checks
prevent stale evidence, unfinished gestures or interrupted streams from clicking.
"""
from __future__ import annotations

import copy
import math
import threading
import time

from .hand_tracking import FIRST_FRAME_TIMEOUT, STARTUP_TIMEOUT
from .eeg_fast_blink import FastCnnBlinkDecoder

DISCLAIMER = ("Voluntary double-blink waveform detection, not intent decoding. "
              "EEG timing uses approximate host timestamps; held-out trial accuracy does not measure physical click latency.")


class MultimodalPreview:
    """One fixed EEG workflow; starting acquisition never arms desktop input."""

    TICK_SECONDS = 0.125  # Preserve the tested 375-ms maximum service-observation gap.
    HAND_TICK_SECONDS = 1.0 / 30.0
    CAMERA_FRESH_SECONDS = 0.5
    EEG_FRESH_SECONDS = 0.75
    STARTUP_GRACE_SECONDS = 5.0
    CAMERA_STARTUP_SECONDS = STARTUP_TIMEOUT + FIRST_FRAME_TIMEOUT + STARTUP_GRACE_SECONDS
    CAMERA_LOSS_SECONDS = EEG_LOSS_SECONDS = 2.0
    EEG_PREDICTION_HOP = 0.25  # A current prediction must be refreshed within twice this display freshness bound.
    EEG_COOLDOWN_SECONDS = 1.0
    EEG_GUARD_SECONDS = 3.0
    MAX_RETIRED_GUARDS = 256
    BLINK_COLLECTION_GRACE = 1.5  # Delivery grace never adds samples to the fixed 250-packet trial.

    def __init__(self, camera, brainbit, gesture_busy=lambda: False,
                 clock=time.monotonic, click_controller=None, pointer_controller=None, camera_mouse_enabled=None):
        self.camera = camera
        self.brainbit = brainbit
        self.gesture_busy = gesture_busy
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
        self._last_hand_click_id = 0  # Controller click IDs increase across resets; count each successful finger click once.
        self._last_finger_at = -math.inf
        self._cnn_finger_active = False  # Cancel CNN evidence only at actual native-input ownership edges.
        self._cnn_cancel_pending = False
        self._finger_click_count = 0  # Keep hand clicks separate from EEG actions and blink clicks.
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
        self._owns_camera = False
        if camera_mouse_enabled is not None and type(camera_mouse_enabled) is not bool:
            raise ValueError("Camera mouse preference must be a boolean.")
        # Combined aiming is the default when its controller exists; acquisition
        # still waits for Start, and the user may explicitly choose EEG-only use.
        self._cnn_camera_enabled = pointer_controller is not None if camera_mouse_enabled is None else camera_mouse_enabled
        self._camera_changing = False
        self._camera_epoch = 0  # Reject camera observations spanning an ownership change.
        self._camera_request_serial = 0  # Request admission cancels older enables waiting on server configuration.
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
        self._gap_count = 0
        self._last_valid_counter = None
        self._eeg_counter_reanchor = False  # A reported discontinuity needs a new watermark and then a later advancing packet.
        self._last_valid_eeg_at = -math.inf
        self._model_source = None  # Describe the selected archive without changing its retained validation evidence.
        self._hand_tracking_suspended = False  # Missing hands pause input once, then resume through a fresh pointing dwell.
        self._decoder = FastCnnBlinkDecoder(clock=self.clock)  # One fixed model family; saved weights are loaded only by explicit request.
        self._blink_reset_serial = 0  # State changes invalidate a blink pair without taking the decoder lock in reverse order.
        self._blink_reset_pending = True  # The next stream update establishes a current-sample watermark before detecting events.
        self._blink_last_event_id = None  # Consume each event once, including events observed while disarmed or blocked.
        self._blink_dispatch_id = None  # Only an event approved by the current service update can reach the mouse boundary.
        self._decoder_lock = threading.RLock()
        self._decoder_status = self._decoder.status()
        self._decoder_epoch = 0
        self._eeg_armed = False
        self._eeg_signal_healthy = False
        self._eeg_prediction = None
        self._eeg_prediction_at = -math.inf
        self._eeg_prediction_session = None
        self._eeg_pending = None
        self._eeg_last_trial_end = -math.inf
        self._eeg_trial_sequence = 0
        self._eeg_last_result = None
        self._eeg_reason = "Collect labelled training trials, freeze the model, then collect held-out validation trials."
        self._eeg_last_disarm_reason = None  # Keep the first actual revocation visible through later prediction refreshes.
        self._eeg_quality_pause_reason = None  # A recoverable fast-CNN artifact pauses only EEG output until fresh rest.
        self._eeg_guard_token = None
        self._eeg_guard_until = -math.inf
        self._eeg_retired_guard_tokens = set()
        self._reset_eeg_debounce()


    @staticmethod
    def _error_of(result):
        if isinstance(result, dict):
            return result.get("error")
        return "Device returned an invalid result."

    @staticmethod
    def _camera_cleanup_unconfirmed(result):
        """Native capture ownership ends only after an explicitly completed release."""
        if not isinstance(result, dict):
            return True
        resource = result.get("camera_resource")
        return bool(result.get("stopping") or result.get("tracker_cleanup_pending")
                    or result.get("camera_cleanup_pending")
                    or isinstance(resource, dict) and resource.get("cleanup_pending"))


    def _result(self, error=None):
        with self._lock:
            self._revision += 1
            result = self.status()
        if error:
            result["error"] = str(error)
        else:
            result["ok"] = True
        return result


    def _camera_required(self):
        return self._cnn_camera_enabled  # Camera ownership is optional; blink inference never needs a visible hand.

    def _hand_pointer_enabled(self):
        return self._cnn_camera_enabled  # A single existing hand controller owns aiming, pinch clicks and dragging.

    def _eeg_cooldown_seconds(self):
        return self.EEG_COOLDOWN_SECONDS  # Match the retained one-second decoder; never queue an event during cooldown.

    def camera_request_epoch(self):
        """Admit a camera request before waiting for the shared camera configuration lock."""
        with self._lock:
            self._camera_request_serial += 1
            return self._camera_request_serial


    def camera_request_is_current(self, epoch):
        with self._lock:
            return type(epoch) is int and epoch == self._camera_request_serial


    def eeg_camera(self, enabled, request_epoch=None):
        """Change optional CNN hand aiming without restarting EEG or replacing its model."""
        with self._lock:
            if type(enabled) is not bool:
                return self._result("Camera mouse requires a boolean enabled value.")
            if request_epoch is None:
                request_epoch = self.camera_request_epoch()
            if not self.camera_request_is_current(request_epoch):
                return self._result("Camera mouse request was superseded by a newer request or Stop.")
            if self._camera_changing or self._state not in ("stopped", "running"):
                return self._result("Wait for the current device operation to finish before changing camera mouse.")
            if self._eeg_pending:
                return self._result("Finish the current trial before changing camera mouse.")
            if self._click_cleanup_pending or self._pointer_cleanup_pending:
                return self._result("Stop again to confirm input cleanup before changing camera mouse.")
            if enabled and self.pointer_controller is None:
                return self._result("Camera mouse requires an EEG-compatible hand pointer controller.")
            self._arm_epoch += 1
            self._disarm_eeg_locked("Camera mouse changed; explicitly arm again after fresh EEG is ready.")
            self._camera_changing = True  # Revoke input before waiting for native operations.
            self._camera_epoch += 1
        try:
            with self._operation_lock:
                with self._action_lock:
                    error = self._cleanup_hand_pointer()
                if error:
                    return self._result("Hand pointer cleanup unconfirmed: " + error)
                with self._lock:
                    if not self.camera_request_is_current(request_epoch):
                        return self._result("Camera mouse request was superseded by a newer request or Stop.")
                    if self._state == "stopped" and not self._owns_camera and not self._owns_eeg:
                        self._cnn_camera_enabled = enabled
                        return self._result()
                    if self._state != "running" or self._halt.is_set():
                        return self._result("Preview stopped while changing camera mouse.")
                    if enabled == self._cnn_camera_enabled and self._owns_camera == enabled:
                        return self._result()  # Same-state requests still revoke authorization.
                if enabled and (self.gesture_busy() or self.camera.is_running()):
                    return self._result("Turn off the normal camera and Hand Mouse controls before enabling camera mouse.")
                with self._lock:
                    if self._halt.is_set() or not self.camera_request_is_current(request_epoch):
                        return self._result("Preview stopped while changing camera mouse.")
                    self._camera = self._camera_display = {}
                    self._camera_sequence = None
                    self._camera_changed_at = self._last_camera_fresh_at = -math.inf
                    self._camera_ready = False
                    self._camera_started_at = self.clock()
                    if enabled:
                        self._owns_camera = True  # Failed native startup must still receive owned cleanup.
                try:
                    result = self.camera.start() if enabled else self.camera.stop()
                    error = self._error_of(result)
                    if error or not enabled and self._camera_cleanup_unconfirmed(result):
                        raise RuntimeError(error or "Camera cleanup is still pending.")
                    with self._lock:
                        self._owns_camera = enabled
                        self._cnn_camera_enabled = enabled
                        self._camera_started_at = self.clock()
                        cancelled = self._halt.is_set()
                        superseded = not self.camera_request_is_current(request_epoch)
                    if cancelled:
                        return self._stop_owned("Preview stopped while changing camera mouse.")
                    if superseded:
                        if enabled:
                            cleanup = self.camera.stop()
                            if self._error_of(cleanup) or self._camera_cleanup_unconfirmed(cleanup):
                                raise RuntimeError(self._error_of(cleanup) or "Superseded camera cleanup is still pending.")
                            with self._lock:
                                self._owns_camera = self._cnn_camera_enabled = False
                        return self._result("Camera mouse request was superseded; camera input remains disarmed.")
                except Exception as exc:
                    return self._stop_owned("Camera mouse change failed: " + str(exc), failed=True)
                return self._result()
        finally:
            with self._lock:
                self._camera_changing = False
                self._camera_epoch += 1
                self._revision += 1


    def start(self, *, request_epoch=None):
        with self._operation_lock:
            with self._lock:
                if request_epoch is not None and not self.camera_request_is_current(request_epoch):
                    return self._result("Preview start was superseded by a newer camera request or Stop.")
                if self._state in ("starting", "running"):
                    return self._result()
                if self._owns_camera or self._owns_eeg or self._click_cleanup_pending or self._pointer_cleanup_pending or self._pointer_needs_cleanup:
                    return self._result("Stop again to confirm device cleanup before restarting.")
            if self.gesture_busy() or self._camera_required() and self.camera.is_running():
                return self._result("Turn off the normal camera and Hand Mouse controls before starting this preview.")
            existing = self.brainbit.acquisition_snapshot()
            if existing.get("state") in ("starting", "running", "stopping"):
                return self._result("Stop the existing EEG/contact acquisition before starting this preview.")
            with self._lock:
                if request_epoch is not None and not self.camera_request_is_current(request_epoch):
                    return self._result("Preview start was superseded by a newer camera request or Stop.")
                self._generation += 1
                self._halt.clear()
                self._clear_live()
                self._state = "starting"
                self._reason = ("Starting camera and EEG; controls remain disarmed." if self._camera_required()
                                else "Starting EEG; controls remain disarmed.")  # An unchecked camera preference acquires EEG only.
                self._error = None
                self._started_at = self.clock()
                self._owns_eeg = True
            try:
                result = self.brainbit.start_acquisition(mode="signal")
                if self._error_of(result):
                    raise RuntimeError(self._error_of(result))
                with self._lock:
                    cancelled = self._halt.is_set() or request_epoch is not None and not self.camera_request_is_current(request_epoch)
                    if not cancelled:
                        # Neither stream's first-data budget includes time spent
                        # waiting for the native start command acknowledgement.
                        self._started_at = self.clock()
                        self._owns_camera = self._camera_required()
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
            tick_started = self.clock()
            try:
                self._tick()
            except Exception as exc:
                if generation == self._generation:
                    self._fail("Preview failed: " + str(exc), generation)
                break
            delay = max(0.0, self.HAND_TICK_SECONDS - max(0.0, self.clock() - tick_started))
            self._halt.wait(delay)  # Thirty-Hz service polling remains interruptible even without the optional camera.

    def stop(self):
        # Revoke input immediately even while a pending Start owns the lifecycle lock.
        with self._lock:
            self._camera_request_serial += 1
            self._retire_eeg_guard_locked()
            self._disarm_eeg_locked("Preview stopped.")
            self._halt.set()
            self._arm_epoch += 1
            self._revision += 1
            owns_eeg = self._owns_eeg
        # The adapter explicitly supports cancelling a pending native operation.
        # Do this before waiting for the lifecycle lock held by Start.
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
            self._halt.set()
            self._generation += 1
            owns_camera, owns_eeg = self._owns_camera, self._owns_eeg
            self._clear_live()
        errors = []
        with self._action_lock:  # Finish hand movement before acknowledging that the preview stopped.
            pointer_error = self._cleanup_hand_pointer()  # Only cleanup is allowed here; no camera frame is processed.
            if pointer_error:
                errors.append("Hand pointer cleanup unconfirmed: " + pointer_error)  # Keep Stop available for an explicit retry.
        if self.click_controller is not None:  # Always retry release cleanup, including after an earlier failure.
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
                if error or self._camera_cleanup_unconfirmed(result):
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
        """Discard sensor evidence while retaining the frozen model and its provenance."""
        self._revision += 1
        self._arm_epoch += 1
        self._camera = {}
        self._camera_display = {}
        self._eeg = {}
        self._waveform = []
        self._camera_sequence = None
        self._camera_changed_at = self._last_camera_fresh_at = self._last_eeg_live_at = -math.inf
        self._camera_ready = False
        self._observed_at = -math.inf
        self._gap_count = 0
        self._last_valid_counter = None
        self._eeg_counter_reanchor = False
        self._last_valid_eeg_at = -math.inf
        self._disarm_eeg_locked("Live samples cleared; fresh current-session EEG is required before arming.")
        self._eeg_signal_healthy = False
        self._eeg_prediction = None
        self._eeg_prediction_at = -math.inf
        self._eeg_prediction_session = None
        if self._eeg_pending:
            self._eeg_last_result = "EEG trial cancelled: preview stopped."
        self._eeg_pending = None
        self._eeg_guard_token = None
        self._eeg_guard_until = -math.inf

    def _fail(self, reason, generation):
        with self._lock:
            if self._halt.is_set() or generation != self._generation:
                return
            self._disarm_eeg_locked(reason)
            self._revision += 1
            self._halt.set()
        with self._operation_lock:
            with self._lock:
                if generation != self._generation or self._state not in ("starting", "running"):
                    return
            self._stop_owned(reason, failed=True)


    @staticmethod
    def _eeg_metadata(acquisition):
        return copy.deepcopy({key: acquisition.get(key) for key in
                              ("state", "mode", "session_id", "channels", "nominal_hz", "units", "stats", "contact_precheck")})


    @staticmethod
    def _eeg_display_snapshot(signal):
        """Derive the bounded display view from one already-private full-rate snapshot."""
        packets = signal.get("samples") or []
        if len(packets) <= 250:
            return signal
        stride = math.ceil(len(packets) / 250)
        display = packets[::stride]
        if display[-1] is not packets[-1]:
            if len(display) == 250:
                display[-1] = packets[-1]
            else:
                display.append(packets[-1])
        return {**signal, "samples": display}  # Never mutate the adapter's independent full-rate decoder input.


    @staticmethod
    def _camera_timing(camera):
        """Expose only finite pipeline timings, aligned with the source camera observation."""
        timing = camera.get("timing")
        if not isinstance(timing, dict):
            return {}
        result = {}
        for key in ("capture_read_ms", "frame_wait_ms", "inference_ms", "frame_age_ms"):
            value = timing.get(key)
            if type(value) in (int, float):
                try:
                    if math.isfinite(value) and value >= 0:
                        result[key] = value
                except OverflowError:
                    pass
        dropped = timing.get("dropped_frames")
        if type(dropped) is int and 0 <= dropped <= 2 ** 53 - 1:
            result["dropped_frames"] = dropped
        return result


    @staticmethod
    def _age(value):
        return float(value) if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 else math.inf


    @staticmethod
    def _valid_row(row, width):
        if not isinstance(row, dict):
            return False  # Malformed adapter data cannot refresh an arming or acquisition timestamp.
        values = row.get("samples") or []
        counter = row.get("counter")
        return (width > 0 and isinstance(values, (list, tuple)) and len(values) == width
                and type(counter) is int and 0 <= counter <= 0xFFFFFFFF
                and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) for value in values))


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
            owns_camera = self._owns_camera and not self._camera_changing
            camera_epoch = self._camera_epoch
        signal_reader = getattr(self.brainbit, "acquisition_signal_window", None)
        if callable(signal_reader):
            signal = signal_reader()
            acquisition = (self._eeg_display_snapshot(signal) if isinstance(signal, dict)
                           else self.brainbit.acquisition_snapshot())
        else:
            acquisition = self.brainbit.acquisition_snapshot()
            signal = None
        # EEG copies/statistics can be expensive. Read the latest completed hand
        # frame afterward, so its landmarks and freshness are current when used.
        try:
            camera_display = self.camera.preview() if owns_camera else {}  # Keep the HUD image and its overlay in one frame.
            tracking_reader = getattr(self.camera, "tracking_snapshot", None)  # Older adapters may expose only a preview.
            camera = (tracking_reader() if owns_camera and callable(tracking_reader)
                      else camera_display)  # Read completed inference directly; cached JPEG sequences cannot throttle the pointer.
        except Exception:
            with self._lock:
                if self._camera_required() and not self._camera_changing and camera_epoch == self._camera_epoch:
                    raise
            camera = camera_display = {}
        now = self.clock()
        failure = None
        with self._lock:
            if self._halt.is_set() or generation != self._generation or self._state not in ("starting", "running"):
                return
            camera_current = camera_epoch == self._camera_epoch and not self._camera_changing
            camera_required = self._camera_required() and camera_current
            if not camera_current:
                camera = camera_display = {}  # Ignore frames read across a camera-only lifecycle change.
            self._revision += 1
            if self._eeg_armed and not self._guard_ready(now):
                self._disarm_eeg_locked("Global Escape guard expired; prepare the guard and explicitly rearm EEG control.")
            if self._eeg.get("session_id") and self._eeg.get("session_id") != acquisition.get("session_id"):
                self._disarm_eeg_locked("EEG session changed; explicitly bind the saved model and rearm.")
                self._eeg_prediction = None
                self._eeg_prediction_session = None
                self._eeg_pending = None
                self._last_valid_counter = None
                self._eeg_counter_reanchor = False
            if now - self._observed_at > self.TICK_SECONDS * 3 and self._eeg_armed:
                self._disarm_eeg_locked("EEG processing paused; wait for fresh signal, then explicitly rearm.")
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
            valid_eeg_age = max(eeg_age, now - self._last_valid_eeg_at)  # Either an old receipt or a stalled counter can make the stream stale.
            eeg_live = (acquisition.get("state") == "running" and acquisition.get("mode") == "signal"
                        and valid_eeg_age <= self.EEG_FRESH_SECONDS and bool(rows)
                        and self._valid_row(rows[-1], width))
            if eeg_live:
                self._last_eeg_live_at = now - valid_eeg_age
            self._eeg_signal_healthy = bool(eeg_live and not gap)
            if not self._eeg_signal_healthy:
                self._disarm_eeg_locked("EEG signal interrupted; wait for fresh uninterrupted samples, then rearm.")
                if self._eeg_pending:
                    self._eeg_pending = None
                    self._eeg_last_result = "Blink trial discarded: EEG became unavailable."
            if eeg_live or camera_required and camera_fresh:
                self._state = "running"
            self._waveform = copy.deepcopy(rows[-250:]) if eeg_live else []  # Status never exposes a stale waveform as live data.
            self._reason = ("Double-blink control armed; optional hand aiming is independent of blink detection."
                            if self._eeg_armed else "EEG preview only; desktop actions are disarmed.")
            elapsed = now - self._started_at
            if (acquisition.get("state") == "error" or acquisition.get("error")):
                failure = "EEG acquisition failed: " + str(acquisition.get("error") or "device error")
            elif camera_required and not camera.get("running") and camera.get("state") in ("error", "unavailable", "off", "stopping"):
                # preview() already exposes the capture worker's cached reason;
                # do not replace model/startup failures with a generic timeout.
                failure = "Camera stopped: " + str(camera.get("hint") or camera.get("error")
                                                   or camera.get("state"))
            elif not eeg_live and (
                    (not math.isfinite(self._last_eeg_live_at) and elapsed >= self.STARTUP_GRACE_SECONDS)
                    or (math.isfinite(self._last_eeg_live_at) and now - self._last_eeg_live_at >= self.EEG_LOSS_SECONDS)):
                failure = "EEG stopped or samples stayed stale; preview stopped."
            elif camera_required and not self._camera_ready and now - self._camera_started_at >= self.CAMERA_STARTUP_SECONDS:
                failure = "Camera startup timed out before the first fresh frame; preview stopped."
            elif camera_required and self._camera_ready and not camera_fresh and now - self._last_camera_fresh_at >= self.CAMERA_LOSS_SECONDS:
                failure = "Camera frames stayed stale; preview stopped."
        if failure:
            self._fail(failure, generation)
        else:
            if self._hand_pointer_enabled() and camera_current:
                self._update_hand_pointer(generation)
            self._process_blink(signal, now, generation)
        self._finish_disarmed_hand_input()  # Every automatic revocation must release a finger drag before this tick completes.


    def _reset_eeg_debounce(self, *, reset_blink_stream=True):
        """Cancel pending output; lifecycle resets also discard every buffered waveform."""
        self._eeg_neutral_ready = False
        self._blink_reset_serial += 1  # A result computed before cancellation cannot publish after it.
        if reset_blink_stream:
            self._blink_reset_pending = True  # The decoder consumes a new packet watermark before collecting fresh context.
            self._cnn_cancel_pending = False
            self._cnn_finger_active = False
        self._blink_dispatch_id = None  # An already computed command is never held for later delivery.

    def _cnn_finger_transition_locked(self, active):
        """Keep the rolling CNN context, while excluding gestures crossing finger ownership."""
        if active != self._cnn_finger_active:
            self._cnn_finger_active = active
            self._reset_eeg_debounce(reset_blink_stream=False)
            self._cnn_cancel_pending = True  # The decoder lock applies the evidence watermark before the next update.


    def _disarm_eeg_locked(self, reason, retire_guard=True):
        if self._eeg_armed:
            self._eeg_last_disarm_reason = reason
            self._arm_epoch += 1
            self._pointer_needs_cleanup = self.pointer_controller is not None  # Native release waits until the caller has dropped the state lock.
        if retire_guard:
            self._retire_eeg_guard_locked()
        self._eeg_armed = False
        self._eeg_quality_pause_reason = None
        self._eeg_reason = reason
        self._reset_eeg_debounce()
        self._pointer_at = -math.inf  # Revoking EEG authorization revokes hand aiming at the same instant.
        self._pointer_sequence = None  # A new explicit arm must start from current camera observations.
        self._hand_tracking_suspended = False  # The next explicit arm begins a new hand tracking lifecycle.
        self._pointer_status = {"state": "mouse_paused", "pointer_ready": False, "finger_active": False,
                                "button_down": False, "hint": reason}  # Cleanup flags separately retain any unresolved native ownership.


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


    def _hand_control_authorized(self, now):
        """The existing explicit arm permits camera input through temporary EEG artifacts."""
        return bool(self._hand_pointer_enabled() and self._owns_camera and self._eeg_control_authorized(now))
        # Actual moves still recheck fresh camera frames; hand loss releases an owned drag.

    def _finger_blocks_eeg(self, now):
        """Optional aiming never gates EEG on a hand pose, only on competing native input."""
        return bool(self._hand_pointer_enabled() and (self._finger_owns_input(self._pointer_status)
                    or now - self._last_finger_at < self._eeg_cooldown_seconds()))


    def _finger_owns_input(self, feedback):
        active = feedback.get("finger_active")
        if type(feedback.get("finger_action_active")) is bool:
            active = feedback["finger_action_active"]  # Approaching a pinch alone cannot erase CNN evidence.
        return active is True or feedback.get("button_down") is True


    def _cnn_event_eligible(self, now, prediction=None):
        prediction = self._eeg_prediction if prediction is None else prediction
        return bool(self._eeg_quality_pause_reason is None
                    and (prediction or {}).get("eligible_for_event", True) is True
                    and now - self._last_dispatch_at >= self._eeg_cooldown_seconds())


    def _recoverable_cnn_amplitude_fault(self, prediction):
        """Only the fast decoder's checked amplitude fault can retain authorization."""
        return (prediction.get("valid") is False and prediction.get("recoverable") is True
                and prediction.get("fault_category") == "signal_quality"
                and prediction.get("fault_code") == "signal_amplitude")


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
            self._cnn_finger_transition_locked(False)  # Releasing an actual drag requires newer EEG; ordinary hand loss does not.
            self._pointer_status = copy.deepcopy(result)  # Publish the adapter's actual paused or fault state.
            self._pointer_at = -math.inf
            self._pointer_sequence = None
            self._hand_tracking_suspended = True  # Repeated missing frames need no repeated reset/native release.
            if error:
                self._pointer_cleanup_pending = True  # Cleanup failures never recover automatically.
                self._disarm_eeg_locked("Hand input cleanup failed: " + error)
                self._pointer_needs_cleanup = False  # This attempt already failed; retain the fault for an explicit Stop retry.
            self._revision += 1


    def _cleanup_hand_pointer(self, reset=False):
        """Forget hand engagement without moving or clicking; caller owns the action lock."""
        if self.pointer_controller is None:
            return None  # EEG-only clicking needs no hand pointer controller.
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
            if not self._hand_pointer_enabled() or not self._eeg_armed:
                return  # EEG-only and disarmed preview never queue a pointer operation.
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
                self._pointer_sequence = sequence  # Consume the frame before calling a potentially reentrant adapter.
            try:
                feedback = self.pointer_controller.update(points, now, hold=False)  # Finger clicks and drags take priority even while EEG holds ordinary aiming.
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
                    self._finger_click_count += 1  # Count completed native finger input independently from blink clicks.
                finger_active = self._finger_owns_input(feedback) or new_click
                self._cnn_finger_transition_locked(finger_active)
                if finger_active:
                    self._last_finger_at = now
                    self._last_dispatch_at = now  # EEG must wait through cooldown and a new rest/select/rest cycle after finger input.
                    self._eeg_reason = "Finger control takes priority; rest again before a new EEG selection."  # Explain why a queued EEG click was cancelled.
                if feedback.get("error"):
                    self._pointer_cleanup_pending = bool(feedback.get("error"))  # Input errors require confirmed cleanup before reuse.
                    self._disarm_eeg_locked(feedback.get("error") or "Hand target lost; point again and explicitly rearm.")  # Never release a queued click after losing its target.
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
        """Initial arming requires the fixed validated model and a fresh usable prediction."""
        return bool(self.click_controller is not None and not self._camera_changing
                    and not self._click_cleanup_pending and not self._pointer_cleanup_pending and not self._pointer_needs_cleanup
                    and (not require_guard or self._guard_ready(now)) and not self._eeg_pending
                    and self._eeg_inputs_fresh(now) and self._decoder_status.get("trained") is True
                    and self._decoder_status.get("passed_validation") is True and self._decoder_status.get("arm_eligible") is True
                    and self._eeg_prediction_session == self._eeg.get("session_id")
                    and now - self._eeg_prediction_at <= self.EEG_PREDICTION_HOP * 2
                    and self._confident_eeg_prediction(self._eeg_prediction or {}))

    def _eeg_control_authorized(self, now):
        """Keep explicit authorization through fresh-signal artifact recovery, never device faults."""
        return bool(self._eeg_armed and not self._camera_changing and self._guard_ready(now)
                    and self._eeg_inputs_fresh(now) and not self._click_cleanup_pending
                    and not self._pointer_cleanup_pending and not self._pointer_needs_cleanup and not self._eeg_pending
                    and self._decoder_status.get("trained") is True and self._decoder_status.get("passed_validation") is True
                    and self._decoder_status.get("arm_eligible") is True)

    def _confident_eeg_prediction(self, prediction):
        return (prediction.get("valid") is True and prediction.get("ready") is True
                and prediction.get("label") in ("rest", "select"))

    def eeg_arm(self, enabled, guard_token=None):
        if enabled is False:  # Escape must revoke queued input before waiting for an already executing click.
            with self._lock:  # Publish revocation immediately, even while the actuator owns the action lock.
                self._arm_epoch += 1  # Invalidate any dispatch waiting to enter its final authorization check.
                self._disarm_eeg_locked("EEG control disarmed.")  # Retire the guard and discard a pending selection.
        with self._action_lock, self._lock:  # Finish any previous press/release before changing mouse authorization.
            if not isinstance(enabled, bool):
                return self._result("EEG arm requires a boolean enabled value.")
            self._arm_epoch += 1
            if not enabled:
                self._retire_eeg_guard_locked()
            self._disarm_eeg_locked("EEG control disarmed.", retire_guard=not enabled)
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
                if self._hand_pointer_enabled():
                    pointer_error = self._cleanup_hand_pointer(reset=True)  # A fresh arm always requires a new deliberate pointing gesture.
                    if pointer_error:
                        return self._result(pointer_error)  # Never arm a hand controller whose cleanup is unconfirmed.
                if self.click_controller is not None:
                    cleanup_error = self._cleanup_eeg_click(reset=True)  # Explicit mouse arming may clear a latched actuator failure.
                    if cleanup_error:
                        return self._result(cleanup_error)  # Require a successful release before granting new authorization.
                if not self._can_eeg_arm(self.clock(), require_guard=True):
                    self._disarm_eeg_locked("EEG arming prerequisites changed; prepare a new Escape guard after recovery.")
                    return self._result("Arming requires a frozen validated fast blink model, fresh EEG and confirmed input cleanup.")
                self._eeg_armed = True
                self._eeg_last_disarm_reason = None  # Only successful explicit arming acknowledges the prior revocation.
                self._eeg_quality_pause_reason = None
                self._eeg_reason = "Armed. Double blink deliberately for one left click. Escape disarms; a visible hand is not required."
            return self._result()


    def eeg_export_cnn_model(self):
        """Capture the fixed fast model under its lock; the server writes a bounded inert archive."""
        with self._decoder_lock:
            return self._decoder.export_state()

    def eeg_use_saved_cnn_model(self, decoder, source=None):
        """Bind an explicitly selected CNN archive without changing its validation."""
        if not isinstance(decoder, FastCnnBlinkDecoder):
            return self._result("A restored CNN blink decoder is required.")
        with self._decoder_lock, self._lock:
            if (self._state != "running"
                    or self._halt.is_set() or self._eeg_pending):
                return self._result("Start EEG in CNN blink mode and finish any trial before loading its model.")
            expected = FastCnnBlinkDecoder
            if not isinstance(decoder, expected) or decoder.feature_profile != expected.feature_profile:
                return self._result("The saved CNN profile does not match the selected window length. Only the one-second fast blink model is supported.")
            context = {"session_id": self._eeg.get("session_id"), "channels": copy.deepcopy(self._eeg.get("channels")),
                       "nominal_hz": self._eeg.get("nominal_hz"), "units": self._eeg.get("units"),
                       "feature_profile": decoder.feature_profile}
            try:
                bound = decoder.bind_prediction_context(context)
                if isinstance(bound, dict) and bound.get("error"):
                    return self._result(bound["error"])
            except (ValueError, TypeError, AttributeError) as exc:
                return self._result(str(exc))
            self._arm_epoch += 1
            self._decoder_epoch += 1
            self._disarm_eeg_locked("CNN model loaded. Wait for fresh EEG, then explicitly arm after validation passes.")
            self._decoder = decoder
            self._decoder_status = decoder.status()
            self._model_source = str(source)[:512] if source is not None else None
            self._eeg_prediction = None
            self._eeg_prediction_at = -math.inf
            self._eeg_prediction_session = None
            self._blink_last_event_id = None
            self._eeg_last_result = "CNN model restored with its original held-out validation. EEG-only clicks remain disarmed."
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
        """Prepare for three seconds, then retain exactly one second of new EEG packets."""
        with self._lock:
            now = self.clock()
            if self._camera_changing:
                return self._result("Wait for the camera change before collecting a trial.")
            if label not in ("rest", "select") or phase not in ("train", "validate"):
                return self._result("EEG trial requires rest/select and train/validate.")
            if self._eeg_pending or now < self._eeg_last_trial_end:
                return self._result("Finish the current trial before recording another.")
            if not self._eeg_inputs_fresh(now) or type(self._last_valid_counter) is not int:
                return self._result("Start EEG and wait for fresh packets before collecting a trial.")
            if phase == "train" and self._decoder_status.get("trained"):
                return self._result("The model is frozen. Collect validation or reset before training again.")
            if phase == "validate" and not self._decoder_status.get("trained"):
                return self._result("Freeze a trained model before collecting separate validation.")
            if phase == "validate" and self._decoder_status.get("state") in ("validated", "validation_failed"):
                return self._result("Held-out validation is complete and frozen. Reset to begin a new experiment.")
            self._arm_epoch += 1
            self._disarm_eeg_locked("Guided EEG trial; all desktop actions are disarmed.")
            self._eeg_trial_sequence += 1
            self._eeg_pending = {"label": label, "phase": phase, "start": now, "end": now + 4,
                                 "stage": "prepare", "prepare_until": now + 3, "capture_seconds": 1.0,
                                 "blink_rows": [], "blink_counter": None, "blink_start_counter": None,
                                 "blink_required": 250, "blink_error": None,
                                 "id": "eeg-trial-" + str(self._eeg_trial_sequence),
                                 "generation": self._generation, "decoder_epoch": self._decoder_epoch}
            if label == "rest":
                counts = self._decoder_status.get("train_counts" if phase == "train" else "validation_counts") or {}
                self._eeg_pending["cue_subtype"] = ("quiet", "single_blink", "movement")[int(counts.get("rest", 0)) % 3]
            self._eeg_last_trial_end = now + 4
            self._eeg_last_result = "Prepare for three seconds. Follow the one-second recording cue when it appears."
        with self._action_lock:
            return self._result(self._cleanup_hand_pointer())  # End any owned drag before acknowledging a cue.

    def eeg_train(self):
        """Freeze the selected training data without collecting or passing validation.

        Fitting may ignore atypical blink examples, but the decoder retains their
        records. A failed fit leaves those records available for another attempt.
        """
        with self._decoder_lock:  # Model loading/reset cannot replace evidence between the precheck and fitting.
            with self._lock:
                self._arm_epoch += 1
                self._decoder_epoch += 1
                self._retire_eeg_guard_locked()
                self._disarm_eeg_locked("Training freezes the EEG model; separate validation is required before arming.")
                if self._eeg_pending:
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
        """Explicitly discard this experiment; saved archives are never deleted by Reset."""
        with self._decoder_lock:
            with self._lock:
                self._arm_epoch += 1
                self._decoder_epoch += 1
                self._disarm_eeg_locked("Fast blink training and validation reset.")
                self._eeg_pending = None
                self._eeg_prediction = None
                self._eeg_prediction_at = -math.inf
                self._eeg_last_trial_end = -math.inf
                self._eeg_last_result = "Fast blink model, training examples and validation cleared."
                self._model_source = None
            self._decoder.reset()
            with self._lock:
                self._decoder_status = self._decoder.status()
        with self._action_lock:
            return self._result(self._cleanup_hand_pointer())

    def _process_blink(self, signal, now, generation):
        """Collect complete labelled cues and consume each completed blink pair once.

        Collection counts contiguous device packets instead of callback arrival
        timestamps. The short grace period waits for delivery only; it never adds
        extra samples to a cue. The decoder separately checks waveform quality,
        and the final dispatch path rechecks arming before any real mouse click.
        """
        dispatch_id = None  # No native input can occur until all model, stream, and hand checks succeed below.
        with self._decoder_lock:  # Reset and update the causal detector under one lock; never process the same stream concurrently.
            with self._lock:
                if generation != self._generation or self._halt.is_set():
                    return
                if (not signal or signal.get("state") != "running" or signal.get("mode") != "signal"
                        or signal.get("session_id") != self._eeg.get("session_id") or not self._eeg_signal_healthy):
                    self._eeg_prediction = None
                    self._decoder.reset_stream()  # Transport loss discards every packet and still revokes all native input.
                    self._decoder_status = self._decoder.status()
                    self._disarm_eeg_locked("Blink clicks disarmed: full-rate EEG was interrupted.")
                    if self._eeg_pending:
                        self._eeg_pending = None
                        self._eeg_last_result = "Blink trial discarded: full-rate EEG was interrupted."
                    return
                rows = signal.get("samples") or []
                # The snapshot is private; trial rows and the decoder retain
                # their own evidence without another full-buffer copy here.
                pending = self._eeg_pending
                if pending:
                    if pending.get("stage") == "prepare" and now >= pending["prepare_until"]:
                        # A complete packet boundary separates preparation from
                        # the labelled clip, even if the BLE callback arrives late.
                        latest = rows[-1] if isinstance(rows, (list, tuple)) and rows else None
                        counter = latest.get("counter") if isinstance(latest, dict) else None
                        if type(counter) is not int or not 0 <= counter <= 0xFFFFFFFF:
                            pending.update(stage="record", start=now, end=now,
                                           blink_error="Blink trial discarded: no valid EEG packet boundary at the recording cue.")
                        else:
                            pending.update(stage="record", start=now, end=now + pending["capture_seconds"],
                                           blink_start_counter=counter)
                        self._eeg_last_trial_end = pending["end"]
                    if now > pending["end"] + self.BLINK_COLLECTION_GRACE and len(pending["blink_rows"]) < pending["blink_required"]:
                        pending["blink_error"] = "Blink trial timed out: the remaining EEG samples arrived after the collection deadline. Accepted trials were kept."
                    # Copy the declared number of contiguous packets; preparation
                    # and later Bluetooth delivery never change the labelled window.
                    for row in (rows if pending.get("stage") != "prepare" else ()):
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
                            pending["blink_error"] = "Blink trial discarded: EEG packet gap in the recorded window. Accepted trials were kept."
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
                cancel_pending = self._cnn_cancel_pending
                self._cnn_cancel_pending = False
            if reset_stream:
                self._decoder.reset_stream()  # First update watermarks buffered samples, so arming cannot replay an earlier blink.
            elif cancel_pending:
                self._decoder.cancel_pending()  # Preserve the rolling context; require evidence newer than the ownership edge.
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
                self._eeg_neutral_ready = prediction.get("ready") is True
                self._revision += 1
                event_id = prediction.get("event_id")
                fresh_event = isinstance(event_id, str) and bool(event_id) and event_id != self._blink_last_event_id
                if fresh_event:
                    self._blink_last_event_id = event_id  # Consume blocked/disarmed detections too; they cannot be queued for a later target.
                self._blink_dispatch_id = None
                if (self._eeg_quality_pause_reason is not None and prediction.get("valid") is True
                        and prediction.get("ready") is True and prediction.get("ready_for_blink") is True
                        and prediction.get("label") == "rest" and not event_id):
                    self._eeg_quality_pause_reason = None  # The decoder rebuilt fresh context and recognized rest after the artifact.
                if not self._eeg_armed:
                    self._eeg_reason = prediction.get("reason") or "Double-blink prediction only; explicitly arm before clicking."
                elif prediction.get("valid") is not True:
                    if self._recoverable_cnn_amplitude_fault(prediction):
                        # The decoder already discarded its waveform and latches.
                        # Cancel service evidence without revoking camera aiming,
                        # finger ownership, or the continuously renewed Escape guard.
                        self._reset_eeg_debounce(reset_blink_stream=False)
                        self._eeg_quality_pause_reason = prediction.get("reason") or "EEG amplitude exceeded the signal guard."
                        self._eeg_reason = "EEG clicks paused: " + self._eeg_quality_pause_reason
                    else:
                        self._disarm_eeg_locked(prediction.get("reason") or "EEG quality failed; wait for recovery and explicitly rearm.")
                elif self._eeg_quality_pause_reason is not None:
                    self._eeg_reason = "EEG clicks paused after a signal artifact. " + (prediction.get("reason") or "Wait for fresh EEG and recognized rest.")
                elif self._finger_blocks_eeg(now):
                    self._eeg_reason = "Finger input takes priority; wait for release and a fresh double blink."
                elif not self._cnn_event_eligible(now, prediction):
                    self._eeg_reason = ("EEG click cooldown; wait briefly before another deliberate double blink."
                                        if now - self._last_dispatch_at < self._eeg_cooldown_seconds() else
                                        prediction.get("reason") or "Waiting for fresh no-command EEG before another double blink.")
                elif not self._can_eeg_arm(now, require_guard=True):
                    self._eeg_reason = prediction.get("reason") or "Waiting for the validated blink detector and fresh EEG."
                elif fresh_event and prediction.get("label") == "select":
                    self._blink_dispatch_id = dispatch_id = event_id  # Final dispatch rechecks this exact event and authorization epoch.
                else:
                    self._eeg_reason = prediction.get("reason") or "Double blink deliberately to click at the current cursor."
                arm_epoch = self._arm_epoch
        if dispatch_id:
            self._dispatch_eeg_click(generation, arm_epoch, dispatch_id)  # Keep native input outside both decoder and state locks.


    def _dispatch_eeg_click(self, generation, arm_epoch, prediction_id):
        """The sole output boundary emits one ordinary left click, never a delayed command."""
        with self._action_lock:  # Serialize press/release with native hand input, Stop and explicit arming.
            with self._lock:
                now = self.clock()
                prediction = self._eeg_prediction or {}
                event_at = prediction.get("event_at")
                cycle_ready = (prediction_id == self._blink_dispatch_id == prediction.get("event_id")
                               and prediction.get("label") == "select" and bool(prediction_id)
                               and isinstance(event_at, (int, float)) and math.isfinite(event_at)
                               and 0 <= now - event_at <= self.EEG_FRESH_SECONDS)
                if (not self._eeg_armed or not self._can_eeg_arm(now, require_guard=True)
                        or self._finger_blocks_eeg(now) or not self._cnn_event_eligible(now, prediction)
                        or generation != self._generation or arm_epoch != self._arm_epoch or not cycle_ready):
                    return  # An expired lease, older acquisition, finger gesture or consumed event forbids native input.
                self._last_dispatch_at = now
                self._reset_eeg_debounce(reset_blink_stream=False)  # The CNN already consumed its rest latch; retain its rolling context.
            try:
                result = self.click_controller.click()
                error = self._error_of(result)
                if error or result.get("ok") is not True:
                    raise RuntimeError(error or "Mouse click was not confirmed.")
                with self._lock:
                    self._click_count += 1  # Count only a confirmed complete press/release, never a prediction.
                    self._eeg_reason = "Mouse click completed. Rest before the next deliberate double blink."
                    self._revision += 1
            except Exception as exc:
                with self._lock:
                    self._disarm_eeg_locked("EEG click failed: " + str(exc))
                self._cleanup_eeg_click()  # Retry release after any input failure; rearming waits for confirmed cleanup.

    def status(self):
        with self._lock:
            now = self.clock()
            elapsed = max(0.0, now - self._observed_at)
            camera = {key: copy.deepcopy(self._camera.get(key)) for key in ("running", "tracked", "sequence", "age_ms", "fps")}
            camera["timing"] = self._camera_timing(self._camera)
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
            eeg_pending = ({"label": self._eeg_pending["label"], "phase": self._eeg_pending["phase"],
                            "remaining_seconds": round(max(0, self._eeg_pending["end"] - now), 1)} if self._eeg_pending else None)
            if eeg_pending:
                trial = self._eeg_pending
                preparing = trial.get("stage") == "prepare"
                eeg_pending.update(stage="prepare" if preparing else "record",
                                   capture_seconds=trial["capture_seconds"],
                                   samples_collected=len(trial["blink_rows"]), samples_required=trial["blink_required"],
                                   prepare_remaining_seconds=round(max(0, trial["prepare_until"] - now), 1))
                if preparing:
                    eeg_pending["remaining_seconds"] = trial["capture_seconds"]
                eeg_pending["total_remaining_seconds"] = round(eeg_pending["prepare_remaining_seconds"] + eeg_pending["remaining_seconds"], 1)
                subtype = trial.get("cue_subtype", "quiet")
                if eeg_pending["label"] == "rest":
                    eeg_pending["cue_subtype"] = subtype
                instruction = ("Blink twice quickly now, then reopen your eyes." if eeg_pending["label"] == "select" else
                               "Make one gentle blink now, then reopen your eyes." if subtype == "single_blink" else
                               "Move your head gently; do not double blink." if subtype == "movement" else
                               "Keep your eyes open and remain relaxed.")
                upcoming = ("blink twice quickly, then reopen your eyes" if eeg_pending["label"] == "select" else
                            "make one gentle blink, then reopen your eyes" if subtype == "single_blink" else
                            "move your head gently without double blinking" if subtype == "movement" else
                            "keep your eyes open and remain relaxed")
                eeg_pending["cue"] = ("Get ready. When the recording cue appears, " + upcoming + "."
                                      if preparing else "Finishing signal collection; keep eyes open." if now >= trial["end"] else instruction)
            return {"revision": self._revision, "state": self._state,
                    "running": self._state in ("starting", "running"),
                    "cleanup_pending": self._click_cleanup_pending or self._pointer_cleanup_pending or self._pointer_needs_cleanup
                    or (self._owns_camera or self._owns_eeg) and self._state in ("error", "stopped", "stopping"),
                    "reason": self._reason, "error": self._error, "control_mode": "eeg_click",
                    "eeg_control": {"armed": self._eeg_control_authorized(now), "method": "cnn_blink_fast", "target": "mouse",
                                    "window_seconds": 1.0, "trial_seconds": 1.0, "click_cooldown_seconds": 1.0,
                                    "can_arm": self._can_eeg_arm(now), "reason": self._eeg_reason,
                                    "last_disarm_reason": self._eeg_last_disarm_reason,
                                    "signal_quality_paused": bool(self._eeg_armed and self._eeg_quality_pause_reason is not None),
                                    "signal_quality_reason": self._eeg_quality_pause_reason,
                                    "can_eeg_click": self._can_eeg_arm(now) and self._cnn_event_eligible(now) and not self._finger_blocks_eeg(now),
                                    "camera_required": self._camera_required(), "camera_mouse_enabled": self._cnn_camera_enabled,
                                    "saved_model_source": self._model_source, "guard_ready": self._guard_ready(now),
                                    "prediction": copy.deepcopy(self._eeg_prediction), "decoder": copy.deepcopy(self._decoder_status),
                                    "pending": eeg_pending, "neutral_ready": self._eeg_neutral_ready,
                                    "hand_pointer": {"required": False, "enabled": self._hand_pointer_enabled(),
                                                     "ready": self._eeg_armed and self._hand_pointer_ready(now),
                                                     "state": self._pointer_status.get("state"), "hint": self._pointer_status.get("hint")},
                                    "click_count": self._click_count, "finger_click_count": self._finger_click_count,
                                    "last_result": self._eeg_last_result},
                    "camera": {**camera, "owned": self._owns_camera}, "eeg": eeg,
                    "timing_note": DISCLAIMER}

    def snapshot(self):
        with self._lock:
            result = self.status()
            if self._state in ("starting", "running"):
                for key in ("running", "tracked", "sequence", "age_ms", "fps", "image", "landmarks", "width", "height"):
                    result["camera"][key] = copy.deepcopy(self._camera_display.get(key))  # Every overlay field refers to the same JPEG frame.
                result["camera"]["timing"] = self._camera_timing(self._camera_display)  # Timings describe the displayed frame, not a newer tracking observation.
                if isinstance(result["camera"].get("age_ms"), (int, float)):
                    result["camera"]["age_ms"] += round(max(0.0, self.clock() - self._observed_at) * 1000)  # Preserve display-frame age between polls.
                result["eeg"]["waveform"] = copy.deepcopy(self._waveform)
            return result
