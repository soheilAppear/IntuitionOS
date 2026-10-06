"""Explicit, ephemeral webcam + EEG preview. EEG never supplies an OS direction.

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


DISCLAIMER = ("Exploratory movement/artifact correlations, not intent decoding. "
              "EEG does not determine or blend into desktop directions. "
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
    CONTACT_SECONDS = 5.0
    TRIAL_SECONDS = 3.0
    MAX_TRIALS = 64

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
                        self._owns_camera = True
                if cancelled:
                    return self._stop_owned("Preview startup cancelled.")
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
        self._observed_at = -math.inf
        self._gap_count = 0
        self._last_valid_counter = None
        self._last_valid_eeg_at = -math.inf
        self._motion.reset()
        if self._pending:
            self._last_trial_result = "Trial cancelled: preview stopped."
        self._pending = None

    def _fail(self, reason, generation):
        with self._lock:
            if self._halt.is_set() or generation != self._generation:
                return
            self._armed = False
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
        camera = self.camera.preview()
        acquisition = self.brainbit.acquisition_snapshot()
        now = self.clock()
        failure = None
        dispatch_direction = None
        arm_epoch = None
        with self._lock:
            if self._halt.is_set() or generation != self._generation or self._state not in ("starting", "running"):
                return
            self._revision += 1
            if now - self._observed_at > self.TICK_SECONDS * 3 and self._armed:
                self._armed = False
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
            eeg_fresh = (acquisition.get("state") == "running" and acquisition.get("mode") == "signal"
                         and eeg_age <= self.EEG_FRESH_SECONDS and advanced and bool(rows)
                         and self._valid_row(rows[-1], width))
            gaps = sum(max(0, int(stats.get(key) or 0)) for key in
                       ("gaps", "queue_drops", "channel_mismatches", "duplicates", "nonfinite"))
            gap = gaps > self._gap_count
            self._gap_count = gaps
            self._healthy = bool(tracked and eeg_fresh and not gap)
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
            if not self._healthy:
                self._armed = False
                self._arm_epoch += 1
                self._motion.reset()
                if self._pending:
                    self._last_trial_result = "Trial discarded: camera/EEG dropout or packet gap."
                    self._pending = None
                if gap:
                    self._raw.clear()
                self._reason = ("Packet gap detected; controls disarmed." if gap else
                                "Waiting for a fresh tracked hand and EEG; controls disarmed.")
            else:
                self._state = "running"
                self._reason = "Armed: mirrored camera direction, with EEG freshness gate." if self._armed else "Preview only; controls are disarmed."
                direction = None
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
                if direction and self._armed and self.dispatch is not None:
                    dispatch_direction, arm_epoch = direction, self._arm_epoch
                self._finish_trial(now)
            elapsed = now - self._started_at
            if acquisition.get("state") == "error" or acquisition.get("error"):
                failure = "EEG acquisition failed: " + str(acquisition.get("error") or "device error")
            elif elapsed > self.STARTUP_GRACE_SECONDS:
                if not camera_fresh:
                    failure = "Camera stopped or frames became stale; preview stopped."
                elif not eeg_fresh:
                    failure = "EEG stopped or samples became stale; preview stopped."
        if failure:
            self._fail(failure, generation)
        elif dispatch_direction:
            self._dispatch_direction(dispatch_direction, generation, arm_epoch)

    def _dispatch_direction(self, direction, generation, arm_epoch):
        # Never hold the status lock across an OS action. Epoch checks reject an
        # event whose ownership/arming changed before it reached this boundary.
        with self._action_lock:
            with self._lock:
                if (not self._armed or not self._can_arm(self.clock())
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
            self._arm_epoch += 1
            self._motion.reset()
            if enabled:
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
        return (self._state == "running" and self._healthy and not self._pending
                and not self._halt.is_set() and elapsed <= self.TICK_SECONDS * 3
                and camera_age <= self.CAMERA_FRESH_SECONDS and eeg_age <= self.EEG_FRESH_SECONDS
                and now - self._last_valid_eeg_at <= self.EEG_FRESH_SECONDS)

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
            return {"revision": self._revision, "state": self._state, "running": self._state in ("starting", "running"),
                    "cleanup_pending": (self._owns_camera or self._owns_eeg) and self._state in ("error", "stopped", "stopping"),
                    "armed": self._armed and self._can_arm(now), "can_arm": self._can_arm(now),
                    "reason": self._reason, "error": self._error,
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
