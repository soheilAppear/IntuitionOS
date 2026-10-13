"""Share deliberate camera navigation with the guarded camera + EEG pointer.

The preview camera only provides landmarks. This adapter owns the short-lived
choice between the mouse and a horizontal desktop swipe, so a hand gesture can
never race a blink click or inherit contacts from a previous arm session.
"""

import math
import threading
import time

from .desktop import DesktopSwipeController
from .gestures import (
    FOUR_FINGER, NONE, PINCH, UNKNOWN, WRIST, MeasuredGestureTracker,
    _holds_navigation_pose, classify, hand_scale, is_navigation_pose, is_pinching,
)
from .hand_control import HandControls


class EegHandController:
    """Coordinate the existing mouse and measured swipe without owning a camera."""

    eeg_compatible = True

    def __init__(self, mouse, *, active, mode="auto", travel_palms=1.2,
                 clock=time.monotonic, dispatch=None,
                 desktop_factory=DesktopSwipeController):
        if getattr(mouse, "eeg_compatible", False) is not True:
            raise ValueError("Combined navigation requires an EEG-compatible mouse.")
        self.mouse = mouse
        self._active = active
        self._clock = clock
        self._lock = threading.RLock()
        self._navigation_active = False
        self._fault = None
        self._button_down = False
        self._feedback = {}
        self._tracker = self._new_tracker(travel_palms)
        # HandControls checks the existing capability gate before native input.
        self.controls = HandControls(
            active=lambda: self._navigation_active and self._authorized(),
            feedback=lambda message: None, mode=mode, clock=clock,
            dispatch=dispatch, desktop_factory=desktop_factory,
        )

    @staticmethod
    def _new_tracker(travel_palms):
        if type(travel_palms) not in (int, float) or not math.isfinite(travel_palms) or travel_palms <= 0:
            raise ValueError("travel_palms must be a positive finite distance")
        # Pose actions stay disabled: a point/pinch must retain its mouse meaning.
        return MeasuredGestureTracker(travel_palms=travel_palms,
                                      clutch_pose=FOUR_FINGER, pose_actions=False)

    def _authorized(self):
        try:
            return self._active() is True
        except Exception:
            return False

    @property
    def bend_click(self):
        return self.mouse.bend_click

    @property
    def pinching(self):
        return self.mouse.pinching

    @property
    def navigation_active(self):
        return self._navigation_active

    def _navigation_feedback(self, *, finished=False):
        desktop = self.controls.desktop.status()
        result = {
            "state": "mouse_error" if self._fault else "mouse_paused" if finished else "desktop_swipe",
            "hint": self._fault or self._feedback.get("text", "Hold four fingers still to navigate desktops"),
            "progress": self._feedback.get("progress", 0.0),
            "eeg_compatible": True, "pointer_ready": False,
            # The final neutral frame still belongs to navigation. A later
            # pointer frame releases this ownership and starts fresh CNN rest.
            "finger_active": True, "finger_action_active": True,
            "button_down": self._button_down if self._fault else False,
            "navigation_active": self._navigation_active,
            "navigation": {"state": self._feedback.get("state", "idle"),
                           "axis": self._feedback.get("axis"),
                           "progress": self._feedback.get("progress", 0.0),
                           "desktop": desktop},
        }
        if finished and not self._fault:
            result["hint"] = "Point briefly to resume the pointer; four fingers to navigate desktops."
        if self._fault:
            result.update(error=self._fault, cleanup_pending=True)
        return result

    def _checked_mouse(self, result):
        if not isinstance(result, dict) or any(type(result.get(key)) is not bool
                                               for key in ("pointer_ready", "finger_active", "button_down")):
            raise RuntimeError("Hand controller returned invalid input ownership.")
        self._button_down = result["button_down"]
        # Forward genuine click IDs unchanged; navigation never creates IDs.
        return {**result, "navigation_active": False}

    def _cleanup(self, reason, *, operation="reset", preserve_fault=False):
        errors = []
        previous_fault = self._fault
        # Always attempt both releases, even if one device rejects cleanup.
        try:
            desktop = self.controls.stop()
            if not isinstance(desktop, dict) or desktop.get("error") or desktop.get("cleanup_pending"):
                errors.append(str(desktop.get("error") or "Desktop input release is pending.")
                              if isinstance(desktop, dict) else "Desktop input release was not confirmed.")
        except Exception as exc:
            errors.append(f"Could not release desktop input: {exc}")
        try:
            callback = getattr(self.mouse, operation)
            mouse = self._checked_mouse(callback() if operation == "close" else callback(reason))
            if mouse.get("error") or mouse.get("button_down"):
                errors.append(str(mouse.get("error") or "Hand Mouse button release is pending."))
        except Exception as exc:
            self._button_down = True
            mouse = {"state": "mouse_error", "pointer_ready": False, "finger_active": False,
                     "button_down": True, "eeg_compatible": True, "navigation_active": False}
            errors.append(f"Could not release Hand Mouse: {exc}")
        self._fault = "; ".join(errors) or (previous_fault if preserve_fault else None)
        if errors:
            # Keep the native ownership latch until an explicit reset confirms
            # release; no later frame may silently restart either controller.
            self._navigation_active = True
            result = self._navigation_feedback()
            result["button_down"] = mouse.get("button_down", True)
            return result
        self._tracker.reset(self._clock(), reason)
        self._navigation_active = False
        self._feedback = {}
        if self._fault:
            return {**mouse, "error": self._fault, "hint": self._fault,
                    "state": "mouse_error", "cleanup_pending": True}
        return mouse

    def reset(self, reason="Hand controls paused; point to resume"):
        with self._lock:
            return self._cleanup(reason)

    def close(self):
        with self._lock:
            return self._cleanup("Hand controls stopped", operation="close")

    def suspend_tracking(self, reason="Hand tracking paused; point briefly to resume"):
        with self._lock:
            # Tracking recovery cannot clear an earlier native input fault.
            return self._cleanup(reason, operation="suspend_tracking", preserve_fault=True)

    def set_bend_click(self, enabled):
        with self._lock:
            result = self._cleanup("Click preference changed; point to resume")
            if result.get("error"):
                return result
            return self._checked_mouse(self.mouse.set_bend_click(enabled))

    def configure_navigation(self, *, mode, travel_palms):
        with self._lock:
            tracker = self._new_tracker(travel_palms)
            if self._navigation_active or self._fault or self.pinching:
                return {"error": "Finish releasing hand input before changing desktop navigation."}
            result = self.controls.desktop.configure(mode=mode)
            if not isinstance(result, dict) or result.get("error") or result.get("cleanup_pending"):
                return result if isinstance(result, dict) else {"error": "Desktop configuration was not confirmed."}
            self._tracker = tracker
            return {"ok": True, "mode": mode, "travel_palms": travel_palms}

    @staticmethod
    def _valid_points(points):
        try:
            return bool(points and len(points) == 21 and all(
                len(point) >= 2 and all(type(value) in (int, float) and math.isfinite(value)
                                        for value in point)
                and all(-0.2 <= value <= 1.2 for value in point[:2]) for point in points))
        except (TypeError, ValueError):
            return False

    def _deliver(self, records, now):
        # Reject vertical intent before its begin reaches HandControls. Keep
        # the measured neutral latch, avoiding a sideways action on recovery.
        if any(record["kind"] == "motion" and record.get("axis") == "vertical"
               and record["phase"] != "cancel" for record in records):
            records = self._tracker._cancel_navigation(
                now, "horizontal_only", "Use a left/right swipe; lower your hand or make a fist to reset.")
        for record in records:
            if record["kind"] == "gesture_progress":
                self._feedback = record
                if record.get("state") == "armed":
                    self._feedback = {**record, "text": "Move left/right between desktops."}
            elif record["kind"] == "motion":
                try:
                    result = self.controls.motion(record)
                    if not isinstance(result, dict) or result.get("error") or result.get("cleanup_pending"):
                        raise RuntimeError(str(result.get("error") or "Desktop input release is pending.")
                                           if isinstance(result, dict) else "Desktop motion was not confirmed.")
                except Exception as exc:
                    self._fault = f"Desktop swipe stopped: {exc}"
                    self._cleanup(self._fault, preserve_fault=True)
                    return

    def update(self, points, now, *, hold=False):
        with self._lock:
            if self._fault:
                return self._navigation_feedback()
            if not self._authorized():
                return self._cleanup("Hand controls are disarmed; explicitly arm to resume.")
            if not self._valid_points(points):
                points = []
            if hold and self._navigation_active:
                self._deliver(self._tracker._cancel_navigation(
                    now, "selection_hold", "Swipe paused; lower your hand or make a fist to reset."), now)
                return self._navigation_feedback()
            # An opening hand may be the release frame of an existing pinch
            # or drag. Its original mouse controller gets that frame first.
            if (not self._navigation_active and not hold and is_navigation_pose(points)
                    and not self.pinching):
                try:
                    result = self._checked_mouse(self.mouse.reset("navigation_started"))
                except Exception as exc:
                    self._button_down = True  # Ambiguous release failure retains possible ownership.
                    self._fault = f"Could not release Hand Mouse before navigation: {exc}"
                    return self._navigation_feedback()
                if result.get("error") or result.get("button_down"):
                    self._fault = str(result.get("error") or "Hand Mouse button release is pending.")
                    return {**result, "error": self._fault, "cleanup_pending": True}
                self._navigation_active = True
            if not self._navigation_active:
                try:
                    result = self._checked_mouse(self.mouse.update(points, now, hold=hold))
                except Exception as exc:
                    self._button_down = True
                    self._fault = f"Hand Mouse stopped: {exc}"
                    return self._navigation_feedback()
                if result.get("error"):
                    self._fault = str(result["error"])
                    result["cleanup_pending"] = True
                return result
            if not points:
                records = self._tracker.update(NONE, 0, 0, 0, now)
            else:
                maintains = self._tracker._anchor is not None and _holds_navigation_pose(points)
                if self._tracker._anchor is not None and is_pinching(points):
                    pose = PINCH
                else:
                    pose = FOUR_FINGER if is_navigation_pose(points) or maintains else classify(points)
                wrist = points[WRIST]
                records = self._tracker.update(pose if pose != NONE else UNKNOWN,
                                               wrist[0], wrist[1], hand_scale(points), now)
            self._deliver(records, now)
            if self._fault:
                return self._navigation_feedback()
            if self._tracker._pose != FOUR_FINGER:
                self._tracker.reset(now, "navigation_ended")
                self._navigation_active = False
                return self._navigation_feedback(finished=True)
            return self._navigation_feedback()
