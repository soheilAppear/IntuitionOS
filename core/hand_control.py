"""Coordinate measured hand input and a separate, target-bound close approval.

The camera supplies evidence; this coordinator checks the capability gate. A
native desktop drag is authorized once before any contacts are injected. Slow
model requests never share this path or delay an in-progress hand movement.
"""

import threading
import time

from .actions import actions
from .capabilities import capabilities, gate
from .desktop import DesktopSwipeController


class HandControls:
    CLOSE_SECONDS = 6.0

    def __init__(self, active, feedback, *, mode="auto", clock=time.monotonic,
                 dispatch=None, confirm=None, desktop_factory=DesktopSwipeController):
        self.active = active
        self.feedback = feedback
        self.clock = clock
        self.dispatch = dispatch or actions.dispatch
        self.confirm = confirm or actions.confirm
        self._lock = threading.RLock()
        self._pending = None
        self._timer = None
        self._invalid_before = float("-inf")
        self._last_event_at = float("-inf")
        self._motion_axis = None
        self.desktop = desktop_factory(
            authorize=self._authorize_desktop, mode=mode,
            authorize_overview=self._authorize_overview,
            shortcut=lambda direction: self.dispatch(
                "os_switch_desktop", {"direction": direction}, actor="gesture"),
            overview_shortcut=lambda direction: self.dispatch(
                "os_desktop_view", {"view": "overview" if direction == "up" else "desktop"},
                actor="gesture"),
        )

    def _authorize_desktop(self):
        return self._authorize_navigation("os_switch_desktop", {"direction": "right"})

    def _authorize_overview(self):
        return self._authorize_navigation("os_desktop_view", {"view": "overview"})

    def _authorize_navigation(self, capability, arguments):
        cap = capabilities.get(capability)
        if not self.active() or cap is None:
            return {"error": "Hand tracking is not ready."}
        decision = gate(cap, arguments, actor="gesture", confidence=1.0)
        return {"ok": True} if decision.verdict == "allow" else {"error": decision.reason}

    def status(self):
        with self._lock:
            pending = self._pending
            return {"desktop": self.desktop.status(), "close": {
                "pending": bool(pending),
                "title": pending["title"] if pending else "",
                "remaining_s": max(0, pending["expires"] - self.clock()) if pending else 0,
            }}

    def _send(self, message):
        try:
            self.feedback(message)
        except Exception:
            pass

    def _cancel_close(self, text="Close cancelled."):
        pending, self._pending = self._pending, None
        if self._timer:
            self._timer.cancel()
            self._timer = None
        if pending:
            self.confirm(pending["token"], False, allow_safe_mode_change=False)
            self._send({"type": "gesture_close", "pending": False, "text": text})

    def _expire(self, token):
        with self._lock:
            if self._pending and self._pending["token"] == token:
                self._cancel_close("Close request expired. Pinch again to choose a window.")

    def handle(self, event, binding=None):
        """Run on a worker; stale camera events never act on a later foreground."""
        with self._lock:
            if (not self.active() or not 0 <= self.clock() - event.at <= 1.0
                    or event.at <= max(self._invalid_before, self._last_event_at)):
                return None
            self._last_event_at = event.at
            name = event.name
            if name == "close_cancel":
                self._cancel_close()
                return None
            if name == "close_confirm":
                pending = self._pending
                if not pending:
                    return None
                if self.clock() >= pending["expires"]:
                    self._cancel_close("Close request expired. Pinch again to choose a window.")
                    return None
                self._pending = None
                if self._timer:
                    self._timer.cancel()
                    self._timer = None
                result = self.confirm(pending["token"], True, allow_safe_mode_change=False)
                self._send({"type": "gesture_close", "pending": False,
                            "text": result.get("error") or f'Close requested for {pending["title"]}.',
                            "ok": not bool(result.get("error"))})
                return ("os_close_window", {}, result)
            if name == "close_request":
                self._cancel_close()
                target = self.dispatch("os_window_close_target", {}, actor="gesture")
                if target.get("error"):
                    return ("os_close_window", {}, target)
                args = {"hwnd": target["hwnd"], "pid": target["pid"]}
                result = self.dispatch("os_close_window", args, actor="gesture")
                if not result.get("needs_confirmation"):
                    return ("os_close_window", args, result)
                self._pending = {"token": result["token"], "title": target["title"],
                                 "expires": self.clock() + self.CLOSE_SECONDS}
                self._timer = threading.Timer(self.CLOSE_SECONDS, self._expire, (result["token"],))
                self._timer.daemon = True
                self._timer.start()
                self._send({"type": "gesture_close", "pending": True,
                            "title": target["title"], "remaining_s": self.CLOSE_SECONDS,
                            "text": "Hold thumbs-up to close. Make a fist or lower your hand to cancel."})
                return None
            if not binding:
                return None
            self._cancel_close()
            capability, args = binding
            return (capability, args, self.dispatch(capability, dict(args), actor="gesture"))

    def motion(self, event):
        """Synchronous camera-thread delivery keeps native input at capture speed."""
        with self._lock:
            phase = event.get("phase")
            if phase == "cancel":
                self._motion_axis = None
                self._invalid_before = max(self._invalid_before, event.get("at", self.clock()))
                self._cancel_close()
                result = self.desktop.end(cancelled=True)
            elif not self.active():
                self._motion_axis = None
                result = self.desktop.end(cancelled=True)
            elif phase == "begin":
                self._invalid_before = max(self._invalid_before, event.get("at", self.clock()))
                self._cancel_close()
                axis = event.get("axis", "horizontal")
                if axis not in ("horizontal", "vertical"):
                    self._motion_axis = None
                    self.desktop.end(cancelled=True)
                    return {"error": "Choose horizontal or vertical navigation."}
                self._motion_axis = None
                result = self.desktop.begin(axis=axis)
                if result.get("ok"):
                    self._motion_axis = axis
                    result = self.desktop.update(event.get("progress", 0.0))
            elif phase == "update":
                if event.get("axis", self._motion_axis) != self._motion_axis:
                    self._motion_axis = None
                    self.desktop.end(cancelled=True)
                    result = {"error": "Navigation direction changed; begin another gesture."}
                else:
                    result = self.desktop.update(event["progress"])
            elif phase == "end":
                changed_axis = event.get("axis", self._motion_axis) != self._motion_axis
                self._motion_axis = None
                result = self.desktop.end(cancelled=changed_axis)
            else:
                return None
            if result.get("error"):
                self._motion_axis = None
                # A failed begin can leave an earlier gesture alive; failed
                # updates can also leave a native release pending. Preserve
                # the original error while asking the device to clean up.
                if phase != "cancel":
                    self.desktop.end(cancelled=True)
            if phase != "update" or result.get("error"):
                self._send({"type": "gesture_desktop", **result})
            return result

    def stop(self):
        with self._lock:
            self._motion_axis = None
            self._invalid_before = self.clock()
            self._cancel_close("Hand tracking stopped. Close cancelled.")
            result = self.desktop.close()
            if isinstance(result, dict) and result.get("error"):
                self._send({"type": "gesture_desktop", **result})
            return result
