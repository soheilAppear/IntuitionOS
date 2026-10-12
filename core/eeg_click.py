"""One explicitly authorized EEG left click, with no pointer movement.

The caller owns EEG validation, fresh rest/select predictions, explicit arming,
and the Escape guard. This actuator cannot arm itself, choose a target, hold a
drag, or invoke semantic OS capabilities. Tests inject a fake mouse factory.
"""

import threading  # Serialize button ownership against concurrent stop requests.

from .hand_mouse import _WindowsMouse  # Reuse the existing, input-free Win32 adapter.


class EegClickController:
    """A lazy, reusable single-click actuator; errors require explicit reset."""

    def __init__(self, *, device_factory=_WindowsMouse):
        self._device_factory = device_factory  # Creating the controller never opens a mouse adapter.
        self._device = None  # Defer native initialization until the first authorized click.
        self._lock = threading.RLock()  # Keep click, reset, and close from racing button ownership.
        self._owned_left = False  # Never release a button unless this controller attempted its press.
        self._fault = None  # A failed injection blocks further clicks until cleanup is confirmed.
        self._click_id = 0  # Count only complete down/up pairs, preserving IDs across resets.

    def _failure_locked(self, reason):
        """Expose failure and unresolved ownership without retrying any input."""
        return {"ok": False, "error": reason,
                "cleanup_pending": self._owned_left}  # The caller must retain failed cleanup.

    def _release_locked(self):
        """Clear ownership only after Windows accepts our button release."""
        if self._owned_left:  # An untouched physical button does not belong to this actuator.
            self._device.left_up()  # Release remains necessary even if pressing reported an error.
            self._owned_left = False  # Keep ownership set when release raises, allowing cleanup retry.

    def click(self):
        """Click once at the current pointer after the caller's final EEG checks."""
        with self._lock:  # A concurrent stop waits for this bounded down/up operation to finish.
            if self._fault is not None:  # Never replay a failed or partially completed click.
                return self._failure_locked(self._fault)  # Explicit reset must resolve the fault first.
            try:  # Device initialization, physical-input checks, and injection can all fail.
                if self._device is None:  # Native construction is lazy and does not inject input.
                    self._device = self._device_factory()  # Tests provide an entirely fake device.
                blocked = self._device.blocked()  # Reject held modifiers and all physical mouse buttons.
                if blocked:  # Do not release or interfere with an existing physical press.
                    return self._failure_locked(
                        str(blocked).replace("Hand Mouse", "EEG clicking"))  # Keep the reason relevant to EEG.
                self._owned_left = True  # A failed down call may still have pressed the native button.
                try:  # Pair every attempted down with a release attempt, including ambiguous failures.
                    self._device.left_down()  # No coordinates or movement flags are ever supplied.
                finally:  # Escape, exceptions, and successful clicks all require an unheld button.
                    self._release_locked()  # A failed up preserves ownership for reset/close.
            except Exception as exc:  # Report failures to the service so it can immediately disarm.
                self._fault = "EEG click failed: " + str(exc)  # Latch the error instead of retrying a click.
                return self._failure_locked(self._fault)  # Report whether release still needs cleanup.
            self._click_id += 1  # Feedback identifies successful full clicks, never failed attempts.
            return {"ok": True, "click_id": self._click_id,
                    "cleanup_pending": False}  # A completed click owns no held input.

    def reset(self):
        """Retry owned-button cleanup and permit a later explicitly armed click."""
        with self._lock:  # Finish an in-flight click before deciding whether cleanup remains.
            try:  # A previous up failure must remain visible until a new up succeeds.
                self._release_locked()  # Never initialize a device or press a button during reset.
            except Exception as exc:  # Keep both ownership and the fault when release is unconfirmed.
                self._fault = "Could not release EEG mouse button: " + str(exc)  # Explain the cleanup failure.
                return self._failure_locked(self._fault)  # A subsequent close/reset can retry release.
            self._fault = None  # Clear the failure latch only after all owned input is released.
            return {"ok": True, "cleanup_pending": False}  # Reset itself never emits a click.

    def close(self):
        """Release owned input without pressing buttons; a later session may reuse us."""
        return self.reset()  # Use the same checked cleanup path for explicit stop and shutdown.
