"""Continuous Windows touchpad input with a measured desktop-shortcut fallback.

The native path injects four contacts horizontally. Windows owns that animation
and uses the user's touchpad gesture settings. Vertical actions use explicit
Windows shortcuts. No private shell interfaces are used.
"""

from __future__ import annotations

import ctypes
import math
import sys
import threading
import time
from typing import Callable


_U32 = ctypes.c_uint32
_I32 = ctypes.c_int32
_UPTR = ctypes.c_size_t


class _Point(ctypes.Structure):
    _fields_ = [("x", _I32), ("y", _I32)]


class _Rect(ctypes.Structure):
    _fields_ = [(n, _I32) for n in ("left", "top", "right", "bottom")]


class _PointerInfo(ctypes.Structure):
    _fields_ = [
        ("pointerType", _U32), ("pointerId", _U32), ("frameId", _U32),
        ("pointerFlags", _U32), ("sourceDevice", ctypes.c_void_p),
        ("hwndTarget", ctypes.c_void_p), ("ptPixelLocation", _Point),
        ("ptHimetricLocation", _Point), ("ptPixelLocationRaw", _Point),
        ("ptHimetricLocationRaw", _Point), ("dwTime", _U32),
        ("historyCount", _U32), ("InputData", _I32), ("dwKeyStates", _U32),
        ("PerformanceCount", ctypes.c_uint64), ("ButtonChangeType", _U32),
    ]


class _TouchInfo(ctypes.Structure):
    _fields_ = [
        ("pointerInfo", _PointerInfo), ("touchFlags", _U32),
        ("touchMask", _U32), ("rcContact", _Rect), ("rcContactRaw", _Rect),
        ("orientation", _U32), ("pressure", _U32),
    ]


class _PointerUnion(ctypes.Union):
    # TOUCH_INFO is the largest member; the unused PEN_INFO fits within it.
    _fields_ = [("pointerInfo", _PointerInfo), ("touchInfo", _TouchInfo)]


class _PointerTypeInfo(ctypes.Structure):
    _anonymous_ = ("data",)
    _fields_ = [("type", _U32), ("data", _PointerUnion)]


class _DeviceParams(ctypes.Structure):
    _fields_ = [
        ("pointerType", _U32), ("maxCount", _U32), ("feedbackMode", _U32),
        ("hMonitor", ctypes.c_void_p), ("deviceWidth", _U32),
        ("deviceHeight", _U32), ("options", _U32),
    ]


class _KeyboardInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_uint16), ("wScan", ctypes.c_uint16),
                ("dwFlags", _U32), ("time", _U32), ("dwExtraInfo", _UPTR)]


class _MouseInput(ctypes.Structure):
    _fields_ = [("dx", _I32), ("dy", _I32), ("mouseData", _U32),
                ("dwFlags", _U32), ("time", _U32), ("dwExtraInfo", _UPTR)]


class _InputUnion(ctypes.Union):
    _fields_ = [("ki", _KeyboardInput), ("mi", _MouseInput)]


class _Input(ctypes.Structure):
    _anonymous_ = ("data",)
    _fields_ = [("type", _U32), ("data", _InputUnion)]


def _load_user32():
    if sys.platform != "win32":
        raise OSError("Desktop control is only available on Windows.")
    return ctypes.WinDLL("user32", use_last_error=True)


def _configure(function, argtypes, restype):
    # Plain Python callables also work as injectable operating-system fakes.
    try:
        function.argtypes = argtypes
        function.restype = restype
    except AttributeError:
        pass
    return function


def _last_error():
    return getattr(ctypes, "get_last_error", lambda: 0)()


def shortcut_switch(direction: str, *, user32=None) -> dict:
    """Request an adjacent desktop with Win+Ctrl+Left/Right once.

    The capability registry gates direct calls. This function cannot verify
    whether an adjacent desktop exists or whether Windows accepted the switch.
    """
    if direction not in ("left", "right"):
        return {"error": "Choose left or right."}
    try:
        api = user32 if user32 is not None else _load_user32()
        send = _configure(api.SendInput,
                          [_U32, ctypes.POINTER(_Input), ctypes.c_int], _U32)
        state = _configure(api.GetAsyncKeyState, [ctypes.c_int], ctypes.c_int16)
        # Never synthesize a chord while the user is holding modifiers: releasing
        # them would interfere with their input, and Alt/Shift changes the chord.
        if any(state(key) & 0x8000 for key in (0x10, 0x11, 0x12, 0x5B, 0x5C)):
            return {"error": "Release keyboard modifier keys before switching desktops."}
        keys = (0x5B, 0x11, 0x25 if direction == "left" else 0x27)
        inputs = (_Input * 6)()
        for index, key in enumerate(keys):
            inputs[index].type = 1
            inputs[index].ki = _KeyboardInput(key, 0, 0, 0, 0)
            inputs[5 - index].type = 1
            inputs[5 - index].ki = _KeyboardInput(key, 0, 0x0002, 0, 0)
        inserted = send(6, inputs, ctypes.sizeof(_Input))
        if inserted != 6:
            # A partial insertion must not leave Win or Ctrl pressed.
            releases = (_Input * 3)(*inputs[3:])
            send(3, releases, ctypes.sizeof(_Input))
            return {"error": "Windows did not accept the complete desktop shortcut."}
        return {"ok": True, "direction": direction, "status": "switch_requested"}
    except Exception as exc:
        return {"error": f"Could not switch desktop: {exc}"}


def shortcut_overview(direction: str, *, user32=None) -> dict:
    """Toggle Task View (up) or show/hide the desktop (down), once on release."""
    if direction not in ("up", "down"):
        return {"error": "Choose up or down."}
    try:
        api = user32 if user32 is not None else _load_user32()
        send = _configure(api.SendInput,
                          [_U32, ctypes.POINTER(_Input), ctypes.c_int], _U32)
        state = _configure(api.GetAsyncKeyState, [ctypes.c_int], ctypes.c_int16)
        if any(state(key) & 0x8000 for key in (0x10, 0x11, 0x12, 0x5B, 0x5C)):
            return {"error": "Release keyboard modifier keys before changing desktop view."}
        keys = (0x5B, 0x09 if direction == "up" else 0x44)
        inputs = (_Input * 4)()
        for index, key in enumerate(keys):
            inputs[index].type = 1
            inputs[index].ki = _KeyboardInput(key, 0, 0, 0, 0)
            inputs[3 - index].type = 1
            inputs[3 - index].ki = _KeyboardInput(key, 0, 0x0002, 0, 0)
        if send(4, inputs, ctypes.sizeof(_Input)) != 4:
            releases = (_Input * 2)(*inputs[2:])
            send(2, releases, ctypes.sizeof(_Input))
            return {"error": "Windows did not accept the complete desktop view shortcut."}
        return {"ok": True, "direction": direction,
                "status": "overview_requested" if direction == "up" else "desktop_requested"}
    except Exception as exc:
        return {"error": f"Could not change desktop view: {exc}"}


class _WindowsTouchpad:
    """Owns one synthetic device. Constructing it injects no contacts."""

    def __init__(self, user32=None, *, sleep: Callable = time.sleep):
        self._api = user32 if user32 is not None else _load_user32()
        self._sleep = sleep
        self._device = None
        self._destroying = False
        self._active = False
        self._progress = 0.0
        self._target = 0.0
        self._axis = "horizontal"
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._error = None
        self._closing = False
        self._last_update = 0.0
        create = _configure(self._api.CreateSyntheticPointerDevice2,
                            [ctypes.POINTER(_DeviceParams)], ctypes.c_void_p)
        self._inject = _configure(self._api.InjectSyntheticPointerInput,
                                  [ctypes.c_void_p, ctypes.POINTER(_PointerTypeInfo), _U32],
                                  ctypes.c_int)
        self._destroy = _configure(self._api.DestroySyntheticPointerDevice,
                                   [ctypes.c_void_p], None)
        # PT_TOUCHPAD, four contacts, no pointer feedback, 100 x 60 mm device,
        # physical units + gesture-only: cannot move/click the mouse.
        params = _DeviceParams(5, 4, 3, None, 10000, 6000, 0x1 | 0x2)
        self._device = create(ctypes.byref(params))
        if not self._device:
            raise OSError(f"Windows touchpad creation failed (error {_last_error()}).")

    def _frame(self, progress, *, release=False, cancelled=False):
        if not self._device:
            raise OSError("Synthetic touchpad is closed.")
        count = 4 if self._axis == "horizontal" else 3
        contacts = (_PointerTypeInfo * count)()
        # Keep every contact within the physical 100 x 60 mm device.
        displacement = round(progress * (3000 if self._axis == "horizontal" else 2400))
        for index, contact in enumerate(contacts):
            contact.type = 5
            pointer = contact.touchInfo.pointerInfo
            pointer.pointerType = 5
            pointer.pointerId = index + 1
            # The touchpad API derives contact transitions from these flags;
            # unlike touchscreen injection, no DOWN/UPDATE/UP flag is needed.
            pointer.pointerFlags = 0x4000 if release else 0x4000 | 0x0002 | 0x0004
            if cancelled:
                pointer.pointerFlags |= 0x8000
            pointer.ptHimetricLocation = _Point(
                4400 + index * 400 + (displacement if self._axis == "horizontal" else 0),
                2500 + (index % 2) * 500 + (displacement if self._axis == "vertical" else 0))
        for attempt in range(3):
            if self._inject(self._device, contacts, count):
                return
            error = _last_error()
            if error != 21 or attempt == 2:  # ERROR_NOT_READY: input frames too close.
                raise OSError(f"Windows touchpad injection failed (error {error}).")
            self._sleep(0.002)

    def begin(self, axis="horizontal"):
        if axis not in ("horizontal", "vertical"):
            raise ValueError("Choose horizontal or vertical swipe axis.")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise OSError("Previous touchpad animation is still stopping.")
            # Mark active first so close() releases after an ambiguous failure.
            self._active = True
            self._axis = axis
            self._progress = self._target = 0.0
            self._error = None
            self._closing = False
            self._last_update = time.monotonic()
            self._stop.clear()
            self._frame(0.0)
            self._thread = threading.Thread(target=self._animate, name="desktop-touchpad", daemon=True)
            self._thread.start()

    def update(self, progress):
        with self._lock:
            if self._error is not None:
                raise self._error
            if not self._active:
                raise OSError("Synthetic touchpad gesture is not active.")
            self._target = max(-1.0, min(1.0, progress))
            self._last_update = time.monotonic()

    def _advance(self, elapsed):
        """Resample camera targets at 60 Hz, without predicting past the target."""
        alpha = 1.0 - math.exp(-max(0.0, elapsed) / 0.035)
        self._progress += (self._target - self._progress) * alpha
        if abs(self._target - self._progress) < 0.0005:
            self._progress = self._target
        self._frame(self._progress)

    def _animate(self):
        previous = time.monotonic()
        interval = 1 / 60
        deadline = previous + interval
        try:
            while not self._stop.wait(max(0.0, deadline - time.monotonic())):
                with self._lock:
                    if not self._active:
                        break
                    now = time.monotonic()
                    # The recognizer allows a 300 ms frame gap and briefly
                    # suppresses updates during a deliberate fist transition.
                    if now - self._last_update > 0.6:
                        self._error = TimeoutError("Touchpad swipe stopped because hand updates became stale.")
                        self._finish_contacts(cancelled=True, settle=False)
                        break
                    self._advance(now - previous)
                    previous = now
                deadline += interval
                finished = time.monotonic()
                if deadline <= finished:
                    # Drop missed ticks instead of injecting catch-up bursts.
                    deadline = finished + interval
        except Exception as exc:
            with self._lock:
                self._error = exc
                try:
                    self._finish_contacts(cancelled=True, settle=False)
                except Exception:
                    pass
                finally:
                    # A failed injection is ambiguous: discard the device even
                    # if capture is stalled and cannot report the error yet.
                    self._destroy_device()
        finally:
            # If a join timed out during close(), this worker retains ownership
            # until its in-flight injection finishes, then releases and destroys.
            if self._closing:
                with self._lock:
                    try:
                        self._finish_contacts(cancelled=True, settle=False)
                    except Exception as exc:
                        self._error = self._error or exc
                    finally:
                        self._destroy_device()

    def _stop_animation(self):
        self._stop.set()
        worker = self._thread
        if worker is not None:
            worker.join(0.5)
            if worker.is_alive():
                raise TimeoutError("Touchpad animation is still releasing its input.")
            self._thread = None

    def _finish_contacts(self, *, cancelled, settle=True):
        if not self._active:
            return
        try:
            target = 0.0 if cancelled else self._target
            if not cancelled and self._axis == "horizontal" and abs(target) >= 1.0:
                # Follow through to 40 mm after an intentional completed swipe.
                # Live hand travel still maps to 30 mm; every contact remains
                # inside the 100 mm device even at this final physical position.
                target = math.copysign(4 / 3, target)
            start = self._progress
            failed = True
            try:
                # Six movement frames plus lift spacing settle in about 120 ms.
                steps = 6 if settle and abs(target - start) > 0.0005 else 1
                for step in range(1, steps + 1):
                    if steps > 1:
                        self._sleep(1 / 60)
                    fraction = step / steps
                    eased = fraction * fraction * (3 - 2 * fraction)
                    self._progress = start + (target - start) * eased
                    self._frame(self._progress)
                failed = False
            finally:
                # Match the native input sample: lift on a distinct frame,
                # rather than immediately after the final movement report.
                self._sleep(1 / 60)
                self._frame(self._progress, release=True, cancelled=cancelled or failed)
        finally:
            self._active = False

    def end(self, cancelled=False):
        self._stop_animation()
        with self._lock:
            error = self._error
            self._finish_contacts(cancelled=cancelled or error is not None)
            if error is not None:
                raise error

    def _destroy_device(self):
        device = self._device
        if device:
            self._destroying = True
            self._device = None
            try:
                self._destroy(device)
            except Exception:
                self._device = device
                raise
            finally:
                self._destroying = False

    @property
    def closed(self):
        # Deliberately nonblocking: an in-flight injection may own _lock.
        return self._device is None and not self._destroying

    @property
    def cleanup_pending(self):
        return self._closing and not self.closed

    def close(self):
        self._closing = True
        try:
            self.end(cancelled=True)
        finally:
            # Never destroy a handle under a still-running injection call.
            if self._thread is None or not self._thread.is_alive():
                with self._lock:
                    self._destroy_device()


class DesktopSwipeController:
    """Serializes an authorized hand swipe, including stop from another thread.

    ``progress`` is signed hand travel in [-1, 1]: negative is a leftward hand
    sweep or upward movement. Native horizontal input follows Windows' touchpad settings.
    The fallback uses natural trackpad direction: a leftward hand sweep requests
    the right desktop. Vertical input always uses explicit Windows shortcuts:
    up toggles Task View; down toggles the desktop.
    It commits once on release only if the final absolute progress reaches 1.

    ``authorize`` (horizontal) and ``authorize_overview`` (vertical) must return
    True or {"ok": True}; the matching callback is called once before each begin.
    A missing callback denies access. All OS work can be replaced by fakes.
    """

    def __init__(self, authorize: Callable | None = None, *, mode: str = "auto",
                 native_factory: Callable = _WindowsTouchpad,
                 shortcut: Callable = shortcut_switch,
                 authorize_overview: Callable | None = None,
                 overview_shortcut: Callable = shortcut_overview):
        if mode not in ("auto", "shortcut"):
            raise ValueError("Choose auto or shortcut desktop mode.")
        self._authorize = authorize
        self._authorize_overview = authorize_overview
        self._native_factory = native_factory
        self._shortcut = shortcut
        self._overview_shortcut = overview_shortcut
        self._lock = threading.RLock()
        self._native = None
        self._native_disabled = False
        self._native_available = False
        self._cleanup_pending = False
        self._preference = mode
        self._reason = "Native touchpad support has not been checked."
        self._mode = "shortcut"
        self._active = False
        self._progress = 0.0
        self._axis = "horizontal"

    def status(self) -> dict:
        with self._lock:
            self._refresh_cleanup()
            return {"mode": self._mode, "active": self._active,
                    "progress": self._progress, "axis": self._axis,
                    "commit_threshold": 1.0, "text": self._reason,
                    "native_available": self._native_available,
                    "fallback": self._native_disabled,
                    "cleanup_pending": self._cleanup_pending,
                    "preference": self._preference}

    describe = status

    def configure(self, *, mode: str) -> dict:
        with self._lock:
            self._refresh_cleanup()
            if mode not in ("auto", "shortcut"):
                return {"error": "Choose auto or shortcut desktop mode."}
            if self._cleanup_pending:
                return {"error": "Finish releasing native input before changing desktop mode.", **self.status()}
            if self._active:
                return {"error": "Finish the current swipe before changing desktop mode."}
            if not self._discard_native():
                return {"error": self._reason, **self.status()}
            self._preference = mode
            self._mode = "shortcut"
            self._reason = ("Measured desktop shortcuts selected."
                            if mode == "shortcut" else "Native touchpad will be checked on the next swipe.")
            return {"ok": True, **self.status()}

    def probe(self) -> dict:
        """Create and destroy a device without sending contacts or keystrokes."""
        with self._lock:
            self._refresh_cleanup()
            if (self._active or self._native is not None or self._native_disabled
                    or self._cleanup_pending
                    or self._preference == "shortcut"):
                return self.status()
            device = None
            try:
                device = self._native_factory()
                device.close()
                device = None
                self._native_available = True
                self._mode = "native"
                self._reason = "Native horizontal touchpad input available; Windows controls animation and four-finger settings. Vertical actions use Windows shortcuts."
                return self.status()
            except Exception as exc:
                self._native_disabled = True
                self._native_available = False
                self._mode = "shortcut"
                self._reason = f"Measured shortcut fallback: {exc}"
                return self.status()
            finally:
                if device is not None:
                    try:
                        device.close()
                    except Exception:
                        pass

    def _refresh_cleanup(self):
        if (self._cleanup_pending and self._native is not None
                and getattr(self._native, "closed", False)):
            self._native = None
            self._cleanup_pending = False

    def _discard_native(self):
        device = self._native
        if device is None:
            self._cleanup_pending = False
            return True
        try:
            device.close()
        except Exception as exc:
            if not getattr(device, "closed", False):
                self._cleanup_pending = True
                self._reason = f"Native input cleanup is still pending: {exc}"
                return False
        if getattr(device, "cleanup_pending", False):
            self._cleanup_pending = True
            self._reason = "Native input cleanup is still pending. Lower your hand to retry."
            return False
        self._native = None
        self._cleanup_pending = False
        return True

    def _native_failure(self, exc) -> dict:
        self._active = False
        self._native_disabled = True
        self._native_available = False
        self._reason = f"Native swipe stopped: {exc}. Next swipe will use the measured shortcut fallback."
        self._discard_native()
        return {"error": self._reason, **self.status()}

    def begin(self, axis: str = "horizontal") -> dict:
        with self._lock:
            self._refresh_cleanup()
            if self._cleanup_pending:
                return {"error": "Native input is still releasing. Lower your hand to retry.", **self.status()}
            if axis not in ("horizontal", "vertical"):
                return {"error": "Choose horizontal or vertical swipe axis.", **self.status()}
            if self._active:
                if axis != self._axis:
                    return {"error": "Finish the current swipe before changing axis.", **self.status()}
                return {"ok": True, **self.status()}
            try:
                authorize = self._authorize_overview if axis == "vertical" else self._authorize
                permission = authorize() if authorize else False
            except Exception as exc:
                return {"error": f"Desktop swipe authorization failed: {exc}", **self.status()}
            allowed = permission is True or (isinstance(permission, dict)
                                             and permission.get("ok") is True
                                             and not permission.get("error"))
            if not allowed:
                reason = permission.get("error") if isinstance(permission, dict) else None
                return {"error": reason or "Desktop swipe was not authorized.", **self.status()}
            self._progress = 0.0
            self._axis = axis
            if (self._native is None and not self._native_disabled
                    and self._preference != "shortcut" and axis == "horizontal"):
                try:
                    self._native = self._native_factory()
                    self._native_available = True
                except Exception as exc:
                    self._native_disabled = True
                    self._native_available = False
                    self._reason = f"Measured shortcut fallback: {exc}"
            self._mode = "native" if self._native is not None and axis == "horizontal" else "shortcut"
            if self._mode == "native":
                try:
                    self._native.begin(axis=axis)
                except Exception as exc:
                    return self._native_failure(exc)
                self._reason = "Native horizontal touchpad input; Windows controls animation and four-finger settings."
            elif axis == "vertical":
                self._reason = "Task View and show desktop use Windows shortcuts."
            self._active = True
            return {"ok": True, **self.status()}

    def update(self, progress: float) -> dict:
        with self._lock:
            if not self._active:
                return {"error": "No desktop swipe is active.", **self.status()}
            try:
                value = float(progress)
                if not math.isfinite(value):
                    raise ValueError("not finite")
            except (ValueError, TypeError, OverflowError):
                self.end(cancelled=True)
                return {"error": "Swipe progress must be a finite number.", **self.status()}
            self._progress = max(-1.0, min(1.0, value))
            if self._mode == "native" and self._native is not None:
                try:
                    self._native.update(self._progress)
                except Exception as exc:
                    return self._native_failure(exc)
            return {"ok": True, **self.status()}

    def end(self, cancelled: bool = False) -> dict:
        with self._lock:
            self._refresh_cleanup()
            if self._cleanup_pending:
                if not self._discard_native():
                    return {"error": self._reason, **self.status()}
                return {"ok": True, "completed": False, **self.status()}
            if not self._active:
                return {"ok": True, "completed": False, **self.status()}
            self._active = False
            cancelled = cancelled or abs(self._progress) < 1.0
            if self._mode == "native" and self._native is not None:
                try:
                    self._native.end(cancelled=cancelled)
                except Exception as exc:
                    return self._native_failure(exc)
                return {"ok": True, "completed": not cancelled,
                        "cancelled": cancelled, "status": "gesture_released", **self.status()}
            if not cancelled:
                vertical = self._axis == "vertical"
                direction = ("up" if self._progress < 0 else "down") if vertical else (
                    "right" if self._progress < 0 else "left")
                try:
                    result = (self._overview_shortcut if vertical else self._shortcut)(direction)
                except Exception as exc:
                    result = {"error": f"Desktop shortcut failed: {exc}"}
                return {**result, "completed": bool(result.get("ok")), **self.status()}
            return {"ok": True, "completed": False, **self.status()}

    def close(self):
        """Release input on camera stop; a later begin lazily opens a new device."""
        with self._lock:
            self.end(cancelled=True)
            if self._cleanup_pending:
                return {"error": self._reason, **self.status()}
            if not self._discard_native():
                return {"error": self._reason, **self.status()}
            self._native_disabled = False
            self._reason = ("Measured desktop shortcuts selected."
                            if self._preference == "shortcut"
                            else "Native touchpad will be checked on the next swipe.")
            return {"ok": True, "completed": False, **self.status()}
