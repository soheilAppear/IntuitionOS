"""Explicit Hand Mouse: point to engage, relaxed movement, pinch to click/drag.

This module owns no camera and emits no semantic actions. Its device is created
only after a deliberate point pose has armed. All input is injectable for tests.
Win32 references: SendInput, MOUSEINPUT, GetAsyncKeyState and
SetThreadDpiAwarenessContext in the Microsoft Windows SDK documentation.
"""

from __future__ import annotations

import ctypes
import math
import sys
import threading
from contextlib import contextmanager


_U32, _I32 = ctypes.c_uint32, ctypes.c_int32


class _Point(ctypes.Structure):
    _fields_ = [("x", _I32), ("y", _I32)]


class _MouseInput(ctypes.Structure):
    _fields_ = [("dx", _I32), ("dy", _I32), ("mouseData", _U32),
                ("dwFlags", _U32), ("time", _U32), ("dwExtraInfo", ctypes.c_size_t)]


class _InputUnion(ctypes.Union):
    # MOUSEINPUT is the largest INPUT union member on both Windows architectures.
    _fields_ = [("mi", _MouseInput)]


class _Input(ctypes.Structure):
    _anonymous_ = ("data",)
    _fields_ = [("type", _U32), ("data", _InputUnion)]


def _configure(function, arguments, result):
    try:
        function.argtypes, function.restype = arguments, result
    except AttributeError:  # Python fakes need no ctypes prototypes.
        pass
    return function


def _clamp(value):
    return max(0.0, min(1.0, value))


class _WindowsMouse:
    """A small, lazy Win32 device; construction never injects input."""

    def __init__(self, user32=None):
        if user32 is None:
            if sys.platform != "win32":
                raise OSError("Hand Mouse is available on Windows only.")
            user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._api = user32
        self._send = _configure(user32.SendInput,
                                [_U32, ctypes.POINTER(_Input), ctypes.c_int], _U32)
        self._key = _configure(user32.GetAsyncKeyState, [ctypes.c_int], ctypes.c_int16)
        self._metric = _configure(user32.GetSystemMetrics, [ctypes.c_int], ctypes.c_int)
        self._cursor = _configure(user32.GetCursorPos, [ctypes.POINTER(_Point)], ctypes.c_int)
        self._dpi = _configure(user32.SetThreadDpiAwarenessContext,
                               [ctypes.c_void_p], ctypes.c_void_p)

    @contextmanager
    def _physical_coordinates(self):
        previous = self._dpi(ctypes.c_void_p(-4))  # PER_MONITOR_AWARE_V2.
        if not previous:
            raise OSError("Windows could not provide physical screen coordinates.")
        try:
            yield
        finally:
            self._dpi(previous)

    def _geometry(self):
        # Windows anchors its primary display at (0, 0). Other displays can
        # extend left/up into negative virtual-desktop coordinates.
        width, height = self._metric(0), self._metric(1)
        vx, vy, vw, vh = (self._metric(key) for key in (76, 77, 78, 79))
        if min(width, height, vw, vh) <= 1:
            raise OSError("Windows returned invalid display dimensions.")
        return width, height, vx, vy, vw, vh

    def position(self):
        with self._physical_coordinates():
            width, height, *_ = self._geometry()
            point = _Point()
            if not self._cursor(ctypes.byref(point)):
                raise OSError("Windows could not read the pointer position.")
            return (_clamp(point.x / (width - 1)), _clamp(point.y / (height - 1)))

    def blocked(self, *, own_left=False):
        if any(self._key(key) & 0x8000 for key in (0x10, 0x11, 0x12, 0x5B, 0x5C)):
            return "Release keyboard modifier keys to use Hand Mouse"
        buttons = (0x02, 0x04, 0x05, 0x06) if own_left else (0x01, 0x02, 0x04, 0x05, 0x06)
        if any(self._key(key) & 0x8000 for key in buttons):
            return "Release physical mouse buttons to use Hand Mouse"
        return None

    def _emit(self, flags, x=0, y=0):
        event = _Input()
        event.type = 0  # INPUT_MOUSE.
        event.mi = _MouseInput(x, y, 0, flags, 0, 0)
        if self._send(1, ctypes.pointer(event), ctypes.sizeof(_Input)) != 1:
            error = getattr(ctypes, "get_last_error", lambda: 0)()
            raise OSError(f"Windows rejected Hand Mouse input (error {error}).")

    def move(self, x, y):
        with self._physical_coordinates():
            width, height, vx, vy, vw, vh = self._geometry()
            px, py = _clamp(x) * (width - 1), _clamp(y) * (height - 1)
            nx = round(_clamp((px - vx) / (vw - 1)) * 65535)
            ny = round(_clamp((py - vy) / (vh - 1)) * 65535)
            self._emit(0x0001 | 0x8000 | 0x4000, nx, ny)  # MOVE|ABSOLUTE|VIRTUALDESK.

    def left_down(self):
        self._emit(0x0002)

    def left_up(self):
        self._emit(0x0004)


class HandMouseController:
    """Track mirrored landmarks and control an explicitly enabled hand pointer.

    Device fakes implement position(), blocked(own_left=...), move(x, y),
    left_down() and left_up(). Coordinates passed to move are normalized within
    the primary monitor. reset/close release only a button this controller owns.

    Only a completed click returns click_id and click_source (bend or pinch).
    IDs increase across resets, and drag release emits no click event.

    pointer_only=True retains hand aiming but bypasses every hand click/drag
    path. Its feedback includes pointer_ready for a separate EEG click gate.
    update(..., hold=True) validates an engaged hand but freezes its target;
    the first frame after releasing the hold reanchors without moving.

    eeg_compatible=True instead retains normal finger clicks and drags while
    reporting their activity to an EEG controller. Finger gestures take priority
    over hold=True, so a pending EEG selection cannot disable a manual click.
    """

    CONTROL_REGION = (0.15, 0.12, 0.85, 0.82)
    ARM_SECONDS = 0.20
    CLICK_MIN_SECONDS = 0.06
    DRAG_SECONDS = 0.35
    BEND_SECONDS = 0.12
    DISTAL_BEND_ALIGNMENT = 0.25
    DISTAL_STRAIGHT_ALIGNMENT = 0.75
    REARM_SECONDS = 0.15
    RECOVERY_SECONDS = 0.15
    MAX_GAP_SECONDS = 0.35
    REST_SMOOTH_SECONDS = 0.13
    FAST_SMOOTH_SECONDS = 0.018
    VELOCITY_SMOOTH_SECONDS = 0.10
    VELOCITY_GAIN = 14.0

    def __init__(self, on_feedback=None, *, device_factory=_WindowsMouse, bend_click=False,
                 pointer_only=False, eeg_compatible=False):
        if type(bend_click) is not bool:
            raise ValueError("bend_click must be a boolean")
        # Require an explicit boolean so strings such as "false" cannot enable a mode.
        if type(pointer_only) is not bool:
            raise ValueError("pointer_only must be a boolean")
        # Preserve a separate explicit mode for hand clicks combined with EEG selection.
        if type(eeg_compatible) is not bool:
            raise ValueError("eeg_compatible must be a boolean")
        # A controller cannot simultaneously prohibit and enable finger button input.
        if pointer_only and eeg_compatible:
            raise ValueError("pointer_only and eeg_compatible are mutually exclusive")
        # Keep the mode fixed for this controller's lifetime; switching uses a reset/new instance.
        self._pointer_only = pointer_only
        # The combined mode is also immutable so authorization assumptions cannot change mid-gesture.
        self._eeg_compatible = eeg_compatible
        self.bend_click = bend_click
        self._factory = device_factory
        self._device = None
        self._callback = on_feedback
        self._lock = threading.RLock()
        self._owned_left = False
        self._fault = None
        self._click_id = 0
        self._clear()

    def _clear(self):
        # Ordinary startup uses its usual mapping; a later suspension explicitly requests reanchoring.
        self._resume_at_physical_cursor = False
        # A previously held bend cannot remain active after the hand controller is reset.
        self._finger_pose_active = False
        # A reset cannot report a manual gesture release from a previous input frame.
        self._finger_release_frame = False
        # Reset any EEG target hold so a new activation cannot resume an old target offset.
        self._pointer_held = False
        self._point_since = None
        self._point_origin = None
        self._armed = False
        self._pinch_since = None
        self._pinch_approaching = False
        self._bend_since = None
        self._bend_latched = False
        self._rearm_since = None
        self._drag_origin = None
        self._drag_cursor = None
        self._cursor = None
        self._virtual_offset = None
        self._filter_target = None
        self._filter_velocity = (0.0, 0.0)
        self._last_at = None
        self._last_input_at = None
        self._last_wrist = None
        self._last_scale = None
        self._last_anchor = None
        self._uncertain_since = None

    def _feedback(self, state, hint, progress=0.0, **extra):
        result = {"state": state, "progress": _clamp(progress), "hint": hint,
                  "control_region": list(self.CONTROL_REGION), **extra}
        # Expose readiness only for the dedicated aiming controller, preserving legacy feedback.
        if self.pointer_only:
            # Missing tracking, arming, pauses, and errors cannot authorize an EEG click.
            result["pointer_ready"] = state == "mouse_pointer" and self._armed
            # Let callers distinguish a movement-only controller from ordinary Hand Mouse.
            result["pointer_only"] = True
        # Combined control reports enough manual activity for the service to cancel competing EEG input.
        if self.eeg_compatible:
            # A near-pinch has not started a click gesture; its stable proximal
            # anchor can still aim while the fingertips move toward contact.
            # CNN clicks do not require a pointing pose, so expose actual action
            # ownership separately from the legacy target-engagement signal.
            result["finger_action_active"] = bool(self._pinch_since is not None
                                                  or self._bend_since is not None or self._owned_left
                                                  or self._finger_pose_active or self._finger_release_frame
                                                  or state == "mouse_clicked")
            result["finger_active"] = result["finger_action_active"] or self._pinch_approaching
            # Button ownership stays visible even if a failed release leaves the controller errored.
            result["button_down"] = self._owned_left
            # EEG may complete only on a tracked, engaged target without a competing finger gesture.
            result["pointer_ready"] = bool(state == "mouse_pointer" and self._armed
                                           and not result["finger_active"])
            # Let the service distinguish combined control from ordinary uncoordinated Hand Mouse.
            result["eeg_compatible"] = True
        if self._callback:
            try:
                self._callback(dict(result))
            except Exception:
                pass  # A display callback cannot interrupt button cleanup.
        return result

    def _release(self):
        if self._owned_left:
            # Leave ownership set if the OS rejects release, so reset/update
            # can retry rather than silently forgetting a possibly held button.
            self._device.left_up()
            self._owned_left = False

    def _cancel(self, reason, *, error=None):
        # Combined control resumes from the actual desktop cursor after any established hand suspension.
        resume_at_cursor = self.eeg_compatible and self._device is not None
        try:
            self._release()
        except Exception as exc:
            error = f"Could not release Hand Mouse button: {exc}"
        self._clear()
        # Preserve this recovery request across clearing all old gesture and filter state.
        self._resume_at_physical_cursor = resume_at_cursor
        self._fault = error
        if error:
            return self._feedback("mouse_error", error, error=error)
        return self._feedback("mouse_paused", reason)

    def reset(self, reason="Hand Mouse paused; point to resume"):
        with self._lock:
            return self._cancel(reason)

    def close(self):
        return self.reset("Hand Mouse stopped")

    def suspend_tracking(self, reason="Hand tracking paused; point briefly to resume"):
        """Release a drag and forget gestures while allowing fresh pointing to recover."""
        with self._lock:
            # Tracking loss must not silently clear an earlier native input/release fault.
            return self._cancel(reason, error=self._fault)

    @property
    def pointer_only(self):
        """A read-only mode: hand gestures can move but cannot press buttons."""
        return self._pointer_only

    @property
    def eeg_compatible(self):
        """A read-only mode coordinating normal finger input with a separate EEG controller."""
        return self._eeg_compatible

    @property
    def pinching(self):
        """Keep an active pinch/drag in mouse control through its release frame."""
        with self._lock:
            return self._pinch_since is not None

    def set_bend_click(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("bend_click must be a boolean")
        with self._lock:
            result = self._cancel("Click preference changed; point to resume")
            if not result.get("error"):
                self.bend_click = enabled
            return result

    def _freeze(self, now, reason):
        """Retain movement engagement briefly, never a click or held button."""
        self._release()
        self._pinch_since = self._bend_since = self._rearm_since = None
        self._pinch_approaching = False
        self._drag_origin = self._drag_cursor = None
        self._bend_latched = True
        if not self._armed or self._cursor is None:
            return self._cancel(reason)
        if self._uncertain_since is None:
            # Include the gap before the first missing sample: low frame rate
            # must not silently extend the advertised recovery allowance.
            self._uncertain_since = self._last_at if self._last_at is not None else now
        if now - self._uncertain_since >= self.RECOVERY_SECONDS:
            return self._cancel("Tracking lost; point briefly to resume")
        return self._feedback("mouse_recovering", reason + "; pointer held still")

    @staticmethod
    def _landmarks(points):
        if points is None or len(points) < 21:
            return None
        try:
            result = [(float(point[0]), float(point[1])) for point in points[:21]]
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
        if any(not math.isfinite(value) or not -0.2 <= value <= 1.2
               for point in result for value in point):
            return None
        return result

    @staticmethod
    def _extended(points, tip, pip):
        return math.dist(points[tip], points[0]) > math.dist(points[pip], points[0]) * 1.08

    @staticmethod
    def _distal_alignment(points, scale):
        # PIP -> DIP -> tip describes a fingertip bend even while the tip
        # remains beyond PIP. Both MediaPipe and WiLoR provide relative depth;
        # retain it here so a bend toward the camera is not mistaken for a
        # straight finger. Two-coordinate trackers remain supported.
        # Image x/y/depth have different normalization scales, so this is a
        # conservative alignment score, not an exact physical joint angle.
        try:
            joints = [points[index] for index in (6, 7, 8)]
            dimensions = {min(len(joint), 3) for joint in joints}
            if len(dimensions) != 1:
                return None
            axes = range(dimensions.pop())
            joints = [tuple(float(joint[axis]) for axis in axes) for joint in joints]
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
        if any(not math.isfinite(value) for joint in joints for value in joint):
            return None
        middle = tuple(b - a for a, b in zip(joints[0], joints[1]))
        distal = tuple(b - a for a, b in zip(joints[1], joints[2]))
        lengths = math.hypot(*middle), math.hypot(*distal)
        # Collapsed/implausible joints have no reliable direction and must not
        # turn occlusion or depth noise into a click or a neutral rearm.
        if any(not scale * 0.08 <= length <= scale * 2.0 for length in lengths):
            return None
        return sum(a * b for a, b in zip(middle, distal)) / math.prod(lengths)

    @staticmethod
    def _anchor(points):
        # Distal finger joints curl to click. The knuckle and proximal joint
        # provide a much steadier motion signal while those joints move.
        return tuple(points[5][axis] * 0.65 + points[6][axis] * 0.35 for axis in (0, 1))

    def _pointer_target(self, points):
        anchor = self._anchor(points)
        if self._virtual_offset is None:
            self._virtual_offset = tuple(points[8][axis] - anchor[axis] for axis in (0, 1))
        tip = tuple(anchor[axis] + self._virtual_offset[axis] for axis in (0, 1))
        left, top, right, bottom = self.CONTROL_REGION
        return (_clamp((tip[0] - left) / (right - left)),
                _clamp((tip[1] - top) / (bottom - top)))

    def _reanchor(self, points):
        if self._cursor is None:
            return
        left, top, right, bottom = self.CONTROL_REGION
        virtual_tip = (left + self._cursor[0] * (right - left),
                       top + self._cursor[1] * (bottom - top))
        anchor = self._anchor(points)
        self._virtual_offset = tuple(virtual_tip[axis] - anchor[axis] for axis in (0, 1))
        self._filter_target = self._cursor
        self._filter_velocity = (0.0, 0.0)

    def _complete_click(self, source):
        # A success event must follow a fully completed click, never just a
        # recognized pose, attempted down, or drag release.
        self._owned_left = True
        self._device.left_down()
        self._release()
        self._click_id += 1
        return {"click_id": self._click_id, "click_source": source}

    def _move(self, target, dt):
        if self._cursor is None:
            self._cursor = self._device.position()
        dt = max(0.001, dt)
        if self._filter_target is None:
            self._filter_target = self._cursor
        # Smooth signed velocity before increasing response speed. Alternating
        # landmark noise largely cancels instead of opening the filter; steady
        # travel rapidly reduces smoothing delay. All constants are in seconds.
        velocity_alpha = 1.0 - math.exp(-dt / self.VELOCITY_SMOOTH_SECONDS)
        velocity = tuple((new - old) / dt for new, old in zip(target, self._filter_target))
        self._filter_velocity = tuple(old + (new - old) * velocity_alpha
                                      for old, new in zip(self._filter_velocity, velocity))
        speed = math.hypot(*self._filter_velocity)
        smooth_seconds = max(self.FAST_SMOOTH_SECONDS,
                             self.REST_SMOOTH_SECONDS / (1.0 + self.VELOCITY_GAIN * speed))
        alpha = 1.0 - math.exp(-dt / smooth_seconds)
        position = tuple(_clamp(old + (new - old) * alpha)
                         for old, new in zip(self._cursor, target))
        position = tuple(new if new in (0.0, 1.0) and abs(new - value) < 0.0005 else value
                         for value, new in zip(position, target))
        self._device.move(*position)
        self._cursor = position
        self._filter_target = target

    def _engage_pointer(self, wrist, scale, straight, others_curled, now):
        """Share the existing deliberate pointing dwell across both control modes."""
        # An already engaged hand may relax while its proximal anchor remains reliable.
        if self._armed:
            return None
        # Starting movement still requires one straight index with the other fingers curled.
        if not straight or not others_curled:
            return self._cancel("Straighten your index finger briefly to start Hand Mouse")
        # Moving too far during the initial dwell starts a fresh activation attempt.
        if self._point_since is None or math.dist(wrist, self._point_origin) > scale * 0.45:
            self._point_since, self._point_origin = now, wrist
        # Measure the dwell using the caller's monotonic tracking time.
        elapsed = now - self._point_since
        # Report progress without creating an input device before the dwell completes.
        if elapsed < self.ARM_SECONDS:
            return self._feedback("mouse_arming", "Hold your pointing hand briefly to start",
                                  elapsed / self.ARM_SECONDS)
        # Construct the device lazily only after deliberate activation.
        if self._device is None:
            self._device = self._factory()
        # Preserve the physical-button/modifier guard before the first movement.
        blocked = self._device.blocked(own_left=False)
        # Cancel instead of competing with existing physical input.
        if blocked:
            return self._cancel(blocked)
        # Authorize motion; this flag alone never presses a mouse button.
        self._armed = True
        # Returning no feedback tells the caller that it may move the pointer now.
        return None

    def update(self, points, now, *, hold=False):
        with self._lock:
            try:
                # A string or integer must never accidentally enable target freezing.
                if type(hold) is not bool:
                    raise ValueError("hold must be a boolean")
                # Freezing requires a controller whose interaction with EEG input is explicit.
                if hold and not (self.pointer_only or self.eeg_compatible):
                    raise ValueError("hold requires pointer_only=True or eeg_compatible=True")
                return self._update(points, now, hold=hold)
            except Exception as exc:
                return self._cancel("Hand Mouse paused", error=f"Hand Mouse input failed: {exc}")

    def _update(self, points, now, *, hold=False):
        # Recompute manual pose intent only from this frame's validated hand geometry.
        self._finger_pose_active = False
        # Report a manual release only on the frame where it actually occurred.
        self._finger_release_frame = False
        if self._fault:
            return self._cancel("Hand Mouse paused", error=self._fault)
        if not isinstance(now, (int, float)) or not math.isfinite(now):
            return self._cancel("Tracking time changed; point again")
        if self._last_input_at is not None and now <= self._last_input_at:
            return self._cancel("Tracking time changed; point again")
        self._last_input_at = now
        if self._device is not None:
            blocked = self._device.blocked(own_left=self._owned_left)
            if blocked:
                return self._cancel(blocked)
        dt = 1 / 30 if self._last_at is None else now - self._last_at
        if self._last_at is not None and (dt <= 0 or dt > self.MAX_GAP_SECONDS):
            return self._cancel("Tracking paused; point again")
        if self._uncertain_since is not None and now < self._uncertain_since:
            return self._cancel("Tracking time changed; point again")
        landmarks = self._landmarks(points)
        if landmarks is None:
            return self._freeze(now, "Hand not visible")
        wrist, tip = landmarks[0], landmarks[8]
        scale = math.dist(wrist, landmarks[9])
        if not 0.025 <= scale <= 0.6:
            return self._cancel("Keep your whole hand visible; point again")
        if self._last_wrist is not None and (
                math.dist(wrist, self._last_wrist) > max(0.22, self._last_scale * 2.0)
                or not 0.55 <= scale / self._last_scale <= 1.8):
            return self._cancel("Hand position changed suddenly; point again")
        others_curled = not any(self._extended(landmarks, t, p)
                               for t, p in ((12, 10), (16, 14), (20, 18)))
        mcp, pip = landmarks[5], landmarks[6]
        proximal = tuple(pip[axis] - mcp[axis] for axis in (0, 1))
        knuckle = tuple(mcp[axis] - wrist[axis] for axis in (0, 1))
        proximal_squared = sum(value * value for value in proximal)
        # A hooked index keeps its proximal joint raised. A full fist folds
        # that joint into the palm, and must pause rather than click.
        index_raised = (proximal_squared > 1e-8
                  and math.dist(pip, wrist) > math.dist(mcp, wrist) * 1.18
                  and sum(a * b for a, b in zip(proximal, knuckle)) > scale * scale * 0.20)
        raised = others_curled and index_raised
        extension = (sum((tip[axis] - pip[axis]) * proximal[axis] for axis in (0, 1))
                     / max(1e-8, proximal_squared))
        point = index_raised and self._extended(landmarks, 8, 6)
        distal_alignment = self._distal_alignment(points, scale)
        straight = (point and extension >= 0.60 and distal_alignment is not None
                    and distal_alignment >= self.DISTAL_STRAIGHT_ALIGNMENT)
        hook = (raised and distal_alignment is not None
                and (extension <= 0.08 or distal_alignment <= self.DISTAL_BEND_ALIGNMENT))
        pinch_distance = math.dist(landmarks[4], tip) / scale
        # Thumb contact naturally curls the fingertip back toward the palm.
        # Keep the proximal index raised and the tip away from the wrist;
        # requiring the tip beyond the palm rejects ordinary relaxed pinches.
        # The full-fist check below still cancels instead of clicking.
        pinch_shape = (index_raised and math.dist(tip, wrist) > scale * 0.80
                       and math.dist(landmarks[6], wrist) > scale * 1.10)
        fist = (others_curled and not point
                and math.dist(tip, wrist) <= scale * 1.25
                and math.dist(pip, wrist) <= scale * 1.20)
        if fist:
            return self._cancel("Hand Mouse paused; point to resume")
        # Once engaged, fingers may relax or open. Only the proximal anchor
        # needs to remain usable for movement; clicks still require their pose.
        movement_shape = (proximal_squared >= (scale * 0.12) ** 2
                          and math.dist(mcp, wrist) > scale * 0.35
                          and math.dist(pip, wrist) > scale * 0.8)
        anchor = self._anchor(landmarks)
        if (self._last_anchor is not None
                and math.dist(anchor, self._last_anchor) > max(0.22, self._last_scale * 2.0)):
            return self._cancel("Pointer tracking jumped; point again")
        if self._uncertain_since is not None:
            if now - self._uncertain_since >= self.RECOVERY_SECONDS:
                return self._cancel("Tracking lost; point briefly to resume")
            # A nearby, similarly sized hand may resume movement. A changed
            # hand position must go through deliberate activation again.
            radius = min(0.12, max(0.04, self._last_scale * 0.5))
            if (math.dist(wrist, self._last_wrist) > radius
                    or math.dist(anchor, self._last_anchor) > radius
                    or not 0.75 <= scale / self._last_scale <= 1.33):
                return self._cancel("Hand position changed; point briefly to resume")
            if not movement_shape:
                return self._freeze(now, "Hand pose uncertain")
            self._uncertain_since = None
            # A physical mouse may have moved during a brief gap; combined control must not jump back.
            if self.eeg_compatible:
                self._cursor = self._device.position()
            self._reanchor(landmarks)
            self._last_at, self._last_wrist, self._last_scale = now, wrist, scale
            self._last_anchor = anchor
            # Reacquisition only reanchors. Pending holds were discarded; a
            # fresh neutral pose is required before another click can start.
            # The EEG aiming mode has no pinch rearm requirement or hand-generated clicks.
            hint = ("Hand found; move your hand to aim; EEG controls selection"
                    if self.pointer_only else "Hand found; separate thumb and index before clicking")
            return self._feedback("mouse_pointer", hint)

        if not movement_shape:
            return self._freeze(now, "Hand pose uncertain")
        self._last_at, self._last_wrist, self._last_scale = now, wrist, scale
        self._last_anchor = anchor
        # A bent finger retains manual priority after its one click until the user relaxes it.
        self._finger_pose_active = bool(self.eeg_compatible and self._armed and self.bend_click and hook)

        # Finish the movement-only path before any pinch, bend, or drag logic can run.
        if self.pointer_only:
            # Reuse ordinary Hand Mouse's deliberate pointing activation and input guards.
            pending = self._engage_pointer(wrist, scale, straight, others_curled, now)
            # Arming or paused feedback means no pointer movement is authorized yet.
            if pending is not None:
                return pending
            # A selection holds the existing target while all tracking/input checks still run above.
            if hold:
                # A hold during initial engagement uses the current physical cursor without moving it.
                if self._cursor is None:
                    self._cursor = self._device.position()
                # Keep the hand-to-cursor offset current while the hand moves during selection.
                self._reanchor(landmarks)
                # Remember to consume one reanchoring frame when selection releases the target.
                self._pointer_held = True
                # A validated held hand remains ready for the separate EEG completion check.
                return self._feedback("mouse_pointer", "Pointer held; EEG selection in progress", 1.0)
            # Releasing a hold must not map accumulated hand movement onto the pointer.
            if self._pointer_held:
                # Clear the hold only after a valid release frame has passed the tracking guards.
                self._pointer_held = False
                # Align this release frame with the held cursor before accepting new motion.
                self._reanchor(landmarks)
                # Resume on the next frame, preserving the target through this transition.
                return self._feedback("mouse_pointer", "Move your hand to aim; EEG controls selection", 1.0)
            # Aim with the same stable palm/index anchor and adaptive smoothing as Hand Mouse.
            self._move(self._pointer_target(landmarks), dt)
            # Report a ready target while leaving all button input to the separate EEG controller.
            return self._feedback("mouse_pointer", "Move your hand to aim; EEG controls selection", 1.0)

        if self._pinch_since is not None:
            duration = now - self._pinch_since
            if not pinch_shape:
                return self._freeze(now, "Pinch interrupted")
            if pinch_distance > 0.55:
                # A release frame still belongs to the finger gesture and cannot complete an EEG click.
                self._finger_release_frame = True
                click = {}
                if self._owned_left:
                    self._release()
                    message, state = "Drag released; move your hand", "mouse_pointer"
                elif duration >= self.CLICK_MIN_SECONDS:
                    # If release is the first frame beyond the drag threshold,
                    # no held frame ever began a drag. Complete one click rather
                    # than dropping it because of camera sampling cadence.
                    click = self._complete_click("pinch")
                    message, state = "Clicked; move your hand", "mouse_clicked"
                else:
                    message, state = "Pinch released; move your hand", "mouse_pointer"
                self._pinch_since = None
                self._bend_latched = True
                self._rearm_since = None
                self._point_since, self._point_origin = now, wrist
                self._reanchor(landmarks)
                return self._feedback(state, message, **click)
            if not self._owned_left and duration >= self.DRAG_SECONDS:
                self._owned_left = True
                self._device.left_down()
                self._drag_origin, self._drag_cursor = wrist, self._cursor
            if self._owned_left:
                left, top, right, bottom = self.CONTROL_REGION
                target = (_clamp(self._drag_cursor[0] + (wrist[0] - self._drag_origin[0]) / (right - left)),
                          _clamp(self._drag_cursor[1] + (wrist[1] - self._drag_origin[1]) / (bottom - top)))
                self._move(target, dt)
                return self._feedback("mouse_dragging", "Dragging; open the pinch to release", 1.0)
            return self._feedback("mouse_pinch", "Release to click; keep pinching to drag",
                                  duration / self.DRAG_SECONDS)

        rearm_action = "straighten your index" if self.bend_click else "separate thumb and index"
        if self._bend_latched:
            # Pinch-only control rearms on a stable opening, even if the index
            # stays curled. Straightening is needed only when a bend can click,
            # so releasing a pinch cannot immediately trigger that second input.
            if (straight or not self.bend_click) and pinch_distance > 0.55:
                if self._rearm_since is None:
                    self._rearm_since = now
                if now - self._rearm_since >= self.REARM_SECONDS:
                    self._bend_latched = False
                    self._rearm_since = None
            else:
                self._rearm_since = None

        if self._armed and pinch_distance < 0.35 and pinch_shape and not self._bend_latched:
            self._bend_since = None
            self._pinch_since = now
            self._pinch_approaching = False
            # Preserve the last pointer position: index movement as the pinch
            # closes must not shift the click target.
            return self._feedback("mouse_pinch", "Release to click; keep pinching to drag")

        self._pinch_approaching = bool(self._armed and pinch_shape and pinch_distance <= 0.55)
        if self._pinch_approaching:
            # Continue aiming from the proximal anchor until actual contact.
            # Freezing/reanchoring on either side of this band discarded hand
            # travel whenever landmark noise crossed the boundary repeatedly.
            # Distal fingertip closure does not change the proximal anchor.
            self._bend_since = None
        elif pinch_distance <= 0.55:
            return self._freeze(now, "Separate thumb and index before clicking")
        # Ordinary Hand Mouse keeps the same activation behavior after its pinch checks.
        pending = self._engage_pointer(wrist, scale, straight, others_curled, now)
        if pending is not None:
            return pending
        # A stable new point resumes a suspended combined controller at the current desktop target.
        if self.eeg_compatible and self._resume_at_physical_cursor:
            # Read physical cursor position only after fresh pointing and the normal input guards pass.
            self._cursor = self._device.position()
            # Bind this recovered hand pose to that cursor instead of its old absolute camera location.
            self._reanchor(landmarks)
            # Consume recovery once; later frames may move relative to the newly anchored pose.
            self._resume_at_physical_cursor = False
            # Returning before click and movement paths guarantees that reacquisition itself emits no input.
            return self._feedback("mouse_pointer", "Hand found; move your hand to aim or pinch to click", 1.0)
        if self.bend_click and hook and not self._bend_latched and not self._pinch_approaching:
            if self._bend_since is None:
                self._bend_since = now
            duration = now - self._bend_since
            if duration >= self.BEND_SECONDS:
                click = self._complete_click("bend")
                self._bend_latched = True
                self._bend_since = None
                self._rearm_since = None
                self._reanchor(landmarks)
                return self._feedback("mouse_clicked", "Clicked; straighten your index to click again", **click)
            return self._feedback("mouse_bend", "Bend held: click coming", duration / self.BEND_SECONDS)
        if self._bend_since is not None:
            self._bend_since = None
            self._reanchor(landmarks)
        # Finger click and drag paths above always run before an EEG target hold can freeze aiming.
        if self.eeg_compatible and hold:
            # Initial engagement during a hold anchors at the current physical pointer without moving it.
            if self._cursor is None:
                self._cursor = self._device.position()
            # Discard hand travel accumulated while EEG is selecting, preserving the selected target.
            self._reanchor(landmarks)
            # Require a reanchoring release frame before ordinary aiming resumes.
            self._pointer_held = True
            # Finger input remains available on the next frame despite this EEG-only movement hold.
            return self._feedback("mouse_pointer", "EEG target held; pinch to click or hold to drag still works", 1.0)
        # Ending an EEG hold must not jump to the hand's new absolute position.
        if self.eeg_compatible and self._pointer_held:
            # Consume the hold only after normal hand gesture handling and tracking validation.
            self._pointer_held = False
            # Bind the released hand pose to the current pointer before resuming motion.
            self._reanchor(landmarks)
            # Preserve the cursor on this first release frame; subsequent movement is intentional.
            return self._feedback("mouse_pointer", "Move your hand; pinch to click or hold to drag", 1.0)
        self._move(self._pointer_target(landmarks), dt)
        hint = (f"Move your hand; {rearm_action} before another click" if self._bend_latched
                else "Move your hand; bring thumb and index together to click" if self._pinch_approaching
                else "Move your hand; pinch to click or hold to drag")
        return self._feedback("mouse_pointer", hint, 1.0)
