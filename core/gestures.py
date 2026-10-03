"""Hand gestures as an input actor.

A camera is the least deliberate input this system takes. There is no keystroke
behind a gesture, the user may have been waving at someone else in the room, and
a frame or two of noise can look like a swipe. Three things follow from that, and
they are the whole design:

  * **Classification is pure.** Turning 21 landmarks into "fist" or "pinch" is a
    function of the landmarks and nothing else, so every gesture in the vocabulary
    is tested against fixed coordinates with no camera involved. Only the capture
    loop needs hardware.
  * **A gesture must persist to count.** Poses use time-based holds; a steady open
    palm arms motion measured in palm lengths. Slow motion keeps its progress,
    and holding a pose never repeats an action.
  * **The gate decides, not this module.** Events dispatch as ``actor="gesture"``,
    which the capability gate confines to reversible actions. Close-request and
    close-confirm poses are separate events for the backend's confirmation flow.

MediaPipe and OpenCV are optional. Import failure is reported as unavailability
with a readable reason, exactly as voice does, rather than taking the backend down.
"""

from __future__ import annotations

import base64
import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional


def _has_legacy_hands(mp) -> bool:
    return hasattr(getattr(getattr(mp, "solutions", None), "hands", None), "Hands")

# MediaPipe hand landmark indices.
WRIST = 0
THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 4, 8, 12, 16, 20
INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP = 6, 10, 14, 18
THUMB_IP, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 3, 5, 9, 17

# Gesture vocabulary.
OPEN_PALM = "open_palm"
FIST = "fist"
POINT = "point"
PINCH = "pinch"
TWO_FINGER = "two_finger"
FOUR_FINGER = "four_finger"
THUMBS_UP = "thumbs_up"
NONE = "none"
UNKNOWN = "unknown"
RESTORE_WINDOW = "restore_window"
CLOSE_REQUEST, CLOSE_CONFIRM, CLOSE_CANCEL = "close_request", "close_confirm", "close_cancel"

SWIPE_LEFT, SWIPE_RIGHT, SWIPE_UP, SWIPE_DOWN = (
    "swipe_left", "swipe_right", "swipe_up", "swipe_down",
)


def _distance(a, b) -> float:
    return math.dist((a[0], a[1]), (b[0], b[1]))


def hand_scale(landmarks) -> float:
    """A length to measure against, so thresholds do not depend on distance.

    Wrist to middle knuckle stays roughly constant as fingers move, which the
    span of the hand does not. Without normalising, every threshold would only
    hold at one distance from the camera.
    """
    return max(1e-6, _distance(landmarks[WRIST], landmarks[MIDDLE_MCP]))


def finger_extended(landmarks, tip: int, pip: int) -> bool:
    """Whether a finger is straight, judged by reach rather than by height.

    Comparing tip.y to pip.y is the usual shortcut and it silently assumes the
    hand points up: it misreads a hand held sideways or upside down. A curled
    finger brings its tip back toward the wrist, so comparing distances is true
    at any rotation.
    """
    wrist = landmarks[WRIST]
    return _distance(landmarks[tip], wrist) > _distance(landmarks[pip], wrist) * 1.08


def thumb_extended(landmarks) -> bool:
    return _distance(landmarks[THUMB_TIP], landmarks[PINKY_MCP]) > hand_scale(landmarks) * 1.05


def is_pinching(landmarks) -> bool:
    return _distance(landmarks[THUMB_TIP], landmarks[INDEX_TIP]) < hand_scale(landmarks) * 0.45


def is_navigation_pose(landmarks) -> bool:
    """Four raised fingers engage navigation; thumb position is free unless pinching."""
    return bool(landmarks and len(landmarks) >= 21
                and all(finger_extended(landmarks, tip, pip)
                        for tip, pip in ((INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
                                         (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP)))
                and not is_pinching(landmarks))


def _holds_navigation_pose(landmarks) -> bool:
    """After deliberate activation, one curling/occluded finger is tolerable."""
    return bool(landmarks and len(landmarks) >= 21
                and sum(finger_extended(landmarks, tip, pip)
                        for tip, pip in ((INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
                                         (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP))) >= 3
                and not is_pinching(landmarks))


def classify(landmarks) -> str:
    """One hand pose from 21 normalised landmarks. Pure, so it is testable."""
    if not landmarks or len(landmarks) < 21:
        return NONE

    index = finger_extended(landmarks, INDEX_TIP, INDEX_PIP)
    middle = finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP)
    ring = finger_extended(landmarks, RING_TIP, RING_PIP)
    pinky = finger_extended(landmarks, PINKY_TIP, PINKY_PIP)
    thumb = thumb_extended(landmarks)

    # Checked before the finger counts: a pinch holds the index near the thumb,
    # which several of the counted poses would otherwise also match.
    if is_pinching(landmarks) and not (middle and ring and pinky):
        return PINCH
    if index and middle and ring and pinky:
        return OPEN_PALM
    if not index and not middle and not ring and not pinky:
        return THUMBS_UP if thumb else FIST
    if index and middle and not ring and not pinky:
        return TWO_FINGER
    if index and not middle and not ring and not pinky:
        return POINT
    return NONE


@dataclass
class _Track:
    x: float
    y: float
    at: float


class MotionTracker:
    """Detects a swipe: travel far enough, fast enough, mostly along one axis.

    All three conditions matter. Distance alone promotes a slow reposition of the
    hand into a command; speed alone promotes a twitch; and without the axis
    ratio a diagonal gesture fires two directions at once.
    """

    def __init__(self, min_travel: float = 0.22, max_duration_s: float = 0.6,
                 axis_ratio: float = 1.6):
        self.min_travel = min_travel
        self.max_duration_s = max_duration_s
        self.axis_ratio = axis_ratio
        self._points: list = []

    def reset(self) -> None:
        self._points.clear()

    def update(self, x: float, y: float, now: float) -> Optional[str]:
        self._points.append(_Track(x, y, now))
        cutoff = now - self.max_duration_s
        self._points = [p for p in self._points if p.at >= cutoff]
        if len(self._points) < 3:
            return None

        first, last = self._points[0], self._points[-1]
        dx, dy = last.x - first.x, last.y - first.y
        adx, ady = abs(dx), abs(dy)

        if adx >= self.min_travel and adx > ady * self.axis_ratio:
            self.reset()
            # The camera image is mirrored for the user, so a hand moving right
            # on screen is the user's own leftward motion.
            return SWIPE_RIGHT if dx > 0 else SWIPE_LEFT
        if ady >= self.min_travel and ady > adx * self.axis_ratio:
            self.reset()
            return SWIPE_DOWN if dy > 0 else SWIPE_UP
        return None


@dataclass
class GestureEvent:
    name: str
    at: float
    position: tuple = (0.0, 0.0)
    scale: float = 0.0


class GestureStabiliser:
    """Turns a stream of per-frame poses into deliberate events.

    A pose must repeat for `hold_frames` before it is believed, and a believed
    gesture cannot fire again until `cooldown_s` has passed. The cooldown is the
    important half: without it a hand held still in front of the camera fires the
    same action every frame, which is how a held open palm becomes forty window
    moves a second.
    """

    def __init__(self, hold_frames: int = 4, cooldown_s: float = 0.8):
        self.hold_frames = hold_frames
        self.cooldown_s = cooldown_s
        self._candidate = NONE
        self._streak = 0
        self._last_fired: dict = {}

    def feed(self, pose: str, now: float) -> Optional[str]:
        if pose != self._candidate:
            self._candidate = pose
            self._streak = 1
            return None
        self._streak += 1
        if pose == NONE or self._streak != self.hold_frames:
            return None
        return pose if self.allow(pose, now) else None

    def allow(self, name: str, now: float) -> bool:
        """Whether this gesture's cooldown has elapsed, recording it if so.

        Absence is None rather than 0.0: defaulting to zero reads as "it fired at
        the start of time", which puts every gesture inside its own cooldown for
        the first few seconds of a clock that starts near zero and silently
        swallows the first one.
        """
        last = self._last_fired.get(name)
        if last is not None and now - last < self.cooldown_s:
            return False
        self._last_fired[name] = now
        return True


class MeasuredGestureTracker:
    """Deliberate, reversible navigation with bounded uncertainty and completion.

    A strict clutch arms once. Trusted travel completes after a short dwell;
    uncertainty can preserve motion briefly, but never completes it. Completed
    or cancelled navigation stays latched until the hand is neutral or lowered.
    """

    POSE_ACTIONS = {
        POINT: (RESTORE_WINDOW, 0.65, "Hold one finger to restore the window"),
        TWO_FINGER: (TWO_FINGER, 0.65, "Hold two fingers to switch windows"),
        PINCH: (CLOSE_REQUEST, 1.0, "Hold a pinch to request closing the window"),
        THUMBS_UP: (CLOSE_CONFIRM, 0.65, "Hold thumbs up to confirm a pending close"),
        FIST: (CLOSE_CANCEL, 0.35, "Hold a fist to cancel a pending close"),
    }
    UNCERTAIN_SECONDS = 0.15
    COMMIT_SECONDS = 0.10
    COMMIT_HYSTERESIS = 0.90
    NEUTRAL_SECONDS = 0.20
    IDLE_SECONDS = 8.0
    SESSION_SECONDS = 20.0
    MAX_FRAME_SECONDS = 0.30

    def __init__(self, travel_palms: float = 1.2, arm_seconds: float = 0.30,
                 release_seconds: float = 0.12, *, clutch_pose: str = OPEN_PALM,
                 pose_actions: bool = True):
        if not math.isfinite(travel_palms) or travel_palms <= 0:
            raise ValueError("travel_palms must be a positive finite distance")
        if clutch_pose not in (OPEN_PALM, FOUR_FINGER):
            raise ValueError("clutch_pose must be open_palm or four_finger")
        self.travel_palms = travel_palms
        self.arm_seconds = arm_seconds
        self.release_seconds = release_seconds
        self.clutch_pose = clutch_pose
        self.pose_actions = pose_actions
        self._clutch_hint = "four fingers" if clutch_pose == FOUR_FINGER else "an open palm"
        self._clear()

    def _clear(self):
        self._pose = NONE
        self._since = 0.0
        self._session_since = 0.0
        self._origin = (0.0, 0.0)
        self._scale = 0.1
        self._anchor = None
        self._axis = None
        self._progress = 0.0
        self._fired = False
        self._last = None
        self._last_scale = None
        self._last_frame_at = None
        self._last_activity_point = None
        self._last_activity_at = 0.0
        self._release_pose = NONE
        self._release_since = 0.0
        self._unclear_since = None
        self._neutral_since = None
        self._commit_direction = 0
        self._commit_elapsed = 0.0
        self._commit_frame_at = None

    @staticmethod
    def _palm_size(scale):
        return max(0.05, min(0.35, scale))

    @staticmethod
    def _feedback(state, progress, text, now, **extra):
        return {"kind": "gesture_progress", "state": state, "progress": progress,
                "text": text, "at": now, **extra}

    def _motion(self, phase, now, **extra):
        return {"kind": "motion", "phase": phase, "progress": self._progress,
                "axis": self._axis, "at": now, **extra}

    def reset(self, now: float = 0.0, reason: str = "hand_lost") -> list[dict]:
        records = []
        # Non-navigation poses may own a pending close approval. A completed
        # navigation latch, however, has already released its native contacts.
        if self._pose != NONE and not (self._pose == self.clutch_pose
                                       and self._fired and self._axis is None):
            records.append(self._motion("cancel", now, reason=reason))
        self._clear()
        records.append(self._feedback("idle", 0.0,
                                     f"Show {self._clutch_hint} and hold still to begin", now))
        return records

    def _begin_pose(self, pose, point, scale, now):
        self._clear()
        self._pose = pose
        self._since = self._session_since = now
        self._origin = point
        self._scale = self._palm_size(scale)
        self._last_frame_at = now

    def _clear_commit(self):
        self._commit_direction = 0
        self._commit_elapsed = 0.0
        self._commit_frame_at = None

    def _is_neutral(self, pose):
        # A pose used to leave navigation must not silently become a window
        # command. Desktop mode requires a fist/lowered hand before new holds.
        # Mouse mode has no pose actions and can safely hand back to pointing.
        return pose in ((NONE, FIST) if self.pose_actions else
                        (NONE, FIST, POINT, TWO_FINGER, THUMBS_UP))

    def _cancel_navigation(self, now, reason, text):
        records = [self._motion("cancel", now, reason=reason)]
        axis = self._axis
        self._anchor = self._axis = None
        self._fired = True
        self._unclear_since = None
        self._clear_commit()
        records.append(self._feedback("cancelled", 0.0, text, now, axis=axis))
        return records

    def _latched(self, pose, now):
        if not self._is_neutral(pose):
            self._neutral_since = None
            return []
        if self._neutral_since is None or now < self._neutral_since:
            self._neutral_since = now
        if now - self._neutral_since + 1e-9 < self.NEUTRAL_SECONDS:
            return []
        self._clear()
        return [self._feedback("idle", 0.0,
                               f"Ready: show {self._clutch_hint} to navigate", now)]

    def _freeze(self, pose, now):
        self._clear_commit()
        if self._unclear_since is None:
            self._unclear_since = self._last[2] if self._last is not None else now
        if self._is_neutral(pose):
            if self._neutral_since is None:
                self._neutral_since = now
        else:
            self._neutral_since = None
        if pose in (FIST, PINCH):
            if self._release_pose != pose:
                self._release_pose, self._release_since = pose, now
            if now - self._release_since + 1e-9 >= self.release_seconds:
                return self._cancel_navigation(now, "pose_cancelled",
                                               "Swipe cancelled; lower your hand or make a fist briefly")
        else:
            self._release_pose = NONE
        if now - self._unclear_since > self.UNCERTAIN_SECONDS + 1e-9:
            reason = "hand_lost" if pose == NONE else "pose_unclear"
            return self._cancel_navigation(now, reason,
                                           "Tracking paused; lower your hand or make a fist before another swipe")
        # Keep an already-started native gesture alive at its last safe position.
        # Heartbeats never add completion time or introduce new contact sets.
        records = [self._motion("update", now, frozen=True)] if self._axis is not None else []
        records.append(self._feedback("uncertain", self._progress,
                                      "Tracking uncertain; keep your hand visible to continue", now,
                                      axis=self._axis, commit_progress=0.0))
        return records

    def _progress_feedback(self, now, *, recovered=False):
        horizontal = self._axis == "horizontal"
        direction = ("right" if self._progress >= 0 else "left") if horizontal else (
            "down" if self._progress >= 0 else "up")
        destination = (("Previous desktop" if direction == "right" else "Next desktop")
                       if horizontal else ("Show desktop" if direction == "down" else "Task View"))
        state = "desktop" if horizontal else "overview"
        text = f"{destination}; keep moving, or make a fist to cancel"
        if recovered:
            text = f"Hand found; continue toward {destination.lower()}"
        if self._commit_frame_at is not None:
            state = "committing"
            text = f"Hold briefly to finish: {destination}"
        return self._feedback(state, self._progress, text, now, axis=self._axis,
                              direction=direction, commit_progress=min(1.0, self._commit_elapsed / self.COMMIT_SECONDS))

    def update(self, pose: str, x: float, y: float, scale: float,
               now: float) -> list[dict]:
        if not isinstance(now, (int, float)) or not math.isfinite(now):
            return self.reset(0.0, "tracking_time_changed")
        previous_at = self._last_frame_at
        self._last_frame_at = now
        if self._pose == self.clutch_pose and self._fired:
            return self._latched(pose, now)
        active = self._pose == self.clutch_pose and self._anchor is not None
        if previous_at is not None and (now < previous_at or now - previous_at > self.MAX_FRAME_SECONDS + 1e-9):
            if active:
                return self._cancel_navigation(now, "tracking_gap",
                                               "Tracking paused; lower your hand or make a fist to restart")
            if self._pose == self.clutch_pose:
                self._pose = NONE
        if active and now - self._session_since >= self.SESSION_SECONDS:
            return self._cancel_navigation(now, "gesture_timeout",
                                           "Swipe timed out; lower your hand or make a fist to restart")
        if active and now - self._last_activity_at >= self.IDLE_SECONDS:
            return self._cancel_navigation(now, "gesture_idle",
                                           "Swipe paused too long; lower your hand or make a fist to restart")
        if pose == NONE or not all(math.isfinite(v) for v in (x, y, scale)) or scale <= 0:
            return self._freeze(NONE, now) if active else self.reset(now, "hand_lost")

        point = (x, y)
        if active:
            if self._last is not None:
                last_x, last_y, last_at = self._last
                distance = math.dist(point, (last_x, last_y))
                limit = max(0.16, self._scale * 1.2) + min(0.20, max(0.0, now - last_at) * 1.5)
                if distance > limit or not 0.6 <= scale / self._last_scale <= 1.67:
                    return self._cancel_navigation(now, "tracking_jump",
                                                   "Hand changed suddenly; lower your hand or make a fist to restart")
            if pose != self.clutch_pose:
                return self._freeze(pose, now)
            if self._unclear_since is not None:
                if now - self._unclear_since > self.UNCERTAIN_SECONDS + 1e-9:
                    return self._cancel_navigation(now, "pose_unclear",
                                                   "Tracking paused; lower your hand or make a fist to restart")
                radius = min(0.12, max(0.04, self._scale * 0.5))
                if (math.dist(point, self._last[:2]) > radius
                        or not 0.75 <= scale / self._last_scale <= 1.33):
                    return self._cancel_navigation(now, "tracking_jump",
                                                   "A different hand position appeared; lower your hand to restart")
                displacement = self._progress * self.travel_palms * self._scale
                self._anchor = ((x - displacement, y) if self._axis == "horizontal" else
                                (x, y - displacement) if self._axis == "vertical" else point)
                self._last, self._last_scale = (x, y, now), scale
                self._unclear_since = self._neutral_since = None
                self._release_pose = NONE
                self._clear_commit()
                if self._axis is None:
                    return [self._feedback("armed", 0.0, "Hand found; move to navigate", now)]
                return [self._motion("update", now, frozen=True), self._progress_feedback(now, recovered=True)]

        if pose != self._pose:
            self._begin_pose(pose, point, scale, now)
        self._last, self._last_scale = (x, y, now), scale
        if pose == self.clutch_pose:
            if now - self._session_since >= self.SESSION_SECONDS:
                return self._cancel_navigation(now, "gesture_timeout",
                                               "Navigation timed out; lower your hand or make a fist to restart")
            if self._anchor is None:
                if math.dist(point, self._origin) > self._scale * 0.22:
                    self._origin, self._since = point, now
                    self._scale = self._palm_size(scale)
                held = now - self._since
                if held + 1e-9 < self.arm_seconds:
                    return [self._feedback("arming", max(0.0, held / self.arm_seconds),
                                           f"Hold {self._clutch_hint} still", now)]
                self._anchor = self._last_activity_point = point
                self._last_activity_at = now
                self._scale = self._palm_size(scale)
                return [self._feedback("armed", 0.0,
                                       "Move left/right for desktops, up for Task View, down for desktop", now)]

            if math.dist(point, self._last_activity_point) >= self._scale * 0.04:
                self._last_activity_point, self._last_activity_at = point, now
            dx, dy = (x - self._anchor[0]) / self._scale, (y - self._anchor[1]) / self._scale
            beginning = False
            if self._axis is None:
                if abs(dx) >= 0.25 and abs(dx) > abs(dy) * 1.35:
                    self._axis, beginning = "horizontal", True
                elif abs(dy) >= 0.25 and abs(dy) > abs(dx) * 1.35:
                    self._axis, beginning = "vertical", True
                else:
                    return []
            travel = dx if self._axis == "horizontal" else dy
            self._progress = max(-1.0, min(1.0, travel / self.travel_palms))
            if abs(self._progress) >= 1.0 - 1e-9:
                self._progress = math.copysign(1.0, self._progress)
            direction = 1 if self._progress >= 0 else -1
            if self._commit_frame_at is not None:
                if direction != self._commit_direction or abs(self._progress) < self.COMMIT_HYSTERESIS:
                    self._clear_commit()
                else:
                    # Count only bounded intervals between trusted frames;
                    # a delayed frame cannot supply an entire completion hold.
                    self._commit_elapsed += min(0.075, max(0.0, now - self._commit_frame_at))
                    self._commit_frame_at = now
            if self._commit_frame_at is None and abs(self._progress) >= 1.0:
                self._commit_direction, self._commit_frame_at = direction, now
            records = [self._motion("begin" if beginning else "update", now)]
            if self._commit_elapsed + 1e-9 >= self.COMMIT_SECONDS:
                self._progress = float(self._commit_direction)
                records[-1]["progress"] = self._progress
                records.append(self._motion("end", now, automatic=True))
                feedback = self._progress_feedback(now)
                feedback.update(state="completed", progress=self._progress, commit_progress=1.0,
                                text="Gesture complete; lower your hand or make a fist before another swipe")
                records.append(feedback)
                self._fired = True
                self._anchor = self._axis = None
                self._neutral_since = None
                self._clear_commit()
            else:
                records.append(self._progress_feedback(now))
            return records

        if not self.pose_actions:
            return []
        spec = self.POSE_ACTIONS.get(pose)
        if spec is None or self._fired:
            return []
        name, duration, text = spec
        if math.dist(point, self._origin) > self._scale * 0.25:
            self._origin, self._since = point, now
            self._scale = self._palm_size(scale)
        held = max(0.0, now - self._since)
        if held + 1e-9 >= duration:
            self._fired = True
            return [{"kind": "action", "name": name, "at": now},
                    self._feedback("triggered", 1.0, "Gesture recognized; relax your hand before repeating",
                                   now, gesture=name)]
        return [self._feedback("holding", held / duration, text, now, gesture=name)]


# ── What each gesture does ───────────────────────────────────────────────────
#
# Every entry is a capability the gate classifies free or reversible. An
# irreversible one would be refused at dispatch, so the binding table cannot
# quietly become a way around the gate.

DEFAULT_BINDINGS: dict = {
    SWIPE_UP:     ("os_window_state", {"state": "maximize"}),
    SWIPE_DOWN:   ("os_window_state", {"state": "minimize"}),
    RESTORE_WINDOW: ("os_window_state", {"state": "restore"}),
    TWO_FINGER:   ("os_cycle_window", {"direction": "next"}),
}


class GestureRecognizer:
    """Owns the camera, classifies frames, and reports deliberate gestures.

    Optional dependencies are resolved at `probe()` rather than import, so a
    machine with no camera or no mediapipe reports why instead of failing to
    start the backend.
    """

    def __init__(self, on_gesture: Optional[Callable] = None, camera_index: int = 0,
                 bindings: Optional[dict] = None, hold_frames: int = 4,
                 cooldown_s: float = 0.8, on_status: Optional[Callable] = None,
                 on_motion: Optional[Callable] = None,
                 on_progress: Optional[Callable] = None, travel_palms: float = 1.2,
                 model_complexity: int = 1, input_mode: str = "desktop",
                 mouse_factory: Optional[Callable] = None,
                 on_click: Optional[Callable] = None,
                 tracker_backend: str = "mediapipe", bend_click: bool = False):
        self._validate_model(model_complexity)
        self._validate_input_mode(input_mode)
        self._validate_tracker_backend(tracker_backend)
        self._validate_bend_click(bend_click)
        self.model_complexity = model_complexity
        self.tracker_backend = tracker_backend
        self.input_mode = input_mode
        self.bend_click = bend_click
        self.on_gesture = on_gesture
        self.on_status = on_status
        self.on_motion = on_motion
        self.on_progress = on_progress
        self.on_click = on_click
        self.camera_index = camera_index
        self.bindings = dict(DEFAULT_BINDINGS if bindings is None else bindings)
        self.stabiliser = GestureStabiliser(hold_frames, cooldown_s)
        self.motion = MotionTracker()
        self.measured = MeasuredGestureTracker(travel_palms=travel_palms)
        self._navigation = MeasuredGestureTracker(
            travel_palms=travel_palms, clutch_pose=FOUR_FINGER, pose_actions=False)
        self._navigation_active = False
        self._navigation_error = None
        self._navigation_axis = None
        self._last_progress = None
        self._last_progress_at = -math.inf
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._running = False
        self._state = "off"
        self._reason = "Gestures have not been started."
        self._lock = threading.RLock()
        self._input_lock = threading.RLock()
        self._tracker_cleanup_lock = threading.RLock()
        self._failed_tracker = None
        self._tracker_error = None
        self._mouse_factory = mouse_factory
        self._mouse = None
        self._mouse_error = None
        self._mouse_pose = NONE
        self._last_mouse_click_id = 0
        self._click_session = uuid.uuid4().hex
        self._click_sequence = 0
        self._preview_lease_until = 0.0
        self._preview_snapshot = None
        self._preview_last_encoded_at = -math.inf
        self._capture_times = deque(maxlen=30)
        self._capture_sequence = 0
        self._diagnostic = self._empty_diagnostic()

    @staticmethod
    def _validate_model(value):
        if type(value) is not int or value not in (0, 1):
            raise ValueError("model_complexity must be 0 (lite) or 1 (full)")

    @staticmethod
    def _validate_input_mode(value):
        if type(value) is not str or value not in ("desktop", "mouse"):
            raise ValueError("input_mode must be 'desktop' or 'mouse'")

    @staticmethod
    def _validate_tracker_backend(value):
        if type(value) is not str or value not in ("mediapipe", "rtmpose", "wilor"):
            raise ValueError("tracker_backend must be 'mediapipe', 'rtmpose' or 'wilor'")

    @staticmethod
    def _validate_bend_click(value):
        if type(value) is not bool:
            raise ValueError("bend_click must be true or false")

    @property
    def model_name(self):
        if self.tracker_backend in ("rtmpose", "wilor"):
            from core.hand_tracking import TRACKER_NAMES
            return TRACKER_NAMES[self.tracker_backend]
        return "MediaPipe Hands full" if self.model_complexity == 1 else "MediaPipe Hands lite"

    @staticmethod
    def _empty_diagnostic():
        return {"image": None, "width": 0, "height": 0, "sequence": 0,
                "tracked": False, "raw_pose": NONE, "pose": NONE,
                "landmarks": [], "fps": 0.0, "state": "idle", "progress": 0.0,
                "hint": "Turn on the camera to see hand tracking", "last_reason": None,
                "captured_at": None}

    # ── Availability ─────────────────────────────────────────────────────

    def probe(self) -> dict:
        """Whether this machine can do gestures at all, and why not if it cannot."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            return {"available": False,
                    "text": "opencv-contrib-python is not installed. Install this project's requirements.txt."}
        if self.tracker_backend in ("rtmpose", "wilor"):
            try:
                from core.hand_tracking import probe_rtmpose, probe_wilor
                probe_tracker = probe_wilor if self.tracker_backend == "wilor" else probe_rtmpose
                probe = probe_tracker()
                return {"available": bool(probe["available"]),
                        "text": probe.get("reason") or f"{self.model_name} hand tracking is available."}
            except Exception as exc:
                return {"available": False, "text": f"{self.model_name} hand tracking is unavailable: {exc}"}
        try:
            import mediapipe as mp
        except ImportError:
            return {"available": False,
                    "text": "mediapipe is not installed. Install this project's requirements.txt."}
        if not _has_legacy_hands(mp):
            return {"available": False,
                    "text": "Installed mediapipe has no solutions.hands.Hands API. "
                            "Install the pinned requirements to enable gestures."}
        return {"available": True, "text": "Gesture recognition is available."}

    def status(self) -> dict:
        probe = self.probe()
        with self._lock:
            return {"available": probe["available"], "running": self._running,
                    "state": self._state if probe["available"] else "unavailable",
                    "text": self._reason if probe["available"] else probe["text"],
                    "model": self.model_name, "model_complexity": self.model_complexity,
                    "model_name": self.model_name,
                    "tracker_backend": self.tracker_backend,
                    "tracker_cleanup_pending": self._failed_tracker is not None,
                    "input_mode": self.input_mode,
                    "navigation_active": self._navigation_active,
                    "navigation_cleanup_pending": self._navigation_error is not None,
                    "bend_click": self.bend_click,
                    "bindings": {k: v[0] for k, v in self.bindings.items()}}

    def is_running(self) -> bool:
        return self._running

    @property
    def navigation_active(self) -> bool:
        """Whether a four-finger navigation candidate owns Hand Mouse input."""
        return self._navigation_active

    def configure(self, *, travel_palms: Optional[float] = None,
                  model_complexity: Optional[int] = None,
                  input_mode: Optional[str] = None,
                  tracker_backend: Optional[str] = None,
                  bend_click: Optional[bool] = None) -> dict:
        """Change capture settings while off, without altering a live drag."""
        with self._input_lock:
            with self._lock:
                if self._navigation_error:
                    return {"error": "Navigation input cleanup is still pending. Turn the camera off again to retry."}
                if self._failed_tracker is not None:
                    return {"error": "Hand tracker cleanup is still pending. Turn the camera off again to retry."}
                if self._running or self._state in ("starting", "stopping") or (self._thread and self._thread.is_alive()):
                    return {"error": "Turn the camera off before changing gesture settings."}
                try:
                    selected_mode = self.input_mode if input_mode is None else input_mode
                    self._validate_input_mode(selected_mode)
                    selected_model = self.model_complexity if model_complexity is None else model_complexity
                    self._validate_model(selected_model)
                    selected_tracker = self.tracker_backend if tracker_backend is None else tracker_backend
                    self._validate_tracker_backend(selected_tracker)
                    selected_bend_click = self.bend_click if bend_click is None else bend_click
                    self._validate_bend_click(selected_bend_click)
                    measured = MeasuredGestureTracker(
                        travel_palms=self.measured.travel_palms if travel_palms is None else travel_palms)
                    navigation = MeasuredGestureTracker(
                        travel_palms=measured.travel_palms, clutch_pose=FOUR_FINGER,
                        pose_actions=False)
                except (TypeError, ValueError) as exc:
                    return {"error": str(exc)}
            if selected_mode != self.input_mode or selected_tracker != self.tracker_backend:
                reason = "mode_changed" if selected_mode != self.input_mode else "tracker_changed"
                self._reset_controls(time.monotonic(), reason)
                if self._navigation_error:
                    return {"error": f"Navigation input could not release: {self._navigation_error}"}
                if self.input_mode == "mouse" and self._mouse_error:
                    return {"error": f"Hand mouse could not release its input: {self._mouse_error}"}
            with self._lock:
                if self._running or self._state in ("starting", "stopping") or (self._thread and self._thread.is_alive()):
                    return {"error": "Turn the camera off before changing gesture settings."}
                if selected_bend_click != self.bend_click and self._mouse is not None:
                    try:
                        diagnostic = self._mouse.set_bend_click(selected_bend_click)
                        self._consume_mouse_click(diagnostic, time.monotonic(), emit=False)
                        if isinstance(diagnostic, dict) and diagnostic.get("error"):
                            raise RuntimeError(str(diagnostic["error"]))
                    except Exception as exc:
                        self._mouse_error = str(exc)
                        return {"error": f"Hand mouse could not change bend clicking: {exc}"}
                    self._mouse_error = None
                self.measured = measured
                self._navigation = navigation
                self.model_complexity = selected_model
                self.tracker_backend = selected_tracker
                self.input_mode = selected_mode
                self.bend_click = selected_bend_click
                result = {"ok": True, "travel_palms": self.measured.travel_palms}
                if model_complexity is not None:
                    result.update(model=self.model_name, model_complexity=self.model_complexity)
                if input_mode is not None:
                    result["input_mode"] = self.input_mode
                if tracker_backend is not None:
                    result.update(tracker_backend=self.tracker_backend, model=self.model_name)
                if bend_click is not None:
                    result["bend_click"] = self.bend_click
                return result

    # ── On-demand camera diagnostics ─────────────────────────────────────

    def preview(self) -> dict:
        """Read the latest frame and renew a short encoding lease, never start capture.

        The capture worker owns the only camera and encodes at most eight small
        JPEGs a second while this method is being polled. The image and landmarks
        are one snapshot, so an overlay never uses a newer hand than the image.
        """
        now = time.monotonic()
        with self._lock:
            running = self._running and not self._stop.is_set()
            if not running:
                self._clear_preview_locked(self._diagnostic.get("last_reason") or "camera_off", end_lease=True)
            else:
                if now >= self._preview_lease_until:
                    self._preview_snapshot = None
                self._preview_lease_until = now + 2.0
            info = dict(self._preview_snapshot or self._diagnostic)
            captured_at = info.pop("captured_at", None)
            age_ms = max(0, round((now - captured_at) * 1000)) if captured_at is not None else None
            if age_ms is not None and age_ms > 500:
                self._preview_snapshot = None
                info.update(image=None, width=0, height=0, tracked=False, landmarks=[],
                            raw_pose=NONE, pose=NONE, state="waiting", progress=0.0,
                            hint="Waiting for fresh camera frames", last_reason="stale_frame")
            if not running:
                info.update(state=self._state, hint=self._reason)
            if self.input_mode != "mouse":
                info.pop("control_region", None)
            elif "control_region" in info:
                info["control_region"] = list(info["control_region"])
            return {**info, "landmarks": [list(point) for point in info["landmarks"]],
                    "running": running, "model": self.model_name,
                    "model_name": self.model_name, "effective_pose": info["pose"],
                    "last_cancel_reason": info["last_reason"],
                    "input_mode": self.input_mode,
                    "navigation_active": self._navigation_active,
                    "navigation_cleanup_pending": self._navigation_error is not None,
                    "bend_click": self.bend_click,
                    "tracker_backend": self.tracker_backend,
                    "model_complexity": self.model_complexity, "age_ms": age_ms}

    def _clear_preview_locked(self, reason, *, end_lease=False):
        self._preview_snapshot = None
        self._preview_last_encoded_at = -math.inf
        self._capture_times.clear()
        self._diagnostic = self._empty_diagnostic()
        self._diagnostic.update(sequence=self._capture_sequence, last_reason=reason)
        if end_lease:
            self._preview_lease_until = 0.0

    def _publish_preview(self, frame, points, frame_at, cv2) -> None:
        """Update cheap metadata every frame; JPEG work happens outside the lock."""
        now = time.monotonic()
        raw_pose = classify(points)
        if self.input_mode == "mouse" and is_navigation_pose(points):
            raw_pose = FOUR_FINGER
        if raw_pose == NONE and points:
            raw_pose = UNKNOWN
        with self._lock:
            if not self._running or self._stop.is_set():
                self._preview_snapshot = None
                return
            self._capture_sequence += 1
            self._capture_times.append(frame_at)
            duration = self._capture_times[-1] - self._capture_times[0]
            fps = (len(self._capture_times) - 1) / duration if duration > 0 else 0.0
            self._diagnostic.update(
                sequence=self._capture_sequence, tracked=bool(points),
                raw_pose=raw_pose,
                pose=self._mouse_pose if self.input_mode == "mouse" else self.measured._pose,
                landmarks=[list(point) for point in points],
                fps=round(fps, 1), captured_at=frame_at)
            if now >= self._preview_lease_until:
                self._preview_snapshot = None
                return
            # Do not keep a tracked-hand overlay after a frame has lost it.
            if not points and self._preview_snapshot and self._preview_snapshot["tracked"]:
                self._preview_snapshot = None
            if now - self._preview_last_encoded_at < 1.0 / 8.0:
                return
            self._preview_last_encoded_at = now
            snapshot = dict(self._diagnostic)
        try:
            height, width = frame.shape[:2]
            if width > 480:
                height = max(1, round(height * 480 / width))
                width = 480
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            if not ok:
                return
            encoded = base64.b64encode(jpeg.tobytes()).decode("ascii")
            snapshot.update(image="data:image/jpeg;base64," + encoded, width=width, height=height)
        except Exception:
            # A preview encoder failure must not interrupt gesture controls.
            return
        with self._lock:
            if self._running and not self._stop.is_set() and time.monotonic() < self._preview_lease_until:
                self._preview_snapshot = snapshot

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> dict:
        with self._lock:
            if self._navigation_error:
                return {"error": "Navigation input cleanup is still pending. Turn the camera off again to retry."}
            if self._failed_tracker is not None:
                return {"error": "Hand tracker cleanup is still pending. Turn the camera off again to retry."}
            if self._state == "stopping":
                return {"error": "The camera is still stopping. Try again shortly."}
            if self._thread and self._thread.is_alive():
                if self._stop.is_set():
                    return {"error": "The camera is still stopping. Try again shortly."}
                return {"ok": True, "already": True}
            probe = self.probe()
            if not probe["available"]:
                self._state = "unavailable"
                self._reason = probe["text"]
                self._report_status()
                return {"error": probe["text"]}
            self._stop.clear()
            self._click_session = uuid.uuid4().hex
            self._click_sequence = 0
            self._clear_preview_locked("starting", end_lease=True)
            self._state = "starting"
            self._reason = "Starting gesture recognition."
            self.stabiliser = GestureStabiliser(
                self.stabiliser.hold_frames, self.stabiliser.cooldown_s)
            self.motion.reset()
            self.measured.reset()
            self._navigation.reset()
            self._navigation_active = False
            self._last_progress = None
            self._last_progress_at = -math.inf
            self._thread = threading.Thread(target=self._run, name="gestures", daemon=True)
            self._report_status()
            try:
                self._thread.start()
            except Exception as exc:
                self._state = "error"
                self._reason = f"Gesture capture could not start: {exc}"
                self._report_status()
                return {"error": self._reason}
            return {"ok": True}

    def stop(self) -> dict:
        with self._lock:
            self._stop.set()
            self._running = False
            self._clear_preview_locked("camera_stopped", end_lease=True)
            thread = self._thread
            self._state = "stopping" if thread and thread.is_alive() else "off"
            self._reason = "Releasing the camera." if self._state == "stopping" else "Gestures are off."
            self._report_status()
        self._reset_controls(time.monotonic(), "camera_stopped")
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        with self._lock:
            if thread and thread.is_alive():
                # The worker owns the device; do not claim release while a
                # driver is still finishing a read or startup operation.
                if self.input_mode == "mouse" and self._mouse_error:
                    self._reason = f"Releasing the camera. Hand mouse button release failed: {self._mouse_error}"
                    self._report_status()
                    return {"error": self._reason, "stopping": True}
                if self._navigation_error:
                    self._reason = f"Releasing the camera. Navigation input release failed: {self._navigation_error}"
                    self._report_status()
                    return {"error": self._reason, "stopping": True}
                return {"ok": True, "stopping": True}
        self._retry_tracker_cleanup()
        with self._lock:
            cleanup_error = self._cleanup_error_text()
            if cleanup_error:
                self._state = "error"
                self._reason = cleanup_error
                self._report_status()
                result = {"error": self._reason, "camera_released": True}
                if self._failed_tracker is not None:
                    result["tracker_cleanup_pending"] = True
                return result
            self._state = "off"
            self._reason = "Gestures are off. The camera is released."
            self._report_status()
            return {"ok": True}

    def _cleanup_error_text(self):
        """Describe retained input/tracker resources after camera release."""
        failures = []
        if self.input_mode == "mouse" and self._mouse_error:
            failures.append(f"hand mouse button release failed: {self._mouse_error}")
        if self._navigation_error:
            failures.append(f"navigation input release failed: {self._navigation_error}")
        if self._failed_tracker is not None:
            failures.append(f"hand tracker cleanup failed: {self._tracker_error}")
        return "The camera is released, but " + "; ".join(failures) if failures else None

    def _retry_tracker_cleanup(self):
        # Normal cleanup belongs to the capture worker. This path is only used
        # once that thread ended with a retained tracker that failed to close.
        with self._tracker_cleanup_lock:
            with self._lock:
                tracker = self._failed_tracker
                if tracker is None:
                    return
                self._state = "stopping"
                self._reason = "The camera is released. Retrying hand tracker cleanup."
                self._report_status()
            try:
                tracker.close()
            except Exception as exc:
                with self._lock:
                    self._tracker_error = str(exc)
            else:
                with self._lock:
                    self._failed_tracker = None
                    self._tracker_error = None

    # ── The capture loop ─────────────────────────────────────────────────

    def _run(self) -> None:
        camera = None
        hands = None
        failure = None
        try:
            import cv2
            if self.tracker_backend in ("rtmpose", "wilor"):
                from core.hand_tracking import TrackerStartupError, create_rtmpose, create_wilor
                create_tracker = create_wilor if self.tracker_backend == "wilor" else create_rtmpose
                try:
                    hands = create_tracker(self._stop)
                except TrackerStartupError as exc:
                    # Constructor cleanup may fail before a tracker is returned.
                    # The shared finally path retries and retains any survivor.
                    hands = exc.tracker
                    raise
            else:
                import mediapipe as mp
                if not _has_legacy_hands(mp):
                    raise RuntimeError("Installed mediapipe has no solutions.hands.Hands API.")
                hands = mp.solutions.hands.Hands(
                    max_num_hands=1, model_complexity=self.model_complexity,
                    min_detection_confidence=0.6, min_tracking_confidence=0.5,
                )

            if self._stop.is_set():
                return
            camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not camera.isOpened():
                camera.release()
                camera = cv2.VideoCapture(self.camera_index)
            if not camera.isOpened():
                raise RuntimeError(f"No camera at index {self.camera_index}. Check it is "
                                   "connected and not in use by another application.")

            # A smaller frame is enough for landmarks and keeps a webcam loop off
            # the CPU budget the rest of the system needs.
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            failed_reads = 0
            while not self._stop.is_set():
                ok, frame = camera.read()
                if not ok:
                    with self._lock:
                        self._clear_preview_locked("camera_read_failed")
                    self._reset_controls(time.monotonic(), "camera_read_failed")
                    failed_reads += 1
                    if failed_reads >= 20:
                        raise RuntimeError("The camera is not providing video frames.")
                    time.sleep(0.05)
                    continue
                failed_reads = 0
                frame_at = time.monotonic()
                frame = cv2.flip(frame, 1)  # mirror, so the user's left is left
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)
                with self._lock:
                    if self._stop.is_set():
                        break
                    if not self._running:
                        self._running = True
                        self._state = "running"
                        self._reason = "Watching for gestures."
                        self._report_status()
                now = time.monotonic()
                if self.tracker_backend in ("rtmpose", "wilor"):
                    points = self._tracker_points(result)
                else:
                    landmarks = (result.multi_hand_landmarks[0].landmark
                                 if result.multi_hand_landmarks else None)
                    points = [(p.x, p.y, p.z) for p in landmarks] if landmarks else []

                if not points:
                    self.stabiliser.feed(NONE, now)
                    self.motion.reset()
                    self._handle([], now)
                    self._publish_preview(frame, [], frame_at, cv2)
                    continue

                self._handle(points, now)
                self._publish_preview(frame, points, frame_at, cv2)
        except Exception as exc:  # a dying camera must not take the backend with it
            failure = f"Gesture capture stopped: {exc}"
        finally:
            self._reset_controls(time.monotonic(), "camera_stopped")
            if hands is not None:
                try:
                    hands.close()
                except Exception as exc:
                    if self.tracker_backend in ("rtmpose", "wilor"):
                        with self._lock:
                            self._failed_tracker = hands
                            self._tracker_error = str(exc)
            if camera is not None:
                try:
                    camera.release()
                except Exception:
                    pass
            with self._lock:
                self._running = False
                self._clear_preview_locked("capture_error" if failure else "camera_stopped", end_lease=True)
                cleanup_error = self._cleanup_error_text()
                if cleanup_error:
                    self._state = "error"
                    self._reason = cleanup_error
                    if failure:
                        self._reason = f"{failure} {self._reason}"
                else:
                    self._state = "off" if self._stop.is_set() or not failure else "error"
                    self._reason = failure if self._state == "error" else "Gestures are off. The camera is released."
                self._report_status()

    @staticmethod
    def _tracker_points(result):
        """Validate an optional tracker's already normalised 21-point output.

        The adapter owns landmark ordering, projection and confidence checks.
        Invalid results mean hand loss; they must never reach input or preview.
        """
        try:
            if len(result) != 21 or any(len(point) != 3 for point in result):
                return []
            points = [tuple(float(value) for value in point) for point in result]
        except (TypeError, ValueError, OverflowError):
            return []
        if not all(math.isfinite(value) for point in points for value in point):
            return []
        if any(not -0.2 <= value <= 1.2 for point in points for value in point[:2]):
            return []
        return points

    def _handle(self, points, now: float) -> None:
        """Interpret one frame without depending on frame rate or swipe speed."""
        with self._input_lock:
            if self._stop.is_set():
                self._reset_controls(now, "camera_stopped")
                return
            if self.input_mode == "mouse":
                self._handle_mouse(points, now)
            else:
                self._handle_desktop(points, now)

    def _handle_desktop(self, points, now: float) -> None:
        if len(points) < 21 or not all(math.isfinite(value) for point in points for value in point):
            if not self._navigation_error and self.measured._pose == self.measured.clutch_pose:
                self._deliver_records(self.measured.update(NONE, 0, 0, 0, now))
            else:
                self._reset_controls(now, "hand_lost")
            return
        if self._navigation_error:
            self._report_navigation_failure(now)
            return
        pose = classify(points)
        if self.measured._anchor is not None:
            if is_pinching(points):
                pose = PINCH
            elif _holds_navigation_pose(points):
                pose = self.measured.clutch_pose
        if pose == NONE:
            pose = UNKNOWN
        wrist = points[WRIST]
        records = self.measured.update(pose, wrist[0], wrist[1], hand_scale(points), now)
        self._deliver_records(records, points)

    def _handle_mouse(self, points, now: float) -> None:
        if len(points) < 21 or not all(math.isfinite(value) for point in points for value in point):
            points = []
            with self._lock:
                self._diagnostic["last_reason"] = "hand_lost"
            # Ordinary detection loss belongs to the controller: it releases
            # buttons immediately, but can preserve movement through a brief
            # dropout. A latched input fault still needs an explicit reset.
            if self._navigation_active and not self._navigation_error:
                self._handle_mouse_navigation([], now)
                return
            if self._navigation_error or self._mouse is None or self._mouse_error:
                self._reset_controls(now, "hand_lost")
                return
        if self._navigation_error:
            self._report_navigation_failure(now)
            return
        if self._navigation_active:
            self._handle_mouse_navigation(points, now)
            return
        self._mouse_pose = classify(points)
        if points and self._mouse_pose == NONE:
            self._mouse_pose = UNKNOWN
        if self._mouse_error:
            self._report_mouse_progress({"state": "error", "progress": 0.0,
                                         "hint": self._mouse_error}, now)
            return
        if is_navigation_pose(points):
            self._handle_mouse_navigation(points, now)
            return
        try:
            if self._mouse is None:
                if self._mouse_factory is None:
                    from core.hand_mouse import HandMouseController
                    self._mouse = HandMouseController(bend_click=self.bend_click)
                else:
                    self._mouse = self._mouse_factory()
                    diagnostic = self._mouse.set_bend_click(self.bend_click)
                    if isinstance(diagnostic, dict) and diagnostic.get("error"):
                        raise RuntimeError(str(diagnostic["error"]))
                self._last_mouse_click_id = 0
            diagnostic = self._mouse.update(points, now)
            if isinstance(diagnostic, dict) and diagnostic.get("error"):
                self._consume_mouse_click(diagnostic, now, emit=False)
                raise RuntimeError(str(diagnostic["error"]))
            self._consume_mouse_click(diagnostic, now, emit=bool(points))
        except Exception as exc:
            message = f"Hand mouse stopped: {exc}. Lower your hand to retry."
            self._reset_controls(now, "mouse_input_error")
            self._mouse_error = message
            diagnostic = {"state": "error", "progress": 0.0, "hint": message}
        self._report_mouse_progress(diagnostic, now)

    def _handle_mouse_navigation(self, points, now: float) -> None:
        """Four fingers temporarily own the hand; no mouse or window actions mix in."""
        if not self._navigation_active:
            # Even a pending pinch must be cancelled before native navigation
            # starts. Releasing four fingers later requires fresh pointing.
            self._reset_controls(now, "navigation_started")
            if self._mouse_error:
                return
            self._navigation_active = True
        if not points:
            self._mouse_pose = NONE
            records = self._navigation.update(NONE, 0, 0, 0, now)
        else:
            maintains = self._navigation._anchor is not None and _holds_navigation_pose(points)
            if self._navigation._anchor is not None and is_pinching(points):
                pose = PINCH
            else:
                pose = FOUR_FINGER if is_navigation_pose(points) or maintains else classify(points)
            self._mouse_pose = pose if pose != NONE else UNKNOWN
            wrist = points[WRIST]
            records = self._navigation.update(self._mouse_pose, wrist[0], wrist[1],
                                               hand_scale(points), now)
        self._deliver_records(records, points, tracker=self._navigation)
        if not self._navigation_error and self._navigation._pose != FOUR_FINGER:
            self._navigation.reset(now, "navigation_ended")
            self._navigation_active = False
            self._report_mouse_progress({"state": "mouse_paused", "progress": 0.0,
                                         "hint": "Point briefly to resume the pointer; four fingers to navigate"}, now)

    def _report_navigation_failure(self, now: float) -> None:
        self._report_progress({"kind": "gesture_progress", "state": "error", "progress": 0.0,
                               "input_mode": self.input_mode, "gesture": "hand_navigation", "at": now,
                               "text": f"Navigation input could not release: {self._navigation_error}. Lower your hand or turn the camera off to retry."})

    def _consume_mouse_click(self, diagnostic, now: float, *, emit=True) -> None:
        """Consume an actual controller click once, independently of HUD progress.

        The controller only issues a click ID after both native button events
        succeed. Keeping its watermark across resets prevents stale diagnostic
        values from becoming new sounds after reconnecting or rearming.
        """
        if not isinstance(diagnostic, dict):
            return
        click_id = diagnostic.get("click_id")
        if type(click_id) is not int or click_id <= self._last_mouse_click_id:
            return
        self._last_mouse_click_id = click_id
        source = diagnostic.get("click_source")
        if not emit or diagnostic.get("error") or source not in ("bend", "pinch"):
            return
        self._click_sequence += 1
        event = {"type": "gesture_click", "id": f"{self._click_session}:{self._click_sequence}",
                 "source": source, "at": now}
        if self.on_click:
            try:
                self.on_click(event)
            except Exception:
                pass  # A feedback failure must not repeat or undo an OS click.

    def _report_mouse_progress(self, diagnostic, now: float) -> None:
        diagnostic = diagnostic if isinstance(diagnostic, dict) else {}
        progress = diagnostic.get("progress", 0.0)
        if not isinstance(progress, (int, float)) or not math.isfinite(progress):
            progress = 0.0
        payload = {"kind": "gesture_progress", "input_mode": "mouse",
                   "gesture": "hand_mouse", "state": diagnostic.get("state", "idle"),
                   "progress": max(0.0, min(1.0, progress)),
                   "text": diagnostic.get("hint", "Show your index finger to move the pointer"),
                   "at": now}
        region = diagnostic.get("control_region")
        if (isinstance(region, (list, tuple)) and len(region) == 4
                and all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 for value in region)
                and region[0] < region[2] and region[1] < region[3]):
            payload["control_region"] = list(region)
        self._report_progress(payload)

    def _reset_controls(self, now: float, reason: str) -> None:
        with self._input_lock:
            with self._lock:
                self._diagnostic["last_reason"] = reason
            if self._navigation_error:
                tracker = self._navigation if self.input_mode == "mouse" else self.measured
                self._deliver_records([{"kind": "motion", "phase": "cancel", "progress": 0.0,
                                        "axis": self._navigation_axis, "at": now, "reason": reason}],
                                      tracker=tracker)
                if not self._navigation_error:
                    self._navigation.reset(now, reason)
                    self.measured.reset(now, reason)
                    self._navigation_active = False
                    self._navigation_axis = None
            elif self._navigation_active:
                self._deliver_records(self._navigation.reset(now, reason), tracker=self._navigation)
                if not self._navigation_error:
                    self._navigation_active = False
            if self.input_mode == "mouse":
                self._mouse_pose = NONE
                self._mouse_error = None
                diagnostic = {"state": "idle", "progress": 0.0,
                              "hint": "Show your index finger to move the pointer"}
                if self._mouse is not None:
                    try:
                        diagnostic = self._mouse.reset(reason)
                        self._consume_mouse_click(diagnostic, now, emit=False)
                        if isinstance(diagnostic, dict) and diagnostic.get("error"):
                            self._mouse_error = str(diagnostic["error"])
                            diagnostic = {**diagnostic, "state": "error", "progress": 0.0,
                                          "hint": diagnostic.get("hint") or f"Hand mouse could not reset: {self._mouse_error}"}
                    except Exception as exc:
                        self._mouse_error = str(exc)
                        diagnostic = {"state": "error", "progress": 0.0,
                                      "hint": f"Hand mouse could not reset: {exc}"}
                self._report_mouse_progress(diagnostic, now)
            else:
                self._deliver_records(self.measured.reset(now, reason))
            if self._navigation_error:
                self._report_navigation_failure(now)

    def _deliver_records(self, records, points=None, *, tracker=None) -> None:
        tracker = self.measured if tracker is None else tracker
        for record in records:
            if record["kind"] == "action":
                if tracker.pose_actions:
                    self._emit(record["name"], points, record["at"])
            elif record["kind"] == "motion":
                if record.get("axis") in ("horizontal", "vertical"):
                    self._navigation_axis = record["axis"]
                if record.get("reason"):
                    with self._lock:
                        self._diagnostic["last_reason"] = record["reason"]
                if not self.on_motion:
                    continue
                try:
                    result = self.on_motion({key: value for key, value in record.items() if key != "kind"})
                    if isinstance(result, dict) and result.get("error"):
                        raise RuntimeError(str(result["error"]))
                    if record["phase"] in ("cancel", "end"):
                        self._navigation_error = None
                        self._navigation_axis = None
                except Exception as exc:
                    # Callback code owns native contacts. Give it an explicit
                    # cleanup opportunity, then require a pose release before
                    # this tracker can begin another desktop session.
                    tracker.reset(record["at"], "motion_callback_failed")
                    cleanup_error = str(exc) if record["phase"] == "cancel" else None
                    if record["phase"] != "cancel":
                        tracker._pose = tracker.clutch_pose
                        tracker._fired = True
                        try:
                            cleanup = self.on_motion({"phase": "cancel", "progress": 0.0,
                                                      "axis": record.get("axis") or self._navigation_axis,
                                                      "at": record["at"], "reason": "motion_callback_failed"})
                            if isinstance(cleanup, dict) and cleanup.get("error"):
                                raise RuntimeError(str(cleanup["error"]))
                            self._navigation_error = None
                            self._navigation_axis = None
                        except Exception as cleanup_exc:
                            cleanup_error = str(cleanup_exc)
                    if cleanup_error and self._navigation_axis is not None:
                        self._navigation_error = cleanup_error
                        if tracker is self._navigation:
                            self._navigation_active = True
                        self._report_navigation_failure(record["at"])
                        return
                    self._report_progress({"kind": "gesture_progress", "state": "cancelled",
                                           "progress": 0.0, "at": record["at"],
                                           "text": "Desktop control stopped; lower your hand to reset"})
                    return
            elif record["kind"] == "gesture_progress":
                if tracker is self._navigation:
                    record = {**record, "input_mode": "mouse", "gesture": "hand_navigation"}
                self._report_progress(record)

    def _report_progress(self, progress) -> None:
        with self._lock:
            self._diagnostic.update(state=progress["state"], progress=progress["progress"],
                                    hint=progress["text"])
            if progress.get("input_mode") == "mouse" and "control_region" in progress:
                self._diagnostic["control_region"] = list(progress["control_region"])
            else:
                self._diagnostic.pop("control_region", None)
        if not self.on_progress:
            return
        previous = self._last_progress
        now = progress["at"]
        identity = ("state", "gesture", "direction")
        changed = previous is None or any(progress.get(key) != previous.get(key) for key in identity)
        same = previous is not None and all(value == previous.get(key) for key, value in progress.items() if key != "at")
        if same or (not changed and now - self._last_progress_at < 1.0 / 15.0):
            return
        self._last_progress, self._last_progress_at = dict(progress), now
        try:
            self.on_progress(progress)
        except Exception:
            pass

    def _emit(self, name: str, points, now: float) -> None:
        event = GestureEvent(name=name, at=now,
                             position=(points[WRIST][0], points[WRIST][1]),
                             scale=hand_scale(points))
        if self.on_gesture:
            try:
                self.on_gesture(event)
            except Exception:
                pass

    def _report_status(self) -> None:
        if self.on_status:
            try:
                self.on_status(self.status())
            except Exception:
                pass

    # ── Dispatch ─────────────────────────────────────────────────────────

    def action_for(self, name: str):
        """The capability a gesture is bound to, or None if it is unbound."""
        return self.bindings.get(name)
