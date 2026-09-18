"""Hand gestures as an input actor.

A camera is the least deliberate input this system takes. There is no keystroke
behind a gesture, the user may have been waving at someone else in the room, and
a frame or two of noise can look like a swipe. Three things follow from that, and
they are the whole design:

  * **Classification is pure.** Turning 21 landmarks into "fist" or "pinch" is a
    function of the landmarks and nothing else, so every gesture in the vocabulary
    is tested against fixed coordinates with no camera involved. Only the capture
    loop needs hardware.
  * **A gesture must persist to count.** One frame is noise. A pose has to hold
    for several consecutive frames before it becomes an event, and then the same
    event cannot fire again until a cooldown has passed. Without that, a held
    hand fires continuously — the same failure the tool loop had when it opened
    one browser tab per iteration.
  * **The gate decides, not this module.** Events dispatch as ``actor="gesture"``,
    which the capability gate confines to reversible actions. Nothing here can
    reach an irreversible capability even if the classifier is badly wrong.

MediaPipe and OpenCV are optional. Import failure is reported as unavailability
with a readable reason, exactly as voice does, rather than taking the backend down.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

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
THUMBS_UP = "thumbs_up"
NONE = "none"

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


# ── What each gesture does ───────────────────────────────────────────────────
#
# Every entry is a capability the gate classifies free or reversible. An
# irreversible one would be refused at dispatch, so the binding table cannot
# quietly become a way around the gate.

DEFAULT_BINDINGS: dict = {
    SWIPE_LEFT:   ("os_snap_window", {"position": "left"}),
    SWIPE_RIGHT:  ("os_snap_window", {"position": "right"}),
    SWIPE_UP:     ("os_window_state", {"state": "maximize"}),
    SWIPE_DOWN:   ("os_window_state", {"state": "minimize"}),
    FIST:         ("os_media_key", {"key": "play_pause"}),
    TWO_FINGER:   ("os_cycle_window", {"direction": "next"}),
    THUMBS_UP:    ("os_media_key", {"key": "next_track"}),
}


class GestureRecognizer:
    """Owns the camera, classifies frames, and reports deliberate gestures.

    Optional dependencies are resolved at `prepare()` rather than import, so a
    machine with no camera or no mediapipe reports why instead of failing to
    start the backend.
    """

    def __init__(self, on_gesture: Optional[Callable] = None, camera_index: int = 0,
                 bindings: Optional[dict] = None, hold_frames: int = 4,
                 cooldown_s: float = 0.8, on_status: Optional[Callable] = None):
        self.on_gesture = on_gesture
        self.on_status = on_status
        self.camera_index = camera_index
        self.bindings = dict(DEFAULT_BINDINGS if bindings is None else bindings)
        self.stabiliser = GestureStabiliser(hold_frames, cooldown_s)
        self.motion = MotionTracker()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._running = False
        self._reason = "Gestures have not been started."
        self._lock = threading.Lock()

    # ── Availability ─────────────────────────────────────────────────────

    def probe(self) -> dict:
        """Whether this machine can do gestures at all, and why not if it cannot."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            return {"available": False,
                    "text": "opencv-python is not installed — run: pip install opencv-python"}
        try:
            import mediapipe  # noqa: F401
        except ImportError:
            return {"available": False,
                    "text": "mediapipe is not installed — run: pip install mediapipe"}
        return {"available": True, "text": "Gesture recognition is available."}

    def status(self) -> dict:
        probe = self.probe()
        return {"available": probe["available"], "running": self._running,
                "text": self._reason if not self._running else "Watching for gestures.",
                "bindings": {k: v[0] for k, v in self.bindings.items()}}

    def is_running(self) -> bool:
        return self._running

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> dict:
        with self._lock:
            if self._running:
                return {"ok": True, "already": True}
            probe = self.probe()
            if not probe["available"]:
                self._reason = probe["text"]
                return {"error": probe["text"]}
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="gestures", daemon=True)
            self._thread.start()
            return {"ok": True}

    def stop(self) -> dict:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        self._running = False
        self._reason = "Gestures are off."
        return {"ok": True}

    # ── The capture loop ─────────────────────────────────────────────────

    def _run(self) -> None:
        import cv2
        import mediapipe as mp

        camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera = cv2.VideoCapture(self.camera_index)
        if not camera.isOpened():
            self._running = False
            self._reason = (f"No camera at index {self.camera_index}. Check it is "
                            "connected and not in use by another application.")
            self._report_status()
            return

        # A smaller frame is enough for landmarks and keeps a webcam loop off the
        # CPU budget the rest of the system needs.
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self._running = True
        self._reason = "Watching for gestures."
        self._report_status()

        hands = mp.solutions.hands.Hands(
            max_num_hands=1, model_complexity=0,
            min_detection_confidence=0.6, min_tracking_confidence=0.5,
        )
        try:
            while not self._stop.is_set():
                ok, frame = camera.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                frame = cv2.flip(frame, 1)  # mirror, so the user's left is left
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                now = time.monotonic()

                if not result.multi_hand_landmarks:
                    self.stabiliser.feed(NONE, now)
                    self.motion.reset()
                    continue

                points = [(p.x, p.y, p.z) for p in result.multi_hand_landmarks[0].landmark]
                self._handle(points, now)
        except Exception as exc:  # a dying camera must not take the backend with it
            self._reason = f"Gesture capture stopped: {exc}"
        finally:
            try:
                hands.close()
            except Exception:
                pass
            camera.release()
            self._running = False
            self._report_status()

    def _handle(self, points, now: float) -> None:
        """One frame's landmarks: emit a swipe, a pose, or nothing."""
        pose = classify(points)
        wrist = points[WRIST]

        # Motion is tracked only while the hand is open or pointing. A pinch or
        # fist that happens to travel is a grab, not a swipe, and tracking it
        # would fire both.
        if pose in (OPEN_PALM, POINT):
            swipe = self.motion.update(wrist[0], wrist[1], now)
            if swipe and self.stabiliser.allow(swipe, now):
                self._emit(swipe, points, now)
                return
        else:
            self.motion.reset()

        settled = self.stabiliser.feed(pose, now)
        if settled and settled != NONE:
            self._emit(settled, points, now)

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
