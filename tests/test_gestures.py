"""Gesture recognition, tested without a camera.

Classification is a pure function of 21 landmarks, so the whole vocabulary can
be checked against fixed coordinates. What needs the most care is not whether a
fist is recognised but whether a *held* fist fires once or forty times a second,
and whether a camera's opinion can reach something irreversible.
"""

import pytest

from core import gestures as g
from core.capabilities import capabilities, gate


# ── Building hands ──────────────────────────────────────────────────────────
#
# Landmarks are normalised to the frame, origin top-left, y increasing downward.
# A hand is built upright with the wrist low and fingers reaching up, which is
# how one is held in front of a webcam.


def _hand(index=True, middle=True, ring=True, pinky=True, thumb=True,
          pinch=False, wrist=(0.5, 0.8)):
    """21 landmarks with each finger either extended or curled."""
    wx, wy = wrist
    points = [(0.0, 0.0, 0.0)] * 21
    points[g.WRIST] = (wx, wy, 0.0)
    points[g.MIDDLE_MCP] = (wx, wy - 0.20, 0.0)      # sets the hand scale
    points[g.INDEX_MCP] = (wx - 0.05, wy - 0.18, 0.0)
    points[g.PINKY_MCP] = (wx + 0.08, wy - 0.16, 0.0)

    def finger(tip_i, pip_i, offset, extended):
        pip_y = wy - 0.26
        tip_y = wy - 0.42 if extended else wy - 0.14
        points[pip_i] = (wx + offset, pip_y, 0.0)
        points[tip_i] = (wx + offset, tip_y, 0.0)

    finger(g.INDEX_TIP, g.INDEX_PIP, -0.05, index)
    finger(g.MIDDLE_TIP, g.MIDDLE_PIP, 0.0, middle)
    finger(g.RING_TIP, g.RING_PIP, 0.05, ring)
    finger(g.PINKY_TIP, g.PINKY_PIP, 0.09, pinky)

    if pinch:
        # Thumb tip meets the index tip.
        points[g.THUMB_TIP] = points[g.INDEX_TIP]
    elif thumb:
        points[g.THUMB_TIP] = (wx - 0.22, wy - 0.16, 0.0)
    else:
        points[g.THUMB_TIP] = (wx + 0.05, wy - 0.10, 0.0)
    points[g.THUMB_IP] = (wx - 0.12, wy - 0.10, 0.0)
    return points


# ── The vocabulary ──────────────────────────────────────────────────────────


def test_an_open_palm_is_recognised():
    assert g.classify(_hand()) == g.OPEN_PALM


def test_a_fist_is_recognised():
    assert g.classify(_hand(False, False, False, False, thumb=False)) == g.FIST


def test_a_thumbs_up_is_a_closed_hand_with_the_thumb_out():
    assert g.classify(_hand(False, False, False, False, thumb=True)) == g.THUMBS_UP


def test_pointing_is_recognised():
    assert g.classify(_hand(True, False, False, False, thumb=False)) == g.POINT


def test_two_fingers_are_recognised():
    assert g.classify(_hand(True, True, False, False, thumb=False)) == g.TWO_FINGER


def test_a_pinch_beats_the_finger_count():
    """Thumb and index together is a pinch even though index reads extended."""
    assert g.classify(_hand(True, False, False, False, pinch=True)) == g.PINCH


def test_nothing_is_claimed_from_missing_landmarks():
    assert g.classify([]) == g.NONE
    assert g.classify([(0.0, 0.0, 0.0)] * 5) == g.NONE


def test_recognition_does_not_depend_on_distance_from_the_camera():
    """Thresholds are relative to hand size, so a hand further away still reads.

    Absolute thresholds would make every gesture work at one distance only.
    """
    near = _hand(wrist=(0.5, 0.9))
    far = [(0.5 + (x - 0.5) * 0.4, 0.5 + (y - 0.5) * 0.4, z) for x, y, z in near]
    assert g.classify(near) == g.OPEN_PALM
    assert g.classify(far) == g.OPEN_PALM


# ── Swipes ──────────────────────────────────────────────────────────────────


def test_a_horizontal_sweep_is_a_swipe():
    tracker = g.MotionTracker()
    assert tracker.update(0.2, 0.5, 0.00) is None
    assert tracker.update(0.4, 0.5, 0.05) is None
    assert tracker.update(0.6, 0.51, 0.10) == g.SWIPE_RIGHT


def test_a_swipe_the_other_way():
    tracker = g.MotionTracker()
    tracker.update(0.8, 0.5, 0.00)
    tracker.update(0.6, 0.5, 0.05)
    assert tracker.update(0.4, 0.5, 0.10) == g.SWIPE_LEFT


def test_vertical_sweeps_are_recognised():
    up = g.MotionTracker()
    up.update(0.5, 0.8, 0.0); up.update(0.5, 0.6, 0.05)
    assert up.update(0.5, 0.4, 0.10) == g.SWIPE_UP

    down = g.MotionTracker()
    down.update(0.5, 0.2, 0.0); down.update(0.5, 0.4, 0.05)
    assert down.update(0.5, 0.6, 0.10) == g.SWIPE_DOWN


def test_a_slow_drift_is_not_a_swipe():
    """Repositioning a hand is not a command."""
    tracker = g.MotionTracker()
    result = None
    for i in range(12):
        result = tracker.update(0.2 + i * 0.05, 0.5, i * 0.5) or result
    assert result is None


def test_a_diagonal_movement_does_not_fire_two_directions():
    tracker = g.MotionTracker()
    tracker.update(0.2, 0.2, 0.00)
    tracker.update(0.35, 0.35, 0.05)
    assert tracker.update(0.5, 0.5, 0.10) is None


def test_a_twitch_is_not_a_swipe():
    tracker = g.MotionTracker()
    tracker.update(0.50, 0.5, 0.00)
    tracker.update(0.53, 0.5, 0.02)
    assert tracker.update(0.56, 0.5, 0.04) is None


# ── Holding still must not repeat ───────────────────────────────────────────


def test_a_pose_must_persist_before_it_counts():
    """One frame is noise, not an instruction."""
    s = g.GestureStabiliser(hold_frames=4, cooldown_s=0.5)
    assert s.feed(g.FIST, 0.00) is None
    assert s.feed(g.FIST, 0.03) is None
    assert s.feed(g.FIST, 0.06) is None
    assert s.feed(g.FIST, 0.09) == g.FIST


def test_a_flickering_pose_never_fires():
    s = g.GestureStabiliser(hold_frames=4, cooldown_s=0.5)
    for i in range(20):
        pose = g.FIST if i % 2 else g.OPEN_PALM
        assert s.feed(pose, i * 0.03) is None


def test_a_held_gesture_fires_once_not_every_frame():
    """The failure that matters. A hand resting in frame at 30fps would
    otherwise dispatch the same action thirty times a second."""
    s = g.GestureStabiliser(hold_frames=3, cooldown_s=1.0)
    fired = [s.feed(g.FIST, i * 0.03) for i in range(40)]
    assert [f for f in fired if f] == [g.FIST], "a held pose fired more than once"


def test_the_same_gesture_fires_again_after_the_cooldown():
    s = g.GestureStabiliser(hold_frames=2, cooldown_s=0.5)
    s.feed(g.FIST, 0.0); assert s.feed(g.FIST, 0.03) == g.FIST
    s.feed(g.NONE, 0.1)
    s.feed(g.FIST, 0.2); assert s.feed(g.FIST, 0.23) is None, "still cooling down"
    s.feed(g.NONE, 0.5)
    s.feed(g.FIST, 0.9); assert s.feed(g.FIST, 0.93) == g.FIST


def test_an_empty_frame_never_fires():
    s = g.GestureStabiliser(hold_frames=2, cooldown_s=0.1)
    assert [s.feed(g.NONE, i * 0.03) for i in range(10)] == [None] * 10


# ── What a gesture is allowed to do ─────────────────────────────────────────


def test_every_bound_gesture_maps_to_a_real_capability():
    """A binding naming a capability that does not exist is a dead gesture."""
    from core.actions import register_os_capabilities
    register_os_capabilities()

    for gesture, (name, _args) in g.DEFAULT_BINDINGS.items():
        assert capabilities.get(name) is not None, f"{gesture} -> unknown {name}"


def test_no_gesture_is_bound_to_something_irreversible():
    """The binding table must not become a way around the gate."""
    from core.actions import register_os_capabilities
    register_os_capabilities()

    for gesture, (name, args) in g.DEFAULT_BINDINGS.items():
        cap = capabilities.get(name)
        assert cap.reversibility != "irreversible", f"{gesture} -> {name}"
        decision = gate(cap, dict(args), actor="gesture", confidence=1.0)
        assert decision.verdict == "allow", f"{gesture} -> {name}: {decision.reason}"


@pytest.mark.parametrize("name,args", [
    ("os_shutdown_computer", {"delay_sec": 30}),
    ("os_restart_computer", {"delay_sec": 30}),
    ("os_kill_process", {"name": "notepad.exe"}),
])
def test_a_camera_can_never_reach_an_irreversible_action(name, args):
    """Even a badly wrong classifier must not be able to shut the machine down."""
    from core.actions import register_os_capabilities
    register_os_capabilities()

    cap = capabilities.get(name)
    if cap is None:
        pytest.skip(f"{name} is not registered on this platform")
    decision = gate(cap, dict(args), actor="gesture", confidence=1.0)
    assert decision.verdict == "deny"
    assert "gesture" in decision.reason


# ── Degrading without hardware ──────────────────────────────────────────────


def test_a_missing_dependency_is_reported_not_raised(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_mediapipe(name, *args, **kwargs):
        if name == "mediapipe":
            raise ImportError("no mediapipe here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_mediapipe)
    recognizer = g.GestureRecognizer()
    probe = recognizer.probe()
    assert probe["available"] is False
    assert "mediapipe" in probe["text"]
    assert "error" in recognizer.start()


def test_stopping_something_never_started_is_harmless():
    assert g.GestureRecognizer().stop()["ok"]
