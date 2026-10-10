"""Gesture recognition, tested without a camera.

Classification is a pure function of 21 landmarks, so the whole vocabulary can
be checked against fixed coordinates. What needs the most care is not whether a
fist is recognised but whether a *held* fist fires once or forty times a second,
and whether a camera's opinion can reach something irreversible.
"""

import math

import pytest
import sys
from types import SimpleNamespace

from core import gestures as g
from core.capabilities import capabilities, gate
from core.hand_tracking import TrackerStartupError


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


# ── Measured controls ──────────────────────────────────────────────────────


def _arm(tracker, x=0.5, y=0.5, scale=0.1, at=0.0):
    for elapsed in (0.0, 0.15, 0.35):
        records = tracker.update(tracker.clutch_pose, x, y, scale, at + elapsed)
    assert records[-1]["state"] == "armed"


def _of_kind(records, kind):
    return [record for record in records if record["kind"] == kind]


def _neutral(tracker, at=2.0):
    tracker.update(g.NONE, 0, 0, 0, at)
    tracker.update(g.NONE, 0, 0, 0, at + 0.21)
    assert tracker._pose == g.NONE


def test_default_navigation_distance_and_arm_are_deliberate():
    tracker = g.MeasuredGestureTracker()
    assert tracker.travel_palms == 1.2
    assert tracker.update(g.OPEN_PALM, 0.5, 0.5, 0.1, 0)[-1]["state"] == "arming"
    assert tracker.update(g.OPEN_PALM, 0.5, 0.5, 0.1, 0.29)[-1]["state"] == "arming"
    assert tracker.update(g.OPEN_PALM, 0.5, 0.5, 0.1, 0.30)[-1]["state"] == "armed"
    assert not _of_kind(tracker.update(g.OPEN_PALM, 0.52, 0.52, 0.1, 0.35), "motion")


@pytest.mark.parametrize("clutch", [g.OPEN_PALM, g.FOUR_FINGER])
@pytest.mark.parametrize("first_at, resumed_at", [(0.0, 0.5), (0.5, 0.4)])
def test_navigation_candidate_does_not_accumulate_hold_across_missing_or_backwards_frames(clutch, first_at, resumed_at):
    tracker = g.MeasuredGestureTracker(clutch_pose=clutch)
    tracker.update(clutch, 0.5, 0.5, 0.1, first_at)
    records = tracker.update(clutch, 0.5, 0.5, 0.1, resumed_at)
    assert records[-1]["state"] == "arming" and records[-1]["progress"] == 0
    assert not _of_kind(records, "motion")
    assert tracker.update(clutch, 0.5, 0.5, 0.1, resumed_at + 0.15)[-1]["state"] == "arming"
    assert tracker.update(clutch, 0.5, 0.5, 0.1, resumed_at + 0.30)[-1]["state"] == "armed"


def test_moving_candidate_cannot_arm_and_never_owns_native_contacts():
    tracker = g.MeasuredGestureTracker()
    records = []
    for i in range(100):
        records += tracker.update(g.OPEN_PALM, 0.5 + 0.06 * math.sin(i * 0.3), 0.5, 0.1, i / 30)
    assert not _of_kind(records, "motion")
    assert not _of_kind(records, "action")


@pytest.mark.parametrize("fps", [15, 30, 60])
@pytest.mark.parametrize("axis, sign", [("horizontal", -1), ("horizontal", 1),
                                       ("vertical", -1), ("vertical", 1)])
def test_realistic_swipe_traces_complete_once_with_jitter_and_remain_latched(fps, axis, sign):
    tracker = g.MeasuredGestureTracker()
    _arm(tracker)
    records = []
    for frame in range(1, fps * 3 + 1):
        now = 0.35 + frame / fps
        travel = min(1.02, max(0.0, (now - 0.4) / 0.7))
        noise = (0.005, -0.007, 0.009, -0.005)[frame % 4]
        distance = sign * (travel + noise) * 0.12
        x, y = (0.5 + distance, 0.5) if axis == "horizontal" else (0.5, 0.5 + distance)
        records += tracker.update(g.OPEN_PALM, x, y, 0.1, now)
    motions = _of_kind(records, "motion")
    assert motions[0]["phase"] == "begin"
    ends = [record for record in motions if record["phase"] == "end"]
    assert len(ends) == 1
    assert ends[0]["automatic"] is True and ends[0]["progress"] == sign
    assert all(record["axis"] == axis for record in motions)
    assert not any(record["phase"] == "cancel" for record in motions)
    assert not _of_kind(records, "action")
    assert [record for record in records if record.get("state") == "completed"]
    assert tracker.update(g.OPEN_PALM, 0.5, 0.5, 0.1, 10.0) == []
    tracker.update(g.FIST, 0.5, 0.5, 0.1, 10.1)
    assert tracker.update(g.OPEN_PALM, 0.5, 0.5, 0.1, 10.25) == []  # brief neutral flicker cannot rearm
    _neutral(tracker, 10.3)
    assert tracker.update(g.OPEN_PALM, 0.5, 0.5, 0.1, 10.55)[-1]["state"] == "arming"


@pytest.mark.parametrize("fps", [15, 30, 60])
def test_resting_landmark_jitter_does_not_start_navigation(fps):
    tracker = g.MeasuredGestureTracker()
    _arm(tracker)
    for frame in range(fps * 2):
        records = tracker.update(g.OPEN_PALM, 0.5 + (-1) ** frame * 0.002,
                                 0.5 + (-1) ** (frame // 2) * 0.002, 0.1, 0.36 + frame / fps)
        assert not _of_kind(records, "motion")


@pytest.mark.parametrize("axis", ["horizontal", "vertical"])
def test_axis_lock_and_reversal_use_final_position_not_peak(axis):
    tracker = g.MeasuredGestureTracker()
    _arm(tracker)
    def update(distance, other, now):
        x, y = (0.5 + distance, 0.5 + other) if axis == "horizontal" else (0.5 + other, 0.5 + distance)
        return tracker.update(g.OPEN_PALM, x, y, 0.1, now)
    assert _of_kind(update(0.04, 0, 0.4), "motion")[-1]["axis"] == axis
    assert update(0.12, 0.01, 0.5)[-1]["state"] == "committing"
    reverse = update(0.08, 0.08, 0.54)
    assert reverse[-1]["state"] in ("desktop", "overview")
    assert _of_kind(reverse, "motion")[-1]["axis"] == axis
    assert not any(record["phase"] == "end" for record in _of_kind(update(0.0, 0.1, 0.6), "motion"))
    tracker.update(g.FIST, 0.5, 0.5, 0.1, 0.65)
    cancelled = tracker.update(g.FIST, 0.5, 0.5, 0.1, 0.78)
    assert _of_kind(cancelled, "motion")[-1]["phase"] == "cancel"


def test_completion_hysteresis_tolerates_small_jitter_but_resets_below_ninety_percent():
    tracker = g.MeasuredGestureTracker()
    _arm(tracker)
    tracker.update(g.OPEN_PALM, 0.62, 0.5, 0.1, 0.5)
    assert tracker.update(g.OPEN_PALM, 0.6128, 0.5, 0.1, 0.54)[-1]["state"] == "committing"
    assert tracker.update(g.OPEN_PALM, 0.6056, 0.5, 0.1, 0.58)[-1]["state"] == "desktop"
    assert not any(r["phase"] == "end" for r in _of_kind(tracker.update(g.OPEN_PALM, 0.618, 0.5, 0.1, 0.7), "motion"))
    tracker.update(g.OPEN_PALM, 0.62, 0.5, 0.1, 0.75)
    tracker.update(g.OPEN_PALM, 0.612, 0.5, 0.1, 0.8)
    records = tracker.update(g.OPEN_PALM, 0.615, 0.5, 0.1, 0.85)
    assert records[-1]["state"] == "completed"
    assert _of_kind(records, "motion")[-1]["progress"] == 1.0


def test_one_delayed_trusted_frame_cannot_supply_the_entire_completion_dwell():
    tracker = g.MeasuredGestureTracker()
    _arm(tracker)
    tracker.update(g.OPEN_PALM, 0.62, 0.5, 0.1, 0.5)
    records = tracker.update(g.OPEN_PALM, 0.62, 0.5, 0.1, 0.7)
    assert records[-1]["state"] == "committing"
    assert not any(r["phase"] == "end" for r in _of_kind(records, "motion"))


@pytest.mark.parametrize("scale", [0.08, 0.1, 0.2])
def test_travel_is_measured_in_palm_lengths(scale):
    tracker = g.MeasuredGestureTracker()
    _arm(tracker, scale=scale)
    records = tracker.update(g.OPEN_PALM, 0.5 + scale * 0.6, 0.5, scale, 0.5)
    assert _of_kind(records, "motion")[-1]["progress"] == pytest.approx(0.5)


def test_scale_is_fixed_when_armed_and_travel_is_configurable():
    tracker = g.MeasuredGestureTracker(travel_palms=2.0)
    _arm(tracker, scale=0.1)
    records = tracker.update(g.OPEN_PALM, 0.6, 0.5, 0.13, 0.5)
    assert _of_kind(records, "motion")[-1]["progress"] == pytest.approx(0.5)


@pytest.mark.parametrize("pose", [g.NONE, g.UNKNOWN])
@pytest.mark.parametrize("progress", [0.5, 1.0])
def test_brief_uncertainty_heartbeats_freezes_and_reanchors_without_completing(pose, progress):
    tracker = g.MeasuredGestureTracker()
    _arm(tracker)
    x = 0.5 + progress * 0.12
    tracker.update(g.OPEN_PALM, x, 0.5, 0.1, 0.5)
    frozen = tracker.update(pose, x, 0.5, 0.1, 0.54)
    heartbeat = _of_kind(frozen, "motion")[-1]
    assert heartbeat["phase"] == "update" and heartbeat["frozen"] is True
    assert heartbeat["progress"] == pytest.approx(progress)
    assert frozen[-1]["state"] == "uncertain" and frozen[-1]["axis"] == "horizontal"
    recovered = tracker.update(g.OPEN_PALM, x + 0.02, 0.5, 0.1, 0.64)
    assert _of_kind(recovered, "motion")[-1]["progress"] == pytest.approx(progress)
    assert recovered[-1]["commit_progress"] == 0
    records = tracker.update(g.OPEN_PALM, x + 0.02, 0.5, 0.1, 0.68)
    assert not any(record["phase"] == "end" for record in _of_kind(records, "motion"))
    assert tracker._anchor[0] == pytest.approx(0.52)


@pytest.mark.parametrize("x, scale", [(0.66, 0.1), (0.56, 0.14), (0.56, 0.07)])
def test_reappearing_hand_must_match_the_previous_position_and_scale(x, scale):
    tracker = g.MeasuredGestureTracker()
    _arm(tracker)
    tracker.update(g.OPEN_PALM, 0.56, 0.5, 0.1, 0.5)
    tracker.update(g.NONE, 0, 0, 0, 0.54)
    records = tracker.update(g.OPEN_PALM, x, 0.5, scale, 0.64)
    assert _of_kind(records, "motion")[-1]["phase"] == "cancel"
    assert _of_kind(records, "motion")[-1]["reason"] == "tracking_jump"


@pytest.mark.parametrize("pose", [g.NONE, g.UNKNOWN, g.OPEN_PALM])
def test_uncertainty_expiry_cancels_even_when_a_valid_pose_returns_late(pose):
    tracker = g.MeasuredGestureTracker()
    _arm(tracker)
    tracker.update(g.OPEN_PALM, 0.62, 0.5, 0.1, 0.5)
    tracker.update(g.UNKNOWN, 0.62, 0.5, 0.1, 0.54)
    records = tracker.update(pose, 0.62, 0.5, 0.1, 0.70)
    assert _of_kind(records, "motion")[-1]["phase"] == "cancel"
    assert not _of_kind(records, "action")
    assert tracker.update(g.OPEN_PALM, 0.62, 0.5, 0.1, 0.73) == []


@pytest.mark.parametrize("pose", [g.FIST, g.PINCH])
def test_fist_or_pinch_cancels_before_completion_and_never_turns_into_a_click(pose):
    tracker = g.MeasuredGestureTracker()
    _arm(tracker)
    tracker.update(g.OPEN_PALM, 0.62, 0.5, 0.1, 0.5)
    tracker.update(pose, 0.62, 0.5, 0.1, 0.53)
    records = tracker.update(pose, 0.62, 0.5, 0.1, 0.66)
    assert _of_kind(records, "motion")[-1]["phase"] == "cancel"
    assert _of_kind(records, "motion")[-1]["reason"] == "pose_cancelled"
    assert not _of_kind(records, "action")
    assert not _of_kind(tracker.update(pose, 0.62, 0.5, 0.1, 0.75), "action")


def test_single_fist_flicker_does_not_cancel_or_carry_a_release_timer_after_recovery():
    tracker = g.MeasuredGestureTracker()
    _arm(tracker)
    tracker.update(g.OPEN_PALM, 0.56, 0.5, 0.1, 0.5)
    tracker.update(g.FIST, 0.56, 0.5, 0.1, 0.53)
    assert tracker.update(g.OPEN_PALM, 0.56, 0.5, 0.1, 0.58)[-1]["state"] == "desktop"
    tracker.update(g.FIST, 0.56, 0.5, 0.1, 0.63)
    assert _of_kind(tracker.update(g.FIST, 0.56, 0.5, 0.1, 0.71), "motion")[-1]["phase"] == "update"
    assert _of_kind(tracker.update(g.FIST, 0.56, 0.5, 0.1, 0.76), "motion")[-1]["phase"] == "cancel"


@pytest.mark.parametrize("when, x", [(0.81, 0.56), (0.49, 0.56), (0.51, 0.95)])
def test_tracking_gap_clock_reversal_and_jump_cancel_and_require_neutral(when, x):
    tracker = g.MeasuredGestureTracker()
    _arm(tracker)
    tracker.update(g.OPEN_PALM, 0.56, 0.5, 0.1, 0.5)
    records = tracker.update(g.OPEN_PALM, x, 0.5, 0.1, when)
    assert _of_kind(records, "motion")[-1]["phase"] == "cancel"
    assert tracker.update(g.OPEN_PALM, x, 0.5, 0.1, when + 0.1) == []
    _neutral(tracker, when + 0.2)


@pytest.mark.parametrize("moving, reason", [(False, "gesture_idle"), (True, "gesture_timeout")])
def test_navigation_contacts_have_bounded_idle_and_total_lifetimes(moving, reason):
    tracker = g.MeasuredGestureTracker()
    _arm(tracker)
    tracker.update(g.OPEN_PALM, 0.56, 0.5, 0.1, 0.5)
    records = []
    for frame in range(1, 30 * 22):
        now = 0.5 + frame / 30
        x = 0.56 + (0.01 * math.sin(frame * 0.25) if moving else 0)
        batch = tracker.update(g.OPEN_PALM, x, 0.5, 0.1, now)
        records += batch
    cancels = [record for record in _of_kind(records, "motion") if record["phase"] == "cancel"]
    assert len(cancels) == 1 and cancels[0]["reason"] == reason
    assert not any(record["phase"] == "end" for record in _of_kind(records, "motion"))


@pytest.mark.parametrize("pose, action, duration", [
    (g.POINT, g.RESTORE_WINDOW, 0.65), (g.TWO_FINGER, g.TWO_FINGER, 0.65),
    (g.PINCH, g.CLOSE_REQUEST, 1.0), (g.THUMBS_UP, g.CLOSE_CONFIRM, 0.65),
    (g.FIST, g.CLOSE_CANCEL, 0.35),
])
def test_held_desktop_poses_have_distinct_timed_actions_and_do_not_repeat(pose, action, duration):
    tracker = g.MeasuredGestureTracker()
    tracker.update(pose, 0.5, 0.5, 0.1, 0.0)
    assert not _of_kind(tracker.update(pose, 0.5, 0.5, 0.1, duration - 0.001), "action")
    assert _of_kind(tracker.update(pose, 0.5, 0.5, 0.1, duration), "action")[-1]["name"] == action
    for i in range(1, 20):
        assert not _of_kind(tracker.update(pose, 0.5, 0.5, 0.1, duration + i), "action")


def test_transition_from_navigation_to_pinch_requires_neutral_then_a_fresh_close_hold():
    tracker = g.MeasuredGestureTracker()
    _arm(tracker)
    tracker.update(g.OPEN_PALM, 0.56, 0.5, 0.1, 0.5)
    tracker.update(g.PINCH, 0.56, 0.5, 0.1, 0.53)
    tracker.update(g.PINCH, 0.56, 0.5, 0.1, 0.66)
    tracker.update(g.PINCH, 0.56, 0.5, 0.1, 0.74)
    tracker.update(g.PINCH, 0.56, 0.5, 0.1, 0.75)
    assert not _of_kind(tracker.update(g.PINCH, 0.56, 0.5, 0.1, 1.74), "action")
    assert not _of_kind(tracker.update(g.PINCH, 0.56, 0.5, 0.1, 2.0), "action")
    _neutral(tracker, 2.1)
    tracker.update(g.PINCH, 0.56, 0.5, 0.1, 2.4)
    assert not _of_kind(tracker.update(g.PINCH, 0.56, 0.5, 0.1, 3.39), "action")
    assert _of_kind(tracker.update(g.PINCH, 0.56, 0.5, 0.1, 3.4), "action")[-1]["name"] == g.CLOSE_REQUEST


@pytest.mark.parametrize("exit_pose", [g.POINT, g.TWO_FINGER, g.THUMBS_UP, g.PINCH])
@pytest.mark.parametrize("completed", [False, True])
def test_held_window_action_pose_after_navigation_cannot_fall_through_without_a_neutral_release(exit_pose, completed):
    tracker = g.MeasuredGestureTracker()
    _arm(tracker)
    tracker.update(g.OPEN_PALM, 0.62, 0.5, 0.1, 0.5)
    if completed:
        tracker.update(g.OPEN_PALM, 0.62, 0.5, 0.1, 0.55)
        assert tracker.update(g.OPEN_PALM, 0.62, 0.5, 0.1, 0.6)[-1]["state"] == "completed"
    records = []
    for frame in range(100):
        records += tracker.update(exit_pose, 0.62, 0.5, 0.1, 0.61 + frame / 30)
    assert not _of_kind(records, "action")
    assert tracker._pose == g.OPEN_PALM and tracker._fired
    _neutral(tracker, 4.0)
    tracker.update(exit_pose, 0.62, 0.5, 0.1, 4.3)
    duration = tracker.POSE_ACTIONS[exit_pose][1]
    assert _of_kind(tracker.update(exit_pose, 0.62, 0.5, 0.1, 4.3 + duration), "action")


def test_dropout_grace_is_bounded_from_last_trusted_frame_not_first_missing_frame():
    tracker = g.MeasuredGestureTracker()
    _arm(tracker)
    tracker.update(g.OPEN_PALM, 0.56, 0.5, 0.1, 0.5)
    assert tracker.update(g.NONE, 0, 0, 0, 0.6)[-1]["state"] == "uncertain"
    records = tracker.update(g.OPEN_PALM, 0.56, 0.5, 0.1, 0.74)
    assert _of_kind(records, "motion")[-1]["phase"] == "cancel"


@pytest.mark.parametrize("mode", ["desktop", "mouse"])
def test_pinching_with_the_other_fingers_raised_cancels_navigation_instead_of_completing(mode):
    motions, actions = [], []
    recognizer = g.GestureRecognizer(input_mode=mode, on_motion=motions.append,
                                     on_gesture=actions.append)
    for now in (0.0, 0.15, 0.3):
        recognizer._handle(_hand(), now)
    recognizer._handle(_hand(wrist=(0.76, 0.8)), 0.5)
    assert motions[-1]["progress"] == 1.0
    pinched = _hand(pinch=True, wrist=(0.76, 0.8))
    assert g.classify(pinched) == g.OPEN_PALM  # Existing classification alone is insufficient here.
    recognizer._handle(pinched, 0.54)
    recognizer._handle(pinched, 0.66)
    assert motions[-1]["phase"] == "cancel"
    assert not any(record["phase"] == "end" for record in motions)
    for frame in range(30):
        recognizer._handle(pinched, 0.7 + frame / 30)
    assert actions == []


def test_desktop_mode_tolerates_one_curled_finger_only_after_activation():
    motions = []
    recognizer = g.GestureRecognizer(on_motion=motions.append)
    three = _hand(True, True, True, False)
    recognizer._handle(three, 0.0)
    recognizer._handle(three, 0.3)
    assert not motions and recognizer.measured._anchor is None
    recognizer._handle(_hand(), 0.4)
    recognizer._handle(_hand(), 0.7)
    recognizer._handle(_hand(True, True, True, False, wrist=(0.56, 0.8)), 0.8)
    assert motions[-1]["phase"] == "begin"
    assert recognizer.measured._pose == g.OPEN_PALM


def test_recognizer_delivers_motion_and_throttles_progress_but_forces_state_changes():
    motions, progress, actions = [], [], []
    recognizer = g.GestureRecognizer(on_gesture=actions.append,
                                     on_motion=motions.append, on_progress=progress.append)
    recognizer._handle(_hand(wrist=(0.5, 0.8)), 0.0)
    recognizer._handle(_hand(wrist=(0.5, 0.8)), 0.30)
    recognizer._handle(_hand(wrist=(0.56, 0.8)), 0.5)
    for i in range(1, 30):
        recognizer._handle(_hand(wrist=(0.56 + i * 0.001, 0.8)), 0.5 + i * 0.005)
    assert len(motions) == 30
    assert len(progress) <= 6
    assert [record["state"] for record in progress[:3]] == ["arming", "armed", "desktop"]
    recognizer._handle([], 0.646)
    assert motions[-1]["frozen"] is True
    assert progress[-1]["state"] == "uncertain"
    recognizer._handle([], 0.81)
    assert motions[-1]["phase"] == "cancel"
    assert progress[-1]["state"] == "cancelled"
    recognizer._handle([], 0.86)
    assert progress[-1]["state"] == "idle"
    assert actions == []


def test_recognizer_emits_window_restore_and_reserves_close_events_for_backend():
    actions = []
    recognizer = g.GestureRecognizer(on_gesture=actions.append)
    points = _hand(True, False, False, False, thumb=False)
    recognizer._handle(points, 0.0)
    recognizer._handle(points, 0.65)
    assert actions[-1].name == g.RESTORE_WINDOW
    assert recognizer.action_for(g.RESTORE_WINDOW) == ("os_window_state", {"state": "restore"})
    assert recognizer.action_for(g.CLOSE_REQUEST) is None
    assert recognizer.action_for(g.CLOSE_CONFIRM) is None
    assert recognizer.action_for(g.CLOSE_CANCEL) is None
    assert recognizer.action_for(g.SWIPE_LEFT) is None
    assert recognizer.action_for(g.SWIPE_RIGHT) is None


def test_hand_loss_notifies_close_controller_once_even_without_a_desktop_drag():
    motions = []
    recognizer = g.GestureRecognizer(on_motion=motions.append)
    pinch = _hand(True, False, False, False, pinch=True)
    recognizer._handle(pinch, 0.0)
    recognizer._handle(pinch, 1.0)
    recognizer._handle([], 1.1)
    recognizer._handle([], 1.2)
    assert motions == [{"phase": "cancel", "progress": 0.0, "axis": None,
                        "at": 1.1, "reason": "hand_lost"}]


@pytest.mark.parametrize("failure_type", ["exception", "result"])
@pytest.mark.parametrize("failure_phase", ["begin", "update"])
def test_a_motion_callback_failure_cancels_and_requires_release_before_rearming(failure_type, failure_phase):
    motions, progress = [], []

    def failing_motion(record):
        motions.append(record)
        if record["phase"] == failure_phase:
            if failure_type == "exception":
                raise RuntimeError("native motion failed")
            return {"error": "native motion failed"}

    recognizer = g.GestureRecognizer(on_motion=failing_motion, on_progress=progress.append)
    recognizer._handle(_hand(wrist=(0.5, 0.8)), 0.0)
    recognizer._handle(_hand(wrist=(0.5, 0.8)), 0.30)
    recognizer._handle(_hand(wrist=(0.56, 0.8)), 0.5)
    expected_phases = ["begin", "cancel"]
    if failure_phase == "update":
        recognizer._handle(_hand(wrist=(0.57, 0.8)), 0.55)
        expected_phases = ["begin", "update", "cancel"]
    assert [record["phase"] for record in motions] == expected_phases
    assert progress[-1]["state"] == "cancelled"
    recognizer._handle(_hand(wrist=(0.56, 0.8)), 3.0)
    assert [record["phase"] for record in motions] == expected_phases


def test_a_failing_cancel_callback_does_not_repeat_on_every_missing_frame():
    motions = []

    def failing_cancel(record):
        motions.append(record)
        return {"error": "cancellation failed"}

    recognizer = g.GestureRecognizer(on_motion=failing_cancel)
    recognizer._handle(_hand(True, False, False, False, pinch=True), 0.0)
    for i in range(10):
        recognizer._handle([], 0.1 + i * 0.1)
    assert [record["phase"] for record in motions] == ["cancel"]


def test_sensitivity_can_only_change_while_camera_is_off():
    recognizer = g.GestureRecognizer()
    assert recognizer.configure(travel_palms=2.5) == {"ok": True, "travel_palms": 2.5}
    assert recognizer.measured.travel_palms == 2.5
    assert "error" in recognizer.configure(travel_palms=0)
    assert recognizer.measured.travel_palms == 2.5
    recognizer._thread = SimpleNamespace(is_alive=lambda: True)
    assert "error" in recognizer.configure(travel_palms=1.0)
    assert recognizer.measured.travel_palms == 2.5


@pytest.mark.parametrize("value", [True, False, -1, 2, 1.0, "1", None])
def test_model_selection_rejects_values_other_than_integer_zero_or_one(value):
    with pytest.raises(ValueError, match="model_complexity"):
        g.GestureRecognizer(model_complexity=value)


def test_model_defaults_to_full_and_can_be_changed_atomically_while_off(monkeypatch):
    recognizer = g.GestureRecognizer()
    monkeypatch.setattr(recognizer, "probe", lambda: {"available": True})
    assert recognizer.status()["model_complexity"] == 1
    assert recognizer.status()["model_name"] == "MediaPipe Hands full"
    assert recognizer.configure(model_complexity=0)["model"] == "MediaPipe Hands lite"
    assert "error" in recognizer.configure(travel_palms=2.5, model_complexity=True)
    assert recognizer.measured.travel_palms == 1.2
    assert recognizer.model_complexity == 0
    recognizer._thread = SimpleNamespace(is_alive=lambda: True)
    assert "error" in recognizer.configure(model_complexity=1)
    assert recognizer.model_complexity == 0


class _FakeMouseController:
    def __init__(self):
        self.updates = []
        self.resets = []
        self.failure = None
        self.bend_click = False

    def set_bend_click(self, enabled):
        self.bend_click = enabled
        return {"state": "idle"}

    def update(self, points, now):
        self.updates.append((points, now))
        if self.failure == "raise":
            raise RuntimeError("Input unavailable")
        if self.failure == "result":
            return {"state": "error", "error": "Input unavailable"}
        return {"state": "moving", "progress": 0.3, "hint": "Move your index finger"}

    def reset(self, reason):
        self.resets.append(reason)
        return {"state": "idle", "progress": 0.0, "hint": "Hand mouse released"}


@pytest.mark.parametrize("fingers, pinched, expected", [
    ((True, True, True, True), False, True),
    ((True, True, True, True), True, False),
    ((True, True, True, False), False, False),
    ((True, True, False, False), False, False),
    ((True, False, False, False), False, False),
    ((False, True, True, False), False, False),
])
@pytest.mark.parametrize("thumb", [False, True])
def test_navigation_clutch_requires_four_fingers_and_separated_thumb(fingers, pinched, expected, thumb):
    assert g.is_navigation_pose(_hand(*fingers, pinch=pinched, thumb=thumb)) is expected


def test_three_fingers_stay_in_mouse_mode_and_four_fingers_require_a_fresh_hold():
    mouse = _FakeMouseController()
    motions, progress = [], []
    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse,
                                     on_motion=motions.append, on_progress=progress.append)
    for at in (0.0, 0.15, 0.3, 0.5):
        recognizer._handle(_hand(True, True, True, False), at)
        assert not recognizer.navigation_active
    assert len(mouse.updates) == 4
    assert mouse.resets == motions == []
    recognizer._handle(_hand(), 0.7)
    assert recognizer.navigation_active
    assert mouse.resets == ["navigation_started"]
    assert progress[-1]["state"] == "arming" and progress[-1]["progress"] == 0
    recognizer._handle(_hand(), 0.99)
    assert progress[-1]["state"] == "arming"
    assert motions == []
    recognizer._handle(_hand(), 1.0)
    assert progress[-1]["state"] == "armed"
    assert len(mouse.updates) == 4


@pytest.mark.parametrize("pose", [g.POINT, g.PINCH, g.FIST, g.THUMBS_UP, g.TWO_FINGER, g.OPEN_PALM])
def test_mouse_navigation_tracker_cannot_emit_window_or_close_actions(pose):
    tracker = g.MeasuredGestureTracker(clutch_pose=g.FOUR_FINGER, pose_actions=False)
    records = []
    for i in range(30):
        records += tracker.update(pose, 0.5, 0.5, 0.1, i * 0.1)
    assert not _of_kind(records, "action")
    assert not _of_kind(records, "motion")


@pytest.mark.parametrize("axis", ["horizontal", "vertical"])
@pytest.mark.parametrize("thumb", [False, True])
def test_four_finger_navigation_owns_mouse_until_automatic_completion_and_neutral(axis, thumb):
    mouse = _FakeMouseController()
    motions, progress, actions = [], [], []
    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse,
                                     on_progress=progress.append, on_gesture=actions.append)

    def motion(record):
        assert recognizer.navigation_active
        assert mouse.resets == ["navigation_started"]
        motions.append(record)

    recognizer.on_motion = motion
    recognizer._handle(_hand(True, False, False, False), 0.0)
    nav = _hand(True, True, True, True, thumb=thumb)
    recognizer._handle(nav, 0.1)
    assert recognizer.navigation_active
    assert progress[-1]["gesture"] == "hand_navigation"
    assert progress[-1]["state"] == "arming"
    recognizer._handle(nav, 0.39)
    assert not motions
    recognizer._handle(nav, 0.41)
    assert progress[-1]["state"] == "armed"
    wrist = (0.86, 0.8) if axis == "horizontal" else (0.5, 0.44)
    recognizer._handle(_hand(True, True, True, True, wrist=wrist, thumb=thumb), 0.61)
    assert motions[-1]["phase"] == "begin" and motions[-1]["axis"] == axis
    assert abs(motions[-1]["progress"]) == pytest.approx(1)
    assert progress[-1]["axis"] == axis and progress[-1]["input_mode"] == "mouse"
    assert progress[-1]["state"] == "committing"
    assert len(mouse.updates) == 1
    # The pinky can curl after deliberate activation without cancelling travel.
    for at in (0.65, 0.69, 0.73):
        recognizer._handle(_hand(True, True, True, False, wrist=wrist, thumb=thumb), at)
    assert [record["phase"] for record in motions].count("end") == 1
    assert progress[-1]["state"] == "completed"
    assert recognizer.navigation_active
    fist = _hand(False, False, False, False, thumb=False, wrist=wrist)
    recognizer._handle(fist, 0.8)
    recognizer._handle(fist, 0.93)
    assert recognizer.navigation_active
    recognizer._handle(fist, 1.01)
    assert not recognizer.navigation_active
    assert len(mouse.updates) == 1
    assert actions == []
    recognizer._handle(_hand(True, False, False, False, wrist=wrist), 1.05)
    assert len(mouse.updates) == 2


@pytest.mark.parametrize("end", ["loss", "stop", "camera_failure", "mode_change"])
def test_navigation_loss_stop_camera_failure_and_mode_changes_cancel_contacts(end):
    mouse = _FakeMouseController()
    motions = []
    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse,
                                     on_motion=motions.append)
    recognizer._handle(_hand(True, False, False, False), 0.0)
    recognizer._handle(_hand(True, True, True, True), 0.1)
    recognizer._handle(_hand(True, True, True, True), 0.4)
    recognizer._handle(_hand(True, True, True, True, wrist=(0.5, 0.44)), 0.6)
    assert motions[-1]["axis"] == "vertical"
    if end == "loss":
        recognizer._handle([], 0.65)
        recognizer._handle([], 0.77)
        recognizer._handle([], 0.86)
    elif end == "stop":
        recognizer.stop()
    elif end == "camera_failure":
        recognizer._reset_controls(0.65, "camera_read_failed")
    else:
        assert recognizer.configure(input_mode="desktop")["ok"]
    assert motions[-1]["phase"] == "cancel"
    assert motions[-1]["axis"] == "vertical"
    assert not recognizer.navigation_active
    assert len(mouse.resets) == (1 if end == "loss" else 2)
    assert len(mouse.updates) == 1


def test_four_finger_candidate_cannot_start_if_pointer_button_release_fails():
    mouse = _FakeMouseController()
    motions, attempts = [], []
    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse,
                                     on_motion=motions.append)
    recognizer._handle(_hand(True, False, False, False), 0.0)

    def failed_release(reason):
        attempts.append(reason)
        return {"error": "left button still held"}

    mouse.reset = failed_release
    for at in (0.1, 0.4, 0.6):
        recognizer._handle(_hand(True, True, True, True), at)
    assert attempts == ["navigation_started"]
    assert not recognizer.navigation_active
    assert not motions
    assert "still held" in recognizer._mouse_error


def test_navigation_callback_failure_stays_in_navigation_until_pose_is_released():
    mouse = _FakeMouseController()
    motions = []

    def failing_motion(record):
        motions.append(record)
        if record["phase"] == "begin":
            return {"error": "native contacts failed"}

    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse,
                                     on_motion=failing_motion)
    recognizer._handle(_hand(True, False, False, False), 0.0)
    for at, wrist in ((0.1, (0.5, 0.8)), (0.4, (0.5, 0.8)),
                       (0.6, (0.5, 0.44)), (0.7, (0.5, 0.44))):
        recognizer._handle(_hand(True, True, True, True, wrist=wrist), at)
    assert [record["phase"] for record in motions] == ["begin", "cancel"]
    assert all(record["axis"] == "vertical" for record in motions)
    assert recognizer.navigation_active
    assert len(mouse.updates) == 1
    recognizer._handle(_hand(False, False, False, False, thumb=False), 0.8)
    recognizer._handle(_hand(False, False, False, False, thumb=False), 0.93)
    recognizer._handle(_hand(False, False, False, False, thumb=False), 1.01)
    assert not recognizer.navigation_active


@pytest.mark.parametrize("failure_type", ["exception", "result"])
@pytest.mark.parametrize("initial_failure", ["begin", "cancel"])
def test_failed_navigation_cleanup_blocks_pointer_and_restart_until_missing_hand_retry(monkeypatch, failure_type, initial_failure):
    mouse = _FakeMouseController()
    motions, progress = [], []
    failing = [True]

    def motion(record):
        motions.append(record)
        if failing[0] and (record["phase"] == "cancel"
                           or (initial_failure == "begin" and record["phase"] == "begin")):
            if failure_type == "exception":
                raise RuntimeError("navigation release is pending")
            return {"error": "navigation release is pending", "cleanup_pending": True}
        return {"ok": True}

    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse,
                                     on_motion=motion, on_progress=progress.append)
    monkeypatch.setattr(recognizer, "probe", lambda: {"available": True})
    recognizer._handle(_hand(True, False, False, False), 0.0)
    recognizer._handle(_hand(True, True, True, True), 0.1)
    recognizer._handle(_hand(True, True, True, True), 0.4)
    recognizer._handle(_hand(True, True, True, True, wrist=(0.5, 0.44)), 0.6)
    if initial_failure == "cancel":
        recognizer._handle([], 0.65)
        recognizer._handle([], 0.77)
    assert recognizer.navigation_active
    assert recognizer.status()["navigation_cleanup_pending"] is True
    assert recognizer.preview()["navigation_cleanup_pending"] is True
    assert "cleanup is still pending" in recognizer.start()["error"]
    assert "cleanup is still pending" in recognizer.configure(input_mode="desktop")["error"]
    assert progress[-1]["state"] == "error"
    attempts = len(motions)
    recognizer._handle(_hand(True, False, False, False), 0.8)
    recognizer._handle(_hand(True, True, True, True), 0.85)
    assert len(motions) == attempts
    assert len(mouse.updates) == 1
    failing[0] = False
    recognizer._handle([], 0.9)
    assert motions[-1]["phase"] == "cancel" and motions[-1]["axis"] == "vertical"
    assert not recognizer.navigation_active
    assert recognizer.status()["navigation_cleanup_pending"] is False
    recognizer._handle(_hand(True, False, False, False), 0.95)
    assert len(mouse.updates) == 2


@pytest.mark.parametrize("mode", ["mouse", "desktop"])
def test_camera_stop_reports_navigation_cleanup_failure_and_can_retry(monkeypatch, mode):
    failing = [True]

    def motion(record):
        if record["phase"] == "cancel" and failing[0]:
            return {"error": "navigation contacts still held", "cleanup_pending": True}
        return {"ok": True}

    recognizer = g.GestureRecognizer(input_mode=mode, on_motion=motion)
    monkeypatch.setattr(recognizer, "probe", lambda: {"available": True})
    for at, wrist in ((0.0, (0.5, 0.8)), (0.3, (0.5, 0.8)), (0.5, (0.56, 0.8))):
        points = _hand(wrist=wrist)
        recognizer._handle(points, at)
    result = recognizer.stop()
    assert result["camera_released"] is True
    assert "navigation input release failed" in result["error"]
    assert recognizer.status()["state"] == "error"
    assert recognizer.status()["navigation_cleanup_pending"] is True
    assert "cleanup is still pending" in recognizer.start()["error"]
    failing[0] = False
    assert recognizer.stop() == {"ok": True}
    assert recognizer.status()["navigation_cleanup_pending"] is False
    assert recognizer.status()["state"] == "off"


def test_navigation_reset_requires_fresh_pointing_before_real_mouse_movement_resumes():
    from core.hand_mouse import HandMouseController

    events, progress = [], []
    device = SimpleNamespace(
        position=lambda: (0.5, 0.5), blocked=lambda **_: None,
        move=lambda x, y: events.append(("move", x, y)),
        left_down=lambda: events.append(("down",)),
        left_up=lambda: events.append(("up",)))
    mouse = HandMouseController(device_factory=lambda: device)
    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse,
                                     on_progress=progress.append, on_motion=lambda _: {"ok": True})
    point = _hand(True, False, False, False)
    point[7] = (point[6][0], (point[6][1] + point[8][1]) / 2, 0)
    for at in (0.0, 0.1, 0.21):
        recognizer._handle(point, at)
    moves = list(events)
    recognizer._handle(_hand(True, True, True, True), 0.3)
    recognizer._handle(_hand(True, True, True, True), 0.6)
    recognizer._handle(_hand(True, True, True, True, wrist=(0.56, 0.8)), 0.7)
    for at in (0.8, 0.93, 1.01):
        recognizer._handle(_hand(False, False, False, False, thumb=False), at)
    assert not recognizer.navigation_active
    recognizer._handle(_hand(True, True, True, False), 1.05)
    assert events == moves
    recognizer._handle(point, 1.1)
    assert progress[-1]["state"] == "mouse_arming"
    assert events == moves
    recognizer._handle(point, 1.31)
    assert progress[-1]["state"] == "mouse_pointer"
    assert len(events) > len(moves)
    assert all(event[0] == "move" for event in events)


@pytest.mark.parametrize("phase", ["click", "drag", "too_short"])
def test_open_hand_pinch_release_finishes_before_four_finger_navigation(phase):
    from core.hand_mouse import HandMouseController

    events, clicks, motions, progress = [], [], [], []
    device = SimpleNamespace(
        position=lambda: (0.5, 0.5), blocked=lambda **_: None,
        move=lambda x, y: events.append(("move", x, y)),
        left_down=lambda: events.append(("down",)),
        left_up=lambda: events.append(("up",)))
    mouse = HandMouseController(device_factory=lambda: device)
    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse,
                                     on_click=clicks.append, on_motion=motions.append,
                                     on_progress=progress.append)
    point = _hand(True, False, False, False)
    point[7] = (point[6][0], (point[6][1] + point[8][1]) / 2, 0)
    for at in (0.0, 0.1, 0.21):
        recognizer._handle(point, at)
    assert progress[-1]["state"] == "mouse_pointer"
    pinch = _hand(pinch=True)
    recognizer._handle(pinch, 0.25)
    assert progress[-1]["state"] == "mouse_pinch"
    release_at = 0.28 if phase == "too_short" else 0.38
    if phase != "too_short":
        if phase == "drag":
            recognizer._handle(pinch, 0.5)
            recognizer._handle(pinch, 0.61)
            assert progress[-1]["state"] == "mouse_dragging"
            release_at = 0.75
        # This separation is already a navigation pose, but remains inside
        # the mouse pinch's release hysteresis. It must keep mouse ownership.
        opening = list(pinch)
        opening[4] = (opening[8][0] + g.hand_scale(opening) * 0.5,
                      opening[8][1], 0)
        assert g.is_navigation_pose(opening)
        recognizer._handle(opening, release_at - 0.05)
        assert not recognizer.navigation_active
        assert clicks == []
    released = _hand()
    assert g.is_navigation_pose(released)
    recognizer._handle(released, release_at)
    assert not recognizer.navigation_active
    assert [event[0] for event in events if event[0] != "move"] == (
        [] if phase == "too_short" else ["down", "up"])
    assert [event["source"] for event in clicks] == (["pinch"] if phase == "click" else [])
    assert motions == []
    # Four fingers can begin their normal fresh hold on the next frame.
    before = list(events)
    recognizer._handle(released, release_at + 0.04)
    assert recognizer.navigation_active
    assert progress[-1]["state"] == "arming"
    assert events == before


def test_open_hand_pinch_release_failure_blocks_navigation():
    from core.hand_mouse import HandMouseController

    events, clicks, motions = [], [], []

    def failed_release():
        events.append("up")
        raise OSError("left button still held")

    device = SimpleNamespace(
        position=lambda: (0.5, 0.5), blocked=lambda **_: None,
        move=lambda *_: None, left_down=lambda: events.append("down"),
        left_up=failed_release)
    mouse = HandMouseController(device_factory=lambda: device)
    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse,
                                     on_click=clicks.append, on_motion=motions.append)
    point = _hand(True, False, False, False)
    point[7] = (point[6][0], (point[6][1] + point[8][1]) / 2, 0)
    for at in (0.0, 0.1, 0.21):
        recognizer._handle(point, at)
    recognizer._handle(_hand(pinch=True), 0.25)
    for at in (0.35, 0.40, 0.65):
        recognizer._handle(_hand(), at)
        assert not recognizer.navigation_active
    assert events[0] == "down"
    assert events.count("down") == 1
    assert "left button still held" in recognizer._mouse_error
    assert clicks == motions == []


def test_successful_mouse_click_feedback_is_discrete_deduplicated_and_not_progress_throttled():
    mouse = _FakeMouseController()
    result = {"state": "mouse_clicked", "progress": 1.0, "hint": "Clicked",
              "click_id": 1, "click_source": "bend"}
    mouse.update = lambda points, now: dict(result)
    clicks, progress = [], []
    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse,
                                     on_click=clicks.append, on_progress=progress.append)
    recognizer._handle(_hand(True, False, False, False), 0.0)
    recognizer._handle(_hand(True, False, False, False), 0.001)  # repeated diagnostic must not click again
    result.update(click_id=2, click_source="pinch")
    recognizer._handle(_hand(True, False, False, False), 0.002)
    assert [event["source"] for event in clicks] == ["bend", "pinch"]
    assert all(event["type"] == "gesture_click" for event in clicks)
    assert clicks[0]["id"] != clicks[1]["id"]
    assert [event["at"] for event in clicks] == [0.0, 0.002]
    assert len(progress) == 1
    assert "click_id" not in progress[0] and "click_source" not in progress[0]


@pytest.mark.parametrize("diagnostic", [
    {"state": "mouse_clicked", "hint": "Pose recognized"},
    {"state": "mouse_pointer", "hint": "Drag released"},
    {"state": "mouse_error", "click_id": 1, "click_source": "bend", "error": "left-up failed"},
    {"state": "mouse_clicked", "click_id": True, "click_source": "bend"},
    {"state": "mouse_clicked", "click_id": 0, "click_source": "bend"},
    {"state": "mouse_clicked", "click_id": 1, "click_source": "drag"},
])
def test_pose_only_drag_release_errors_and_invalid_click_markers_make_no_click_feedback(diagnostic):
    mouse = _FakeMouseController()
    mouse.update = lambda points, now: dict(diagnostic)
    clicks = []
    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse,
                                     on_click=clicks.append)
    recognizer._handle(_hand(True, False, False, False), 0.0)
    assert clicks == []


def test_reset_error_and_preview_diagnostics_cannot_replay_click_feedback(monkeypatch):
    mouse = _FakeMouseController()
    diagnostic = {"state": "mouse_clicked", "click_id": 1, "click_source": "bend", "error": "failed click"}
    mouse.update = lambda points, now: dict(diagnostic)
    clicks = []
    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse,
                                     on_click=clicks.append)
    monkeypatch.setattr(recognizer, "probe", lambda: {"available": True})
    recognizer._handle(_hand(True, False, False, False), 0.0)
    recognizer._handle([], 0.1)
    diagnostic.pop("error")
    recognizer._handle(_hand(True, False, False, False), 0.2)
    assert clicks == []  # a formerly failed result cannot become a new click
    diagnostic["click_id"] = 2
    mouse.reset = lambda reason: dict(diagnostic)
    recognizer._handle([], 0.3)
    recognizer._handle(_hand(True, False, False, False), 0.4)
    assert clicks == []  # reset diagnostics are consumed without notification
    diagnostic["click_id"] = 3
    recognizer._handle(_hand(True, False, False, False), 0.5)
    assert len(clicks) == 1
    for _ in range(3):
        assert "click_id" not in recognizer.preview()
        assert "click_id" not in recognizer.status()
    assert len(clicks) == 1


def test_click_callback_failure_is_not_retried_and_does_not_reset_mouse_input():
    mouse = _FakeMouseController()
    mouse.update = lambda points, now: {"click_id": 1, "click_source": "pinch"}
    clicks = []

    def fail_feedback(event):
        clicks.append(event)
        raise RuntimeError("Audio unavailable")

    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse, on_click=fail_feedback)
    recognizer._handle(_hand(True, False, False, False), 0.0)
    recognizer._handle(_hand(True, False, False, False), 0.1)
    assert len(clicks) == 1
    assert mouse.resets == []
    assert recognizer._mouse_error is None


def test_click_event_ids_are_unique_between_recognizers_and_camera_sessions(monkeypatch):
    class HeldThread:
        def __init__(self, **kwargs):
            self.alive = False

        def is_alive(self):
            return self.alive

        def start(self):
            self.alive = True

        def join(self, **kwargs):
            self.alive = False

    monkeypatch.setattr(g.threading, "Thread", HeldThread)
    clicks = []
    diagnostic = {"click_id": 1, "click_source": "bend"}
    mouse = _FakeMouseController()
    mouse.update = lambda points, now: dict(diagnostic)
    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse, on_click=clicks.append)
    monkeypatch.setattr(recognizer, "probe", lambda: {"available": True})
    recognizer._handle(_hand(True, False, False, False), 0.0)
    first_id = clicks[-1]["id"]
    assert recognizer.start()["ok"]
    recognizer._handle(_hand(True, False, False, False), 0.1)
    assert len(clicks) == 1  # existing controller watermark survives restart
    diagnostic["click_id"] = 2
    recognizer._handle(_hand(True, False, False, False), 0.2)
    assert clicks[-1]["id"].split(":")[0] != first_id.split(":")[0]
    other = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse, on_click=clicks.append)
    other._handle(_hand(True, False, False, False), 0.3)
    assert len({event["id"] for event in clicks}) == 3
    assert recognizer.stop()["ok"]


@pytest.mark.parametrize("value", [None, True, 1, "", "MOUSE", "eye"])
def test_input_mode_requires_explicit_desktop_or_mouse(value):
    with pytest.raises(ValueError, match="input_mode"):
        g.GestureRecognizer(input_mode=value)


def test_mouse_controller_is_lazy_and_desktop_remains_the_constructor_default(monkeypatch):
    created = []
    recognizer = g.GestureRecognizer(mouse_factory=lambda: created.append(True))
    assert recognizer.input_mode == "desktop"
    recognizer._handle(_hand(), 0.0)
    recognizer._handle(_hand(), 0.35)
    assert created == []
    recognizer.configure(input_mode="mouse")
    recognizer._handle([], 0.4)
    recognizer.preview()
    assert created == []
    monkeypatch.setattr(recognizer, "probe", lambda: {"available": True})
    assert recognizer.status()["input_mode"] == "mouse"
    assert recognizer.preview()["input_mode"] == "mouse"


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", [], {}])
def test_bend_click_constructor_requires_an_explicit_boolean(value):
    with pytest.raises(ValueError, match="bend_click"):
        g.GestureRecognizer(bend_click=value)


def test_bend_click_is_optional_preserved_and_applied_to_existing_controller(monkeypatch):
    mouse = _FakeMouseController()
    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse)
    monkeypatch.setattr(recognizer, "probe", lambda: {"available": True})
    assert recognizer.status()["bend_click"] is False
    assert recognizer.preview()["bend_click"] is False
    recognizer._handle(_hand(True, False, False, False), 0.0)
    assert mouse.bend_click is False
    assert recognizer.configure(bend_click=True)["bend_click"] is True
    assert mouse.bend_click is True
    assert recognizer.configure(model_complexity=0)["ok"]
    assert recognizer.status()["bend_click"] is True
    assert recognizer.preview()["bend_click"] is True
    assert "error" in recognizer.configure(bend_click="false", travel_palms=2.5)
    assert recognizer.measured.travel_palms == 1.2
    recognizer._running = True
    assert "error" in recognizer.configure(bend_click=False)
    assert mouse.bend_click is recognizer.bend_click is True


def test_bend_click_reaches_lazy_controller_without_opening_devices(monkeypatch):
    from core import hand_mouse

    created = []

    def make_mouse(**kwargs):
        created.append(kwargs)
        return _FakeMouseController()

    monkeypatch.setattr(hand_mouse, "HandMouseController", make_mouse)
    recognizer = g.GestureRecognizer(input_mode="mouse", bend_click=True)
    recognizer._handle([], 0.0)
    assert created == []
    recognizer._handle(_hand(True, False, False, False), 0.1)
    assert created == [{"bend_click": True}]


@pytest.mark.parametrize("failure", ["raise", "result"])
def test_bend_click_change_keeps_settings_when_button_release_fails(failure):
    mouse = _FakeMouseController()
    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse)
    recognizer._handle(_hand(True, False, False, False), 0.0)

    def fail_change(_enabled):
        if failure == "raise":
            raise RuntimeError("button release failed")
        return {"error": "button release failed"}

    mouse.set_bend_click = fail_change
    result = recognizer.configure(bend_click=True, travel_palms=2.5)
    assert "button release failed" in result["error"]
    assert recognizer.bend_click is False
    assert recognizer.measured.travel_palms == 1.2
    mouse.set_bend_click = lambda enabled: {"state": "idle"}
    assert recognizer.configure(bend_click=True)["ok"]
    assert recognizer._mouse_error is None


def test_mouse_mode_routes_ordinary_poses_without_desktop_window_or_close_actions(monkeypatch):
    mouse = _FakeMouseController()
    created, gestures, motions, progress = [], [], [], []

    def make_mouse():
        created.append(True)
        return mouse

    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=make_mouse,
                                     on_gesture=gestures.append, on_motion=motions.append,
                                     on_progress=progress.append)
    monkeypatch.setattr(recognizer.measured, "update", lambda *_: pytest.fail("Mouse frame reached desktop tracker"))
    monkeypatch.setattr(recognizer.measured, "reset", lambda *_: pytest.fail("Mouse reset reached desktop tracker"))
    poses = [_hand(True, True, True, False), _hand(True, False, False, False, pinch=True),
             _hand(False, False, False, False, thumb=True),
             _hand(True, True, False, False, thumb=False)]
    for i, points in enumerate(poses):
        recognizer._handle(points, i * 2.0)
    assert created == [True]
    assert len(mouse.updates) == len(poses)
    assert gestures == motions == []
    assert progress[-1]["input_mode"] == "mouse"
    assert progress[-1]["gesture"] == "hand_mouse"
    assert progress[-1]["text"] == "Move your index finger"
    recognizer._handle([], 7.0)
    assert mouse.resets == []
    assert mouse.updates[-1] == ([], 7.0)
    assert gestures == motions == []


@pytest.mark.parametrize("missing", [[], [(0.5, 0.5, 0)] * 20,
                                    [(float("nan"), 0.5, 0)] * 21])
def test_missing_mouse_frames_reach_controller_grace_without_replaying_clicks(missing):
    mouse = _FakeMouseController()
    clicks, progress = [], []
    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse,
                                     on_click=clicks.append, on_progress=progress.append)
    recognizer._handle(_hand(True, False, False, False), 0.0)

    def lost_update(points, now):
        mouse.updates.append((points, now))
        # Even malformed feedback on a missing frame must not create a click
        # sound now or replay its watermark on a later recovered frame.
        return {"state": "mouse_uncertain", "hint": "Keep your hand visible",
                "click_id": 1, "click_source": "pinch"}

    mouse.update = lost_update
    recognizer._handle(missing, 0.05)
    assert mouse.updates[-1] == ([], 0.05)
    assert mouse.resets == []
    assert progress[-1]["state"] == "mouse_uncertain"
    assert recognizer._mouse_pose == g.NONE
    recognizer._handle(_hand(True, False, False, False), 0.1)
    assert clicks == []
    assert mouse.resets == []


@pytest.mark.parametrize("phase", ["moving", "pinching", "dragging"])
def test_recognizer_preserves_only_movement_across_a_real_controller_dropout(phase):
    from core.hand_mouse import HandMouseController

    events, clicks, progress = [], [], []
    device = SimpleNamespace(
        position=lambda: (0.5, 0.5), blocked=lambda **_: None,
        move=lambda x, y: events.append(("move", x, y)),
        left_down=lambda: events.append(("down",)),
        left_up=lambda: events.append(("up",)))
    mouse = HandMouseController(device_factory=lambda: device)
    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse,
                                     on_click=clicks.append, on_progress=progress.append)
    point = _hand(True, False, False, False)
    point[7] = (point[6][0], (point[6][1] + point[8][1]) / 2, 0)
    for now in (0.0, 0.1, 0.21):
        recognizer._handle(point, now)
    assert progress[-1]["state"] == "mouse_pointer"
    lost_at = 0.26
    if phase != "moving":
        pinch = list(point)
        pinch[4] = pinch[8]
        recognizer._handle(pinch, 0.25)
        assert progress[-1]["state"] == "mouse_pinch"
        if phase == "dragging":
            recognizer._handle(pinch, 0.45)
            recognizer._handle(pinch, 0.61)
            assert progress[-1]["state"] == "mouse_dragging"
            lost_at = 0.63
    moves_before_loss = len([event for event in events if event[0] == "move"])
    recognizer._handle([], lost_at)
    assert progress[-1]["state"] == "mouse_recovering"
    assert len([event for event in events if event[0] == "move"]) == moves_before_loss
    expected_buttons = ["down", "up"] if phase == "dragging" else []
    assert [event[0] for event in events if event[0] != "move"] == expected_buttons

    # Relax the other fingers during recovery: movement resumes without a new
    # activation hold, while missing-frame click and drag state stays cancelled.
    relaxed = list(point)
    for tip in (12, 16):
        relaxed[tip] = (relaxed[tip][0], point[8][1], 0)
    recognizer._handle(relaxed, lost_at + 0.05)
    assert progress[-1]["state"] == "mouse_pointer"
    assert len([event for event in events if event[0] == "move"]) == moves_before_loss
    recognizer._handle(relaxed, lost_at + 0.08)
    assert len([event for event in events if event[0] == "move"]) > moves_before_loss
    assert [event[0] for event in events if event[0] != "move"] == expected_buttons
    assert clicks == []


def test_mouse_control_region_reaches_progress_and_same_frame_preview(monkeypatch):
    recognizer, frame, cv2, clock, _encoded, _resized = _preview_fixture(monkeypatch)
    recognizer.input_mode = "mouse"
    progress = []
    recognizer.on_progress = progress.append
    recognizer.preview()
    recognizer._report_mouse_progress({"state": "moving", "control_region": (0.18, 0.12, 0.82, 0.88)}, clock[0])
    recognizer._publish_preview(frame, _hand(), clock[0], cv2)
    assert progress[-1]["control_region"] == [0.18, 0.12, 0.82, 0.88]
    assert recognizer.preview()["control_region"] == [0.18, 0.12, 0.82, 0.88]
    copied = recognizer.preview()
    copied["control_region"][0] = 99
    assert recognizer.preview()["control_region"][0] == 0.18
    recognizer.input_mode = "desktop"
    assert "control_region" not in recognizer.preview()


def test_mouse_navigation_preview_reports_its_clutch_and_hides_pointer_region(monkeypatch):
    recognizer, frame, cv2, clock, _encoded, _resized = _preview_fixture(monkeypatch)
    recognizer.input_mode = "mouse"
    monkeypatch.setattr(recognizer, "probe", lambda: {"available": True})
    recognizer.preview()
    recognizer._report_mouse_progress({"state": "moving", "control_region": (0.18, 0.12, 0.82, 0.88)}, clock[0])
    points = _hand(True, True, True, True)
    recognizer._handle(points, clock[0])
    recognizer._publish_preview(frame, points, clock[0], cv2)
    preview = recognizer.preview()
    assert preview["raw_pose"] == preview["pose"] == g.FOUR_FINGER
    assert preview["state"] == "arming"
    assert preview["navigation_active"] is True
    assert recognizer.status()["navigation_active"] is True
    assert "control_region" not in preview


@pytest.mark.parametrize("region", [[0, 0, 1], [0, 0, 2, 1], [0.8, 0.1, 0.2, 0.9],
                                    [0, 0, float("nan"), 1], [False, 0, 1, 1], "0,0,1,1"])
def test_invalid_mouse_control_regions_are_not_forwarded(region):
    progress = []
    recognizer = g.GestureRecognizer(input_mode="mouse", on_progress=progress.append)
    recognizer._report_mouse_progress({"state": "moving", "control_region": region}, 0.0)
    assert "control_region" not in progress[-1]
    assert "control_region" not in recognizer._diagnostic


@pytest.mark.parametrize("failure", ["raise", "result"])
def test_mouse_failures_are_visible_release_input_and_wait_for_hand_loss_before_retry(failure):
    mouse = _FakeMouseController()
    mouse.failure = failure
    progress = []
    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse,
                                     on_progress=progress.append)
    recognizer._handle(_hand(True, False, False, False), 0.0)
    assert mouse.resets == ["mouse_input_error"]
    assert progress[-1]["state"] == "error"
    assert "Input unavailable" in progress[-1]["text"]
    recognizer._handle(_hand(True, False, False, False), 0.1)
    assert len(mouse.updates) == 1
    recognizer._handle([], 0.2)
    mouse.failure = None
    recognizer._handle(_hand(True, False, False, False), 0.3)
    assert len(mouse.updates) == 2
    assert progress[-1]["state"] == "moving"


def test_mouse_factory_failure_is_visible_and_does_not_create_a_retry_storm():
    attempts, progress = [], []

    def fail_factory():
        attempts.append(True)
        raise RuntimeError("Mouse controller unavailable")

    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=fail_factory,
                                     on_progress=progress.append)
    recognizer._handle(_hand(True, False, False, False), 0.0)
    recognizer._handle(_hand(True, False, False, False), 0.1)
    assert attempts == [True]
    assert "Mouse controller unavailable" in progress[-1]["text"]


def test_mode_configuration_preserves_omitted_mode_and_resets_before_switching(monkeypatch):
    mouse = _FakeMouseController()
    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse)
    recognizer._handle(_hand(True, False, False, False), 0.0)
    assert recognizer.configure(model_complexity=0)["ok"]
    assert recognizer.input_mode == "mouse"
    assert recognizer.configure(input_mode="desktop")["input_mode"] == "desktop"
    assert mouse.resets == ["mode_changed"]
    assert "error" in recognizer.configure(input_mode="bad", travel_palms=2.5)
    assert recognizer.input_mode == "desktop"
    assert recognizer.measured.travel_palms == 1.2
    recognizer._running = True
    assert "error" in recognizer.configure(input_mode="mouse")
    recognizer._running = False
    recognizer._state = "starting"
    assert "error" in recognizer.configure(input_mode="mouse")


@pytest.mark.parametrize("failure", ["raise", "result"])
def test_mode_switch_does_not_abandon_a_mouse_button_that_failed_to_release(failure):
    mouse = _FakeMouseController()

    def failed_reset(reason):
        if failure == "raise":
            raise RuntimeError("button release failed")
        return {"error": "button release failed"}

    mouse.reset = failed_reset
    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse)
    recognizer._handle(_hand(True, False, False, False), 0.0)
    assert "error" in recognizer.configure(input_mode="desktop")
    assert recognizer.input_mode == "mouse"


def test_stop_releases_mouse_and_rejects_late_capture_frames():
    mouse = _FakeMouseController()
    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse)
    recognizer._handle(_hand(True, False, False, False), 0.0)
    assert recognizer.stop()["ok"]
    assert mouse.resets == ["camera_stopped"]
    recognizer._handle(_hand(True, False, False, False), 1.0)
    assert len(mouse.updates) == 1
    assert mouse.resets[-1] == "camera_stopped"


@pytest.mark.parametrize("failure", ["raise", "result"])
def test_stop_reports_mouse_release_failure_persistently_with_camera_off_and_can_retry(monkeypatch, failure):
    mouse = _FakeMouseController()
    statuses = []

    def failed_reset(reason):
        if failure == "raise":
            raise RuntimeError("left button is still held")
        return {"error": "left button is still held"}

    mouse.reset = failed_reset
    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse,
                                     on_status=statuses.append)
    monkeypatch.setattr(recognizer, "probe", lambda: {"available": True})
    recognizer._handle(_hand(True, False, False, False), 0.0)
    result = recognizer.stop()
    assert result["camera_released"] is True
    assert "button release failed" in result["error"]
    assert recognizer.is_running() is False
    assert recognizer.status()["state"] == "error"
    assert "camera is released" in recognizer.status()["text"]
    assert "left button is still held" in statuses[-1]["text"]
    assert statuses[-1]["running"] is False
    assert recognizer._mouse is mouse  # ownership is retained for a later retry
    mouse.reset = lambda reason: {"state": "idle", "hint": "released"}
    assert recognizer.stop() == {"ok": True}
    assert recognizer.status()["state"] == "off"


def test_worker_finally_keeps_mouse_release_failure_visible_after_releasing_camera(monkeypatch):
    mouse = _FakeMouseController()
    mouse.reset = lambda reason: {"error": "left button is still held"}
    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse)
    recognizer._handle(_hand(True, False, False, False), 0.0)
    released = []

    class Hands:
        def process(self, frame):
            raise RuntimeError("capture processing failed")

        def close(self):
            pass

    camera = SimpleNamespace(isOpened=lambda: True, set=lambda *_: None,
                             read=lambda: (True, object()), release=lambda: released.append(True))
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        CAP_DSHOW=1, CAP_PROP_FRAME_WIDTH=2, CAP_PROP_FRAME_HEIGHT=3, COLOR_BGR2RGB=4,
        VideoCapture=lambda *_: camera, flip=lambda frame, _axis: frame,
        cvtColor=lambda frame, _mode: frame))
    monkeypatch.setitem(sys.modules, "mediapipe", SimpleNamespace(
        solutions=SimpleNamespace(hands=SimpleNamespace(Hands=lambda **_: Hands()))))
    recognizer._run()
    assert released == [True]
    status = recognizer.status()
    assert status["running"] is False
    assert status["state"] == "error"
    assert "capture processing failed" in status["text"]
    assert "camera is released" in status["text"]
    assert "button release failed" in status["text"]
    assert recognizer._mouse is mouse


def test_stop_serializes_mouse_release_after_an_in_progress_frame_without_deadlock():
    entered, finish = g.threading.Event(), g.threading.Event()
    order = []
    mouse = _FakeMouseController()

    def held_update(points, now):
        entered.set()
        assert finish.wait(timeout=2.0)
        order.append("update")
        return {"state": "moving"}

    def reset(reason):
        order.append("reset")
        return {"state": "idle"}

    mouse.update, mouse.reset = held_update, reset
    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse)
    worker = g.threading.Thread(target=lambda: recognizer._handle(_hand(True, False, False, False), 0.0))
    worker.start()
    assert entered.wait(timeout=2.0)
    stopper = g.threading.Thread(target=recognizer.stop)
    stopper.start()
    assert recognizer._stop.wait(timeout=2.0)
    finish.set()
    worker.join(timeout=2.0)
    stopper.join(timeout=2.0)
    assert not worker.is_alive() and not stopper.is_alive()
    assert order == ["update", "reset"]


@pytest.mark.parametrize("failure", ["read", "processing"])
def test_mouse_input_is_reset_on_camera_read_failure_and_worker_error(monkeypatch, failure):
    mouse = _FakeMouseController()
    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse)
    recognizer._handle(_hand(True, False, False, False), 0.0)

    class Hands:
        def process(self, frame):
            raise RuntimeError("capture processing failed")

        def close(self):
            pass

    camera = SimpleNamespace(isOpened=lambda: True, set=lambda *_: None,
                             read=lambda: (failure != "read", object()), release=lambda: None)
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        CAP_DSHOW=1, CAP_PROP_FRAME_WIDTH=2, CAP_PROP_FRAME_HEIGHT=3, COLOR_BGR2RGB=4,
        VideoCapture=lambda *_: camera, flip=lambda frame, _axis: frame,
        cvtColor=lambda frame, _mode: frame))
    monkeypatch.setitem(sys.modules, "mediapipe", SimpleNamespace(
        solutions=SimpleNamespace(hands=SimpleNamespace(Hands=lambda **_: Hands()))))

    def stop_after_failed_read(_seconds):
        assert mouse.resets == ["camera_read_failed"]
        recognizer._stop.set()

    monkeypatch.setattr(g.time, "sleep", stop_after_failed_read)
    recognizer._run()
    assert mouse.resets[-1] == "camera_stopped"
    assert recognizer.is_running() is False
    if failure == "processing":
        assert recognizer.status()["state"] == "error"


def _preview_fixture(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(g.time, "monotonic", lambda: clock[0])
    encoded, resized = [], []

    def resize(_frame, size, **_kwargs):
        resized.append(size)
        return SimpleNamespace(shape=(size[1], size[0], 3))

    def imencode(extension, frame, options):
        encoded.append((extension, frame.shape, options))
        return True, SimpleNamespace(tobytes=lambda: b"jpeg data")

    cv2 = SimpleNamespace(resize=resize, imencode=imencode, INTER_AREA=1, IMWRITE_JPEG_QUALITY=2)
    recognizer = g.GestureRecognizer()
    recognizer._running, recognizer._state = True, "running"
    frame = SimpleNamespace(shape=(480, 640, 3))
    return recognizer, frame, cv2, clock, encoded, resized


def test_preview_never_starts_the_camera_or_grants_a_lease_while_off(monkeypatch):
    recognizer = g.GestureRecognizer()
    monkeypatch.setattr(recognizer, "start", lambda: pytest.fail("Preview started camera"))
    preview = recognizer.preview()
    assert preview["running"] is False
    assert preview["image"] is None
    assert preview["landmarks"] == []
    assert preview["age_ms"] is None
    assert recognizer._preview_lease_until == 0.0


def test_preview_only_encodes_during_a_lease_and_limits_size_and_rate(monkeypatch):
    recognizer, frame, cv2, clock, encoded, resized = _preview_fixture(monkeypatch)
    points = _hand()
    recognizer._handle(points, clock[0])
    recognizer._publish_preview(frame, points, clock[0], cv2)
    assert encoded == []
    assert recognizer.preview()["image"] is None
    recognizer._publish_preview(frame, points, clock[0], cv2)
    preview = recognizer.preview()
    assert preview["image"].startswith("data:image/jpeg;base64,")
    assert (preview["width"], preview["height"]) == (480, 360)
    assert resized == [(480, 360)]
    assert preview["landmarks"] == [list(point) for point in points]
    assert preview["raw_pose"] == preview["effective_pose"] == g.OPEN_PALM
    assert preview["model_name"] == "MediaPipe Hands full"
    first_sequence = preview["sequence"]
    first_points = preview["landmarks"]
    clock[0] += 0.05
    new_points = _hand(wrist=(0.51, 0.8))
    recognizer._publish_preview(frame, new_points, clock[0], cv2)
    preview = recognizer.preview()
    assert len(encoded) == 1
    assert preview["sequence"] == first_sequence
    assert preview["landmarks"] == first_points  # image and skeleton stay aligned
    assert preview["age_ms"] == 50
    preview["landmarks"][0][0] = 99
    assert recognizer.preview()["landmarks"][0][0] != 99
    clock[0] += 0.08
    recognizer._publish_preview(frame, new_points, clock[0], cv2)
    assert len(encoded) == 2
    assert recognizer.preview()["sequence"] > first_sequence
    assert recognizer.preview()["fps"] > 0
    clock[0] += 2.1
    recognizer._publish_preview(frame, new_points, clock[0], cv2)
    assert len(encoded) == 2
    assert recognizer._preview_snapshot is None
    assert recognizer.preview()["image"] is None


def test_no_hand_preview_has_current_camera_image_and_no_old_landmarks(monkeypatch):
    recognizer, frame, cv2, clock, encoded, _resized = _preview_fixture(monkeypatch)
    recognizer.preview()
    recognizer._publish_preview(frame, _hand(), clock[0], cv2)
    clock[0] += 0.03
    recognizer._handle([], clock[0])
    recognizer._publish_preview(frame, [], clock[0], cv2)
    assert recognizer.preview()["landmarks"] == []
    clock[0] += 0.1
    recognizer._publish_preview(frame, [], clock[0], cv2)
    preview = recognizer.preview()
    assert preview["image"] is not None
    assert preview["tracked"] is False
    assert preview["raw_pose"] == preview["effective_pose"] == g.NONE
    assert preview["landmarks"] == []
    assert preview["last_cancel_reason"] == "hand_lost"
    assert len(encoded) == 2


def test_preview_distinguishes_a_visible_unknown_pose_from_a_missing_hand(monkeypatch):
    recognizer, frame, cv2, clock, _encoded, _resized = _preview_fixture(monkeypatch)
    recognizer.preview()
    recognizer._handle(_hand(wrist=(0.3, 0.8)), 9.45)
    recognizer._handle(_hand(wrist=(0.3, 0.8)), 9.75)
    recognizer._handle(_hand(wrist=(0.66, 0.8)), 9.95)
    unclear = _hand(True, False, True, False, wrist=(0.66, 0.8))
    recognizer._handle(unclear, clock[0])
    recognizer._publish_preview(frame, unclear, clock[0], cv2)
    preview = recognizer.preview()
    assert preview["tracked"] is True
    assert preview["raw_pose"] == g.UNKNOWN
    assert preview["effective_pose"] == g.OPEN_PALM
    assert preview["state"] == "uncertain"
    assert preview["progress"] == pytest.approx(1.0)


def test_stale_preview_and_camera_stop_clear_pixels_and_landmarks(monkeypatch):
    recognizer, frame, cv2, clock, _encoded, _resized = _preview_fixture(monkeypatch)
    recognizer.preview()
    recognizer._publish_preview(frame, _hand(), clock[0], cv2)
    clock[0] += 0.6
    stale = recognizer.preview()
    assert stale["image"] is None
    assert stale["landmarks"] == []
    assert stale["last_cancel_reason"] == "stale_frame"
    recognizer._publish_preview(frame, _hand(), clock[0], cv2)
    assert recognizer.preview()["image"] is not None
    recognizer.stop()
    stopped = recognizer.preview()
    assert stopped["image"] is None
    assert stopped["landmarks"] == []
    assert stopped["running"] is False
    assert recognizer._preview_lease_until == 0.0
    recognizer._publish_preview(frame, _hand(), clock[0], cv2)
    assert recognizer._diagnostic["landmarks"] == []  # late frame cannot refill a stopped preview


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


def test_mediapipe_without_legacy_hands_is_reported_before_start(monkeypatch):
    """A newer MediaPipe wheel must not crash the capture thread on start."""
    monkeypatch.setitem(sys.modules, "mediapipe", SimpleNamespace(tasks=SimpleNamespace()))
    recognizer = g.GestureRecognizer()

    probe = recognizer.probe()
    assert probe["available"] is False
    assert "solutions.hands.Hands" in probe["text"]
    assert "error" in recognizer.start()
    assert recognizer._thread is None


def test_capture_failure_closes_hands_and_releases_camera(monkeypatch):
    camera = SimpleNamespace(
        isOpened=lambda: True,
        set=lambda *_: None,
        read=lambda: (True, object()),
        release=lambda: released.append(True),
    )
    released, closed = [], []

    class FailingHands:
        def __init__(self, **_kwargs):
            pass

        def process(self, _frame):
            raise RuntimeError("camera frame failed")

        def close(self):
            closed.append(True)

    cv2 = SimpleNamespace(
        CAP_DSHOW=1, CAP_PROP_FRAME_WIDTH=2, CAP_PROP_FRAME_HEIGHT=3,
        COLOR_BGR2RGB=4, VideoCapture=lambda *_: camera,
        flip=lambda frame, _axis: frame,
        cvtColor=lambda frame, _mode: frame,
    )
    mp = SimpleNamespace(solutions=SimpleNamespace(
        hands=SimpleNamespace(Hands=FailingHands)))
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setitem(sys.modules, "mediapipe", mp)

    statuses = []
    recognizer = g.GestureRecognizer(on_status=statuses.append)
    recognizer._run()

    assert closed == [True]
    assert released == [True]
    assert recognizer.is_running() is False
    assert "camera frame failed" in recognizer.status()["text"]
    assert recognizer.status()["state"] == "error"
    assert all(status["running"] is False for status in statuses)


def test_start_and_stop_wait_for_the_worker_before_claiming_camera_release(monkeypatch):
    class HeldThread:
        def __init__(self, **_kwargs):
            self.alive = False

        def start(self):
            self.alive = True

        def is_alive(self):
            return self.alive

        def join(self, **_kwargs):
            pass

    monkeypatch.setattr(g.threading, "Thread", HeldThread)
    statuses = []
    recognizer = g.GestureRecognizer(on_status=statuses.append)
    monkeypatch.setattr(recognizer, "probe", lambda: {"available": True})
    assert recognizer.start()["ok"]
    assert recognizer.status()["state"] == "starting"
    assert recognizer.status()["running"] is False
    original_worker = recognizer._thread
    assert recognizer.start()["already"]
    assert recognizer._thread is original_worker

    assert recognizer.stop()["stopping"]
    assert recognizer.status()["state"] == "stopping"
    assert "released" not in recognizer.status()["text"]
    assert "error" in recognizer.start()
    assert recognizer._thread is original_worker

    original_worker.alive = False
    assert recognizer.stop()["ok"]
    assert recognizer.status()["state"] == "off"
    assert "released" in recognizer.status()["text"]
    assert [s["state"] for s in statuses][:2] == ["starting", "stopping"]


@pytest.mark.parametrize("complexity", [0, 1])
def test_first_successful_frame_marks_running_and_shutdown_releases_before_off(monkeypatch, complexity):
    lifecycle = []
    recognizer = g.GestureRecognizer(model_complexity=complexity)

    class Hands:
        def __init__(self, **kwargs):
            assert kwargs["model_complexity"] == complexity
            assert kwargs["max_num_hands"] == 1

        def process(self, _frame):
            assert recognizer.status()["state"] == "starting"
            assert recognizer.is_running() is False
            return SimpleNamespace(multi_hand_landmarks=None)

        def close(self):
            lifecycle.append("hands closed")

    camera = SimpleNamespace(
        isOpened=lambda: True, set=lambda *_: None,
        read=lambda: (True, object()),
        release=lambda: lifecycle.append("camera released"),
    )
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        CAP_DSHOW=1, CAP_PROP_FRAME_WIDTH=2, CAP_PROP_FRAME_HEIGHT=3,
        COLOR_BGR2RGB=4, VideoCapture=lambda *_: camera,
        flip=lambda frame, _axis: frame, cvtColor=lambda frame, _mode: frame))
    monkeypatch.setitem(sys.modules, "mediapipe", SimpleNamespace(
        solutions=SimpleNamespace(hands=SimpleNamespace(Hands=Hands))))

    def report(status):
        lifecycle.append(status["state"])
        if status["state"] == "running":
            recognizer._stop.set()

    recognizer.on_status = report
    recognizer._state = "starting"
    recognizer._run()
    assert lifecycle == ["running", "hands closed", "camera released", "off"]
    assert recognizer.status()["running"] is False


def test_camera_that_opens_but_has_no_frames_reports_error_and_releases(monkeypatch):
    released = []
    camera = SimpleNamespace(
        isOpened=lambda: True, set=lambda *_: None,
        read=lambda: (False, None), release=lambda: released.append(True))
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        CAP_DSHOW=1, CAP_PROP_FRAME_WIDTH=2, CAP_PROP_FRAME_HEIGHT=3,
        VideoCapture=lambda *_: camera))
    monkeypatch.setitem(sys.modules, "mediapipe", SimpleNamespace(
        solutions=SimpleNamespace(hands=SimpleNamespace(
            Hands=lambda **_: SimpleNamespace(close=lambda: None)))))
    monkeypatch.setattr(g.time, "sleep", lambda _: None)
    statuses = []
    recognizer = g.GestureRecognizer(on_status=statuses.append)
    recognizer._state = "starting"
    recognizer._run()
    assert recognizer.status()["state"] == "error"
    assert "not providing video frames" in recognizer.status()["text"]
    assert all(s["running"] is False for s in statuses)
    assert released == [True]


def test_first_failed_camera_read_cancels_contacts_before_waiting_or_aborting(monkeypatch):
    motions, released, sleeps = [], [], []
    recognizer = g.GestureRecognizer(on_motion=motions.append)
    recognizer._handle(_hand(wrist=(0.5, 0.8)), 0.0)
    recognizer._handle(_hand(wrist=(0.5, 0.8)), 0.30)
    recognizer._handle(_hand(wrist=(0.56, 0.8)), 0.5)
    recognizer._preview_snapshot = {"image": "previous camera frame"}
    recognizer._diagnostic["landmarks"] = [[0.5, 0.5, 0.0]] * 21
    assert motions[-1]["phase"] == "begin"
    camera = SimpleNamespace(
        isOpened=lambda: True, set=lambda *_: None,
        read=lambda: (False, None), release=lambda: released.append(True))
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        CAP_DSHOW=1, CAP_PROP_FRAME_WIDTH=2, CAP_PROP_FRAME_HEIGHT=3,
        VideoCapture=lambda *_: camera))
    monkeypatch.setitem(sys.modules, "mediapipe", SimpleNamespace(
        solutions=SimpleNamespace(hands=SimpleNamespace(
            Hands=lambda **_: SimpleNamespace(close=lambda: None)))))
    monkeypatch.setattr(g.time, "monotonic", lambda: 0.6)

    def first_sleep(seconds):
        assert motions[-1]["phase"] == "cancel"
        assert motions[-1]["reason"] == "camera_read_failed"
        assert recognizer._preview_snapshot is None
        assert recognizer._diagnostic["landmarks"] == []
        sleeps.append(seconds)
        recognizer._stop.set()

    monkeypatch.setattr(g.time, "sleep", first_sleep)
    recognizer._run()
    assert sleeps == [0.05]
    assert [record["phase"] for record in motions] == ["begin", "cancel"]
    assert released == [True]


# Optional GPU trackers keep the same mirrored, normalised landmark contract.

GPU_TRACKER_NAMES = {"rtmpose": "RTMPose Hand5 · GPU", "wilor": "WiLoR + AnyHand · GPU"}


def _gpu_tracker_fixture(monkeypatch, recognizer, process=None, create=None):
    lifecycle = []
    original, mirrored, rgb = object(), object(), object()

    class Tracker:
        def process(self, image):
            assert image is rgb
            lifecycle.append("inference")
            return process(image) if process else _hand()

        def close(self):
            lifecycle.append("tracker closed")

    def factory(stop_event):
        assert stop_event is recognizer._stop
        lifecycle.append("tracker created")
        return create(Tracker()) if create else Tracker()

    def selected_factory(backend, stop_event):
        assert recognizer.tracker_backend == backend
        return factory(stop_event)

    camera = SimpleNamespace(
        isOpened=lambda: True, set=lambda *_: None,
        read=lambda: (True, original),
        release=lambda: lifecycle.append("camera released"),
    )

    def open_camera(*_):
        lifecycle.append("camera opened")
        return camera

    def flip(frame, axis):
        assert frame is original and axis == 1
        return mirrored

    def convert(frame, color):
        assert frame is mirrored and color == 4
        return rgb

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        CAP_DSHOW=1, CAP_PROP_FRAME_WIDTH=2, CAP_PROP_FRAME_HEIGHT=3,
        COLOR_BGR2RGB=4, VideoCapture=open_camera, flip=flip, cvtColor=convert))
    monkeypatch.setitem(sys.modules, "mediapipe", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "core.hand_tracking", SimpleNamespace(
        TRACKER_NAMES=GPU_TRACKER_NAMES,
        TrackerStartupError=TrackerStartupError,
        probe_rtmpose=lambda: {"available": True, "reason": "GPU tracker ready"},
        probe_wilor=lambda: {"available": True, "reason": "GPU tracker ready"},
        create_rtmpose=lambda stop: selected_factory("rtmpose", stop),
        create_wilor=lambda stop: selected_factory("wilor", stop)))
    return lifecycle, mirrored


@pytest.mark.parametrize("value", [None, True, 1, "", "auto", "RTMPose", [], {}])
def test_tracker_backend_rejects_unknown_selections(value):
    with pytest.raises(ValueError, match="tracker_backend"):
        g.GestureRecognizer(tracker_backend=value)


@pytest.mark.parametrize("backend", ["rtmpose", "wilor"])
def test_tracker_backend_selection_is_atomic_and_preserves_mediapipe_model(monkeypatch, backend):
    recognizer = g.GestureRecognizer(model_complexity=0)
    _gpu_tracker_fixture(monkeypatch, recognizer)
    assert recognizer.configure(tracker_backend=backend) == {
        "ok": True, "travel_palms": 1.2, "tracker_backend": backend,
        "model": GPU_TRACKER_NAMES[backend]}
    assert recognizer.model_complexity == 0
    assert recognizer.configure(travel_palms=2.3)["ok"]
    assert recognizer.tracker_backend == backend
    assert "error" in recognizer.configure(tracker_backend="wrong", travel_palms=1.2)
    assert recognizer.measured.travel_palms == 2.3
    assert recognizer.tracker_backend == backend
    assert "error" in recognizer.configure(tracker_backend="mediapipe", model_complexity=True)
    assert recognizer.tracker_backend == backend
    assert recognizer.configure(tracker_backend="mediapipe")["model"] == "MediaPipe Hands lite"


@pytest.mark.parametrize("backend", ["rtmpose", "wilor"])
def test_tracker_change_requires_mouse_input_to_release_successfully(monkeypatch, backend):
    mouse = _FakeMouseController()
    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse)
    _gpu_tracker_fixture(monkeypatch, recognizer)
    recognizer._handle(_hand(True, False, False, False), 0.0)
    mouse.reset = lambda _: {"error": "left button is still held"}
    result = recognizer.configure(tracker_backend=backend, travel_palms=2.5)
    assert "left button is still held" in result["error"]
    assert recognizer.tracker_backend == "mediapipe"
    assert recognizer.measured.travel_palms == 1.2
    mouse.reset = lambda _: {"state": "idle"}
    assert recognizer.configure(tracker_backend=backend)["ok"]


@pytest.mark.parametrize("state", ["starting", "running", "stopping", "live_worker"])
@pytest.mark.parametrize("backend", ["rtmpose", "wilor"])
def test_tracker_cannot_change_until_capture_worker_has_stopped(monkeypatch, backend, state):
    recognizer = g.GestureRecognizer(tracker_backend=backend)
    _gpu_tracker_fixture(monkeypatch, recognizer)
    recognizer._state = state
    recognizer._running = state == "running"
    if state == "live_worker":
        recognizer._thread = SimpleNamespace(is_alive=lambda: True)
    assert "error" in recognizer.configure(tracker_backend="mediapipe")
    assert recognizer.tracker_backend == backend


@pytest.mark.parametrize("backend", ["rtmpose", "wilor"])
def test_gpu_tracker_probe_status_and_preview_do_not_require_mediapipe(monkeypatch, backend):
    recognizer = g.GestureRecognizer(tracker_backend=backend)
    lifecycle, _ = _gpu_tracker_fixture(monkeypatch, recognizer)
    assert recognizer.probe() == {"available": True, "text": "GPU tracker ready"}
    status = recognizer.status()
    preview = recognizer.preview()
    for info in (status, preview):
        assert info["tracker_backend"] == backend
        assert info["model_name"] == GPU_TRACKER_NAMES[backend]
        assert info["model_complexity"] == 1
    assert lifecycle == []  # Neither model loading nor camera access in status.


@pytest.mark.parametrize("raises", [False, True])
@pytest.mark.parametrize("backend", ["rtmpose", "wilor"])
def test_gpu_tracker_unavailability_is_reported_before_start(monkeypatch, backend, raises):
    recognizer = g.GestureRecognizer(tracker_backend=backend)
    lifecycle, _ = _gpu_tracker_fixture(monkeypatch, recognizer)

    def probe():
        if raises:
            raise RuntimeError("GPU runtime is missing")
        return {"available": False, "reason": "GPU runtime is missing"}

    monkeypatch.setattr(sys.modules["core.hand_tracking"], f"probe_{backend}", probe)
    assert recognizer.probe()["available"] is False
    assert "GPU runtime is missing" in recognizer.start()["error"]
    assert recognizer.status()["state"] == "unavailable"
    assert recognizer._thread is None
    assert lifecycle == []


@pytest.mark.parametrize("backend", ["rtmpose", "wilor"])
def test_gpu_tracker_keeps_landmarks_aligned_with_mirrored_preview(monkeypatch, backend):
    recognizer = g.GestureRecognizer(tracker_backend=backend)
    lifecycle, mirrored = _gpu_tracker_fixture(monkeypatch, recognizer)
    handled, previews = [], []
    monkeypatch.setattr(recognizer, "_handle", lambda points, now: handled.append(points))

    def publish(frame, points, *_):
        previews.append((frame, points))
        recognizer._stop.set()

    monkeypatch.setattr(recognizer, "_publish_preview", publish)
    recognizer._run()
    assert handled == [_hand()]
    assert previews == [(mirrored, _hand())]
    assert lifecycle == ["tracker created", "camera opened", "inference",
                         "tracker closed", "camera released"]
    assert recognizer.status()["state"] == "off"


@pytest.mark.parametrize("output", [None, [], [(0.5, 0.5, 0)] * 20,
                                    [(0.5, 0.5)] * 21, [(float("nan"), 0.5, 0)] * 21,
                                    [(0.5, float("inf"), 0)] * 21,
                                    [(1.3, 0.5, 0)] * 21, [(0.5, 0.5, "bad")] * 21])
@pytest.mark.parametrize("backend", ["rtmpose", "wilor"])
def test_invalid_gpu_landmarks_become_hand_loss(monkeypatch, backend, output):
    recognizer = g.GestureRecognizer(tracker_backend=backend, input_mode="mouse",
                                     mouse_factory=_FakeMouseController)
    recognizer._handle(_hand(True, False, False, False), 0.0)
    lifecycle, _ = _gpu_tracker_fixture(monkeypatch, recognizer, process=lambda _: output)
    previews = []

    def publish(frame, points, *_):
        previews.append(points)
        recognizer._stop.set()

    monkeypatch.setattr(recognizer, "_publish_preview", publish)
    recognizer._run()
    assert previews == [[]]
    assert recognizer._mouse.resets == ["camera_stopped"]
    assert len(recognizer._mouse.updates) == 2
    assert recognizer._mouse.updates[-1][0] == []
    assert lifecycle[-2:] == ["tracker closed", "camera released"]


@pytest.mark.parametrize("backend", ["rtmpose", "wilor"])
def test_stop_during_gpu_initialization_never_opens_camera(monkeypatch, backend):
    recognizer = g.GestureRecognizer(tracker_backend=backend)

    def create(tracker):
        recognizer._stop.set()
        return tracker

    lifecycle, _ = _gpu_tracker_fixture(monkeypatch, recognizer, create=create)
    recognizer._run()
    assert lifecycle == ["tracker created", "tracker closed"]
    assert recognizer.status()["state"] == "off"


@pytest.mark.parametrize("backend", ["rtmpose", "wilor"])
def test_stop_during_gpu_inference_discards_late_input_and_preview(monkeypatch, backend):
    recognizer = g.GestureRecognizer(tracker_backend=backend)

    def process(_):
        recognizer._stop.set()
        return _hand()

    lifecycle, _ = _gpu_tracker_fixture(monkeypatch, recognizer, process=process)
    handled, published = [], []
    monkeypatch.setattr(recognizer, "_handle", lambda *_: handled.append(True))
    monkeypatch.setattr(recognizer, "_publish_preview", lambda *_: published.append(True))
    recognizer._run()
    assert handled == []
    assert published == []
    assert lifecycle[-2:] == ["tracker closed", "camera released"]
    assert recognizer.status()["state"] == "off"


@pytest.mark.parametrize("stage", ["initialization", "inference"])
@pytest.mark.parametrize("backend", ["rtmpose", "wilor"])
def test_gpu_tracker_errors_release_input_and_report_failure(monkeypatch, backend, stage):
    recognizer = g.GestureRecognizer(tracker_backend=backend, input_mode="mouse",
                                     mouse_factory=_FakeMouseController)
    recognizer._handle(_hand(True, False, False, False), 0.0)

    def fail(_):
        raise RuntimeError("GPU worker exited")

    lifecycle, _ = _gpu_tracker_fixture(monkeypatch, recognizer,
                                    process=fail if stage == "inference" else None,
                                    create=fail if stage == "initialization" else None)
    recognizer._run()
    assert recognizer._mouse.resets == ["camera_stopped"]
    assert recognizer.status()["state"] == "error"
    assert "GPU worker exited" in recognizer.status()["text"]
    if stage == "initialization":
        assert lifecycle == ["tracker created"]
    else:
        assert lifecycle[-2:] == ["tracker closed", "camera released"]


@pytest.mark.parametrize("backend", ["rtmpose", "wilor"])
def test_gpu_cleanup_failure_is_visible_and_blocks_restart_until_retry_succeeds(monkeypatch, backend):
    recognizer = g.GestureRecognizer(tracker_backend=backend)
    held = []
    attempts = []

    def create(tracker):
        held.append(tracker)

        def close():
            attempts.append(True)
            raise RuntimeError("GPU worker has not exited")

        tracker.close = close
        return tracker

    def process(_):
        recognizer._stop.set()
        return []

    lifecycle, _ = _gpu_tracker_fixture(monkeypatch, recognizer, create=create, process=process)
    recognizer._run()
    assert lifecycle[-1] == "camera released"
    assert recognizer._failed_tracker is held[0]
    status = recognizer.status()
    assert status["state"] == "error" and status["running"] is False
    assert status["tracker_cleanup_pending"] is True
    assert "camera is released" in status["text"]
    assert "GPU worker has not exited" in status["text"]
    assert "cleanup is still pending" in recognizer.start()["error"]
    assert "cleanup is still pending" in recognizer.configure(tracker_backend="mediapipe")["error"]
    assert recognizer._thread is None
    assert recognizer.tracker_backend == backend
    result = recognizer.stop()
    assert result["camera_released"] is True
    assert result["tracker_cleanup_pending"] is True
    assert "GPU worker has not exited" in result["error"]
    assert len(attempts) == 2
    held[0].close = lambda: attempts.append(True)
    assert recognizer.stop() == {"ok": True}
    assert len(attempts) == 3
    assert recognizer._failed_tracker is None and recognizer._tracker_error is None
    assert recognizer.status()["state"] == "off"
    assert recognizer.status()["tracker_cleanup_pending"] is False
    assert recognizer.configure(tracker_backend="mediapipe")["ok"]


@pytest.mark.parametrize("backend", ["rtmpose", "wilor"])
def test_gpu_cleanup_is_not_retried_while_capture_worker_still_owns_it(monkeypatch, backend):
    recognizer = g.GestureRecognizer(tracker_backend=backend)
    _gpu_tracker_fixture(monkeypatch, recognizer)
    attempts = []
    recognizer._failed_tracker = SimpleNamespace(close=lambda: attempts.append(True))
    recognizer._tracker_error = "close failed"
    alive = [True]
    recognizer._thread = SimpleNamespace(is_alive=lambda: alive[0], join=lambda **_: None)
    result = recognizer.stop()
    assert result["stopping"] is True
    assert "camera_released" not in result
    assert attempts == []
    alive[0] = False
    assert recognizer.stop()["ok"]
    assert attempts == [True]


@pytest.mark.parametrize("backend", ["rtmpose", "wilor"])
def test_gpu_cleanup_retry_blocks_new_capture_during_close(monkeypatch, backend):
    recognizer = g.GestureRecognizer(tracker_backend=backend)
    _gpu_tracker_fixture(monkeypatch, recognizer)
    attempts = []

    def close():
        attempts.append(recognizer.start())
        attempts.append(recognizer.configure(tracker_backend="mediapipe"))

    recognizer._failed_tracker = SimpleNamespace(close=close)
    recognizer._tracker_error = "close failed"
    assert recognizer.stop()["ok"]
    assert all("error" in result for result in attempts)
    assert recognizer._thread is None
    assert recognizer.tracker_backend == backend


@pytest.mark.parametrize("backend", ["rtmpose", "wilor"])
def test_gpu_cleanup_error_is_preserved_alongside_capture_and_mouse_release_errors(monkeypatch, backend):
    mouse = _FakeMouseController()
    recognizer = g.GestureRecognizer(tracker_backend=backend, input_mode="mouse",
                                     mouse_factory=lambda: mouse)
    recognizer._handle(_hand(True, False, False, False), 0.0)
    mouse.reset = lambda _: {"error": "mouse release failed"}

    def create(tracker):
        def close():
            raise RuntimeError("GPU cleanup failed")
        tracker.close = close
        return tracker

    def process(_):
        raise RuntimeError("inference failed")

    lifecycle, _ = _gpu_tracker_fixture(monkeypatch, recognizer, create=create, process=process)
    recognizer._run()
    text = recognizer.status()["text"]
    assert "inference failed" in text
    assert "mouse release failed" in text
    assert "GPU cleanup failed" in text
    assert "camera is released" in text
    assert lifecycle[-1] == "camera released"


@pytest.mark.parametrize("backend", ["rtmpose", "wilor"])
@pytest.mark.parametrize("cleanup_still_fails", [False, True])
def test_failed_constructor_retains_partial_tracker_until_cleanup_succeeds(
        monkeypatch, backend, cleanup_still_fails):
    recognizer = g.GestureRecognizer(tracker_backend=backend, input_mode="mouse",
                                     mouse_factory=_FakeMouseController)
    recognizer._handle(_hand(True, False, False, False), 0.0)
    held, attempts = [], []
    failing = [cleanup_still_fails]

    def create(tracker):
        held.append(tracker)
        def close():
            attempts.append(True)
            if failing[0]:
                raise RuntimeError("GPU process still owns resources")
        tracker.close = close
        raise TrackerStartupError(RuntimeError("GPU initialization failed"),
                                  RuntimeError("Initial worker cleanup failed"), tracker)

    lifecycle, _ = _gpu_tracker_fixture(monkeypatch, recognizer, create=create)
    recognizer._run()
    assert lifecycle == ["tracker created"]  # No camera opened during failed model startup.
    assert recognizer._mouse.resets == ["camera_stopped"]
    assert attempts == [True]  # Capture finally retries the partial worker.
    assert "GPU initialization failed" in recognizer.status()["text"]
    assert recognizer.status()["state"] == "error"
    if cleanup_still_fails:
        assert recognizer._failed_tracker is held[0]
        assert recognizer.status()["tracker_cleanup_pending"] is True
        assert "cleanup is still pending" in recognizer.start()["error"]
        assert "cleanup is still pending" in recognizer.configure(tracker_backend="mediapipe")["error"]
        assert recognizer.stop()["tracker_cleanup_pending"] is True
        assert recognizer._failed_tracker is held[0]
        failing[0] = False
        assert recognizer.stop() == {"ok": True}
        assert len(attempts) == 3
    assert recognizer._failed_tracker is None
    assert recognizer.status()["tracker_cleanup_pending"] is False
