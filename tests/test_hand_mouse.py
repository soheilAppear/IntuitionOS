"""Hand Mouse tests use only fake input devices; they never move the real mouse."""

import ctypes
import math

import pytest

from core.hand_mouse import HandMouseController, _Input, _Point, _WindowsMouse


def hand(pose="point", dx=0.0, dy=0.0):
    points = [[0.5, 0.6, 0.0] for _ in range(21)]
    points[0] = [0.5, 0.8, 0.0]
    points[4] = [0.25, 0.61, 0.0]
    points[5], points[6], points[7], points[8] = (
        [0.44, 0.61, 0], [0.44, 0.48, 0], [0.44, 0.395, 0], [0.44, 0.31, 0])
    for mcp, pip, tip, x in ((9, 10, 12, 0.5), (13, 14, 16, 0.56), (17, 18, 20, 0.62)):
        points[mcp], points[pip], points[tip] = [x, 0.6, 0], [x, 0.45, 0], [x, 0.68, 0]
    if pose == "pinch":
        points[8], points[4] = [0.44, 0.41, 0], [0.45, 0.41, 0]
    elif pose == "fist":
        points[6], points[7] = [0.44, 0.60, 0], [0.44, 0.65, 0]
        points[8], points[4] = [0.44, 0.68, 0], [0.45, 0.68, 0]
    elif pose == "partial":
        points[7], points[8] = [0.49, 0.435, 0], [0.54, 0.43, 0]
    elif pose == "tip_bend":
        points[8] = [0.525, 0.395, 0]
    elif pose == "depth_bend":
        points[8] = [0.44, 0.395, -0.085]
    elif pose == "hook":
        points[7], points[8] = [0.46, 0.46, 0], [0.46, 0.54, 0]
    elif pose == "palm":
        for tip in (12, 16, 20):
            points[tip][1] = 0.3
    return [[x + dx, y + dy, z] for x, y, z in points]


class Device:
    def __init__(self):
        self.events = []
        self.block = None
        self.fail = {}

    def _event(self, name, *args):
        self.events.append((name, *args))
        if self.fail.get(name, 0):
            self.fail[name] -= 1
            raise OSError(f"failed {name}")

    def position(self):
        return 0.5, 0.5

    def blocked(self, *, own_left=False):
        return self.block

    def move(self, x, y):
        self._event("move", x, y)

    def left_down(self):
        self._event("down")

    def left_up(self):
        self._event("up")


@pytest.fixture
def mouse():
    device = Device()
    return HandMouseController(device_factory=lambda: device), device


@pytest.fixture
def bend_mouse(mouse):
    controller, device = mouse
    controller.set_bend_click(True)
    return controller, device


def arm(controller, at=0):
    controller.update(hand(), at)
    controller.update(hand(), at + 0.1)
    return controller.update(hand(), at + 0.21)


def drag(controller):
    arm(controller)
    controller.update(hand("pinch"), 0.25)
    controller.update(hand("pinch"), 0.5)
    return controller.update(hand("pinch"), 0.61)


def buttons(device):
    return [event[0] for event in device.events if event[0] != "move"]


def pinch_distance(distance):
    points = hand("pinch")
    points[4] = [points[8][0] + distance * 0.2, points[8][1], 0]
    return points


def relaxed_pinch(distance=0.05):
    # Thumb meets a curled fingertip near the palm while the proximal index
    # stays raised. Opening the thumb does not straighten the index.
    points = hand("hook")
    points[7], points[8] = [0.47, 0.51, 0], [0.48, 0.61, 0]
    points[4] = [points[8][0] - distance * 0.2, points[8][1], 0]
    return points


def test_constructor_reset_and_unarmed_poses_create_no_device():
    created = []
    controller = HandMouseController(device_factory=lambda: created.append(True))
    controller.reset()
    for pose in ("fist", "pinch", "palm"):
        for index in range(10):
            controller.update(hand(pose), index * 0.1)
    controller.close()
    assert created == []


def test_point_arms_before_moving_and_reports_camera_region(mouse):
    controller, device = mouse
    assert controller.update(hand(), 0)["state"] == "mouse_arming"
    assert controller.update(hand(), 0.1)["progress"] == pytest.approx(0.5)
    assert device.events == []
    result = controller.update(hand(), 0.21)
    assert result["state"] == "mouse_pointer"
    assert result["control_region"] == [0.15, 0.12, 0.85, 0.82]
    assert device.events[0][0] == "move"


def test_point_with_extended_thumb_still_moves(mouse):
    controller, device = mouse
    arm(controller)
    assert controller.update(hand(), 0.25)["state"] == "mouse_pointer"
    assert buttons(device) == []


def test_moving_hand_during_initial_arm_restarts_dwell(mouse):
    controller, device = mouse
    controller.update(hand(), 0)
    controller.update(hand(dx=0.1), 0.1)
    assert controller.update(hand(dx=0.1), 0.21)["state"] == "mouse_arming"
    assert device.events == []
    assert controller.update(hand(dx=0.1), 0.32)["state"] == "mouse_pointer"


def test_deliberate_short_pinch_clicks_once_on_release_without_pointer_jump(mouse):
    controller, device = mouse
    arm(controller)
    before = list(device.events)
    assert controller.update(hand("pinch"), 0.25)["state"] == "mouse_pinch"
    controller.update(hand("pinch"), 0.30)
    assert device.events == before
    assert controller.update(hand(), 0.34)["state"] == "mouse_clicked"
    assert buttons(device) == ["down", "up"]
    controller.update(hand(), 0.40)
    assert buttons(device) == ["down", "up"]


def test_pinch_hysteresis_prevents_chattering(mouse):
    controller, device = mouse
    arm(controller)
    controller.update(hand("pinch"), 0.25)
    mid = hand("pinch")
    mid[4] = [mid[8][0] + 0.09, mid[8][1], 0]  # 0.45 palm: neither start nor release.
    assert controller.update(mid, 0.33)["state"] == "mouse_pinch"
    assert buttons(device) == []
    controller.update(hand(), 0.38)
    assert buttons(device) == ["down", "up"]


def test_curled_index_pinch_clicks_on_thumb_release_without_pointer_jump(mouse):
    controller, device = mouse
    arm(controller)
    before = list(device.events)
    assert controller.update(relaxed_pinch(), 0.25)["state"] == "mouse_pinch"
    assert controller.update(relaxed_pinch(0.45), 0.30)["state"] == "mouse_pinch"
    result = controller.update(relaxed_pinch(0.7), 0.35)
    assert result["click_source"] == "pinch"
    assert device.events == before + [("down",), ("up",)]


def test_pinch_release_does_not_require_a_pointing_pose(mouse):
    controller, device = mouse
    arm(controller)
    controller.update(hand("pinch"), 0.25)
    result = controller.update(hand("hook"), 0.35)
    assert result["click_source"] == "pinch"
    assert buttons(device) == ["down", "up"]


def test_separated_relaxed_fingers_rearm_pinch_without_bend_click(mouse):
    controller, device = mouse
    arm(controller)
    controller.update(hand("pinch"), 0.25)
    assert controller.update(hand(), 0.35)["click_id"] == 1
    for at in (0.40, 0.48, 0.57):
        result = controller.update(hand("hook"), at)
        assert "click_id" not in result
        assert "straighten" not in result["hint"].lower()
    controller.update(hand("pinch"), 0.61)
    assert controller.update(hand("hook"), 0.71)["click_id"] == 2
    assert buttons(device) == ["down", "up", "down", "up"]


def test_brief_separation_does_not_rearm_a_relaxed_pinch(mouse):
    controller, device = mouse
    arm(controller)
    controller.update(relaxed_pinch(), 0.25)
    assert controller.update(relaxed_pinch(0.7), 0.35)["click_id"] == 1
    for at, distance in ((0.39, 0.7), (0.44, 0.05), (0.50, 0.7),
                         (0.58, 0.05), (0.68, 0.7)):
        assert "click_id" not in controller.update(relaxed_pinch(distance), at)
    assert buttons(device) == ["down", "up"]


def test_curled_pinch_drag_releases_without_a_point_or_extra_click(mouse):
    controller, device = mouse
    arm(controller)
    controller.update(relaxed_pinch(), 0.25)
    controller.update(relaxed_pinch(), 0.45)
    assert controller.update(relaxed_pinch(), 0.61)["state"] == "mouse_dragging"
    result = controller.update(relaxed_pinch(0.7), 0.70)
    assert result["state"] == "mouse_pointer"
    assert "click_id" not in result
    assert buttons(device) == ["down", "up"]


@pytest.mark.parametrize("held", [False, True])
@pytest.mark.parametrize("thumb_separated", [False, True])
def test_curled_pinch_to_fist_cancels_without_click(mouse, held, thumb_separated):
    controller, device = mouse
    arm(controller)
    controller.update(relaxed_pinch(), 0.25)
    controller.update(relaxed_pinch(), 0.45)
    if held:
        controller.update(relaxed_pinch(), 0.61)
    fist = hand("fist")
    if thumb_separated:
        fist[4] = [0.25, 0.61, 0]
    assert controller.update(fist, 0.70)["state"] == "mouse_paused"
    assert buttons(device) == (["down", "up"] if held else [])


def test_gradual_pinch_keeps_arming_and_clicks_once_after_release(mouse):
    controller, device = mouse
    arm(controller)
    controller.update(pinch_distance(0.7), 0.25)
    frozen = list(device.events)
    assert controller.update(pinch_distance(0.5), 0.30)["state"] == "mouse_pointer"
    assert controller.update(pinch_distance(0.4), 0.35)["state"] == "mouse_pointer"
    assert device.events == frozen
    assert controller.update(pinch_distance(0.3), 0.40)["state"] == "mouse_pinch"
    assert controller.update(pinch_distance(0.45), 0.45)["state"] == "mouse_pinch"
    assert controller.update(pinch_distance(0.6), 0.50)["state"] == "mouse_clicked"
    assert device.events == frozen + [("down",), ("up",)]


def test_gradual_pinch_band_does_not_count_toward_drag_hold(mouse):
    controller, device = mouse
    arm(controller)
    controller.update(pinch_distance(0.7), 0.25)
    for at, distance in ((0.30, 0.5), (0.50, 0.4), (0.70, 0.4)):
        controller.update(pinch_distance(distance), at)
    assert buttons(device) == []
    controller.update(pinch_distance(0.3), 0.75)
    controller.update(pinch_distance(0.3), 0.95)
    assert buttons(device) == []
    assert controller.update(pinch_distance(0.3), 1.11)["state"] == "mouse_dragging"
    assert buttons(device) == ["down"]
    controller.update(pinch_distance(0.6), 1.20)
    assert buttons(device) == ["down", "up"]


def test_one_frame_accidental_pinch_does_not_click(mouse):
    controller, device = mouse
    arm(controller)
    controller.update(hand("pinch"), 0.25)
    controller.update(hand(), 0.28)
    assert buttons(device) == []


def test_release_crossing_drag_threshold_completes_click_when_drag_never_started(mouse):
    controller, device = mouse
    arm(controller)
    controller.update(hand("pinch"), 0.25)
    controller.update(hand("pinch"), 0.55)
    assert buttons(device) == []
    assert controller.update(hand(), 0.62)["state"] == "mouse_clicked"
    assert buttons(device) == ["down", "up"]


@pytest.mark.parametrize("pose", ["fist"])
def test_pinch_to_cancel_pose_does_not_click(mouse, pose):
    controller, device = mouse
    arm(controller)
    controller.update(hand("pinch"), 0.25)
    assert controller.update(hand(pose), 0.34)["state"] == "mouse_paused"
    controller.update(hand(), 0.4)
    assert buttons(device) == []


def test_pinch_held_during_mode_entry_cannot_click_until_point_rearms(mouse):
    controller, device = mouse
    controller.reset("mode_changed")
    for index in range(6):
        controller.update(hand("pinch"), index * 0.1)
    controller.update(hand(), 0.6)
    controller.update(hand("pinch"), 0.65)
    controller.update(hand(), 0.75)
    assert device.events == []


def test_held_pinch_starts_drag_once_and_uses_wrist_motion(mouse):
    controller, device = mouse
    assert drag(controller)["state"] == "mouse_dragging"
    assert buttons(device) == ["down"]
    before = device.events[-1]
    controller.update(hand("pinch", dx=0.07, dy=0.035), 0.68)
    after = device.events[-1]
    assert after[1] > before[1]
    assert after[2] > before[2]
    # Finger movement with a still wrist cannot move a held drag.
    changed = hand("pinch", dx=0.07, dy=0.035)
    changed[8][1] -= 0.02
    changed[4][1] -= 0.02
    controller.update(changed, 0.75)
    assert buttons(device) == ["down"]
    assert controller.update(hand(dx=0.07, dy=0.035), 0.82)["state"] == "mouse_pointer"
    assert buttons(device) == ["down", "up"]


@pytest.mark.parametrize("reason", ["stop", "mode_changed", "camera_error"])
def test_reset_releases_drag_once_and_is_reusable(mouse, reason):
    controller, device = mouse
    drag(controller)
    controller.reset(reason)
    controller.reset(reason)
    assert buttons(device) == ["down", "up"]
    assert arm(controller, at=1)["state"] == "mouse_pointer"


@pytest.mark.parametrize("points,now,state", [
    (None, 0.7, "mouse_recovering"), ([], 0.7, "mouse_recovering"),
    (hand("fist"), 0.7, "mouse_paused"), (hand("palm"), 0.7, "mouse_pointer"),
    (hand("pinch"), 1.2, "mouse_paused"), (hand("pinch"), 0.5, "mouse_paused"),
    (hand("pinch", dx=0.5), 0.7, "mouse_paused"), (hand("pinch"), float("nan"), "mouse_paused"),
])
def test_tracking_interruptions_release_drag_without_replaying(mouse, points, now, state):
    controller, device = mouse
    drag(controller)
    result = controller.update(points, now)
    assert result["state"] == state
    assert buttons(device) == ["down", "up"]
    controller.update(hand("pinch"), 1.3)
    assert buttons(device) == ["down", "up"]


def test_landmark_nan_is_not_sent_to_windows(mouse):
    controller, device = mouse
    drag(controller)
    invalid = hand("pinch")
    invalid[8][0] = float("nan")
    assert controller.update(invalid, 0.7)["state"] == "mouse_recovering"
    assert buttons(device) == ["down", "up"]


def test_existing_physical_input_blocks_mouse_without_releasing_unowned_button(mouse):
    controller, device = mouse
    device.block = "Release physical mouse buttons"
    assert arm(controller)["state"] == "mouse_paused"
    assert device.events == []


def test_modifier_during_drag_still_releases_our_button(mouse):
    controller, device = mouse
    drag(controller)
    device.block = "Release modifier keys"
    assert controller.update(hand("pinch"), 0.7)["state"] == "mouse_paused"
    assert buttons(device) == ["down", "up"]


@pytest.mark.parametrize("operation", ["down", "move", "up"])
def test_injection_failures_release_ownership_and_latch_error(mouse, operation):
    controller, device = mouse
    if operation == "down":
        arm(controller)
        controller.update(hand("pinch"), 0.25)
        controller.update(hand("pinch"), 0.5)
        device.fail[operation] = 1
        result = controller.update(hand("pinch"), 0.61)
    else:
        drag(controller)
        device.fail[operation] = 1
        result = controller.update(hand() if operation == "up" else hand("pinch"), 0.7)
    assert result["state"] == "mouse_error"
    assert "error" in result
    assert buttons(device)[-1] == "up"
    count = len(device.events)
    assert controller.update(hand(), 0.8)["state"] == "mouse_error"
    assert len(device.events) == count
    controller.reset()
    assert arm(controller, at=1)["state"] == "mouse_pointer"


def test_failed_release_remains_owned_until_cleanup_succeeds(mouse):
    controller, device = mouse
    drag(controller)
    device.fail["up"] = 2
    assert controller.reset()["state"] == "mouse_error"
    assert controller.reset()["state"] == "mouse_error"
    assert controller.reset()["state"] == "mouse_paused"
    assert buttons(device) == ["down", "up", "up", "up"]


def test_feedback_callback_failure_cannot_prevent_release():
    device = Device()
    controller = HandMouseController(lambda result: 1 / 0, device_factory=lambda: device)
    drag(controller)
    controller.close()
    assert buttons(device) == ["down", "up"]


def test_adaptive_smoothing_has_comparable_response_at_different_frame_rates():
    def smooth(fps):
        device = Device()
        controller = HandMouseController(device_factory=lambda: device)
        controller._device = device
        controller._cursor = (0.5, 0.5)
        for frame in range(1, fps + 1):
            controller._move((0.5 + 0.3 * frame / fps, 0.5), 1 / fps)
        return controller._cursor
    assert smooth(30) == pytest.approx(smooth(60), abs=0.004)


def settle_pointer(controller, start=0.25):
    for frame in range(30):
        controller.update(hand(), start + frame / 30)
    return start + 29 / 30


def test_small_finger_bends_keep_tracking_without_target_drift_or_rearming(mouse):
    controller, device = mouse
    arm(controller)
    at = settle_pointer(controller)
    original = device.events[-1][1:]
    for frame in range(1, 25):
        result = controller.update(hand("partial" if frame < 13 else "point"), at + frame / 30)
        assert result["state"] == "mouse_pointer"
        assert "click_id" not in result
        assert device.events[-1][1:] == pytest.approx(original, abs=1e-7)
    assert buttons(device) == []


def test_partially_bent_hand_still_moves_pointer_with_proximal_motion(mouse):
    controller, device = mouse
    arm(controller)
    at = settle_pointer(controller)
    original = device.events[-1][1:]
    for frame in range(1, 16):
        result = controller.update(hand("partial", dx=frame * 0.007, dy=frame * 0.0035), at + frame / 30)
        assert result["state"] == "mouse_pointer"
    moved = device.events[-1][1:]
    assert moved[0] - original[0] == pytest.approx(0.105 / 0.7, abs=0.008)
    assert moved[1] - original[1] == pytest.approx(0.0525 / 0.7, abs=0.008)
    assert buttons(device) == []


def test_hook_clicks_once_after_hold_and_straightening_preserves_target(bend_mouse):
    controller, device = bend_mouse
    arm(controller)
    at = settle_pointer(controller)
    original = device.events[-1][1:]
    assert controller.update(hand("hook"), at + 0.05)["state"] == "mouse_bend"
    assert controller.update(hand("hook"), at + 0.10)["state"] == "mouse_bend"
    result = controller.update(hand("hook"), at + 0.18)
    assert result["state"] == "mouse_clicked"
    assert result["click_id"] == 1 and result["click_source"] == "bend"
    for frame in range(1, 16):
        result = controller.update(hand("hook" if frame < 8 else "point"), at + 0.18 + frame / 30)
        assert "click_id" not in result
        assert device.events[-1][1:] == pytest.approx(original, abs=1e-7)
    assert buttons(device) == ["down", "up"]


def test_hook_entry_without_neutral_point_cannot_click_or_create_device():
    created = []
    controller = HandMouseController(device_factory=lambda: created.append(True))
    for frame in range(12):
        assert controller.update(hand("hook"), frame / 30)["state"] == "mouse_paused"
    assert created == []


def test_hook_hold_requires_continuity_and_click_rearm_requires_stable_straight(bend_mouse):
    controller, device = bend_mouse
    arm(controller)
    controller.update(hand("hook"), 0.25)
    controller.update(hand("partial"), 0.32)
    controller.update(hand("hook"), 0.36)
    assert "click_id" not in controller.update(hand("hook"), 0.43)
    assert controller.update(hand("hook"), 0.50)["click_id"] == 1
    # Brief straight excursions and partial-bend jitter cannot rearm a click.
    for at, pose in ((0.56, "point"), (0.63, "partial"), (0.70, "hook"),
                     (0.80, "point"), (0.87, "partial"), (1.00, "hook")):
        assert "click_id" not in controller.update(hand(pose), at)
    assert buttons(device) == ["down", "up"]
    controller.update(hand(), 1.10)
    controller.update(hand(), 1.27)
    controller.update(hand("hook"), 1.30)
    assert controller.update(hand("hook"), 1.43)["click_id"] == 2
    assert buttons(device) == ["down", "up", "down", "up"]


def test_full_fist_with_thumb_away_pauses_instead_of_bend_click(bend_mouse):
    controller, device = bend_mouse
    arm(controller)
    fist = hand("fist")
    fist[4] = [0.25, 0.61, 0]
    for at in (0.25, 0.35, 0.50):
        assert controller.update(fist, at)["state"] == "mouse_paused"
    assert buttons(device) == []


def test_pinch_takes_priority_over_pending_hook_and_emits_only_pinch_click(bend_mouse):
    controller, device = bend_mouse
    arm(controller)
    controller.update(hand("hook"), 0.25)
    controller.update(hand("pinch"), 0.30)
    result = controller.update(hand(), 0.40)
    assert result["click_source"] == "pinch" and result["click_id"] == 1
    controller.update(hand("hook"), 0.45)
    controller.update(hand("hook"), 0.60)
    assert buttons(device) == ["down", "up"]


def test_completed_hook_cannot_also_click_from_pinch_until_neutral_rearm(bend_mouse):
    controller, device = bend_mouse
    arm(controller)
    controller.update(hand("hook"), 0.25)
    assert controller.update(hand("hook"), 0.38)["click_source"] == "bend"
    for at, pose in ((0.42, "pinch"), (0.55, "pinch"), (0.65, "point")):
        assert "click_id" not in controller.update(hand(pose), at)
    assert buttons(device) == ["down", "up"]


@pytest.mark.parametrize("operation", ["down", "up"])
def test_failed_bend_click_never_emits_success_id(bend_mouse, operation):
    controller, device = bend_mouse
    arm(controller)
    controller.update(hand("hook"), 0.25)
    device.fail[operation] = 1
    result = controller.update(hand("hook"), 0.38)
    assert result["state"] == "mouse_error" and "click_id" not in result
    assert controller._click_id == 0
    assert buttons(device)[-1] == "up"


def test_click_ids_survive_reset_and_exclude_drag_release(bend_mouse):
    controller, device = bend_mouse
    arm(controller)
    controller.update(hand("pinch"), 0.25)
    assert controller.update(hand(), 0.35)["click_id"] == 1
    assert "click_id" not in controller.reset()
    arm(controller, 1)
    controller.update(hand("hook"), 1.25)
    assert controller.update(hand("hook"), 1.38)["click_id"] == 2
    controller.reset()
    arm(controller, 2)
    controller.update(hand("pinch"), 2.25)
    controller.update(hand("pinch"), 2.5)
    controller.update(hand("pinch"), 2.61)
    assert "click_id" not in controller.update(hand(), 2.70)
    assert controller._click_id == 2


def test_adaptive_filter_reduces_stationary_jitter_vs_previous_45ms_filter():
    device = Device()
    controller = HandMouseController(device_factory=lambda: device)
    controller._device, controller._cursor = device, (0.5, 0.5)
    controller._filter_target = (0.5, 0.5)
    old, current_values, old_values = 0.5, [], []
    alpha = 1 - math.exp(-(1 / 30) / 0.045)
    for frame in range(180):
        target = 0.5 + (0.004 if frame % 2 else -0.004)
        old += (target - old) * alpha
        controller._move((target, 0.5), 1 / 30)
        if frame >= 30:
            current_values.append(controller._cursor[0] - 0.5)
            old_values.append(old - 0.5)
    rms = lambda values: math.sqrt(sum(value * value for value in values) / len(values))
    assert rms(current_values) < rms(old_values) * 0.65
    assert rms(current_values) < 0.001


def test_adaptive_filter_tracks_intentional_ramp_faster_than_previous_filter():
    device = Device()
    controller = HandMouseController(device_factory=lambda: device)
    controller._device, controller._cursor = device, (0.2, 0.5)
    controller._filter_target = (0.2, 0.5)
    old, current_lag, old_lag = 0.2, [], []
    alpha = 1 - math.exp(-(1 / 30) / 0.045)
    for frame in range(1, 31):
        target = 0.2 + frame / 30 * 0.5
        old += (target - old) * alpha
        controller._move((target, 0.5), 1 / 30)
        if frame >= 10:
            current_lag.append(target - controller._cursor[0])
            old_lag.append(target - old)
    assert max(current_lag) < 0.008
    assert sum(current_lag) < sum(old_lag) * 0.5


class Win32:
    def __init__(self):
        self.events, self.dpi_calls = [], []
        self.keys = set()
        self.inserted = 1
        self.metrics = {0: 1920, 1: 1080, 76: -1920, 77: -100,
                        78: 3840, 79: 1180}

    def SetThreadDpiAwarenessContext(self, context):
        self.dpi_calls.append(context.value if hasattr(context, "value") else context)
        return 123

    def SendInput(self, count, events, size):
        item = events[0]
        self.events.append((item.mi.dwFlags, item.mi.dx, item.mi.dy, count, size))
        return self.inserted

    def GetAsyncKeyState(self, key):
        return 0x8000 if key in self.keys else 0

    def GetSystemMetrics(self, key):
        return self.metrics[key]

    def GetCursorPos(self, pointer):
        point = ctypes.cast(pointer, ctypes.POINTER(_Point)).contents
        point.x, point.y = 960, 540
        return 1


def test_native_device_construction_is_input_free_and_pointer_safe():
    api = Win32()
    _WindowsMouse(api)
    assert api.events == []
    assert api.dpi_calls == []
    assert ctypes.sizeof(_Input) == (40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28)


@pytest.mark.parametrize("x,y", [(0, 0), (1, 1), (0.5, 0.5)])
def test_native_absolute_input_maps_primary_display_inside_virtual_desktop(x, y):
    api = Win32()
    device = _WindowsMouse(api)
    device.move(x, y)
    flags, nx, ny, count, size = api.events[0]
    assert flags == 0xC001
    assert nx == round((x * 1919 + 1920) / 3839 * 65535)
    assert ny == round((y * 1079 + 100) / 1179 * 65535)
    assert count == 1 and size == ctypes.sizeof(_Input)
    assert api.dpi_calls[-1] == 123


def test_native_position_restores_thread_dpi_context():
    api = Win32()
    device = _WindowsMouse(api)
    assert device.position() == pytest.approx((960 / 1919, 540 / 1079))
    assert api.dpi_calls[-1] == 123


def test_native_failed_move_restores_dpi_context_and_reports_error():
    api = Win32()
    api.inserted = 0
    with pytest.raises(OSError, match="rejected"):
        _WindowsMouse(api).move(0.5, 0.5)
    assert api.dpi_calls[-1] == 123


@pytest.mark.parametrize("key", [0x10, 0x11, 0x12, 0x5B, 0x5C, 1, 2, 4, 5, 6])
def test_native_modifier_and_mouse_guards(key):
    api = Win32()
    api.keys.add(key)
    assert _WindowsMouse(api).blocked()
    assert api.events == []


def test_native_our_button_is_excluded_from_drag_guard_and_up_has_no_guard():
    api = Win32()
    device = _WindowsMouse(api)
    api.keys.add(1)
    assert device.blocked(own_left=True) is None
    api.keys.add(0x11)
    assert device.blocked(own_left=True)
    device.left_up()
    assert api.events[0][0] == 0x0004


@pytest.mark.parametrize("pose", ["tip_bend", "depth_bend"])
def test_distal_finger_bend_clicks_while_tip_remains_beyond_proximal_joint(bend_mouse, pose):
    controller, device = bend_mouse
    arm(controller)
    # A bent fingertip can remain above PIP: the former extension-only check
    # regarded these poses as straight, even with a sustained bend.
    assert controller.update(hand(pose), 0.25)["state"] == "mouse_bend"
    result = controller.update(hand(pose), 0.38)
    assert result["state"] == "mouse_clicked"
    assert result["click_source"] == "bend"
    assert buttons(device) == ["down", "up"]
    for frame in range(1, 25):
        assert "click_id" not in controller.update(hand(pose), 0.38 + frame / 30)
    assert buttons(device) == ["down", "up"]
    # Distal alignment, as well as projected extension, must return to neutral.
    controller.update(hand(), 1.20)
    controller.update(hand(), 1.37)
    controller.update(hand(pose), 1.40)
    assert controller.update(hand(pose), 1.53)["click_id"] == 2
    assert buttons(device) == ["down", "up", "down", "up"]


@pytest.mark.parametrize("pose", ["tip_bend", "depth_bend"])
def test_distal_bend_requires_point_and_a_continuous_hold(bend_mouse, pose):
    controller, device = bend_mouse
    for frame in range(8):
        controller.update(hand(pose), frame / 30)
    assert device.events == []
    arm(controller, at=0.3)
    controller.update(hand(pose), 0.55)
    controller.update(hand("partial"), 0.60)
    controller.update(hand(pose), 0.65)
    assert "click_id" not in controller.update(hand(pose), 0.72)
    assert controller.update(hand(pose), 0.78)["click_source"] == "bend"
    assert buttons(device) == ["down", "up"]


def test_distal_bend_with_two_coordinate_landmarks_is_supported(bend_mouse):
    controller, device = bend_mouse
    for at in (0.0, 0.1, 0.21):
        controller.update([point[:2] for point in hand()], at)
    controller.update([point[:2] for point in hand("tip_bend")], 0.25)
    assert controller.update([point[:2] for point in hand("tip_bend")], 0.38)["click_source"] == "bend"
    assert buttons(device) == ["down", "up"]


@pytest.mark.parametrize("pose", ["tip_bend", "hook"])
@pytest.mark.parametrize("invalid", ["collapsed_middle", "collapsed_tip", "nan_depth", "huge_depth"])
def test_unreliable_distal_joints_cannot_complete_bend(bend_mouse, invalid, pose):
    controller, device = bend_mouse
    arm(controller)
    points = hand(pose)
    if invalid == "collapsed_middle":
        points[7] = list(points[6])
    elif invalid == "collapsed_tip":
        points[8] = list(points[7])
    elif invalid == "nan_depth":
        points[8][2] = float("nan")
    else:
        points[8][2] = 100.0
    for at in (0.25, 0.38, 0.51):
        assert "click_id" not in controller.update(points, at)
    assert buttons(device) == []


def test_pinch_takes_priority_over_a_distal_bend(bend_mouse):
    controller, device = bend_mouse
    arm(controller)
    points = hand("tip_bend")
    points[4] = [points[8][0] + 0.01, points[8][1], points[8][2]]
    assert controller.update(points, 0.25)["state"] == "mouse_pinch"
    assert controller.update(hand(), 0.38)["click_source"] == "pinch"
    assert buttons(device) == ["down", "up"]


@pytest.mark.parametrize("operation", ["down", "up"])
def test_failed_distal_bend_injection_never_emits_success_id(bend_mouse, operation):
    controller, device = bend_mouse
    arm(controller)
    controller.update(hand("depth_bend"), 0.25)
    device.fail[operation] = 1
    result = controller.update(hand("depth_bend"), 0.38)
    assert result["state"] == "mouse_error" and "click_id" not in result
    assert controller._click_id == 0
    assert buttons(device)[-1] == "up"


def test_relaxed_open_hand_keeps_movement_and_can_pinch_click(mouse):
    controller, device = mouse
    arm(controller)
    for at, dx in ((0.25, 0.01), (0.30, 0.03), (0.35, 0.05)):
        assert controller.update(hand("palm", dx=dx), at)["state"] == "mouse_pointer"
    assert device.events[-1][1] > device.events[0][1]
    pinch = hand("palm", dx=0.05)
    pinch[4] = [pinch[8][0] + 0.01, pinch[8][1], 0]
    assert controller.update(pinch, 0.40)["state"] == "mouse_pinch"
    assert controller.update(hand("palm", dx=0.05), 0.50)["click_source"] == "pinch"
    assert buttons(device) == ["down", "up"]
    assert controller.update(hand("palm", dx=0.06), 0.55)["state"] == "mouse_pointer"


@pytest.mark.parametrize("pose", ["hook", "tip_bend", "depth_bend"])
def test_bending_finger_moves_without_clicking_by_default(mouse, pose):
    controller, device = mouse
    assert controller.bend_click is False
    arm(controller)
    for frame in range(1, 16):
        result = controller.update(hand(pose, dx=frame * 0.005), 0.21 + frame / 30)
        assert result["state"] == "mouse_pointer" and "click_id" not in result
    assert buttons(device) == []
    assert device.events[-1][1] > device.events[0][1]


def test_brief_missing_hand_freezes_then_reanchors_without_pointer_jump(mouse):
    controller, device = mouse
    arm(controller)
    before = list(device.events)
    assert controller.update([], 0.25)["state"] == "mouse_recovering"
    assert controller.update([], 0.30)["state"] == "mouse_recovering"
    assert device.events == before
    assert controller.update(hand("palm", dx=0.03), 0.34)["state"] == "mouse_pointer"
    assert device.events == before  # First recovered frame must never move/click.
    assert controller.update(hand("palm", dx=0.03), 0.38)["state"] == "mouse_pointer"
    assert device.events[-1][1:] == pytest.approx(before[-1][1:])
    controller.update(hand("palm", dx=0.05), 0.42)
    assert device.events[-1][1] > before[-1][1]
    assert buttons(device) == []


def test_ambiguous_index_frame_does_not_require_rearming(mouse):
    controller, device = mouse
    arm(controller)
    before = list(device.events)
    ambiguous = hand()
    ambiguous[6] = list(ambiguous[5])  # Collapsed proximal anchor is uncertain.
    assert controller.update(ambiguous, 0.25)["state"] == "mouse_recovering"
    assert controller.update(hand(), 0.30)["state"] == "mouse_pointer"
    assert device.events == before
    assert controller.update(hand(dx=0.01), 0.34)["state"] == "mouse_pointer"


@pytest.mark.parametrize("recovered", [False, True])
def test_repeated_missing_frames_cannot_extend_grace(mouse, recovered):
    controller, device = mouse
    arm(controller)
    before = list(device.events)
    controller.update([], 0.25)
    controller.update([], 0.32)
    result = controller.update(hand() if recovered else [], 0.41)
    assert result["state"] == "mouse_paused"
    assert controller.update(hand("palm"), 0.45)["state"] == "mouse_paused"
    assert device.events == before
    assert arm(controller, at=0.50)["state"] == "mouse_pointer"


def test_delayed_first_missing_sample_counts_toward_recovery_deadline(mouse):
    controller, device = mouse
    arm(controller)  # Last reliable sample at 0.21.
    before = list(device.events)
    assert controller.update([], 0.34)["state"] == "mouse_recovering"
    assert controller.update(hand("palm"), 0.45)["state"] == "mouse_paused"
    assert device.events == before


def test_first_missing_sample_after_grace_disarms_immediately(mouse):
    controller, device = mouse
    arm(controller)
    before = list(device.events)
    assert controller.update([], 0.40)["state"] == "mouse_paused"
    assert device.events == before


@pytest.mark.parametrize("source", ["pinch", "bend", "drag"])
def test_loss_cancels_gesture_and_requires_fresh_neutral_before_next_click(mouse, source):
    controller, device = mouse
    if source == "bend":
        controller.set_bend_click(True)
    if source == "drag":
        drag(controller)
        loss_at = 0.65
        expected_buttons = ["down", "up"]
    else:
        arm(controller)
        controller.update(hand("hook" if source == "bend" else "pinch"), 0.25)
        loss_at = 0.30
        expected_buttons = []
    assert controller.update([], loss_at)["state"] == "mouse_recovering"
    assert buttons(device) == expected_buttons
    # A released or still-pinched recovery frame must not finish the old hold.
    assert "click_id" not in controller.update(hand(), loss_at + 0.04)
    assert "click_id" not in controller.update(hand("pinch"), loss_at + 0.08)
    assert buttons(device) == expected_buttons
    controller.update(hand(), loss_at + 0.12)
    controller.update(hand(), loss_at + 0.28)
    controller.update(hand("pinch"), loss_at + 0.32)
    assert controller.update(hand(), loss_at + 0.42)["click_source"] == "pinch"
    assert buttons(device) == expected_buttons + ["down", "up"]


@pytest.mark.parametrize("guard", ["modifier", "physical_button", "fist", "stop", "reverse_time", "new_position"])
def test_explicit_stops_and_discontinuities_override_recovery(mouse, guard):
    controller, device = mouse
    arm(controller)
    controller.update([], 0.25)
    before = list(device.events)
    if guard in ("modifier", "physical_button"):
        device.block = guard
        result = controller.update([], 0.28)
    elif guard == "stop":
        result = controller.reset("Camera off")
    elif guard == "reverse_time":
        controller.update([], 0.30)
        result = controller.update(hand(), 0.28)
    else:
        points = hand("fist") if guard == "fist" else hand(dx=0.15)
        result = controller.update(points, 0.28)
    assert result["state"] == "mouse_paused"
    assert device.events == before
    device.block = None
    assert controller.update(hand("palm"), 0.34)["state"] == "mouse_paused"


@pytest.mark.parametrize("value", [1, "false", None, []])
def test_bend_click_setting_requires_a_boolean(mouse, value):
    controller, _ = mouse
    with pytest.raises(ValueError, match="boolean"):
        controller.set_bend_click(value)
    with pytest.raises(ValueError, match="boolean"):
        HandMouseController(bend_click=value)


def test_bend_click_setting_releases_drag_and_commits_only_after_cleanup(mouse):
    controller, device = mouse
    drag(controller)
    device.fail["up"] = 1
    assert controller.set_bend_click(True)["state"] == "mouse_error"
    assert controller.bend_click is False
    assert controller.set_bend_click(True)["state"] == "mouse_paused"
    assert controller.bend_click is True
    assert buttons(device) == ["down", "up", "up"]
    assert controller.update(hand("pinch"), 1.0)["state"] == "mouse_paused"
