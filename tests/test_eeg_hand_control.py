"""Combined navigation replays use fake mouse/touchpad devices, never OS input."""

from types import SimpleNamespace

import pytest

from core.actions import register_os_capabilities
from core.desktop import DesktopSwipeController
from core.eeg_hand_control import EegHandController


def hand(x=0.5, y=0.7, fingers=4):
    points = [(x, y, 0.0)] * 21
    points[5], points[9], points[17] = (x - 0.05, y - 0.18, 0), (x, y - 0.2, 0), (x + 0.08, y - 0.16, 0)
    for index, (tip, pip, offset) in enumerate(((8, 6, -0.05), (12, 10, 0),
                                              (16, 14, 0.05), (20, 18, 0.09))):
        points[pip] = (x + offset, y - 0.26, 0.0)
        points[tip] = (x + offset, y - (0.42 if index < fingers else 0.14), 0.0)
        points[tip - 1] = (x + offset, (points[tip][1] + points[pip][1]) / 2, 0.0)
    points[4], points[3] = (x - 0.22, y - 0.16, 0), (x - 0.12, y - 0.1, 0)
    return points


class Mouse:
    eeg_compatible = True
    bend_click = False
    pinching = False
    fail_release = False
    button_down = False
    click_id = 0

    def __init__(self):
        self.calls = []

    def feedback(self, **extra):
        return {"state": "mouse_pointer", "pointer_ready": True,
                "finger_active": False, "finger_action_active": False,
                "button_down": self.button_down, "eeg_compatible": True, **extra}

    def update(self, points, now, *, hold=False):
        self.calls.append(("update", now, hold))
        if self.pinching:
            self.pinching = self.button_down = False
            self.click_id += 1
            return self.feedback(state="mouse_clicked", finger_active=True,
                                 finger_action_active=True, click_id=self.click_id, click_source="pinch")
        return self.feedback()

    def reset(self, reason):
        self.calls.append(("reset", reason))
        if self.fail_release:
            return self.feedback(pointer_ready=False, error="mouse release failed")
        self.button_down = self.pinching = False
        return self.feedback(state="mouse_paused", pointer_ready=False)

    def suspend_tracking(self, reason):
        return self.reset(reason)

    def close(self):
        return self.reset("close")

    def set_bend_click(self, enabled):
        self.bend_click = enabled
        return self.reset("bend preference")


@pytest.fixture
def control():
    register_os_capabilities()
    now, active = [100.0], [True]
    native, dispatches = [], []
    devices = []

    class Touchpad:
        fail_close = False
        closed = False

        def __init__(self):
            devices.append(self)

        def begin(self, axis="horizontal"):
            native.append(("begin", axis))

        def update(self, progress):
            native.append(("update", progress))

        def end(self, cancelled=False):
            native.append(("end", cancelled))

        def close(self):
            native.append(("close",))
            if self.fail_close:
                raise OSError("native release failed")
            self.closed = True

    mouse = Mouse()
    controller = EegHandController(
        mouse, active=lambda: active[0], clock=lambda: now[0],
        dispatch=lambda name, args, **kwargs: dispatches.append((name, args)) or {"ok": True},
        desktop_factory=lambda **kwargs: DesktopSwipeController(native_factory=Touchpad, **kwargs),
    )

    def feed(points=None, dt=1 / 30, **kwargs):
        now[0] += dt
        return controller.update(hand() if points is None else points, now[0], **kwargs)

    context = SimpleNamespace(controller=controller, mouse=mouse, now=now, active=active,
                              native=native, devices=devices, dispatches=dispatches, feed=feed)
    yield context
    for device in devices:
        device.fail_close = False
    mouse.fail_release = False
    controller.close()


def arm(control, fps=30):
    result = None
    for _ in range(int(fps * 0.5) + 1):
        result = control.feed(dt=1 / fps)
    return result


def sweep(control, direction=1, fps=30):
    arm(control, fps)
    for index in range(1, fps + 1):
        result = control.feed(hand(x=0.5 + direction * 0.27 * index / fps), dt=1 / fps)
    for _ in range(fps // 2):
        result = control.feed(hand(x=0.5 + direction * 0.27), dt=1 / fps)
    return result


@pytest.mark.parametrize("fps", [15, 30, 60])
@pytest.mark.parametrize("direction", [-1, 1])
def test_native_swipe_completes_once_and_owns_both_click_sources(control, fps, direction):
    c = control
    result = sweep(c, direction, fps)
    assert [call for call in c.native if call[0] == "begin"] == [("begin", "horizontal")]
    assert [call for call in c.native if call[0] == "end"] == [("end", False)]
    assert c.dispatches == []
    assert not any(call[0] == "update" for call in c.mouse.calls)
    assert result["state"] == "desktop_swipe"
    assert result["finger_active"] and result["finger_action_active"] and result["navigation_active"]
    assert not result["pointer_ready"] and not result["button_down"]
    assert "click_id" not in result
    for _ in range(fps):
        c.feed(hand(x=0.5 + direction * 0.27), dt=1 / fps)
    assert len([call for call in c.native if call[0] == "begin"]) == 1


@pytest.mark.parametrize("direction,expected", [(-1, "right"), (1, "left")])
def test_shortcut_fallback_preserves_camera_only_direction(control, direction, expected):
    c = control
    assert c.controller.configure_navigation(mode="shortcut", travel_palms=1.2)["ok"]
    sweep(c, direction)
    assert c.native == []
    assert c.dispatches == [("os_switch_desktop", {"direction": expected})]


@pytest.mark.parametrize("direction", [-1, 1])
def test_vertical_motion_cancels_without_os_input_or_pointer_fallthrough(control, direction):
    c = control
    arm(c)
    for index in range(1, 31):
        result = c.feed(hand(y=0.7 + direction * 0.27 * index / 30))
    assert result["state"] == "desktop_swipe"
    assert "left/right" in result["hint"]
    assert result["finger_active"]
    assert c.native == c.dispatches == []
    assert not any(call[0] == "update" for call in c.mouse.calls)


def test_short_travel_and_hand_loss_cancel_without_switch(control):
    c = control
    arm(c)
    for index in range(1, 11):
        c.feed(hand(x=0.5 + index * 0.01))
    for _ in range(12):
        c.feed([])
    assert ("end", True) in c.native
    assert ("end", False) not in c.native
    assert c.dispatches == []


def test_completion_requires_neutral_before_another_swipe_or_pointer(control):
    c = control
    sweep(c)
    for _ in range(5):
        result = c.feed(hand(fingers=1))
        assert result["finger_active"]
    assert not any(call[0] == "update" for call in c.mouse.calls)
    for _ in range(5):
        result = c.feed(hand(fingers=1))
    assert not result["navigation_active"]
    assert any(call[0] == "update" for call in c.mouse.calls)


@pytest.mark.parametrize("dragging", [False, True])
def test_pinch_release_keeps_real_click_before_navigation_acquires(control, dragging):
    c = control
    c.mouse.pinching, c.mouse.button_down = True, dragging
    result = c.feed()
    assert result["click_id"] == 1
    assert not result["navigation_active"]
    assert c.mouse.calls[0][0] == "update"
    result = c.feed()
    assert result["navigation_active"]
    assert "click_id" not in result
    assert c.mouse.click_id == 1


@pytest.mark.parametrize("operation", ["reset", "close", "suspend_tracking"])
def test_every_lifecycle_release_cancels_desktop_and_releases_mouse(control, operation):
    c = control
    arm(c)
    for index in range(1, 11):
        c.feed(hand(x=0.5 + index * 0.01))
    result = getattr(c.controller, operation)()
    assert not result.get("error")
    assert ("end", True) in c.native and ("close",) in c.native
    assert c.mouse.calls[-1][0] == "reset"
    assert not c.controller.navigation_active
    assert not c.controller.controls.desktop.status()["active"]


def test_revoking_authorization_cancels_without_finishing_or_pointer_input(control):
    c = control
    arm(c)
    for index in range(1, 11):
        c.feed(hand(x=0.5 + index * 0.01))
    c.active[0] = False
    result = c.feed(hand(x=0.77))
    assert not result.get("error")
    assert ("end", True) in c.native and ("end", False) not in c.native
    assert c.dispatches == []
    assert not any(call[0] == "update" for call in c.mouse.calls)


def test_native_release_failure_blocks_pointer_and_swipes_until_checked_reset(control):
    c = control
    arm(c)
    c.feed(hand(x=0.58))
    c.devices[0].fail_close = True
    result = c.controller.reset()
    assert result["error"] and result["cleanup_pending"]
    assert c.mouse.calls[-1][0] == "reset", "Mouse cleanup must still be attempted."
    call_count = len(c.mouse.calls)
    assert c.feed(hand(fingers=1))["error"]
    assert len(c.mouse.calls) == call_count
    assert c.controller.configure_navigation(mode="shortcut", travel_palms=1.2).get("error")
    c.devices[0].fail_close = False
    assert not c.controller.reset().get("error")
    assert c.feed(hand(fingers=1))["pointer_ready"]


def test_mouse_release_failure_blocks_swipe_and_retains_button_ownership(control):
    c = control
    c.mouse.button_down = c.mouse.fail_release = True
    result = c.feed()
    assert result["error"] and result["button_down"]
    assert c.native == []
    assert c.feed()["button_down"]
    c.mouse.fail_release = False
    assert not c.controller.reset().get("error")


def test_hold_cannot_arm_or_complete_desktop_navigation(control):
    c = control
    for _ in range(20):
        c.feed(hold=True)
    assert not c.controller.navigation_active and c.native == []
    arm(c)
    for index in range(1, 11):
        c.feed(hand(x=0.5 + index * 0.01))
    result = c.feed(hand(x=0.77), hold=True)
    assert result["finger_active"]
    assert ("end", True) in c.native and ("end", False) not in c.native


@pytest.mark.parametrize("travel", [0, -1, True, float("nan"), float("inf")])
def test_navigation_settings_reject_invalid_distance(control, travel):
    with pytest.raises(ValueError):
        control.controller.configure_navigation(mode="auto", travel_palms=travel)


def test_changed_navigation_distance_is_used_for_completion(control):
    c = control
    assert c.controller.configure_navigation(mode="shortcut", travel_palms=2.0)["ok"]
    sweep(c)
    assert c.dispatches == []
    assert c.controller.set_bend_click(True)["navigation_active"] is False
    assert c.controller.bend_click is True


def test_armed_hint_only_offers_horizontal_desktops(control):
    result = arm(control)
    assert result["hint"] == "Move left/right between desktops."


@pytest.mark.parametrize("points", [hand(x=1.5), hand(y=-0.8), [(float("nan"), 0, 0)] * 21,
                                   [(0.5,)] * 21, [None] * 21])
def test_invalid_landmarks_never_acquire_navigation(control, points):
    result = control.feed(points)
    assert not result["navigation_active"]
    assert control.native == control.dispatches == []


@pytest.mark.parametrize("operation", ["reset", "update"])
def test_mouse_exception_latches_uncertain_ownership_until_explicit_cleanup(control, monkeypatch, operation):
    c = control
    original = getattr(c.mouse, operation)

    def fail(*args, **kwargs):
        raise OSError("input API failed")

    monkeypatch.setattr(c.mouse, operation, fail)
    result = c.feed(hand(fingers=4 if operation == "reset" else 1))
    assert result["error"] and result["button_down"] and result["cleanup_pending"]
    assert c.native == []
    monkeypatch.setattr(c.mouse, operation, original)
    assert c.feed()["error"], "A healthy later frame must not clear an input fault."
    assert not c.controller.reset().get("error")
