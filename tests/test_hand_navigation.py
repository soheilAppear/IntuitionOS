"""Replay camera-like traces through recognition, gating, and desktop control.

The clocks, landmarks, mouse, touchpad and action dispatcher are all local fakes.
No camera, keyboard shortcut, mouse movement or desktop switch is performed.
"""

from types import SimpleNamespace

import pytest

from core import gestures as g
from core.actions import register_os_capabilities
from core.desktop import DesktopSwipeController
from core.hand_control import HandControls


def hand(x=0.5, y=0.7, fingers=4):
    points = [(x, y, 0.0)] * 21
    points[g.INDEX_MCP] = (x - 0.05, y - 0.18, 0.0)
    points[g.MIDDLE_MCP] = (x, y - 0.20, 0.0)
    points[g.PINKY_MCP] = (x + 0.08, y - 0.16, 0.0)
    for i, (tip, pip, offset) in enumerate(((8, 6, -0.05), (12, 10, 0),
                                          (16, 14, 0.05), (20, 18, 0.09))):
        points[pip] = (x + offset, y - 0.26, 0.0)
        points[tip] = (x + offset, y - (0.42 if i < fingers else 0.14), 0.0)
        points[tip - 1] = (x + offset, (points[tip][1] + points[pip][1]) / 2, 0.0)
    points[g.THUMB_TIP] = (x - 0.22, y - 0.16, 0.0)
    points[g.THUMB_IP] = (x - 0.12, y - 0.10, 0.0)
    return points


@pytest.fixture
def navigation():
    register_os_capabilities()
    clock = [100.0]
    dispatches, native, pointer, feedback = [], [], [], []

    class Touchpad:
        def begin(self, axis="horizontal"):
            native.append(("begin", axis))

        def update(self, progress):
            native.append(("update", progress))

        def end(self, cancelled=False):
            native.append(("end", cancelled))

        def close(self):
            native.append(("close",))

    mouse = SimpleNamespace(
        set_bend_click=lambda enabled: {},
        update=lambda points, now: pointer.append((points, now)) or {},
        reset=lambda reason: {},
    )
    recognizer = g.GestureRecognizer(input_mode="mouse", mouse_factory=lambda: mouse,
                                     on_progress=feedback.append)
    controls = HandControls(
        active=lambda: recognizer.navigation_active, feedback=feedback.append,
        clock=lambda: clock[0],
        dispatch=lambda name, args, **kwargs: dispatches.append((name, args)) or {"ok": True},
        desktop_factory=lambda **kwargs: DesktopSwipeController(
            native_factory=Touchpad, **kwargs),
    )
    recognizer.on_motion = controls.motion

    def feed(points, dt):
        clock[0] += dt
        recognizer._handle(points, clock[0])

    yield SimpleNamespace(recognizer=recognizer, controls=controls, feed=feed,
                          dispatches=dispatches, native=native, pointer=pointer,
                          feedback=feedback)
    controls.stop()


@pytest.mark.parametrize("fps", [15, 30, 60])
@pytest.mark.parametrize("direction", ["left", "right", "up", "down"])
def test_a_complete_sweep_finishes_once_without_a_fist(navigation, fps, direction):
    n = navigation
    dt = 1 / fps
    for _ in range(int(fps * 0.5) + 1):
        n.feed(hand(), dt)
    dx = {"left": -0.27, "right": 0.27}.get(direction, 0)
    dy = {"up": -0.27, "down": 0.27}.get(direction, 0)
    for i in range(1, fps + 1):
        fraction = i / fps
        n.feed(hand(0.5 + dx * fraction, 0.7 + dy * fraction), dt)
    for _ in range(fps):
        n.feed(hand(0.5 + dx, 0.7 + dy), dt)
    assert n.pointer == [], "Navigation must never leak into pointer movement"
    assert not n.controls.desktop.status()["active"], "Completed sweeps must lift their contacts"
    if direction in ("left", "right"):
        assert [event for event in n.native if event[0] == "begin"] == [("begin", "horizontal")]
        assert [event for event in n.native if event[0] == "end"] == [("end", False)]
        assert n.dispatches == [], "A native action must not also issue a shortcut"
    else:
        assert n.native == [], "Task View must not depend on touchpad settings"
        assert n.dispatches == [("os_desktop_view", {
            "view": "overview" if direction == "up" else "desktop"})]


def test_short_travel_and_hand_loss_never_dispatch_an_action(navigation):
    n = navigation
    for _ in range(16):
        n.feed(hand(), 1 / 30)
    for i in range(1, 11):
        n.feed(hand(0.5 + i * 0.01), 1 / 30)
    for _ in range(10):
        n.feed([], 1 / 30)
    assert not n.controls.desktop.status()["active"]
    assert n.dispatches == []
    assert ("end", True) in n.native


def test_reversing_below_threshold_before_completion_cancels(navigation):
    n = navigation
    for _ in range(16):
        n.feed(hand(), 1 / 30)
    for step in list(range(1, 19)) + list(range(17, -1, -1)):
        n.feed(hand(0.5 + step * 0.01), 1 / 30)
    for _ in range(10):
        n.feed([], 1 / 30)
    assert n.dispatches == []
    assert [event for event in n.native if event[0] == "end"] == [("end", True)]
