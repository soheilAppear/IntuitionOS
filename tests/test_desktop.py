"""Desktop input lifecycle tests use fakes: never switch the user's desktops."""

import ctypes
from queue import Queue
import threading
import time
from types import SimpleNamespace

import pytest

from core import desktop as d


class FakeTouchpad:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    def _call(self, name, value=None):
        self.calls.append((name, value))
        if name == self.fail:
            raise OSError("injection refused")

    def begin(self, axis="horizontal"):
        self._call("begin", axis)

    def update(self, progress):
        self._call("update", progress)

    def end(self, cancelled=False):
        self._call("end", cancelled)

    def close(self):
        self._call("close")


def unavailable():
    raise OSError("API unavailable")


def test_authorization_happens_before_creating_device():
    calls = []
    controller = d.DesktopSwipeController(native_factory=lambda: calls.append("create"))
    assert "error" in controller.begin()
    assert calls == []
    assert not controller.status()["active"]


@pytest.mark.parametrize("permission", [False, None, {"ok": False}, {"verdict": "allow"},
                                         {"ok": True, "error": "denied"}])
def test_truthy_objects_do_not_bypass_authorization(permission):
    controller = d.DesktopSwipeController(lambda: permission,
                                          native_factory=lambda: pytest.fail("OS call"))
    assert "error" in controller.begin()


def test_native_input_tracks_slow_progress_and_reversal_until_release():
    device, auth = FakeTouchpad(), []
    controller = d.DesktopSwipeController(lambda: auth.append(True) or True,
                                          native_factory=lambda: device)
    assert controller.begin()["mode"] == "native"
    assert controller.begin()["ok"]  # repeated begin cannot double-inject.
    for progress in (0.02, 0.10, 0.30, 0.25, -0.10, -0.70):
        assert controller.update(progress)["ok"]
    result = controller.end()
    assert result["status"] == "gesture_released"
    assert auth == [True]
    assert device.calls == [("begin", "horizontal"), ("update", .02), ("update", .10),
                            ("update", .30), ("update", .25), ("update", -.10),
                            ("update", -.70), ("end", True)]
    assert not result["completed"]  # Releasing before full travel eases back.
    assert not controller.end()["completed"]


@pytest.mark.parametrize("progress,direction", [(-1, "right"), (1, "left")])
def test_shortcut_commits_once_on_release_in_trackpad_direction(progress, direction):
    switches = []
    controller = d.DesktopSwipeController(lambda: True, native_factory=unavailable,
        shortcut=lambda value: switches.append(value) or {"ok": True})
    assert controller.begin()["mode"] == "shortcut"
    for _ in range(30):
        controller.update(progress)
    assert not switches
    assert controller.end()["completed"]
    controller.end()
    assert switches == [direction]


def test_reverse_below_threshold_or_cancel_never_commits_shortcut():
    controller = d.DesktopSwipeController(lambda: True, native_factory=unavailable,
                                          shortcut=lambda _: pytest.fail("unexpected shortcut"))
    controller.begin()
    controller.update(1)
    controller.update(.6)
    assert not controller.end()["completed"]
    controller.begin()
    controller.update(-1)
    assert not controller.end(cancelled=True)["completed"]


@pytest.mark.parametrize("failure", ["begin", "update", "end"])
def test_native_failure_never_double_dispatches_a_shortcut(failure):
    device, switches = FakeTouchpad(fail=failure), []
    controller = d.DesktopSwipeController(lambda: True, native_factory=lambda: device,
        shortcut=lambda direction: switches.append(direction) or {"ok": True})
    results = [controller.begin(), controller.update(-1), controller.end()]
    assert any("error" in result for result in results)
    assert switches == []
    assert ("close", None) in device.calls
    assert controller.begin()["mode"] == "shortcut"
    controller.update(-1)
    controller.end()
    assert switches == ["right"]


def test_explicit_shortcut_preference_does_not_create_native_device():
    controller = d.DesktopSwipeController(lambda: True, mode="shortcut",
        native_factory=lambda: pytest.fail("native should remain unused"))
    assert controller.probe()["mode"] == "shortcut"
    assert controller.begin()["mode"] == "shortcut"
    assert "error" in controller.configure(mode="auto")
    controller.end(cancelled=True)
    assert controller.configure(mode="auto")["preference"] == "auto"


def test_probe_creates_and_destroys_without_injecting_or_authorizing():
    device = FakeTouchpad()
    controller = d.DesktopSwipeController(native_factory=lambda: device)
    result = controller.probe()
    assert result["native_available"]
    assert device.calls == [("close", None)]
    assert controller.status()["native_available"]
    assert "error" in controller.begin()


@pytest.mark.parametrize("progress", [float("nan"), float("inf"), None, "bad"])
def test_invalid_progress_cancels_contacts(progress):
    device = FakeTouchpad()
    controller = d.DesktopSwipeController(lambda: True, native_factory=lambda: device)
    controller.begin()
    assert "error" in controller.update(progress)
    assert device.calls[-1] == ("end", True)
    assert not controller.status()["active"]


def test_close_serializes_with_an_inflight_update_and_releases_once():
    started, finish, closed = threading.Event(), threading.Event(), threading.Event()
    device = FakeTouchpad()
    original_update = device.update

    def update(progress):
        started.set()
        assert finish.wait(2)
        original_update(progress)

    device.update = update
    controller = d.DesktopSwipeController(lambda: True, native_factory=lambda: device)
    controller.begin()
    worker = threading.Thread(target=lambda: controller.update(.5))
    stopper = threading.Thread(target=lambda: (controller.close(), closed.set()))
    worker.start()
    assert started.wait(2)
    stopper.start()
    assert not closed.wait(.03)
    finish.set()
    worker.join(2)
    stopper.join(2)
    assert closed.is_set()
    assert device.calls[-3:] == [("update", .5), ("end", True), ("close", None)]
    controller.close()
    assert device.calls.count(("close", None)) == 1
    assert controller.begin()["ok"]  # Camera may restart with the same controller.
    controller.close()


class FakeUser32:
    def __init__(self):
        self.params = None
        self.frames = []
        self.destroyed = []
        self.keys = []
        self.held = False
        self.partial = False
        self.fail_inject = False

    def CreateSyntheticPointerDevice2(self, pointer):
        self.params = d._DeviceParams.from_buffer_copy(
            ctypes.string_at(pointer, ctypes.sizeof(d._DeviceParams)))
        return 123

    def InjectSyntheticPointerInput(self, device, pointers, count):
        assert device == 123
        self.frames.append([(p.type, p.touchInfo.pointerInfo.pointerId,
                             p.touchInfo.pointerInfo.pointerFlags,
                             p.touchInfo.pointerInfo.ptHimetricLocation.x,
                             p.touchInfo.pointerInfo.ptHimetricLocation.y)
                            for p in pointers[:count]])
        return not self.fail_inject

    def DestroySyntheticPointerDevice(self, device):
        self.destroyed.append(device)

    def SendInput(self, count, inputs, size):
        assert size == ctypes.sizeof(d._Input)
        self.keys.append([(v.ki.wVk, v.ki.dwFlags) for v in inputs[:count]])
        return 2 if self.partial else count

    def GetAsyncKeyState(self, key):
        return 0x8000 if self.held else 0


def test_native_frames_use_physical_four_contact_input_and_release_all_contacts():
    api = FakeUser32()
    device = d._WindowsTouchpad(api)
    assert api.frames == []
    assert (api.params.pointerType, api.params.maxCount, api.params.options) == (5, 4, 3)
    device.begin()
    device.update(-.5)
    device.end()
    device.close()
    assert [p[3] for p in api.frames[-2]] == [2900, 3300, 3700, 4100]
    assert len({p[1] for p in api.frames[0]}) == 4
    assert all(p[2] == 0x4006 for p in api.frames[0])
    assert all(p[2] == 0x4000 for p in api.frames[-1])
    assert api.destroyed == [123]


def test_native_cancel_returns_contacts_to_origin_and_destroys_after_failure():
    api = FakeUser32()
    device = d._WindowsTouchpad(api)
    device.begin()
    device.update(.5)
    device.close()
    assert api.frames[-2][0][3] == 4400
    assert all(p[2] == 0xC000 for p in api.frames[-1])
    assert api.destroyed == [123]
    failing = FakeUser32()
    device = d._WindowsTouchpad(failing)
    device.begin()
    failing.fail_inject = True
    with pytest.raises(OSError):
        device.close()
    assert failing.destroyed == [123]


def test_shortcut_uses_a_single_complete_sendinput_chord():
    api = FakeUser32()
    assert d.shortcut_switch("right", user32=api)["ok"]
    assert api.keys == [[(0x5B, 0), (0x11, 0), (0x27, 0),
                         (0x27, 2), (0x11, 2), (0x5B, 2)]]


def test_partial_shortcut_releases_keys_and_reports_failure():
    api = FakeUser32()
    api.partial = True
    assert "error" in d.shortcut_switch("left", user32=api)
    assert api.keys[-1] == [(0x25, 2), (0x11, 2), (0x5B, 2)]


def test_shortcut_refuses_held_modifiers_and_invalid_direction():
    api = FakeUser32()
    api.held = True
    assert "error" in d.shortcut_switch("left", user32=api)
    assert "error" in d.shortcut_switch("down", user32=api)
    assert api.keys == []


def test_vertical_swipe_has_separate_authorization_and_locks_axis():
    device, authorized, requests = FakeTouchpad(), [], []
    controller = d.DesktopSwipeController(
        lambda: pytest.fail("horizontal authorization must not grant vertical input"),
        native_factory=lambda: device)
    assert "error" in controller.begin(axis="vertical")
    assert device.calls == []
    controller = d.DesktopSwipeController(
        lambda: pytest.fail("wrong authorization"),
        authorize_overview=lambda: authorized.append("overview") or {"ok": True},
        native_factory=lambda: pytest.fail("vertical input must not create a native device"),
        overview_shortcut=lambda direction: requests.append(direction) or {"ok": True})
    assert "error" in controller.begin(axis="diagonal")
    assert controller.begin(axis="vertical")["axis"] == "vertical"
    assert controller.begin(axis="vertical")["ok"]
    assert "error" in controller.begin(axis="horizontal")
    assert authorized == ["overview"]
    assert device.calls == []
    controller.update(-1)
    assert controller.end()["completed"]
    assert requests == ["up"]


@pytest.mark.parametrize("progress,direction", [(-1, "up"), (1, "down")])
def test_vertical_shortcut_commits_once_only_at_final_threshold(progress, direction):
    requests = []
    controller = d.DesktopSwipeController(
        authorize_overview=lambda: True, mode="shortcut",
        shortcut=lambda _: pytest.fail("horizontal shortcut used"),
        overview_shortcut=lambda value: requests.append(value) or {"ok": True})
    controller.begin(axis="vertical")
    controller.update(progress)
    controller.update(progress * .99)
    assert not controller.end()["completed"]
    assert requests == []
    controller.begin(axis="vertical")
    controller.update(progress)
    assert requests == []
    assert controller.end()["completed"]
    assert not controller.end()["completed"]
    assert requests == [direction]


def test_native_threshold_can_be_reversed_before_release():
    device = FakeTouchpad()
    controller = d.DesktopSwipeController(lambda: True, authorize_overview=lambda: True,
                                          native_factory=lambda: device)
    controller.begin()
    controller.update(1)
    controller.update(.8)
    assert not controller.end()["completed"]
    assert device.calls[-1] == ("end", True)
    controller.begin()
    controller.update(-1)
    assert controller.end()["completed"]
    assert device.calls[-1] == ("end", False)


@pytest.mark.parametrize("axis,count,coordinate", [("horizontal", 4, 3), ("vertical", 3, 4)])
@pytest.mark.parametrize("progress", [-1, 1])
def test_both_axes_keep_all_contacts_inside_physical_device(axis, count, coordinate, progress):
    api = FakeUser32()
    device = d._WindowsTouchpad(api, sleep=lambda _: None)
    device.begin(axis=axis)
    device.update(progress)
    device.end()
    device.close()
    assert all(len(frame) == count for frame in api.frames)
    assert all(0 <= contact[3] <= 10000 and 0 <= contact[4] <= 6000
               for frame in api.frames for contact in frame)
    origin, final = api.frames[0], api.frames[-1]
    assert (final[0][coordinate] - origin[0][coordinate]) * progress > 0
    other_coordinate = 4 if coordinate == 3 else 3
    assert final[0][other_coordinate] == origin[0][other_coordinate]
    assert all(contact[2] == 0x4000 for contact in final)
    assert api.destroyed == [123]


def wait_until(condition, timeout=1):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting for fake input worker"
        threading.Event().wait(.003)


@pytest.mark.parametrize("elapsed", [1 / 60, .03125])
def test_native_worker_interpolates_camera_targets_and_reverses_without_overshoot(monkeypatch, elapsed):
    # Control elapsed time while exercising the real animation thread. Windows
    # may report the same monotonic tick for consecutive frames; that is valid
    # and must not be mistaken for interpolation failing to advance.
    clock = SimpleNamespace(now=100.0)
    ticks, waiting = Queue(), Queue()

    class SteppedStop(threading.Event):
        def wait(self, timeout=None):
            waiting.put(timeout)
            duration = ticks.get()
            if self.is_set():
                return True
            clock.now += duration
            return False

        def set(self):
            super().set()
            ticks.put(0.0)

    monkeypatch.setattr(d, "time", SimpleNamespace(monotonic=lambda: clock.now))
    api = FakeUser32()
    device = d._WindowsTouchpad(api, sleep=lambda _: None)
    device._stop = SteppedStop()

    def step(duration):
        before = len(api.frames)
        ticks.put(duration)
        waiting.get(timeout=2)  # The next wait acknowledges this frame finished.
        assert len(api.frames) == before + 1
        return api.frames[-1][0][3]

    try:
        device.begin()
        waiting.get(timeout=2)
        device.update(1)
        forward = []
        for _ in range(3):
            forward.append(step(elapsed))
            assert step(0.0) == forward[-1]
        assert 4400 < forward[0] < forward[1] < forward[2] < 7400
        device.update(-1)
        reversed_positions = []
        for _ in range(3):
            reversed_positions.append(step(elapsed))
            assert step(0.0) == reversed_positions[-1]
        assert forward[-1] > reversed_positions[0] > reversed_positions[1] > reversed_positions[2] > 1400
        worker = device._thread
        device.end(cancelled=True)
        assert api.frames[-2][0][3] == 4400
        assert all(contact[2] == 0xC000 for contact in api.frames[-1])
        assert not worker.is_alive(), "no animation survives gesture release"
        assert device._thread is None
    finally:
        device.close()
    assert api.destroyed == [123]


def test_native_worker_releases_stale_camera_input_without_waiting_for_controller():
    api = FakeUser32()
    device = d._WindowsTouchpad(api, sleep=lambda _: None)
    device.begin()
    device.update(.7)
    with device._lock:
        device._last_update -= 1
    wait_until(lambda: api.frames[-1][0][2] == 0xC000)
    with pytest.raises(TimeoutError, match="stale"):
        device.update(.8)
    with pytest.raises(TimeoutError, match="stale"):
        device.close()
    assert api.frames[-1][0][3] == 4400
    assert api.destroyed == [123]


def test_native_close_waits_for_inflight_animation_and_never_destroys_during_injection():
    started, finish, closed = threading.Event(), threading.Event(), threading.Event()

    class BlockingUser32(FakeUser32):
        def InjectSyntheticPointerInput(self, device, pointers, count):
            if self.frames and not started.is_set():
                started.set()
                assert finish.wait(2)
                assert self.destroyed == []
            return super().InjectSyntheticPointerInput(device, pointers, count)

    api = BlockingUser32()
    device = d._WindowsTouchpad(api, sleep=lambda _: None)
    device.begin()
    device.update(.5)
    assert started.wait(1)
    stopper = threading.Thread(target=lambda: (device.close(), closed.set()))
    stopper.start()
    assert not closed.wait(.03)
    finish.set()
    stopper.join(1)
    assert closed.is_set()
    assert api.destroyed == [123]
    assert all(contact[2] == 0xC000 for contact in api.frames[-1])
    device.close()
    assert api.destroyed == [123]


def test_native_animation_failure_releases_and_destroys_without_another_camera_frame():
    api = FakeUser32()
    device = d._WindowsTouchpad(api, sleep=lambda _: None)
    device.begin()
    api.fail_inject = True
    wait_until(lambda: api.destroyed == [123])
    assert all(contact[2] == 0xC000 for contact in api.frames[-1])
    with pytest.raises(OSError, match="injection failed"):
        device.update(.5)
    with pytest.raises(OSError, match="injection failed"):
        device.close()
    assert api.destroyed == [123]


def test_native_close_timeout_leaves_cleanup_with_the_inflight_worker():
    started, finish = threading.Event(), threading.Event()

    class BlockingUser32(FakeUser32):
        def InjectSyntheticPointerInput(self, device, pointers, count):
            if self.frames and not started.is_set():
                started.set()
                assert finish.wait(2)
                assert self.destroyed == []
            return super().InjectSyntheticPointerInput(device, pointers, count)

    api = BlockingUser32()
    device = d._WindowsTouchpad(api, sleep=lambda _: None)
    device.begin()
    assert started.wait(1)
    try:
        with pytest.raises(TimeoutError, match="still releasing"):
            device.close()
        assert api.destroyed == []
    finally:
        finish.set()
    wait_until(lambda: api.destroyed == [123])
    assert all(contact[2] == 0xC000 for contact in api.frames[-1])
    device.close()
    assert api.destroyed == [123]


@pytest.mark.parametrize("direction,key", [("up", 0x09), ("down", 0x44)])
def test_overview_shortcuts_are_one_complete_chord(direction, key):
    api = FakeUser32()
    assert d.shortcut_overview(direction, user32=api)["ok"]
    assert api.keys == [[(0x5B, 0), (key, 0), (key, 2), (0x5B, 2)]]
    api.partial = True
    assert "error" in d.shortcut_overview(direction, user32=api)
    assert api.keys[-1] == [(key, 2), (0x5B, 2)]


def test_overview_shortcut_rejects_held_modifiers_and_invalid_direction():
    api = FakeUser32()
    api.held = True
    assert "error" in d.shortcut_overview("up", user32=api)
    assert "error" in d.shortcut_overview("left", user32=api)
    assert api.keys == []


def test_controller_retains_failed_cleanup_and_blocks_every_new_input_path():
    class PendingTouchpad(FakeTouchpad):
        closed = False
        cleanup_pending = False

        def end(self, cancelled=False):
            self.cleanup_pending = True
            raise TimeoutError("still releasing")

        def close(self):
            self.calls.append(("close", None))
            if not self.closed:
                raise TimeoutError("still releasing")

    device, creates, shortcuts = PendingTouchpad(), [], []
    controller = d.DesktopSwipeController(
        lambda: True, authorize_overview=lambda: True,
        native_factory=lambda: creates.append(True) or device,
        shortcut=lambda direction: shortcuts.append(direction) or {"ok": True})
    controller.begin()
    controller.update(1)
    result = controller.end()
    assert result["cleanup_pending"] and "error" in result
    assert controller._native is device, "pending contacts must retain their owner"
    for _ in range(2):
        assert "error" in controller.end(cancelled=True)
        assert controller.status()["cleanup_pending"]
        assert "error" in controller.begin(axis="horizontal")
        assert "error" in controller.begin(axis="vertical")
        assert "error" in controller.configure(mode="shortcut")
        assert controller.probe()["cleanup_pending"]
        assert "error" in controller.close()
    assert creates == [True]
    assert shortcuts == []
    device.closed = True
    device.cleanup_pending = False
    result = controller.end(cancelled=True)
    assert result["ok"] and not result["cleanup_pending"]
    assert controller._native is None
    assert controller.begin()["mode"] == "shortcut"
    controller.update(-1)
    assert controller.end()["completed"]
    assert shortcuts == ["right"]


def test_controller_reports_pending_until_blocked_native_worker_actually_releases():
    started, finish = threading.Event(), threading.Event()

    class BlockingUser32(FakeUser32):
        def InjectSyntheticPointerInput(self, device, pointers, count):
            if self.frames and not started.is_set():
                started.set()
                assert finish.wait(5)
            return super().InjectSyntheticPointerInput(device, pointers, count)

    api = BlockingUser32()
    device = d._WindowsTouchpad(api, sleep=lambda _: None)
    controller = d.DesktopSwipeController(lambda: True, native_factory=lambda: device,
                                          shortcut=lambda _: pytest.fail("unexpected fallback"))
    controller.begin()
    controller.update(.5)
    assert started.wait(1)
    try:
        result = controller.end(cancelled=True)
        assert "error" in result and result["cleanup_pending"]
        assert "error" in controller.end(cancelled=True)
        assert "error" in controller.begin()
        assert controller.status()["cleanup_pending"]
        assert api.destroyed == []
    finally:
        finish.set()
    wait_until(lambda: device.closed)
    result = controller.end(cancelled=True)
    assert result["ok"] and not result["cleanup_pending"]
    assert api.destroyed == [123]
    assert all(contact[2] == 0xC000 for contact in api.frames[-1])


@pytest.mark.parametrize("cancelled", [False, True])
def test_vertical_shortcuts_never_touch_cached_horizontal_native_device(cancelled):
    device, requests = FakeTouchpad(), []
    controller = d.DesktopSwipeController(
        lambda: True, authorize_overview=lambda: True,
        native_factory=lambda: device,
        overview_shortcut=lambda direction: requests.append(direction) or {"ok": True})
    assert controller.begin()["mode"] == "native"
    controller.update(1)
    assert controller.end()["completed"]
    previous_calls = list(device.calls)
    result = controller.begin(axis="vertical")
    assert result["mode"] == "shortcut" and result["native_available"]
    for progress in (-.2, -.5, -1):
        controller.update(progress)
    assert controller.end(cancelled=cancelled)["completed"] is not cancelled
    assert requests == ([] if cancelled else ["up"])
    assert device.calls == previous_calls, "vertical input must not inject or release cached contacts"
    assert controller.begin()["mode"] == "native"
    assert device.calls[-1] == ("begin", "horizontal")
    controller.close()


@pytest.mark.parametrize("progress", [-1, 1])
@pytest.mark.parametrize("cancelled", [False, True])
def test_native_commit_follows_through_40mm_but_cancel_returns_origin_before_spaced_lift(progress, cancelled):
    events = []

    class TimedUser32(FakeUser32):
        def InjectSyntheticPointerInput(self, device, pointers, count):
            events.append(("frame", pointers[0].touchInfo.pointerInfo.pointerFlags))
            return super().InjectSyntheticPointerInput(device, pointers, count)

    api = TimedUser32()
    device = d._WindowsTouchpad(api, sleep=lambda seconds: events.append(("sleep", seconds)))
    device.begin()
    device.update(progress)
    device.end(cancelled=cancelled)
    device.close()
    expected_x = 4400 if cancelled else 4400 + progress * 4000
    assert api.frames[-2][0][3] == expected_x
    assert api.frames[-1][0][3] == expected_x
    assert all(0 <= contact[3] <= 10000 for frame in api.frames for contact in frame)
    assert events[-2] == ("sleep", 1 / 60), "lift needs a distinct time step after the final move"
    assert events[-1] == ("frame", 0xC000 if cancelled else 0x4000)
    assert sum(value for name, value in events if name == "sleep") <= .12


def test_worker_deadlines_do_not_add_injection_cost_or_catch_up_in_bursts(monkeypatch):
    from types import SimpleNamespace

    instant, starts, delays = [0.0], [], []
    api = FakeUser32()
    device = d._WindowsTouchpad(api, sleep=lambda _: None)
    device._active = True
    device._last_update = 0

    class FakeStop:
        def wait(self, duration):
            delays.append(duration)
            instant[0] += duration
            return len(starts) >= 4

    def advance(elapsed):
        starts.append(instant[0])
        # Two normal frames, then an overrun. The next frame must wait again.
        instant[0] += .05 if len(starts) == 3 else .005

    device._stop = FakeStop()
    device._advance = advance
    monkeypatch.setattr(d, "time", SimpleNamespace(monotonic=lambda: instant[0]))
    device._animate()
    assert starts[:3] == pytest.approx([1 / 60, 2 / 60, 3 / 60])
    assert starts[3] == pytest.approx(starts[2] + .05 + 1 / 60)
    assert all(delay > 0 for delay in delays)
    device._active = False
    device._destroy_device()


def test_failed_native_followthrough_cancels_releases_and_never_replays_a_shortcut():
    class FollowthroughFailure(FakeUser32):
        def InjectSyntheticPointerInput(self, device, pointers, count):
            accepted = super().InjectSyntheticPointerInput(device, pointers, count)
            return False if len(self.frames) == 4 else accepted

    api, shortcuts = FollowthroughFailure(), []
    device = d._WindowsTouchpad(api, sleep=lambda _: None)
    controller = d.DesktopSwipeController(
        lambda: True, native_factory=lambda: device,
        shortcut=lambda direction: shortcuts.append(direction) or {"ok": True})
    controller.begin()
    controller.update(1)
    result = controller.end()
    assert "error" in result
    assert not result["cleanup_pending"]
    assert shortcuts == []
    assert all(contact[2] == 0xC000 for contact in api.frames[-1])
    assert api.destroyed == [123]
