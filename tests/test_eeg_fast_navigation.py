"""Combined EEG/hand navigation uses real coordinators and fake OS devices only."""

import pytest

from core.actions import register_os_capabilities
from core.desktop import DesktopSwipeController
from core.eeg_hand_control import EegHandController
from core.hand_mouse import HandMouseController
from fast_blink_fakes import CameraHarness
from test_hand_mouse import Device, hand


class NativeSwipe:
    """Record native ownership without creating a touchpad or sending input."""

    def __init__(self):
        self.events = []
        self.active = False
        self.closed = False
        self.cleanup_pending = False
        self.fail_release = False

    def begin(self, axis="horizontal"):
        self.events.append(("begin", axis))
        self.active = True
        self.closed = False

    def update(self, progress):
        self.events.append(("update", progress))

    def end(self, cancelled=False):
        self.events.append(("end", cancelled))
        if self.fail_release:
            self.cleanup_pending = True
            raise OSError("Synthetic native contacts could not be released")
        self.active = self.cleanup_pending = False

    def close(self):
        self.events.append(("close",))
        if self.fail_release:
            self.cleanup_pending = True
            raise OSError("Synthetic native contacts could not be released")
        self.active = self.cleanup_pending = False
        self.closed = True


class NavigationHarness(CameraHarness):
    """Drive the service using camera landmarks and fresh synthetic EEG packets."""

    def __init__(self, monkeypatch, *, mode="shortcut"):
        super().__init__(monkeypatch)
        register_os_capabilities()
        self.points = hand()
        self.device = Device()
        self.native = NativeSwipe()
        self.dispatches = []
        self.native_creations = 0
        original_preview = self.camera.preview

        def preview():
            result = original_preview()
            if result.get("tracked"):
                result["landmarks"] = self.points
            return result

        def dispatch(name, args, **kwargs):
            self.dispatches.append((name, dict(args), kwargs))
            return {"ok": True}

        def create_native():
            self.native_creations += 1
            return self.native

        def create_desktop(**kwargs):
            return DesktopSwipeController(native_factory=create_native, **kwargs)

        self.camera.preview = preview
        mouse = HandMouseController(device_factory=lambda: self.device, eeg_compatible=True)
        self.pointer = EegHandController(
            mouse, active=self.service.hand_navigation_authorized,
            mode=mode, clock=self.clock, dispatch=dispatch, desktop_factory=create_desktop)
        self.service.pointer_controller = self.pointer
        self.enable()

    def pose(self, name="point", *, dx=0, count=1, step=1 / 30):
        self.points = hand(name, dx=dx)
        self.tick(count, step=step)

    def clutch(self):
        self.pose("palm", count=11)

    def travel(self, sign=1, *, complete=True):
        for index in range(1, 10 if complete else 5):
            self.pose("palm", dx=sign * index * .03)
        if complete:
            self.pose("palm", dx=sign * .27, count=5)

    def begin_native(self):
        self.arm()
        self.clutch()
        self.travel(complete=False)
        assert self.native.active
        assert self.native_creations == 1


@pytest.fixture
def navigation(monkeypatch):
    return NavigationHarness(monkeypatch)


@pytest.fixture
def native_navigation(monkeypatch):
    return NavigationHarness(monkeypatch, mode="auto")


def test_preview_never_dispatches_desktop_mouse_or_blink_without_shared_arm(navigation):
    h = navigation
    h.clutch()
    h.travel()
    h.emit("unarmed-pair")
    assert not h.control()["armed"]
    assert h.dispatches == h.device.events == []
    assert h.mouse.clicks == h.native_creations == 0
    assert not h.pointer.navigation_active


@pytest.mark.parametrize("sign,destination", [(1, "left"), (-1, "right")])
def test_deliberate_horizontal_swipe_switches_once_until_neutral(navigation, sign, destination):
    h = navigation
    h.arm()
    h.clutch()
    h.travel(sign)
    assert h.dispatches == [("os_switch_desktop", {"direction": destination}, {"actor": "gesture"})]
    assert h.pointer.navigation_active
    h.pose("palm", dx=sign * .27, count=45)
    assert len(h.dispatches) == 1
    assert h.control()["armed"] and h.mouse.clicks == 0
    assert not any(event[0] in ("down", "up") for event in h.device.events)
    h.pose("point", dx=sign * .27, count=9)
    assert not h.pointer.navigation_active


def test_swipe_candidate_and_completion_latch_cancel_blink_without_replay(navigation):
    h = navigation
    h.arm()
    resets, cancelled = h.decoder.stream_resets, h.decoder.cancel_calls
    h.points = hand("palm")
    h.emit("blocked-during-navigation")
    assert h.pointer.navigation_active
    assert not h.control()["can_eeg_click"] and h.mouse.clicks == 0
    assert h.decoder.cancel_calls == cancelled + 1
    h.clutch()
    h.travel()
    h.emit("blocked-during-navigation")
    assert h.mouse.clicks == 0 and len(h.dispatches) == 1
    assert h.decoder.stream_resets == resets
    h.pose("point", dx=.27, count=12)
    h.tick(12)  # The ordinary one-second post-hand-action cooldown expires.
    assert h.control()["armed"] and not h.pointer.navigation_active
    h.emit("blocked-during-navigation")
    assert h.mouse.clicks == 0
    h.emit("new-pair-after-navigation")
    assert h.mouse.clicks == 1
    assert h.decoder.cancel_calls == cancelled + 2


def test_an_owned_finger_drag_finishes_before_a_new_navigation_clutch(navigation):
    h = navigation
    h.arm()
    h.pose("point", count=9)
    h.pose("pinch", count=12)
    assert h.service._pointer_status["button_down"]
    assert [event[0] for event in h.device.events].count("down") == 1
    h.pose("palm")
    assert not h.pointer.navigation_active  # This frame belongs to releasing the existing drag.
    assert [event[0] for event in h.device.events].count("up") == 1
    assert h.dispatches == []
    h.clutch()
    h.travel()
    assert len(h.dispatches) == 1
    assert h.mouse.clicks == 0


@pytest.mark.parametrize("revocation", ["escape", "stop", "guard", "eeg_stale", "hand_loss", "camera_loss"])
def test_revocation_releases_native_contacts_and_cannot_finish_old_swipe(native_navigation, revocation):
    h = native_navigation
    h.begin_native()
    if revocation == "escape":
        assert h.service.eeg_arm(False).get("ok")
    elif revocation == "stop":
        assert h.service.stop().get("ok")
    elif revocation == "guard":
        h.guard_active = False
        h.service._eeg_guard_until = h.clock.now
        h.tick()
    elif revocation == "eeg_stale":
        h.tick(age=1.0)
    elif revocation == "hand_loss":
        h.show_hand = False
        h.tick()
    else:
        h.camera.running = False
        h.camera.frame = {"running": False, "state": "error", "tracked": False}
        h.tick()
    assert not h.native.active
    assert ("end", True) in h.native.events
    assert ("end", False) not in h.native.events
    assert h.mouse.clicks == 0 and h.dispatches == []
    if revocation == "hand_loss":
        assert h.control()["armed"]  # Hand disappearance must not disable camera-independent EEG clicking.
    else:
        assert not h.control()["armed"]
    # A later frame at the old destination cannot resume those cancelled contacts.
    h.show_hand = True
    h.pose("palm", dx=.27, count=5)
    assert ("end", False) not in h.native.events
    assert h.dispatches == [] and h.mouse.clicks == 0


@pytest.mark.parametrize("boundary", ["begin", "end"])
@pytest.mark.parametrize("revocation", ["guard", "camera_stale", "eeg_stale"])
def test_navigation_rechecks_authorization_at_native_boundary(native_navigation, monkeypatch, boundary, revocation):
    h = native_navigation
    h.arm()
    h.clutch()
    original = h.pointer.controls.motion
    revoked = []

    def motion(record):
        if record["phase"] == boundary and not revoked:
            # Revoke after the service admitted this camera frame but before
            # the real HandControls coordinator reaches the desktop device.
            revoked.append(record["phase"])
            if revocation == "guard":
                h.guard_active = False
                h.service._eeg_guard_until = h.clock.now
            elif revocation == "camera_stale":
                h.service._camera["age_ms"] = 1000
            else:
                h.service._eeg["stats"]["age_seconds"] = 1.0
        return original(record)

    monkeypatch.setattr(h.pointer.controls, "motion", motion)
    h.travel()
    assert revoked == [boundary]
    assert h.dispatches == [] and h.mouse.clicks == 0
    assert ("end", False) not in h.native.events
    if boundary == "begin":
        assert h.native_creations == 0
    else:
        assert ("end", True) in h.native.events and not h.native.active


def test_native_release_failure_disarms_and_blocks_restart_until_stop_confirms_cleanup(native_navigation):
    h = native_navigation
    h.begin_native()
    h.native.fail_release = True
    h.show_hand = False
    h.tick()
    assert not h.control()["armed"] and h.service.status()["cleanup_pending"]
    assert h.native.cleanup_pending
    assert h.service.stop().get("error")
    assert h.service.start().get("error")
    assert h.native_creations == 1 and h.mouse.clicks == 0
    assert h.dispatches == [] and ("end", False) not in h.native.events
    h.native.fail_release = False
    assert h.service.stop().get("ok")
    assert not h.service.status()["cleanup_pending"]
    assert not h.native.active and h.native.closed
    assert h.service.start().get("ok")
    assert not h.control()["armed"]
