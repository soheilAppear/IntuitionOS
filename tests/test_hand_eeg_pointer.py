"""Join real hand aiming and click actuators using fake camera, EEG, and Windows input."""

import pytest

from core.eeg_click import EegClickController
from core.hand_mouse import HandMouseController
from test_eeg_click_control import ClickHarness
from test_hand_mouse import Device, buttons, hand


class HandEegHarness(ClickHarness):
    def __init__(self, monkeypatch, *, bend_click=False):
        super().__init__(monkeypatch)
        self.device = Device()
        self.pointer = HandMouseController(device_factory=lambda: self.device, eeg_compatible=True, bend_click=bend_click)
        self.service.pointer_controller = self.pointer
        self.service.click_controller = EegClickController(device_factory=lambda: self.device)
        self.pose, self.dx = "point", 0.0
        original_preview = self.camera.preview

        def preview():
            frame = original_preview()
            if frame.get("tracked"):
                frame["landmarks"] = hand(self.pose, dx=self.dx)
            return frame

        self.camera.preview = preview

    def engage(self):
        self.decoder.label = "rest"
        self.tick(16)
        assert self.control()["hand_pointer"]["ready"]
        assert self.control()["neutral_ready"]


def test_real_hand_aims_and_finger_clicks_work_alongside_eeg_practice_and_mouse(monkeypatch):
    h = HandEegHarness(monkeypatch)
    h.tick(12)
    assert h.device.events == []
    h.arm()
    h.engage()
    h.dx = 0.06
    h.tick(4)
    assert any(event[0] == "move" for event in h.device.events)
    assert len(set(event[1:] for event in h.device.events)) > 1
    h.pose = "pinch"
    h.tick()
    h.pose = "point"
    h.tick()
    assert buttons(h.device) == ["down", "up"]
    assert h.control()["finger_click_count"] == 1
    assert h.control()["selection_count"] == 0
    h.engage()
    h.select()
    assert h.control()["selection_pending"]
    held_events = list(h.device.events)
    h.dx = 0.08
    h.tick(4)
    assert h.device.events == held_events
    h.release()
    assert h.control()["selection_count"] == 1
    assert buttons(h.device) == ["down", "up"]

    assert h.service.eeg_target("mouse").get("ok")
    assert not h.control()["armed"]  # A changed EEG destination still requires an explicit arm.
    h.arm()
    h.engage()
    h.select()
    held_events = list(h.device.events)
    h.dx = 0.10
    h.tick(4)
    assert h.device.events == held_events
    h.release()
    assert buttons(h.device) == ["down", "up", "down", "up"]
    assert h.control()["click_count"] == 1
    assert h.control()["finger_click_count"] == 1
    assert h.actions == []
    assert h.service.stop().get("ok")
    assert buttons(h.device) == ["down", "up", "down", "up"]


@pytest.mark.parametrize("loss", ["fist", "missing", "jump", "physical_button"])
def test_real_pointer_rejects_lost_or_changed_hand_target_before_eeg_click(monkeypatch, loss):
    h = HandEegHarness(monkeypatch)
    h.service.eeg_target("mouse")
    h.arm()
    h.engage()
    h.select()
    assert h.control()["selection_pending"]
    if loss == "fist":
        h.pose = "fist"
    elif loss == "jump":
        h.dx = 0.6
    elif loss == "physical_button":
        h.device.block = "Release physical mouse buttons"
    h.tick(tracked=loss != "missing")
    assert h.control()["armed"]  # Temporary pose loss pauses targeting rather than revoking explicit hand authorization.
    assert not h.control()["selection_pending"]
    h.pose, h.dx, h.device.block = "point", 0.0, None
    h.release()
    assert buttons(h.device) == []
    assert h.control()["hand_pointer"]["ready"]  # Fresh stable pointing recovers without a second shared arm.


def test_hand_controller_without_ownership_feedback_cannot_be_installed_for_eeg(monkeypatch):
    from core.multimodal import MultimodalPreview
    from test_eeg_control import Camera, EEG

    created = []
    pointer = HandMouseController(device_factory=lambda: created.append(True))
    with pytest.raises(ValueError, match="EEG-compatible"):
        MultimodalPreview(Camera(), EEG(), pointer_controller=pointer)
    assert created == []


@pytest.mark.parametrize("gesture", ["pinch", "hook"])
def test_finger_click_during_pending_eeg_selection_cancels_duplicate(monkeypatch, gesture):
    h = HandEegHarness(monkeypatch, bend_click=gesture == "hook")
    h.service.eeg_target("mouse")
    h.arm()
    h.engage()
    h.select()
    assert h.control()["selection_pending"]
    h.pose = gesture
    h.tick(2 if gesture == "hook" else 1)
    assert not h.control()["selection_pending"]
    h.pose = "point"
    h.tick()
    assert buttons(h.device) == ["down", "up"]
    assert h.control()["finger_click_count"] == 1
    assert h.control()["armed"]
    h.release()
    assert buttons(h.device) == ["down", "up"]
    assert h.control()["click_count"] == 0
    h.engage()
    h.select()
    h.release()
    assert buttons(h.device) == ["down", "up", "down", "up"]
    assert h.control()["click_count"] == 1


@pytest.mark.parametrize("reason", ["hand_loss", "eeg_stale", "raw_invalid", "session", "guard", "uncertainty", "disarm", "stop"])
def test_disarm_releases_owned_finger_drag_before_returning(monkeypatch, reason):
    h = HandEegHarness(monkeypatch)
    h.arm()
    h.engage()
    h.pose = "pinch"
    h.tick(4)
    assert buttons(h.device) == ["down"]
    assert h.service._pointer_status["button_down"]
    if reason == "hand_loss":
        h.tick(tracked=False)
    elif reason == "eeg_stale":
        h.tick(eeg_age=1)
    elif reason == "raw_invalid":
        h.extract_ok = False
        h.tick(2)
    elif reason == "session":
        h.eeg.acq["session_id"] = "changed-while-dragging"
        h.tick()
    elif reason == "guard":
        h.guard_active = False
        h.tick(25)
    elif reason == "uncertainty":
        h.decoder.label = None
        h.tick(28)
    elif reason == "disarm":
        assert h.service.eeg_arm(False).get("ok")
    else:
        assert h.service.stop().get("ok")
    if reason in ("hand_loss", "guard", "disarm", "stop"):
        assert buttons(h.device) == ["down", "up"]  # A missing hand or explicit/emergency revocation releases the owned drag.
    else:
        assert buttons(h.device) == ["down"]  # EEG interruption does not interrupt a still-tracked finger drag.
    assert h.control()["armed"] is (reason not in ("guard", "disarm", "stop"))
    assert not h.service.status()["cleanup_pending"]
    assert h.control()["finger_click_count"] == 0


def test_failed_drag_release_stays_visible_and_stop_retries_only_release(monkeypatch):
    h = HandEegHarness(monkeypatch)
    h.arm()
    h.engage()
    h.pose = "pinch"
    h.tick(4)
    h.device.fail["up"] = 1
    h.tick(tracked=False)  # A tracking loss, rather than unrelated EEG freshness, releases the drag.
    assert not h.control()["armed"]
    assert h.service.status()["cleanup_pending"]
    assert h.pointer._owned_left
    assert h.service.stop().get("ok")
    assert buttons(h.device) == ["down", "up", "up"]
    assert not h.pointer._owned_left
    assert not h.service.status()["cleanup_pending"]
