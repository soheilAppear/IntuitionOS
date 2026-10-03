"""Hand coordination uses fake input devices, targets, confirmations and clocks."""

from types import SimpleNamespace

import pytest

from core import hand_control
from core.actions import register_os_capabilities


class FakeTimer:
    instances = None

    def __init__(self, duration, callback, args=()):
        self.duration, self.callback, self.args = duration, callback, args
        self.cancelled = False
        self.started = False
        self.instances.append(self)

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True

    def fire(self):
        self.callback(*self.args)


class FakeDesktop:
    def __init__(self, authorize, mode, shortcut, authorize_overview=None, overview_shortcut=None):
        self.authorize, self.shortcut, self.mode = authorize, shortcut, mode
        self.authorize_overview, self.overview_shortcut = authorize_overview, overview_shortcut
        self.calls = []
        self.active = False

    def status(self):
        return {"active": self.active, "mode": self.mode}

    def begin(self, axis="horizontal"):
        self.calls.append(("begin",) if axis == "horizontal" else ("begin", axis))
        permitted = (self.authorize_overview if axis == "vertical" else self.authorize)()
        if permitted.get("error"):
            return permitted
        self.active = True
        return {"ok": True, "active": True}

    def update(self, progress):
        self.calls.append(("update", progress))
        return {"ok": True, "active": self.active, "progress": progress}

    def end(self, cancelled=False):
        self.calls.append(("end", cancelled))
        self.active = False
        return {"ok": True, "active": False}

    def close(self):
        self.calls.append(("close",))
        self.active = False


@pytest.fixture
def hands(monkeypatch):
    timers, messages, dispatches, confirmations = [], [], [], []
    monkeypatch.setattr(FakeTimer, "instances", timers)
    monkeypatch.setattr(hand_control.threading, "Timer", FakeTimer)
    register_os_capabilities()
    now, active = [100.0], [True]
    target = {"ok": True, "hwnd": 10, "pid": 20, "title": "First document"}
    confirmation_result = {"ok": True, "status": "close_requested"}

    def dispatch(name, args, **kwargs):
        dispatches.append((name, dict(args), kwargs))
        if name == "os_window_close_target":
            return dict(target)
        if name == "os_close_window":
            return {"needs_confirmation": True, "token": f"close-{len(dispatches)}"}
        return {"ok": True}

    def confirm(token, granted=True, **kwargs):
        confirmations.append((token, granted, kwargs))
        return dict(confirmation_result) if granted else {"ok": True, "cancelled": True}

    controls = hand_control.HandControls(
        lambda: active[0], messages.append, clock=lambda: now[0],
        dispatch=dispatch, confirm=confirm, desktop_factory=FakeDesktop,
    )

    def event(name, at=None):
        if at is None:
            now[0] += 0.01
            at = now[0]
        return SimpleNamespace(name=name, at=at)

    return SimpleNamespace(
        controls=controls, now=now, active=active, target=target, messages=messages,
        dispatches=dispatches, confirmations=confirmations, timers=timers,
        confirmation_result=confirmation_result,
        event=event,
    )


def test_close_confirmation_consumes_the_original_target_token_only_once(hands):
    h = hands
    h.controls.handle(h.event("close_request"))
    assert h.dispatches[:2] == [
        ("os_window_close_target", {}, {"actor": "gesture"}),
        ("os_close_window", {"hwnd": 10, "pid": 20}, {"actor": "gesture"}),
    ]
    assert h.confirmations == []
    assert h.controls.status()["close"]["title"] == "First document"
    h.target.update(hwnd=11, pid=21, title="Another document")
    result = h.controls.handle(h.event("close_confirm"))
    assert result[2]["status"] == "close_requested"
    assert h.confirmations == [("close-2", True, {"allow_safe_mode_change": False})]
    assert h.controls.status()["close"]["pending"] is False
    assert h.controls.handle(h.event("close_confirm")) is None
    assert len(h.dispatches) == 2
    assert len(h.confirmations) == 1
    assert h.timers[0].cancelled


def test_close_error_is_reported_without_claiming_success(hands):
    h = hands
    h.controls.handle(h.event("close_request"))
    h.confirmation_result.clear()
    h.confirmation_result["error"] = "The active window changed."
    result = h.controls.handle(h.event("close_confirm"))
    assert result[2] == {"error": "The active window changed."}
    assert h.messages[-1] == {"type": "gesture_close", "pending": False,
                              "text": "The active window changed.", "ok": False}


def test_bad_snapshot_never_creates_an_approval(hands):
    h = hands
    h.target.clear()
    h.target["error"] = "No active application window."
    result = h.controls.handle(h.event("close_request"))
    assert result[2]["error"] == "No active application window."
    assert len(h.dispatches) == 1
    assert h.timers == []
    assert not h.controls.status()["close"]["pending"]


@pytest.mark.parametrize("cancel", ["fist", "hand_lost", "stop", "other_action"])
def test_cancellation_paths_consume_pending_close_without_granting(hands, cancel):
    h = hands
    h.controls.handle(h.event("close_request"))
    if cancel == "fist":
        h.controls.handle(h.event("close_cancel"))
    elif cancel == "hand_lost":
        h.controls.motion({"phase": "cancel", "at": h.now[0], "reason": "hand_lost"})
    elif cancel == "stop":
        h.active[0] = False
        h.controls.stop()
    else:
        h.controls.handle(h.event("restore_window"), ("os_window_state", {"state": "restore"}))
    assert h.confirmations == [("close-2", False, {"allow_safe_mode_change": False})]
    assert h.timers[0].cancelled
    assert not h.controls.status()["close"]["pending"]


def test_expiry_refuses_confirmation_even_if_timer_delivery_is_delayed(hands):
    h = hands
    h.controls.handle(h.event("close_request"))
    h.now[0] += h.controls.CLOSE_SECONDS
    assert h.controls.handle(h.event("close_confirm")) is None
    assert h.confirmations == [("close-2", False, {"allow_safe_mode_change": False})]
    assert "expired" in h.messages[-1]["text"]


def test_old_expiry_callback_cannot_cancel_a_new_request(hands):
    h = hands
    h.controls.handle(h.event("close_request"))
    first_timer = h.timers[0]
    h.now[0] += 1
    h.controls.handle(h.event("close_request"))
    first_timer.fire()
    assert h.controls.status()["close"]["pending"]
    assert len(h.confirmations) == 1
    h.timers[-1].fire()
    assert not h.controls.status()["close"]["pending"]
    assert all(not entry[1] for entry in h.confirmations)


@pytest.mark.parametrize("age", [1.01, -0.01])
def test_stale_or_future_actions_never_dispatch(hands, age):
    h = hands
    assert h.controls.handle(h.event("close_request", at=h.now[0] - age)) is None
    assert h.dispatches == []


def test_inactive_camera_cannot_authorize_actions_or_native_input(hands):
    h = hands
    h.active[0] = False
    assert h.controls.handle(h.event("close_request")) is None
    assert "error" in h.controls._authorize_desktop()
    h.controls.motion({"phase": "begin", "at": h.now[0]})
    assert h.controls.desktop.calls == [("end", True)]
    assert h.dispatches == []


def test_continuous_motion_preserves_signed_progress_and_releases_on_stop(hands):
    h = hands
    h.controls.motion({"phase": "begin", "at": h.now[0]})
    for progress in (0.2, 0.65, 0.1, -0.3):
        h.controls.motion({"phase": "update", "progress": progress, "at": h.now[0]})
    h.active[0] = False
    h.controls.motion({"phase": "update", "progress": -0.6, "at": h.now[0]})
    assert h.controls.desktop.calls == [
        ("begin",), ("update", 0.0), ("update", 0.2), ("update", 0.65),
        ("update", 0.1), ("update", -0.3), ("end", True),
    ]
    assert h.dispatches == []


def test_first_motion_frame_is_delivered_even_when_it_already_reaches_threshold(hands):
    hands.controls.motion({"phase": "begin", "progress": -1.0, "at": hands.now[0]})
    assert hands.controls.desktop.calls == [("begin",), ("update", -1.0)]


def test_hand_loss_invalidates_an_action_queued_before_the_loss(hands):
    h = hands
    queued = h.event("close_request")
    h.now[0] += 0.05
    h.controls.motion({"phase": "cancel", "at": h.now[0], "reason": "hand_lost"})
    h.controls.handle(queued)
    assert h.dispatches == [], "A delayed pinch must not arm close after the hand left the camera"
    assert not h.controls.status()["close"]["pending"]


def test_stop_and_restart_cannot_replay_an_action_from_previous_camera_session(hands):
    h = hands
    queued = h.event("restore_window")
    h.now[0] += 0.05
    h.active[0] = False
    h.controls.stop()
    h.now[0] += 0.05
    h.active[0] = True
    h.controls.handle(queued, ("os_window_state", {"state": "restore"}))
    assert h.dispatches == [], "A fresh camera session must not execute old queued window actions"


def test_out_of_order_action_cannot_override_a_newer_cancel(hands):
    h = hands
    older = h.event("close_request")
    newer = h.event("close_cancel")
    h.controls.handle(newer)
    h.controls.handle(older)
    assert h.dispatches == []


def test_duplicate_action_event_does_not_replace_its_pending_approval(hands):
    h = hands
    event = h.event("close_request")
    h.controls.handle(event)
    h.controls.handle(event)
    assert len(h.dispatches) == 2
    assert len(h.timers) == 1
    assert h.confirmations == []


def test_native_begin_obeys_capability_gate_before_becoming_active(hands, monkeypatch):
    h = hands
    monkeypatch.setattr(hand_control, "gate", lambda *args, **kwargs: SimpleNamespace(
        verdict="deny", reason="Desktop control is disabled."))
    outcome = h.controls.motion({"phase": "begin", "at": h.now[0]})
    assert outcome == {"error": "Desktop control is disabled."}
    assert h.controls.desktop.active is False
    assert h.dispatches == []


def test_vertical_navigation_checks_its_own_capability_before_input(hands, monkeypatch):
    checked = []

    def allow(cap, args, **kwargs):
        checked.append((cap.name, args, kwargs))
        return SimpleNamespace(verdict="allow")

    monkeypatch.setattr(hand_control, "gate", allow)
    outcome = hands.controls.motion({"phase": "begin", "axis": "vertical",
                                     "progress": -0.4, "at": hands.now[0]})
    assert outcome["ok"]
    assert checked == [("os_desktop_view", {"view": "overview"},
                        {"actor": "gesture", "confidence": 1.0})]
    assert hands.controls.desktop.calls == [("begin", "vertical"), ("update", -0.4)]
    assert hands.dispatches == []


def test_vertical_navigation_denial_never_starts_input(hands, monkeypatch):
    monkeypatch.setattr(hand_control, "gate", lambda *args, **kwargs: SimpleNamespace(
        verdict="deny", reason="Desktop overview is disabled."))
    result = hands.controls.motion({"phase": "begin", "axis": "vertical"})
    assert result == {"error": "Desktop overview is disabled."}
    assert hands.controls.desktop.calls == [("begin", "vertical"), ("end", True)]
    assert not hands.controls.desktop.active
    assert hands.controls._motion_axis is None


@pytest.mark.parametrize("direction,view", [("up", "overview"), ("down", "desktop")])
def test_overview_fallback_dispatches_through_the_capability_gate(hands, direction, view):
    assert hands.controls.desktop.overview_shortcut(direction)["ok"]
    assert hands.dispatches == [("os_desktop_view", {"view": view}, {"actor": "gesture"})]


@pytest.mark.parametrize("phase", ["update", "end"])
def test_axis_change_cancels_instead_of_committing_another_action(hands, phase):
    hands.controls.motion({"phase": "begin", "axis": "vertical", "progress": -1.0})
    result = hands.controls.motion({"phase": phase, "axis": "horizontal", "progress": 1.0})
    assert hands.controls.desktop.calls[-1] == ("end", True)
    assert not hands.controls.desktop.active
    assert hands.controls._motion_axis is None
    assert hands.dispatches == []
    if phase == "update":
        assert "error" in result


def test_invalid_axis_cancels_without_starting_contacts(hands):
    result = hands.controls.motion({"phase": "begin", "axis": "diagonal"})
    assert "error" in result
    assert hands.controls.desktop.calls == [("end", True)]
    assert hands.dispatches == []


def test_failed_rebegin_cancels_previous_motion_and_clears_its_axis(hands, monkeypatch):
    hands.controls.motion({"phase": "begin", "axis": "vertical"})
    monkeypatch.setattr(hands.controls.desktop, "begin", lambda **kwargs: {
        "error": "Previous contacts are still active."})
    result = hands.controls.motion({"phase": "begin", "axis": "horizontal"})
    assert "error" in result
    assert hands.controls.desktop.calls[-1] == ("end", True)
    assert not hands.controls.desktop.active
    assert hands.controls._motion_axis is None
