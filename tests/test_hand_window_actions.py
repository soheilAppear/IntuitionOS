"""A close approval is bound to one foreground window and never force-kills it."""

import sys
from types import SimpleNamespace

import pytest

from core import actions as actions_mod
from core import os_sandbox
from core.capabilities import capabilities, gate, is_safe_mode, set_safe_mode


class FakeUser32:
    def __init__(self):
        self.active = 101
        self.pid = 202
        self.title = "Unsaved document — Editor"
        self.window_class = "EditorWindow"
        self.exists = True
        self.visible = True
        self.posted = []

    def GetForegroundWindow(self):
        return self.active

    def GetShellWindow(self):
        return 1

    def GetDesktopWindow(self):
        return 2

    def IsWindow(self, hwnd):
        return self.exists and hwnd == 101

    def IsWindowVisible(self, hwnd):
        return self.visible

    def GetClassNameW(self, hwnd, buffer, length):
        buffer.value = self.window_class
        return len(buffer.value)

    def GetWindowTextLengthW(self, hwnd):
        return len(self.title)

    def GetWindowTextW(self, hwnd, buffer, length):
        buffer.value = self.title
        return len(buffer.value)

    def GetWindowThreadProcessId(self, hwnd, pointer):
        pointer._obj.value = self.pid
        return 11

    def PostMessageW(self, hwnd, message, wparam, lparam):
        self.posted.append((hwnd, message, wparam, lparam))
        return True


@pytest.fixture
def window_api(monkeypatch):
    api = FakeUser32()
    monkeypatch.setattr(os_sandbox, "_user32", lambda: api)
    actions_mod.register_os_capabilities()
    return api


def test_close_posts_only_wm_close_to_the_captured_foreground_window(window_api):
    target = os_sandbox.snapshot_active_window()
    assert target == {"ok": True, "hwnd": 101, "pid": 202,
                      "title": "Unsaved document — Editor"}
    outcome = os_sandbox.close_window(target["hwnd"], target["pid"])
    assert outcome["status"] == "close_requested"
    assert "save prompt" in outcome["message"]
    assert window_api.posted == [(101, 0x0010, 0, 0)]


@pytest.mark.parametrize("attribute,value", [
    ("active", 303), ("pid", 404), ("exists", False), ("visible", False),
])
def test_close_refuses_a_changed_or_missing_target(window_api, attribute, value):
    target = os_sandbox.snapshot_active_window()
    setattr(window_api, attribute, value)
    assert "error" in os_sandbox.close_window(target["hwnd"], target["pid"])
    assert window_api.posted == []


@pytest.mark.parametrize("window_class,title", [
    ("Progman", "Program Manager"), ("WorkerW", ""),
    ("Shell_TrayWnd", ""), ("Shell_SecondaryTrayWnd", ""),
    ("Chrome_WidgetWin_1", "IntuitionOS"),
])
def test_snapshot_and_close_exclude_desktop_taskbars_and_hud(window_api, window_class, title):
    window_api.window_class = window_class
    window_api.title = title
    assert "error" in os_sandbox.snapshot_active_window()
    assert "error" in os_sandbox.close_window(101, 202)
    assert window_api.posted == []


@pytest.mark.parametrize("hwnd", [1, 2])
def test_shell_and_desktop_handles_are_never_close_targets(window_api, monkeypatch, hwnd):
    window_api.active = hwnd
    monkeypatch.setattr(window_api, "IsWindow", lambda _hwnd: True)
    assert "error" in os_sandbox.snapshot_active_window()
    assert "error" in os_sandbox.close_window(hwnd, 202)
    assert window_api.posted == []


def test_failed_post_does_not_claim_the_window_closed(window_api, monkeypatch):
    monkeypatch.setattr(window_api, "PostMessageW", lambda *args: False)
    assert "error" in os_sandbox.close_window(101, 202)


def test_snapshot_is_a_free_read_and_close_is_always_confirmed(window_api):
    assert capabilities.get("os_window_close_target").reversibility == "free"
    cap = capabilities.get("os_close_window")
    assert cap.reversibility == "irreversible"
    assert cap.needs_confirmation({"hwnd": 101, "pid": 202}, actor="gesture")
    for safe in (True, False):
        set_safe_mode(safe)
        decision = gate(cap, {"hwnd": 101, "pid": 202}, actor="gesture", confidence=1.0)
        assert decision.verdict == "confirm"
        assert not decision.requires_safe_mode_off


def test_confirmation_closes_once_without_disabling_safe_mode(window_api, wired):
    registry, journal, _ = wired
    set_safe_mode(True)
    request = registry.dispatch("os_close_window", {"hwnd": 101, "pid": 202}, actor="gesture")
    assert request["needs_confirmation"]
    assert request["requires_safe_mode_off"] is False
    assert window_api.posted == []
    outcome = registry.confirm(request["token"])
    assert outcome["status"] == "close_requested"
    assert is_safe_mode() is True
    assert "error" in registry.confirm(request["token"])
    assert window_api.posted == [(101, 0x0010, 0, 0)]
    assert journal.recent(limit=1)[0]["decision"] == "confirm_granted"


def test_direct_confirmed_flag_cannot_close_a_window_without_a_token(window_api, wired):
    registry, _, _ = wired
    request = registry.dispatch("os_close_window", {"hwnd": 101, "pid": 202},
                                actor="gesture", confirmed=True)
    assert request["needs_confirmation"]
    assert window_api.posted == []
    assert registry.confirm(request["token"], granted=False)["cancelled"]


@pytest.mark.parametrize("attribute,value", [("active", 303), ("pid", 404)])
def test_confirmation_rechecks_the_target_and_consumes_failed_approval(window_api, wired, attribute, value):
    registry, _, _ = wired
    request = registry.dispatch("os_close_window", {"hwnd": 101, "pid": 202}, actor="gesture")
    setattr(window_api, attribute, value)
    assert "error" in registry.confirm(request["token"])
    assert "error" in registry.confirm(request["token"])
    assert window_api.posted == []


@pytest.mark.parametrize("name,args", [
    ("os_shutdown_computer", {}), ("os_restart_computer", {}),
    ("os_kill_process", {"name": "editor.exe"}),
])
@pytest.mark.parametrize("safe", [True, False])
def test_gesture_close_exception_does_not_authorize_other_irreversible_actions(window_api, name, args, safe):
    set_safe_mode(safe)
    decision = gate(capabilities.get(name), args, actor="gesture", confidence=1.0,
                    offer_safe_mode_confirmation=True)
    assert decision.verdict == "deny"
    assert "gesture" in decision.reason


@pytest.mark.parametrize("args", [{}, {"hwnd": 101}, {"hwnd": True, "pid": 202},
                                 {"hwnd": 101, "pid": 0}, {"hwnd": "101", "pid": 202}])
def test_close_requires_real_positive_integer_target_fields(window_api, args):
    decision = gate(capabilities.get("os_close_window"), args, actor="gesture", confidence=1.0)
    assert decision.verdict == "deny"
    assert window_api.posted == []


def test_switch_desktop_dispatches_only_declared_directions(window_api, wired, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "core.desktop", SimpleNamespace(
        shortcut_switch=lambda direction: calls.append(direction) or {"ok": True}))
    registry, _, _ = wired
    assert registry.dispatch("os_switch_desktop", {"direction": "left"}, actor="gesture")["ok"]
    assert registry.dispatch("os_switch_desktop", {"direction": "right"}, actor="gesture")["ok"]
    assert registry.dispatch("os_switch_desktop", {"direction": "up"}, actor="gesture")["denied"]
    assert calls == ["left", "right"]


def test_desktop_views_dispatch_only_declared_views(window_api, wired, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "core.desktop", SimpleNamespace(
        shortcut_overview=lambda direction: calls.append(direction) or {"ok": True}))
    registry, _, _ = wired
    for view in ("overview", "desktop"):
        result = registry.dispatch("os_desktop_view", {"view": view}, actor="gesture")
        assert result["ok"]
        assert result["view"] == view
    assert registry.dispatch("os_desktop_view", {"view": "close"}, actor="gesture")["denied"]
    assert "error" in os_sandbox.desktop_view("close")
    assert calls == ["up", "down"]


def test_desktop_view_reports_input_failure_without_claiming_success(window_api, monkeypatch):
    def fail(direction):
        raise OSError("Input is unavailable")

    monkeypatch.setitem(sys.modules, "core.desktop", SimpleNamespace(shortcut_overview=fail))
    outcome = os_sandbox.desktop_view("overview")
    assert "Input is unavailable" in outcome["error"]
    assert not outcome.get("ok")


class CyclingUser32:
    def __init__(self):
        self.windows = {
            10: ("Editor", "Application"), 20: ("Browser", "Application"),
            30: ("Files", "Application"), 90: ("IntuitionOS", "Chrome_WidgetWin_1"),
            80: ("Other desktop app", "Application"), 70: ("Taskbar", "Shell_TrayWnd"),
            1: ("Shell", "Application"), 2: ("Desktop", "Application"),
        }
        self.order = list(self.windows)
        self.active = 10
        self.minimized = {20}
        self.shown = []
        self.focused = []
        self.refuse_focus = False

    def GetForegroundWindow(self):
        return self.active

    def GetShellWindow(self):
        return 1

    def GetDesktopWindow(self):
        return 2

    def IsWindow(self, hwnd):
        return hwnd in self.windows

    def IsWindowVisible(self, hwnd):
        return True

    def GetClassNameW(self, hwnd, buffer, length):
        buffer.value = self.windows[hwnd][1]
        return len(buffer.value)

    def GetWindowTextLengthW(self, hwnd):
        return len(self.windows[hwnd][0])

    def GetWindowTextW(self, hwnd, buffer, length):
        buffer.value = self.windows[hwnd][0]
        return len(buffer.value)

    def IsIconic(self, hwnd):
        return hwnd in self.minimized

    def ShowWindow(self, hwnd, command):
        self.shown.append((hwnd, command))
        self.minimized.discard(hwnd)
        return True

    def SetForegroundWindow(self, hwnd):
        self.focused.append(hwnd)
        if self.refuse_focus:
            return False
        self.active = hwnd
        self.order = [hwnd] + [other for other in self.order if other != hwnd]
        return True


@pytest.fixture
def cycling_api(monkeypatch):
    api = CyclingUser32()
    monkeypatch.setattr(os_sandbox, "_user32", lambda: api)
    monkeypatch.setattr(os_sandbox, "_enumerate_windows", lambda user32: [
        (hwnd, user32.windows[hwnd][0]) for hwnd in user32.order])
    monkeypatch.setattr(os_sandbox, "_window_is_cloaked", lambda hwnd: hwnd == 80)
    return api


def test_next_window_traverses_all_apps_despite_z_order_changes(cycling_api):
    results = [os_sandbox.cycle_window() for _ in range(3)]
    assert all(result["ok"] for result in results)
    assert cycling_api.focused == [20, 30, 10]
    assert cycling_api.shown == [(20, os_sandbox._SW_RESTORE)]
    assert 20 not in cycling_api.minimized


def test_previous_window_traverses_in_reverse_without_hud_shell_or_other_desktop(cycling_api):
    for _ in range(3):
        assert os_sandbox.cycle_window("previous")["ok"]
    assert cycling_api.focused == [30, 20, 10]


@pytest.mark.parametrize("target", [90, 1, 2, 70, 80])
@pytest.mark.parametrize("state", ["maximize", "minimize", "restore"])
def test_window_state_never_changes_hud_shell_or_cloaked_window(cycling_api, target, state):
    cycling_api.active = target
    assert "error" in os_sandbox.set_window_state(state=state)
    assert cycling_api.shown == []


def test_cycle_can_return_from_hud_to_the_only_application(cycling_api):
    cycling_api.active = 90
    cycling_api.order = [90, 10]
    assert os_sandbox.cycle_window()["ok"]
    assert cycling_api.focused == [10]


def test_cycle_reports_no_other_app_without_selecting_the_hud(cycling_api):
    cycling_api.order = [90, 10]
    assert "error" in os_sandbox.cycle_window()
    assert cycling_api.focused == []


def test_cycle_does_not_claim_focus_when_windows_rejects_it(cycling_api):
    cycling_api.refuse_focus = True
    result = os_sandbox.cycle_window()
    assert "error" in result
    assert not result.get("ok")
    assert cycling_api.active == 10
