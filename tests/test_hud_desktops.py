"""HUD desktop pinning uses fake COM and windows; no live window is changed."""

import ctypes
import json

import pytest

from core import hud_desktops as h


class FakeWindows:
    def __init__(self, *, exists=True, pid=42, title="IntuitionOS", window_class="Chrome_WidgetWin_1"):
        self.exists, self.pid = exists, pid
        self.title, self.window_class = title, window_class
        self.validations = 0

    def IsWindow(self, hwnd):
        self.validations += 1
        return self.exists

    def GetWindowThreadProcessId(self, hwnd, result):
        ctypes.cast(result, ctypes.POINTER(h._U32))[0] = self.pid
        return 1

    def GetWindowTextW(self, hwnd, buffer, size):
        buffer.value = self.title
        return len(self.title)

    def GetClassNameW(self, hwnd, buffer, size):
        buffer.value = self.window_class
        return len(self.window_class)


class FakeShell:
    def __init__(self, *, pinned=False, failure=None, after_view=None):
        self.pinned, self.failure = pinned, failure
        self.calls = []
        self.after_view = after_view

    def _call(self, name):
        self.calls.append(name)
        if self.failure == name:
            raise OSError("shell disconnected")

    def __enter__(self):
        self._call("enter")
        return self

    def __exit__(self, *_args):
        self._call("close")

    def view_for(self, hwnd):
        self._call("view")
        if self.after_view:
            self.after_view()
        return object()

    def is_pinned(self, view):
        self._call("check")
        return self.pinned

    def pin(self, view):
        self._call("pin")
        self.pinned = True

    def on_current_desktop(self, hwnd):
        self._call("current")
        return True


def run(api=None, shell=None, **kwargs):
    return h.pin_window(123, 42, window_api=api or FakeWindows(),
                        shell_factory=lambda: shell or FakeShell(),
                        platform_check=lambda: True, **kwargs)


@pytest.mark.parametrize("kwargs", [{"exists": False}, {"pid": 43}, {"title": "Browser"},
                                    {"title": "IntuitionOS DevTools"}, {"window_class": "ConsoleWindowClass"}])
def test_only_exact_expected_hud_can_reach_com(kwargs):
    shell = FakeShell()
    result = run(FakeWindows(**kwargs), shell)
    assert not result["ok"]
    assert shell.calls == []


@pytest.mark.parametrize("hwnd,pid", [(0, 42), (-1, 42), (True, 42), (123, 0),
                                       (123, 1 << 32), (1 << 64, 42)])
def test_invalid_identifiers_fail_before_any_windows_calls(hwnd, pid):
    result = h.pin_window(hwnd, pid, platform_check=lambda: pytest.fail("OS reached"))
    assert not result["ok"]


def test_unsupported_platform_reports_error_without_loading_com():
    result = h.pin_window(123, 42, platform_check=lambda: False,
                          shell_factory=lambda: pytest.fail("COM reached"))
    assert "Windows 10" in result["error"]


def test_check_does_not_pin_or_change_window():
    shell = FakeShell()
    result = run(shell=shell, check_only=True)
    assert result == {"ok": True, "pinned": False, "hwnd": 123, "pid": 42,
                      "on_current_desktop": True}
    assert shell.calls == ["enter", "view", "check", "current", "close"]


def test_pin_rechecks_identity_and_verifies_result():
    api, shell = FakeWindows(), FakeShell()
    result = run(api, shell)
    assert result["ok"] and result["pinned"]
    assert api.validations == 3
    assert shell.calls == ["enter", "view", "check", "pin", "check", "current", "close"]


def test_already_pinned_hud_is_not_pinned_again():
    shell = FakeShell(pinned=True)
    assert run(shell=shell)["pinned"]
    assert "pin" not in shell.calls


def test_reused_window_handle_during_lookup_is_rejected_before_pinning():
    api = FakeWindows()
    shell = FakeShell(after_view=lambda: setattr(api, "pid", 777))
    result = run(api, shell)
    assert not result["ok"]
    assert "pin" not in shell.calls
    assert shell.calls[-1] == "close"


@pytest.mark.parametrize("failure", ["view", "check", "pin"])
def test_shell_errors_still_release_context(failure):
    shell = FakeShell(failure=failure)
    result = run(shell=shell)
    assert not result["ok"]
    assert shell.calls[-1] == "close"


def test_unverified_pin_reports_failure():
    shell = FakeShell()
    shell.pin = lambda view: shell.calls.append("pin")
    result = run(shell=shell)
    assert not result["ok"]
    assert "did not confirm" in result["error"]


def test_optional_public_desktop_check_can_fail_without_losing_pin_result():
    result = run(shell=FakeShell(failure="current"))
    assert result["ok"] and result["pinned"]
    assert result["on_current_desktop"] is None


class FakeCOM:
    def __init__(self, *, failure_slot=None, initialize=0, null_view=False):
        self.failure_slot, self.initialize = failure_slot, initialize
        self.null_view = null_view
        self.calls, self.released = [], []
        self.next_pointer = 100
        self.uninitialized = 0

    def _out(self, output):
        ctypes.cast(output, ctypes.POINTER(h._PTR))[0] = self.next_pointer
        self.next_pointer += 100

    def CoInitializeEx(self, reserved, mode):
        assert mode == 2
        return self.initialize

    def CoUninitialize(self):
        self.uninitialized += 1

    def CoCreateInstance(self, clsid, outer, context, iid, output):
        assert context == 4
        self._out(output)
        return 0

    def method(self, pointer, slot, argtypes, restype=h._HRESULT):
        def call(this, *args):
            key = (pointer.value, slot)
            self.calls.append(key)
            if slot == 2:
                self.released.append(pointer.value)
                return 0
            if self.failure_slot == key:
                return -2147467262  # E_NOINTERFACE.
            if pointer.value == 100 and slot == 3:
                self._out(args[-1])
            elif pointer.value == 200 and slot == 6:
                if not self.null_view:
                    self._out(args[-1])
            elif slot == 6 or (pointer.value == 500 and slot == 3):
                ctypes.cast(args[-1], ctypes.POINTER(h._BOOL))[0] = 1
            return 0
        return call


def test_com_uses_expected_slots_and_releases_all_references_in_reverse_order():
    com = FakeCOM()
    with h._WindowsPinning(com, method_factory=com.method) as shell:
        view = shell.view_for(123)
        assert shell.is_pinned(view)
        shell.pin(view)
        assert shell.on_current_desktop(123)
    assert (200, 6) in com.calls  # GetViewForHwnd.
    assert (300, 6) in com.calls  # IsViewPinned.
    assert (300, 7) in com.calls  # PinView.
    assert com.released == [500, 400, 300, 200, 100]
    assert com.uninitialized == 1


def test_partial_com_initialization_failure_releases_provider_and_apartment():
    com = FakeCOM(failure_slot=(100, 3))
    with pytest.raises(OSError, match="0x80004002"):
        with h._WindowsPinning(com, method_factory=com.method):
            pytest.fail("Should not enter")
    assert com.released == [100]
    assert com.uninitialized == 1


def test_null_interface_is_rejected_and_previous_references_released():
    com = FakeCOM(null_view=True)
    with pytest.raises(OSError, match="empty interface"):
        with h._WindowsPinning(com, method_factory=com.method) as shell:
            shell.view_for(123)
    assert com.released == [300, 200, 100]
    assert com.uninitialized == 1


@pytest.mark.parametrize("initialization,uninitializations", [(1, 1), (-2147417850, 0)])
def test_existing_com_apartment_reference_counts_are_balanced(initialization, uninitializations):
    com = FakeCOM(initialize=initialization)
    with h._WindowsPinning(com, method_factory=com.method):
        pass
    assert com.uninitialized == uninitializations


def test_cli_returns_one_json_object_for_parse_errors(capsys):
    assert h.main(["--hwnd", "invalid", "--pid", "42"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert not result["ok"]


def test_cli_check_passes_only_explicit_identifiers(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(h, "pin_window", lambda hwnd, pid, **kw:
                        calls.append((hwnd, pid, kw)) or {"ok": True, "pinned": False})
    assert h.main(["--hwnd", "123", "--pid", "42", "--check"]) == 0
    assert calls == [(123, 42, {"check_only": True})]
    assert json.loads(capsys.readouterr().out)["ok"]
