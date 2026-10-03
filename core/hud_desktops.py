"""Pin only IntuitionOS's verified Electron HUD across Windows desktops.

Run this module in a short-lived helper process: the shell pinning interfaces
are private Windows interfaces and may change after an OS update. The public
desktop-manager interface is used only for the optional visibility check.

Interface identities and vtable order are corroborated by:
https://github.com/Ciantic/VirtualDesktopAccessor/blob/rust/src/interfaces.rs
https://github.com/MScholtes/VirtualDesktop/blob/master/VirtualDesktop11.cs
No downloaded binary or third-party Python dependency is required.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import threading
import uuid


_HRESULT = ctypes.c_int32
_BOOL = ctypes.c_int32
_U32 = ctypes.c_uint32
_PTR = ctypes.c_void_p


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", _U32), ("Data2", ctypes.c_uint16),
                ("Data3", ctypes.c_uint16), ("Data4", ctypes.c_ubyte * 8)]

    @classmethod
    def parse(cls, value):
        return cls.from_buffer_copy(uuid.UUID(value).bytes_le)


_SHELL = _GUID.parse("C2F03A33-21F5-47FA-B4BB-156362A2F239")
_SERVICE_PROVIDER = _GUID.parse("6D5140C1-7436-11CE-8034-00AA006009FA")
_VIEWS = _GUID.parse("1841C6D7-4F9D-42C0-AF41-8747538F10E5")
_PIN_SERVICE = _GUID.parse("B5A399E7-1C87-46B8-88E9-FC5747B171BD")
_PIN_INTERFACE = _GUID.parse("4CE81583-1E4C-4632-A621-07A53543148F")
_DESKTOP_MANAGER = _GUID.parse("A5CD92FF-29BE-454C-8D04-D82879FB3F1B")


def _bind(function, argtypes, restype):
    try:
        function.argtypes = argtypes
        function.restype = restype
    except AttributeError:  # Bound Python methods in injected test fakes.
        pass
    return function


def _check_hresult(result, operation):
    if ctypes.c_int32(result).value < 0:
        code = result & 0xFFFFFFFF
        raise OSError(f"{operation} failed (HRESULT 0x{code:08X}). "
                      "Windows shell desktop pinning may be unavailable on this build or session.")


def _method(pointer, slot, argtypes, restype=_HRESULT):
    if not pointer or not pointer.value:
        raise OSError("Windows shell returned an empty interface pointer.")
    vtable = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(_PTR))).contents
    address = vtable[slot]
    if not address:
        raise OSError("Windows shell returned an empty interface method.")
    return ctypes.WINFUNCTYPE(restype, _PTR, *argtypes)(address)


class _WindowsPinning:
    """Acquire, use and release shell COM pointers on the same calling thread."""

    def __init__(self, ole32=None, *, method_factory=None):
        self._ole = ole32 if ole32 is not None else ctypes.WinDLL("ole32")
        self._method = method_factory or _method
        self._objects = []
        self._initialized = False
        self._provider = self._views = self._pins = None
        _bind(self._ole.CoInitializeEx, [_PTR, _U32], _HRESULT)
        _bind(self._ole.CoUninitialize, [], None)
        _bind(self._ole.CoCreateInstance,
              [ctypes.POINTER(_GUID), _PTR, _U32, ctypes.POINTER(_GUID),
               ctypes.POINTER(_PTR)], _HRESULT)

    def _take(self, result, pointer, operation):
        # Release even an anomalous non-null output on a failed HRESULT.
        if pointer.value:
            self._objects.append(pointer)
        _check_hresult(result, operation)
        if not pointer.value:
            raise OSError(f"{operation} returned an empty interface pointer.")
        return pointer

    def _query(self, service, interface):
        pointer = _PTR()
        query = self._method(self._provider, 3,
                             [ctypes.POINTER(_GUID), ctypes.POINTER(_GUID),
                              ctypes.POINTER(_PTR)])
        result = query(self._provider, ctypes.byref(service), ctypes.byref(interface),
                       ctypes.byref(pointer))
        return self._take(result, pointer, "QueryService")

    def __enter__(self):
        try:
            result = self._ole.CoInitializeEx(None, 2)  # COINIT_APARTMENTTHREADED.
            if result >= 0:
                self._initialized = True  # S_OK and S_FALSE both need a matching uninitialize.
            elif (result & 0xFFFFFFFF) != 0x80010106:  # Existing MTA: use it, do not uninitialize it.
                _check_hresult(result, "CoInitializeEx")
            provider = _PTR()
            result = self._ole.CoCreateInstance(
                ctypes.byref(_SHELL), None, 4, ctypes.byref(_SERVICE_PROVIDER),
                ctypes.byref(provider))  # CLSCTX_LOCAL_SERVER.
            self._provider = self._take(result, provider, "Create shell service provider")
            self._views = self._query(_VIEWS, _VIEWS)
            self._pins = self._query(_PIN_SERVICE, _PIN_INTERFACE)
            return self
        except Exception:
            self.close()
            raise

    def __exit__(self, *_args):
        self.close()

    def view_for(self, hwnd):
        pointer = _PTR()
        get_view = self._method(self._views, 6, [_PTR, ctypes.POINTER(_PTR)])
        result = get_view(self._views, hwnd, ctypes.byref(pointer))
        return self._take(result, pointer, "Find HUD application view")

    def is_pinned(self, view):
        pinned = _BOOL()
        query = self._method(self._pins, 6, [_PTR, ctypes.POINTER(_BOOL)])
        _check_hresult(query(self._pins, view, ctypes.byref(pinned)), "Check HUD pin")
        return bool(pinned.value)

    def pin(self, view):
        pin = self._method(self._pins, 7, [_PTR])
        _check_hresult(pin(self._pins, view), "Pin HUD to all desktops")

    def on_current_desktop(self, hwnd):
        # This is the documented interface; no private desktop-manager vtable.
        manager = self._query(_DESKTOP_MANAGER, _DESKTOP_MANAGER)
        present = _BOOL()
        query = self._method(manager, 3, [_PTR, ctypes.POINTER(_BOOL)])
        _check_hresult(query(manager, hwnd, ctypes.byref(present)), "Check current desktop")
        return bool(present.value)

    def close(self):
        objects, self._objects = self._objects, []
        try:
            for pointer in reversed(objects):
                try:
                    self._method(pointer, 2, [], _U32)(pointer)  # IUnknown.Release.
                except Exception:
                    # Still release every other acquired reference and apartment.
                    pass
        finally:
            if self._initialized:
                self._initialized = False
                self._ole.CoUninitialize()
            self._provider = self._views = self._pins = None


def _load_windows_api():
    api = ctypes.WinDLL("user32", use_last_error=True)
    _bind(api.IsWindow, [_PTR], _BOOL)
    _bind(api.GetWindowThreadProcessId, [_PTR, ctypes.POINTER(_U32)], _U32)
    for name in ("GetWindowTextW", "GetClassNameW"):
        _bind(getattr(api, name), [_PTR, ctypes.c_wchar_p, ctypes.c_int], ctypes.c_int)
    return api


def _supported_windows():
    return sys.platform == "win32" and sys.getwindowsversion().major >= 10


def _validate_window(api, hwnd, pid):
    if not api.IsWindow(hwnd):
        raise ValueError("The IntuitionOS HUD window no longer exists.")
    actual_pid = _U32()
    if not api.GetWindowThreadProcessId(hwnd, ctypes.byref(actual_pid)) or actual_pid.value != pid:
        raise ValueError("The HUD handle no longer belongs to the expected process.")
    title, window_class = ctypes.create_unicode_buffer(512), ctypes.create_unicode_buffer(256)
    if not api.GetWindowTextW(hwnd, title, len(title)) or title.value != "IntuitionOS":
        raise ValueError("Only the IntuitionOS HUD window can be pinned.")
    if (not api.GetClassNameW(hwnd, window_class, len(window_class))
            or window_class.value != "Chrome_WidgetWin_1"):
        raise ValueError("The target is not the expected IntuitionOS Electron window.")


def pin_window(hwnd: int, pid: int, *, check_only: bool = False,
               window_api=None, shell_factory=None, platform_check=None) -> dict:
    """Pin or inspect one exact HUD HWND/PID, with no focus or desktop changes.

    OS boundaries are injectable for tests. Check mode reports ``pinned=False``
    as a successful query and never calls PinView. The CLI isolates private COM
    from the backend; Electron should also apply a timeout to that process.
    """
    result = {"ok": False, "pinned": False, "hwnd": hwnd, "pid": pid}
    try:
        if (not isinstance(hwnd, int) or isinstance(hwnd, bool) or not 0 < hwnd < 1 << (8 * ctypes.sizeof(_PTR))
                or not isinstance(pid, int) or isinstance(pid, bool) or not 0 < pid < 1 << 32):
            raise ValueError("A positive decimal HWND and process ID are required.")
        if not (platform_check or _supported_windows)():
            raise OSError("HUD desktop pinning requires Windows 10 or later.")
        api = window_api if window_api is not None else _load_windows_api()
        _validate_window(api, hwnd, pid)
        with (shell_factory or _WindowsPinning)() as shell:
            view = shell.view_for(hwnd)
            pinned = shell.is_pinned(view)
            if not pinned and not check_only:
                _validate_window(api, hwnd, pid)
                shell.pin(view)
                pinned = shell.is_pinned(view)
                if not pinned:
                    raise OSError("Windows did not confirm pinning the HUD to all desktops.")
            _validate_window(api, hwnd, pid)
            result.update(ok=True, pinned=pinned)
            try:
                result["on_current_desktop"] = shell.on_current_desktop(hwnd)
            except Exception:
                # Pin verification is authoritative; this public query is extra telemetry.
                result["on_current_desktop"] = None
        return result
    except Exception as exc:
        result["ok"] = False
        result["error"] = str(exc)
        return result


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main(argv=None):
    try:
        parser = _ArgumentParser(description="Keep the IntuitionOS HUD on every Windows desktop.")
        parser.add_argument("--hwnd", type=int, required=True)
        parser.add_argument("--pid", type=int, required=True)
        parser.add_argument("--check", action="store_true", help="Inspect only; do not pin")
        args = parser.parse_args(argv)
        result = pin_window(args.hwnd, args.pid, check_only=args.check)
    except Exception as exc:
        result = {"ok": False, "pinned": False, "error": str(exc)}
    print(json.dumps(result, ensure_ascii=True))
    return 0 if result.get("ok") else 1


def _run_cli():
    # COM can block in the shell. Terminate this isolated worker itself so the
    # venv launcher cannot leave a hung Python child behind after Electron's
    # slightly longer execFile timeout. Importing the library starts no timer.
    def timed_out():
        print(json.dumps({"ok": False, "pinned": False,
                          "error": "Windows desktop pinning timed out."}), flush=True)
        os._exit(2)

    watchdog = threading.Timer(5.0, timed_out)
    watchdog.daemon = True
    watchdog.start()
    try:
        return main()
    finally:
        watchdog.cancel()


if __name__ == "__main__":
    raise SystemExit(_run_cli())
