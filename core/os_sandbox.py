"""OS interaction sandbox — open apps, screenshot, process management, clipboard,
system info, and basic input control.

All actions are explicit and named. Nothing runs automatically.
"""

import os
import shutil
import subprocess
import sys
import time

import psutil

# ── App launcher ──────────────────────────────────────────────────────────────

_HOME = os.path.expanduser("~")
_LOCAL = os.environ.get("LOCALAPPDATA", os.path.join(_HOME, "AppData", "Local"))
_ROAMING = os.environ.get("APPDATA", os.path.join(_HOME, "AppData", "Roaming"))
_PROG = r"C:\Program Files"
_PROG86 = r"C:\Program Files (x86)"

_APP_ALIASES: dict[str, list[str]] = {
    "chrome":        ["chrome.exe", "google-chrome", "chromium"],
    "firefox":       ["firefox.exe", "firefox"],
    "edge":          ["msedge.exe"],
    "vscode":        ["code.exe", "code"],
    "notepad":       ["notepad.exe"],
    "explorer":      ["explorer.exe"],
    "terminal":      ["wt.exe", "cmd.exe"],
    "calculator":    ["calc.exe"],
    "paint":         ["mspaint.exe"],
    "task manager":  ["taskmgr.exe"],
    "control panel": ["control.exe"],
    "settings":      ["ms-settings:"],
    "spotify":       ["Spotify.exe"],
    "discord":       ["Discord.exe"],
    "slack":         ["slack.exe"],
    "word":          ["WINWORD.EXE"],
    "excel":         ["EXCEL.EXE"],
    "powerpoint":    ["POWERPNT.EXE"],
    "teams":         ["ms-teams.exe", "Teams.exe"],
    "obs":           ["obs64.exe", "obs.exe"],
    "vlc":           ["vlc.exe"],
    "steam":         ["steam.exe"],
}

# Common install paths Windows doesn't always put in PATH
_WIN_EXTRA_PATHS: list[str] = [
    os.path.join(_LOCAL, "Google", "Chrome", "Application"),
    os.path.join(_PROG, "Google", "Chrome", "Application"),
    os.path.join(_PROG86, "Google", "Chrome", "Application"),
    os.path.join(_PROG, "Mozilla Firefox"),
    os.path.join(_PROG86, "Mozilla Firefox"),
    os.path.join(_LOCAL, "Programs", "Microsoft VS Code"),
    os.path.join(_PROG, "Microsoft VS Code"),
    os.path.join(_ROAMING, "Spotify"),
    os.path.join(_LOCAL, "Discord", "app-*"),  # wildcard handled below
    os.path.join(_ROAMING, "Microsoft", "Teams"),
    os.path.join(_PROG, "VideoLAN", "VLC"),
    os.path.join(_PROG86, "VideoLAN", "VLC"),
    os.path.join(_PROG, "OBS Studio"),
    r"C:\Program Files (x86)\Steam",
    r"C:\Program Files\Steam",
    os.path.join(_LOCAL, "Steam"),
]


def _find_exe(exe: str) -> str | None:
    """Find exe in PATH or Windows install directories."""
    found = shutil.which(exe)
    if found:
        return found
    if sys.platform != "win32":
        return None
    import glob
    for base in _WIN_EXTRA_PATHS:
        # Handle glob patterns (e.g. Discord versioned dirs)
        candidates = glob.glob(base) if "*" in base else [base]
        for d in candidates:
            full = os.path.join(d, exe)
            if os.path.isfile(full):
                return full
    return None


def open_app(name: str) -> dict:
    """Launch an application by friendly name or executable path."""
    key = name.lower().strip()
    candidates = _APP_ALIASES.get(key, [name])

    for exe in candidates:
        if exe.startswith("ms-settings:"):
            try:
                os.startfile(exe)
                return {"ok": True, "launched": exe}
            except Exception:
                continue

        found = _find_exe(exe)
        if found:
            try:
                os.startfile(found)  # ShellExecuteEx — the Windows-native way to open apps
                return {"ok": True, "launched": found}
            except Exception as e:
                return {"error": str(e)}

        if sys.platform == "win32":
            # Last resort: Windows `start` command handles PATH + registry lookups
            try:
                subprocess.Popen(
                    ["cmd.exe", "/c", "start", "", exe],
                    creationflags=0x08000000,  # CREATE_NO_WINDOW
                )
                return {"ok": True, "launched": exe}
            except Exception:
                continue

    return {"error": f"Could not find '{name}'. Try typing the exact executable name."}


# ── Screenshot ────────────────────────────────────────────────────────────────

def take_screenshot() -> dict:
    """Take a screenshot and save to data/screenshots/. Returns the file path."""
    try:
        import pyautogui
    except ImportError:
        return {"error": "pyautogui not installed — run: pip install pyautogui"}

    try:
        out_dir = os.path.join("data", "screenshots")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"screenshot_{int(time.time())}.png")
        img = pyautogui.screenshot()
        img.save(path)
        return {"ok": True, "path": path, "size": f"{img.width}x{img.height}"}
    except Exception as e:
        return {"error": str(e)}


# ── Processes ─────────────────────────────────────────────────────────────────

def list_processes(filter_name: str = "") -> dict:
    """List running processes (top 30 by name). Optionally filter by name."""
    procs = []
    for p in psutil.process_iter(["name", "pid", "cpu_percent", "memory_percent"]):
        try:
            info = p.info
            n = info.get("name") or ""
            if filter_name and filter_name.lower() not in n.lower():
                continue
            procs.append({
                "pid":  info["pid"],
                "name": n,
                "cpu":  f"{round(info.get('cpu_percent') or 0, 1)}%",
                "mem":  f"{round(info.get('memory_percent') or 0, 1)}%",
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    procs.sort(key=lambda x: x["name"].lower())
    return {"processes": procs[:30]}


def kill_process(name: str) -> dict:
    """Terminate the first process matching the given name (with or without .exe)."""
    nl = name.lower().strip()
    # Build a set of candidate names to match against
    targets = {nl, nl + ".exe", nl.removesuffix(".exe")}
    for p in psutil.process_iter(["name", "pid"]):
        try:
            pname = (p.info.get("name") or "").lower()
            if pname in targets or pname.removesuffix(".exe") in targets:
                p.terminate()
                return {"ok": True, "killed": p.info["name"], "pid": p.info["pid"]}
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return {"error": f"No process named '{name}' found. Try listing processes first."}


# ── Clipboard ─────────────────────────────────────────────────────────────────

def get_clipboard() -> dict:
    """Read current clipboard text (Windows)."""
    try:
        r = subprocess.run(
            ["powershell", "-NonInteractive", "-Command", "Get-Clipboard"],
            capture_output=True, text=True, timeout=5,
        )
        return {"text": r.stdout.rstrip("\n")}
    except Exception as e:
        return {"error": str(e)}


def set_clipboard(text: str) -> dict:
    """Write text to clipboard (Windows)."""
    try:
        subprocess.run(
            ["powershell", "-NonInteractive", "-Command",
             f"Set-Clipboard -Value @'\n{text}\n'@"],
            capture_output=True, timeout=5,
        )
        return {"ok": True}
    except Exception as e:
        return {"error": str(e)}


# ── System info ───────────────────────────────────────────────────────────────

def system_info() -> dict:
    """Return OS, uptime, RAM, and disk usage."""
    import platform
    vm   = psutil.virtual_memory()
    disk = psutil.disk_usage("C:\\" if sys.platform == "win32" else "/")
    return {
        "os":            f"{platform.system()} {platform.release()}",
        "uptime_hours":  round((time.time() - psutil.boot_time()) / 3600, 1),
        "ram_total_gb":  round(vm.total / 1e9, 1),
        "ram_used_pct":  f"{vm.percent}%",
        "disk_total_gb": round(disk.total / 1e9, 1),
        "disk_used_pct": f"{disk.percent}%",
    }


# ── Input control ─────────────────────────────────────────────────────────────

def type_text(text: str) -> dict:
    """Type text at the current cursor position (keyboard simulation)."""
    try:
        import pyautogui
    except ImportError:
        return {"error": "pyautogui not installed"}
    try:
        pyautogui.write(text, interval=0.03)
        return {"ok": True, "typed": len(text)}
    except Exception as e:
        return {"error": str(e)}


def move_mouse(x: int, y: int) -> dict:
    """Move mouse to screen coordinates."""
    try:
        import pyautogui
        pyautogui.moveTo(int(x), int(y), duration=0.25)
        return {"ok": True, "x": x, "y": y}
    except ImportError:
        return {"error": "pyautogui not installed"}
    except Exception as e:
        return {"error": str(e)}


def click(x: int = None, y: int = None, button: str = "left") -> dict:
    """Click at (x, y) or at the current mouse position."""
    try:
        import pyautogui
        if x is not None and y is not None:
            pyautogui.click(int(x), int(y), button=button)
        else:
            pyautogui.click(button=button)
        return {"ok": True}
    except ImportError:
        return {"error": "pyautogui not installed"}
    except Exception as e:
        return {"error": str(e)}


# ── Volume ────────────────────────────────────────────────────────────────────

def set_volume(level: int) -> dict:
    """Set system master volume 0-100 (Windows, no external modules needed)."""
    level = max(0, min(100, int(level)))

    # Primary: PowerShell inline C# — works on Windows 10/11 without extra deps,
    # and has no COM threading issues since it runs in a separate process.
    import tempfile
    scalar = round(level / 100.0, 4)
    ps_script = (
        'Add-Type -TypeDefinition @"\n'
        'using System;\n'
        'using System.Runtime.InteropServices;\n'
        '[Guid("5CDF2C82-841E-4546-9722-0CF74078229A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]\n'
        'interface IAudioEndpointVolume {\n'
        '  int a();int b();int c();int d();int e();int f();int g();int h();\n'
        '  int SetMasterVolumeLevelScalar(float fLevel, Guid ctx);\n'
        '  int i();\n'
        '  int GetMasterVolumeLevelScalar(out float v);\n'
        '}\n'
        '[Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]\n'
        'interface IMMDevice { int Activate([MarshalAs(UnmanagedType.LPStruct)] Guid iid, int ctx, IntPtr p, [MarshalAs(UnmanagedType.IUnknown)] out object pp); }\n'
        '[Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]\n'
        'interface IMMDeviceEnumerator { int x(); int GetDefaultAudioEndpoint(int flow, int role, out IMMDevice dev); }\n'
        '[ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")] class MMDevEnum {}\n'
        'public class AV {\n'
        '  public static void Set(float v) {\n'
        '    var e = new MMDevEnum() as IMMDeviceEnumerator;\n'
        '    IMMDevice d; e.GetDefaultAudioEndpoint(0,1,out d);\n'
        '    object ep; d.Activate(typeof(IAudioEndpointVolume).GUID,23,IntPtr.Zero,out ep);\n'
        '    ((IAudioEndpointVolume)ep).SetMasterVolumeLevelScalar(v,Guid.Empty);\n'
        '  }\n'
        '}\n'
        '"@\n'
        # Note: [float] cast, not C# "f" suffix — PowerShell parses this line, not C#
        f'[AV]::Set([float]{scalar})\n'
    )
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ps1', delete=False, encoding='utf-8') as f:
            f.write(ps_script)
            tmp_ps = f.name
        r = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-NonInteractive", "-File", tmp_ps],
            capture_output=True, timeout=12, text=True,
        )
        try:
            os.unlink(tmp_ps)
        except Exception:
            pass
        if r.returncode == 0:
            return {"ok": True, "volume": level}
        # PowerShell failed — last resort: pycaw (may work if COM is initialized)
        try:
            from pycaw.pycaw import AudioUtilities
            speakers = AudioUtilities.GetSpeakers()
            speakers.EndpointVolume.SetMasterVolumeLevelScalar(level / 100.0, None)
            return {"ok": True, "volume": level}
        except Exception:
            pass
        return {"error": r.stderr.strip() or "Volume adjustment failed"}
    except Exception as e:
        return {"error": str(e)}


def get_volume() -> dict:
    """Return current master volume level 0-100."""
    try:
        from pycaw.pycaw import AudioUtilities
        speakers = AudioUtilities.GetSpeakers()
        current = speakers.EndpointVolume.GetMasterVolumeLevelScalar()
        return {"volume": round(current * 100)}
    except Exception as e:
        return {"error": str(e)}


# ── Power management ──────────────────────────────────────────────────────────

def shutdown_computer(delay_sec: int = 30) -> dict:
    """Schedule a Windows shutdown (default 30-second grace period)."""
    try:
        subprocess.run(["shutdown", "/s", "/t", str(delay_sec)], check=True)
        return {"ok": True, "action": "shutdown", "delay_sec": delay_sec}
    except Exception as e:
        return {"error": str(e)}


def restart_computer(delay_sec: int = 30) -> dict:
    """Schedule a Windows restart (default 30-second grace period)."""
    try:
        subprocess.run(["shutdown", "/r", "/t", str(delay_sec)], check=True)
        return {"ok": True, "action": "restart", "delay_sec": delay_sec}
    except Exception as e:
        return {"error": str(e)}


def cancel_shutdown() -> dict:
    """Cancel a pending Windows shutdown or restart."""
    try:
        subprocess.run(["shutdown", "/a"], check=True)
        return {"ok": True, "action": "cancelled"}
    except subprocess.CalledProcessError:
        return {"error": "No shutdown scheduled to cancel"}
    except Exception as e:
        return {"error": str(e)}


# ── Display / brightness ──────────────────────────────────────────────────────

def set_brightness(level: int) -> dict:
    """Set display brightness 0-100 (requires WMI — built into Windows)."""
    level = max(0, min(100, int(level)))
    ps = (
        f"(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods)"
        f".WmiSetBrightness(1, {level})"
    )
    r = subprocess.run(
        ["powershell", "-NonInteractive", "-Command", ps],
        capture_output=True, text=True, timeout=8,
    )
    if r.returncode == 0:
        return {"ok": True, "brightness": level}
    return {"error": "Cannot set brightness — try updating display drivers or use laptop hotkeys"}


def get_brightness() -> dict:
    """Get current display brightness (WMI)."""
    ps = "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness).CurrentBrightness"
    r = subprocess.run(
        ["powershell", "-NonInteractive", "-Command", ps],
        capture_output=True, text=True, timeout=8,
    )
    val = r.stdout.strip()
    if r.returncode == 0 and val.isdigit():
        return {"brightness": int(val)}
    return {"error": "Could not read brightness"}


# ── Battery ───────────────────────────────────────────────────────────────────

def get_battery() -> dict:
    """Return battery level, charging status, and estimated time remaining."""
    b = psutil.sensors_battery()
    if b is None:
        return {"error": "No battery — this appears to be a desktop"}
    mins = int(b.secsleft / 60) if b.secsleft not in (psutil.POWER_TIME_UNKNOWN, psutil.POWER_TIME_UNLIMITED) else None
    return {
        "percent": f"{round(b.percent)}%",
        "charging": b.power_plugged,
        "time_remaining": f"{mins} min" if mins else ("Calculating…" if b.power_plugged else "Unknown"),
    }


# ── Network ───────────────────────────────────────────────────────────────────

def get_network_info() -> dict:
    """Return active network connections, Wi-Fi SSID, and IP addresses."""
    addrs = psutil.net_if_addrs()
    stats = psutil.net_if_stats()
    ifaces = []
    for name, addr_list in addrs.items():
        st = stats.get(name)
        if not st or not st.isup:
            continue
        ipv4 = next((a.address for a in addr_list if a.family.name == "AF_INET"), None)
        if ipv4 and not ipv4.startswith("127."):
            ifaces.append({"interface": name, "ip": ipv4})

    # Wi-Fi SSID
    ssid = None
    try:
        r = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True, text=True, timeout=5,
        )
        for line in r.stdout.splitlines():
            if "SSID" in line and "BSSID" not in line:
                ssid = line.split(":", 1)[-1].strip()
                break
    except Exception:
        pass

    return {"interfaces": ifaces, "wifi_ssid": ssid}


def toggle_wifi(state: str) -> dict:
    """Enable or disable Wi-Fi adapter (state: 'on' or 'off')."""
    action = "Enable" if state.lower() in ("on", "enable", "1") else "Disable"
    ps = f"Get-NetAdapter | Where-Object {{$_.Name -like '*Wi-Fi*' -or $_.Name -like '*WiFi*' -or $_.Name -like '*Wireless*'}} | {action}-NetAdapter -Confirm:$false"
    r = subprocess.run(
        ["powershell", "-NonInteractive", "-Command", ps],
        capture_output=True, text=True, timeout=10,
    )
    return {"ok": r.returncode == 0, "wifi": state}


# ── Window management ─────────────────────────────────────────────────────────

def list_windows() -> dict:
    """List visible application windows by title."""
    ps = (
        "Add-Type -Name Win -Namespace Native -MemberDefinition "
        "'[DllImport(\"user32.dll\")] public static extern bool IsWindowVisible(IntPtr h); "
        "[DllImport(\"user32.dll\")] public static extern int GetWindowTextLength(IntPtr h); "
        "[DllImport(\"user32.dll\")] public static extern int GetWindowText(IntPtr h, System.Text.StringBuilder s, int n);' 2>$null; "
        "[System.Diagnostics.Process]::GetProcesses() | "
        "Where-Object {$_.MainWindowHandle -ne 0 -and $_.MainWindowTitle -ne ''} | "
        "Select-Object -Property @{n='pid';e={$_.Id}},@{n='title';e={$_.MainWindowTitle}},@{n='name';e={$_.ProcessName}} | "
        "ConvertTo-Json"
    )
    r = subprocess.run(
        ["powershell", "-NonInteractive", "-Command", ps],
        capture_output=True, text=True, timeout=10,
    )
    try:
        import json as _json
        wins = _json.loads(r.stdout)
        if isinstance(wins, dict):
            wins = [wins]
        return {"windows": [{"pid": w["pid"], "title": w["title"], "app": w["name"]} for w in wins]}
    except Exception:
        return {"error": "Could not list windows"}


# ── Sleep / lock ──────────────────────────────────────────────────────────────

def sleep_computer() -> dict:
    """Put the computer to sleep immediately."""
    try:
        subprocess.Popen(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"])
        return {"ok": True}
    except Exception as e:
        return {"error": str(e)}


def lock_screen() -> dict:
    """Lock the Windows screen."""
    try:
        subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"])
        return {"ok": True}
    except Exception as e:
        return {"error": str(e)}
