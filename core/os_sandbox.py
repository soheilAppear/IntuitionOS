"""OS interaction sandbox — open apps, screenshot, process management, clipboard,
system info, and basic input control.

All actions are explicit and named. Nothing runs automatically.
"""

import ipaddress
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from html import unescape
from typing import Optional
from urllib.parse import urlsplit
import webbrowser

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
    os.path.join(_LOCAL, "Microsoft", "Edge", "Application"),
    os.path.join(_PROG, "Microsoft", "Edge", "Application"),
    os.path.join(_PROG86, "Microsoft", "Edge", "Application"),
    os.path.join(_PROG, "Mozilla Firefox"),
    os.path.join(_PROG86, "Mozilla Firefox"),
    os.path.join(_LOCAL, "Mozilla Firefox"),
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
    # App Paths covers registered applications (for example Office) that do not
    # live on PATH. Read the executable itself; never dispatch through cmd/start.
    if not any(separator in exe for separator in ("/", "\\")):
        try:
            import winreg
            key_name = rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{exe}"
            for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
                    try:
                        with winreg.OpenKey(hive, key_name, 0, winreg.KEY_READ | view) as key:
                            path, _ = winreg.QueryValueEx(key, None)
                        if isinstance(path, str):
                            path = os.path.expandvars(path).strip('"')
                            if os.path.isfile(path):
                                return path
                    except OSError:
                        continue
        except ImportError:
            pass
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
    if not isinstance(name, str) or not name.strip():
        return {"error": "An application name is required."}
    key = name.lower().strip()
    if "://" in key:
        return {"error": "Use the open URL action to open a website."}
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
                if sys.platform == "win32":
                    os.startfile(found)
                else:
                    subprocess.Popen([found], shell=False)
                return {"ok": True, "launched": found}
            except Exception as e:
                return {"error": str(e)}

    return {"error": f"Could not find '{name}'. Try typing the exact executable name."}


def normalize_url(url: str) -> str:
    """Validate an HTTP(S) destination and add HTTPS when no scheme is given.

    Preserve the URL's spelling, including case-sensitive paths and queries.
    This helper does not contact the destination or launch anything.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("A website URL is required.")
    value = url.strip()
    if any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError("Website URLs cannot contain spaces or control characters.")
    if any(c in value for c in '\\<>"'):
        raise ValueError("Use an HTTP or HTTPS website URL with a valid hostname.")
    # Host:port is a destination, while arbitrary URI schemes are not.
    if "://" not in value:
        first = value.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
        if ":" in first and not first.startswith("["):
            _, port = first.rsplit(":", 1)
            if not port.isdigit():
                raise ValueError("Only HTTP and HTTPS website URLs are supported.")
        value = "https://" + value
    try:
        parts = urlsplit(value)
        host = parts.hostname
        port = parts.port  # Also checks invalid or out-of-range ports.
    except ValueError as exc:
        raise ValueError("Use a valid HTTP or HTTPS website URL.") from exc
    if parts.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only HTTP and HTTPS website URLs are supported.")
    if not host or parts.username is not None or parts.password is not None:
        raise ValueError("Use a website URL with a hostname and no embedded credentials.")
    if port == 0:
        raise ValueError("Website ports must be between 1 and 65535.")
    if ":" not in host:
        try:
            ascii_host = host.encode("idna").decode("ascii").rstrip(".")
        except UnicodeError as exc:
            raise ValueError("Use a website URL with a valid hostname.") from exc
        labels = ascii_host.split(".")
        if len(ascii_host) > 253 or any(
            not label or len(label) > 63
            or not label[0].isalnum() or not label[-1].isalnum()
            or any(not (c.isalnum() or c == "-") for c in label)
            for label in labels
        ):
            raise ValueError("Use a website URL with a valid hostname.")
    return value


_FETCH_TIMEOUT_S = 12.0
_FETCH_MAX_BYTES = 2_000_000
_SCRIPT_STYLE = re.compile(r"<(script|style|noscript)\b.*?</\1>", re.I | re.S)
_TAGS = re.compile(r"<[^>]+>")
_BLANKS = re.compile(r"\n\s*\n\s*")


def _refuse_internal_address(host: str) -> Optional[str]:
    """Refuse a host that resolves anywhere inside this machine or network.

    Without this the model can reach the loopback interface — Ollama's own API
    on 11434, the IntuitionOS backend on 7432 — plus the router, anything else
    on the LAN, and cloud metadata endpoints. A page fetch is supposed to read
    the public web, so everything else is refused by address rather than by
    hostname spelling.

    This is a check at resolution time, not a guarantee: a name that resolves
    again between this call and the connection can point somewhere else, and
    a redirect is followed by the HTTP library. It raises the cost of reaching
    the private network; it does not make it impossible.
    """
    try:
        resolved = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return f"Could not resolve {host}."
    for family, _type, _proto, _canon, sockaddr in resolved:
        if family not in (socket.AF_INET, socket.AF_INET6):
            continue
        try:
            address = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            return f"Could not interpret the address for {host}."
        if (address.is_private or address.is_loopback or address.is_link_local
                or address.is_reserved or address.is_multicast
                or address.is_unspecified):
            return (
                f"Refusing to fetch {host}: it resolves to {address}, which is on "
                "this machine or a private network rather than the public web."
            )
    return None


def html_to_text(html: str) -> str:
    """Reduce a page to readable text without pulling in a parser dependency."""
    text = _SCRIPT_STYLE.sub(" ", html)
    text = _TAGS.sub(" ", text)
    text = unescape(text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _BLANKS.sub("\n\n", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def fetch_url(url: str, max_chars: int = 4000) -> dict:
    """Fetch a public web page and return its readable text.

    This is the read that `open_url` cannot do. Opening a page in a browser
    hands it to the user; it can never hand the contents back to the model, so
    a question like "how is the weather" had no tool that could answer it.
    """
    try:
        destination = normalize_url(url)
    except ValueError as exc:
        return {"error": str(exc)}

    host = urlsplit(destination).hostname or ""
    refusal = _refuse_internal_address(host)
    if refusal:
        return {"error": refusal}

    try:
        limit = max(200, min(int(max_chars), 20000))
    except (TypeError, ValueError):
        limit = 4000

    try:
        import requests
    except ImportError:
        return {"error": "requests is not installed — run: pip install requests"}

    try:
        response = requests.get(
            destination,
            timeout=_FETCH_TIMEOUT_S,
            stream=True,
            headers={"User-Agent": "IntuitionOS/1.0 (local assistant)"},
        )
    except Exception as exc:
        return {"error": f"Could not fetch {destination}: {exc}"}

    with response:
        content_type = (response.headers.get("Content-Type") or "").lower()
        if not any(t in content_type for t in ("text/", "json", "xml", "")):
            return {"error": f"{destination} returned {content_type or 'unknown'} content, not a readable page."}
        try:
            raw = response.raw.read(_FETCH_MAX_BYTES, decode_content=True) or b""
        except Exception as exc:
            return {"error": f"Could not read {destination}: {exc}"}
        if not response.ok:
            return {"error": f"{destination} returned HTTP {response.status_code}.",
                    "status_code": response.status_code}

    body = raw.decode(response.encoding or "utf-8", errors="replace")
    text = html_to_text(body) if "html" in content_type else body.strip()
    truncated = len(text) > limit
    return {
        "ok": True,
        "url": response.url,
        "status_code": response.status_code,
        "truncated": truncated,
        "text": text[:limit] + ("… (truncated)" if truncated else ""),
    }


def open_url(url: str, browser: str = "default") -> dict:
    """Ask a browser to open an HTTP(S) URL, without claiming the page loaded."""
    try:
        destination = normalize_url(url)
    except ValueError as exc:
        return {"error": str(exc)}
    if not isinstance(browser, str) or browser not in {"default", "chrome", "edge", "firefox"}:
        return {"error": "Choose default, chrome, edge, or firefox as the browser."}
    try:
        if browser == "default":
            if sys.platform == "win32":
                os.startfile(destination)
            elif not webbrowser.open(destination, new=2):
                return {"error": "No default browser accepted the URL. Choose an installed browser."}
        else:
            executable = next(
                (found for name in _APP_ALIASES[browser] if (found := _find_exe(name))),
                None,
            )
            if executable is None:
                return {"error": f"Could not find {browser}. Install it or choose another installed browser."}
            subprocess.Popen([executable, destination], shell=False)
    except Exception as exc:
        return {"error": f"Could not open the website in {browser}: {exc}"}
    return {
        "ok": True,
        "url": destination,
        "browser": browser,
        "status": "open_requested",
        "message": f"Sent {destination} to {browser if browser != 'default' else 'your default browser'}. Page loading is not verified.",
    }


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

_MEDIA_KEYS = {
    "play_pause": 0xB3, "next_track": 0xB0, "previous_track": 0xB1,
    "stop": 0xB2, "mute": 0xAD, "volume_down": 0xAE, "volume_up": 0xAF,
}


def media_key(key: str) -> dict:
    """Send one media key, the same event a keyboard's media button sends.

    Goes to the shell rather than a particular window, so whichever player is
    running responds — which is what makes a gesture for "next track" useful
    without having to find and focus the player first.
    """
    name = (key or "").strip().lower()
    code = _MEDIA_KEYS.get(name)
    if code is None:
        return {"error": f"Choose one of: {', '.join(sorted(_MEDIA_KEYS))}."}
    user32 = _user32()
    if user32 is None:
        return {"error": "Media keys are only implemented on Windows."}
    _KEYEVENTF_KEYUP = 0x0002
    user32.keybd_event(code, 0, 0, 0)
    user32.keybd_event(code, 0, _KEYEVENTF_KEYUP, 0)
    return {"ok": True, "key": name}


# ── Window control ────────────────────────────────────────────────────────────
#
# Done with ctypes against user32 rather than pywin32, which would be a new
# binary dependency for a handful of calls this module can make directly.
# Everything here is reversible: a window that was moved can be moved back, and
# the gate is what says so.


_SW_MINIMIZE, _SW_MAXIMIZE, _SW_RESTORE = 6, 3, 9
_SWP_NOZORDER, _SWP_NOACTIVATE = 0x0004, 0x0010


def _user32():
    if sys.platform != "win32":
        return None
    import ctypes
    return ctypes.WinDLL("user32", use_last_error=True)


def _window_title(user32, handle) -> str:
    import ctypes
    length = user32.GetWindowTextLengthW(handle)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(handle, buffer, length + 1)
    return buffer.value


def _enumerate_windows(user32) -> list:
    """Every visible, titled, non-minimised-to-tray top-level window."""
    import ctypes
    from ctypes import wintypes

    found = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def collect(handle, _param):
        if not user32.IsWindowVisible(handle):
            return True
        title = _window_title(user32, handle)
        if title:
            found.append((handle, title))
        return True

    user32.EnumWindows(callback_type(collect), 0)
    return found


def _resolve_window(user32, title: str):
    """Find one window by a case-insensitive substring of its title.

    Returns (handle, error). Ambiguity is an error rather than a guess: moving
    the wrong window is confusing and the caller can always be more specific.
    """
    windows = _enumerate_windows(user32)
    if not title or not title.strip():
        handle = user32.GetForegroundWindow()
        if not handle:
            return None, "There is no active window."
        return handle, None
    needle = title.strip().lower()
    matches = [(h, t) for h, t in windows if needle in t.lower()]
    if not matches:
        return None, f"No visible window matching {title!r}."
    exact = [(h, t) for h, t in matches if t.lower() == needle]
    if exact:
        return exact[0][0], None
    if len(matches) > 1:
        names = ", ".join(repr(t) for _h, t in matches[:5])
        return None, f"{len(matches)} windows match {title!r}: {names}. Be more specific."
    return matches[0][0], None


def _work_area(user32) -> tuple:
    """The desktop minus the taskbar, so a snapped window is not covered."""
    import ctypes
    from ctypes import wintypes

    class RECT(ctypes.Structure):
        _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                    ("right", wintypes.LONG), ("bottom", wintypes.LONG)]

    rect = RECT()
    if user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
        return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
    return 0, 0, user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def window_geometry(title: str = "") -> dict:
    """Where a window currently is, so a move can be undone."""
    import ctypes
    from ctypes import wintypes

    user32 = _user32()
    if user32 is None:
        return {"error": "Window control is only implemented on Windows."}
    handle, error = _resolve_window(user32, title)
    if error:
        return {"error": error}

    class RECT(ctypes.Structure):
        _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                    ("right", wintypes.LONG), ("bottom", wintypes.LONG)]

    rect = RECT()
    if not user32.GetWindowRect(handle, ctypes.byref(rect)):
        return {"error": "Could not read the window's position."}
    return {
        "ok": True, "title": _window_title(user32, handle),
        "x": rect.left, "y": rect.top,
        "width": rect.right - rect.left, "height": rect.bottom - rect.top,
    }


def move_window(title: str = "", x: int = 0, y: int = 0,
                width: int = 0, height: int = 0) -> dict:
    """Move and optionally resize a window. Zero width/height keeps the current size."""
    user32 = _user32()
    if user32 is None:
        return {"error": "Window control is only implemented on Windows."}
    handle, error = _resolve_window(user32, title)
    if error:
        return {"error": error}

    current = window_geometry(_window_title(user32, handle))
    if current.get("error"):
        return current
    width = int(width) or current["width"]
    height = int(height) or current["height"]
    if width < 100 or height < 100:
        return {"error": "A window smaller than 100x100 would be unusable."}

    user32.ShowWindow(handle, _SW_RESTORE)
    if not user32.SetWindowPos(handle, None, int(x), int(y), width, height,
                               _SWP_NOZORDER | _SWP_NOACTIVATE):
        return {"error": "The window refused to move."}
    return {"ok": True, "title": current["title"], "x": int(x), "y": int(y),
            "width": width, "height": height}


def snap_window(title: str = "", position: str = "left") -> dict:
    """Snap a window to a half, quarter, or the full work area."""
    user32 = _user32()
    if user32 is None:
        return {"error": "Window control is only implemented on Windows."}
    handle, error = _resolve_window(user32, title)
    if error:
        return {"error": error}

    origin_x, origin_y, full_w, full_h = _work_area(user32)
    half_w, half_h = full_w // 2, full_h // 2
    layouts = {
        "left":         (origin_x,          origin_y,          half_w, full_h),
        "right":        (origin_x + half_w, origin_y,          half_w, full_h),
        "top":          (origin_x,          origin_y,          full_w, half_h),
        "bottom":       (origin_x,          origin_y + half_h, full_w, half_h),
        "top-left":     (origin_x,          origin_y,          half_w, half_h),
        "top-right":    (origin_x + half_w, origin_y,          half_w, half_h),
        "bottom-left":  (origin_x,          origin_y + half_h, half_w, half_h),
        "bottom-right": (origin_x + half_w, origin_y + half_h, half_w, half_h),
        "full":         (origin_x,          origin_y,          full_w, full_h),
    }
    key = (position or "").strip().lower().replace("_", "-").replace(" ", "-")
    if key not in layouts:
        return {"error": f"Choose one of: {', '.join(sorted(layouts))}."}

    x, y, width, height = layouts[key]
    user32.ShowWindow(handle, _SW_RESTORE)
    if not user32.SetWindowPos(handle, None, x, y, width, height,
                               _SWP_NOZORDER | _SWP_NOACTIVATE):
        return {"error": "The window refused to move."}
    return {"ok": True, "title": _window_title(user32, handle), "position": key,
            "x": x, "y": y, "width": width, "height": height}


def set_window_state(title: str = "", state: str = "restore") -> dict:
    """Minimise, maximise or restore a window."""
    user32 = _user32()
    if user32 is None:
        return {"error": "Window control is only implemented on Windows."}
    handle, error = _resolve_window(user32, title)
    if error:
        return {"error": error}
    commands = {"minimize": _SW_MINIMIZE, "maximize": _SW_MAXIMIZE, "restore": _SW_RESTORE}
    key = (state or "").strip().lower()
    if key not in commands:
        return {"error": "Choose minimize, maximize or restore."}
    name = _window_title(user32, handle)
    user32.ShowWindow(handle, commands[key])
    return {"ok": True, "title": name, "state": key}


def focus_window(title: str = "") -> dict:
    """Bring a window to the front."""
    user32 = _user32()
    if user32 is None:
        return {"error": "Window control is only implemented on Windows."}
    handle, error = _resolve_window(user32, title)
    if error:
        return {"error": error}
    name = _window_title(user32, handle)
    user32.ShowWindow(handle, _SW_RESTORE)
    # A process that does not own the foreground cannot simply take it, so the
    # input queues are briefly attached. Failure here is not fatal: the window
    # is still restored and flashing in the taskbar.
    user32.SetForegroundWindow(handle)
    return {"ok": True, "title": name, "focused": True}


def cycle_window(direction: str = "next") -> dict:
    """Focus the next or previous visible window, in a stable order."""
    user32 = _user32()
    if user32 is None:
        return {"error": "Window control is only implemented on Windows."}
    windows = [(h, t) for h, t in _enumerate_windows(user32)]
    if len(windows) < 2:
        return {"error": "There are not two visible windows to switch between."}
    current = user32.GetForegroundWindow()
    handles = [h for h, _t in windows]
    step = -1 if (direction or "next").strip().lower() in ("previous", "prev", "back") else 1
    try:
        index = handles.index(current)
    except ValueError:
        index = -1 if step == 1 else 0
    target, name = windows[(index + step) % len(windows)]
    user32.ShowWindow(target, _SW_RESTORE)
    user32.SetForegroundWindow(target)
    return {"ok": True, "title": name, "direction": "previous" if step < 0 else "next"}


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
