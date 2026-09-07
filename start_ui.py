"""Launch the backend and Electron HUD as one supervised local session.

The HUD opens only after the backend answers /health. Keep this launcher running;
Ctrl+C here or Ctrl+Q in the HUD stops the children owned by this invocation.

Usage: python start_ui.py
"""

import http.client
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(PROJECT_ROOT, "ui")
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 7432
STARTUP_TIMEOUT = 60.0


def _find_electron():
    """Prefer the native binary so the launcher owns Electron, not a CMD shim."""
    system = platform.system()
    dist = os.path.join(UI_DIR, "node_modules", "electron", "dist")
    if system == "Windows":
        candidates = [os.path.join(dist, "electron.exe")]
    elif system == "Darwin":
        candidates = [
            os.path.join(dist, "Electron.app", "Contents", "MacOS", "Electron")
        ]
    else:
        candidates = [os.path.join(dist, "electron")]
    if system != "Windows":
        candidates.append(os.path.join(UI_DIR, "node_modules", ".bin", "electron"))
    return next((path for path in candidates if os.path.isfile(path)), None)


def _ensure_electron():
    """Install the declared local dependency if absent, then require its binary."""
    electron = _find_electron()
    if electron:
        return electron
    is_windows = platform.system() == "Windows"
    npm = shutil.which("npm.cmd" if is_windows else "npm")
    if not npm:
        raise RuntimeError(
            "Node.js/npm is missing. Install Node.js, then run npm install in ui."
        )
    print("Installing the local Electron dependency…", flush=True)
    if is_windows:
        # A .cmd file needs CMD. Use a fixed install command, with both paths
        # quoted, instead of passing the npm shim directly to CreateProcess.
        comspec = os.environ.get("COMSPEC", "cmd.exe")
        command = f'"{comspec}" /d /s /c ""{npm}" install"'
    else:
        command = [npm, "install"]
    result = subprocess.run(command, cwd=UI_DIR)
    if result.returncode:
        raise RuntimeError(
            "npm install failed. Read its output above, then retry from the ui directory."
        )
    electron = _find_electron()
    if not electron:
        raise RuntimeError(
            "Electron's binary is missing after installation. Run npm rebuild electron in ui, then retry."
        )
    return electron


def _port_in_use():
    try:
        with socket.create_connection((BACKEND_HOST, BACKEND_PORT), timeout=0.5):
            return True
    except OSError:
        return False


def _backend_healthy():
    """Check local API readiness directly, without environment proxy settings."""
    connection = http.client.HTTPConnection(BACKEND_HOST, BACKEND_PORT, timeout=0.5)
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        if response.status != 200:
            return False
        payload = json.loads(response.read(4096))
        return (
            isinstance(payload, dict)
            and payload.get("ok") is True
            and payload.get("version") == "1.0"
        )
    except (OSError, http.client.HTTPException, ValueError):
        return False
    finally:
        connection.close()


def _check_port_available():
    """Diagnose existing listeners; never stop or adopt another process."""
    if not _port_in_use():
        return
    address = f"{BACKEND_HOST}:{BACKEND_PORT}"
    if _backend_healthy():
        raise RuntimeError(
            f"A backend is already responding on {address}. Keep its launcher running "
            "and open only the overlay with cd ui followed by npm start. "
            "To restart everything, close the existing launcher first."
        )
    raise RuntimeError(
        f"Port {address} is occupied, but /health is not ready. An existing backend "
        "may still be starting, or another service may own the port. Check its terminal "
        "or identify the listener before retrying; this launcher has not stopped it."
    )


def _wait_for_backend(server, timeout=None):
    """Wait for completed API startup while checking the exact child we spawned."""
    if timeout is None:
        timeout = STARTUP_TIMEOUT
        if os.environ.get("INTUITION_SHELL", "").lower() in {"powershell", "pwsh"}:
            from core.shell_environment import POWERSHELL_DISCOVERY_TIMEOUT

            # Shell discovery runs before /health becomes available. Allow its
            # complete setup budget in addition to ordinary backend startup.
            timeout += POWERSHELL_DISCOVERY_TIMEOUT
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code = server.poll()
        if code is not None:
            raise RuntimeError(
                f"Backend exited during startup (code {code}). Read its error above. "
                "Check the active Python environment, requirements.txt and config/config.yaml."
            )
        if _backend_healthy() and server.poll() is None:
            return
        time.sleep(0.2)
    raise RuntimeError(
        f"Backend did not become ready within {timeout:g} seconds. Read its startup "
        "output above and check configuration/dependencies before retrying."
    )


def _stop_process(process):
    """Reap only a child owned by this launcher, escalating if it will not exit."""
    if process is None or process.poll() is not None:
        return
    try:
        if platform.system() == "Windows":
            # A venv python.exe can be a redirector with the real interpreter
            # below it. Terminating that parent first would orphan the backend.
            # Stop its still-attached tree by this Popen's PID, never by name.
            result = subprocess.run(
                ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode and process.poll() is None:
                raise OSError(
                    result.stderr.strip()
                    or f"taskkill exited with code {result.returncode}"
                )
            process.wait(timeout=5)
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"Could not finish stopping a launcher child: {error}", flush=True)


def _monitor(server, electron):
    """Keep both processes supervised instead of leaving an offline HUD open."""
    while True:
        code = electron.poll()
        if code is not None:
            if code:
                raise RuntimeError(
                    f"Electron exited with code {code}. Read its output above."
                )
            return
        code = server.poll()
        if code is not None:
            raise RuntimeError(
                f"Backend stopped unexpectedly (code {code}); closing this launcher's HUD. "
                "Read the backend error above, then restart python start_ui.py."
            )
        time.sleep(0.25)


def main():
    server = electron = None
    try:
        _check_port_available()
        electron_bin = _ensure_electron()
        print("Starting IntuitionOS backend; waiting for /health…", flush=True)
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "interface.server:app",
                "--host",
                BACKEND_HOST,
                "--port",
                str(BACKEND_PORT),
                "--log-level",
                "warning",
            ],
            cwd=PROJECT_ROOT,
        )
        _wait_for_backend(server)
        print(f"Backend ready at ws://{BACKEND_HOST}:{BACKEND_PORT}/ws", flush=True)
        # Some development hosts run on Electron and export this switch. It
        # would turn our HUD binary into a Node process instead of opening a UI.
        # Remove it only for this child; keep the caller's environment intact.
        electron_env = os.environ.copy()
        electron_env.pop("ELECTRON_RUN_AS_NODE", None)
        electron = subprocess.Popen([electron_bin, "."], cwd=UI_DIR, env=electron_env)
        print(
            "HUD launched. Keep this terminal open. Alt+Space toggles; Ctrl+Q quits.",
            flush=True,
        )
        _monitor(server, electron)
        return 0
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError) as error:
        print(f"ERROR: {error}", flush=True)
        return 1
    finally:
        _stop_process(electron)
        _stop_process(server)
        if server is not None or electron is not None:
            print("IntuitionOS stopped.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
