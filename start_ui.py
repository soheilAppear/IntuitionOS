"""
IntuitionOS HUD launcher.

Starts the FastAPI WebSocket backend, then opens the Electron HUD overlay.
Press Ctrl+C or Ctrl+Q inside the HUD to quit.

Usage:
    python start_ui.py
"""

import os
import platform
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(PROJECT_ROOT, "ui")


def _find_electron():
    is_win = platform.system() == "Windows"
    candidates = [
        os.path.join(UI_DIR, "node_modules", ".bin", "electron.cmd" if is_win else "electron"),
        os.path.join(UI_DIR, "node_modules", "electron", "dist", "electron.exe" if is_win else "electron"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def main():
    # ── 1. Check ui/node_modules exists ──
    if not os.path.isdir(os.path.join(UI_DIR, "node_modules")):
        print("First-time setup: installing Electron…")
        result = subprocess.run(["npm", "install"], cwd=UI_DIR, shell=platform.system() == "Windows")
        if result.returncode != 0:
            print("ERROR: npm install failed. Make sure Node.js is installed.")
            sys.exit(1)

    # ── 2. Start Python backend ──
    print("Starting IntuitionOS backend…")
    server = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "interface.server:app",
            "--host", "127.0.0.1",
            "--port", "7432",
            "--log-level", "warning",
        ],
        cwd=PROJECT_ROOT,
    )

    # Give uvicorn a moment to bind
    time.sleep(2.0)

    if server.poll() is not None:
        print("ERROR: Backend failed to start.")
        print("  • Is Ollama running?  (ollama serve)")
        print("  • Is config/config.yaml valid?")
        sys.exit(1)

    print("Backend ready at ws://127.0.0.1:7432")

    # ── 3. Launch Electron HUD ──
    electron_bin = _find_electron()
    electron = None

    if electron_bin:
        electron = subprocess.Popen([electron_bin, "."], cwd=UI_DIR)
    else:
        # Fall back to npx
        npx = "npx.cmd" if platform.system() == "Windows" else "npx"
        try:
            electron = subprocess.Popen(
                [npx, "electron", "."],
                cwd=UI_DIR,
                shell=platform.system() == "Windows",
            )
        except (FileNotFoundError, OSError) as e:
            print(f"ERROR: Could not launch Electron: {e}")
            print("  Run:  cd ui && npm install")
            server.terminate()
            sys.exit(1)

    print("HUD launched. Use Alt+Space to toggle, Ctrl+Q to quit.")

    try:
        electron.wait()
    except KeyboardInterrupt:
        electron.terminate()
    finally:
        server.terminate()
        print("IntuitionOS stopped.")


if __name__ == "__main__":
    main()
