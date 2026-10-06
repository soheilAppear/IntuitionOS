"""Optional BrainBit connection adapter with isolated, bounded SDK operations."""

import copy
import importlib.util
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import uuid

from .hw_base import HardwareDriver


TRANSPORT = "Windows Bluetooth LE; SDK does not identify radio/dongle"
MAX_FRAME = 65536


class _WorkerClient:
    """One request at a time over bounded JSON frames; never load the SDK here."""

    def __init__(self, on_event, on_exit, command=None):
        self._responses = queue.Queue(maxsize=4)
        self._closed = threading.Event()
        self._expected_exit = threading.Event()
        self._on_event = on_event
        self._on_exit = on_exit
        if command is None:
            # Windows venv python.exe can be a redirector with a child interpreter.
            # Launch the base binary directly so kill() owns the SDK process.
            executable = getattr(sys, "_base_executable", None) or sys.executable
            command = [executable, "-I", "-u", "-c",
                       "import json,runpy,sys; sys.path[:]=json.loads(sys.argv[1]); "
                       "runpy.run_path(sys.argv[2],run_name='__main__')",
                       json.dumps(sys.path), str(Path(__file__).with_name("brainbit_worker.py"))]
        self._process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self._reader = threading.Thread(target=self._read, daemon=True,
                                        name="brainbit-ipc")
        self._reader.start()

    def _read(self):
        try:
            while not self._closed.is_set():
                line = self._process.stdout.readline(MAX_FRAME + 1)
                if not line:
                    break
                if len(line) > MAX_FRAME or not line.endswith("\n"):
                    break
                message = json.loads(line)
                if not isinstance(message, dict):
                    break
                if "event" in message:
                    self._on_event(message["event"])
                else:
                    if message.get("error"):
                        self._expected_exit.set()
                    self._responses.put_nowait(message)
        except (OSError, ValueError, queue.Full):
            pass
        finally:
            try:
                self._responses.put_nowait(None)
            except queue.Full:
                pass
            if not self._closed.is_set() and not self._expected_exit.is_set():
                self._on_exit()

    def request(self, action, args, *, timeout, scan_seconds):
        request_id = str(uuid.uuid4())
        data = json.dumps({"id": request_id, "action": action, "args": args,
                           "scan_seconds": scan_seconds}) + "\n"
        if len(data) > MAX_FRAME:
            raise ValueError("BrainBit request is too large")
        if self._closed.is_set() or self._process.poll() is not None:
            raise OSError("BrainBit worker stopped")
        if action == "disconnect":
            self._expected_exit.set()
        self._process.stdin.write(data)
        self._process.stdin.flush()
        try:
            message = self._responses.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("BrainBit operation timed out") from None
        if message is None or message.get("id") != request_id:
            raise OSError("BrainBit worker stopped")
        return message

    def close(self):
        if self._closed.is_set():
            return
        self._closed.set()
        # Kill first: closing a pipe before the reader exits can wait on its I/O
        # lock forever when the native SDK is stuck.
        if self._process.poll() is None:
            try:
                self._process.kill()
            except OSError:
                pass
        try:
            self._process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        if threading.current_thread() is not self._reader:
            self._reader.join(timeout=1)
        for stream in (self._process.stdin, self._process.stdout):
            if stream is not None and (stream is self._process.stdin or not self._reader.is_alive()):
                try:
                    stream.close()
                except OSError:
                    pass


class BrainBit(HardwareDriver):
    """Explicit discovery/connect/disconnect with immediate cached status.

    ``call`` is synchronous and belongs on an executor; ``status`` only copies
    cached data. Disconnect cancels an in-flight operation by killing its worker.
    Closing permanently disables this instance. No samples are acquired.
    """

    name = "brainbit"

    def __init__(self, enabled=True, scan_seconds=5, connect_timeout=15,
                 request_timeout=20):
        self._lock = threading.RLock()
        self._worker = None
        self._closed = not bool(enabled)
        self._epoch = 0
        self._revision = 0
        self._scan_seconds = min(15.0, max(0.0, float(scan_seconds)))
        self._connect_timeout = max(0.05, min(60.0, float(connect_timeout)))
        self._request_timeout = max(0.05, min(60.0, float(request_timeout)))
        available = False
        if enabled:
            try:
                available = importlib.util.find_spec("neurosdk") is not None
            except (ImportError, ValueError):
                pass
        state = "disabled" if not enabled else "disconnected" if available else "unavailable"
        self._snapshot = {
            "state": state, "available": available, "busy": False,
            "text": "BrainBit integration disabled" if not enabled else
                    "Ready to discover BrainBit devices" if available else
                    "Install the optional pyneurosdk2 package for this Python runtime",
            "devices": [], "device": None, "transport": TRANSPORT, "revision": 0,
        }

    def schema(self):
        return {"actions": [
            {"name": "discover", "args": [], "confirm": False},
            {"name": "connect", "args": ["device_id"], "confirm": False},
            {"name": "disconnect", "args": [], "confirm": False},
            {"name": "status", "args": ["refresh"], "confirm": False},
        ]}

    def status(self):
        with self._lock:
            return copy.deepcopy(self._snapshot)

    def _update(self, **values):
        self._snapshot.pop("error", None)
        self._snapshot.update(values)
        self._revision += 1
        self._snapshot["revision"] = self._revision

    def _reject(self, message):
        result = self.status()
        result["error"] = message
        return result

    def _event(self, epoch, payload):
        if not isinstance(payload, dict):
            return
        stopped = None
        with self._lock:
            if epoch != self._epoch or self._closed:
                return
            # Completion owns the busy flag. Callbacks may still report battery
            # or link loss while a metadata refresh is waiting on the DLL.
            values = {key: payload[key] for key in
                      ("state", "text", "device", "devices") if key in payload}
            if values.get("state") == "disconnected":
                self._epoch += 1
                stopped, self._worker = self._worker, None
                values.update(devices=[], device=None, busy=False,
                              text="BrainBit connection lost; discover again")
            self._update(**values)
        if stopped is not None:
            stopped.close()

    def _exited(self, epoch):
        stopped = None
        with self._lock:
            if epoch != self._epoch or self._closed:
                return
            self._epoch += 1
            stopped, self._worker = self._worker, None
            self._update(state="error", device=None, devices=[], busy=False,
                         text="BrainBit worker stopped; discover again",
                         error="BrainBit worker stopped unexpectedly")
        if stopped is not None:
            stopped.close()

    def call(self, action: str, **kwargs):
        if action not in ("discover", "connect", "disconnect", "status"):
            return self._reject("Unsupported BrainBit action")
        allowed = {"connect": {"device_id"}, "status": {"refresh"}}.get(action, set())
        if set(kwargs) - allowed:
            return self._reject("Unsupported BrainBit arguments")
        if action == "status" and type(kwargs.get("refresh", False)) is not bool:
            return self._reject("refresh must be a boolean")
        if action == "status" and not kwargs.get("refresh", False):
            return self.status()
        if action == "connect" and (not isinstance(kwargs.get("device_id"), str)
                                    or len(kwargs["device_id"]) > 64):
            return self._reject("Choose a discovered BrainBit device")

        cancelled = None
        with self._lock:
            if self._closed or not self._snapshot["available"]:
                return self.status()
            if self._snapshot["busy"]:
                if action != "disconnect":
                    return self.status()
                self._epoch += 1
                cancelled, self._worker = self._worker, None
                self._update(state="disconnected", busy=False, devices=[], device=None,
                             text="BrainBit operation cancelled; discover again")
            elif action == "status" and self._worker is None:
                return self.status()
            elif action == "disconnect" and self._worker is None:
                self._update(state="disconnected", busy=False, devices=[], device=None,
                             text="BrainBit disconnected")
                return self.status()
            elif action == "discover" and self._snapshot["state"] == "connected":
                return self._reject("Disconnect before discovering devices")
            elif action == "connect" and self._snapshot["state"] == "connected":
                return self._reject("Already connected; disconnect before selecting a device")
            elif action == "connect" and kwargs["device_id"] not in {
                    item["id"] for item in self._snapshot["devices"]}:
                return self._reject("Device selection expired; discover again")
            else:
                state = {"discover": "scanning", "connect": "connecting",
                         "disconnect": "disconnecting"}.get(action, self._snapshot["state"])
                self._update(state=state, busy=True,
                             text={"discover": "Discovering BrainBit devices…",
                                   "connect": "Connecting to BrainBit…",
                                   "disconnect": "Disconnecting BrainBit…",
                                   "status": "Refreshing BrainBit status…"}[action])
                epoch = self._epoch
                worker = self._worker
                cancelled = False
        if cancelled is not False:
            if cancelled is not None:
                cancelled.close()
            return self.status()

        try:
            if worker is None:
                worker = _WorkerClient(lambda payload: self._event(epoch, payload),
                                       lambda: self._exited(epoch))
                with self._lock:
                    stale = epoch != self._epoch or self._closed
                    if not stale:
                        self._worker = worker
                if stale:
                    worker.close()
                    return self.status()
            timeout = self._connect_timeout if action == "connect" else self._request_timeout
            response = worker.request(action, kwargs if action == "connect" else {},
                                      timeout=timeout, scan_seconds=self._scan_seconds)
            if response.get("error"):
                raise _WorkerFailure(bool(response.get("unavailable")))
            payload = response.get("result")
            if not isinstance(payload, dict) or "state" not in payload:
                raise OSError("Invalid BrainBit worker response")
            stopped = None
            with self._lock:
                if epoch != self._epoch or self._closed:
                    return self.status()
                if action == "disconnect" or (action in ("status", "connect")
                                              and payload["state"] == "disconnected"):
                    self._epoch += 1
                    stopped, self._worker = self._worker, None
                    payload.update(devices=[], device=None)
                self._update(**{key: value for key, value in payload.items()
                                if key in ("state", "available", "text", "device", "devices")},
                             busy=False)
            if stopped is not None:
                stopped.close()
            return self.status()
        except (OSError, ValueError, TimeoutError, _WorkerFailure) as exc:
            with self._lock:
                if epoch != self._epoch or self._closed:
                    return self.status()
                unavailable = isinstance(exc, _WorkerFailure) and exc.unavailable
                message = ("BrainBit SDK could not load; check the optional pyneurosdk2 installation"
                           if unavailable else "BrainBit operation timed out; discover again"
                           if isinstance(exc, TimeoutError) else
                           "BrainBit operation failed; check the device, then discover again")
                self._update(state="unavailable" if unavailable else "error", busy=False,
                             available=not unavailable, device=None, devices=[],
                             text=message, error=message)
                self._epoch += 1
                stopped, self._worker = self._worker, None
            if stopped is not None:
                stopped.close()
            return self.status()

    def close(self):
        with self._lock:
            was_busy = self._snapshot["busy"]
            self._closed = True
            self._epoch += 1
            worker, self._worker = self._worker, None
            self._update(state="disabled", available=False, busy=False, device=None,
                         devices=[], text="BrainBit integration stopped")
        if worker is not None:
            try:
                # An idle worker gets a bounded graceful disconnect. Busy native
                # operations are cancelled immediately; do not queue behind them.
                if not was_busy:
                    worker.request("disconnect", {}, timeout=min(2, self._request_timeout),
                                   scan_seconds=self._scan_seconds)
            except (OSError, ValueError, TimeoutError):
                pass
            finally:
                worker.close()


class _WorkerFailure(Exception):
    def __init__(self, unavailable=False):
        self.unavailable = unavailable
