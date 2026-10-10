"""Optional BrainBit connection adapter with isolated, bounded SDK operations."""

import copy
from collections import deque
import importlib.util
import json
import math
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
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
                if "acquisition" in message:
                    self._on_event({"_acquisition": message["acquisition"]})
                elif "event" in message:
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
    Closing permanently disables this instance. Acquisition is a separate,
    explicit local-preview API and is never advertised to hardware actions.
    """

    name = "brainbit"

    def __init__(self, enabled=True, scan_seconds=5, connect_timeout=15,
                 request_timeout=20):
        self._lock = threading.RLock()
        self._worker = None
        self._closed = not bool(enabled)
        self._epoch = 0
        self._revision = 0
        self._samples = deque(maxlen=1250)
        self._contact_precheck = None
        self._acq = self._empty_acquisition()
        self._last_counter = None
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

    @staticmethod
    def _empty_acquisition():
        return {"state": "stopped", "mode": None, "session_id": None,
                "channels": [], "nominal_hz": None, "units": None,
                "timing": "Host monotonic callback receipt; packet times are estimates",
                "stats": {"nonfinite": 0, "gaps": 0, "duplicates": 0,
                          "channel_mismatches": 0, "queue_drops": 0}}

    def _clear_acquisition(self, *, error=None, keep_contact=False):
        self._samples.clear()
        self._acq = self._empty_acquisition()
        self._last_counter = None
        if error:
            self._acq.update(state="error", error=error)
        if not keep_contact:
            self._contact_precheck = None

    def acquisition_snapshot(self):
        """Return a finite, bounded local-preview view; never journal this data."""
        return self._acquisition_snapshot(sample_limit=250)

    def acquisition_signal_window(self):
        """Copy up to five seconds/1250 cached packets for local EEG features.

        This internal service API preserves full resolution; it is not a
        hardware action or a public status payload and never reads the SDK.
        """
        return self._acquisition_snapshot(sample_limit=None)

    def _acquisition_snapshot(self, sample_limit):
        with self._lock:
            now = time.monotonic()
            result = copy.deepcopy(self._acq)
            packets = copy.deepcopy([packet for packet in self._samples
                                     if now - packet["host_received_monotonic"] <= 5])
            contact = copy.deepcopy(self._contact_precheck)
            last_receipt = self._samples[-1]["host_received_monotonic"] if self._samples else None
        if sample_limit is not None and len(packets) > sample_limit:
            stride = math.ceil(len(packets) / sample_limit)
            display = packets[::stride]
            # The display must include the newest packet, not end a stride early.
            if display[-1] is not packets[-1]:
                if len(display) == sample_limit:
                    display[-1] = packets[-1]
                else:
                    display.append(packets[-1])
            result["samples"] = display
        else:
            result["samples"] = packets
        result["stats"]["age_seconds"] = (max(0, now - last_receipt)
                                           if last_receipt is not None else None)
        duration = (packets[-1]["host_received_monotonic"] - packets[0]["host_received_monotonic"]
                    if len(packets) > 1 else 0)
        result["stats"]["received_rate_hz"] = ((len(packets) - 1) / duration
                                                if duration > 0 else None)
        channel_stats = []
        if result["mode"] == "signal":
            for index, channel in enumerate(result["channels"]):
                values = [packet["samples"][index] for packet in packets
                          if len(packet["samples"]) > index and packet["samples"][index] is not None]
                mean = sum(values) / len(values) if values else 0
                channel_stats.append({"name": channel["name"],
                                      "rms_v": math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
                                      if values else None,
                                      "peak_to_peak_v": max(values) - min(values) if values else None})
        result["stats"]["channels"] = channel_stats
        if contact is not None:
            contact["age_seconds"] = max(0, now - contact["host_received_monotonic"])
        result["contact_precheck"] = contact
        return result

    def _acquisition_event(self, epoch, payload):
        if not isinstance(payload, dict):
            return
        with self._lock:
            if (epoch != self._epoch or self._closed or
                    self._acq["state"] not in ("starting", "running") or
                    payload.get("session_id") != self._acq["session_id"]):
                return
            channels = payload.get("channels", [])[:16]
            if channels:
                self._acq["channels"] = copy.deepcopy(channels)
            for key in ("nominal_hz", "units", "timing"):
                if key in payload:
                    self._acq[key] = payload[key]
            for key in ("nonfinite", "channel_mismatches", "queue_drops"):
                if type(payload.get(key)) is int:
                    self._acq["stats"][key] = max(self._acq["stats"][key], payload[key])
            for packet in payload.get("packets", [])[:100]:
                if not isinstance(packet, dict):
                    continue
                counter = packet.get("counter")
                receipt = packet.get("host_received_monotonic")
                if type(counter) is not int or not isinstance(receipt, (float, int)) or not math.isfinite(receipt):
                    continue
                counter &= 0xffffffff
                values = [float(value) if isinstance(value, (int, float)) and math.isfinite(value)
                          else None for value in packet.get("samples", [])[:16]]
                if self._last_counter is not None:
                    delta = (counter - self._last_counter) & 0xffffffff
                    if delta == 0:
                        self._acq["stats"]["duplicates"] += 1
                    elif delta != 1:
                        # A reset is not evidence of billions of lost samples.
                        self._acq["stats"]["gaps"] += 1
                self._last_counter = counter
                estimated = packet.get("estimated_monotonic")
                if not isinstance(estimated, (float, int)) or not math.isfinite(estimated):
                    estimated = None
                safe = {"counter": counter, "marker": packet.get("marker"), "samples": values,
                        "host_received_monotonic": float(receipt), "estimated_monotonic": estimated}
                self._samples.append(safe)
                if self._acq["mode"] == "contact":
                    self._contact_precheck = {
                        "values": list(values), "channels": copy.deepcopy(self._acq["channels"]),
                        "host_received_monotonic": float(receipt), "units": "ohm",
                        "unit_note": "Python SDK 1.0.15 specifies ohms; the SDK website conflicts. No contact threshold is assumed.",
                    }

    def start_acquisition(self, mode="signal"):
        """Start an explicitly requested local preview; not a HardwareDriver action."""
        if mode not in ("signal", "contact"):
            result = self.acquisition_snapshot()
            result["error"] = "Acquisition mode must be signal or contact"
            return result
        return self._acquisition_call("start_acquisition", mode)

    def stop_acquisition(self):
        """Stop acquisition with a bounded SDK request and release on failure."""
        return self._acquisition_call("stop_acquisition")

    def _acquisition_call(self, action, mode=None):
        cancelled = False
        with self._lock:
            if action == "stop_acquisition" and self._acq["state"] == "stopped":
                return self.acquisition_snapshot()
            if self._snapshot["busy"]:
                if action == "stop_acquisition":
                    message = "Pending SDK operation cancelled; device stop is unconfirmed. Reconnect before retrying."
                    self._clear_acquisition(error=message)
                    self._epoch += 1
                    cancelled, self._worker = self._worker, None
                    self._update(state="error", busy=False, devices=[], device=None,
                                 text=message, error=message)
                else:
                    result = self.acquisition_snapshot()
                    result["error"] = "BrainBit is busy; retry after the current operation"
                    return result
            elif self._closed or self._worker is None or self._snapshot["state"] != "connected":
                result = self.acquisition_snapshot()
                result.setdefault("error", "Connect a BrainBit before starting acquisition")
                return result
            elif action == "start_acquisition" and self._acq["state"] == "running" and self._acq["mode"] == mode:
                return self.acquisition_snapshot()
            else:
                epoch, worker = self._epoch, self._worker
                self._clear_acquisition(keep_contact=True)
                session_id = str(uuid.uuid4()) if action == "start_acquisition" else None
                self._acq.update(state="starting" if session_id else "stopping", mode=mode,
                                 session_id=session_id,
                                 units="V" if mode == "signal" else "ohm" if mode == "contact" else None)
                self._update(busy=True)
        if cancelled is not False:
            if cancelled is not None:
                cancelled.close()
            return self.acquisition_snapshot()
        try:
            args = {"mode": mode, "session_id": session_id} if session_id else {}
            response = worker.request(action, args, timeout=min(5, self._request_timeout),
                                      scan_seconds=self._scan_seconds)
            if response.get("error"):
                raise OSError("Acquisition command failed")
            payload = response.get("result", {})
            if payload.get("state") not in ("running", "stopped"):
                raise OSError("Invalid acquisition response")
            with self._lock:
                if epoch != self._epoch or self._closed:
                    return self.acquisition_snapshot()
                for key in ("state", "mode", "session_id", "channels", "nominal_hz", "units", "timing"):
                    if key in payload:
                        self._acq[key] = copy.deepcopy(payload[key])
                self._update(busy=False)
            return self.acquisition_snapshot()
        except (OSError, ValueError, TimeoutError):
            with self._lock:
                if epoch != self._epoch or self._closed:
                    return self.acquisition_snapshot()
                message = "Acquisition command failed; device stop is unconfirmed. Worker released; reconnect before retrying."
                self._clear_acquisition(error=message)
                self._epoch += 1
                self._worker = None
                self._update(state="error", busy=False, devices=[], device=None,
                             text=message, error=message)
            worker.close()
            return self.acquisition_snapshot()

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
        if "_acquisition" in payload:
            self._acquisition_event(epoch, payload["_acquisition"])
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
                active = self._acq["state"] in ("starting", "running", "stopping")
                self._clear_acquisition(error="Connection lost; device stop is unconfirmed" if active else None)
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
            active = self._acq["state"] in ("starting", "running", "stopping")
            self._clear_acquisition(error="Worker stopped; device stop is unconfirmed" if active else None)
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
                active = self._acq["state"] in ("starting", "running", "stopping")
                self._clear_acquisition(error="Operation cancelled; device stop is unconfirmed" if active else None)
                self._update(state="disconnected", busy=False, devices=[], device=None,
                             text="BrainBit operation cancelled; discover again")
            elif action == "status" and self._worker is None:
                return self.status()
            elif action == "disconnect" and self._worker is None:
                self._clear_acquisition()
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
                if action == "disconnect":
                    active = self._acq["state"] in ("starting", "running", "stopping")
                    self._clear_acquisition()
                    if active:
                        self._acq["state"] = "stopping"
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
                    lost_active = action != "disconnect" and self._acq["state"] in ("starting", "running", "stopping")
                    self._clear_acquisition(error="Connection lost; device stop is unconfirmed"
                                            if lost_active else None)
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
                active = self._acq["state"] in ("starting", "running", "stopping")
                if active:
                    message = "BrainBit operation failed; device stop is unconfirmed. Reconnect before retrying."
                self._clear_acquisition(error="SDK operation failed; device stop is unconfirmed" if active else None)
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
            if self._closed:
                return
            was_busy = self._snapshot["busy"]
            was_acquiring = self._acq["state"] in ("starting", "running", "stopping")
            self._clear_acquisition(error=self._acq.get("error"))
            if was_acquiring:
                self._acq["state"] = "stopping"
            self._closed = True
            self._epoch += 1
            worker, self._worker = self._worker, None
            self._update(state="disabled", available=False, busy=False, device=None,
                         devices=[], text="BrainBit integration stopped")
        if worker is not None:
            confirmed = not was_acquiring
            try:
                # An idle worker gets a bounded graceful disconnect. Busy native
                # operations are cancelled immediately; do not queue behind them.
                if not was_busy:
                    response = worker.request("disconnect", {}, timeout=min(2, self._request_timeout),
                                              scan_seconds=self._scan_seconds)
                    confirmed = not bool(response.get("error"))
            except (OSError, ValueError, TimeoutError):
                pass
            finally:
                worker.close()
                with self._lock:
                    self._clear_acquisition(error=None if confirmed else
                                            "Shutdown released worker; device stop is unconfirmed")


class _WorkerFailure(Exception):
    def __init__(self, unavailable=False):
        self.unavailable = unavailable
