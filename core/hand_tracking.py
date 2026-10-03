"""Optional GPU trackers: local shared memory, bounded latency, no camera/input.

The separate interpreters keep GPU runtimes out of the voice/app
environment. Only one frame is in flight. The camera owner closes this object;
EOF also stops the worker if the owning backend exits unexpectedly.
"""
from __future__ import annotations

import json
import math
from multiprocessing import shared_memory
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "data" / "tracking-env" / "Scripts" / "python.exe"
MODEL_DIR = ROOT / "data" / "hand-models"
WILOR_PYTHON = ROOT / "data" / "wilor-env" / "Scripts" / "python.exe"
WILOR_MODEL_DIR = ROOT / "data" / "wilor-models"
TRACKER_NAMES = {"mediapipe": "MediaPipe Hands", "rtmpose": "RTMPose Hand5 · GPU",
                 "wilor": "WiLoR + AnyHand · GPU"}
MAX_FRAME_BYTES = 1920 * 1080 * 3
MAX_FRAME_AGE = 0.25
STARTUP_TIMEOUT = 90.0
FIRST_FRAME_TIMEOUT = 10.0
FRAME_TIMEOUT = 1.0
SETUP_HINT = r"Run .\.venv\Scripts\python.exe setup_hand_tracking.py"


def probe_rtmpose():
    """Cheap installation check; status/preview never start a GPU or a camera."""
    if sys.platform != "win32":
        return {"available": False, "reason": "RTMPose GPU currently requires Windows."}
    if not PYTHON.is_file() or any(not (MODEL_DIR / name).is_file()
                                   for name in ("detector.onnx", "pose.onnx")):
        return {"available": False, "reason": "RTMPose is not installed. " + SETUP_HINT}
    return {"available": True, "reason": "RTMPose is installed; GPU is checked when the camera starts."}


def create_rtmpose(stop_event):
    return RTMPoseWorker(stop_event)


def probe_wilor():
    """Probe files only; GPU loading happens after explicit camera activation."""
    if sys.platform != "win32":
        return {"available": False, "reason": "WiLoR GPU currently requires Windows and NVIDIA CUDA."}
    required = ("anyhand_wilor.ckpt", "detector.pt", "MANO_RIGHT.pkl", "mano_mean_params.npz")
    if not WILOR_PYTHON.is_file() or any(not (WILOR_MODEL_DIR / name).is_file() for name in required):
        return {"available": False,
                "reason": r"WiLoR is not installed. Run .\.venv\Scripts\python.exe setup_wilor.py"}
    return {"available": True, "reason": "WiLoR + AnyHand is installed; CUDA is checked when the camera starts."}


def create_wilor(stop_event):
    return RTMPoseWorker(stop_event, backend="wilor")


class TrackerStartupError(RuntimeError):
    """Startup failed with resources whose owner must retry cleanup."""

    def __init__(self, startup_error, cleanup_error, tracker):
        super().__init__(f"{startup_error}; {cleanup_error}")
        self.tracker = tracker


class RTMPoseWorker:
    """Shared GPU process transport (legacy class name retained for callers)."""

    def __init__(self, stop_event, backend="rtmpose"):
        if backend not in ("rtmpose", "wilor"):
            raise ValueError("Unknown GPU tracker backend")
        self.backend = backend
        self._stop = stop_event
        self._process = None
        self._memory = None
        self._reader = None
        self._log = None
        self._job = None
        self._closed = False
        self._messages = queue.Queue(maxsize=4)
        self._sequence = 0
        self.metadata = {}
        try:
            check = probe_wilor() if backend == "wilor" else probe_rtmpose()
            if not check["available"]:
                raise RuntimeError(check["reason"])
            if self._stop.is_set():
                raise RuntimeError("Hand tracking startup cancelled")
            self._memory = shared_memory.SharedMemory(create=True, size=MAX_FRAME_BYTES)
            self._log = (ROOT / "data" / "hand-tracker.log").open("w", encoding="utf-8")
            python = WILOR_PYTHON if backend == "wilor" else PYTHON
            self._process = subprocess.Popen(
                [str(python), "-u", "-m", "core.hand_tracking", "--worker", self._memory.name, backend],
                cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), bufsize=0,
            )
            self._job = _worker_job(self._process)
            self._reader = threading.Thread(target=self._read, name="hand-tracker-results", daemon=True)
            self._reader.start()
            ready = self._wait(STARTUP_TIMEOUT)
            if ready.get("type") != "ready":
                raise RuntimeError("Unexpected hand tracker startup response")
            self.metadata = ready.get("metadata", {})
        except BaseException as startup_error:
            try:
                self.close()
            except Exception as cleanup_error:
                # Construction did not return, so expose the partial object
                # explicitly. Its camera owner can retain it for stop/retry.
                raise TrackerStartupError(startup_error, cleanup_error, self) from startup_error
            raise

    def _read(self):
        try:
            while not self._closed:
                line = self._process.stdout.readline(16385)
                if not line:
                    raise RuntimeError("GPU hand tracker worker exited. See data/hand-tracker.log")
                if len(line) > 16384:
                    raise RuntimeError("Invalid GPU hand tracker response size")
                self._messages.put_nowait(json.loads(line))
        except Exception as exc:
            try:
                self._messages.put_nowait({"error": str(exc)})
            except queue.Full:
                pass

    def _wait(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            if self._closed or self._stop.is_set():
                raise RuntimeError("Hand tracking stopped")
            left = deadline - time.monotonic()
            if left <= 0:
                phase = "model startup" if self._sequence == 0 else f"frame {self._sequence}"
                raise RuntimeError(
                    f"GPU hand tracker timed out during {phase} after {timeout:g} seconds. "
                    "See data/hand-tracker.log. Turn the camera off, then retry."
                )
            try:
                message = self._messages.get(timeout=min(0.05, left))
            except queue.Empty:
                continue
            if not isinstance(message, dict):
                raise RuntimeError("Invalid GPU hand tracker response")
            if message.get("error"):
                raise RuntimeError(str(message["error"]))
            return message

    def process(self, rgb):
        import numpy as np
        if self._closed or self._stop.is_set():
            return []
        if (rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1
                or rgb.dtype != np.uint8 or rgb.nbytes > MAX_FRAME_BYTES):
            raise ValueError("GPU hand tracker requires a uint8 RGB frame up to 1920x1080")
        started = time.monotonic()
        height, width = rgb.shape[:2]
        frame = np.ndarray(rgb.shape, dtype=np.uint8, buffer=self._memory.buf)
        frame[:] = rgb
        del frame
        self._sequence += 1
        command = {"type": "frame", "id": self._sequence, "width": width, "height": height}
        self._process.stdin.write((json.dumps(command) + "\n").encode("utf-8"))
        # Real camera content can initialize detector/postprocessing paths that
        # synthetic warmup did not exercise. Allow that first response to finish
        # without accepting an old pose or submitting overlapping frames.
        result = self._wait(FIRST_FRAME_TIMEOUT if self._sequence == 1 else FRAME_TIMEOUT)
        if (result.get("type") != "frame" or type(result.get("id")) is not int
                or result["id"] != self._sequence):
            raise RuntimeError("Out-of-order GPU hand tracker frame")
        # Cold compilation, a busy GPU or a suspended process must not replay
        # an old hand pose into the live pointer. No frame queue accumulates.
        if self._stop.is_set() or time.monotonic() - started > MAX_FRAME_AGE:
            return []
        points = result.get("points")
        if points == []:
            return []
        if (not isinstance(points, list) or len(points) != 21
                or any(not isinstance(p, list) or len(p) != 3
                       or any(type(v) not in (int, float) or not math.isfinite(v) for v in p)
                       for p in points)):
            raise RuntimeError("Invalid GPU hand tracker landmarks")
        return points

    def close(self):
        self._closed = True
        process = self._process
        failures = []
        try:
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=0.5)
                except (OSError, subprocess.TimeoutExpired):
                    process.kill()
                    process.wait(timeout=0.5)
        except Exception as exc:
            failures.append(exc)
        # A driver/process cleanup failure must never skip releasing the
        # parent-owned handles. A subsequent close can retry a surviving child.
        resources = []
        if self._job is not None:
            # Also terminates the child if TerminateProcess could not complete.
            resources.append(self._job.close)
        if process is not None:
            resources.extend(pipe.close for pipe in (process.stdin, process.stdout) if pipe is not None)
        if self._reader is not None:
            resources.append(lambda: self._reader.join(timeout=0.2))
        if self._memory is not None:
            resources.extend((self._memory.close, self._memory.unlink))
        if self._log is not None:
            resources.append(self._log.close)
        for release in resources:
            try:
                release()
            except Exception as exc:
                failures.append(exc)
        if not failures:
            self._memory = None
        if failures:
            raise RuntimeError(f"Hand tracker cleanup failed: {failures[0]}")


def _worker_job(process):
    """Windows kills the GPU worker if its owning backend exits or crashes."""
    import ctypes as ct
    from ctypes import wintypes as wt

    class Basic(ct.Structure):
        _fields_ = [("per_process", ct.c_longlong), ("per_job", ct.c_longlong),
                    ("flags", wt.DWORD), ("min_working_set", ct.c_size_t),
                    ("max_working_set", ct.c_size_t), ("active_processes", wt.DWORD),
                    ("affinity", ct.c_size_t), ("priority", wt.DWORD), ("scheduling", wt.DWORD)]

    class Io(ct.Structure):
        _fields_ = [(name, ct.c_ulonglong) for name in
                    ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

    class Extended(ct.Structure):
        _fields_ = [("basic", Basic), ("io", Io), ("process_memory", ct.c_size_t),
                    ("job_memory", ct.c_size_t), ("peak_process", ct.c_size_t), ("peak_job", ct.c_size_t)]

    kernel = ct.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ct.c_void_p, wt.LPCWSTR]
    kernel.CreateJobObjectW.restype = wt.HANDLE
    kernel.SetInformationJobObject.argtypes = [wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD]
    kernel.SetInformationJobObject.restype = wt.BOOL
    kernel.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
    kernel.AssignProcessToJobObject.restype = wt.BOOL
    kernel.CloseHandle.argtypes = [wt.HANDLE]
    kernel.CloseHandle.restype = wt.BOOL
    handle = kernel.CreateJobObjectW(None, None)
    if not handle:
        raise ct.WinError(ct.get_last_error())
    limits = Extended()
    limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if (not kernel.SetInformationJobObject(handle, 9, ct.byref(limits), ct.sizeof(limits))
            or not kernel.AssignProcessToJobObject(handle, wt.HANDLE(int(process._handle)))):
        error = ct.get_last_error()
        kernel.CloseHandle(handle)
        raise ct.WinError(error)

    class Job:
        def close(self):
            nonlocal handle
            if handle:
                if not kernel.CloseHandle(handle):
                    raise ct.WinError(ct.get_last_error())
                handle = None

    return Job()


def _select_windows_gpu():
    """Get the actual DXGI ordinal DirectML uses (not nvidia-smi's ordinal)."""
    import ctypes as ct
    from ctypes import wintypes as wt
    import uuid

    class Luid(ct.Structure):
        _fields_ = [("low", wt.DWORD), ("high", wt.LONG)]

    class Description(ct.Structure):
        _fields_ = [("name", wt.WCHAR * 128), ("vendor", wt.UINT),
                    ("device", wt.UINT), ("subsystem", wt.UINT), ("revision", wt.UINT),
                    ("video_memory", ct.c_size_t), ("system_memory", ct.c_size_t),
                    ("shared_memory", ct.c_size_t), ("luid", Luid), ("flags", wt.UINT)]

    def method(obj, index, result, *args):
        vtable = ct.cast(obj, ct.POINTER(ct.POINTER(ct.c_void_p))).contents
        return ct.WINFUNCTYPE(result, ct.c_void_p, *args)(vtable[index])

    iid = (ct.c_ubyte * 16).from_buffer_copy(uuid.UUID("770aae78-f26f-4dba-a829-253c83d1b387").bytes_le)
    factory = ct.c_void_p()
    create = ct.WinDLL("dxgi").CreateDXGIFactory1
    create.argtypes = [ct.c_void_p, ct.POINTER(ct.c_void_p)]
    create.restype = ct.c_long
    if create(ct.byref(iid), ct.byref(factory)) < 0:
        raise RuntimeError("Windows could not enumerate graphics adapters")
    adapters = []
    try:
        for index in range(32):
            adapter = ct.c_void_p()
            hr = method(factory, 12, ct.c_long, wt.UINT, ct.POINTER(ct.c_void_p))(
                factory, index, ct.byref(adapter))
            if hr < 0:
                break
            try:
                description = Description()
                hr = method(adapter, 10, ct.c_long, ct.POINTER(Description))(adapter, ct.byref(description))
                if hr >= 0 and not (description.flags & 2):  # exclude software rasterizers
                    adapters.append({"device_id": index, "gpu_name": description.name,
                                     "vram_bytes": description.video_memory,
                                     "vendor": description.vendor})
            finally:
                method(adapter, 2, wt.ULONG)(adapter)
    finally:
        method(factory, 2, wt.ULONG)(factory)
    if not adapters:
        raise RuntimeError("No hardware graphics adapter is available for RTMPose")
    return max(adapters, key=lambda item: (item["vendor"] == 0x10DE, item["vram_bytes"]))


def _worker(memory_name, backend="rtmpose"):
    import contextlib
    import socket
    import traceback

    started = time.monotonic()
    phase = "dependency imports"

    def report(stage):
        nonlocal phase
        phase = stage
        print(f"[hand-tracker +{time.monotonic() - started:.2f}s] {backend}: {stage}",
              file=sys.stderr, flush=True)

    # Models and dependencies are installed explicitly. Runtime inference has
    # no need for outbound connections, telemetry, or automatic downloads.
    def offline(*args, **kwargs):
        raise OSError("Hand tracking runs offline. Use the setup script for missing assets.")
    socket.socket.connect = offline
    socket.socket.connect_ex = lambda *args, **kwargs: 10051  # WSAENETUNREACH

    # stdout is reserved for the protocol, including dependency diagnostics.
    output = sys.stdout
    def send(message):
        output.write(json.dumps(message, allow_nan=False) + "\n")
        output.flush()

    memory = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            report("dependency imports")
            import numpy as np
            import cv2

            report("model loading")
            if backend == "wilor":
                from core.wilor_model import WiLoRModel
                model = WiLoRModel(WILOR_MODEL_DIR, device_id=0)
                gpu = {}  # WiLoR metadata reports CUDA's own adapter identity.
            elif backend == "rtmpose":
                from core.rtmpose_model import RTMPoseModel
                gpu = _select_windows_gpu()
                model = RTMPoseModel(MODEL_DIR, device_id=gpu["device_id"])
            else:
                raise ValueError("Unknown hand tracker backend")
            report("GPU warmup")
            model.warmup()
            metadata = {**model.metadata, **gpu}
        report("shared memory attachment")
        memory = shared_memory.SharedMemory(name=memory_name)
        send({"type": "ready", "metadata": metadata})
        report("ready")
        for line in sys.stdin:
            command = json.loads(line)
            phase = f"frame {command.get('id')}"
            width, height = command["width"], command["height"]
            if (type(width) is not int or type(height) is not int or min(width, height) < 1
                    or width * height * 3 > MAX_FRAME_BYTES or command.get("type") != "frame"):
                raise ValueError("Invalid hand tracker frame dimensions")
            frame = np.ndarray((height, width, 3), np.uint8, buffer=memory.buf)
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            del frame
            with contextlib.redirect_stdout(sys.stderr):
                if command["id"] == 1:
                    report("first frame inference")
                points = model.predict(bgr)
            send({"type": "frame", "id": command["id"], "points": points})
            if command["id"] == 1:
                report("first frame complete")
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        send({"error": f"GPU hand tracker failed during {phase}: {exc}. See data/hand-tracker.log"})
    finally:
        if memory is not None:
            memory.close()


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4) or sys.argv[1] != "--worker":
        raise SystemExit("This is an internal tracker worker. Use a setup script for installation.")
    _worker(sys.argv[2], sys.argv[3] if len(sys.argv) == 4 else "rtmpose")
