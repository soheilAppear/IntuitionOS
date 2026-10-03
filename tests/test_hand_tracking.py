"""Parent GPU tracker protocol tests with no GPU, camera, or child processes."""

import io
import json
import queue
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from core import hand_tracking as tracking


class _Memory:
    def __init__(self, size=48):
        self.buf = bytearray(size)
        self.name = "fake-tracker-memory"
        self.closed = False
        self.unlinked = False

    def close(self):
        self.closed = True

    def unlink(self):
        self.unlinked = True


class _Pipe(io.BytesIO):
    def __init__(self, contents=b"", failure=None):
        super().__init__(contents)
        self.failure = failure

    def close(self):
        if self.failure:
            raise self.failure
        super().close()


class _Process:
    def __init__(self, *, terminate_error=None, waits=None, stdout=None):
        self.stdin = _Pipe()
        self.stdout = _Pipe(stdout or b"")
        self.actions = []
        self.returncode = None
        self.terminate_error = terminate_error
        self.waits = list(waits or [])

    def poll(self):
        return self.returncode

    def terminate(self):
        self.actions.append("terminate")
        if self.terminate_error:
            raise self.terminate_error

    def kill(self):
        self.actions.append("kill")

    def wait(self, timeout):
        self.actions.append(("wait", timeout))
        response = self.waits.pop(0) if self.waits else 0
        if isinstance(response, Exception):
            raise response
        self.returncode = response
        return response


def _worker(process=None):
    worker = tracking.RTMPoseWorker.__new__(tracking.RTMPoseWorker)
    worker._stop = threading.Event()
    worker._process = process or _Process()
    worker._memory = _Memory()
    worker._reader = SimpleNamespace(join=lambda **_: None)
    worker._log = _Pipe()
    worker._job = None
    worker._closed = False
    worker._messages = queue.Queue(maxsize=4)
    worker._sequence = 0
    worker.metadata = {}
    return worker


def _response(points=None, **extra):
    return {"type": "frame", "id": 1,
            "points": [[0.4, 0.6, 0.0] for _ in range(21)] if points is None else points,
            **extra}


def test_wait_returns_only_a_valid_object_and_reports_worker_errors():
    worker = _worker()
    message = {"type": "ready", "metadata": {"gpu_name": "test GPU"}}
    worker._messages.put(message)
    assert worker._wait(1) is message
    worker._messages.put({"error": "DirectML initialization failed"})
    with pytest.raises(RuntimeError, match="DirectML initialization failed"):
        worker._wait(1)
    worker._messages.put(["not an object"])
    with pytest.raises(RuntimeError, match="Invalid GPU hand tracker response"):
        worker._wait(1)


@pytest.mark.parametrize("closed", [False, True])
def test_wait_cancels_before_consuming_an_already_queued_frame(closed):
    worker = _worker()
    if closed:
        worker._closed = True
    else:
        worker._stop.set()
    worker._messages.put(_response())
    with pytest.raises(RuntimeError, match="stopped"):
        worker._wait(90)
    assert worker._messages.qsize() == 1


def test_wait_checks_stop_again_after_a_short_empty_poll():
    worker = _worker()
    waits = []

    def empty(timeout):
        waits.append(timeout)
        worker._stop.set()
        raise queue.Empty

    worker._messages = SimpleNamespace(get=empty)
    with pytest.raises(RuntimeError, match="stopped"):
        worker._wait(90)
    assert len(waits) == 1 and 0 < waits[0] <= 0.05


@pytest.mark.parametrize("sequence, phase", [(0, "model startup"), (1, "frame 1"), (8, "frame 8")])
def test_wait_timeout_is_bounded_without_real_sleep(monkeypatch, sequence, phase):
    worker = _worker()
    worker._sequence = sequence
    clock = iter((10.0, 10.01, 11.01))
    monkeypatch.setattr(tracking.time, "monotonic", lambda: next(clock))

    def empty(timeout):
        assert timeout <= 0.05
        raise queue.Empty

    worker._messages = SimpleNamespace(get=empty)
    with pytest.raises(RuntimeError, match=f"timed out during {phase} after 1 seconds") as error:
        worker._wait(1)
    assert "data/hand-tracker.log" in str(error.value)


@pytest.mark.parametrize("contents, message", [
    (b"", "worker exited"),
    (b"not-json\n", "Expecting value"),
    (b"x" * 16385, "response size"),
])
def test_reader_turns_eof_or_invalid_json_into_a_reported_error(contents, message):
    worker = _worker(_Process(stdout=contents))
    worker._read()
    with pytest.raises(RuntimeError, match=message):
        worker._wait(1)


def test_process_copies_mirrored_rgb_without_channel_or_landmark_changes():
    worker = _worker()
    source = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)[:, ::-1]
    response = _response()
    worker._messages.put(response)
    assert worker.process(source) == response["points"]
    assert worker._memory.buf[:source.nbytes] == source.tobytes()
    command = json.loads(worker._process.stdin.getvalue())
    assert command == {"type": "frame", "id": 1, "width": 4, "height": 2}


@pytest.mark.parametrize("stop", [True, False])
def test_process_does_not_send_a_frame_after_stop_or_close(stop):
    worker = _worker()
    if stop:
        worker._stop.set()
    else:
        worker._closed = True
    assert worker.process(np.zeros((2, 2, 3), dtype=np.uint8)) == []
    assert worker._process.stdin.getvalue() == b""


@pytest.mark.parametrize("elapsed, stop", [(0.251, False), (0.01, True)])
def test_process_discards_stale_or_cancelled_results(monkeypatch, elapsed, stop):
    worker = _worker()
    now = [10.0]
    monkeypatch.setattr(tracking.time, "monotonic", lambda: now[0])

    def result(_):
        now[0] += elapsed
        if stop:
            worker._stop.set()
        return _response()

    monkeypatch.setattr(worker, "_wait", result)
    assert worker.process(np.zeros((2, 2, 3), dtype=np.uint8)) == []


def test_slow_first_frame_is_drained_and_discarded_then_tracking_recovers(monkeypatch):
    worker = _worker()
    now = [10.0]
    monkeypatch.setattr(tracking.time, "monotonic", lambda: now[0])
    responses = [(11.2, _response()), (11.21, _response(id=2))]

    def receive(timeout):
        ready_at, response = responses[0]
        now[0] = min(now[0] + timeout, ready_at)
        if now[0] < ready_at:
            raise queue.Empty
        responses.pop(0)
        return response

    worker._messages = SimpleNamespace(get=receive)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    # A first prediction over one second must not shut down capture, but its
    # stale landmarks must not be replayed into the live pointer either.
    assert worker.process(frame) == []
    assert worker.process(frame) == _response()["points"]
    sent = [json.loads(line) for line in worker._process.stdin.getvalue().splitlines()]
    assert [command["id"] for command in sent] == [1, 2]
    assert responses == []


@pytest.mark.parametrize("sequence, timeout", [(0, tracking.FIRST_FRAME_TIMEOUT), (1, tracking.FRAME_TIMEOUT)])
def test_stalled_frame_still_times_out_with_a_bounded_wait(monkeypatch, sequence, timeout):
    worker = _worker()
    worker._sequence = sequence
    now = [10.0]
    monkeypatch.setattr(tracking.time, "monotonic", lambda: now[0])

    def empty(timeout):
        now[0] += timeout
        raise queue.Empty

    worker._messages = SimpleNamespace(get=empty)
    with pytest.raises(RuntimeError, match=f"during frame {sequence + 1}"):
        worker.process(np.zeros((2, 2, 3), dtype=np.uint8))
    assert now[0] == pytest.approx(10.0 + timeout)


@pytest.mark.parametrize("points", [
    "bad", {}, [[0.5, 0.5, 0.0]] * 20, [[0.5, 0.5]] * 21,
    [[True, 0.5, 0]] * 21, [[float("nan"), 0.5, 0]] * 21,
    [[0.5, float("inf"), 0]] * 21, [[0.5, "0.5", 0]] * 21,
])
def test_process_rejects_malformed_landmarks(points):
    worker = _worker()
    worker._messages.put(_response(points))
    with pytest.raises(RuntimeError, match="Invalid GPU hand tracker landmarks"):
        worker.process(np.zeros((2, 2, 3), dtype=np.uint8))


def test_process_returns_hand_loss_for_an_empty_detection():
    worker = _worker()
    worker._messages.put(_response([]))
    assert worker.process(np.zeros((2, 2, 3), dtype=np.uint8)) == []


@pytest.mark.parametrize("identity", [0, 2, "1", None])
def test_process_rejects_responses_from_another_frame(identity):
    worker = _worker()
    worker._messages.put(_response(id=identity))
    with pytest.raises(RuntimeError, match="Out-of-order"):
        worker.process(np.zeros((2, 2, 3), dtype=np.uint8))


@pytest.mark.parametrize("extra", [{"id": True}, {"id": 1.0}, {"type": "ready"}, {"type": None}])
def test_process_requires_a_frame_message_with_an_integer_identity(extra):
    worker = _worker()
    worker._messages.put(_response(**extra))
    with pytest.raises(RuntimeError):
        worker.process(np.zeros((2, 2, 3), dtype=np.uint8))


@pytest.mark.parametrize("source", [
    np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2, 4), dtype=np.uint8),
    np.zeros((2, 2, 3), dtype=np.float32), np.zeros((0, 2, 3), dtype=np.uint8),
    np.zeros((2, 0, 3), dtype=np.uint8),
])
def test_invalid_frames_are_rejected_before_shared_memory_or_protocol_write(source):
    worker = _worker()
    with pytest.raises(ValueError, match="RGB frame"):
        worker.process(source)
    assert worker._process.stdin.getvalue() == b""


def test_close_terminates_then_kills_a_worker_that_does_not_exit():
    process = _Process(waits=[subprocess.TimeoutExpired("worker", 0.5), 0])
    worker = _worker(process)
    memory, log = worker._memory, worker._log
    worker.close()
    assert process.actions == ["terminate", ("wait", 0.5), "kill", ("wait", 0.5)]
    assert process.stdin.closed and process.stdout.closed
    assert memory.closed and memory.unlinked and log.closed
    worker.close()
    assert len(process.actions) == 4


@pytest.mark.parametrize("failure", ["terminate", "second_wait", "pipe_close"])
def test_close_releases_all_local_resources_even_when_worker_cleanup_fails(failure):
    process = _Process(
        terminate_error=OSError("process termination raced") if failure == "terminate" else None,
        waits=[subprocess.TimeoutExpired("worker", 0.5)] * 2 if failure == "second_wait" else None,
    )
    if failure == "pipe_close":
        process.stdin.failure = OSError("pipe close failed")
    worker = _worker(process)
    memory, log = worker._memory, worker._log
    try:
        worker.close()
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        pass  # Reporting a failure is fine; skipping the remaining cleanup is not.
    assert memory.closed and memory.unlinked
    assert log.closed
    assert process.stdout.closed


@pytest.mark.parametrize("backend", ["rtmpose", "wilor"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_startup_failure_cleans_up_the_partially_created_worker(monkeypatch, tmp_path, backend, cleanup_fails):
    (tmp_path / "data").mkdir()
    process, memory = _Process(), _Memory()
    if cleanup_fails:
        process.stdin.failure = OSError("Cannot release worker pipe")
    monkeypatch.setattr(tracking, "ROOT", tmp_path)
    monkeypatch.setattr(tracking, f"probe_{backend}", lambda: {"available": True})
    monkeypatch.setattr(tracking.shared_memory, "SharedMemory", lambda **_: memory)
    monkeypatch.setattr(tracking.subprocess, "Popen", lambda *_, **__: process)
    monkeypatch.setattr(tracking, "_worker_job", lambda _: None)
    monkeypatch.setattr(tracking.threading, "Thread", lambda **_: SimpleNamespace(
        start=lambda: None, join=lambda **_: None))

    def failed_wait(self, timeout):
        raise RuntimeError("GPU warmup failed")

    monkeypatch.setattr(tracking.RTMPoseWorker, "_wait", failed_wait)
    with pytest.raises(RuntimeError, match="GPU warmup failed") as failed:
        tracking.RTMPoseWorker(threading.Event(), backend=backend)
    assert process.actions == ["terminate", ("wait", 0.5)]
    assert memory.closed and memory.unlinked
    if cleanup_fails:
        assert isinstance(failed.value, tracking.TrackerStartupError)
        worker = failed.value.tracker
        assert worker.backend == backend
        assert worker._process is process and worker._memory is memory
        assert "Cannot release worker pipe" in str(failed.value)
        assert str(failed.value.__cause__) == "GPU warmup failed"
        process.stdin.failure = None
        worker.close()
        assert worker._memory is None
    else:
        assert type(failed.value) is RuntimeError
        assert str(failed.value) == "GPU warmup failed"
    assert process.stdin.closed and process.stdout.closed


@pytest.mark.parametrize("backend", ["rtmpose", "wilor"])
def test_cancelled_startup_does_not_allocate_or_launch(monkeypatch, backend):
    monkeypatch.setattr(tracking, f"probe_{backend}", lambda: {"available": True})
    calls = []
    monkeypatch.setattr(tracking.shared_memory, "SharedMemory", lambda **_: calls.append("memory"))
    monkeypatch.setattr(tracking.subprocess, "Popen", lambda *_, **__: calls.append("process"))
    stop = threading.Event()
    stop.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        tracking.RTMPoseWorker(stop, backend=backend)
    assert calls == []


@pytest.fixture
def wilor_assets(monkeypatch, tmp_path):
    python = tmp_path / "wilor-env" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.touch()
    model_dir = tmp_path / "wilor-models"
    model_dir.mkdir()
    assets = {name: model_dir / name for name in (
        "anyhand_wilor.ckpt", "detector.pt", "MANO_RIGHT.pkl", "mano_mean_params.npz")}
    for path in assets.values():
        path.touch()
    monkeypatch.setattr(tracking, "WILOR_PYTHON", python)
    monkeypatch.setattr(tracking, "WILOR_MODEL_DIR", model_dir)
    monkeypatch.setattr(tracking.sys, "platform", "win32")
    return {"python.exe": python, **assets}


@pytest.mark.parametrize("missing", ["python.exe", "anyhand_wilor.ckpt", "detector.pt",
                                     "MANO_RIGHT.pkl", "mano_mean_params.npz"])
def test_wilor_probe_requires_its_own_runtime_and_every_model_asset(wilor_assets, missing):
    assert tracking.probe_wilor()["available"] is True
    wilor_assets[missing].unlink()
    result = tracking.probe_wilor()
    assert result["available"] is False
    assert "setup_wilor.py" in result["reason"]


def test_wilor_probe_rejects_unsupported_platform_without_starting_worker(monkeypatch, wilor_assets):
    monkeypatch.setattr(tracking.sys, "platform", "linux")
    result = tracking.probe_wilor()
    assert result["available"] is False
    assert "Windows" in result["reason"] and "CUDA" in result["reason"]


def test_wilor_status_and_preview_only_probe_files_without_gpu_or_camera(monkeypatch, wilor_assets):
    import builtins
    from core.gestures import GestureRecognizer

    def forbidden(*args, **kwargs):
        raise AssertionError("Camera-off status cannot start a GPU, camera, or child process")

    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in ("torch", "ultralytics", "onnxruntime", "wilor_mini"):
            forbidden()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(tracking.subprocess, "Popen", forbidden)
    monkeypatch.setattr(tracking.shared_memory, "SharedMemory", forbidden)
    monkeypatch.setattr(tracking, "_select_windows_gpu", forbidden)
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(VideoCapture=forbidden))
    recognizer = GestureRecognizer(tracker_backend="wilor")
    assert tracking.probe_wilor()["available"] is True
    status = recognizer.status()
    assert status["available"] is True and status["running"] is False
    assert status["tracker_backend"] == "wilor"
    preview = recognizer.preview()
    assert preview["running"] is False and preview["image"] is None
    assert preview["model_name"] == "WiLoR + AnyHand · GPU"


@pytest.mark.parametrize("backend", ["rtmpose", "wilor"])
def test_factory_launches_correct_isolated_environment_and_shared_transport(monkeypatch, tmp_path, backend):
    (tmp_path / "data").mkdir()
    process, memory = _Process(), _Memory()
    launches = []
    monkeypatch.setattr(tracking, "ROOT", tmp_path)
    monkeypatch.setattr(tracking, "PYTHON", tmp_path / "rtmpose-python.exe")
    monkeypatch.setattr(tracking, "WILOR_PYTHON", tmp_path / "wilor-python.exe")
    monkeypatch.setattr(tracking, f"probe_{backend}", lambda: {"available": True})
    other = "wilor" if backend == "rtmpose" else "rtmpose"
    def forbidden():
        raise AssertionError("A selected tracker must never silently load the other backend")
    monkeypatch.setattr(tracking, f"probe_{other}", forbidden)
    monkeypatch.setattr(tracking.shared_memory, "SharedMemory", lambda **_: memory)
    def launch(command, **kwargs):
        launches.append((command, kwargs))
        return process
    monkeypatch.setattr(tracking.subprocess, "Popen", launch)
    monkeypatch.setattr(tracking, "_worker_job", lambda _: None)
    monkeypatch.setattr(tracking.threading, "Thread", lambda **_: SimpleNamespace(
        start=lambda: None, join=lambda **_: None))
    metadata = {"model": backend, "gpu_name": "test GPU"}
    monkeypatch.setattr(tracking.RTMPoseWorker, "_wait",
                        lambda self, timeout: {"type": "ready", "metadata": metadata})
    stop = threading.Event()
    worker = getattr(tracking, f"create_{backend}")(stop)
    try:
        assert worker.backend == backend and worker._stop is stop
        assert worker.metadata == metadata
        command, kwargs = launches[0]
        python = tracking.WILOR_PYTHON if backend == "wilor" else tracking.PYTHON
        assert command == [str(python), "-u", "-m", "core.hand_tracking",
                           "--worker", memory.name, backend]
        assert kwargs["cwd"] == tmp_path
        assert kwargs["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
        stop.set()
        assert worker.process(np.zeros((2, 2, 3), dtype=np.uint8)) == []
        assert process.stdin.getvalue() == b""
    finally:
        worker.close()
    assert process.actions == ["terminate", ("wait", 0.5)]
    assert process.stdin.closed and process.stdout.closed and memory.closed and memory.unlinked


@pytest.mark.parametrize("backend", [None, True, 0, "", "WiLoR", "mediapipe", [], {}])
def test_unknown_worker_backend_is_rejected_before_resources(monkeypatch, backend):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid backend must not allocate or launch")
    monkeypatch.setattr(tracking.shared_memory, "SharedMemory", forbidden)
    monkeypatch.setattr(tracking.subprocess, "Popen", forbidden)
    with pytest.raises(ValueError, match="Unknown"):
        tracking.RTMPoseWorker(threading.Event(), backend=backend)


def test_missing_wilor_installation_does_not_fall_back_to_rtmpose(monkeypatch):
    monkeypatch.setattr(tracking, "probe_wilor", lambda: {
        "available": False, "reason": "WiLoR checkpoint is missing"})
    def forbidden(*args, **kwargs):
        raise AssertionError("Unavailable WiLoR must not load a different tracker")
    monkeypatch.setattr(tracking, "probe_rtmpose", forbidden)
    monkeypatch.setattr(tracking.subprocess, "Popen", forbidden)
    monkeypatch.setattr(tracking.shared_memory, "SharedMemory", forbidden)
    with pytest.raises(RuntimeError, match="WiLoR checkpoint is missing"):
        tracking.create_wilor(threading.Event())
