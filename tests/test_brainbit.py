"""Hardware-free adapter, IPC and cancellation coverage."""

import json
import sys
import threading
import time

import pytest

from plugins import brainbit


def snapshot(state="disconnected"):
    return {
        "state": state, "available": True, "busy": False, "text": state,
        "devices": [{"id": "temporary-choice", "name": "BrainBit", "family": "LEBrainBit2"}],
        "device": {"name": "BrainBit", "family": "LEBrainBit2", "battery": 81,
                   "firmware": "1.2.3"} if state == "connected" else None,
    }


class FakeWorker:
    instances = []

    def __init__(self, on_event, on_exit):
        self.on_event, self.on_exit = on_event, on_exit
        self.calls = []
        self.closed = False
        self.failure = None
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.state = "disconnected"
        self.instances.append(self)

    def request(self, action, args, **kwargs):
        self.calls.append((action, args, kwargs))
        self.started.set()
        if self.block:
            assert self.release.wait(3), "test worker was not released"
        if self.failure:
            raise self.failure
        if action == "connect":
            self.state = "connected"
        elif action == "disconnect":
            self.state = "disconnected"
        return {"result": snapshot(self.state)}

    def close(self):
        self.closed = True
        self.release.set()


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.setattr(brainbit.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(brainbit, "_WorkerClient", FakeWorker)
    FakeWorker.instances = []
    adapter = brainbit.BrainBit(scan_seconds=0, connect_timeout=.1, request_timeout=.2)
    yield adapter
    adapter.close()


def test_lazy_disabled_missing_and_cached_status(monkeypatch):
    monkeypatch.setattr(brainbit.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(brainbit, "_WorkerClient", lambda *_args: pytest.fail("worker must stay lazy"))
    assert brainbit.BrainBit(enabled=False).status()["state"] == "disabled"
    adapter = brainbit.BrainBit()
    assert adapter.call("discover")["state"] == "unavailable"
    assert not adapter.status()["available"]
    adapter.close()


def test_discover_connect_refresh_disconnect_repeated(driver):
    initial = driver.status()
    assert FakeWorker.instances == []
    assert all(action["confirm"] is False for action in driver.schema()["actions"])
    for _ in range(3):
        found = driver.call("discover")
        worker = FakeWorker.instances[-1]
        found["devices"].clear()
        assert driver.status()["devices"]
        connected = driver.call("connect", device_id="temporary-choice")
        assert connected["state"] == "connected"
        assert connected["device"]["battery"] == 81
        count = len(worker.calls)
        assert driver.call("status")["device"] == connected["device"]
        assert len(worker.calls) == count
        assert driver.call("status", refresh=True)["state"] == "connected"
        disconnected = driver.call("disconnect")
        assert disconnected["state"] == "disconnected"
        assert disconnected["devices"] == []
        assert worker.closed
        assert disconnected["revision"] > initial["revision"]
    assert len(FakeWorker.instances) == 3


def test_selection_and_action_validation_preserve_connection(driver):
    assert "error" in driver.call("connect", device_id="not-discovered")
    assert "error" in driver.call("status", refresh="yes")
    assert "error" in driver.call("record")
    assert "error" in driver.call("discover", arbitrary=True)
    assert FakeWorker.instances == []
    driver.call("discover")
    driver.call("connect", device_id="temporary-choice")
    worker = FakeWorker.instances[-1]
    assert "error" in driver.call("discover")
    assert "error" in driver.call("connect", device_id="temporary-choice")
    assert not worker.closed
    assert driver.status()["state"] == "connected"


@pytest.mark.parametrize("failure", [TimeoutError("MAC-secret"), OSError("serial-secret")])
def test_failure_terminates_worker_sanitizes_error_and_allows_retry(driver, failure):
    driver.call("discover")
    worker = FakeWorker.instances[-1]
    worker.failure = failure
    result = driver.call("connect", device_id="temporary-choice")
    assert result["state"] == "error"
    assert not result["busy"]
    assert "secret" not in json.dumps(result)
    assert worker.closed and result["devices"] == []
    assert driver.call("discover")["state"] == "disconnected"
    assert len(FakeWorker.instances) == 2


def test_busy_is_immediate_and_disconnect_cancels_stale_completion(driver):
    driver.call("discover")
    worker = FakeWorker.instances[-1]
    worker.started.clear()
    worker.block = True
    results = []
    operation = threading.Thread(target=lambda: results.append(
        driver.call("connect", device_id="temporary-choice")))
    operation.start()
    assert worker.started.wait(1)
    before = time.monotonic()
    assert driver.status()["state"] == "connecting"
    assert driver.call("discover")["busy"]
    assert driver.call("status", refresh=True)["busy"]
    assert time.monotonic() - before < .5
    disconnected = driver.call("disconnect")
    operation.join(1)
    assert not operation.is_alive()
    assert disconnected["state"] == "disconnected"
    assert not disconnected["busy"] and worker.closed
    assert driver.status()["state"] == "disconnected"
    assert len(worker.calls) == 2
    assert results[0]["state"] == "disconnected"
    worker.on_event(snapshot("connected"))
    assert driver.status()["state"] == "disconnected"


def test_close_cancels_operation_and_never_restarts(driver):
    driver.call("discover")
    worker = FakeWorker.instances[-1]
    worker.started.clear()
    worker.block = True
    operation = threading.Thread(target=lambda: driver.call("connect", device_id="temporary-choice"))
    operation.start()
    assert worker.started.wait(1)
    driver.close()
    operation.join(1)
    assert not operation.is_alive()
    assert worker.closed
    assert driver.call("discover")["state"] == "disabled"
    assert not driver.status()["busy"]


def test_crash_during_connect_cannot_publish_a_late_success(driver):
    driver.call("discover")
    worker = FakeWorker.instances[-1]
    worker.started.clear()
    worker.block = True
    operation = threading.Thread(target=lambda: driver.call("connect", device_id="temporary-choice"))
    operation.start()
    assert worker.started.wait(1)
    worker.on_exit()
    operation.join(1)
    assert not operation.is_alive()
    assert worker.closed
    assert driver.status()["state"] == "error"
    assert not driver.status()["busy"]


def test_callbacks_crash_and_link_loss_recover(driver):
    driver.call("discover")
    driver.call("connect", device_id="temporary-choice")
    worker = FakeWorker.instances[-1]
    status = snapshot("connected")
    status["device"]["battery"] = 62
    revision = driver.status()["revision"]
    worker.on_event(status)
    assert driver.status()["device"]["battery"] == 62
    assert driver.status()["revision"] > revision
    worker.on_event(snapshot("disconnected"))
    assert worker.closed
    assert driver.status()["state"] == "disconnected"
    assert driver.status()["devices"] == []
    driver.call("discover")
    new_worker = FakeWorker.instances[-1]
    new_worker.on_exit()
    assert new_worker.closed
    assert driver.status()["state"] == "error"
    driver.call("discover")
    assert driver.status()["state"] == "disconnected"


def test_sdk_load_failure_is_unavailable(driver, monkeypatch):
    driver.call("discover")
    worker = FakeWorker.instances[-1]
    monkeypatch.setattr(worker, "request", lambda *_args, **_kwargs: {
        "unavailable": True, "error": "raw sensitive SDK exception"})
    result = driver.call("connect", device_id="temporary-choice")
    assert result["state"] == "unavailable" and not result["available"]
    assert "sensitive" not in json.dumps(result)
    assert worker.closed


def test_real_action_gate_accepts_refresh_and_rejects_signal_args(driver, monkeypatch, wired):
    from core import actions as actions_mod
    monkeypatch.setattr(actions_mod, "_drivers", {"brainbit": driver})
    registry, _journal, _memory = wired
    result = registry.dispatch("hw_call", {"device": "brainbit", "action": "status",
                                          "args": {"refresh": True}}, actor="user")
    assert result["result"]["state"] == "disconnected"
    rejected = registry.dispatch("hw_call", {"device": "brainbit", "action": "status",
                                            "args": {"record": True}}, actor="user")
    assert rejected.get("denied")


def test_idle_close_requests_graceful_disconnect_and_ignores_late_events(driver):
    driver.call("discover")
    driver.call("connect", device_id="temporary-choice")
    worker = FakeWorker.instances[-1]
    driver.close()
    assert worker.calls[-1][0] == "disconnect"
    assert worker.calls[-1][2]["timeout"] <= 2
    assert worker.closed
    worker.on_event(snapshot("connected"))
    worker.on_exit()
    assert driver.status()["state"] == "disabled"


def test_idle_close_reaps_worker_after_graceful_timeout(driver):
    driver.call("discover")
    worker = FakeWorker.instances[-1]
    worker.failure = TimeoutError("stuck disconnect")
    driver.close()
    assert worker.calls[-1][0] == "disconnect"
    assert worker.closed and not driver.status()["busy"]


def test_real_ipc_framing_timeout_and_process_reaping():
    # A harmless child implements the wire protocol, without importing the SDK.
    code = (
        "import json,sys,time\n"
        "for line in sys.stdin:\n"
        " r=json.loads(line)\n"
        " if r['action']=='hang': time.sleep(30)\n"
        " print(json.dumps({'id':r['id'],'result':{'state':'disconnected'}}),flush=True)\n"
    )
    events = []
    worker = brainbit._WorkerClient(events.append, lambda: None,
                                   command=[sys.executable, "-u", "-c", code])
    try:
        result = worker.request("status", {}, timeout=2, scan_seconds=0)
        assert result["result"]["state"] == "disconnected"
        with pytest.raises(TimeoutError):
            worker.request("hang", {}, timeout=.1, scan_seconds=0)
    finally:
        worker.close()
    assert worker._process.poll() is not None
    assert not worker._reader.is_alive()


@pytest.mark.parametrize("code", ["raise SystemExit(0)", "print('x'*70000,flush=True)"])
def test_real_ipc_crash_and_oversized_frame_fail_promptly(code):
    worker = brainbit._WorkerClient(lambda _event: None, lambda: None,
                                   command=[sys.executable, "-u", "-c", code])
    try:
        with pytest.raises(OSError):
            worker.request("status", {}, timeout=2, scan_seconds=0)
    finally:
        worker.close()
    assert worker._process.poll() is not None
