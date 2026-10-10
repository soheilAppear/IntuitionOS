"""Local acquisition previews with a fake SDK process; no device access."""

import json
import sys
import threading
import time

import pytest

from plugins import brainbit


class AcquisitionWorker:
    def __init__(self, on_event, on_exit):
        self.on_event, self.on_exit = on_event, on_exit
        self.closed = False
        self.calls = []
        self.failure = None
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.acquisition = None

    def request(self, action, args, **kwargs):
        self.calls.append(action)
        self.started.set()
        if self.block:
            assert self.release.wait(2)
        if self.failure:
            raise self.failure
        if action == "start_acquisition":
            self.acquisition = dict(args)
            return {"result": {**args, "state": "running", "channels": [{"num": 0, "name": "O1"}],
                               "nominal_hz": 250, "units": "V" if args["mode"] == "signal" else "ohm"}}
        if action == "stop_acquisition":
            return {"result": {"state": "stopped", "mode": None, "session_id": None}}
        return {"result": {"state": "disconnected" if action == "disconnect" else "connected",
                           "available": True, "device": {"name": "BrainBit"}, "devices": []}}

    def packet(self, counters=(0,), values=None, **extra):
        assert self.acquisition is not None
        now = time.monotonic()
        data = {**self.acquisition, "channels": [{"num": 0, "name": "O1"}], "nominal_hz": 250,
                "units": "V" if self.acquisition["mode"] == "signal" else "ohm",
                "packets": [{"counter": counter, "marker": 0, "samples": [values[index] if values else index * .000001],
                             "host_received_monotonic": now - (len(counters) - 1 - index) / 250,
                             "estimated_monotonic": now - (len(counters) - 1 - index) / 250}
                            for index, counter in enumerate(counters)], **extra}
        self.on_event({"_acquisition": data})
        return data

    def close(self):
        self.closed = True
        self.release.set()


@pytest.fixture
def connected(monkeypatch):
    monkeypatch.setattr(brainbit.importlib.util, "find_spec", lambda _name: object())
    driver = brainbit.BrainBit(request_timeout=.1)
    worker = AcquisitionWorker(lambda payload: driver._event(0, payload), lambda: driver._exited(0))
    driver._worker = worker
    driver._snapshot.update(state="connected", device={"name": "BrainBit", "family": "LEBrainBit2"})
    yield driver, worker
    driver.close()


def test_explicit_only_separate_samples_and_stats(connected):
    driver, worker = connected
    assert not worker.calls
    assert driver.call("start_acquisition", mode="signal")["error"]
    assert "start_acquisition" not in [entry["name"] for entry in driver.schema()["actions"]]
    assert driver.start_acquisition()["state"] == "running"
    worker.packet(counters=(0xfffffffe, 0xffffffff, 0, 0, 2),
                  values=(.000001, .000002, float("inf"), .000004, .000005), nonfinite=1, queue_drops=3)
    result = driver.acquisition_snapshot()
    assert len(result["samples"]) == 5
    assert result["samples"][2]["samples"] == [None]
    assert result["stats"]["gaps"] == 1 and result["stats"]["duplicates"] == 1
    assert result["stats"]["nonfinite"] == 1 and result["stats"]["queue_drops"] == 3
    assert result["stats"]["age_seconds"] is not None
    assert result["stats"]["channels"][0]["peak_to_peak_v"] == pytest.approx(.000004)
    json.dumps(result, allow_nan=False)
    status = json.dumps(driver.call("status"))
    assert "samples" not in status and "contact_precheck" not in status and "packets" not in status


def test_contact_memory_and_stop_clear_stale_signal_packets(connected):
    driver, worker = connected
    driver.start_acquisition("contact")
    old = worker.packet(values=[5000])
    assert driver.acquisition_snapshot()["contact_precheck"]["values"] == [5000]
    driver.stop_acquisition()
    result = driver.acquisition_snapshot()
    assert result["samples"] == [] and result["state"] == "stopped"
    assert result["contact_precheck"]["units"] == "ohm"
    driver.start_acquisition("signal")
    worker.on_event({"_acquisition": old})
    result = driver.acquisition_snapshot()
    assert result["samples"] == []
    assert result["contact_precheck"]["age_seconds"] >= 0
    worker.packet(counters=(21,))
    assert driver.acquisition_snapshot()["stats"]["gaps"] == 0
    driver.call("disconnect")
    result = driver.acquisition_snapshot()
    assert result["contact_precheck"] is None and result["samples"] == []


def test_five_second_bounded_cache_and_downsampling(connected):
    driver, worker = connected
    driver.start_acquisition()
    for offset in range(0, 2000, 100):
        worker.packet(counters=tuple(range(offset, offset + 100)))
    assert len(driver._samples) == 1250
    result = driver.acquisition_snapshot()
    assert len(result["samples"]) <= 250
    result["samples"].clear()
    assert driver.acquisition_snapshot()["samples"]


def test_signal_window_preserves_full_contiguous_counters_and_newest_packet(connected, monkeypatch):
    driver, worker = connected
    driver.start_acquisition()
    monkeypatch.setattr(brainbit.time, "monotonic", lambda: 100.0)
    for offset in range(0, 2000, 100):
        worker.packet(counters=tuple(range(offset, offset + 100)))
    calls = list(worker.calls)

    window = driver.acquisition_signal_window()
    preview = driver.acquisition_snapshot()

    assert [packet["counter"] for packet in window["samples"]] == list(range(750, 2000))
    assert window["samples"][-1]["counter"] == preview["samples"][-1]["counter"] == 1999
    assert len(preview["samples"]) <= 250
    assert {key: value for key, value in window.items() if key != "samples"} == {
        key: value for key, value in preview.items() if key != "samples"}
    assert worker.calls == calls
    window["samples"][0]["samples"][0] = 123.0
    window["channels"][0]["name"] = "changed"
    again = driver.acquisition_signal_window()
    assert again["samples"][0]["samples"][0] != 123.0
    assert again["channels"][0]["name"] == "O1"


def test_signal_window_expires_old_packets_and_keeps_finite_values(connected, monkeypatch):
    driver, worker = connected
    driver.start_acquisition()
    monkeypatch.setattr(brainbit.time, "monotonic", lambda: 100.0)
    worker.packet(counters=(0, 1, 2), values=(float("nan"), float("inf"), -.000001))
    monkeypatch.setattr(brainbit.time, "monotonic", lambda: 104.99)
    window = driver.acquisition_signal_window()
    assert [packet["samples"] for packet in window["samples"]] == [[None], [None], [-.000001]]
    json.dumps(window, allow_nan=False)
    monkeypatch.setattr(brainbit.time, "monotonic", lambda: 105.001)
    assert driver.acquisition_signal_window()["samples"] == []


def test_signal_window_is_not_a_public_action_and_stop_clears_it(connected):
    driver, worker = connected
    driver.start_acquisition()
    worker.packet(counters=(0, 1, 2))
    assert driver.acquisition_signal_window()["samples"]
    assert "acquisition_signal_window" not in [action["name"] for action in driver.schema()["actions"]]
    assert "error" in driver.call("acquisition_signal_window")
    public_status = json.dumps(driver.call("status"))
    assert "samples" not in public_status and "packets" not in public_status
    driver.stop_acquisition()
    assert driver.acquisition_signal_window()["samples"] == []
    assert driver.acquisition_snapshot()["samples"] == []


@pytest.mark.parametrize("action", ["stop_acquisition", "start_acquisition"])
def test_acquisition_failure_releases_worker_and_is_not_stopped(connected, action):
    driver, worker = connected
    driver.start_acquisition()
    worker.packet()
    worker.failure = TimeoutError("secret native device identifier")
    result = driver.stop_acquisition() if action == "stop_acquisition" else driver.start_acquisition("contact")
    assert result["state"] == "error" and "unconfirmed" in result["error"]
    assert result["samples"] == [] and result["contact_precheck"] is None
    assert worker.closed and driver.status()["state"] == "error"
    assert "secret" not in json.dumps(result)


def test_stop_cancels_pending_start_and_rejects_late_success(connected):
    driver, worker = connected
    worker.block = True
    results = []
    thread = threading.Thread(target=lambda: results.append(driver.start_acquisition()))
    thread.start()
    assert worker.started.wait(1)
    assert driver.acquisition_snapshot()["state"] == "starting"
    stopped = driver.stop_acquisition()
    thread.join(1)
    assert not thread.is_alive()
    assert stopped["state"] == "error" and worker.closed
    assert results[0]["state"] == "error"


def test_device_loss_clears_contact_and_signals(connected):
    driver, worker = connected
    driver.start_acquisition("contact")
    old = worker.packet(values=[8000])
    worker.on_event({"state": "disconnected"})
    result = driver.acquisition_snapshot()
    assert result["state"] == "error" and result["samples"] == []
    assert result["contact_precheck"] is None and worker.closed
    worker.on_event({"_acquisition": old})
    assert driver.acquisition_snapshot()["samples"] == []


def test_polled_device_loss_does_not_claim_acquisition_stopped(connected, monkeypatch):
    driver, worker = connected
    driver.start_acquisition()
    worker.packet()
    monkeypatch.setattr(worker, "request", lambda *_args, **_kwargs: {
        "result": {"state": "disconnected", "device": None, "devices": []}})
    driver.call("status", refresh=True)
    result = driver.acquisition_snapshot()
    assert result["state"] == "error" and "unconfirmed" in result["error"]
    assert worker.closed and result["samples"] == []


def test_shutdown_stop_failure_not_claimed_as_stopped(connected):
    driver, worker = connected
    driver.start_acquisition()
    worker.failure = TimeoutError()
    driver.close()
    result = driver.acquisition_snapshot()
    assert result["state"] == "error" and "unconfirmed" in result["error"]
    assert worker.closed


def test_ipc_acquisition_events_do_not_consume_request_response():
    code = ("import json,sys\n"
            "r=json.loads(sys.stdin.readline())\n"
            "print(json.dumps({'acquisition':{'session_id':'test','packets':[]}}),flush=True)\n"
            "print(json.dumps({'id':r['id'],'result':{'state':'running'}}),flush=True)\n")
    events = []
    worker = brainbit._WorkerClient(events.append, lambda: None,
                                   command=[sys.executable, "-u", "-c", code])
    try:
        result = worker.request("start_acquisition", {}, timeout=2, scan_seconds=0)
        assert result["result"]["state"] == "running"
        assert events == [{"_acquisition": {"session_id": "test", "packets": []}}]
    finally:
        worker.close()


def test_signal_status_is_cached_private_and_tracks_stream_freshness(connected, monkeypatch):
    from core.multimodal import MultimodalPreview

    driver, worker = connected
    monkeypatch.setattr(brainbit.time, "monotonic", lambda: 100.0)
    assert driver.status()["signal"]["state"] == "stopped"
    driver.start_acquisition()
    assert driver.status()["signal"]["state"] == "starting"
    worker.packet(counters=(0, 1, 2), values=(1e-6, 2e-6, 3e-6))
    calls = list(worker.calls)
    signal = driver.call("status")["signal"]
    assert signal["state"] == "running"
    assert signal["channel_count"] == 1 and signal["sample_count"] == 3
    assert signal["nominal_hz"] == 250
    assert signal["received_rate_hz"] == pytest.approx(250)
    assert signal["freshness_seconds"] == MultimodalPreview.EEG_FRESH_SECONDS
    assert signal["age_seconds"] == 0
    assert set(signal).isdisjoint({"samples", "packets", "channels", "session_id", "contact_precheck"})
    assert signal["contact"] == {"available": False, "age_seconds": None}
    signal["issue_counts"]["gaps"] = 999
    assert driver.status()["signal"]["issue_counts"]["gaps"] == 0
    monkeypatch.setattr(brainbit.time, "monotonic", lambda: 100.76)
    assert driver.status()["signal"]["state"] == "stale"
    assert worker.calls == calls
    worker.packet(counters=(3,))
    assert driver.status()["signal"]["state"] == "running"
    driver.stop_acquisition()
    signal = driver.status()["signal"]
    assert signal["state"] == "stopped" and signal["sample_count"] == 0
    assert signal["age_seconds"] is None
    json.dumps(signal, allow_nan=False)


def test_signal_status_distinguishes_contact_and_invalid_data(connected, monkeypatch):
    driver, worker = connected
    monkeypatch.setattr(brainbit.time, "monotonic", lambda: 100.0)
    driver.start_acquisition("contact")
    worker.packet(values=[5000])
    signal = driver.status()["signal"]
    assert signal["state"] == "contact"
    assert signal["contact"] == {"available": True, "age_seconds": 0.0}
    assert "5000" not in json.dumps(signal)
    driver.start_acquisition("signal")
    worker.packet(values=[float("nan")], nonfinite=1)
    signal = driver.status()["signal"]
    assert signal["state"] == "error"
    assert signal["issue_counts"]["nonfinite"] == 1
    assert signal["contact"]["available"]
    worker.packet(counters=(1,), values=[1e-6])
    assert driver.status()["signal"]["state"] == "running"
    worker.on_event({"state": "disconnected"})
    signal = driver.status()["signal"]
    assert signal["state"] == "error" and "unconfirmed" in signal["text"]
    assert signal["sample_count"] == 0 and not signal["contact"]["available"]


def test_signal_status_handles_idle_connection_states(monkeypatch):
    monkeypatch.setattr(brainbit.importlib.util, "find_spec", lambda _name: None)
    assert brainbit.BrainBit(enabled=False).status()["signal"]["state"] == "disabled"
    assert brainbit.BrainBit().status()["signal"]["state"] == "unavailable"
    monkeypatch.setattr(brainbit.importlib.util, "find_spec", lambda _name: object())
    driver = brainbit.BrainBit()
    assert driver.status()["signal"]["state"] == "disconnected"
    driver.close()
    assert driver.status()["signal"]["state"] == "disabled"


def test_finite_extreme_values_cannot_overflow_preview_statistics(connected):
    driver, worker = connected
    driver.start_acquisition()
    worker.packet(counters=(0, 1), values=(1e308, -1e308))
    result = driver.acquisition_snapshot()
    assert result["stats"]["channels"][0]["rms_v"] == pytest.approx(1e308)
    assert result["stats"]["channels"][0]["peak_to_peak_v"] is None
    json.dumps(result, allow_nan=False)
