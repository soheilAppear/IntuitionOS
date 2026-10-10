"""Connection panel API behavior with no SDK import or hardware access."""
import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from core import actions as action_module
from interface import server


HEADERS = {"host": "127.0.0.1:7432", "x-intuition-brainbit": "1"}


class Driver:
    name = "brainbit"

    def __init__(self):
        self.calls = []
        self.state = "disconnected"
        self.block = False
        self.entered = threading.Event()
        self.release = threading.Event()

    def schema(self):
        return {"actions": [
            {"name": name, "args": args, "confirm": False}
            for name, args in (("status", ["refresh"]), ("discover", []),
                               ("connect", ["device_id"]), ("disconnect", []))]}

    def status(self):
        return {"state": self.state, "busy": self.state == "connecting",
                "available": True, "devices": [], "device": None, "text": self.state}

    def call(self, action, **kwargs):
        self.calls.append((action, kwargs, threading.get_ident()))
        if action == "connect" and self.block:
            self.state = "connecting"
            self.entered.set()
            assert self.release.wait(3), "disconnect was blocked behind connect"
        if action == "disconnect":
            self.state = "disconnected"
            self.release.set()
        if action == "discover":
            return {**self.status(), "error": "Bluetooth unavailable."}
        return self.status()


@pytest.fixture
def api(monkeypatch):
    driver = Driver()
    monkeypatch.setattr(server, "_state", {"brainbit": driver})
    monkeypatch.setattr(server, "_clients", set())
    monkeypatch.setattr(action_module, "_drivers", {"brainbit": driver})
    monkeypatch.setattr(action_module, "_memory", None)
    monkeypatch.setattr(action_module, "_journal_ref", [None])
    monkeypatch.setattr(action_module, "_logger", lambda _: None)
    return driver, TestClient(server.app)


@pytest.mark.parametrize("headers", [
    {}, {**HEADERS, "origin": "null"}, {**HEADERS, "origin": "https://example.org"},
    {**HEADERS, "host": "attacker.example:7432"}, {**HEADERS, "x-intuition-brainbit": "0"},
])
def test_only_native_local_hud_can_access_status_or_controls(api, headers):
    driver, client = api
    assert client.get("/brainbit/status", headers=headers).status_code == 403
    assert client.post("/brainbit/disconnect", headers=headers, json={}).status_code == 403
    assert driver.calls == []


def test_get_is_cached_and_never_discovers_or_connects(api):
    driver, client = api
    response = client.get("/brainbit/status", headers=HEADERS)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["state"] == "disconnected"
    assert driver.calls == []


@pytest.mark.parametrize("operation,payload", [
    ("connect", {}), ("connect", {"device_id": ""}),
    ("connect", {"device_id": True}), ("connect", {"device_id": "x" * 129}),
    ("connect", {"device_id": "opaque", "record": True}), ("disconnect", {"force": True}),
    ("discover", None), ("refresh", []),
])
def test_invalid_controls_cannot_reach_driver(api, operation, payload):
    driver, client = api
    assert client.post(f"/brainbit/{operation}", headers=HEADERS, json=payload).status_code == 400
    assert driver.calls == []


def test_explicit_connect_uses_user_gate_and_does_not_change_safe_mode(api, monkeypatch):
    driver, client = api
    seen = []
    dispatch = server.actions.dispatch

    def gated(name, args, **kwargs):
        seen.append((name, args, kwargs))
        return dispatch(name, args, **kwargs)

    monkeypatch.setattr(server.actions, "dispatch", gated)
    response = client.post("/brainbit/connect", headers=HEADERS, json={"device_id": "opaque"})
    assert response.status_code == 200
    assert seen == [("hw_call", {"device": "brainbit", "action": "connect",
                                "args": {"device_id": "opaque"}}, {"actor": "user"})]
    assert driver.calls[0][:2] == ("connect", {"device_id": "opaque"})


def test_discovery_error_is_visible_and_refresh_is_only_metadata(api):
    driver, client = api
    response = client.post("/brainbit/discover", headers=HEADERS, json={})
    assert response.status_code == 503
    assert response.json()["error"] == "Bluetooth unavailable."
    response = client.post("/brainbit/refresh", headers=HEADERS, json={})
    assert response.status_code == 200
    assert driver.calls[-1][:2] == ("status", {"refresh": True})
    assert client.post("/brainbit/record", headers=HEADERS, json={}).status_code == 404


def request(payload):
    async def receive():
        return {"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}
    return Request({"type": "http", "method": "POST", "path": "/brainbit/connect",
                    "headers": [(k.encode(), v.encode()) for k, v in HEADERS.items()]}, receive)


def test_connect_does_not_block_cache_health_or_disconnect(api):
    driver, _ = api
    driver.block = True

    async def exercise():
        task = asyncio.create_task(server.brainbit_operation("connect", request({"device_id": "opaque"})))
        assert await asyncio.to_thread(driver.entered.wait, 1)
        try:
            cached = await asyncio.wait_for(server.brainbit_status(request({})), 0.2)
            assert json.loads(cached.body)["state"] == "connecting"
            assert (await asyncio.wait_for(server.health(), 0.2))["ok"]
            disconnected = await asyncio.wait_for(server.brainbit_operation("disconnect", request({})), 1)
            assert disconnected.status_code == 200
            assert (await asyncio.wait_for(task, 1)).status_code == 200
            assert all(tid != threading.get_ident() for _, _, tid in driver.calls)
        finally:
            driver.release.set()
            await task

    asyncio.run(exercise())


def test_background_refresh_never_scans_or_reconnects(api):
    driver, _ = api
    driver.state = "connected"

    async def exercise():
        task = asyncio.create_task(server._poll_brainbit(driver))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert [c[:2] for c in driver.calls] == [("status", {"refresh": True})]


@pytest.mark.parametrize("signal_state", ["starting", "running", "stale", "contact", "stopping"])
def test_active_signal_broadcasts_before_hud_expiry_without_starting_acquisition(api, monkeypatch, signal_state):
    from plugins.brainbit import SIGNAL_FRESH_SECONDS

    driver, _ = api
    driver.state = "connected"
    elapsed = 0.0
    broadcasts = []
    metadata_refreshes = []
    status = driver.status
    call = driver.call
    sleep = asyncio.sleep

    # Even identical healthy summaries must renew the HUD's freshness clock.
    def signal_status():
        return {**status(), "signal": {"state": signal_state, "age_seconds": 0.05,
                                       "freshness_seconds": SIGNAL_FRESH_SECONDS}}

    def tracked_call(action, **kwargs):
        metadata_refreshes.append(elapsed)
        return call(action, **kwargs)

    async def broadcast(message):
        broadcasts.append((elapsed, message))

    async def immediate_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    async def advance(seconds):
        nonlocal elapsed
        await sleep(0)
        elapsed += seconds
        if elapsed > 3.5:
            raise asyncio.CancelledError

    monkeypatch.setattr(driver, "status", signal_status)
    monkeypatch.setattr(driver, "call", tracked_call)
    monkeypatch.setattr(server, "time", SimpleNamespace(monotonic=lambda: elapsed))
    monkeypatch.setattr(server, "_broadcast", broadcast)
    monkeypatch.setattr(server.asyncio, "sleep", advance)
    monkeypatch.setattr(server.asyncio, "to_thread", immediate_thread)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(server._poll_brainbit(driver))

    assert broadcasts[-1][0] >= 3.5
    assert all(current[0] - previous[0] + 0.05 < SIGNAL_FRESH_SECONDS
               for previous, current in zip(broadcasts, broadcasts[1:]))
    assert all(message["type"] == "brainbit_status" for _, message in broadcasts)
    assert metadata_refreshes == [0.0, 3.0]
    assert [entry[:2] for entry in driver.calls] == [("status", {"refresh": True})] * 2


def test_blocked_metadata_refresh_cannot_delay_signal_broadcasts_and_is_cancelled(api, monkeypatch):
    driver, _ = api
    driver.state = "connected"
    elapsed = 0.0
    broadcasts = []
    refreshes = []
    cleanup = []
    sleep = asyncio.sleep
    status = driver.status

    async def blocked_thread(function, *args, **kwargs):
        refreshes.append((function, args, kwargs))
        try:
            await asyncio.Event().wait()
        finally:
            cleanup.append(True)

    async def broadcast(message):
        broadcasts.append((elapsed, message))

    async def advance(seconds):
        nonlocal elapsed
        await sleep(0)
        elapsed += seconds
        if elapsed > 3.5:
            raise asyncio.CancelledError

    monkeypatch.setattr(driver, "status", lambda: {
        **status(), "signal": {"state": "running", "age_seconds": 0.01}})
    monkeypatch.setattr(server, "time", SimpleNamespace(monotonic=lambda: elapsed))
    monkeypatch.setattr(server, "_broadcast", broadcast)
    monkeypatch.setattr(server.asyncio, "sleep", advance)
    monkeypatch.setattr(server.asyncio, "to_thread", blocked_thread)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(server._poll_brainbit(driver))

    assert broadcasts[-1][0] >= 3.5
    assert len(broadcasts) >= 14
    assert refreshes == [(driver.call, ("status",), {"refresh": True})]
    assert cleanup == [True]


def test_speculative_actor_cannot_connect(api):
    driver, _ = api
    result = server.actions.dispatch("hw_call", {"device": "brainbit", "action": "connect",
                                               "args": {"device_id": "opaque"}}, actor="anticipator")
    assert result["denied"]
    assert driver.calls == []
