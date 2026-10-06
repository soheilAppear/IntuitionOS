"""BrainBit discovery and connection lifecycle without native SDK or hardware."""

import builtins
from enum import Enum
import importlib
import json
from types import SimpleNamespace
from uuid import UUID

import pytest


class FakeSensorFamily(Enum):
    LEBrainBit = 1
    LEBrainBitBlack = 2
    LEBrainBit2 = 3
    LEBrainBitPro = 4
    LEBrainBitFlex = 5
    LECallibri = 6


class FakeSensorState(Enum):
    StateInRange = 1
    StateOutOfRange = 2


ALLOWED_FAMILIES = {
    FakeSensorFamily.LEBrainBit,
    FakeSensorFamily.LEBrainBitBlack,
    FakeSensorFamily.LEBrainBit2,
    FakeSensorFamily.LEBrainBitPro,
    FakeSensorFamily.LEBrainBitFlex,
}


def sensor_info(name="BrainBit test", family=FakeSensorFamily.LEBrainBit, suffix="01"):
    return SimpleNamespace(
        Name=name,
        SensFamily=family,
        Address=f"AA:BB:CC:DD:EE:{suffix}",
        SerialNumber=f"private-serial-{suffix}",
    )


class FakeSensor:
    """Only metadata, connection state, and their callbacks are supported."""

    def __init__(self):
        self.forbidden_accesses = []
        self.state = FakeSensorState.StateInRange
        self.batt_power = 73
        self.version = SimpleNamespace(FwMajor=1, FwMinor=2, FwPatch=3)
        self.sensorStateChanged = None
        self.batteryChanged = None
        self.disconnect_calls = 0

    @staticmethod
    def _forbidden(name):
        lowered = name.lower()
        return (
            "signal" in lowered
            or "resist" in lowered
            or name in {"exec_command", "execute", "send_command", "commands"}
            or name in {"name", "sens_family", "serial_number", "address"}
        )

    def __getattribute__(self, name):
        if name not in {"_forbidden", "forbidden_accesses"} and FakeSensor._forbidden(name):
            object.__getattribute__(self, "forbidden_accesses").append(("read", name))
            raise AssertionError(f"Unsupported sensor API read: {name}")
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if FakeSensor._forbidden(name):
            self.forbidden_accesses.append(("write", name))
            raise AssertionError(f"Unsupported sensor API write: {name}")
        object.__setattr__(self, name, value)

    def disconnect(self):
        self.disconnect_calls += 1
        self.state = FakeSensorState.StateOutOfRange
        if self.sensorStateChanged:
            self.sensorStateChanged(self, self.state)


class MetadataFailureSensor(FakeSensor):
    def __getattribute__(self, name):
        if name in {"batt_power", "version"}:
            raise RuntimeError("optional metadata is unavailable")
        return super().__getattribute__(name)


@pytest.fixture
def worker_module(monkeypatch):
    """Fail instead of loading a native package, even when locally installed."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "neurosdk" or name.startswith("neurosdk."):
            raise AssertionError("Tests must never import the real native SDK")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    return importlib.import_module("plugins.brainbit_worker")


@pytest.fixture
def harness(worker_module):
    sessions = []

    class FakeScanner:
        infos = [sensor_info()]
        sensor_factory = FakeSensor
        instances = []

        def __init__(self, filters):
            self.filters = list(filters)
            self.start_calls = 0
            self.stop_calls = 0
            self.created_infos = []
            self.created_sensors = []
            self.__class__.instances.append(self)

        def start(self):
            self.start_calls += 1

        def stop(self):
            self.stop_calls += 1

        def sensors(self):
            return list(self.infos)

        def create_sensor(self, info):
            self.created_infos.append(info)
            sensor = self.__class__.sensor_factory()
            self.created_sensors.append(sensor)
            return sensor

    def create_session(**kwargs):
        session = worker_module.SDKSession(
            scanner_class=FakeScanner,
            family_enum=FakeSensorFamily,
            state_enum=FakeSensorState,
            scan_seconds=0,
            **kwargs,
        )
        sessions.append(session)
        return session

    yield SimpleNamespace(scanner=FakeScanner, session=create_session)
    for session in sessions:
        session.close()
    for scanner in FakeScanner.instances:
        for sensor in scanner.created_sensors:
            assert sensor.forbidden_accesses == []


def test_discovery_filters_families_and_exposes_only_opaque_ids(harness):
    harness.scanner.infos = [
        sensor_info(f"BrainBit {family.name}", family, f"{index:02}")
        for index, family in enumerate(FakeSensorFamily)
    ]
    session = harness.session()

    status = session.handle("discover")

    scanner = harness.scanner.instances[0]
    assert set(scanner.filters) == ALLOWED_FAMILIES
    assert scanner.start_calls == 1
    assert scanner.stop_calls >= 1
    assert scanner.created_sensors == []
    assert status["device"] is None
    assert status["available"] is True
    assert status["busy"] is False
    assert len(status["devices"]) == len(ALLOWED_FAMILIES)
    ids = [device["id"] for device in status["devices"]]
    assert len(set(ids)) == len(ids)
    for device in status["devices"]:
        assert set(device) == {"id", "name", "family"}
        assert str(UUID(device["id"])) == device["id"]
        assert device["family"] != "LECallibri"
    public_json = json.dumps(status)
    assert "AA:BB:CC" not in public_json
    assert "private-serial" not in public_json


def test_connect_uses_explicit_selected_device_and_refreshes_metadata(harness):
    first = sensor_info("First headset", suffix="01")
    second = sensor_info("Selected headset", FakeSensorFamily.LEBrainBit2, "02")
    harness.scanner.infos = [first, second]
    session = harness.session()
    discovered = session.handle("discover")["devices"]
    selected = next(device for device in discovered if device["name"] == second.Name)

    connected = session.handle("connect", device_id=selected["id"])

    scanner = harness.scanner.instances[0]
    assert scanner.created_infos == [second]
    assert connected["state"] == "connected"
    assert connected["device"] == {
        "name": second.Name,
        "family": selected["family"],
        "battery": 73,
        "firmware": "1.2.3",
    }
    sensor = scanner.created_sensors[0]
    sensor.batt_power = 42
    sensor.version = SimpleNamespace(FwMajor=2, FwMinor=0, FwPatch=8)

    refreshed = session.handle("status")

    assert refreshed["device"]["battery"] == 42
    assert refreshed["device"]["firmware"] == "2.0.8"
    assert session.snapshot()["device"] == refreshed["device"]


def test_unknown_device_id_never_connects_an_arbitrary_headset(harness):
    session = harness.session()
    session.handle("discover")

    with pytest.raises(ValueError):
        session.handle("connect", device_id="not-a-discovered-device")

    assert harness.scanner.instances[0].created_sensors == []
    assert session.snapshot()["device"] is None


def test_disconnect_is_repeatable_and_detaches_callbacks(harness):
    session = harness.session()
    device_id = session.handle("discover")["devices"][0]["id"]
    session.handle("connect", device_id=device_id)
    sensor = harness.scanner.instances[0].created_sensors[0]
    assert callable(sensor.sensorStateChanged)
    assert callable(sensor.batteryChanged)

    first = session.handle("disconnect")
    second = session.handle("disconnect")
    session.close()
    session.close()

    assert first["device"] is None
    assert second["device"] is None
    assert sensor.disconnect_calls == 1
    assert sensor.sensorStateChanged is None
    assert sensor.batteryChanged is None


def test_close_disconnects_connected_sensor_and_stops_scanner(harness):
    session = harness.session()
    device_id = session.handle("discover")["devices"][0]["id"]
    session.handle("connect", device_id=device_id)
    scanner = harness.scanner.instances[0]
    sensor = scanner.created_sensors[0]
    stops_before_close = scanner.stop_calls

    session.close()

    assert sensor.disconnect_calls == 1
    assert sensor.sensorStateChanged is None
    assert sensor.batteryChanged is None
    assert scanner.stop_calls > stops_before_close


def test_optional_metadata_failures_do_not_prevent_connection(harness):
    harness.scanner.sensor_factory = MetadataFailureSensor
    session = harness.session()
    device_id = session.handle("discover")["devices"][0]["id"]

    connected = session.handle("connect", device_id=device_id)
    refreshed = session.handle("status")

    for status in (connected, refreshed):
        assert status["state"] == "connected"
        assert status["device"]["name"] == "BrainBit test"
        assert status["device"]["battery"] is None
        assert status["device"]["firmware"] is None


def test_battery_and_connection_callbacks_update_public_status(harness):
    emitted = []
    session = harness.session(emit=emitted.append)
    device_id = session.handle("discover")["devices"][0]["id"]
    session.handle("connect", device_id=device_id)
    sensor = harness.scanner.instances[0].created_sensors[0]
    emitted.clear()

    sensor.batteryChanged(sensor, 29)

    assert session.snapshot()["device"]["battery"] == 29
    assert emitted
    assert emitted[-1]["device"]["battery"] == 29

    sensor.state = FakeSensorState.StateOutOfRange
    sensor.sensorStateChanged(sensor, sensor.state)

    assert session.snapshot()["state"] != "connected"
    assert emitted[-1]["state"] != "connected"


def test_reconnecting_after_disconnect_creates_a_fresh_sensor(harness):
    session = harness.session()
    device_id = session.handle("discover")["devices"][0]["id"]
    session.handle("connect", device_id=device_id)
    previous_sensor = harness.scanner.instances[0].created_sensors[0]
    previous_battery_callback = previous_sensor.batteryChanged
    previous_state_callback = previous_sensor.sensorStateChanged
    session.handle("disconnect")

    with pytest.raises(ValueError):
        session.handle("connect", device_id=device_id)
    new_device_id = session.handle("discover")["devices"][0]["id"]
    connected = session.handle("connect", device_id=new_device_id)

    assert new_device_id != device_id
    assert len(harness.scanner.instances) == 2
    current_sensor = harness.scanner.instances[1].created_sensors[0]
    assert previous_sensor.disconnect_calls == 1
    assert previous_sensor.sensorStateChanged is None
    assert previous_sensor.batteryChanged is None
    assert connected["state"] == "connected"
    assert current_sensor.disconnect_calls == 0

    # An SDK callback already queued when disconnect occurred must not alter
    # the newly selected sensor's state or metadata.
    previous_battery_callback(previous_sensor, 1)
    previous_state_callback(previous_sensor, FakeSensorState.StateOutOfRange)
    assert session.snapshot() == connected
