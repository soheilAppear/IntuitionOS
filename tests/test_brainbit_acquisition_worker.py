"""Deterministic BrainBit 2 acquisition tests; no SDK import or hardware access."""

import builtins
from enum import Enum
import importlib
import json
import math
from types import SimpleNamespace
from uuid import uuid4

import pytest


class SensorFamily(Enum):
    LEBrainBit2 = 3


class SensorState(Enum):
    StateInRange = 1
    StateOutOfRange = 2


class SensorParameter(Enum):
    BattPower = 1


class SensorCommand(Enum):
    StartSignal = 0
    StopSignal = 1
    StartResist = 2
    StopResist = 3


class SamplingFrequency(Enum):
    # The SDK enum value is an ordinal, not the sampling frequency in Hz.
    FrequencyHz250 = 4


class FakeSensor:
    """Match the SDK's Python packet boundary without loading its native DLL."""

    native_reads = {
        "state", "batt_power", "version", "supported_channels",
        "sampling_frequency", "is_supported_command", "is_supported_parameter",
        "exec_command", "disconnect",
    }

    def __init__(self):
        self.in_callback = False
        self.callback_native_accesses = []
        self.command_log = []
        self.support_checks = []
        self.unsupported = set()
        self.fail_commands = set()
        self.state = SensorState.StateInRange
        self.batt_power = 80
        self.version = SimpleNamespace(FwMajor=1, FwMinor=2, FwPatch=3)
        self.supported_channels = [
            SimpleNamespace(Num=3, Name="T4"),
            SimpleNamespace(Num=0, Name="O1"),
            SimpleNamespace(Num=2, Name="T3"),
            SimpleNamespace(Num=1, Name="O2"),
        ]
        self.sampling_frequency = SamplingFrequency.FrequencyHz250
        self.sensorStateChanged = None
        self.batteryChanged = None
        self.signalDataReceived = None
        self.resistDataReceived = None
        self.disconnect_calls = 0

    def __getattribute__(self, name):
        if name in {"commands", "sampling_frequency_resist"}:
            object.__getattribute__(self, "callback_native_accesses").append(name)
            raise AssertionError(f"Unsafe SDK accessor: {name}")
        if name in FakeSensor.native_reads and object.__getattribute__(self, "in_callback"):
            object.__getattribute__(self, "callback_native_accesses").append(name)
            raise AssertionError(f"Native SDK access from callback: {name}")
        return object.__getattribute__(self, name)

    def is_supported_parameter(self, _parameter):
        return True

    def is_supported_command(self, command):
        self.support_checks.append(command)
        return command not in self.unsupported

    def exec_command(self, command):
        self.command_log.append(command)
        if command in self.fail_commands:
            raise RuntimeError("private SDK failure with private-device-serial")

    def disconnect(self):
        self.disconnect_calls += 1
        self.state = SensorState.StateOutOfRange

    def deliver(self, mode, packets, callback=None):
        callback = callback or getattr(
            self, "signalDataReceived" if mode == "signal" else "resistDataReceived"
        )
        assert callable(callback)
        self.in_callback = True
        try:
            callback(self, packets)
        finally:
            self.in_callback = False


def packet(counter=17, samples=None, marker=5, mode="signal"):
    values = {"PackNum": counter, "Samples": list(samples if samples is not None else [4e-6, 1e-6, 3e-6, 2e-6])}
    if mode == "signal":
        values["Marker"] = marker
    else:
        values["Referents"] = [1234.0, 5678.0]
    return SimpleNamespace(**values)


@pytest.fixture
def harness(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "neurosdk" or name.startswith("neurosdk."):
            raise AssertionError("Acquisition tests must never load the real SDK")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    worker = importlib.import_module("plugins.brainbit_worker")
    sensor = FakeSensor()
    frames = []
    statuses = []

    class FakeScanner:
        def __init__(self, families):
            assert families == [SensorFamily.LEBrainBit2]

        def start(self):
            pass

        def stop(self):
            pass

        def sensors(self):
            return [SimpleNamespace(Name="BrainBit 2 test", SensFamily=SensorFamily.LEBrainBit2)]

        def create_sensor(self, _info):
            return sensor

    def emit_frame(frame):
        assert not sensor.in_callback, "I/O emission must happen outside SDK callbacks"
        frames.append(frame)

    def emit_status(status):
        assert not sensor.in_callback, "Metadata I/O must happen outside SDK callbacks"
        statuses.append(status)

    session = worker.SDKSession(
        scanner_class=FakeScanner,
        family_enum=SensorFamily,
        state_enum=SensorState,
        parameter_enum=SensorParameter,
        command_enum=SensorCommand,
        scan_seconds=0,
        emit=emit_status,
        emit_acquisition=emit_frame,
        start_pump=False,
    )

    def connect():
        device = session.handle("discover")["devices"][0]
        return session.handle("connect", device_id=device["id"])

    def start(mode="signal", session_id=None):
        return session.handle("start_acquisition", mode=mode, session_id=session_id or str(uuid4()))

    yield SimpleNamespace(
        session=session, sensor=sensor, frames=frames, statuses=statuses,
        connect=connect, start=start,
    )
    sensor.fail_commands.clear()
    if session.acquisition_snapshot()["state"] == "error":
        # An unconfirmed native stop makes the disposable process unsafe to
        # reuse. Its caller terminates it instead of retrying native operations.
        with pytest.raises(RuntimeError):
            session.close()
    else:
        session.close()
    assert sensor.callback_native_accesses == []


def test_connect_and_status_do_not_start_acquisition(harness):
    harness.connect()
    harness.session.handle("status")
    harness.session._pump_once()

    assert harness.sensor.command_log == []
    assert harness.frames == []
    assert harness.session.acquisition_snapshot()["state"] == "stopped"


@pytest.mark.parametrize("mode,start_command,stop_command,units", [
    ("signal", SensorCommand.StartSignal, SensorCommand.StopSignal, "V"),
    ("contact", SensorCommand.StartResist, SensorCommand.StopResist, "ohm"),
])
def test_start_reports_sorted_channels_true_frequency_and_mode_units(
    harness, mode, start_command, stop_command, units
):
    harness.connect()
    session_id = str(uuid4())

    result = harness.start(mode, session_id)

    assert result["state"] == "running"
    assert result["mode"] == mode
    assert result["session_id"] == session_id
    assert result["channels"] == [
        {"num": 0, "name": "O1"}, {"num": 1, "name": "O2"},
        {"num": 2, "name": "T3"}, {"num": 3, "name": "T4"},
    ]
    assert result["nominal_hz"] == 250
    assert result["units"] == units
    assert isinstance(result["timing"], str) and result["timing"]
    assert harness.sensor.command_log == [start_command]
    assert {start_command, stop_command}.issubset(harness.sensor.support_checks)


@pytest.mark.parametrize("mode,missing", [
    ("signal", SensorCommand.StartSignal), ("signal", SensorCommand.StopSignal),
    ("contact", SensorCommand.StartResist), ("contact", SensorCommand.StopResist),
])
def test_start_requires_both_start_and_stop_support(harness, mode, missing):
    harness.connect()
    harness.sensor.unsupported.add(missing)

    with pytest.raises((ValueError, RuntimeError)):
        harness.start(mode)

    assert harness.sensor.command_log == []
    assert harness.session.acquisition_snapshot()["state"] != "running"


def test_start_requires_connection(harness):
    with pytest.raises((ValueError, RuntimeError)):
        harness.start()
    assert harness.sensor.command_log == []


@pytest.mark.parametrize("mode", ["signal", "contact"])
def test_callback_only_enqueues_then_pump_maps_samples_and_emits(harness, mode):
    harness.connect()
    started = harness.start(mode)
    harness.frames.clear()
    samples = [4000.0, 1000.0, 3000.0, 2000.0] if mode == "contact" else [4e-6, 1e-6, 3e-6, 2e-6]

    harness.sensor.deliver(mode, [packet(17, samples, mode=mode)])

    assert harness.frames == []
    assert harness.session._packet_queue.qsize() == 1
    harness.session._pump_once()
    assert len(harness.frames) == 1
    frame = harness.frames[0]
    assert frame["session_id"] == started["session_id"]
    assert frame["mode"] == mode
    assert frame["channels"] == started["channels"]
    assert frame["nominal_hz"] == 250
    assert len(frame["packets"]) == 1
    sample = frame["packets"][0]
    assert sample["counter"] == 17
    # Samples is indexed by channel Num, not by supported_channels list order.
    assert sample["samples"] == samples
    if mode == "signal":
        assert sample["marker"] == 5
    for key in ("host_received_monotonic",):
        assert isinstance(sample[key], (int, float))
        assert math.isfinite(sample[key])
    if mode == "signal":
        assert math.isfinite(sample["estimated_monotonic"])
    else:
        assert sample["estimated_monotonic"] is None
    json.dumps(frame, allow_nan=False)


def test_public_status_never_contains_acquisition_metadata_or_raw_samples(harness):
    harness.connect()
    harness.start()
    harness.sensor.deliver("signal", [packet()])
    harness.session._pump_once()

    for status in [harness.session.snapshot(), harness.session.handle("status"), *harness.statuses]:
        assert set(status).isdisjoint({"acquisition", "packets", "samples", "channels", "session_id"})
        serialized = json.dumps(status)
        assert "host_received_monotonic" not in serialized
        assert "estimated_monotonic" not in serialized


def test_mode_switch_stops_old_command_and_discards_old_session_packets(harness):
    harness.connect()
    first = harness.start("signal")
    old_callback = harness.sensor.signalDataReceived
    harness.sensor.deliver("signal", [packet(20)])

    second = harness.start("contact")

    assert second["session_id"] != first["session_id"]
    assert harness.sensor.command_log == [
        SensorCommand.StartSignal, SensorCommand.StopSignal, SensorCommand.StartResist,
    ]
    harness.frames.clear()
    harness.sensor.deliver("signal", [packet(21)], callback=old_callback)
    harness.sensor.deliver("contact", [packet(22, [40, 10, 30, 20], mode="contact")])
    harness.session._pump_once()
    packets = [item for frame in harness.frames for item in frame["packets"]]
    assert [item["counter"] for item in packets] == [22]
    assert all(frame["session_id"] == second["session_id"] for frame in harness.frames)


def test_stop_is_repeatable_clears_queue_and_ignores_late_callbacks(harness):
    harness.connect()
    harness.start()
    old_callback = harness.sensor.signalDataReceived
    harness.sensor.deliver("signal", [packet()])

    stopped = harness.session.handle("stop_acquisition")
    again = harness.session.handle("stop_acquisition")

    assert stopped["state"] == "stopped"
    assert again["state"] == "stopped"
    assert stopped["session_id"] is None
    assert stopped["mode"] is None
    assert harness.sensor.command_log == [SensorCommand.StartSignal, SensorCommand.StopSignal]
    assert harness.session._packet_queue.empty()
    harness.frames.clear()
    harness.sensor.deliver("signal", [packet()], callback=old_callback)
    harness.session._pump_once()
    assert harness.frames == []


def test_stop_failure_invalidates_session_and_clears_buffer(harness):
    harness.connect()
    harness.start()
    old_callback = harness.sensor.signalDataReceived
    harness.sensor.deliver("signal", [packet()])
    harness.sensor.fail_commands.add(SensorCommand.StopSignal)

    with pytest.raises((ValueError, RuntimeError)):
        harness.session.handle("stop_acquisition")

    assert harness.session._packet_queue.empty()
    status = harness.session.acquisition_snapshot()
    assert status["state"] == "error"
    assert "private-device-serial" not in json.dumps(status)
    harness.frames.clear()
    harness.sensor.deliver("signal", [packet()], callback=old_callback)
    harness.session._pump_once()
    assert harness.frames == []


def test_nonfinite_and_wrong_length_samples_cannot_reach_invalid_json(harness):
    harness.connect()
    harness.start()
    harness.sensor.deliver("signal", [
        packet(1, [1e-6, float("nan"), float("inf"), -float("inf")]),
        packet(2, [1e-6]),
        packet(3),
    ])
    harness.session._pump_once()

    assert harness.frames
    for frame in harness.frames:
        json.dumps(frame, allow_nan=False)
        assert frame["nonfinite"] >= 3
        assert frame["channel_mismatches"] >= 1
        for sample in frame["packets"]:
            assert len(sample["samples"]) == 4
            assert all(value is None or math.isfinite(value) for value in sample["samples"])


def test_backpressure_bounds_queue_and_each_frame(harness):
    harness.connect()
    harness.start()
    harness.sensor.deliver("signal", [packet(index) for index in range(1200)])

    assert harness.session._packet_queue.maxsize == 500
    assert harness.session._packet_queue.qsize() <= 500
    for _ in range(6):
        harness.session._pump_once()

    assert harness.session._packet_queue.empty()
    assert harness.frames
    assert all(1 <= len(frame["packets"]) <= 100 for frame in harness.frames)
    assert sum(len(frame["packets"]) for frame in harness.frames) <= 500
    assert harness.frames[-1]["queue_drops"] >= 700
    for frame in harness.frames:
        json.dumps(frame, allow_nan=False)


def test_disconnect_stops_acquisition_and_detaches_data_callbacks(harness):
    harness.connect()
    harness.start("contact")
    harness.sensor.deliver("contact", [packet(mode="contact")])

    result = harness.session.handle("disconnect")

    assert result["state"] == "disconnected"
    assert harness.sensor.command_log == [SensorCommand.StartResist, SensorCommand.StopResist]
    assert harness.sensor.signalDataReceived is None
    assert harness.sensor.resistDataReceived is None
    assert harness.sensor.disconnect_calls == 1
    assert harness.session._packet_queue.empty()


def test_connection_loss_callback_invalidates_queue_without_native_reads_or_io(harness):
    harness.connect()
    harness.start()
    old_callback = harness.sensor.signalDataReceived
    harness.sensor.deliver("signal", [packet()])
    harness.frames.clear()
    harness.statuses.clear()
    harness.sensor.state = SensorState.StateOutOfRange
    harness.sensor.in_callback = True
    try:
        harness.sensor.sensorStateChanged(harness.sensor, SensorState.StateOutOfRange)
    finally:
        harness.sensor.in_callback = False

    assert harness.frames == []
    assert harness.statuses == []
    assert harness.sensor.command_log == [SensorCommand.StartSignal]
    assert harness.session._packet_queue.empty()
    assert harness.session.acquisition_snapshot()["state"] == "error"
    assert harness.session.snapshot()["state"] == "disconnected"
    harness.sensor.deliver("signal", [packet()], callback=old_callback)
    harness.session._pump_once()
    assert harness.frames == []
    assert harness.statuses[-1]["state"] == "disconnected"


def test_battery_callback_defers_metadata_io_until_pump(harness):
    harness.connect()
    harness.statuses.clear()
    harness.sensor.in_callback = True
    try:
        harness.sensor.batteryChanged(harness.sensor, 31)
    finally:
        harness.sensor.in_callback = False

    assert harness.statuses == []
    assert harness.session.snapshot()["device"]["battery"] == 31
    harness.session._pump_once()
    assert harness.statuses[-1]["device"]["battery"] == 31
    assert harness.sensor.command_log == []
