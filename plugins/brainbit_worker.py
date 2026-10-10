"""Private SDK worker with explicit, bounded BrainBit2 preview acquisition.

Run by ``BrainBit`` in a disposable process. No native SDK import happens when
IntuitionOS imports the hardware driver, and SDK handles never cross IPC.
"""

import copy
import json
import math
import os
import queue
import re
import sys
import threading
import time
import uuid


TRANSPORT = "Windows Bluetooth LE; SDK does not identify radio/dongle"
FAMILIES = ("LEBrainBit", "LEBrainBitBlack", "LEBrainBit2", "LEBrainBitPro", "LEBrainBitFlex")
MAX_FRAME = 65536
MAX_CHANNELS = 16
MAX_PACKETS = 100
QUEUE_PACKETS = 500
ACQUISITION_FAMILIES = {"LEBrainBit2", "LEBrainBitPro", "LEBrainBitFlex"}
TIMING = "host receive time; signal sample times estimated from callback order and nominal rate"


class SDKSession:
    """Own native handles; callbacks enqueue data and never write to IPC."""

    def __init__(self, scanner_class=None, family_enum=None, state_enum=None,
                 scan_seconds=5, emit=None, parameter_enum=None,
                 command_enum=None, emit_acquisition=None, start_pump=True):
        if scanner_class is None:
            from neurosdk.scanner import Scanner
            from neurosdk.cmn_types import SensorFamily, SensorState, SensorParameter, SensorCommand
            scanner_class, family_enum, state_enum = Scanner, SensorFamily, SensorState
            parameter_enum = SensorParameter
            command_enum = SensorCommand
        self._scanner_class = scanner_class
        self._families = [getattr(family_enum, name) for name in FAMILIES
                          if hasattr(family_enum, name)]
        self._in_range = state_enum.StateInRange
        self._battery_parameter = parameter_enum.BattPower if parameter_enum else None
        self._scan_seconds = min(15.0, max(0.0, float(scan_seconds)))
        self._emit = emit or (lambda _snapshot: None)
        self._emit_acquisition = emit_acquisition or (lambda _frame: None)
        self._command_enum = command_enum
        self._lock = threading.RLock()
        self._scanner = None
        self._sensor = None
        self._infos = {}
        self._selected = None
        self._packet_queue = queue.Queue(maxsize=QUEUE_PACKETS)
        self._metadata_queue = queue.Queue(maxsize=1)
        self._pump_stop = threading.Event()
        self._pump_thread = None
        self._start_pump = start_pump
        self._active_token = None
        self._stop_command = None
        self._acquisition_callback = None
        self._queue_drops = self._nonfinite = self._channel_mismatches = 0
        self._fatal_acquisition = False
        self._output_failed = False
        self._acquisition = {
            "state": "stopped", "mode": None, "session_id": None,
            "channels": [], "nominal_hz": 250, "units": None,
            "timing": TIMING,
        }
        self._snapshot = {
            "state": "disconnected", "available": True, "busy": False,
            "text": "Ready to discover BrainBit devices", "devices": [],
            "device": None, "transport": TRANSPORT,
        }

    def snapshot(self):
        with self._lock:
            return copy.deepcopy(self._snapshot)

    def _update(self, **values):
        with self._lock:
            self._snapshot.update(values)

    def handle(self, action, **kwargs):
        if self._output_failed:
            raise RuntimeError("BrainBit preview output failed; reconnect the device")
        if action == "start_acquisition":
            return self._start_acquisition(kwargs.get("mode"), kwargs.get("session_id"))
        if action == "stop_acquisition":
            return self._stop_acquisition()
        if action == "discover":
            self._discover()
        elif action == "connect":
            self._connect(kwargs.get("device_id"))
        elif action == "disconnect":
            self.close()
        elif action == "status":
            self._refresh()
        else:
            raise ValueError("Unsupported BrainBit action")
        return self.snapshot()

    def _discover(self):
        if self._sensor is not None:
            raise ValueError("Disconnect before discovering devices")
        if self._scanner is None:
            self._scanner = self._scanner_class(self._families)
        self._infos.clear()
        self._update(devices=[])
        try:
            self._scanner.start()
            time.sleep(self._scan_seconds)
            infos = self._scanner.sensors()
        finally:
            self._scanner.stop()
        devices = []
        for info in infos[:64]:
            if info.SensFamily not in self._families:
                continue
            key = str(uuid.uuid4())
            self._infos[key] = info
            devices.append({"id": key, "name": str(info.Name or "BrainBit")[:128],
                            "family": info.SensFamily.name})
        self._update(state="disconnected", device=None, devices=devices,
                     text=f"Found {len(devices)} BrainBit device(s)" if devices
                     else "No BrainBit devices found; check power and Bluetooth")

    def _connect(self, device_id):
        if self._sensor is not None:
            if device_id == self._selected:
                self._refresh()
                return
            raise ValueError("Disconnect before selecting another device")
        info = self._infos.get(device_id)
        if info is None:
            raise ValueError("Device selection expired; discover again")
        # create_sensor already connects. Calling connect() again is redundant.
        sensor = self._scanner.create_sensor(info)
        self._sensor = sensor
        self._selected = device_id
        sensor.sensorStateChanged = self._state_changed
        sensor.batteryChanged = self._battery_changed
        self._update(device={"name": str(info.Name or "BrainBit")[:128],
                             "family": info.SensFamily.name,
                             "battery": None, "firmware": None})
        self._refresh()
        self._ensure_pump()

    def acquisition_snapshot(self):
        """Acquisition control metadata only; raw packets are never cached here."""
        with self._lock:
            return copy.deepcopy(self._acquisition)

    def _ensure_pump(self):
        if not self._start_pump or (self._pump_thread and self._pump_thread.is_alive()):
            return
        self._pump_stop.clear()
        self._pump_thread = threading.Thread(target=self._pump, name="brainbit-output", daemon=True)
        self._pump_thread.start()

    def _pump(self):
        while not self._pump_stop.wait(0.1):
            try:
                self._pump_once()
            except Exception:
                # IPC failures must not propagate into a native callback.
                self._output_failed = True
                if self._stop_command is not None:
                    self._acquisition_error("Preview output failed; acquisition state is unconfirmed")
                self._pump_stop.set()

    def _pump_once(self):
        """Drain bounded Python queues; this path never touches the SDK."""
        try:
            metadata = self._metadata_queue.get_nowait()
        except queue.Empty:
            metadata = None
        if metadata is not None:
            self._emit(metadata)
        token = self._active_token
        if token is None:
            return
        packets = []
        for _ in range(MAX_PACKETS):
            try:
                queued_token, packet = self._packet_queue.get_nowait()
            except queue.Empty:
                break
            if queued_token is token:
                packets.append(packet)
        if not packets or token is not self._active_token:
            return
        metadata = self.acquisition_snapshot()
        payload = {
            "session_id": metadata["session_id"], "mode": metadata["mode"],
            "channels": metadata["channels"], "nominal_hz": metadata["nominal_hz"],
            "units": metadata["units"],
            "packets": packets, "queue_drops": self._queue_drops,
            "nonfinite": self._nonfinite, "channel_mismatches": self._channel_mismatches,
        }
        # Worst-case float encodings and escaped channel names also fit IPC.
        while len(json.dumps({"acquisition": payload}, allow_nan=False, ensure_ascii=True)) >= MAX_FRAME:
            payload["packets"].pop()
            self._queue_drops += 1
            payload["queue_drops"] = self._queue_drops
        if token is self._active_token and payload["packets"]:
            self._emit_acquisition(payload)

    def _queue_metadata(self):
        metadata = self.snapshot()
        try:
            self._metadata_queue.put_nowait(metadata)
        except queue.Full:
            try:
                self._metadata_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._metadata_queue.put_nowait(metadata)
            except queue.Full:
                pass

    def _clear_packets(self):
        previous_queue = self._packet_queue
        self._packet_queue = queue.Queue(maxsize=QUEUE_PACKETS)
        while True:
            try:
                previous_queue.get_nowait()
            except queue.Empty:
                return

    def _acquisition_error(self, message):
        self._active_token = None
        self._clear_packets()
        self._fatal_acquisition = True
        with self._lock:
            self._acquisition.update(state="error", error=message, stop_confirmed=False)

    def _start_acquisition(self, mode, session_id):
        if mode not in {"signal", "contact"}:
            raise ValueError("Unsupported acquisition mode")
        try:
            session_id = str(uuid.UUID(str(session_id)))
        except (ValueError, TypeError, AttributeError):
            raise ValueError("A valid acquisition session is required") from None
        if self._fatal_acquisition:
            raise RuntimeError("Acquisition state is unconfirmed; reconnect the device")
        sensor = self._sensor
        if sensor is None or self.snapshot()["state"] != "connected":
            raise ValueError("Connect a BrainBit2 device before acquisition")
        info = self._infos[self._selected]
        if info.SensFamily.name not in ACQUISITION_FAMILIES:
            raise ValueError("Preview acquisition requires a BrainBit2 family device")
        suffix = "Signal" if mode == "signal" else "Resist"
        start = getattr(self._command_enum, "Start" + suffix, None)
        stop = getattr(self._command_enum, "Stop" + suffix, None)
        if start is None or stop is None:
            raise ValueError("Acquisition commands are unavailable")
        # Never use sensor.commands: SDK 1.0.15's bulk accessor is unsafe.
        supported_start = sensor.is_supported_command(start)
        supported_stop = sensor.is_supported_command(stop)
        if not supported_start or not supported_stop:
            raise ValueError("The device does not support this acquisition start/stop pair")
        if self._stop_command is not None:
            self._stop_acquisition()
        reported = sensor.supported_channels
        if not 1 <= len(reported) <= MAX_CHANNELS:
            raise ValueError("Unsupported acquisition channel count")
        channels = []
        for channel in reported:
            number = channel.Num
            if type(number) is not int or not 0 <= number < MAX_CHANNELS:
                raise ValueError("Unsupported acquisition channel number")
            channels.append({"num": number, "name": str(channel.Name)[:32]})
        channels.sort(key=lambda channel: channel["num"])
        if len({channel["num"] for channel in channels}) != len(channels):
            raise ValueError("Duplicate acquisition channel numbers")
        frequency = sensor.sampling_frequency
        match = re.fullmatch(r"FrequencyHz(\d+)", getattr(frequency, "name", ""))
        if match is None or int(match.group(1)) != 250:
            raise ValueError("Preview requires the nominal 250 Hz signal rate")
        # sampling_frequency_resist is deliberately never accessed. Contact
        # packets have host receive timestamps and no estimated sample time.
        self._clear_packets()
        self._queue_drops = self._nonfinite = self._channel_mismatches = 0
        token = object()
        self._active_token = token
        self._stop_command = stop
        self._acquisition_callback = "signalDataReceived" if mode == "signal" else "resistDataReceived"
        with self._lock:
            self._acquisition = {
                "state": "running", "mode": mode, "session_id": session_id,
                "channels": channels, "nominal_hz": 250,
                "units": "V" if mode == "signal" else "ohm", "timing": TIMING,
            }
        callback = lambda source, packets: self._receive_packets(source, packets, token, mode, channels)
        try:
            setattr(sensor, self._acquisition_callback, callback)
            self._ensure_pump()
            sensor.exec_command(start)
        except Exception:
            self._acquisition_error("Acquisition start failed; device state is unconfirmed")
            raise RuntimeError("Acquisition start failed; reconnect the device") from None
        return self.acquisition_snapshot()

    def _receive_packets(self, source, packets, token, mode, channels):
        """Copy already-materialized SDK packet values; no native calls or I/O."""
        if source is not self._sensor or token is not self._active_token:
            return
        if not isinstance(packets, (list, tuple)):
            self._channel_mismatches += 1
            return
        received = time.monotonic()
        packet_queue = self._packet_queue
        count = len(packets)
        offset = max(0, count - QUEUE_PACKETS)
        self._queue_drops += offset
        expected_samples = channels[-1]["num"] + 1
        for index in range(offset, count):
            if token is not self._active_token:
                return
            packet = packets[index]
            counter = getattr(packet, "PackNum", None)
            if type(counter) is not int or not 0 <= counter <= 0xFFFFFFFF:
                self._channel_mismatches += 1
                continue
            values = getattr(packet, "Samples", ())
            if not isinstance(values, (list, tuple)):
                values = ()
            if len(values) != expected_samples:
                self._channel_mismatches += 1
            samples = []
            for channel in channels:
                position = channel["num"]
                value = values[position] if position < len(values) else None
                if value is None:
                    samples.append(None)
                    continue
                try:
                    value = float(value)
                    finite = math.isfinite(value)
                except (ValueError, TypeError, OverflowError):
                    finite = False
                if not finite:
                    self._nonfinite += 1
                samples.append(value if finite else None)
            marker = getattr(packet, "Marker", None) if mode == "signal" else None
            if type(marker) is not int or not 0 <= marker <= 255:
                marker = None
            converted = {
                "counter": counter, "marker": marker, "samples": samples,
                "host_received_monotonic": received,
                "estimated_monotonic": received - (count - 1 - index) / 250 if mode == "signal" else None,
            }
            try:
                packet_queue.put_nowait((token, converted))
            except queue.Full:
                self._queue_drops += 1
                try:
                    packet_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    packet_queue.put_nowait((token, converted))
                except queue.Full:
                    self._queue_drops += 1

    def _stop_acquisition(self):
        self._active_token = None
        self._clear_packets()
        if self._fatal_acquisition:
            raise RuntimeError("Acquisition stop is unconfirmed; reconnect the device")
        if self._stop_command is not None:
            try:
                self._sensor.exec_command(self._stop_command)
            except Exception:
                self._acquisition_error("Acquisition stop is unconfirmed; reconnect the device")
                raise RuntimeError("Acquisition stop is unconfirmed; reconnect the device") from None
            # Preserve callbacks until the native stop succeeds.
            setattr(self._sensor, self._acquisition_callback, None)
            self._stop_command = None
            self._acquisition_callback = None
        self._clear_packets()
        with self._lock:
            self._acquisition.update(state="stopped", mode=None, session_id=None)
        return self.acquisition_snapshot()

    def _refresh(self):
        sensor = self._sensor
        if sensor is None:
            return
        if sensor.state != self._in_range:
            if self._active_token is not None or self._stop_command is not None:
                self._acquisition_error("Connection lost; acquisition stop is unconfirmed")
            self._update(state="disconnected", device=None,
                         text="BrainBit disconnected; disconnect to reset, then discover again")
            return
        # Optional metadata failure must not turn an established link into a
        # connection failure. A native hang is still bounded by the parent.
        battery = None
        firmware = None
        try:
            supported = (self._battery_parameter is None or
                         sensor.is_supported_parameter(self._battery_parameter))
            value = sensor.batt_power if supported else None
            if type(value) is int and 0 <= value <= 100:
                battery = value
        except Exception:
            pass
        try:
            version = sensor.version
            firmware = f"{int(version.FwMajor)}.{int(version.FwMinor)}.{int(version.FwPatch)}"
        except Exception:
            pass
        info = self._infos[self._selected]
        self._update(state="connected", text="BrainBit connected",
                     device={"name": str(info.Name or "BrainBit")[:128],
                             "family": info.SensFamily.name,
                             "battery": battery, "firmware": firmware})

    def _state_changed(self, sensor, state):
        if sensor is not self._sensor:
            return
        # No SDK reads from SDK callback threads: these can deadlock the DLL.
        if state != self._in_range:
            if self._active_token is not None or self._stop_command is not None:
                self._acquisition_error("Connection lost; acquisition stop is unconfirmed")
            self._update(state="disconnected", device=None,
                         text="BrainBit connection lost; disconnect to reset, then discover again")
            self._queue_metadata()

    def _battery_changed(self, sensor, battery):
        if sensor is not self._sensor or type(battery) is not int or not 0 <= battery <= 100:
            return
        with self._lock:
            if self._snapshot["device"] is None:
                return
            self._snapshot["device"]["battery"] = battery
        self._queue_metadata()

    def close(self):
        if self._stop_command is not None:
            self._stop_acquisition()
        self._pump_stop.set()
        if self._pump_thread and self._pump_thread is not threading.current_thread():
            self._pump_thread.join(timeout=0.25)
        self._active_token = None
        self._clear_packets()
        sensor, self._sensor = self._sensor, None
        scanner, self._scanner = self._scanner, None
        self._selected = None
        self._infos.clear()
        try:
            if sensor is not None:
                sensor.sensorStateChanged = None
                sensor.batteryChanged = None
                sensor.disconnect()
        finally:
            try:
                if scanner is not None:
                    scanner.stop()
            finally:
                self._update(state="disconnected", devices=[], device=None,
                             text="BrainBit disconnected")
        # The parent exits this process after disconnect. SDK 1.0.15 scanner
        # destructor cleanup is unreliable, so GC is not our resource boundary.


def main():
    # Keep stdout exclusively for framed IPC, even if a native dependency writes
    # diagnostics to file descriptor 1. Never send raw SDK errors or identifiers.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), sys.stdout.fileno())
    write_lock = threading.Lock()

    def send(message):
        data = json.dumps(message, ensure_ascii=True, allow_nan=False)
        if len(data) > MAX_FRAME:
            raise ValueError("BrainBit response is too large")
        with write_lock:
            protocol.write(data + "\n")
            protocol.flush()

    session = None
    while True:
        line = sys.stdin.readline(MAX_FRAME + 1)
        if not line:
            break
        if len(line) > MAX_FRAME or not line.endswith("\n"):
            break
        request_id = None
        try:
            request = json.loads(line)
            request_id = request["id"]
            if session is None:
                try:
                    session = SDKSession(scan_seconds=request.get("scan_seconds", 5),
                                         emit=lambda state: send({"event": state}),
                                         emit_acquisition=lambda frame: send({"acquisition": frame}))
                except Exception:
                    send({"id": request_id, "unavailable": True,
                          "error": "BrainBit SDK could not load; install the optional pyneurosdk2 package for this Python runtime"})
                    break
            result = session.handle(request["action"], **request.get("args", {}))
            send({"id": request_id, "result": result})
            if request["action"] == "disconnect":
                break
        except Exception:
            error = "BrainBit operation failed; check the device, then discover again"
            if session is not None and session.acquisition_snapshot()["state"] == "error":
                error = "BrainBit acquisition failed; stop is unconfirmed; disconnect and reconnect"
            send({"id": request_id, "error": error})
            break
    protocol.close()
    # Finalizers can invoke the DLL and hang. Process teardown is the bounded
    # resource cleanup boundary; normal disconnect already ran above.
    os._exit(0)


if __name__ == "__main__":
    main()
