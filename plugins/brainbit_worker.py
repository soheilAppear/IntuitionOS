"""Private SDK worker. Only connection and device metadata are exposed.

Run by ``BrainBit`` in a disposable process. No native SDK import happens when
IntuitionOS imports the hardware driver, and SDK handles never cross IPC.
"""

import copy
import json
import os
import sys
import threading
import time
import uuid


TRANSPORT = "Windows Bluetooth LE; SDK does not identify radio/dongle"
FAMILIES = ("LEBrainBit", "LEBrainBitBlack", "LEBrainBit2", "LEBrainBitPro", "LEBrainBitFlex")
MAX_FRAME = 65536


class SDKSession:
    """Own one scanner and sensor; callbacks publish cached metadata only."""

    def __init__(self, scanner_class=None, family_enum=None, state_enum=None,
                 scan_seconds=5, emit=None, parameter_enum=None):
        if scanner_class is None:
            from neurosdk.scanner import Scanner
            from neurosdk.cmn_types import SensorFamily, SensorState, SensorParameter
            scanner_class, family_enum, state_enum = Scanner, SensorFamily, SensorState
            parameter_enum = SensorParameter
        self._scanner_class = scanner_class
        self._families = [getattr(family_enum, name) for name in FAMILIES
                          if hasattr(family_enum, name)]
        self._in_range = state_enum.StateInRange
        self._battery_parameter = parameter_enum.BattPower if parameter_enum else None
        self._scan_seconds = min(15.0, max(0.0, float(scan_seconds)))
        self._emit = emit or (lambda _snapshot: None)
        self._lock = threading.RLock()
        self._scanner = None
        self._sensor = None
        self._infos = {}
        self._selected = None
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

    def _refresh(self):
        sensor = self._sensor
        if sensor is None:
            return
        if sensor.state != self._in_range:
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
            self._update(state="disconnected", device=None,
                         text="BrainBit connection lost; disconnect to reset, then discover again")
            self._emit(self.snapshot())

    def _battery_changed(self, sensor, battery):
        if sensor is not self._sensor or type(battery) is not int or not 0 <= battery <= 100:
            return
        with self._lock:
            if self._snapshot["device"] is None:
                return
            self._snapshot["device"]["battery"] = battery
        self._emit(self.snapshot())

    def close(self):
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
        data = json.dumps(message, ensure_ascii=True)
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
                                         emit=lambda state: send({"event": state}))
                except Exception:
                    send({"id": request_id, "unavailable": True,
                          "error": "BrainBit SDK could not load; install the optional pyneurosdk2 package for this Python runtime"})
                    break
            result = session.handle(request["action"], **request.get("args", {}))
            send({"id": request_id, "result": result})
            if request["action"] == "disconnect":
                break
        except Exception:
            send({"id": request_id, "error": "BrainBit operation failed; check the device, then discover again"})
            break
    protocol.close()
    # Finalizers can invoke the DLL and hang. Process teardown is the bounded
    # resource cleanup boundary; normal disconnect already ran above.
    os._exit(0)


if __name__ == "__main__":
    main()
