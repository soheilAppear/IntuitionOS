import platform
import subprocess

import psutil

from .hw_base import HardwareDriver


def _temp_windows() -> float | None:
    """Read thermal zone temperature via Windows WMI (no extra drivers needed)."""
    try:
        ps = (
            "(Get-WmiObject -Namespace root/wmi "
            "-Class MSAcpi_ThermalZoneTemperature "
            "-ErrorAction SilentlyContinue).CurrentTemperature"
        )
        r = subprocess.run(
            ["powershell", "-NonInteractive", "-Command", ps],
            capture_output=True, text=True, timeout=6
        )
        if r.returncode == 0 and r.stdout.strip():
            vals = [float(v) for v in r.stdout.split() if v.strip().lstrip("-").isdigit() or v.strip().replace(".", "", 1).isdigit()]
            if vals:
                # WMI gives units of 1/10 Kelvin
                temps_c = [(v / 10) - 273.15 for v in vals if v > 0]
                if temps_c:
                    return round(sum(temps_c) / len(temps_c), 1)
    except Exception:
        pass
    return None


class CPUInfo(HardwareDriver):
    name = "cpu_info"

    def schema(self):
        return {"actions": [{"name": "status", "args": []}]}

    def call(self, action: str, **kwargs):
        if action != "status":
            return {"error": "only status is supported"}

        try:
            usage = psutil.cpu_percent(interval=0.5)
            freq = psutil.cpu_freq()
            physical = psutil.cpu_count(logical=False)
            logical = psutil.cpu_count(logical=True)

            result = {
                "cpu_usage": f"{usage}%",
                "cores": f"{physical} physical / {logical} logical",
            }

            if freq:
                result["frequency"] = f"{round(freq.current)} MHz  (max {round(freq.max)} MHz)"

            # Temperature — try psutil first (Linux/macOS), then Windows WMI fallback
            temp = None
            try:
                sensors = psutil.sensors_temperatures()
                if sensors:
                    for key in ("coretemp", "k10temp", "cpu_thermal", "cpu-thermal", "acpitz"):
                        if key in sensors:
                            readings = sensors[key]
                            temp = round(sum(e.current for e in readings) / len(readings), 1)
                            break
            except AttributeError:
                pass  # Windows — psutil doesn't implement this

            if temp is None and platform.system() == "Windows":
                temp = _temp_windows()

            if temp is not None:
                result["temperature"] = f"{temp} °C"
            else:
                result["temperature"] = "unavailable (install LibreHardwareMonitor for Windows sensor access)"

            return result

        except Exception as e:
            return {"error": str(e)}
