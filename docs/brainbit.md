# BrainBit connection and status

The HUD's **BrainBit** panel can discover nearby devices, connect to a selected
headset, display its battery and firmware, and disconnect. It does not start EEG
streams. The separate, explicitly started [experimental combined preview](#experimental-eeg--camera-preview)
can display live EEG and webcam frames. Neither panel records signals or updates firmware.

## Setup

Use the same Python environment as the IntuitionOS backend. On Windows 10 or
newer, the tested runtime is 64-bit Python 3.10.11 with `pyneurosdk2==1.0.15`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-brainbit.txt
.\.venv\Scripts\python.exe start_ui.py
```

The dependency is optional. Missing or unloadable SDK libraries produce an
availability message; they do not prevent the rest of IntuitionOS from starting.
The native DLL runs in a separate worker, never inside the backend process.

`config/config.yaml` enables this adapter under `hardware.drivers`:

```yaml
- name: brainbit
  enabled: true
```

Set `enabled: false` to disable it. Restart the backend after changing hardware
configuration or installing this feature into an already running checkout.
Starting IntuitionOS and opening the panel do not scan or connect automatically.

## Use the panel

1. Keep the charged headset powered on and nearby, and make Windows Bluetooth
   available. Close another application's connection to the headset if needed.
2. Expand **BrainBit** below Hand controls and choose **Discover devices**.
3. Select the intended entry, then choose **Connect**.
4. While connected, battery and firmware metadata update automatically.
   **Refresh status** requests a fresh read. Unsupported metadata says Unknown.
5. Choose **Disconnect** when finished. During discovery or connection, the
   same control can cancel the pending operation. Discover again before a new
   connection because selections belong to one worker session.

The SDK uses Windows Bluetooth LE. It does not expose which Bluetooth radio or
USB dongle carries the connection, so a connected dongle alone does not establish
that it was used. Device family `LEBrainBit2` is supported, along with the other
BrainBit families exposed by this SDK version.

A failed or timed-out operation releases the worker and offers discovery again.
The app does not reconnect automatically. If the backend goes offline, the HUD
marks device state unconfirmed and disables controls until fresh status arrives.
Closing or restarting the backend closes its owned worker and connection.

## Implementation and tests

`plugins/brainbit.py` implements `HardwareDriver` and owns bounded JSON IPC,
cached status, cancellation, and a revision counter. `brainbit_worker.py` alone
loads the SDK and owns native scanner/sensor handles. Public hardware actions are limited to
discovery, connection, metadata, and disconnect. Device addresses and serial
numbers are not returned to the HUD or action journal; selection IDs are opaque
and expire when the worker closes.

Both interfaces register the adapter through their existing hardware registry.
HUD user operations use the existing `hw_call` capability gate. The panel's
native HTTP requests use `/brainbit/status` and explicit POST actions; these
routes reject browser Origins and non-loopback Hosts. `/health` remains a cheap
backend responsiveness check. Connected-device metadata refresh runs off the
event loop; a blocked SDK operation cannot block cached status or cancellation.

Tests use fake SDK objects or harmless subprocesses rather than live devices:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_brainbit.py tests/test_brainbit_worker.py tests/test_brainbit_protocol.py tests/test_startup.py
node --test --test-isolation=none tests/brainbit_renderer_test.cjs tests/brainbit_http_test.cjs tests/hud_renderer_test.cjs
```

Live verification on 2026-10-06 found one BrainBit2, completed two API connection,
metadata and disconnect cycles, and a third cycle through the rendered Electron
panel. Battery read 89% initially and 88% during the panel check; firmware was
1.11.0. No signal-start commands were sent. The test used temporary data on a
separate local backend, leaving the existing IntuitionOS session running.

Official references: [Python installation](https://sdk.brainbit.com/sdk2_python_install/),
[package](https://pypi.org/project/pyneurosdk2/),
[connection API](https://sdk.brainbit.com/sdk2_sensor/),
[BrainBit2](https://sdk.brainbit.com/brainbit-2-devices/).

## Experimental EEG + camera preview

This separate HUD panel is an exploratory movement experiment for **BrainBit2**.
It displays mirrored webcam hand tracking next to four EEG channels, using
approximate host arrival timing. It does not decode intentions: movement,
muscle activity and electrode motion can all affect the signal. EEG never
supplies the direction of a desktop action.

1. Connect BrainBit in the connection panel. Wear it according to its manual;
   keep it powered on and disconnect the charger if the device manual requires it.
2. Turn off the normal Hand controls camera. Expand **EEG + camera preview**.
3. Optionally choose **Check contacts** while preview is stopped. This acquires
   resistance for up to five seconds, then stops. Contact and signal acquisition
   are separate modes. Readings are historical and show their age, with no
   validated good/bad cutoff. The versioned Python SDK defines these values as
   ohms; its web documentation has conflicting units, noted in the panel.
4. Choose **Start preview**. Opening the panel never starts either sensor.
   Keep one hand visible and still briefly, then move it horizontally left or
   right. Directions refer to the mirrored camera view. Return to a neutral
   position between movements. EEG is shown in microvolts with descriptive
   age, rate, counter-discontinuity, nonfinite-value and queue-drop diagnostics.
5. For an exploratory calibration, choose a left or right trial and perform
   that movement within three seconds. Each accepted trial needs matching
   camera movement and fresh EEG. Features stay in memory, up to 64 trials;
   raw signal windows are not retained in calibration. Collect balanced trials
   across both directions. Evaluation uses a chronological held-out set and
   needs at least six training and three held-out trials per class. Accuracy
   and the baseline are descriptive movement/artifact correlations, not proof
   of neural intent or generalization. Calibration never enables control.
6. Desktop control is **off by default**. Only if desired, explicitly enable
   **Arm webcam swipes**. Fresh webcam data determines left/right, while fresh
   EEG is a required availability gate. A neutral reset, debounce and cooldown
   limit repeated actions. Lost tracking, stale data or device failure disarms
   the session. Safe Mode and the existing desktop capability gate still apply.
7. Choose **Stop & disarm** when finished. It cancels startup/contact checking,
   releases the preview camera and stops EEG. If native shutdown cannot be
   confirmed, the panel reports the error and the worker is terminated.

Frames and waveform samples are transient in memory and pass only between the
local backend and HUD. There is no recording, export, cloud upload, model input,
or raw-signal action logging. The waveform cache is bounded to five seconds and
at most 250 display points per response; it is unsuitable for spectral analysis.
Hiding the panel clears its displayed frames and pauses raw-data HTTP polling;
acquisition continues until Stop, a device failure, or backend shutdown. Stop
before leaving the experiment. Restarting the backend clears calibration.

The private `/multimodal/preview` route carries raw display data. Other preview
status/actions contain metadata only, and acquisition methods are excluded from
the public hardware action schema. Browser-origin and non-loopback requests are
rejected; this is not an authentication boundary against other local processes.

Automated verification uses synthetic signals, fake sensors and fake camera
frames. Live EEG acquisition, contact measurement and user-specific calibration
require an explicit user-started hardware trial; passing the software tests does
not establish working hardware synchronization or EEG intention decoding.
