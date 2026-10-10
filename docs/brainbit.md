# BrainBit connection, EEG signals, and status

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

### EEG signal status

The panel reports EEG signal status separately from the Bluetooth connection.
A connected headset with acquisition stopped is not a live signal. Start the
experimental preview explicitly to acquire samples; opening the BrainBit panel
only reads cached status.

The signal display distinguishes waiting for samples, receiving samples, stale
data, stopped acquisition, contact checking, and errors. It shows the channel
count, nominal and observed sample rates, last-sample age, and packet diagnostics
when available. Data older than 0.75 seconds is stale. A missed response also
expires the live indicator locally, so an old reading cannot remain live.
Contact-check age refers to the last separate contact measurement. These are
acquisition diagnostics, not a validated assessment of electrode contact or
medical signal quality.

The SDK uses Windows Bluetooth LE. A compatible Windows Bluetooth LE USB dongle
can supply the radio in place of the computer's built-in Bluetooth, but Windows must
recognize it as a working Bluetooth adapter. The dongle still communicates with
the headset over Bluetooth. The app cannot select or identify which radio carries
the connection, so plugging in a dongle alone does not establish that it was used.
If discovery fails, check the adapter's status in Windows Device Manager as well
as the headset connection. Device family `LEBrainBit2` is supported, along with
the other BrainBit families exposed by this SDK version.

A failed or timed-out operation releases the worker and offers discovery again.
The app does not reconnect automatically. If the backend goes offline, the HUD
marks device state unconfirmed and disables controls until fresh status arrives.
Closing or restarting the backend closes its owned worker and connection.

## View EEG without the camera

1. Connect the headset in **BrainBit** and turn off the normal Hand controls
   camera.
2. Expand **Experimental hand + EEG**. While stopped, optionally choose
   **Contact check (5 s)** and wait for it to finish.
3. Set **Control source → EEG classifier** before starting. This skips the
   webcam for this acquisition; the selector also controls which experimental
   classifier could later be armed.
4. Choose **Start preview**. Leave **Arm webcam swipes** and **Arm EEG swipes**
   unchecked to view signals without desktop actions.
5. Watch the channel traces and **EEG live** indicator. BrainBit2 supplies four
   channels at a nominal 250 Hz. The observed rate is estimated from host packet
   arrivals and can fluctuate around that value.
6. Choose **Stop & disarm (Esc)** when finished. Closing the panel alone does
   not stop acquisition.

Waveforms and signal status work before any training and do not require Ollama.
An untrained-model message refers to movement prediction, not signal acquisition.
To train later, use **Webcam direction** for the guided trials described below.

## Checking signal quality

Check transport and signal quality separately:

| Indicator | What it establishes |
|---|---|
| Connected | The SDK has a device connection; acquisition may still be stopped. |
| EEG live | Recent packets are arriving; this is not a clean-signal rating. |
| EEG stale | The newest confirmed sample is older than 0.75 seconds, or a live status update has expired locally. |
| Gaps, duplicates, nonfinite values, channel mismatches, queue drops | Cumulative acquisition diagnostics; compare their change over the observation period. Gaps count counter discontinuities, not the number of missing samples. |
| Channel RMS and peak-to-peak values | Descriptive amplitude statistics over the cached full-rate samples; RMS removes the mean but still includes slow drift. |
| Contact-check age | Time since a separate resistance measurement; it is not a continuous contact reading. |

Each channel trace uses its own automatic vertical scale. Compare the numeric
RMS and peak-to-peak values rather than assuming equal trace heights mean equal
signal amplitudes.

A stream can have no packet errors and still contain substantial movement,
electrode drift, or 50/60 Hz electrical interference. Sit still with a relaxed jaw
for a short comparison. If needed, stop preview, check contact, ensure electrodes
touch the scalp with hair moved aside, and restart. See the manufacturer's
[electrode-contact guidance](https://sdk.brainbit.com/device-recommendation/).
Investigate nearby power supplies and cables if an analysis shows a strong mains
frequency peak. A peak alone does not identify the interference source or prove
that a Bluetooth adapter is at fault.

The HUD does not automatically diagnose contact quality, measure mains-noise
power, or apply a 50/60 Hz notch filter. Large constant baseline offsets alone
do not fail the feature extractor's amplitude check; raw variation, flat signals,
and suspicious plateaus are checked separately. Passing those broad engineering
checks does not establish clean EEG or reliable movement classification.

**For signal analysis:** the waveform response contains at most 250 display
points across a cache of up to five seconds. It uses stride sampling without an
anti-alias filter, so do not treat the displayed points as a continuous 250 Hz
recording or use them directly for frequency-band or 50/60 Hz measurements.
Spectral analysis requires a verified contiguous full-rate sample sequence from
one acquisition session. The cached channel statistics and the internal EEG
feature extractor use the full-rate buffer. The app has no recording or file
export feature for raw EEG.

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
approximate host arrival timing. **Webcam direction** uses camera direction;
**EEG classifier** predicts left, right, or rest exclusively from EEG features.
Camera movement verifies training and validation labels, and never enters the
EEG classifier or supplies its action direction. Muscle activity and electrode
motion can explain successful classification: this is not thought reading or
a medical measurement, and software tests do not demonstrate reliable control.

1. Connect BrainBit in the connection panel. Wear it according to its manual;
   keep it powered on and disconnect the charger if the device manual requires it.
2. Turn off the normal Hand controls camera. Expand **Experimental hand + EEG**.
3. Optionally choose **Contact check (5 s)** while preview is stopped. This acquires
   resistance for up to five seconds, then stops. Contact and signal acquisition
   are separate modes. Readings are historical and show their age, with no
   validated good/bad cutoff. The versioned Python SDK defines these values as
   ohms; its web documentation has conflicting units, noted in the panel.
4. Choose **Start preview**. Opening the panel never starts either sensor.
   In Webcam mode this starts both sensors; do not also enable the normal Hand camera.
   A connected BrainBit alone does not prevent normal Hand camera use. GPU
   trackers such as WiLoR can take time to load: the combined preview allows
   up to 105 seconds for initial camera readiness and remains disarmed while
   waiting. Short frame or packet pauses disarm immediately; sustained loss
   stops acquisition. Stop remains available throughout startup.
   Keep one hand visible and still briefly, then move it horizontally left or
   right. Directions refer to the mirrored camera view. Return to a neutral
   position between movements. EEG is shown in microvolts with descriptive
   age, rate, counter-discontinuity, nonfinite-value and queue-drop diagnostics.
5. Keep Webcam mode selected for guided trials, with controls disarmed.
   Collect at least **eight training trials each for left, right, and rest**.
   Each trial lasts three seconds. Hold still for the first half-second, perform
   the requested horizontal movement during the middle two seconds, and hold
   still for the final half-second; remain still throughout rest trials. Each accepted trial needs
   fresh tracked camera data and a valid complete EEG window. Alternate labels
   and return to a neutral position between trials. No raw trial data is saved.
6. Choose **Train & freeze model**, then collect **eight new validation trials per
   class**. These use the frozen model and separate, nonoverlapping EEG samples.
   Training and validation cannot be mixed, and completed validation cannot be
   extended until a favorable score appears. Read the class precision/recall,
   confusion matrix, uncertain predictions, and rest false activations. The
   fixed engineering gates are described below. Failing a gate keeps EEG arming
   unavailable; reset starts a new experiment.
7. Select **EEG classifier** without stopping acquisition to retain the calibrated
   session. Review prediction-only output first. If the gates passed and current
   EEG is valid, **Arm EEG swipes** explicitly enables only left/right desktop
   navigation. A working global **Escape** shortcut is required before arming.
   Hold a confident rest prediction for one second, then sustain a confident
   direction across three predictions. A two-second cooldown and another rest
   period limit repeats. The camera is not required for EEG predictions or EEG
   actions, even if its preview becomes unavailable. Starting in EEG mode skips
   the camera, but a new acquisition requires new guided calibration.
8. Webcam control remains available through **Arm webcam swipes** in Webcam
   mode. Fresh camera data determines direction and fresh EEG gates availability.
   Webcam and EEG controls cannot be armed together. Brief uncertain or
   out-of-distribution predictions suppress actions and reset the direction
   streak; three seconds of continuous uncertainty disarms. Mode changes,
   stale/invalid EEG, packet gaps, disconnect, or failed actions disarm EEG
   immediately; recovery never rearms it. Safe Mode and the desktop capability
   gate apply.
9. Press **Escape** from any application while EEG control is armed, or choose
   **Stop & disarm (Esc)** when finished. It cancels startup/contact checking,
   releases the preview camera and stops EEG. If native shutdown cannot be
   confirmed, the panel reports the error and the worker is terminated.

Frames and waveform samples are transient in memory and pass only between the
local backend and HUD. There is no recording, export, cloud upload, language-model
input, or raw-signal action logging. The waveform cache is bounded to five seconds
and at most 250 display points per response. The local EEG classifier separately
reads the full-rate in-memory buffer (at most 1,250 packets) and uses complete
two-second windows. Calibration retains only derived training features and
validation predictions, with 120 total trials maximum, and has no save/load path.
Hiding the panel clears its displayed frames and pauses raw-data HTTP polling;
acquisition continues until Stop, a device failure, or backend shutdown. Stop
before leaving the experiment. Restarting the backend clears calibration.
Models are bound to the acquisition session, channel layout, and sample rate;
after stopping/restarting acquisition, reset and repeat training/validation.
The main process renews a three-second Escape guard lease while armed. Shortcut
loss, renderer failure, or app exit stops renewal and disarms control; this is
not an authentication boundary against other programs on the same machine.

### Experimental EEG model and validation

Each channel is linearly detrended. Features are log variance and log integrated
Hann-periodogram power in 4–8, 8–13, and 13–30 Hz. NumPy implements a diagonal
linear discriminant classifier with 20% variance shrinkage, equal class priors,
training-only standardization, and training-only distance rejection. Camera
coordinates, motion, and labels are absent from inference inputs. Checks reject
incomplete windows, counter gaps, nonfinite values, flat channels, suspicious
extreme plateaus, and raw peak-to-peak variation above 10 mV in any channel.
Constant baseline offsets are removed during feature processing; they do not
alone fail the amplitude check. The peak-to-peak limit is applied before
detrending so large spikes or drift are still rejected. Rejection messages name
the affected channel and distinguish excessive variation from flat signals.
These are engineering checks, not validated diagnoses of electrode contact or
signal quality.

The first eight accepted validation trials of each class are fixed. Abstentions
count as incorrect. Arming eligibility requires balanced accuracy ≥75%, precision
and recall ≥70% for every class, and ≤10% accepted left/right predictions on rest
trials. With eight rest trials, any such false activation fails the gate. This
metric is **per held-out rest trial before debounce**, not activations per hour.
Live and validation predictions use the same fixed score ≥0.80, margin ≥0.25,
and distance limits. Displayed scores are uncalibrated, not probabilities.
These small-sample engineering gates do not establish future reliability or
neural intent, especially when labels involve physical movement.

The private `/multimodal/preview` route carries raw display data. Other preview
status/actions contain metadata only, and acquisition methods are excluded from
the public hardware action schema. Browser-origin and non-loopback requests are
rejected; this is not an authentication boundary against other local processes.

Automated verification uses synthetic signals, fake sensors and fake camera
frames. Live EEG acquisition, contact measurement and user-specific calibration
require an explicit user-started hardware trial; passing the software tests does
not establish working hardware synchronization or EEG intention decoding.
