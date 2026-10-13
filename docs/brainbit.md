# BrainBit and fast double-blink mouse clicks

Updated **October 12, 2026** on
[`update_10_12_2026_eeg_signal_status`](https://github.com/soheilAppear/IntuitionOS/tree/update_10_12_2026_eeg_signal_status).

The EEG workflow has one detector: a small causal CNN that recognizes a deliberate
double blink from the latest one second of four-channel EEG. A recognized pair
sends one left click at the current cursor. Camera aiming is optional, and the
camera does not need to see your hand for an EEG click. Standalone Hand controls
remain available separately.

## Setup and connection

Use the project's Python environment. The local Windows installation uses
64-bit Python 3.12.3 and the pinned `pyneurosdk2==1.0.15` package:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-brainbit.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-eeg-cnn.txt
.\.venv\Scripts\python.exe start_ui.py
```

The second requirements file installs the CPU trainer. Live inference uses NumPy;
PyTorch is imported only when you explicitly train. Neither EEG preview nor blink
recognition needs Ollama. A missing headset SDK reports an availability error
without preventing the rest of IntuitionOS from starting.

1. Power on the charged headset near the computer. Close another app's connection
   to it before connecting from IntuitionOS.
2. Open **BrainBit**, choose **Discover devices**, select the headset, and **Connect**.
3. Battery and firmware metadata refresh while connected. The connection panel
   itself does not start EEG acquisition.
4. Wear the headset with the electrodes and reference touching skin/scalp. Turn
   off standalone Hand controls if the EEG workflow will use the camera.
5. Open **Fast double blink**, set **Camera mouse** as desired, then choose **Start**.
   The camera setting is enabled by default; acquisition and input still require
   their separate explicit Start and Arm actions.

The native SDK runs in a separate worker process. Windows Bluetooth LE carries
the connection. A working compatible Bluetooth USB adapter can provide the radio,
but the SDK does not identify which adapter Windows selected. A dongle's presence
alone does not prove the connection uses it. The tested headset family is
`LEBrainBit2`; the app supports the BrainBit families exposed by the pinned SDK.

Connection setup has a 40-second deadline and **Disconnect** cancels a pending
operation. A failure releases the worker; rediscover before retrying because device
selection IDs belong to one worker session. The app does not reconnect automatically.
`hardware.drivers` in `config/config.yaml` enables or disables the BrainBit adapter;
restart after changing the hardware configuration or installing its dependencies.

## Fast double-blink calibration

An existing saved fast model can be reused: start a compatible live EEG stream,
choose **Use saved fast CNN**, wait for fresh signal, and explicitly arm. Loading
preserves the original validation results and does not train, start a recording,
or arm controls. Older detector archives cannot be loaded into this profile.

For a new calibration:

1. Leave controls disarmed and collect **eight No command** and **eight Double blink**
   training trials. Follow the guided batches or collect individual trials.
2. Each cue provides a three-second preparation countdown, followed by exactly
   one second of recording. For a Double blink cue, complete two quick gentle
   blinks during that recording and reopen your eyes. Keep your jaw relaxed.
3. No command cues include relaxed eyes-open rest, ordinary looking, a single
   blink, and gentle head movement. These examples teach the model when to stay
   quiet. Follow the displayed cue rather than making a double blink.
4. Choose **Train & freeze fast CNN**. Training uses only accepted training trials.
   If the examples cannot be separated, collect more before freezing; the limit
   is 48 training trials in total.
5. Collect **eight new trials of each label in Validation**. The frozen model
   scores these separately; validation examples never update its weights.
6. Save after validation. Arming requires complete passing validation and a
   compatible fresh stream. A failed result remains failed when saved and loaded.

Accepted trials contain 250 consecutive four-channel samples. Missing packets,
incorrect timestamps, invalid values, clipping, flat signals, and excessive
amplitude reject a trial instead of counting incomplete data as a success.
Starting collection disarms input. Cancelling a batch does not turn a partial
recording into an accepted example.

Held-out validation requires balanced accuracy of at least 75%, per-class precision
and recall of at least 70%, and a no-command false-activation rate no greater than
10%. These are engineering gates on separate one-second trials, not measurements
of continuous false clicks or physical blink-to-click latency.

## Camera aiming and mouse input

After validation, explicitly check the arm control. With Camera mouse enabled,
point your index finger briefly to engage movement, then move your hand to aim.
The preview shows the same hand joints and connections as camera-only mode.
The backend copies the current standalone tracker, camera index, model complexity,
and optional bend-click preference before starting the EEG camera.

- A recognized **double blink sends one left click**, even if no hand is visible.
- **Pinch and release** sends a finger click; **pinch and hold** drags until release.
- Finger input takes priority. It discards pending EEG commands; after finger
  release, EEG clicking requires one second of newer signal and recognized rest.
- Losing hand tracking pauses movement and releases an owned drag. Show your
  hand and point briefly to resume aiming. Hand loss alone does not block an EEG click.
- Changing Camera mouse disarms input but retains the model and running EEG session.
- **Escape** disarms input from any app. **Stop** also ends EEG and releases the camera.
  Closing a panel alone does not stop acquisition or silently disarm a live control.

After a blink click, the rolling EEG window remains available. The detector waits
for recognized rest and its one-second click cooldown before another command.
It evaluates new samples every 40 ms and requires 80 ms of persistent evidence.
A one-second window does not guarantee that every physical blink produces a click
in under one second; signal delivery, the gesture, classification, and recovery
all contribute to observed latency.

A physical mouse button or keyboard modifier prevents the EEG actuator from
issuing a click. Failed release retains ownership so Stop can retry release
without repeating the press. The explicit input path does not change Safe Mode
or the confirmation rules for model-proposed operating-system actions.

## Yellow camera indicator

The header's yellow **CAMERA ON** indicator covers native capture by either mode.
It is separate from **HAND ON/OFF**, which controls standalone hand tracking.
Camera ownership stays visible while the EEG panel is closed or no hand is detected.

Starting and stopping have distinct states. A stop request does not imply that a
native capture handle has been released. If release fails or status is unavailable,
the indicator remains visibly uncertain rather than incorrectly showing OFF.
A confirmed release from both owners turns it off. The HUD polls metadata even
with both control panels closed, and rejects older status revisions.

## Signal faults and recovery

| Condition | Behavior |
|---|---|
| Temporary raw variation above 10 mV peak-to-peak | Pause EEG clicks; retain the existing arm and camera/finger controls. Discard pending evidence, then require fresh clean EEG and recognized rest. |
| Packet gap, stale delivery, invalid context/data, flatline, repeated samples, or clipping | Disarm; show and retain the reason. Recover the stream and explicitly arm again. |
| Escape lease expires or native input fails | Disarm and release owned input; do not replay a pending click. |
| Hand disappears | Pause movement and release a drag; EEG clicks may still use the current cursor. |
| Camera start/stop overlaps another request | Newer Stop/disable supersedes queued starts; release must complete before ownership can change. |
| Backend or headset disconnects | Mark status unconfirmed and revoke input. Reconnect and explicitly resume. |

The original repeated-unchecking issue was traced to raw amplitude excursions:
the previous service treated each excursion as a full-control shutdown. Only this
specific recoverable amplitude fault now pauses EEG independently. Transport,
clipping, contact-related flatline, and guard checks remain strict; the amplitude
limit and trained weights were not loosened.

**EEG live** means recent samples are arriving, not that they are clean. The panel
shows nominal/received rates, last-sample age, and cumulative packet diagnostics.
A sample older than 0.75 seconds is stale. Each waveform has its own display scale;
use numeric RMS and peak-to-peak values when comparing channels.

Movement, electrode drift, and electrical interference can affect a stream with
no packet errors. Recheck electrode contact and compare a quiet eyes-open interval.
The app does not diagnose contact, measure mains interference, or establish medical
signal quality. Eye and muscle artifacts can resemble the learned command.

## Model storage and privacy

**Save fast CNN & trials** writes bounded, inert JSON under the Git-ignored path
`data/eeg-fast-cnn-blink-models/fast-cnn-blink-*.json`. It retains numeric weights,
processed one-second training and validation waveforms, labels, and trial metadata.
These are personal signal-derived data. They are neither uploaded nor committed.
Camera frames and unprocessed live EEG remain in bounded memory buffers.

The fast format and existing saved weights remain compatible. Loading selects the
newest fast archive from that fixed directory. A corrupt newest archive fails
visibly instead of silently loading another model. Reads reject oversized/deep
JSON, duplicate fields, nonfinite values, symbolic redirects, wrong tensor shapes,
and incompatible profiles. Writes validate first and replace atomically.

Old personal model files remain untouched, but retired detectors, their selectors,
practice counters, EEG swipe experiments, and finger-snap analysis are removed from
the application. The current API provides no legacy-model or validation-bypass path.
Unfinished collection remains in memory until reset or shutdown; save a frozen
model to retain it. The camera preview's downsampled waveform is for display and
must not be treated as a full-rate recording for spectral analysis.

## Implementation and verification

| File | Responsibility |
|---|---|
| `plugins/brainbit.py`, `plugins/brainbit_worker.py` | Isolated SDK worker, bounded samples, cached status and cancellation. |
| `core/eeg_fast_blink.py` | One-second preprocessing, local training, frozen validation, causal NumPy inference and event rearming. |
| `core/eeg_fast_blink_store.py` | Strict inert JSON parsing and atomic fast-model persistence. |
| `core/multimodal.py` | Preview lifecycle, camera ownership, trial collection, Escape lease and final click authorization. |
| `core/hand_mouse.py`, `core/eeg_click.py` | Pointer/finger ownership and one current-position left press/release. |
| `interface/server.py` | Exact private HTTP operations, camera exclusion, capture-status aggregation and model storage. |
| `ui/renderer/multimodal-panel.cjs` | Fast-only calibration and live controls. |
| `ui/renderer/hand-overlay.cjs` | Shared hand-joint rendering for both previews. |

Inline comments explain the control transitions, numeric processing, authorization,
and cleanup steps. Tests use synthetic waveforms, fake clocks, fake cameras, and
fake actuators; they do not generate desktop clicks or connect to a headset.

```powershell
.\.venv\Scripts\python.exe -m pytest
$jsTests = rg --files tests -g '*_test.cjs'
node --test @jsTests
```

Live checks and automated results for this update are recorded in the
[October 12 update report](2026-10-12-fast-blink.md). Synthetic classification tests
cannot establish a wearer's real-world recognition accuracy.
