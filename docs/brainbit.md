# BrainBit connection, EEG signals, and status

The HUD's **BrainBit** panel can discover nearby devices, connect to a selected
headset, display its battery and firmware, and disconnect. It does not start EEG
streams. The separate, explicitly started **Experimental hand + EEG** panel
defaults to [EEG selection and mouse click](#eeg-selection-and-mouse-click), with
a hand-controlled pointer and a practice counter. The [advanced direction modes](#experimental-eeg--camera-preview)
retain webcam and EEG swipe experiments. Neither panel records signals or updates firmware.

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

## View the hand camera and EEG

1. Connect the headset in **BrainBit** and turn off the normal Hand controls
   camera.
2. Expand **Experimental hand + EEG**. While stopped, optionally choose
   **Contact check (5 s)** and wait for it to finish.
3. Keep the default **Control source → EEG selection / click** and **Selection
   target → Practice counter**. Choosing a mode does not start either sensor.
4. Choose **Start preview** to acquire the camera and EEG together. Leave **Arm
   hand controls + EEG practice** unchecked to view signals and hand tracking
   without moving the pointer, clicking, or incrementing the practice counter.
5. Watch the channel traces and **EEG live** indicator. BrainBit2 supplies four
   channels at a nominal 250 Hz. The observed rate is estimated from host packet
   arrivals and can fluctuate around that value.
6. Choose **Stop & disarm (Esc)** when finished. Closing the panel alone does
   not stop acquisition.

Waveforms and signal status work before any training and do not require Ollama.
An untrained-model message refers to classification, not signal acquisition.
The default rest/select labels below come from EEG and your explicit trial cues,
not camera-verified hand movements. Keep the camera running during calibration;
you do not need to point or hold your hand in view until arming. The advanced
**EEG classifier (advanced swipes)** mode still permits EEG-only acquisition
while its controls remain disarmed.

## EEG selection and mouse click

The default **EEG click method → Eyes closed (existing model)** within
**EEG selection / click** learns two labels: eyes-open **rest**
and gently closed-eye **select**. Your hand movements in the camera steer the
pointer; both finger gestures and EEG can click there. Point your index finger
briefly after arming to engage the pointer, then move your hand to aim.
Pinch and release for a finger click; hold the pinch to drag, then separate your
fingers to release. Optional index bend clicks follow your existing Hand controls
setting; this mode does not silently enable that preference. EEG adds one left
click per completed rest/select cycle, without scanning screen targets or
performing a double-click or drag. These are trained signal patterns. Eye activity,
muscle activity, and electrode movement can explain classification, so this is
not thought reading or a medical measurement. The separate
[Deliberate double blink method](#deliberate-double-blink-clicks) below preserves
this model and uses a brief paired-blink command instead.

1. Connect the headset and start the hand camera and EEG preview as above. Leave
   **Practice counter** selected and controls disarmed while collecting trials.
2. The trial phase starts at **Training**. Choose **Collect rest trial** and keep your
   eyes open, head still, and jaw relaxed for the five-second trial.
3. Choose **Collect select trial**. Gently close your eyes, count four seconds,
   then reopen. Each five-second trial excludes the first settling second and
   measures a complete two-second EEG window before you reopen. The labels come
   from your selected cue; a camera does not verify them.
4. Alternate until **8 accepted training trials per label** are shown. Choose
   **Train & freeze model**; the phase automatically changes to **Validation**. Collect **8 new
   trials per label**, using the same cues without retraining. These are separate
   samples scored by the frozen model. Failed validation blocks the normal EEG
   click path; hand aiming and finger clicks remain independently available.
   **Save EEG model & trials** preserves the model and original results on disk.
   **Use saved EEG model** selects it explicitly for a compatible running stream,
   retaining its original validation result even when it failed.
5. Keep acquisition running and your hand visible. Explicitly choose
   **Arm hand controls + EEG practice** to enable hand controls. A usable model
   additionally enables EEG selections; uncertain EEG does not prevent manual
   aiming or finger clicks.
   Point your index finger briefly, then move your hand to aim. **Practice affects
   EEG only: finger clicks and drags still affect the desktop.** The EEG selection
   result goes to **EEG practice** instead of sending a mouse click. The panel
   counts EEG clicks and finger clicks separately. Keep your hand visible and still during the
   selection cycle; check the hand-pointer hint next to the camera.
6. To click, choose **Selection target → Mouse left click**. Changing
   the target disarms and revokes the previous Escape guard; explicitly choose
   **Arm hand controls + EEG clicks** again. Point briefly, move your hand to the
   intended target, hold it still, and perform the same selection cycle. The
   pointer holds only after an EEG selection is confirmed. Ordinary aiming continues
   while EEG is uncertain or unavailable; that condition suppresses EEG clicks.
   Finger gestures take priority: beginning a finger interaction cancels the
   queued EEG selection and requires a fresh EEG rest cycle before another one.
   A completed click is counted only after
   both the button press and release succeed. Hand tracking loss releases an owned
   drag and cancels pending EEG input. Return your hand and point briefly to resume
   without another arm-button click or a cursor jump.
7. Press **Escape** from any application while armed, or use **Stop & disarm
   (Esc)**. Keep Stop available if the app reports unconfirmed input cleanup.

One selection follows **rest → select → rest**. First, the model must recognize
one second of confident rest. It then needs three distinct confident select
prediction windows. Reopen your eyes: one second of confident rest completes
the action. The click occurs after this return to rest, not when your eyes first
close. Holding select cannot repeat an action. Every later action needs a new
cycle, and actions are separated by at least two seconds. A prepared selection
that does not complete within eight seconds expires without disabling hand controls.

Arming hand controls requires fresh initial hand tracking and a working global
Escape shortcut. EEG clicks additionally need valid current-session predictions
and either passing validation or explicit selection of an experimental saved model.
The combined pointer reads current inference landmarks on a 30 Hz control
schedule; it is independent of the eight-FPS image preview and 250 ms EEG
prediction interval. Each camera frame is consumed once, with no extrapolated
movement when the hand is missing.
Starting preview, loading a model or collecting trials never arms input.
Brief model transitions suppress EEG actions while preserving a pending selection
for up to three seconds. Longer uncertainty, stale/invalid EEG or packet gaps cancel
the EEG cycle; manual hand input remains available. Hand loss pauses movement and
releases an owned drag; stable pointing resumes at the actual cursor without
replaying an old click. Escape, Stop, lost guard renewal, target/mode changes and
native input failures revoke authorization and require explicit arming again.
Held physical mouse buttons or
keyboard modifiers block a mouse click. If release fails, the actuator retains
ownership and Stop retries release without repeating the press.

This explicitly armed pointer-input path is separate from model-proposed OS
commands, like the existing Hand Mouse controller. It does not change Safe Mode
or weaken the confirmation rules for the generic `os_click` capability. Switching
between click mode and the advanced direction modes resets the incompatible
model. After restarting acquisition, collect a new calibration or explicitly use
a saved model for compatible-session experimental inference.

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
checks does not establish clean EEG or reliable classification.

The click model's first feature is integrated **4–30 Hz power**, replacing the
legacy model's broadband variance. Its remaining features cover 4–8, 8–13, and
13–30 Hz. Thus the first click feature does not integrate the 60 Hz frequency
bins. This is bounded feature extraction, not a cosmetic filter on the raw
waveforms, a notch filter, or proof that interference is gone. Raw amplitude,
flatline, plateau, and packet checks still run before band selection with their
existing limits.

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

`core/multimodal.py` owns hand-pointer authorization, the rest/select cycle, and
the final camera/EEG freshness, session, validation, and Escape checks.
`core/hand_mouse.py` supplies pointer movement, pinch clicks, pinch-hold dragging,
and optional bend clicks using the existing Hand controls preference. Finger
interactions take priority over queued EEG output. `core/eeg_click.py` lazily creates the existing
Windows mouse adapter for one current-pointer left press/release; construction
injects no input. The practice target changes only EEG output to a counter;
explicitly armed finger controls still move, click, and drag. The new feature, control, actuator, and HTTP
logic has inline comments explaining its state changes and cleanup. The private
`eeg_target` operation accepts only `practice` or `mouse`, never coordinates or
an immediate click request.

Tests use fake SDK objects or harmless subprocesses rather than live devices:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_brainbit.py tests/test_brainbit_worker.py tests/test_brainbit_protocol.py tests/test_startup.py
.\.venv\Scripts\python.exe -m pytest -q tests/test_eeg_decoder.py tests/test_eeg_click.py tests/test_eeg_click_control.py tests/test_eeg_click_integration.py tests/test_multimodal_protocol.py
node --test tests/brainbit_renderer_test.cjs tests/brainbit_http_test.cjs tests/hud_renderer_test.cjs tests/multimodal_renderer_test.cjs tests/multimodal_http_test.cjs tests/eeg_guard_test.cjs
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

These are the **advanced direction modes**, not the default selection workflow.
In the same HUD panel, **Webcam direction (advanced)** displays mirrored webcam
hand tracking next to four EEG channels, using approximate host arrival timing.
**EEG classifier (advanced swipes)** predicts left, right, or rest exclusively from EEG features.
Camera movement verifies training and validation labels, and never enters the
EEG classifier or supplies its action direction. Muscle activity and electrode
motion can explain successful classification: this is not thought reading or
a medical measurement, and software tests do not demonstrate reliable control.

1. Connect BrainBit in the connection panel. Wear it according to its manual;
   keep it powered on and disconnect the charger if the device manual requires it.
2. Turn off the normal Hand controls camera. Expand **Experimental hand + EEG**
   and select **Control source → Webcam direction (advanced)** while stopped.
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
7. Select **EEG classifier (advanced swipes)** without stopping acquisition to retain the calibrated
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
local backend and local preview clients. There is no raw recording, raw export,
cloud upload, language-model input, or raw-signal action logging. The waveform cache is bounded to five seconds
and at most 250 display points per response. The local EEG classifier separately
reads the full-rate in-memory buffer (at most 1,250 packets) and uses complete
two-second windows. Calibration retains only derived training features and
validation predictions, with 120 total trials maximum. Explicit JSON model archives
are supported by `core.eeg_model_store` and the HUD's **Save EEG model & trials**
and **Use saved EEG model** controls.
Hiding the panel clears its displayed frames and pauses raw-data HTTP polling;
acquisition continues until Stop, a device failure, or backend shutdown. Stop
before leaving the experiment. Restarting clears the in-memory instance; saved
archives retain the model and its original trials. After reconnecting and starting
a compatible stream, **Use saved EEG model** binds inference to that stream without
claiming its historical validation applies to the new acquisition.
The main process renews a three-second Escape guard lease while armed. Shortcut
loss, renderer failure, or app exit stops renewal and disarms control; this is
not an authentication boundary against other programs on the same machine.

### Model backups and reuse

A saved model is different from a status report. An eyes-closed model archive contains the
original coefficient arrays, derived training vectors, validation predictions,
trial identities and acquisition context. A status report contains only metrics
and cannot restore the classifier. Neither contains recorded raw EEG trials.

`core.eeg_model_store.save_decoder` writes a validated JSON archive atomically;
callers accessing a live service must hold its decoder lock. `load_decoder` checks
the stored dimensions, finite values, trial identities and parameter consistency
against training features. It restores the original validation result, including
failure. It never enables desktop input, changes the acquisition identity or
converts historical validation into permission for a new session. No pickle or
executable object format is used. Keep personal archives in `data/eeg-models/`,
which Git ignores. Choose **Save EEG model & trials** after completing validation
to write a new archive containing the full results; future trials are not saved
automatically. There is no raw EEG recording in a model archive.

In the existing HUD, start the compatible camera/EEG preview and choose **Use saved
EEG model**. The app selects the newest `model-*.json` in its fixed local archive
folder. It rejects incompatible channels, sample rate or feature processing. Model
selection keeps controls disarmed and preserves the original trial counts and
validation results. For the eyes-closed method, explicit saved-model use allows experimental EEG selections
even when historical validation failed; the failure remains visible.

Choose **Mouse left click** and **Arm hand controls + EEG clicks** for real desktop
use. Hand aiming and finger clicks work independently of EEG confidence. EEG adds
one left click only after its full rest/select/rest sequence and current-data
checks. Press **Escape** to disarm both. No separate practice app is required.

### Deliberate double-blink clicks

Choose **EEG click method → Deliberate double blink** in **Experimental hand + EEG**.
It keeps the same real Windows cursor, hand aiming, pinch clicks, pinch-hold drag,
and optional bend-click setting. The detector learns your local ocular waveform
from the EEG headset. It recognizes a deliberate pair as one left-click command;
a single blink is not a command. This is eye-movement detection, not thought
reading. It uses a separate model: existing eyes-closed training and validation
remain available when you switch back. Switching methods always disarms input.

1. Start the compatible camera and EEG preview, select **Deliberate double blink**,
   and leave controls disarmed while collecting trials.
2. Choose **Collect remaining training**. It alternates unfinished classes until
   each has **8 accepted training trials**, skipping any quota already completed.
   A visible three-second countdown prepares you for each cue. For no command,
   read normally and allow natural single blinks. For double blink, follow the
   cue and blink twice gently at a comfortable quick pace, letting your eyes reopen
   between blinks; then keep them open for the end of the trial. Keep your head
   and jaw relaxed. Manual **Collect no-command trial** and **Collect double-blink
   trial** buttons remain available when the guided sequence is stopped.
3. Choose **Train & freeze model**, then explicitly choose **Collect remaining
   validation** for **8 new no-command and 8 new double-blink validation trials**
   against that frozen model. No-command validation should
   include ordinary single blinks, reading, and gentle jaw/head movement. The
   validation samples do not retrain the model. Hand controls stay disarmed during trials.
4. After validation passes, choose **Mouse left click**, then explicitly check
   **Arm hand controls + EEG clicks**. Move your hand to aim and make one deliberate
   double blink for one left click. Keep your eyes open afterwards. There is no
   sustained eyes-closed hold or one-second rest/select/rest completion sequence.
   Finger gestures still work and take priority over pending EEG input. **Practice
   counter** remains an optional EEG-only destination; finger input still affects
   the real desktop there. **Escape** stops both controls.
5. Choose **Save double-blink model & trials** after validation. To reuse it with a
   compatible running stream, select the same method and choose **Use saved
   double-blink model**, then arm separately. A failed blink validation remains
   failed and cannot be bypassed by loading the archive. Hand controls remain
   available even while the EEG add-on waits for validation or a usable signal.

Each blink trial collects exactly 1,250 consecutive samples, representing five
seconds at 250 Hz. The sample counter defines the captured window; variable
Bluetooth callback arrival times do not cut a valid trial short. If necessary,
the backend waits up to another 1.5 seconds for delivery. **Finishing signal
collection** means the trial is still pending: keep your eyes open and wait.
Missing packets or an incomplete window still reject the trial. The guided
sequence advances only after its accepted count increases; a rejection pauses
it and keeps all previously accepted trials. **Cancel queued trials** stops
future cues without aborting the current trial or clearing data. It never trains
or arms automatically. The pulse detector separates quick pairs even when the
first blink's filtered signal has not fully settled, while retaining the
single-blink rejection and separate validation requirements.

Training retains all accepted examples and uses a consistent set of whole
double-blink trials—at least 6 of 8, or 75% for larger sets—with small timing
alignment to tolerate modest differences between blinks. After both classes
reach eight and before freezing, **Collect another double-blink trial** adds
clearer examples without resetting the existing data, up to 80 accepted training
trials total; the model status reports any retained examples excluded from
learning. Fresh validation against the frozen model is still required, with its
separate limit of eight new trials per class.

Keep the preview running through training and validation. To pause the guided
cues, choose **Cancel queued trials**; accepted trials stay in the current
calibration. Choose **Collect remaining training** or **Collect remaining
validation** when ready to continue the current phase.

Blink backups use `double-blink-*.json` in the same Git-ignored `data/eeg-models/`
directory; the eyes-closed loader continues to use `model-*.json`. Each method's
Save, Load, and Reset controls affect only its own calibration. Blink archives
contain learned waveform features and trial/validation metadata: normalized,
resampled pulse shapes (24 bins across four channels), amplitude, duration and
timing summaries, noise scales, and the learned waveform template. They do not
contain original full-rate EEG packets or camera frames. Live scores describe
waveform similarity, not probabilities. Saved historical validation remains
labeled as historical when inference binds to a new acquisition.

The detector processes incoming samples and waits for a complete recognized pair;
the UI's five-second calibration trial is not the live click delay. End-to-end
response time and false activations with the actual wearer must be measured;
automated checks do not establish those values.

### Experimental EEG model and validation

For the eyes-closed and advanced direction models, each channel is linearly detrended. The binary click profile uses log integrated
Hann-periodogram power in 4–30, 4–8, 8–13, and 13–30 Hz; the legacy direction
profile uses log variance followed by the three narrower band powers. The same
profile is used for training, held-out validation, and live prediction, and
features from different profiles cannot be mixed. NumPy implements a diagonal
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
count as incorrect. Normal EEG-click eligibility requires balanced accuracy ≥75%, precision
and recall ≥70% for every class, and ≤10% accepted action predictions on rest
trials: **select** for the binary model, **left/right** for the legacy model.
With eight rest trials, any such false activation fails the gate. This
metric is **per held-out rest trial before debounce**, not activations per hour.
Manual hand controls do not depend on this score. Explicit eyes-closed saved-model selection
is separately labeled experimental use and leaves these evaluation gates unchanged.
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
