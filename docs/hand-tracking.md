# Hand tracker selection — October 2026

## Laptop default

The 8 GB RTX 4070 Laptop configuration selects **MediaPipe Hands Full**
(`tracker_backend: mediapipe`, `model_complexity: 1`). It is included in the
project's Python requirements and performs hand inference on the CPU, leaving
GPU memory available for Ollama. WiLoR and RTMPose require separate setup.

## Optional GPU model

**Full WiLoR with AnyHand fine-tuned weights**, using all 32 transformer layers.
The app uses the dedicated hand detector, reconstructs 21 joints in 3D, then
projects them into the mirrored camera frame. Gesture recognition remains the
app's own control logic. It is selected for improved hand retention and 3D pose
robustness; a universal "best webcam tracker" is not established by the evidence.

The [AnyHand paper](https://arxiv.org/html/2603.25726v3) reports WiLoR+AnyHand
PA-MPJPE of 5.394 mm on FreiHAND and 7.355 mm on HO3D, compared with original
WiLoR's 5.5 and 7.5 mm. These are aligned 3D benchmark errors, not screen-cursor
errors. [Released checkpoints and inference code](https://github.com/chen-si-cs/AnyHand).

An [independent June 2026 comparison](https://arxiv.org/html/2606.17427v1) found
WiLoR had the lowest error among WiLoR, HaMeR, HoloLens 2, MediaPipe and WildHands.
WiLoR, HaMeR and WildHands lacked predictions on fewer than 1% of accepted frames;
MediaPipe lacked predictions on about 22%. The setting was egocentric object
interaction, and only 38% of recorded frames met the study's ground-truth checks.
This supports testing WiLoR for occlusion, but does not establish frontal webcam
mouse-control accuracy or evaluate the AnyHand checkpoint itself.

## Local comparison

The following cross-model comparison was recorded on September 27, before the
October detector-resolution change below.

Hardware: RTX 5090, Ryzen 9 9900X, Windows. Four public annotated OneHand10K
fixtures from [MMPose](https://github.com/open-mmlab/mmpose/tree/main/tests/data/onehand10k),
each tested unchanged, rotated -25°, rotated +25°, and with a small synthetic
index-fingertip occlusion. All 16 inputs preserve aspect ratio in 640×480 frames.
Models use this app's actual detection and rejection rules, with identity reset
between cases. No camera, mouse clicks, or desktop actions were used.

| Model | Accepted hands | 2D PCK@0.1 including misses | Median inference |
| --- | ---: | ---: | ---: |
| MediaPipe Full | 16/16 | 96.0% | 4.5 ms tracked; 20.3 ms initial |
| RTMPose Hand5 | 11/16 | 79.8% | 8.2 ms |
| WiLoR + AnyHand, FP32 | 16/16 | 91.1% | 37.4 ms |
| WiLoR + AnyHand, FP16 (selected) | 16/16 | 91.5% | 31.9 ms |

PCK uses 10% of the annotated bounding-box diagonal, annotation-marked joints
inside the frame, and excludes joints covered by the added occluder. Missed hands
count as failures. Mean normalized visible-joint error among accepted predictions
was 0.0338 for MediaPipe, 0.0275 for RTMPose, and 0.0423 for selected WiLoR. RTMPose's lower
conditional error excludes its rejected hands and should not be treated as better
overall performance. WiLoR's visible 2D localization did not beat MediaPipe here.

The selected FP16 precision retains every model layer and checkpoint tensor.
A separate four-image precision comparison measured maximum projected-joint
deviation of 0.575 pixels from FP32 at 640×480, with all predictions finite.
This limited comparison supports the lower-latency default; it is not a numerical
equivalence guarantee for every frame. `WiLoRModel(..., precision="float32")`
remains available for further diagnostic comparisons.

This is a bounded reference/synthetic check, not a live webcam or statistically
representative benchmark. It does not measure real movement jitter, input latency,
gesture accuracy, or this user's hand. Raw reports and the local comparison script
remain in ignored `data/hand-benchmark/` and `data/compare_hand_models.py`.

## October 10 RTX 5090 tuning

The full AnyHand-WiLoR model already runs on CUDA with FP16 and all 32 layers.
Increasing model size alone would not fix combined hand+EEG control: that path
previously consumed landmarks from the JPEG preview, capped at eight updates per
second, and its control loop also waited 125 ms between iterations.

Combined click mode now reads a separate, fresh landmark snapshot on a 30 Hz
control schedule. Each camera sequence is consumed once. The display keeps its
image-aligned, eight-FPS preview, and the legacy bandpower EEG classifier keeps
its separate 250 ms cadence. Slow or missing inference still pauses controls;
this does not invent hand positions between camera frames or continue moving
after hand loss.

Camera reads now run independently of hand inference and JPEG encoding, with
only one pending frame. If inference falls behind, the next pass takes the
newest frame instead of processing a backlog. Frames older than 500 ms cannot
drive input. This keeps the selected pretrained tracker and camera resolution;
it does not synthesize motion between observations. Combined blink mode also
shares one detached EEG snapshot between its display and decoder to reduce
repeated copying. Camera status exposes capture-read, pending-frame wait, and
inference timings for diagnosis; those host timings do not measure sensor
exposure-to-screen latency.

Capture requests 30 FPS after setting 640×480, falling back to the camera's
working default if the driver rejects that setting. On the current Windows
webcam, a read-only capture probe measured 19.8 FPS at its default and 29.6 FPS
after the explicit request, without changing resolution or pixel format.

The detector now uses a **960-pixel inference size**, retaining the trained
256-pixel reconstruction crop and the same confidence and anatomy checks.
The camera frame remains 640×480; upsampling detection adds no sensor detail.
A same-run comparison on the RTX 5090 used the same 16 fixtures plus eight fixed
small-hand cases (four originals reduced to 50% and 30% scale):

| Detector size | Original/rotated/occluded cases accepted | Small-hand cases accepted | Median complete inference |
| --- | ---: | ---: | ---: |
| 640 | 16/16 | 6/8 | 24.0 ms |
| **960 (selected)** | **16/16** | **7/8** | **25.9 ms** |
| 1280 | 16/16 | 7/8 | 29.6 ms |

The 1280 setting also reduced mean per-case PCK on these fixtures. This small
comparison supports 960 as a practical setting, not a guarantee for every hand,
lighting condition, or webcam. Timing excludes camera capture, IPC and mouse
dispatch. The ignored runner is `data/compare_wilor_detection.py`, with detailed
results in `data/hand-benchmark/detector-scales.json`. Diagnostic comparisons can
select `WiLoRModel(..., detector_size=640)` or `detector_size=1280` explicitly.

An eight-second read-only check after loading the updated main app observed
28.3 combined-control ticks/s and 23.7 distinct camera updates/s, with 47 ms
median and 93 ms 95th-percentile control-frame age. The saved EEG model remained
loaded with all 32 trials. Input was disarmed and a hand was visible in only
25 of 357 status polls, so this verifies the faster delivery path rather than
sustained hand accuracy or real click performance. The local report is
`data/hand-benchmark/live-control-cadence.json`.

The [AnyHand authors](https://github.com/chen-si-cs/AnyHand) recommend WiLoR for
most users. Newer temporal reconstruction research is not automatically suitable
for live control: [ViDiHand](https://arxiv.org/html/2606.30308) reports 5.5 FPS on
four A100 GPUs and describes its current use as offline annotation. No claim is
made that this app uses a universally best tracker.

## Runtime behavior

- Install with `python setup_wilor.py`; runtime and assets are isolated under `data/`.
- Setup pins source revisions and verifies downloaded model SHA-256 checksums.
- All 443 inference state tensors must match; incomplete model loads fail visibly.
- CUDA initialization failure never silently selects a different model or CPU.
- The worker has no camera or desktop input access; the recognizer owns capture.
- The worker blocks outbound socket connections and disables detector telemetry,
  cloud integrations and automatic dependency installation.
- One frame is processed at a time. Responses older than 250 ms are discarded.
- Camera Off cancels input and terminates the worker. A Windows job also terminates
  the worker if the owning backend exits. Cleanup failures remain visible/retryable.

## Hand Mouse controls

Point with your index finger and keep the other fingers curled briefly to activate
the pointer. Once active, you can relax your fingers and move your hand;
the pointer follows the knuckles rather than requiring a continuously raised
index fingertip. Fully opening all four fingers enters navigation. A full fist
pauses immediately. Lowering your hand out of view
also stops movement; point again to resume after a longer interruption.

Pinch thumb and index, then separate them for a click; the index fingertip can
stay curled as long as its proximal joint stays raised. Keep the fingers apart
briefly before the next pinch. Hold the pinch for about 0.35 seconds to drag, and
separate them to drop. Opening all four fingers finishes an active pinch or drag
before navigation can take over on a following frame. Index bend clicking is off
by default so ordinary finger flexing does not compete with pointing. To enable
it, stop the camera, select **Index bend click (optional)** in Hand controls, and
click **Apply**.
With bend clicking enabled, straighten the index briefly between clicks to avoid
triggering both click gestures. The option lasts until the backend restarts. Its
guide appears only when the backend confirms it is enabled.

Moving the hand still moves the cursor while the fingertips approach a pinch.
Only actual pinch contact holds the click target. The pointer uses the proximal
index joints, so curling the fingertip alone does not move that target. This
also prevents tracking noise near the approach boundary from repeatedly
freezing the cursor and discarding hand travel.

Brief tracking uncertainty freezes the pointer for up to 150 ms, allowing the
same hand to recover without repeating the activation pose. Missing or unreliable
input cancels a pending click and releases a drag immediately. Recovery itself
never clicks: release any pinch before making a new click gesture. A different
hand or a longer interruption requires pointing again.

## Camera mouse with fast double-blink clicks

Updated October 13, 2026. **Fast double blink** is the only EEG detector.
Enable **Camera mouse** in that panel to use the same tracker, camera index,
model complexity, optional bend-click preference, desktop movement mode, and
hand travel as standalone Hand controls.
Both previews share one renderer for the hand skeleton and its 21 joints.
The configured laptop default is MediaPipe Hands Full; selecting another supported
tracker in Hand controls applies to the next EEG camera start as well.

Only one mode can own the camera and pointer at a time. Turn off standalone
Hand controls before starting the combined preview. Camera mouse is selected by
default, but capture still starts only after an explicit Start. You may disable
it and use the fast EEG detector with your ordinary mouse or touchpad.

After loading a validated model and explicitly arming, point briefly to engage
movement, move your hand to aim, and double blink for one left click. A missing
hand pauses movement and releases an owned drag, while EEG clicking at the
current cursor remains available. Pinch/release clicks, pinch-hold drags, and
left/right desktop swipes take priority over EEG. After these hand actions end,
the detector requires one second of newer EEG and recognized rest so a buffered
blink cannot become a late click. The ordinary click cooldown also still applies.

For desktop navigation, raise four fingers with your thumb relaxed, hold still
for 0.30 seconds, and move horizontally. Left reveals the next desktop to the
right; right reveals the previous desktop to the left. Reach 100% and hold for
0.10 seconds to finish. The navigation hint beneath the camera follows progress;
the pointer and EEG clicks wait while navigation owns input. Lower your hand or
make a fist for 0.20 seconds before another swipe, then point again to aim. A fist
or pinch before completion cancels; a cancelling pinch never becomes a click.

The combined mode supports horizontal navigation only, using the same hand-travel
and desktop-movement settings as camera-only mode. Native smooth gestures follow
Windows' four-finger touchpad assignment; measured steps send an adjacent-desktop
shortcut when complete. At least two desktops must already exist. Configure
these options in Hand controls with the camera off. No new EEG training or extra
arm switch is needed; Escape and Stop cancel desktop navigation too.

Raw EEG amplitude alone no longer pauses clicks or rejects a collected trial;
the October 13 update removes the former 10 mV peak-to-peak cutoff. The CNN
still uses the same normalized waveform input and existing saved models remain
compatible. Large movement or contact artifacts can still resemble a blink;
normalization does not guarantee their correct classification. Invalid data,
flatlines, repeated samples, clipping, transport, guard, or input faults still
disarm and retain a visible reason. Escape disarms both forms of input; Stop
releases acquisition. Toggling Camera mouse disarms input and preserves EEG
calibration and its session.

The yellow **CAMERA ON** header indicator covers both standalone and EEG capture.
It remains active during release, and indicates uncertainty if release fails or
status becomes unavailable. It turns off after confirmed release, independently
of hand visibility, the selected panel, or whether mouse control is armed.
The separate **HAND ON/OFF** button controls only standalone hand tracking.

See the [fast CNN guide](brainbit.md#fast-double-blink-calibration) for collection,
validation, model reuse, and timing. A one-second input window is not a guarantee
of subsecond physical click latency.

## Desktop and Task View navigation

Navigation works alongside Hand Mouse. Raise your index, middle, ring and pinky
fingers, keep your thumb relaxed and apart from your index, and hold still for
0.30 seconds. The pointer pauses while you navigate. In Desktop gestures mode,
start with an open palm. After activation, three raised fingers are enough to
keep the swipe active; a slightly relaxed pinky no longer interrupts it.

After activation, a 0.25-palm dead zone filters small movements. The first clear
movement locks the swipe to a horizontal or vertical axis. Left/right moves
between desktops; up reveals Task View with windows and desktops; down shows the
desktop. Progress follows signed hand travel and reverses when you move back.
The **Hand travel to 100%** slider sets the travel needed to commit (1.2 palms by
default); it is available in either mode with the camera off.

Reach 100% and hold for 0.10 seconds to finish automatically. The **Hold briefly
to finish** hint marks this short completion dwell. Once it starts, progress
between 90% and 100% tolerates small tremors; reversing below 90% resets the dwell.
There is no separate finishing pose. A fist or pinch before completion cancels
the swipe. Brief missing or uncertain tracking freezes progress for up to 150 ms;
recovery continues only for a nearby hand, and the gap cannot finish a gesture.
Longer loss, a distant reappearance, stopping the camera, or an input error cancels.

One gesture produces at most one action. Lower the hand or make a fist for 0.20
seconds before another swipe; simply keeping the hand raised does not repeat.
Point again to use the mouse. In Desktop gestures, the reset is required before
new window-control poses too: held pointing, two fingers, thumbs-up, or pinch
cannot become a window action after navigation. Navigation cannot emit a mouse
click, and a pinch used to cancel cannot become a click or close request. An inactive swipe
times out after 8 seconds without meaningful movement, or 20 seconds total.

**Smooth when supported** uses a synthetic Windows precision touchpad for
horizontal navigation, with four contacts. A 60 Hz interpolation loop smooths
movement between camera samples and finishes with a bounded follow-through and
release. Windows owns the animation and final snap. Configure **Windows Settings
→ Bluetooth & devices → Touchpad → Four-finger gestures** for desktop switching.
Smoothing cannot improve tracking accuracy or remove camera latency. Availability
is checked without injecting contacts; this check does not verify how a live
swipe looks or whether Windows honors the configured gesture.

Up/down navigation always sends one Windows shortcut at completion: Win+Tab for
Task View, or Win+D to show/hide the desktop. This avoids depending on a vertical
touchpad assignment. It does not reveal Task View progressively with hand travel.
Both are toggles, so a new completed gesture can close Task View or restore the
windows hidden by Show desktop.
To choose a desktop in Hand Mouse, swipe up to open Task View, lower your hand
briefly, point to resume the pointer, and pinch/release on a desktop thumbnail.

**Measured steps**, also used when native horizontal support is unavailable,
sends Win+Ctrl+Left/Right once at completion for adjacent desktops. It does not
move desktop content continuously while the hand moves. Windows then runs its
own transition. No desktop is created automatically; at least two are needed.

The native path follows Microsoft's [precision touchpad injection guide](https://learn.microsoft.com/en-us/windows/win32/input-precisiontouchpad/precision-touchpad-guide#injecting-touchpad-input)
and [standard gesture mappings](https://support.microsoft.com/en-au/windows/hardware/input-devices/touch-gestures-for-windows).

See [navigation validation](hand-navigation-validation.md) for reproducible
checks, expected event sequences, and the limits of the current verification.
Controlled Windows input checks switched to an adjacent desktop and restored it,
displayed Task View, and showed/restored the desktop. Camera gesture accuracy and
perceived smoothness have not been measured by those checks.

## Camera startup timeout

The detector's first nonempty hand prediction can initialize CUDA NMS lazily.
On the local RTX 5090, a reference hand image took 2.9 seconds on its first
prediction and 31–32 ms afterward. A blank warmup frame did not exercise that
path, so the former one-second frame deadline could stop capture at startup.

Warmup now exercises detector NMS and both one-hand and two-hand reconstruction
before the camera opens. The first camera frame also has a bounded ten-second
response allowance; later frames retain the one-second deadline. Results older
than 250 ms are still discarded, including a slow first result.

Restart the backend after updating, then enable the camera again. If it still
fails, the error identifies model startup (90 seconds) or the frame number.
`data/hand-tracker.log` records startup stages and timings, first-frame completion,
and worker exception tracebacks. A timeout alone does not establish that CUDA
or the model installation is broken.

Weights and MANO assets have separate noncommercial/research terms; see the
[WiLoR license declaration](https://github.com/rolpotamias/WiLoR#license),
[AnyHand repository](https://github.com/chen-si-cs/AnyHand), and
[MANO license](https://mano.is.tue.mpg.de/license.html). Assets are not committed.
