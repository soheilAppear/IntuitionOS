# Hand tracker selection — September 2026

## Selected model

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

Pinch thumb and index, then release for a click. Hold the pinch for about 0.35
seconds to drag, and separate them to drop. Index bend clicking is off by default
so ordinary finger flexing does not compete with pointing. To enable it, stop the
camera, select **Index bend click (optional)** in Hand controls, and click **Apply**.
The option lasts until the backend restarts. Its guide appears only when the
backend confirms it is enabled.

Brief tracking uncertainty freezes the pointer for up to 150 ms, allowing the
same hand to recover without repeating the activation pose. Missing or unreliable
input cancels a pending click and releases a drag immediately. Recovery itself
never clicks: release any pinch before making a new click gesture. A different
hand or a longer interruption requires pointing again.

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
