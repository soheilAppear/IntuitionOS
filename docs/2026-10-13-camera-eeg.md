# Camera + EEG update - October 13, 2026

Branch: `update_10_12_2026_eeg_signal_status`.

Camera + EEG now supports the same four-finger left/right virtual-desktop swipes
as camera-only mode, alongside pointer movement, pinch clicks, dragging, and
fast double-blink clicks. The saved fast CNN format and waveform preprocessing
remain unchanged; this update does not require new EEG calibration.

## Desktop navigation

With Camera mouse enabled and the shared controls armed, hold four fingers still
for 0.30 seconds, sweep horizontally, and hold at completion for 0.10 seconds.
Lower the hand or make a fist for 0.20 seconds before the next swipe. Movement
mode and travel distance come from the existing Hand controls settings.

Navigation pauses the pointer and suppresses blink clicks. Pending blink evidence
is discarded, and EEG input resumes only after fresh signal and recognized rest.
Horizontal movement switches adjacent desktops; vertical gestures do not trigger
Task View or show-desktop actions in this mode. Existing pinch/drag release takes
priority, and a held navigation pose cannot repeatedly switch desktops.

Stop, Escape, lost authorization, stale sensors, and tracking loss cancel owned
navigation. Cleanup failures remain visible and block input until release succeeds.
The same tracker and 21-joint overlay continue to serve both camera modes, and
the yellow camera indicator continues to reflect native capture ownership.

## EEG and readiness

The former raw 10 mV peak-to-peak cutoff has been removed from both trial collection
and live detection. The CNN still receives the same normalized waveform. Invalid
numbers, flatlines, repeated samples, clipping, packet gaps, stale input, and
expired Escape guards still stop input. Large movement or contact artifacts can
resemble blinks; normalization alone cannot ensure correct classification.

The UI now explains failed saved-model validation beside the arm control and
camera status, including the measured result when available. Camera motion and
EEG clicks continue to share one explicit arm and validation requirement.

## Model files stay local

This update publishes code, tests, configuration, and documentation only.
Pretrained networks, model weights, personal EEG examples, and saved calibration
archives are excluded from Git. The ignored `data/` directory retains local
models; model loading, training, and download/setup code remain available.

## Verification

- 497 targeted Python checks passed across camera protocols, startup, hand control,
  fast EEG lifecycle and quality, desktop navigation, and combined-mode integration.
- 250 JavaScript checks passed, including readiness and navigation feedback.
- Model compatibility was checked by loading saved fast CNN archives and
  recomputing their held-out validation without changing their weights.
- Simulated-input checks covered both swipe directions, blink suppression,
  pinch/drag release, tracking loss, failed cleanup, and Stop during navigation.
- Whitespace/diff validation passed.

Automated checks use fake sensors and actuators. No live desktop-swipe verification
was performed for this update; trial and synthetic results do not establish
continuous real-world recognition accuracy or physical blink-to-click latency.
One existing Starlette/httpx deprecation warning remains in the protocol tests.

See the [BrainBit guide](brainbit.md) and [hand-tracking guide](hand-tracking.md)
for setup and use.
