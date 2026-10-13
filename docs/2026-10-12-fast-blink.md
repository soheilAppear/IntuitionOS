# Fast double-blink cleanup — October 12, 2026

Branch: `update_10_12_2026_eeg_signal_status`, continued from the previous EEG branch.

## Result

The EEG panel now exposes one workflow: a one-second fast double-blink CNN sends
one left click at the current cursor. Camera aiming is optional; pinch clicks
and dragging remain available. A visible hand is not required for an EEG click.

The retired eyes-closed, calibrated blink, automatic blink, five-second CNN,
EEG swipe, practice-counter, and finger-snap paths have been removed, including
their obsolete modules, commands, selectors, API routes, and documentation.
Applicable lifecycle, transport, and native-input regression cases were migrated
to the fast-only suite. Standalone Hand Mouse and desktop gestures remain intact.

The existing fast archive format and saved calibration remain compatible.
Weights, preprocessing, validation thresholds, and the double-blink-to-single-click
mapping are unchanged. Personal model archives stay in the ignored local data
directory and are not included in this branch.

## Camera and recovery changes

- Both camera previews share one renderer for all 21 hand joints and connections.
  The EEG camera uses the standalone tracker, model complexity, camera index,
  and optional bend-click preference.
- The yellow CAMERA indicator aggregates both native capture owners, separately
  from the HAND ON/OFF control. It remains visible through pending or failed
  release, hidden panels, stale status, and disconnected status channels. OFF
  requires confirmed release. Revision and backend-instance checks reject late replies.
- A failed native camera release retains the capture object for cleanup retry
  and blocks the other mode from acquiring the camera.
- Temporary excessive EEG amplitude pauses only blink output. Recovery discards
  pending commands and requires fresh clean EEG plus rest. Other input, packet,
  clipping, freshness, and Escape-lease failures still disarm.
- All queued Start requests, including EEG-only starts, carry cancellation epochs.
  A later Stop cannot be undone by an earlier request waiting for the device lock.
- Inline comments explain signal processing, state transitions, cancellation,
  ownership, model persistence, and final input authorization.

The branch also retains the preceding laptop configuration and performance work:
the compact Qwen3 4B Ollama default with an 8,192-token context, MediaPipe Full
camera default, fresh-frame capture/control scheduling, and BrainBit connection
timeout/status improvements.

## Verification

| Check | Result |
|---|---|
| Complete Python suite | 2,473 passed; 5 platform-dependent tests skipped |
| Complete JavaScript suite | 245 passed |
| Real-service/renderer contract | Camera-on/off stopped snapshots validated in both languages |
| Existing local fast archive | Loaded successfully with unchanged weights and original held-out validation |
| Native HUD inspection | Fast-only controls and confirmed CAMERA OFF displayed correctly |
| Live headset check after cleanup | Skipped at the user's request; no new live recognition claim |
| Whitespace/diff validation | Passed |

Commands used:

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q
$jsTests = rg --files tests -g '*_test.cjs'
node --test --test-isolation=none @jsTests
git diff --check
```

Tests cover amplitude pause/recovery without delayed click replay, stale and
malformed EEG, packet discontinuities, saved-model validation, hand loss,
finger/EEG ownership, failed button and camera release, camera-enable/Stop races,
expired guards, late image decoding, hidden panels, and out-of-order status.
They use fake actuators and sensors; automated tests generate no desktop clicks.

One dependency deprecation warning remains from FastAPI's Starlette test client
using `httpx`; it does not fail the suite. Trial classification and synthetic
regressions do not establish continuous real-world accuracy or physical click latency.

See the [usage guide](brainbit.md) and [hand-tracking guide](hand-tracking.md).
