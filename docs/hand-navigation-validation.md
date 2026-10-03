# Hand navigation validation

The automated tests exercise synthetic landmark shapes, timed gesture sequences,
fake Windows input, and the renderer. A separate controlled check exercised real
Windows navigation. Neither measures a person's camera tracking accuracy or
establishes how smooth the gesture feels.

## Run the automated checks

From the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_gestures.py tests/test_hand_navigation.py tests/test_desktop.py tests/test_hand_control.py tests/test_hand_mouse.py tests/test_hand_protocol.py tests/test_hand_window_actions.py tests/test_startup.py
node --test tests/hud_renderer_test.cjs tests/camera_preview_test.cjs tests/desktop_visibility_test.cjs
```

These checks replace camera and input calls with fakes. They cover the following
behavior; a passing run should not open a camera, change desktops, or click.

| Area | Expected behavior |
|---|---|
| Activation | Four fingers and a steady 0.30-second hold are required; ordinary mouse pointing does not navigate. |
| Continued movement | Three visible extended fingers can sustain an armed swipe. The first clear movement locks the horizontal or vertical axis. |
| Completion | Reaching 100% starts a 0.10-second dwell. Progress down to 90% keeps it alive; reversing below 90% resets it. |
| Repetition | Lowering the hand or holding a fist for 0.20 seconds resets a completed/cancelled gesture. Holding another pose cannot repeat navigation or become a Desktop-mode window action. Hand Mouse can also use a fresh point to return to pointer activation. |
| Tracking loss | Missing/uncertain input freezes progress for at most 150 ms, without advancing completion. A distant return or longer loss cancels. |
| Cancellation | Fist, pinch, camera stop, stale input, and input failure cannot commit an unfinished swipe or fall through to a click/close request. |
| Windows input | Horizontal native movement interpolates at 60 Hz; measured horizontal fallback and both vertical directions issue at most one shortcut at completion. |
| Cleanup | Failed release remains visible and retryable. Restart/settings stay blocked until native and tracker cleanup succeeds. |
| HUD | Signed travel, completion dwell, cancellation, and rearm hints match the backend without submitting a command or changing a draft. |

The integration traces in `tests/test_hand_navigation.py` pass synthetic hand
landmarks through the recognizer and `HandControls` into the desktop controller,
with only OS input replaced. They cover all four directions at 15, 30, and 60
frames per second, so state-machine success alone is not the completion oracle.

## Expected event sequence

For a synthetic leftward swipe with wrist anchor `(0.50, 0.50)`, palm scale `0.10`,
and travel setting `1.2`, the horizontal completion point is `x=0.38`.
Negative progress represents left/up movement. A leftward hand swipe requests
the next desktop, to the right of the current one.

| Input | Expected progress / motion |
|---|---|
| Four raised fingers, held still for 0.30 seconds | `arming` → `armed`; no Windows input yet. |
| Move left past the 0.25-palm dead zone | `desktop`, `axis=horizontal`; one motion `begin`, followed by `update` records. |
| Reach `x=0.38` | `committing`, `progress=-1`; completion dwell starts. |
| Small tremor to `x=0.386` (95%) | `committing` remains active; no extra action. |
| Keep a valid hand through the 0.10-second dwell | One motion `end`; after successful input, `completed` with a rearm hint. |
| Keep four fingers raised | No second motion `begin` or shortcut. |
| Lower the hand or make a fist for 0.20 seconds | Ready for a new activation; in Hand Mouse, point again to resume the pointer. |

Reversing from 100% to 80% during the dwell must return to ordinary signed
movement without committing. A fist or pinch then produces `cancel`, and no
shortcut. A brief missing frame must report `uncertain` with the same axis and
frozen signed progress. Recovery must not use the missing time to finish.
If the Windows callback fails, cancellation/error takes precedence over
`completed`; the HUD must not claim the action succeeded.

## Live check on Windows

Restart the backend and HUD after updating. Start with **Hand Mouse**, default
**Hand travel to 100% = 1.2**, and **Show camera preview** enabled. Create a second
virtual desktop if needed. Use the preview to distinguish a missing hand skeleton
from a recognized hand with the wrong pose. Test one movement at a time.

1. Point to move, relax your fingers, then pinch/release to click and hold a pinch
   to drag. Confirm the existing pointer controls still work.
2. Raise four fingers, keep your thumb comfortably apart from the index, and hold
   still until ready. Move left slowly. The pointer should pause, the meter should
   move left, and the desktop should finish after the short hold at 100%.
3. Keep the same pose raised. There should be no second switch. Lower the hand for
   0.20 seconds, then repeat in the opposite direction.
4. Repeat with a slightly relaxed pinky after activation. Repeat a partial swipe
   and reverse before completion. Make a fist or pinch to cancel; neither should
   click or close an application. In Desktop gestures, hold the cancelling pinch
   or a pointing/two-finger/thumbs-up pose: no window action should follow until
   you lower the hand or make a fist to reset, then start a fresh pose.
5. Make an upward gesture to open Task View. The meter should progress, then one
   shortcut should toggle the overview after completion. In Hand Mouse, lower
   your hand briefly, point to resume the pointer, then pinch/release on a desktop
   thumbnail to select it. A downward gesture
   should show the desktop; a new completed downward gesture restores windows.
6. Briefly obscure the hand during a partial swipe, then return nearby. Progress
   should freeze and resume without jumping. Keeping the hand out of view longer
   than 150 ms or returning far away should cancel.
7. With the camera off, select **Measured steps**, apply, and repeat. Desktop
   content should stay still until completion, then switch once. Return to
   **Smooth when supported** to compare horizontal movement.
8. Stop the camera mid-swipe. Input must release, the preview must stop, and no
   action should fire later. If **Retry cleanup** appears, use it before starting
   again or changing settings.

For native horizontal movement, Windows' **Settings → Bluetooth & devices →
Touchpad → Four-finger gestures** must assign sideways movement to desktops.
Up/down uses Win+Tab/Win+D independently of that setting. Windows controls the
native animation and final snap; camera FPS and tracking jitter still affect feel.

## What has and has not been verified

Release checks on 2026-10-03 passed against a clean export of the exact staged
source: **1,863 Python tests passed**, one shell-specific test skipped because
`sh` is unavailable on this Windows machine, and **93 Node tests passed**.
Whitespace checks were clean.

A controlled Windows input check on this development machine verified:

- A native horizontal swipe moved to the expected neighboring desktop among six
  desktops. The Windows desktop identifier confirmed the change; an explicit
  shortcut restored the original desktop afterward.
- Win+Tab displayed Task View with all six desktops and their windows in a local
  screenshot. The overview was then toggled closed.
- Win+D made the desktop visible, with `Progman` as the foreground window. Toggling
  it again restored the same original foreground window handle.

The local evidence is in ignored `data/native_navigation_smoke.json` and
`data/task-view-live-check.png`; these files remain private and are not repository
artifacts. Local WiLoR diagnostics also reached worker ready and completed a first
frame. These checks establish working OS actions and model startup, not a measured
live hand-navigation success rate.

The revised camera gesture feel, perceived smoothness, and recovery under a real
user's occlusions still need the live sequence above. No claim of Mac trackpad
smoothness follows from a successful desktop identifier change. Record the
selected tracker, movement mode, travel
setting, Windows build, approximate capture FPS, observed state/hint, and whether
the skeleton stayed visible when reporting a failure. Worker startup diagnostics
are in ignored `data/hand-tracker.log`; camera frames are not saved.
