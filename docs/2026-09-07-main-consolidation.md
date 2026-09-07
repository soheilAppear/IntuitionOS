# Main branch consolidation and bug fixes — 2026-09-07

The repository had three branches: `main` at `564dae4`, the current application
on `phase-0-hygiene` at `c756848`, and the original prototype on
`copilot/add-intuition-model-os` at `9321495`.

The consolidation retains both development histories through a merge commit.
The supported root application is the current local Ollama/HUD implementation.
All 21 files of the earlier prototype are preserved under
[`archive/prototype-2025`](../archive/README.md); its conflicting cloud-model
configuration, packaging, launchers and documentation do not replace the current
application. Original prototype content is retained, including its limitations.

## Bugs reproduced and fixed

- PowerShell command discovery had a 15-second timeout. GitHub CI run
  [34153378041](https://github.com/soheilAppear/IntuitionOS/actions/runs/34153378041)
  failed three discovery tests on each Windows Python version, while the Linux
  jobs and evaluation passed. Cold discovery now allows a bounded 120 seconds,
  retains the full installed-command scan, and reports actionable timeout errors.
  Warm discovery is cached; changing `PSModulePath` or `PATHEXT` invalidates it.
  This does not change shell action execution timeouts. Full installed-command
  enumeration follows [PowerShell Get-Command behavior](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/get-command).
- A shell catalog containing a JSON array, scalar or null raised an unexpected
  attribute error. It now receives an explicit validation error.
- Snoozing an already-fired reminder left it fired and could leave its due time
  in the past. Snooze now reactivates it and delays from now or its future due
  time. Missing reminders and zero durations report errors. Undo restores exact
  due/status values and remains compatible with older delta-based journal rows.
- Missed repeating reminders advanced one interval per poll and could replay
  days of notifications/payloads. They now advance to the next future occurrence
  in one step, preserving their interval cadence.
- Two processes could deliver the same reminder after reading the same pending
  row. A conditional SQLite update now claims the occurrence before callbacks.
  Stale rows cannot bypass a user's snooze, and a callback's completion is not
  overwritten by a later scheduler status update.
- Valid JSON with the wrong payload shape could block the whole reminder queue.
  Such payloads are ignored with a diagnostic and the reminder still fires.
- Stopping the scheduler left its polling thread asleep until the next tick.
  An event now wakes it immediately and shutdown joins the worker.
- Returning from the terminal left its current scheduler and anticipator alive.
  Cleanup now runs on normal exit, EOF, interruption and exceptions, reading the
  current workers after reload/forget replacements.

## Verification

The starting local suite passed 552 tests with one POSIX-only skip. The first
18 added regression cases all failed against that implementation, independently
reproducing the defects above. The focused suites passed after the fixes.

Final local Windows verification: **572 Python tests passed, 1 POSIX-only test
skipped**, and **17 renderer tests passed**. JavaScript syntax and generated
capability manifest checks passed. The prediction/tool-loop gate passed (12/12
plans), and the chronological correction gate passed (96.2% top-1, 100% top-3,
zero changes to 14 valid inputs on the synthetic fixture). The existing
Starlette/httpx deprecation warning remains. Remote CI is checked separately
before advancing `main`.

## Remaining boundaries

Tests establish the covered behavior, not an absence of all possible bugs.
Reminder delivery is at most once per occurrence: a crash after the database
claim can miss the notification; fired one-offs remain visible in open tasks.
Repeat intervals retain the existing fixed-duration semantics. Physical speech
recognition and real hardware still require checks with the intended devices.
The action gate is not operating-system process isolation.
