# Command resolution contributor reference

The terminal and HUD share [core/command_resolver.py](../core/command_resolver.py).
The resolver proposes a precise text replacement; `CorrectionSession` binds the
user's selection to that displayed text; the existing capability gate separately
decides whether the resulting action may run. Resolution and ranking do not use
Ollama. For measured results and a UI showcase, see the
[milestone report](2026-09-07-command-resolution.md).

## Provider and result contracts

A provider implements `commands(context=None) -> Iterable[Command]`. Each
`Command` declares its `name`, `namespace`, `kind`, and `case_sensitive` flag.
Providers describe names available in their execution environment. They must not
run candidate commands to check whether they work. A future application or file
provider can implement the same protocol, but its execution handler and
capability declaration must be added separately.

This complete example uses a declared demonstration vocabulary. It performs no
command discovery, starts no subprocess, writes no database, and executes no
selected command. Run it with the project Python from the repository root:

```python
from core.command_resolver import Command, CommandResolver, CorrectionSession


class DemoProvider:
    def commands(self, context=None):
        yield Command("python", namespace="shell", kind="executable",
                      case_sensitive=True)


resolver = CommandResolver([DemoProvider()], shell="sh")
session = CorrectionSession(resolver)
raw = '  pyhton\t"train file.py"  --lr 0.001  '

result = session.update(raw)
candidate = result.candidates[0]
assert result.status == "correction"
assert candidate.span == (2, 8)
assert candidate.text == '  python\t"train file.py"  --lr 0.001  '

# An interface renders this snapshot before allowing the user to select it.
shown = session.snapshot()
selected = session.commit(
    raw, token=shown["token"], revision=shown["revision"], candidate_index=0,
)
assert selected == candidate.text
print(result.status, candidate.token, candidate.span)
print(repr(selected))
# No dispatch here: commit returns a string, not an execution permission.
```

Expected output:

```text
correction python (2, 8)
'  python\t"train file.py"  --lr 0.001  '
```

Use `create_default_resolver(memory=None, enabled=True, shell=None)` for actual
environment discovery. Passing the application's `Memory` enables correction
feedback. `enabled` may be a boolean or a callable, such as
`lambda: episodes.enabled`. `context` may be a dictionary or the project's
`Context` object: the built-in providers use `cwd`, and feedback ranking uses
`cwd` and optional `ts`. A controlled `ts` allows chronological evaluation.

`Resolution.to_dict()` contains `original`, `status`, `candidates`, `reason`,
`span`, and `namespace`. A candidate contains `text`, `token`, `namespace`,
`score`, `reason`, and `span`.

| Field or status | Meaning |
| --- | --- |
| `original` | Entire input, unchanged. |
| Candidate `text` | Entire proposed command, including the original argument suffix. |
| Candidate `token` | Replacement command name or understood subcommand. |
| `span` | Half-open Python string indices `[start, end)` in the **original** input; `None` when no safe span is established. JSON serializes a tuple as an array. |
| `namespace` | `intuitionos`, `shell`, `git`, or `unknown`; it scopes matching and informs interface routing. |
| `exact` | Recognized valid input; no correction candidates. It does not certify that execution will succeed. |
| `incomplete` | More command text is needed, or a command-name prefix has completion candidates. |
| `correction` | Plausible spelling replacement with a clear highest-ranked candidate. |
| `ambiguous` | The leading candidate scores are close; the user can select an alternative or keep the original. |
| `unsupported` | No plausible available command or no safely understood syntax. Read `reason`; no replacement is made. |

Replacement construction is exactly
`original[:start] + replacement + original[end:]`. Do not split and rejoin the
line in a caller: that loses quoting and whitespace. The replacement may be
longer or shorter than the old span; highlight its new length when rendering
`candidate.text`.

Ranking first preserves exact names under their declared case rules. It then
uses optimal-string-alignment distance, including adjacent transpositions.
Prefix completion has its own baseline score. Explicit acceptance history adds
bounded frequency, recency, repeated-typo, and project-context weights. By
default, at most three candidates are returned, sorted deterministically.
`score` is a ranking weight, can exceed 1, and is **not a probability**. The
prediction calibrator is not used to reinterpret these scores as confidence.

## Display, selection, and execution are separate steps

`CorrectionSession` holds one interface's current review state:

1. `update(text, context)` resolves a draft and records a display event when it
   has candidates and logging is enabled. Present its result to the user;
   background exploration should call `resolver.resolve()` instead.
2. `snapshot()` adds an opaque `token` and increasing `revision` to the result.
   Repeated `update()` with the same still-valid draft reuses the snapshot.
3. `commit(original, token=..., revision=..., candidate_index=...)` returns the
   selected full text. A zero-based index chooses a displayed candidate;
   `None` explicitly keeps the original. It checks the exact original text,
   token, revision, index, and feedback generation. A stale or reused commitment
   raises `ValueError`.
4. The interface routes that text to an action through `actions.dispatch()` or
   `actions.call()`. A shell command reaches `run_command`; the resolver itself
   never dispatches it.
5. `outcome(category)` records what happened separately from acceptance. A
   parked action is `pending`; its later result can be `ok`, `error`, `denied`,
   or `cancelled`. The store also accepts `undone` and `unsupported`; unrecognized
   categories become `unknown`.

An input edit must invalidate the displayed selection before it can be reused.
`invalidate()` removes the usable token and advances the revision; it does not
delete historical feedback. An uncommitted manual edit is recorded as ambiguous
evidence at the next `update()`. Keep a separate session per input surface.

The [terminal prompt](../interface/terminal.py) renders the selected full command
in its toolbar. Ctrl+N/P cycles choices, Escape selects the original, and Enter
only commits text and a selection that have actually been rendered. Enter after
a rapid unseen edit requests a redraw first. If the capability gate requires
confirmation, `run_action()` shows the literal action arguments and asks
`proceed? [y/N]`.

The [HUD renderer](../ui/renderer/app.js) shows highlighted candidate rows and a
Keep original row. It also supports Up/Down. Typing, including clearing the input,
immediately informs the backend and clears local suggestions and approvals.
Voice transcription fills the editable draft; it does not submit by itself.
The HUD's action confirmation bar uses a separate token and Enter/Escape or its
buttons. [The server](../interface/server.py) binds that token to its WebSocket
and input revision, and cancels pending actions on edits and disconnects.

`run_command` is an irreversible capability: Safe Mode must be off, and a
separate confirmation is still required. Confirming rechecks current gate
conditions. The action uses the stored arguments, not replacement arguments in
a confirmation message, and the journal records its execution outcome. A
correctly selected command that exits unsuccessfully remains an accepted
interpretation with an error outcome.

## HUD WebSocket example

The endpoint is `ws://127.0.0.1:7432/ws`. The connection first receives a `status`
message. The following exchange uses `/hlep` because its correction does not
depend on which external executables are installed. Tokens below are symbolic:
clients must echo the actual server-issued value.

Client draft:

```json
{"type":"buffer","text":"/hlep","client_revision":1}
```

Server resolution on a fresh connection with no correction history:

```json
{
  "type": "resolution",
  "original": "/hlep",
  "status": "correction",
  "candidates": [{
    "text": "/help",
    "token": "/help",
    "namespace": "intuitionos",
    "score": 0.8,
    "reason": "Spelling distance 1 (adjacent transpositions count as one)",
    "span": [0, 5]
  }],
  "reason": "Choose a displayed suggestion or keep the original.",
  "span": [0, 5],
  "namespace": "intuitionos",
  "token": "SERVER_RESOLUTION_TOKEN",
  "revision": 1,
  "client_revision": 1
}
```

After rendering and choosing suggestion 0, the client submits:

```json
{"type":"input","text":"/hlep","selected_text":"/help","token":"SERVER_RESOLUTION_TOKEN","revision":1,"candidate_index":0,"client_revision":1}
```

The server returns a `reply` containing help. To keep the original instead,
submit `candidate_index: null` and `selected_text: "/hlep"`; the unchanged
unknown slash command produces an `error`. Never overwrite `text` with the
candidate: `text` binds the review to the original buffer.

`client_revision` is the renderer's edit counter, echoed so it can discard late
responses. `revision` is the session's server-issued revision. They serve
different purposes and need not stay equal. `resolve` accepts the same draft
fields as `buffer` to request a review without submitting, which the renderer
uses if Enter arrives before a current snapshot.

For a shell command, an allowed review may instead lead to `confirm_request`
with `token`, `capability`, validated `args`, `reason`, `reversibility`, `summary`,
and `client_revision`. Reply using that **confirmation** token:

```json
{"type":"confirm","token":"SERVER_CONFIRMATION_TOKEN","granted":true}
```

Use `false` to decline. Editing the input makes a pending approval unusable;
another connection cannot approve it. Server-driven resets send
`{"type":"input_invalidated"}`. Legacy clients can still submit exact raw
commands without a resolution token, but a likely typo receives a review
response and is never silently corrected at submission.

## Feedback and retention

`CorrectionFeedbackStore` adds tables to the existing SQLite database without
replacing notes, tasks, or earlier schemas:

| Table / fields | Stored meaning |
| --- | --- |
| `command_corrections.id`, `ts`, `namespace` | Display-event identity, time, and command namespace. |
| `original_token`, `candidates_json` | Original correction token and displayed candidate tokens, namespaces, and scores. No full command lines. |
| `project_key` | SHA-256 of the normalized absolute working directory, used for project preference matching. |
| `selected_token`, `selected_ts`, `accepted` | Explicit candidate selection; acceptance is `1` for a selected correction and otherwise `NULL`. |
| `manual_token` | An edited command token or understood Git subcommand; it does not automatically count as acceptance or rejection. |
| `outcome`, `outcome_ts` | Execution category and time, independently of selection. |
| `command_correction_meta.generation` | Invalidation generation advanced when correction records are forgotten. |

Only a restricted command-token character set is retained, capped at 100
characters; unsupported token forms become empty strings. Argument values,
quoted strings, flags, file paths, and command output are not correction-learning
fields. `learning_text()` provides argument-free command representations for
episode/context learning. This is not a promise that the entire application
retains no arguments: intentional notes, conversation memory, and the execution
journal have their own purposes, and the journal can contain real action args.

Ranking reads prior explicit acceptances with both display and selection times
at or before the query's `ts`. It does not treat ignored suggestions, manual
edits, or command failure as rejected interpretations. A disabled feedback store
neither records new events nor adds learned ranking boosts.

Set `episodes.enabled: false` in [config/config.yaml](../config/config.yaml) and
use `/reload` to apply logging changes. `/forget` deletes episodes, correction
evidence, and derived predictor/calibration/rule state. Interfaces replace their
live learning state; the HUD invalidates every connection on that backend and
revokes its parked approvals. A changed feedback generation also makes an older
`CorrectionSession.commit()` fail. Notes, tasks, and the audit journal remain.
For retention code, `EpisodeLog.forget_before(ts)` and
`CorrectionFeedbackStore.forget(before=ts)` remove records older than the cutoff;
the store's temporary `since` spelling has the same older-than meaning.

## Environment discovery and grammar limits

[core/shell_environment.py](../core/shell_environment.py) binds discovery and
execution to `INTUITION_SHELL`: CMD by default on Windows, `sh` elsewhere, or
explicit `powershell` / `pwsh`. The configured shell must be available. Starting
the app from a PowerShell terminal alone does not change the default CMD runner.

| Provider | Source and constraints |
| --- | --- |
| `IntuitionCommandProvider` | Registered slash-command vocabulary plus `ls` and `tree`. Existing application phrases are recognized by [core/os_intents.py](../core/os_intents.py). |
| `ShellBuiltinProvider` | Declared CMD or POSIX sh builtins with the appropriate case rules. |
| `PathExecutableProvider` | Filesystem-only PATH scan; Windows PATHEXT names and extensions, or executable files on POSIX. CMD also searches the working directory. Cache key includes PATH, PATHEXT, cwd, and shell; default TTL is 3 seconds. |
| `EnvironmentCatalogProvider` | A command catalog accepted only for the same execution shell; PowerShell metadata includes aliases, functions, cmdlets, and module-qualified names. |
| `InstalledGitSubcommandProvider` | The default factory's fixed `git --list-cmds=main,others,alias` metadata query, cached for 30 seconds. Git resolves its own aliases and included/conditional/worktree configuration. A failed query disables subcommand correction. |
| `GitSubcommandProvider` | Injectable fixed vocabulary and direct Git config reads, useful for controlled tests. It is not the installed-metadata provider used by the default factory. |

PowerShell discovery uses a qualified `Get-Command` metadata query in the same
`-NoProfile` setup as execution. It does not invoke the discovered candidate
functions. To include aliases/functions from the current PowerShell session,
launch through [scripts/start-intuition.ps1](../scripts/start-intuition.ps1),
with `-Interface terminal` or `-Interface hud`. It creates a temporary local JSON
catalog, sets `INTUITION_SHELL` and `INTUITION_SHELL_CATALOG`, and removes the
snapshot when the child exits. These are startup snapshots: restart after
changing definitions. Function definitions that depend on captured variables,
closures, or module state may need adaptation. CMD and sh do not import these
PowerShell catalogs.

Git correction understands the immediate subcommand in `git SUBCOMMAND ...`.
It does not search beyond leading global options such as `git -C path ...`,
rewrite flags or later arguments, or apply Git grammar to an alias/function
merely named `git`. External `git-*` commands can contribute subcommands.

The parser conservatively leaves shell operators, unsupported expansions or
escapes, unmatched quotes, quoted executable names, and environment assignments
unchanged. A quoted argument is not a correction target. Single quotes are not
quoting in CMD. Existing executable paths are preserved; nonexistent paths are
not repaired. PowerShell module-qualified corrections stay within the specified
module. Bare unsupported shell expressions are explained without dispatch;
unknown free text may still use the existing local LLM path.

Unquoted `/exec git status` uses the new full-command path. Legacy
`/exec "python script.py" [cwd]` retains its `run_local` parsing and project-venv
Python selection. Both paths go through capability controls. Argument text is
preserved by the resolver; the selected shell still determines native argument
semantics. In particular, Windows PowerShell 5.1 can drop empty native arguments.

## Tests and extension checklist

| Concern | Relevant coverage |
| --- | --- |
| Provider availability, namespaces, case, transpositions, spans, Git grammar, ranking, feedback and forgetting | [test_command_resolver.py](../tests/test_command_resolver.py) |
| Native shell arguments, catalog consistency, discovery without invoking shadowing candidates, gate rechecks, journal outcomes | [test_command_execution.py](../tests/test_command_execution.py) |
| Actual terminal render/keyboard state, stale edits, explicit original, action confirmation | [test_terminal_resolution.py](../tests/test_terminal_resolution.py) |
| Real WebSocket requests, candidate/argument binding, cross-connection tokens, voice drafts, reset and shutdown retention | [test_hud_resolution.py](../tests/test_hud_resolution.py) |
| Renderer highlights, keyboard choices, immediate edits, stale responses, voice draft behavior | [hud_renderer_test.cjs](../tests/hud_renderer_test.cjs) |
| Same-case old/baseline/personalized comparison and chronological learning | [test_correction_evaluation.py](../tests/test_correction_evaluation.py), [eval/command_resolution.py](../eval/command_resolution.py) |

When adding a provider, preserve valid existing names before ranking, declare
the actual shell's case and namespace rules, and keep arguments opaque unless a
dedicated grammar owns their correction span. Test discovery without executing
candidates, a cold-history typo, exact-name preservation, and argument fidelity.
When adding an interface, reuse `CorrectionSession` and test rendered-state
binding and action confirmation independently. Reuse the existing action gate;
a higher ranking score must never grant execution permission.

Focused checks from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_command_resolver.py tests/test_command_execution.py tests/test_terminal_resolution.py tests/test_hud_resolution.py tests/test_correction_evaluation.py
node --test tests/hud_renderer_test.cjs
.\.venv\Scripts\python.exe -m eval.command_resolution --check
```
