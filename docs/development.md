# Working on IntuitionOS

Read [the architecture guide](architecture.md) for component ownership and
[the resolver reference](command-resolution.md) for command-review contracts.
The [README](../README.md) is the user-facing setup and command guide; dated
reports in `docs/` describe measured results rather than API specifications.

## Set up and run

Python 3.10 or newer is required. Use the project's virtual environment so the
launcher, tests and optional voice packages see the same dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
cd ui
npm install
cd ..
```

`requirements-dev.txt` includes application dependencies and the Python test
tools. Electron is installed locally under `ui/node_modules`. There is no web
bundler or separate browser development server.

```powershell
# Terminal, with its own Python runtime
.\.venv\Scripts\python.exe intuitionos.py

# Backend plus Electron HUD
.\.venv\Scripts\python.exe start_ui.py

# Renderer only, when this project's backend is already running
cd ui
npm start
```

Run project commands from the repository root except the renderer-only npm
command. Backend port `7432` must be free before `start_ui.py` starts. The HUD
toggle is Alt+Space; Ctrl+Q exits. Start Ollama separately for model-assisted
requests and install the model named in `config/config.yaml`. Correction,
ordinary built-ins and the automated test fixtures do not require Ollama.

For PowerShell aliases/functions, run the snapshot launcher from the intended
session: `./scripts/start-intuition.ps1 -Interface terminal` or `-Interface hud`.
Otherwise launching from a PowerShell prompt still uses the default CMD command
environment. See the resolver reference for snapshot limitations.

## Make changes at the right boundary

| Desired change | Start here | Preserve |
|---|---|---|
| Command spelling/ranking or availability | `core/command_resolver.py` provider and resolver contracts | Exact commands, namespaces, argument spans, no candidate execution during discovery |
| Windows intention vocabulary | `core/os_intents.py` | Existing app phrase precedence without scanning arbitrary shell arguments as intentions |
| New operation | `core/actions.py` and `core/capabilities.py` | Schema validation, actor, reversibility, confirmation, path scope, journal |
| A new shell environment | `core/shell_environment.py` plus its resolver provider | Discovery and execution describe the same environment |
| HUD interaction or messages | `ui/renderer/app.js` and `interface/server.py` | Server-authoritative tokens, original/selected text binding, disconnect/edit invalidation |
| Terminal interaction | `interface/terminal.py` | Rendered choice is the submitted choice, literal argument display |
| History or persistence | Owning feature store plus `core/memory.py` | Additive migration, locked access, retention/forgetting behavior |
| Background behavior | `core/anticipator.py` or `core/scheduler.py` | Worker lifecycle and actor-specific gate restrictions |

Keep comments close to the code whose invariant they explain. A useful comment
states why a shortcut would be incorrect: why argument text cannot be rebuilt,
why metadata cmdlets are module-qualified, or why an old worker must stop after
forgetting. Function/class docstrings describe input/output contracts and who
owns state. Avoid comments that simply repeat an assignment or name.

The readability pass uses conventional multiline Python formatting. No formatter
is added to runtime dependencies. Follow the surrounding style and avoid
unrelated whole-repository formatting in feature patches.

## Add a capability

1. Write a small implementation that returns its result. Actions commonly return
   dictionaries; directory queries can return lists. Expected failures should be
   explicit. Do not print UI output from the implementation.
2. Register it with the callable registry **and** a `Capability` entry. The `_cap`
   helper in `core/actions.py` does both. OS wrappers are registered through
   `register_os_capabilities`; ensure the intended interface actually calls the
   registration path.
3. Declare the complete argument schema, reversibility and confirmation policy.
   For path-bearing arguments, declare both `path_scope` and `path_args`. Add
   argument-dependent validation/confirmation where a driver requires it.
4. For automatic undo, supply `undo` and a capture hook. Capture previous content
   before a write; capture generated IDs from the result when they do not exist
   beforehand. A reversible classification alone does not create undo support.
5. Dispatch through `actions.call` for an explicit user request or
   `actions.dispatch` with the actual actor. Never call the implementation
   directly from the UI, model, scheduler or anticipator. Correction ranking must
   never be passed as execution confidence.
6. Add behavioral coverage for the relevant gate, confirmation, journal and
   platform behavior. Regenerate the model's advertised manifest:

```powershell
.\.venv\Scripts\python.exe -m core.gen_manifest
git diff -- config/planner_schema.json
```

`config/planner_schema.json` is generated data. Change the capability definition
first rather than manually editing the generated tool list. The existing
manifest-completeness test detects registry/advertisement drift.

## Add a provider or change correction

Use `CandidateProvider.commands(context)` to expose names that really exist in
the execution environment. The [verified provider example](command-resolution.md)
shows the minimal implementation without running subprocesses.

Keep availability discovery separate from ranking, UI selection, and execution.
An exact command wins before ranking. Apply a subcommand edit only when a
provider understands that command's grammar. Preserve the prefix and suffix of
the chosen string span verbatim; do not tokenize and rejoin the command line.

Any UI change that edits the draft must invalidate its previous resolution and
pending approval. Test the real race—editing or cycling a selection and pressing
Enter before a redraw—not merely a helper method in isolation. Correction
acceptance and execution outcome remain separate database fields.

## Test locally

These commands match the repository's main checks:

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q
node --test tests/hud_renderer_test.cjs
node --check ui/renderer/app.js
.\.venv\Scripts\python.exe -m eval.check
.\.venv\Scripts\python.exe -m eval.command_resolution --check
git diff --check
```

CI runs Python tests on Windows and Ubuntu with Python 3.10 and 3.12, plus the
renderer tests under Node 24, a generated-manifest check, and evaluation gates.
Local Windows success does not certify another OS or an unobserved CI run.

| Area | Existing coverage |
|---|---|
| Resolver, providers and private feedback | `tests/test_command_resolver.py` |
| Actual shell round trips and gate binding | `tests/test_command_execution.py` |
| Prompt rendering and terminal dispatch | `tests/test_terminal_resolution.py` |
| WebSocket state, confirmation and forgetting | `tests/test_hud_resolution.py` |
| HUD keyboard/rendering behavior | `tests/hud_renderer_test.cjs` |
| Component wiring and fresh reminder startup | `tests/test_startup.py` |
| Gate policy, journal and reversal | `tests/test_capabilities.py`, `tests/test_journal.py` |
| Memory use and learning | `tests/test_episodes.py`, `tests/test_retrieval.py`, `tests/test_predictor.py`, `tests/test_calibration.py`, `tests/test_consolidation.py` |
| Workers and model loop | `tests/test_scheduler.py`, `tests/test_anticipator.py`, `tests/test_brain_loop.py` |

Use the existing `project`, `memory`, and `wired` fixtures for temporary
directories/databases and a real registry/journal. The startup fixtures disable
voice and hardware so tests do not download a model or operate real devices.
When testing a dangerous capability, replace the implementation with a harmless
stub while retaining the real gate and confirmation flow. Stop any workers a
test creates before releasing their dependencies.

Tests for `Memory`'s raw helpers are not a substitute for startup coverage:
the scheduler must receive that memory object in the actual interface bootstrap.
Keep initialization order and reload/forget/shutdown transitions represented in
integration tests.

## Evaluate and show the result

```powershell
.\.venv\Scripts\python.exe -m eval.run --json
.\.venv\Scripts\python.exe -m eval.command_resolution --json
.\.venv\Scripts\python.exe -m eval.command_resolution --showcase
```

Prediction evaluation compares the old heuristic, learned predictor and
calibrated predictor. Correction evaluation compares the old slash matcher, the
deterministic resolver and personalized ranking on the same fixture. Both use
chronological evidence. Report synthetic data as synthetic, include valid-input
preservation, and distinguish warm resolution timing from startup and execution.

The optional showcase script renders the actual HUD with a fixed response:

```powershell
.\ui\node_modules\electron\dist\electron.exe scripts/showcase-hud.cjs
```

It uses a hidden offscreen window, blocks WebSocket traffic, and does not execute
the displayed command. It writes to the default dated PNG path used by the
milestone report; pass a different output path for a new report so historical
evidence is retained. An `ELECTRON_RUN_AS_NODE` environment setting prevents the
script from using Electron's app runtime; launch it from an ordinary shell or
clear that variable only for the showcase process.

Before delivering, inspect the diff, run checks appropriate to the change,
record measured results and limitations, and keep source, generated manifest,
README and examples consistent. Publication follows the user's requested branch
and workflow; a successful local test run is not proof of a remote CI result.
