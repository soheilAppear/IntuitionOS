# IntuitionOS v1.0

Local, model-aware shell that listens, anticipates, and helps while keeping your work local and safe.

## The story

Computers got fast. Interfaces did not. We still type exact commands, watch the cursor blink, and repeat little rituals to make simple things happen. That is fine when you have time and full attention. It is painful when you are building, testing, and juggling ideas at the speed you think.

IntuitionOS started from that pain. The goal was simple: build a local, model-aware shell that behaves like a careful lab partner. It listens, anticipates, and helps. It does not nag. It does not steal control. It bends toward your intent even when your input is imperfect.

The first seeds came from day to day research work. Rapid VR and AR prototyping needs a loop like this:

- Capture an idea or a todo the moment it appears
- Run a quick script, log the result, and schedule a follow up if needed
- Tweak hardware in a safe sandbox without switching tools
- Try commands that are half typed or slightly wrong and still get the right action

Traditional shells are literal. Chat assistants are helpful but detached from your local machine. IntuitionOS tries to land in the middle. It feels like a shell that thinks with you, not for you.

## Why this exists

### Intent beats syntax
You should not lose flow because you typed `/hlp` instead of `/help`. The system should meet you where you are.

### Anticipation reduces wait time
While you type, the system can warm up likely results like directory listings or a file preview. You press Enter and it responds instantly. No flashing output until you confirm.

### Local by default
Models run on your box through Ollama. Your files stay local. You decide what can execute and where.

### Memory that matters
Notes and recalls are built in. Save a thought now, find it later. Schedule reminders with natural time.

### Safe edges
Execution is sandboxed to the project folder. Safe Mode is on by default. You can turn it off with a clear command, then turn it back on just as easily.

## What makes IntuitionOS different from a traditional OS

- A traditional OS gives you primitives and expects precision. IntuitionOS wraps the same primitives with a model guided layer that tolerates typos, guesses intent, and precomputes likely next steps.
- A traditional OS has a strong separation between shell, scheduler, and device tools. IntuitionOS keeps them in one conversational surface so you can say things like “remind me test build in 15m” and it just schedules it.
- A traditional OS executes exactly what you type. IntuitionOS tries to help you avoid foot guns by keeping execution inside the current working directory and by keeping Safe Mode on until you say otherwise.

## What makes IntuitionOS different from chat assistants

- Chat assistants are great at explanations. They are weak at operating your machine. IntuitionOS is designed to operate locally. It writes files, runs scripts, lists directories, and calls safe hardware adapters.
- Chat assistants often require cloud access. IntuitionOS uses a local model via Ollama by default.
- Chat assistants optimize conversations. IntuitionOS optimizes short feedback loops. It is a tool for doing, not only discussing.

## Design principles

- **Local first** - Use local models and local files. Network use is optional and explicit.
- **Transparent safety** - Show what is about to execute. Keep execution inside the project. Make Safe Mode a first class toggle.
- **Small, predictable actions** - Expose a tiny set of clear actions: read, write, run, list, recall, schedule, hardware call.
- **Anticipate quietly** - Prewarm likely responses in the background. Reveal nothing until the user presses Enter.
- **Forgive typos** - Fuzzy match only the first token so that `/safe off` stays intact while `/hlp` still maps to `/help`.

## A quick scene

You open the terminal and type `tre`. IntuitionOS corrects to `tree`, warms the directory tree in the background, and shows it the instant you hit Enter. You say “remind me export figures in 40m”. It sets a reminder. You say “write file test.py with a small plot and run it”. It writes, then executes inside the sandbox. You say “set led color to ff8800”. The simulated driver reports state without touching any real hardware unless you opt in.

It feels like a shell that understands the rhythm of your work.

## Core capabilities at a glance

- Fuzzy commands that preserve arguments
- Background anticipator for `ls`, `tree`, and file previews
- Memory and recall with a tiny SQLite store
- Natural language reminders with a polling scheduler
- Sandboxed execution inside the project directory
- Safe Mode on by default and easy to toggle
- Local LLM via Ollama with a configurable model name
- Minimal hardware adapters to grow into real devices later

## Who is this for

- Builders who live in scripts and want a smoother loop
- Researchers who prototype quickly and need frictionless notes and reminders
- Tinkerers who want model help without giving up local control
- Anyone who likes the shell but wishes it felt a bit more like a thoughtful teammate

## What this is not

- A full desktop replacement
- A general chat portal
- A license to execute arbitrary commands across your system

It is a focused layer that makes the things you already do feel faster and more humane.

## Roadmap ideas

- Per project memory with tagging and export
- Smarter task recurrence and time zone handling
- Optional remote device adapters guarded by explicit allowlists
- A small UI viewer for plans and logs next to the terminal

## Credits

- Built for local use with Ollama
- Thanks to everyone who pushed on fuzzy parsing, safe exec, and background prewarming ideas

## Highlights

- Fuzzy slash commands. `/hlp` becomes `/help`. Only the first token is corrected.
- Anticipator. Prewarms likely results while you type. Shows nothing until you press Enter.
- Safe Mode. On by default. Toggle with `/safe on` or `/safe off`.
- Sandboxed exec. `/exec "python script.py"` runs inside this directory only.
- Local LLM via Ollama. Default model is `gpt-oss:20b`.
- Memory and recall. `/save "note"`, `/recall "term"`.
- Tasks. `remind me drink water in 10m`, `/tasks`, `/done <id>`.
- Files. `/write <path> "text"`, `/read <path>`, `ls`, `tree`.
- Hardware demo. `plugins/led_strip.py` and GPU info plugin. All safe and simulated by default.

## Quick start

```powershell
cd IntuitionOS_v1.0
python -m venv .venv
.\.venv\Scriptsctivate
pip install -r requirements.txt
$env:OLLAMA_HOST = "http://127.0.0.1:11434"
python .\intuitionos.py
```

If `ollama serve` is already running on 127.0.0.1:11434 you are good.

## Safe Mode

Safe Mode blocks `/exec` and any write action that touches the file system unless you use the explicit built ins. Use:

```
/safe off
/exec "python -V"
/safe on
```

You can also start with Safe Mode off:

```powershell
$env:INTUITION_SAFE = "0"
python .\intuitionos.py
```

## Commands

- `/help`
- `/exit`
- `/memory`
- `/dream`
- `/save "text"`
- `/recall "term"`
- `/config`
- `/actions`
- `/reload`
- `/tasks`, `/done <id>`, `/delete <id>`, `/snooze <id> 15m`
- `/hw`, `/hw schema <name>`
- `/safe on|off`
- `/exec "python your_script.py" [cwd]`
- `/write <path> "text"`
- `/read <path>`
- `ls`
- `tree`

## Troubleshooting

- If you see `venv python not found at ...`, create `.venv` as shown above or set: `$env:INTUITION_ALLOW_SYSTEM_PY = "1"` to permit system Python.
- If a command times out talking to Ollama, ensure `ollama serve` is running and the model is pulled: `ollama pull gpt-oss:20b`.
