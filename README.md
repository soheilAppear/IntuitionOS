# IntuitionOS

> A local, model-aware shell that listens, anticipates, and helps — without leaving your machine.

IntuitionOS sits between traditional shells (exact, literal) and cloud AI assistants (helpful, but remote). It behaves like a careful lab partner: it tolerates typos, precomputes likely results while you type, remembers your notes, schedules reminders in natural language, and keeps execution safely sandboxed — all running on your own hardware via Ollama.

---

## Two interfaces

### HUD overlay (Electron)

A frameless, always-on-top ambient overlay that lives at the top of your screen. Press `Alt+Space` anywhere to summon or dismiss it. It expands when you interact and collapses to a minimal bar otherwise — an OS layer, not another app window.

- Dark glassmorphic panel — no window chrome, no taskbar entry
- Live memory and task panels (click ◎ and ≡ in the header)
- Cyan glow on `›` when the anticipator is running in the background
- Ghost hint shows the precomputed result before you press Enter
- Reminder toasts flash the HUD border when a scheduled task fires

### Terminal (classic)

A REPL with fuzzy command correction, Rich-formatted output, and the same brain and memory as the HUD.

---

## Quick start

### HUD overlay

```powershell
# 1. Clone and set up Python environment
git clone https://github.com/soheilAppear/IntuitionOS
cd IntuitionOS
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# 2. Install Electron (one time)
cd ui
npm install
cd ..

# 3. Start Ollama (optional — file and task features work without it)
ollama serve
ollama pull llama3        # or whichever model you set in config/config.yaml

# 4. Launch
python start_ui.py
```

Use `Alt+Space` to toggle the HUD. `Ctrl+Q` to quit.

### Terminal

```powershell
python intuitionos.py
```

---

## What you can do

### File system (instant, no Ollama needed)

```
ls
tree
/read config/config.yaml
/read requirements.txt
```

### Memory

```
/save "working on HUD overlay feature"
/save "Ollama model: llama3"
/recall Ollama
/recall HUD
```

Click **◎** in the HUD header to browse the full memory panel.

### Natural language scheduling

```
remind me check the build in 10m
remind me push to GitHub in 1h
/tasks
/done 1
/snooze 2 15m
/delete 3
```

The HUD border flashes and a toast appears when a reminder fires.

### Safe mode

```
/safe off
/exec "python your_script.py"
/safe on
```

Safe Mode is **on by default**. The green dot in the HUD header turns red when it is off.

### AI (requires Ollama)

```
what does the anticipator do?
explain the memory system
write a python function that reads a csv file
what are we building?
```

After `/save`-ing notes, the LLM uses them as context.

### Hardware

```
/hw
/hw schema led_strip
```

Drivers are simulated by default. See `config/config.yaml` to configure real hardware.

---

## What is recorded on your machine

IntuitionOS keeps an **episode log**: one row for every input you submit, stored
in `data/intuition.db` on your own disk. Each row holds what you typed, a snapshot
of the situation it arrived in (working directory, git branch and dirtiness, hour
of day, how long you paused before pressing Enter, the last few commands), and —
when the HUD showed you a hint — whether you took it or ignored it.

This is deliberate and it is the point. A system cannot learn that you run
`pytest` after `git commit` unless something records *after what*. The rows the
predictor learns most from are the ones where it was **wrong**: a hint shown and
ignored is the negative signal that keeps its confidence honest.

Two things are worth being explicit about:

- **Nothing leaves this machine.** There is no telemetry, no upload, and no
  hosted API in the path. The log is a table in a local SQLite file.
- **It records without being asked.** Unlike `/save`, you do not opt in per entry.

So it comes with an off switch and an eraser:

| | |
|---|---|
| See what has been recorded | `/episodes` |
| Erase all of it | `/forget` |
| Stop recording entirely | set `episodes.enabled: false` in `config/config.yaml` |

The action journal (`/journal`) is separate and smaller: it records only actions
that changed something, so that `/undo` has something to reverse and so that
"scoped exec" has an audit trail behind it.

---

## Anticipation

Type `tree`, `ls`, or `read file <path>` **slowly**. Watch the `›` glow cyan — the anticipator has already computed the result in the background. Press Enter and see `⚡ cached` in the response. No waiting.

---

## Commands reference

| Command | Description |
|---------|-------------|
| `ls` | List directory |
| `tree` | Recursive directory view |
| `/read <path>` | Read a file |
| `/write <path> "text"` | Write a file |
| `/save "text"` | Save a memory note |
| `/recall "term"` | Search memory |
| `/memory` | Show recent memory |
| `/dream` | Run a reflection pass |
| `/tasks` | List pending tasks |
| `/done <id>` | Mark task complete |
| `/delete <id>` | Delete a task |
| `/snooze <id> 15m\|2h\|1d` | Snooze a task |
| `/safe on\|off` | Toggle Safe Mode |
| `/exec "python script.py"` | Run inside sandbox |
| `/hw` | List hardware devices |
| `/hw schema <name>` | Show device schema |
| `/actions` | List all registered actions |
| `/config` | Print current config |
| `/reload` | Reload config without restart |
| `/help` | Show help |
| `/exit` | Quit terminal mode |
| `remind me <title> in/at <when>` | Natural language scheduling |

Fuzzy correction applies to all `/` commands — `/hlp` becomes `/help`, `/taks` becomes `/tasks`.

---

## Configuration

Edit `config/config.yaml`:

```yaml
backend: ollama
model: llama3              # any model pulled in Ollama
temperature: 0.2
max_tokens: 600
timezone: America/New_York
memory_db_path: data/intuition.db
anticipation:
  enabled: true
  debounce_ms: 180
  match_threshold: 0.6
hardware:
  drivers:
    - name: led_strip
      simulate: true
    - name: gpu_nvml
      enabled: true
```

Environment variables (`.env` or shell):

```
OLLAMA_HOST=http://127.0.0.1:11434
INTUITION_SAFE=1
```

---

## Architecture

```
start_ui.py
  ├── uvicorn → interface/server.py   (FastAPI + WebSocket)
  │               ├── core/brain.py      (LLM coordination)
  │               ├── core/memory.py     (SQLite store)
  │               ├── core/scheduler.py  (reminder polling)
  │               ├── core/anticipator.py (background prewarming)
  │               └── core/actions.py    (file, exec, task, hw)
  └── Electron → ui/
                  ├── main.js            (frameless window, shortcuts)
                  └── renderer/          (HUD interface)

intuitionos.py → interface/terminal.py  (classic REPL, same core)
```

---

## Design philosophy

IntuitionOS is built around cognitive science concepts:

| Human faculty | IntuitionOS feature |
|--------------|---------------------|
| Fast automatic system | Fuzzy commands that forgive typos |
| Predictive processing | Anticipator precomputes results while you type |
| Episodic memory | `/save`, `/recall`, SQLite memory store |
| Prospective memory | Natural language reminders, scheduler |
| Risk management | Safe Mode, sandboxed exec, visible plans |

**Core rules:**
- Favor momentum over exactness — correct typos in the first token, preserve arguments
- Predict quietly, reveal on confirmation — prewarm but show nothing until Enter
- Constrain the blast radius — exec is sandboxed, Safe Mode is on by default
- Local by default — Ollama runs on your machine, your files stay local

---

## Tech stack

- **Python** — core, backend, terminal REPL
- **FastAPI + WebSocket** — bridge between HUD and brain
- **Electron** — native HUD window (frameless, always-on-top, transparent)
- **Ollama** — local LLM inference
- **SQLite** — memory and task storage
- **Rich** — terminal formatting
- **prompt_toolkit** — terminal REPL input

---

## Troubleshooting

**HUD commands do nothing**
The WebSocket might not be connected. Check that `python start_ui.py` is running and the terminal shows no errors. The HUD input placeholder reads "Reconnecting…" when disconnected.

**LLM errors**
Make sure Ollama is running (`ollama serve`) and the model is pulled (`ollama pull <model>`). File and task commands work without Ollama.

**venv Python not found**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```
Or allow system Python: `$env:INTUITION_ALLOW_SYSTEM_PY = "1"`

**Safe Mode blocking exec**
```
/safe off
/exec "python your_script.py"
```

---

## Roadmap

- Per-project memory with tagging and export
- Streaming LLM replies token by token in the HUD
- Voice input trigger via hotword
- Real hardware adapter support (LED strips, serial devices)
- Plugin system for custom actions

---

## Credits

Built by Soheil Sepahyar. Runs locally on Ollama. No data leaves your machine.
