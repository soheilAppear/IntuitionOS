# IntuitionOS v1.0

A local, LLM driven, intuitive shell that blends memory, task scheduling, fuzzy commands, and gentle hardware hooks.

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
.\.venv\Scripts\activate
pip install -r requirements.txt
$env:OLLAMA_HOST = "http://127.0.0.1:11434"
python .\intuitionos.py
```

If `ollama serve` is already running on 127.0.0.1:11434 you are good.

## Safe Mode

Safe Mode blocks `/exec` and any write action that touches the file system unless you use the explicit built-ins.
Use:

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

- If you see `venv python not found at ...`, create `.venv` as shown above or set:
  `$env:INTUITION_ALLOW_SYSTEM_PY = "1"` to permit system Python.
- If a command times out talking to Ollama, ensure `ollama serve` is running and the model is pulled:
  `ollama pull gpt-oss:20b`.

