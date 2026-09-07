# Action registry and safe helpers used by the terminal, the HUD and the planner.
#
# The plain functions below are unchanged in spirit: they do one thing and return
# a dict. What changed is how they are reached. Every dispatch now goes through
# ActionRegistry.dispatch, which asks core.capabilities.gate for a verdict first
# and writes a core.journal row for anything that is not free. The functions stay
# plain on purpose — they remain directly importable and directly testable, and
# the policy lives in exactly one place instead of being re-improvised at each
# call site the way it was when run_local, write_file and hw_call each had their
# own idea of what "safe" meant.

import os, json, shlex, subprocess, sys, shutil, time

from .capabilities import (
    Capability,
    capabilities,
    gate,
    is_safe_mode,
    pending_confirmations,
    set_safe_mode,
)
from .journal import Journal
from .logger import make_logger


class ActionRegistry:
    def __init__(self):
        # Registry of action names to callables
        self._actions = {}
        # Expose names for UI
        self.names = set()

    def register(self, name, fn):
        # Register callable
        self._actions[name] = fn
        self.names.add(name)

    # ── Gated dispatch ───────────────────────────────────────────────────

    def call(self, name, **kwargs):
        """Backwards-compatible entry point: a human at the keyboard.

        Existing call sites in the terminal and the HUD pass their arguments as
        keywords, so this keeps working exactly as before — but it now carries an
        actor, which means even a typed command is judged and journalled.
        """
        return self.dispatch(name, kwargs, actor="user", confidence=1.0)

    def dispatch(self, name, args=None, *, actor="user", confidence=1.0, confirmed=False):
        """The one road to a side effect.

        Returns the action's own dict on success, or one of:
          {"error": ..., "denied": True}          — the gate refused
          {"needs_confirmation": True, "token": ...} — a human must say yes
        """
        cap = capabilities.get(name)
        if not cap:
            # A name that is registered but has no manifest entry is a bug, not a
            # reason to run it unjudged.
            if name in self._actions:
                return {"error": f"action {name} has no capability manifest entry", "denied": True}
            return {"error": f"unknown action: {name}"}

        decision = gate(cap, args or {}, confidence=confidence, actor=actor, thresholds=_thresholds)

        if decision.verdict == "deny":
            _record(actor=actor, capability=name, args=args or {}, confidence=confidence,
                    decision="deny", outcome=None)
            _logger(f"gate DENY {actor}/{name}: {decision.reason}")
            return {"error": decision.reason, "denied": True, "capability": name}

        if decision.verdict == "confirm" and not confirmed:
            p = pending_confirmations.put(name, decision.args, actor, confidence, decision.reason)
            return {
                "needs_confirmation": True,
                "token": p.token,
                "capability": name,
                "args": decision.args,
                "reason": decision.reason,
                "reversibility": cap.reversibility,
                "summary": cap.summary,
            }

        label = "confirm_granted" if decision.verdict == "confirm" else "allow"
        return self._execute(cap, decision.args, actor, confidence, label)

    def confirm(self, token, granted=True):
        """Resolve a parked confirmation. The token is good for one use."""
        p = pending_confirmations.take(token)
        if not p:
            return {"error": "confirmation expired or already used"}
        cap = capabilities.get(p.capability)
        if not cap:
            return {"error": f"unknown capability: {p.capability}"}
        if not granted:
            _record(actor=p.actor, capability=p.capability, args=p.args,
                    confidence=p.confidence, decision="confirm_denied", outcome=None)
            return {"ok": True, "cancelled": True, "capability": p.capability}
        # The arguments in the store were already validated and jailed by the
        # gate, so they are executed as-is rather than re-read from the wire.
        return self._execute(cap, p.args, p.actor, p.confidence, "confirm_granted")

    def _execute(self, cap, args, actor, confidence, decision_label):
        undo_payload = {}
        if cap.capture_undo:
            try:
                undo_payload.update(cap.capture_undo(args) or {})
            except Exception as e:
                # Without a reversal payload the action is not reversible, so
                # refuse rather than quietly performing an unrepeatable write.
                return {"error": f"could not capture undo state for {cap.name}: {e}", "denied": True}

        entry_id = None
        if cap.reversibility != "free":
            entry_id = _record(actor=actor, capability=cap.name, args=args,
                               confidence=confidence, decision=decision_label, outcome=None)

        try:
            result = cap.fn(**args)
        except Exception as e:
            result = {"error": str(e)}

        outcome = "error" if isinstance(result, dict) and result.get("error") else "ok"

        if outcome == "ok" and cap.capture_undo_result:
            try:
                undo_payload.update(cap.capture_undo_result(args, result) or {})
            except Exception:
                undo_payload = {}

        if entry_id is not None:
            _journal_ref[0].finish(
                entry_id, outcome,
                undo=undo_payload if (outcome == "ok" and cap.undo and undo_payload) else None,
            )
        return result


actions = ActionRegistry()

# Globals set by terminal
_scheduler = None
_memory = None
_logger = make_logger("data/log.txt")
_journal_ref = [None]
# Cost-gated confidence thresholds, populated from config.yaml in Phase 5.
_thresholds = None


def set_scheduler(s):
    # Bind scheduler and tell it about memory later
    global _scheduler
    _scheduler = s


def set_memory(mem):
    # Bind memory for actions that need it, and open the journal on the same db
    global _memory
    _memory = mem
    _journal_ref[0] = Journal(mem)
    if _scheduler:
        _scheduler.set_memory(mem)


def set_logger(logger):
    # Replace default logger
    global _logger
    _logger = logger


def set_thresholds(thresholds):
    # Bind the cost-gated threshold policy (Phase 5)
    global _thresholds
    _thresholds = thresholds


def get_journal():
    return _journal_ref[0]


def _record(**kw):
    j = _journal_ref[0]
    return j.record(**kw) if j else None


def _is_safe():
    # Kept as a name other modules may import; the state itself now lives in
    # core.capabilities rather than in os.environ (Appendix A #4).
    return is_safe_mode()


def set_safe_mode_action(state: str):
    # Toggle Safe Mode for this process
    if state.lower() in ("off", "0", "false", "no"):
        set_safe_mode(False)
        return {"result": "Safe Mode OFF"}
    if state.lower() in ("on", "1", "true", "yes"):
        set_safe_mode(True)
        return {"result": "Safe Mode ON"}
    return {"error": "usage: /safe on|off"}


# File actions

def list_dir(path: str = "."):
    # Return flat list of items in a directory
    try:
        out = []
        for name in os.listdir(path):
            tp = "dir" if os.path.isdir(os.path.join(path, name)) else "file"
            out.append({"name": name, "type": tp})
        return out
    except Exception as e:
        return {"error": str(e)}


def list_tree(path: str = ".", depth: int = 3):
    # Return nested tree with limited depth
    try:
        def walk(p, d):
            if d < 0:
                return []
            items = []
            for name in os.listdir(p):
                full = os.path.join(p, name)
                if os.path.isdir(full):
                    items.append({"name": name, "type": "dir", "children": walk(full, d - 1)})
                else:
                    items.append({"name": name, "type": "file"})
            return items
        return walk(path, depth)
    except Exception as e:
        return {"error": str(e)}


def read_file(path: str):
    # Read a small text file
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {"path": path, "text": f.read()}
    except Exception as e:
        return {"error": str(e)}


# Cap a single write so a runaway model cannot fill the disk one call at a time.
MAX_WRITE_BYTES = 1_000_000


def write_file(path: str, text: str):
    # Write text to a file. The path jail and the size limit are enforced by the
    # gate before this is reached; the check here is the backstop for direct calls.
    try:
        if len(text.encode("utf-8")) > MAX_WRITE_BYTES:
            return {"error": f"refusing to write more than {MAX_WRITE_BYTES} bytes"}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return {"ok": True, "path": path, "bytes": len(text.encode("utf-8"))}
    except Exception as e:
        return {"error": str(e)}


def _capture_write(args):
    # Snapshot whatever write_file is about to destroy.
    path = args["path"]
    if not os.path.exists(path):
        return {"path": path, "existed": False}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {"path": path, "existed": True, "text": f.read()}
    except Exception as e:
        # Binary or unreadable: say so now rather than promise an undo we cannot keep.
        return {"path": path, "existed": True, "unreadable": str(e)}


def _undo_write(payload):
    path = payload["path"]
    if not payload.get("existed"):
        if os.path.exists(path):
            os.remove(path)
        return {"removed": path}
    if "text" not in payload:
        raise RuntimeError(f"prior contents of {path} were not readable: {payload.get('unreadable')}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(payload["text"])
    return {"restored": path, "bytes": len(payload["text"].encode("utf-8"))}


# Task actions

def create_task(text: str = None, when: str = None, repeat: str = "", payload: dict = None):
    # Create a reminder
    if not _scheduler or not _memory:
        return {"error": "scheduler not ready"}
    title = text or "reminder"
    return _scheduler.create(title=title, when=when, payload=payload or {}, repeat=repeat)


def list_tasks(status: str = "pending"):
    # List tasks by status
    return {"result": _memory.list_tasks(status=status)}


def complete_task(task_id: int):
    # Complete task
    _memory.complete_task(task_id)
    return {"ok": True}


def delete_task(task_id: int):
    # Delete task
    _memory.delete_task(task_id)
    return {"ok": True}


def snooze_task(task_id: int, delta: str):
    # Parse short form like 15m 2h 1d into seconds
    _memory.snooze_task(task_id, _delta_seconds(delta))
    return {"ok": True}


def _delta_seconds(delta: str) -> int:
    mult = {"m": 60, "h": 3600, "d": 86400}.get(delta[-1:], 1)
    return int(delta[:-1] if mult != 1 else delta) * mult


# Exec action

def run_local(cmd: str, cwd: str = "."):
    # The cwd jail is enforced by the gate (path_scope="project"). The old check
    # here was target.startswith(base_dir), which let /proj-evil pass for /proj;
    # see core.capabilities.jail_path.
    base_dir = os.getcwd()
    target = os.path.abspath(cwd)

    # Resolve venv python
    venv_py = (
        os.path.join(base_dir, ".venv", "Scripts", "python.exe")
        if os.name == "nt"
        else os.path.join(base_dir, ".venv", "bin", "python")
    )
    allow_system = os.environ.get("INTUITION_ALLOW_SYSTEM_PY", "0") == "1"

    # Pick interpreter
    if os.path.exists(venv_py):
        py = venv_py
    elif allow_system:
        py = sys.executable or "python"
    else:
        return {"error": f"venv python not found at {venv_py}"}

    # If the user wrote "python something", swap in our interpreter. Tokenising
    # first matters: a plain cmd.replace("python", py, 1) also rewrote the
    # "python" inside a filename like python_script.py (Appendix A #15).
    cmd = _swap_interpreter(cmd, py)

    if is_safe_mode():
        return {"error": "Safe Mode is ON — use /safe off first"}
    try:
        result = subprocess.run(cmd, cwd=target, shell=True, capture_output=True, text=True, timeout=60)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except Exception as e:
        return {"error": str(e)}


def _swap_interpreter(cmd: str, py: str) -> str:
    """Replace a leading `python` token with our interpreter, and only a token."""
    try:
        tokens = shlex.split(cmd, posix=(os.name != "nt"))
    except ValueError:
        return cmd
    if not tokens or tokens[0].lower() not in ("python", "python3", "py"):
        return cmd
    quoted = f'"{py}"' if (" " in py and os.name == "nt") else shlex.quote(py) if os.name != "nt" else py
    return " ".join([quoted] + tokens[1:])


# Hardware registry and adapters

_drivers = {}


def register_driver(drv):
    # Register a driver instance by its name
    _drivers[drv.name] = drv


def hw_list():
    # List registered hardware drivers
    return {"devices": list(_drivers.keys())}


def hw_schema(device: str):
    # Return schema for a device
    d = _drivers.get(device)
    if not d:
        return {"error": f"no such device: {device}"}
    return {"device": device, "schema": d.schema()}


def hw_call(device: str, action: str, args: dict = None):
    # Dispatch a hardware action. Arguments were checked against the driver's own
    # declared schema by _validate_hw before this was reached, so the **splat is
    # no longer splatting whatever the caller invented (Appendix A #3).
    d = _drivers.get(device)
    if not d:
        return {"error": f"no such device: {device}"}
    try:
        return {"result": d.call(action, **(args or {}))}
    except Exception as e:
        return {"error": str(e)}


def _driver_action(device: str, action: str):
    d = _drivers.get(device)
    if not d:
        return None, None
    for entry in (d.schema() or {}).get("actions", []):
        if entry.get("name") == action:
            return d, entry
    return d, None


def _validate_hw(args: dict):
    """Check a hw_call against the schema the driver itself declares."""
    device, action = args.get("device"), args.get("action")
    d, entry = _driver_action(device, action)
    if d is None:
        return f"no such device: {device}"
    if entry is None:
        names = [a.get("name") for a in (d.schema() or {}).get("actions", [])]
        return f"device {device} declares no action {action!r} (has: {', '.join(names) or 'none'})"
    declared = set(entry.get("args", []))
    supplied = set((args.get("args") or {}).keys())
    unknown = supplied - declared
    if unknown:
        return f"{device}.{action} does not accept {', '.join(sorted(unknown))}"
    return None


def _hw_needs_confirmation(args: dict):
    """Confirmation for hardware depends on the driver, as the manifest says.

    A driver opts an action in by listing it under `confirm` in its schema, or by
    marking the action entry `{"confirm": true}`. Drivers that say nothing get
    the safe default for a device that can move something in the physical world:
    reads are free, writes are confirmed.
    """
    device, action = args.get("device"), args.get("action")
    d, entry = _driver_action(device, action)
    if d is None or entry is None:
        return True
    if "confirm" in entry:
        return bool(entry["confirm"])
    schema = d.schema() or {}
    if "confirm" in schema:
        return action in (schema.get("confirm") or [])
    return not entry.get("name", "").startswith(("status", "get", "read", "list"))


# ── Manifest ─────────────────────────────────────────────────────────────────
#
# Every action above is declared here with what it costs to be wrong. Nothing
# reaches dispatch without an entry; a registered function with no entry is
# refused rather than run unjudged.

_PATH = {"type": "string", "minLength": 1}


def _cap(name, fn, schema, reversibility, cost, confirm, summary, **kw):
    actions.register(name, fn)
    return capabilities.register(
        Capability(
            name=name, fn=fn, arg_schema=schema, reversibility=reversibility,
            est_cost_ms=cost, requires_confirmation=confirm, summary=summary, **kw
        )
    )


def _schema(props, required=()):
    return {
        "type": "object",
        "properties": props,
        "required": list(required),
        # Unknown keys are rejected rather than ignored: an argument the action
        # never declared is a sign the caller misunderstood it.
        "additionalProperties": False,
    }


_cap("list_dir", list_dir, _schema({"path": _PATH}), "free", 5, False,
     "List the entries of one directory.", path_scope="project", path_args=("path",))

_cap("list_tree", list_tree, _schema({"path": _PATH, "depth": {"type": "integer", "minimum": 0, "maximum": 6}}),
     "free", 60, False, "List a directory recursively to a bounded depth.",
     path_scope="project", path_args=("path",))

_cap("read_file", read_file, _schema({"path": _PATH}, ["path"]), "free", 10, False,
     "Read a UTF-8 text file and return its contents.",
     path_scope="project", path_args=("path",))

_cap("write_file", write_file,
     _schema({"path": _PATH, "text": {"type": "string", "maxLength": MAX_WRITE_BYTES}}, ["path", "text"]),
     "reversible", 15, False, "Overwrite a file with new text. Undoable from the journal.",
     path_scope="project", path_args=("path",),
     capture_undo=_capture_write, undo=_undo_write)

_cap("list_tasks", list_tasks, _schema({"status": {"type": "string"}}), "free", 5, False,
     "List reminders by status.")

_cap("create_task", create_task,
     _schema({"text": {"type": ["string", "null"]}, "when": {"type": ["string", "null"]},
              "repeat": {"type": "string"}, "payload": {"type": ["object", "null"]}}),
     "reversible", 20, False, "Schedule a reminder for a time given in plain language.",
     capture_undo_result=lambda args, result: ({"task_id": result["id"]} if result.get("ok") else {}),
     undo=lambda p: (_memory.delete_task(p["task_id"]), {"deleted_task": p["task_id"]})[1])

_cap("complete_task", complete_task, _schema({"task_id": {"type": "integer"}}, ["task_id"]),
     "reversible", 10, False, "Mark a reminder done.",
     capture_undo=lambda args: {"task_id": args["task_id"],
                                "status": (_memory.get_task(args["task_id"]) or {}).get("status", "pending")},
     undo=lambda p: (_memory.set_task_status(p["task_id"], p["status"]), {"restored_task": p["task_id"]})[1])

_cap("snooze_task", snooze_task,
     _schema({"task_id": {"type": "integer"}, "delta": {"type": "string", "pattern": r"^\d+[mhd]?$"}},
             ["task_id", "delta"]),
     "reversible", 10, False, "Push a reminder later by 15m, 2h or 1d.",
     capture_undo=lambda args: {"task_id": args["task_id"], "delta": args["delta"]},
     undo=lambda p: (_memory.snooze_task(p["task_id"], -_delta_seconds(p["delta"])),
                     {"unsnoozed_task": p["task_id"]})[1])

_cap("delete_task", delete_task, _schema({"task_id": {"type": "integer"}}, ["task_id"]),
     "irreversible", 10, True, "Delete a reminder permanently.")

_cap("run_local", run_local, _schema({"cmd": {"type": "string", "minLength": 1}, "cwd": _PATH}, ["cmd"]),
     "irreversible", 500, True, "Run a shell command inside the project directory.",
     path_scope="project", path_args=("cwd",))

_cap("hw_list", hw_list, _schema({}), "free", 1, False, "List registered hardware devices.")

_cap("hw_schema", hw_schema, _schema({"device": {"type": "string"}}, ["device"]), "free", 1, False,
     "Show the actions one hardware device declares.")

_cap("hw_call", hw_call,
     _schema({"device": {"type": "string"}, "action": {"type": "string"},
              "args": {"type": ["object", "null"]}}, ["device", "action"]),
     "reversible", 50, False, "Invoke a declared action on a hardware device.",
     extra_validate=_validate_hw, dynamic_confirm=_hw_needs_confirmation)


def _undo_screenshot(payload):
    # Undoing a screenshot means deleting the file it left behind.
    path = payload.get("path")
    if path and os.path.exists(path):
        os.remove(path)
        return {"removed": path}
    return {"removed": None}


def register_os_capabilities():
    """Register the OS sandbox surface.

    Kept behind a call rather than done at import so a headless test run does not
    have to care about the host OS. Both the terminal and the HUD call it, which
    is the point: before this, the HUD registered these actions directly on the
    plain registry and "shut down my pc" — typed *or spoken into the microphone*
    — reached `shutdown /s /t 30` with no gate, no confirmation and no record.
    """
    try:
        from . import os_sandbox as _os
    except Exception as e:  # pragma: no cover - platform dependent
        _logger(f"os sandbox unavailable: {e}")
        return []

    def _os_cap(name, fn_name, reversibility, cost, confirm, summary, schema=None, **kw):
        fn = getattr(_os, fn_name, None)
        if fn is None:
            return None
        return _cap(name, fn, schema or _schema({}), reversibility, cost, confirm, summary, **kw)

    level = _schema({"level": {"type": "integer", "minimum": 0, "maximum": 100}}, ["level"])

    # Reads: cheap and harmless, so the anticipator may prewarm them.
    _os_cap("os_system_info", "system_info", "free", 120, False, "Report OS, RAM, disk and uptime.")
    _os_cap("os_get_volume", "get_volume", "free", 80, False, "Read the current master volume.")
    _os_cap("os_get_brightness", "get_brightness", "free", 200, False, "Read display brightness.")
    _os_cap("os_get_battery", "get_battery", "free", 20, False, "Read battery level and charge state.")
    _os_cap("os_get_network_info", "get_network_info", "free", 400, False, "Report interfaces and Wi-Fi SSID.")
    _os_cap("os_get_clipboard", "get_clipboard", "free", 300, False, "Read the clipboard.")
    _os_cap("os_list_windows", "list_windows", "free", 600, False, "List visible window titles.")
    _os_cap("os_list_processes", "list_processes", "free", 300, False, "List running processes.",
            schema=_schema({"filter_name": {"type": "string"}}))
    _os_cap("os_cancel_shutdown", "cancel_shutdown", "free", 50, False,
            "Cancel a pending shutdown or restart.")

    # Writes that can be put back, each with the prior value captured first.
    _os_cap("os_set_volume", "set_volume", "reversible", 400, False, "Set master volume 0-100.",
            schema=level,
            capture_undo=lambda a: {"level": _os.get_volume().get("volume")},
            undo=lambda p: _os.set_volume(p["level"]) if p.get("level") is not None else {"skipped": True})
    _os_cap("os_set_brightness", "set_brightness", "reversible", 400, False, "Set display brightness 0-100.",
            schema=level,
            capture_undo=lambda a: {"level": _os.get_brightness().get("brightness")},
            undo=lambda p: _os.set_brightness(p["level"]) if p.get("level") is not None else {"skipped": True})
    _os_cap("os_set_clipboard", "set_clipboard", "reversible", 300, False, "Replace the clipboard contents.",
            schema=_schema({"text": {"type": "string", "maxLength": 100_000}}, ["text"]),
            capture_undo=lambda a: {"text": _os.get_clipboard().get("text", "")},
            undo=lambda p: _os.set_clipboard(p["text"]))
    _os_cap("os_toggle_wifi", "toggle_wifi", "reversible", 2000, True, "Enable or disable the Wi-Fi adapter.",
            schema=_schema({"state": {"enum": ["on", "off"]}}, ["state"]),
            capture_undo=lambda a: {"state": "off" if a["state"] == "on" else "on"},
            undo=lambda p: _os.toggle_wifi(p["state"]))
    _os_cap("os_take_screenshot", "take_screenshot", "reversible", 900, False,
            "Capture the screen to data/screenshots.",
            capture_undo_result=lambda a, r: ({"path": r["path"]} if r.get("path") else {}),
            undo=_undo_screenshot)
    # Launching an app is reversible by the person sitting there — they close the
    # window. It carries no automatic undo, because killing an application the
    # user has since started typing into is worse than leaving it open.
    _os_cap("os_open_app", "open_app", "reversible", 800, False, "Launch an application by name.",
            schema=_schema({"name": {"type": "string", "minLength": 1}}, ["name"]))
    # Likewise: the user unlocks or wakes the machine themselves.
    _os_cap("os_lock_screen", "lock_screen", "reversible", 100, False, "Lock the screen.")
    _os_cap("os_sleep_computer", "sleep_computer", "reversible", 100, True, "Put the computer to sleep.")

    # Cannot be taken back by anyone. These are the ones that were completely
    # ungated before, reachable from a single regex match on speech.
    _os_cap("os_kill_process", "kill_process", "irreversible", 200, True, "Terminate a process by name.",
            schema=_schema({"name": {"type": "string", "minLength": 1}}, ["name"]))
    _os_cap("os_shutdown_computer", "shutdown_computer", "irreversible", 100, True, "Shut the computer down.",
            schema=_schema({"delay_sec": {"type": "integer", "minimum": 0, "maximum": 3600}}))
    _os_cap("os_restart_computer", "restart_computer", "irreversible", 100, True, "Restart the computer.",
            schema=_schema({"delay_sec": {"type": "integer", "minimum": 0, "maximum": 3600}}))
    _os_cap("os_type_text", "type_text", "irreversible", 500, True, "Type text as synthetic keystrokes.",
            schema=_schema({"text": {"type": "string", "maxLength": 4000}}, ["text"]))
    _os_cap("os_move_mouse", "move_mouse", "irreversible", 100, True, "Move the mouse pointer.",
            schema=_schema({"x": {"type": "integer"}, "y": {"type": "integer"}}, ["x", "y"]))
    _os_cap("os_click", "click", "irreversible", 100, True, "Click the mouse.",
            schema=_schema({"x": {"type": ["integer", "null"]}, "y": {"type": ["integer", "null"]},
                            "button": {"enum": ["left", "right", "middle"]}}))

    return [n for n in capabilities.names() if n.startswith("os_")]


# ── Journal-backed commands ──────────────────────────────────────────────────

def undo_last():
    """Reverse the most recent reversible action."""
    j = _journal_ref[0]
    if not j:
        return {"error": "journal not ready"}
    return j.undo_last(capabilities)


def journal_recent(limit: int = 20):
    """Recent journal entries, newest first."""
    j = _journal_ref[0]
    if not j:
        return {"error": "journal not ready"}
    return {"rows": j.recent(limit=limit)}
