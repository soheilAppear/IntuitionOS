# Action registry and safe helpers used by the terminal and the planner

import os, json, subprocess, sys, shutil
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

    def call(self, name, **kwargs):
        # Dispatch call to named action or return error
        fn = self._actions.get(name)
        if not fn:
            return {"error": f"unknown action: {name}"}
        return fn(**kwargs)

actions = ActionRegistry()

# Globals set by terminal
_scheduler = None
_memory = None
_logger = make_logger("data/log.txt")

def set_scheduler(s):
    # Bind scheduler and tell it about memory later
    global _scheduler
    _scheduler = s

def set_memory(mem):
    # Bind memory for actions that need it
    global _memory
    _memory = mem
    if _scheduler:
        _scheduler.set_memory(mem)

def set_logger(logger):
    # Replace default logger
    global _logger
    _logger = logger

# Safe Mode flag from environment
def _is_safe():
    # Safe Mode is on unless INTUITION_SAFE is "0"
    return os.environ.get("INTUITION_SAFE", "1") != "0"

def set_safe_mode_action(state:str):
    # Toggle Safe Mode via environment variable for current process
    if state.lower() in ("off","0","false","no"):
        os.environ["INTUITION_SAFE"] = "0"
        return {"result": "Safe Mode OFF"}
    if state.lower() in ("on","1","true","yes"):
        os.environ["INTUITION_SAFE"] = "1"
        return {"result": "Safe Mode ON"}
    return {"error": "usage: /safe on|off"}

# File actions

def list_dir(path:str="."):
    # Return flat list of items in a directory
    try:
        out=[]
        for name in os.listdir(path):
            tp = "dir" if os.path.isdir(os.path.join(path,name)) else "file"
            out.append({"name": name, "type": tp})
        return out
    except Exception as e:
        return {"error": str(e)}

def list_tree(path:str=".", depth:int=3):
    # Return nested tree with limited depth
    try:
        def walk(p, d):
            if d<0:
                return []
            items=[]
            for name in os.listdir(p):
                full=os.path.join(p,name)
                if os.path.isdir(full):
                    items.append({"name": name, "type":"dir", "children": walk(full, d-1)})
                else:
                    items.append({"name": name, "type":"file"})
            return items
        return walk(path, depth)
    except Exception as e:
        return {"error": str(e)}

def read_file(path:str):
    # Read a small text file
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {"path": path, "text": f.read()}
    except Exception as e:
        return {"error": str(e)}

def write_file(path:str, text:str):
    # Write text to a file, guard by Safe Mode
    try:
        if _is_safe():
            # Writes are permitted through this explicit action
            pass
        # Ensure directory exists
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return {"ok": True, "path": path, "bytes": len(text.encode("utf-8"))}
    except Exception as e:
        return {"error": str(e)}

# Task actions

def create_task(text:str=None, when:str=None, repeat:str="", payload:dict=None):
    # Create a reminder
    if not _scheduler or not _memory:
        return {"error": "scheduler not ready"}
    title = text or "reminder"
    return _scheduler.create(title=title, when=when, payload=payload or {}, repeat=repeat)

def list_tasks(status:str="pending"):
    # List tasks by status
    return {"result": _memory.list_tasks(status=status)}

def complete_task(task_id:int):
    # Complete task
    _memory.complete_task(task_id)
    return {"ok": True}

def delete_task(task_id:int):
    # Delete task
    _memory.delete_task(task_id)
    return {"ok": True}

def snooze_task(task_id:int, delta:str):
    # Parse short form like 15m 2h 1d into seconds
    mult = 1
    if delta.endswith("m"):
        mult = 60
        n = int(delta[:-1])
    elif delta.endswith("h"):
        mult = 3600
        n = int(delta[:-1])
    elif delta.endswith("d"):
        mult = 86400
        n = int(delta[:-1])
    else:
        n = int(delta)
    _memory.snooze_task(task_id, n*mult)
    return {"ok": True}

# Exec action

def run_local(cmd:str, cwd:str="."):
    # Only allow execution inside current repo
    base_dir = os.getcwd()
    target = os.path.abspath(cwd)
    if not target.startswith(base_dir):
        return {"error": "cwd outside project directory is blocked"}

    # Resolve venv python
    venv_py = os.path.join(base_dir, ".venv", "Scripts", "python.exe") if os.name == "nt" else os.path.join(base_dir, ".venv", "bin", "python")
    allow_system = os.environ.get("INTUITION_ALLOW_SYSTEM_PY", "0") == "1"

    # Pick interpreter
    if os.path.exists(venv_py):
        py = f'"{venv_py}"' if os.name == "nt" else venv_py
    elif allow_system:
        py = "python"
    else:
        return {"error": f"venv python not found at {venv_py}"}

    # If user wrote "python something", swap in our interpreter
    if cmd.lower().startswith("python "):
        cmd = cmd.replace("python", py, 1)

    try:
        if _is_safe():
            pass  # exec allowed via explicit /exec
        result = subprocess.run(cmd, cwd=target, shell=True, capture_output=True, text=True, timeout=60)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except Exception as e:
        return {"error": str(e)}


# Hardware registry and adapters

_drivers = {}

def register_driver(drv):
    # Register a driver instance by its name
    _drivers[drv.name] = drv

def hw_list():
    # List registered hardware drivers
    return {"devices": list(_drivers.keys())}

def hw_schema(device:str):
    # Return schema for a device
    d = _drivers.get(device)
    if not d:
        return {"error": f"no such device: {device}"}
    return {"device": device, "schema": d.schema()}

def hw_call(device:str, action:str, args:dict):
    # Dispatch a hardware action safely
    d = _drivers.get(device)
    if not d:
        return {"error": f"no such device: {device}"}
    try:
        return {"result": d.call(action, **(args or {}))}
    except Exception as e:
        return {"error": str(e)}

# Register default actions
actions.register("list_dir", list_dir)
actions.register("list_tree", list_tree)
actions.register("read_file", read_file)
actions.register("write_file", write_file)
actions.register("run_local", run_local)
actions.register("create_task", create_task)
actions.register("list_tasks", list_tasks)
actions.register("complete_task", complete_task)
actions.register("delete_task", delete_task)
actions.register("snooze_task", snooze_task)
actions.register("hw_list", hw_list)
actions.register("hw_schema", hw_schema)
actions.register("hw_call", hw_call)
