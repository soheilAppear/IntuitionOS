import asyncio
import difflib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from core.actions import actions, set_logger, set_memory, set_scheduler, set_safe_mode_action
from core.anticipator import Anticipator
from core.brain import Brain
from core.logger import make_logger
from core.llm import LLMClient
from core.memory import Memory
from core.scheduler import Scheduler

_state: dict = {}
_clients: set = set()
_executor = ThreadPoolExecutor(max_workers=4)


def _load_config():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def _broadcast(msg: dict):
    dead = set()
    for ws in list(_clients):
        try:
            await ws.send_json(msg)
        except Exception:
            dead.add(ws)
    _clients.difference_update(dead)


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    cfg = _load_config()
    sys_prompt = _read_text(cfg.get("system_prompt_path", "config/system_prompt.txt"))
    schema = _read_json(cfg.get("planner_schema_path", "config/planner_schema.json"))

    mem = Memory(cfg.get("memory_db_path", "data/intuition.db"))
    logger = make_logger(cfg.get("log_path", "data/log.txt"))
    set_logger(logger)
    set_memory(mem)

    llm = LLMClient(
        cfg.get("backend", "ollama"),
        cfg.get("model", "gpt-oss:20b"),
        cfg.get("temperature", 0.2),
        cfg.get("max_tokens", 600),
    )
    brain = Brain(llm, mem, sys_prompt, schema, logger=logger)

    def notify(task_id: int, title: str):
        mem.add("reminder", f"#{task_id} {title}", tags="reminder")
        asyncio.run_coroutine_threadsafe(
            _broadcast({"type": "reminder", "id": task_id, "title": title}),
            loop,
        )

    def execute(payload: dict):
        name = payload.get("action")
        kwargs = payload.get("kwargs", {})
        if name in {"hw_call"}:
            res = actions.call(name, **kwargs)
            mem.add("tool", json.dumps({"scheduled": True, "action": name, "result": res})[:2000])

    sched = Scheduler(
        db_path=cfg.get("memory_db_path", "data/intuition.db"),
        tz=cfg.get("timezone", "America/New_York"),
        tick_seconds=10,
        notify_cb=notify,
        execute_cb=execute,
    )
    set_scheduler(sched)

    for d in cfg.get("hardware", {}).get("drivers", []):
        if d.get("name") == "led_strip":
            from plugins.led_strip import LEDStrip
            from core.actions import register_driver
            register_driver(LEDStrip(simulate=d.get("simulate", True), port=d.get("port")))
        if d.get("name") == "gpu_nvml" and d.get("enabled", True):
            from plugins.gpu_nvml import GPUNVML
            from core.actions import register_driver
            register_driver(GPUNVML(enabled=True))

    def predict(buf: str):
        t = buf.strip()
        if not t:
            return {"confidence": 0.0}
        if t == "tree" or t.startswith("tree "):
            return {"intent": "tree", "text": t, "confidence": 0.9}
        if t == "ls":
            return {"intent": "ls", "text": t, "confidence": 0.9}
        if t.startswith("read file "):
            return {"intent": "read_file", "text": t, "confidence": 0.85}
        return {"intent": "plan", "text": t, "confidence": 0.65}

    def prewarm(intent: dict):
        t = intent.get("text", "")
        kind = intent.get("intent")
        if kind == "tree":
            return (t, {"reply": str(actions.call("list_tree", path="."))[:4000]})
        if kind == "ls":
            return (t, {"reply": str(actions.call("list_dir", path="."))[:4000]})
        if kind == "read_file":
            path = t[len("read file "):].strip()
            return (t, {"reply": str(actions.call("read_file", path=path))[:4000]})
        if kind == "plan":
            return (t, {"plan": brain.plan_dryrun(t).get("plan", [])})
        return None

    a = cfg.get("anticipation", {}) or {}
    ant = Anticipator(
        predict_fn=predict,
        prewarm_fn=prewarm,
        enabled=bool(a.get("enabled", True)),
        debounce_ms=int(a.get("debounce_ms", 180)),
        match_threshold=float(a.get("match_threshold", 0.6)),
    )
    ant.start()

    _state.update({"cfg": cfg, "brain": brain, "mem": mem, "sched": sched, "ant": ant})

    yield

    ant.stop()
    try:
        sched.stop()
    except Exception:
        pass


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_KNOWN_CMDS = [
    "/help", "/exit", "/memory", "/dream", "/save", "/recall", "/actions",
    "/config", "/reload", "/tasks", "/done", "/delete", "/snooze",
    "/hw", "/task_payload", "/safe", "/exec", "/write", "/read",
]


def _fuzzy_cmd(base: str) -> str:
    if base in _KNOWN_CMDS:
        return base
    m = difflib.get_close_matches(base, _KNOWN_CMDS, n=1, cutoff=0.55)
    return m[0] if m else base


async def _handle_command(ws: WebSocket, text: str):
    mem: Memory = _state["mem"]
    brain: Brain = _state["brain"]
    loop = asyncio.get_running_loop()

    if text == "/memory":
        rows = mem.recent(limit=12)
        await ws.send_json({"type": "memory", "rows": [
            {"id": r[0], "ts": r[1], "role": r[2], "text": r[3], "tags": r[4]}
            for r in reversed(rows)
        ]})
        return

    if text == "/tasks":
        result = actions.call("list_tasks", status="pending")
        await ws.send_json({"type": "tasks", "rows": result.get("result", [])})
        return

    if text == "/dream":
        out = await loop.run_in_executor(_executor, lambda: brain.plan_dryrun("reflection"))
        await ws.send_json({"type": "reply", "plan": out.get("plan", []), "text": ""})
        return

    if text.startswith("/save "):
        note = text[6:].strip().strip('"')
        mem.add("note", note, tags="note")
        await ws.send_json({"type": "reply", "text": "Saved."})
        return

    if text.startswith("/recall "):
        term = text[8:].strip().strip('"')
        rows = mem.search(term, limit=12)
        await ws.send_json({"type": "memory", "rows": [
            {"id": r[0], "ts": r[1], "role": r[2], "text": r[3], "tags": r[4]}
            for r in reversed(rows)
        ]})
        return

    if text.startswith("/done "):
        try:
            tid = int(text.split(" ", 1)[1])
            actions.call("complete_task", task_id=tid)
            await ws.send_json({"type": "reply", "text": f"Task {tid} marked done."})
            await ws.send_json({"type": "status",
                                "tasks_count": len(mem.list_tasks(status="pending")),
                                "safe_mode": os.environ.get("INTUITION_SAFE", "1") != "0"})
        except Exception:
            await ws.send_json({"type": "error", "text": "usage: /done <id>"})
        return

    if text.startswith("/delete "):
        try:
            tid = int(text.split(" ", 1)[1])
            actions.call("delete_task", task_id=tid)
            await ws.send_json({"type": "reply", "text": f"Task {tid} deleted."})
        except Exception:
            await ws.send_json({"type": "error", "text": "usage: /delete <id>"})
        return

    if text.startswith("/snooze "):
        try:
            parts = text.split(" ")
            actions.call("snooze_task", task_id=int(parts[1]), delta=parts[2])
            await ws.send_json({"type": "reply", "text": f"Task {parts[1]} snoozed {parts[2]}."})
        except Exception:
            await ws.send_json({"type": "error", "text": "usage: /snooze <id> <15m|2h|1d>"})
        return

    if text.startswith("/safe"):
        parts = text.split()
        if len(parts) >= 2:
            res = set_safe_mode_action(state=parts[1])
            safe_on = os.environ.get("INTUITION_SAFE", "1") != "0"
            await ws.send_json({
                "type": "status",
                "safe_mode": safe_on,
                "tasks_count": len(mem.list_tasks(status="pending")),
                "text": res.get("result", ""),
            })
        else:
            await ws.send_json({"type": "error", "text": "usage: /safe on|off"})
        return

    if text.startswith("/exec "):
        rest = text[6:].strip()
        cmd, cwd = None, "."
        if rest.startswith('"'):
            idx = rest.find('"', 1)
            if idx != -1:
                cmd = rest[1:idx]
                cwd = rest[idx + 1:].strip() or "."
        if not cmd:
            pts = rest.split(" ", 1)
            cmd, cwd = pts[0], (pts[1] if len(pts) == 2 else ".")
        res = await loop.run_in_executor(_executor, lambda: actions.call("run_local", cmd=cmd, cwd=cwd))
        out = res.get("stdout", "") or res.get("error", "") or json.dumps(res)
        await ws.send_json({"type": "reply", "text": out})
        return

    if text.startswith("/read "):
        path = text.split(" ", 1)[1].strip()
        res = await loop.run_in_executor(_executor, lambda: actions.call("read_file", path=path))
        await ws.send_json({"type": "reply", "text": res.get("text", res.get("error", ""))})
        return

    if text == "/hw":
        await ws.send_json({"type": "reply", "text": json.dumps(actions.call("hw_list"), indent=2)})
        return

    if text == "/actions":
        await ws.send_json({"type": "reply", "text": ", ".join(sorted(actions.names))})
        return

    await ws.send_json({"type": "error", "text": f"Unknown command: {text}"})


async def _handle_input(ws: WebSocket, text: str):
    brain: Brain = _state["brain"]
    mem: Memory = _state["mem"]
    ant: Anticipator = _state["ant"]
    loop = asyncio.get_running_loop()

    if text.startswith("/"):
        tokens = text.split()
        fixed = _fuzzy_cmd(tokens[0])
        if fixed != tokens[0]:
            text = " ".join([fixed] + tokens[1:]).strip()
        await _handle_command(ws, text)
        return

    m = re.match(r"^remind\s+me\s+(.+?)\s+((?:in|at)\s+\S.*)$", text, re.IGNORECASE)
    if m:
        res = actions.call("create_task", text=m.group(1).strip(), when=m.group(2).strip())
        await ws.send_json({"type": "reply", "text": json.dumps(res)})
        return

    if text == "ls":
        res = await loop.run_in_executor(_executor, lambda: actions.call("list_dir", path="."))
        await ws.send_json({"type": "reply", "text": str(res)})
        return

    if text == "tree":
        res = await loop.run_in_executor(_executor, lambda: actions.call("list_tree", path="."))
        await ws.send_json({"type": "reply", "text": str(res)})
        return

    pre = ant.try_serve(text)
    if pre and isinstance(pre, dict) and ("reply" in pre or "plan" in pre):
        mem.add("assistant", pre.get("reply", ""))
        await ws.send_json({
            "type": "reply",
            "text": pre.get("reply", ""),
            "plan": pre.get("plan", []),
            "from_cache": True,
        })
        return

    await ws.send_json({"type": "thinking"})
    try:
        out = await loop.run_in_executor(_executor, lambda: brain.step(text))
        await ws.send_json({
            "type": "reply",
            "text": out.get("reply", ""),
            "plan": out.get("plan", []),
        })
    except Exception as e:
        await ws.send_json({"type": "error", "text": f"LLM error: {e}"})


async def _maybe_send_anticipation(ws: WebSocket, text: str):
    await asyncio.sleep(0.35)
    ant = _state.get("ant")
    if not ant:
        return
    pre = ant.try_serve(text)
    if pre and isinstance(pre, dict):
        try:
            await ws.send_json({"type": "anticipation", "data": pre, "text": text})
        except Exception:
            pass


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.add(ws)

    mem: Memory = _state["mem"]
    await ws.send_json({
        "type": "status",
        "safe_mode": os.environ.get("INTUITION_SAFE", "1") != "0",
        "tasks_count": len(mem.list_tasks(status="pending")),
        "version": "1.0",
    })

    try:
        while True:
            data = await ws.receive_json()
            t = data.get("type")

            if t == "buffer":
                _state["ant"].update_buffer(data.get("text", ""))
                asyncio.create_task(_maybe_send_anticipation(ws, data.get("text", "")))

            elif t == "input":
                text = data.get("text", "").strip()
                if text:
                    await _handle_input(ws, text)

            elif t == "get_status":
                await ws.send_json({
                    "type": "status",
                    "safe_mode": os.environ.get("INTUITION_SAFE", "1") != "0",
                    "tasks_count": len(mem.list_tasks(status="pending")),
                })

    except WebSocketDisconnect:
        _clients.discard(ws)
    except Exception:
        _clients.discard(ws)


@app.get("/health")
async def health():
    return {"ok": True, "version": "1.0"}
