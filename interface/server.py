"""Local FastAPI/WebSocket adapter for the Electron HUD.

The application lifespan owns shared services; each socket owns its draft,
prediction window, and correction session. Client revisions identify edited
drafts on the wire. Core correction tokens bind a displayed selection, while
separate capability tokens authorize actions through the existing gate.

Blocking model/action work runs in the executor. WebSocket state and message
delivery stay on the event loop, including callbacks from scheduler/voice work.
"""

import asyncio
import datetime
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from core.actions import (
    actions,
    get_journal,
    journal_recent,
    register_os_capabilities,
    set_logger,
    set_memory,
    set_safe_mode_action,
    set_scheduler,
    set_thresholds,
    undo_last,
)
from core.anticipator import Anticipator
from core.brain import Brain
from core.calibration import CalibrationStore, load_thresholds, reliability
from core.capabilities import capabilities, is_safe_mode
from core.consolidation import RuleStore, consolidate, render_rules
from core.context import ContextSensor
from core.os_intents import _try_os_intent
from core.command_resolver import (
    KNOWN_COMMANDS,
    CorrectionSession,
    create_default_resolver,
    legacy_fuzzy_slash,
    learning_text,
)
from core.episodes import EpisodeLog, PredictionWindow
from core.predictor import Predictor, PredictorStore
from core.logger import make_logger
from core.llm import LLMClient
from core.memory import Memory
from core.retrieval import Retriever
from core.scheduler import Scheduler
from core.voice import VoiceRecognizer

# Shared services are created during lifespan, rather than per HUD connection.
_state: dict = {}
_clients: set = set()
# One prediction window per connection: what the user is typing, and what we put
# in front of them. Keyed by socket so two HUDs do not credit each other's hints.
_windows: dict = {}
# Socket-owned selection state is separate from the renderer's draft revision.
_resolutions: dict = {}
_connections: dict = {}
# Gate tokens carry socket/selection ownership until answered or invalidated.
_confirmations: dict = {}
# A dispatch fills in the outcome of the input currently handled on that socket.
_active_notes: dict = {}
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
    """Send a server event to connected HUDs and discard failed recipients."""
    dead = set()
    for ws in list(_clients):
        try:
            await ws.send_json(msg)
        except Exception:
            dead.add(ws)
    _clients.difference_update(dead)


def _voice_info():
    return dict(
        _state.get(
            "voice_status",
            {
                "state": "disabled",
                "available": False,
                "text": "Voice is disabled in config.yaml.",
            },
        )
    )


async def _set_voice_status(state, available, text):
    """Voice workers report lifecycle on the event loop, including failures."""
    info = {"state": state, "available": available, "text": text}
    _state["voice_status"] = info
    await _broadcast({"type": "voice_status", **info})


def _queue_voice_callback(loop, callback):
    if loop.is_closed():
        return
    future = asyncio.run_coroutine_threadsafe(callback(), loop)
    # Delivery to a disconnected socket must not produce an unobserved future.
    future.add_done_callback(
        lambda done: done.exception() if not done.cancelled() else None
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create shared services, then stop workers and save enabled learning."""
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
    bcfg = cfg.get("brain", {}) or {}
    rcfg = cfg.get("retrieval", {}) or {}
    retriever = Retriever(mem, budget_tokens=int(rcfg.get("budget_tokens", 700)))
    brain = Brain(
        llm,
        mem,
        sys_prompt,
        schema,
        logger=logger,
        max_iters=int(bcfg.get("max_iters", 5)),
        budget_ms=int(bcfg.get("budget_ms", 20000)),
        history_turns=int(bcfg.get("history_turns", 6)),
        retriever=retriever,
        retrieve_k=int(rcfg.get("k", 4)),
        prompt_budget_tokens=int(bcfg.get("prompt_budget_tokens", 2400)),
    )

    def notify(task_id: int, title: str):
        mem.add("reminder", f"#{task_id} {title}", tags="reminder")
        asyncio.run_coroutine_threadsafe(
            _broadcast({"type": "reminder", "id": task_id, "title": title}),
            loop,
        )

    def execute(outcome: dict):
        # The scheduler now dispatches through the gate itself, as actor
        # "scheduler", so this is a report rather than a second execution path
        # with its own private allowlist.
        mem.add("tool", json.dumps({"scheduled": True, **outcome}, default=str)[:2000])
        asyncio.run_coroutine_threadsafe(
            _broadcast(
                {
                    "type": "reply",
                    "text": f"Scheduled action {outcome.get('action')} ran.",
                }
            ),
            loop,
        )

    sched = Scheduler(
        db_path=cfg.get("memory_db_path", "data/intuition.db"),
        tz=cfg.get("timezone", "America/New_York"),
        tick_seconds=10,
        notify_cb=notify,
        execute_cb=execute,
        logger=logger,
        dispatcher=actions,
    )
    set_scheduler(sched)

    for d in cfg.get("hardware", {}).get("drivers", []):
        if d.get("name") == "led_strip":
            from plugins.led_strip import LEDStrip
            from core.actions import register_driver

            register_driver(
                LEDStrip(simulate=d.get("simulate", True), port=d.get("port"))
            )
        if d.get("name") == "gpu_nvml" and d.get("enabled", True):
            from plugins.gpu_nvml import GPUNVML
            from core.actions import register_driver

            register_driver(GPUNVML(enabled=True))
        if d.get("name") == "cpu_info" and d.get("enabled", True):
            from plugins.cpu_info import CPUInfo
            from core.actions import register_driver

            register_driver(CPUInfo())

    # ── Prediction ───────────────────────────────────────────────────────
    # The four literals that used to live here (0.9, 0.9, 0.85, 0.65) were never
    # compared to anything, so they could not be wrong. The predictor learns from
    # the episode log and falls back to exactly those heuristics until it has
    # seen enough to do better.
    ecfg = cfg.get("episodes", {}) or {}
    episodes = EpisodeLog(mem, enabled=bool(ecfg.get("enabled", True)))
    resolver = create_default_resolver(mem, enabled=lambda: episodes.enabled)
    sensor = ContextSensor(journal=get_journal())

    thresholds = load_thresholds(cfg.get("thresholds"))
    set_thresholds(thresholds)

    calibration_store = CalibrationStore(mem)
    calibrator = calibration_store.load()
    rule_store = RuleStore(mem)

    pcfg = cfg.get("prediction", {}) or {}
    predictor = Predictor(
        store=PredictorStore(mem),
        half_life_s=float(pcfg.get("half_life_s", 7 * 24 * 3600)),
        min_episodes=int(pcfg.get("min_episodes", 50)),
        calibrator=calibrator,
        rules=rule_store,
    )
    if predictor.seen == 0:
        # No saved state: relearn from the log rather than starting cold.
        predictor.fit(episodes.recent(limit=int(pcfg.get("replay_limit", 5000))))

    def prewarm(prediction):
        """Run the predicted action speculatively, as the anticipator.

        That actor is what confines this to free capabilities: it is guessing at
        something the user has not submitted and may never submit.
        """
        text = prediction.action
        conf = prediction.confidence

        def warm(name, args):
            return str(
                actions.dispatch(name, args, actor="anticipator", confidence=conf)
            )[:4000]

        t = text.strip()
        if t == "tree" or t.startswith("tree "):
            return (
                text,
                {
                    "reply": warm("list_tree", {"path": "."}),
                    "confidence": conf,
                    "why": prediction.why,
                    "action": text,
                },
            )
        if t == "ls":
            return (
                text,
                {
                    "reply": warm("list_dir", {"path": "."}),
                    "confidence": conf,
                    "why": prediction.why,
                    "action": text,
                },
            )
        if t.startswith("read file "):
            path = t[len("read file ") :].strip()
            return (
                text,
                {
                    "reply": warm("read_file", {"path": path}),
                    "confidence": conf,
                    "why": prediction.why,
                    "action": text,
                },
            )
        # Anything else is still worth predicting even though there is nothing
        # cheap to precompute: the hint alone has value.
        return (text, {"confidence": conf, "why": prediction.why, "action": text})

    a = cfg.get("anticipation", {}) or {}
    ant = Anticipator(
        prewarm_fn=prewarm,
        predictor=predictor,
        context_fn=lambda: sensor.snapshot(),
        enabled=bool(a.get("enabled", True)),
        debounce_ms=int(a.get("debounce_ms", 180)),
        match_threshold=float(a.get("match_threshold", 0.6)),
        thresholds=thresholds,
    )
    ant.start()

    # ── OS sandbox actions ───────────────────────────────────────────────
    # These used to be registered straight onto the plain registry, which meant
    # "shut down my pc" — typed or spoken — reached shutdown /s /t 30 with no
    # gate, no confirmation and no record. They now come with a declared cost.
    register_os_capabilities()

    # ── Voice ────────────────────────────────────────────────────────────
    voice: VoiceRecognizer | None = None
    voice_cfg = cfg.get("voice", {}) or {}
    voice_status = {
        "state": "disabled",
        "available": False,
        "text": "Voice is disabled in config.yaml.",
    }
    if voice_cfg.get("enabled", True):
        try:
            voice = VoiceRecognizer(
                model_size=voice_cfg.get("model", "base"),
                language=voice_cfg.get("language", "en"),
            )
            voice_status = {
                "state": "loading",
                "available": False,
                "text": "Preparing voice model (first setup may download model files)…",
            }
        except Exception as error:
            voice = None
            voice_status = {
                "state": "error",
                "available": False,
                "text": f"Voice setup failed: {error}",
            }

    _state.update(
        {
            "cfg": cfg,
            "brain": brain,
            "mem": mem,
            "sched": sched,
            "ant": ant,
            "voice": voice,
            "voice_status": voice_status,
            "voice_owner": None,
            "episodes": episodes,
            "sensor": sensor,
            "predictor": predictor,
            "calibrator": calibrator,
            "calibration_store": calibration_store,
            "thresholds": thresholds,
            "rules": rule_store,
            "retriever": retriever,
            "resolver": resolver,
        }
    )

    # Preload off the event loop to reduce the first voice request's latency.
    if voice:

        def _preload():
            try:
                voice.prepare()
                state, available, detail = "ready", True, "Voice is ready."
            except Exception as error:
                state, available = "error", False
                detail = f"Voice setup failed ({voice.model_size}): {error}"
                logger(detail)

            async def report():
                if _state.get("voice") is voice:
                    await _set_voice_status(state, available, detail)

            _queue_voice_callback(loop, report)

        threading.Thread(target=_preload, daemon=True, name="whisper-preload").start()

    yield

    # Read replaceable services from _state: /forget may have replaced the
    # initial predictor/anticipator while this lifespan was active.
    _state["ant"].stop()
    if _state.get("voice"):
        _state["voice"].stop_now()
    try:
        if _state["episodes"].enabled:
            _state["predictor"].save()
    except Exception:
        logger("could not save predictor state")
    try:
        sched.stop()
    except Exception:
        pass


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def _format_os_result(action: str, result: dict, kwargs: dict) -> str:
    """Render an OS action result using user-facing names and useful fields."""
    if result.get("error"):
        return f"⚠  {result['error']}"
    if action == "os_open_app":
        # Show the friendly name the user requested, not the full exe path
        return f"Opened {kwargs.get('name', result.get('launched', '?')).title()}"
    if action == "os_set_volume":
        return f"Volume set to {result.get('volume', kwargs.get('level', '?'))}%"
    if action == "os_take_screenshot":
        return f"Screenshot saved → {result.get('path', '?')}"
    if action == "os_system_info":
        r = result
        return (
            f"OS: {r.get('os')}\n"
            f"RAM: {r.get('ram_used_pct')} used of {r.get('ram_total_gb')} GB\n"
            f"Disk C:: {r.get('disk_used_pct')} used of {r.get('disk_total_gb')} GB\n"
            f"Uptime: {r.get('uptime_hours')} h"
        )
    if action == "os_list_processes":
        procs = result.get("processes", [])
        return "\n".join(
            f"{p['name']}  PID {p['pid']}  CPU {p['cpu']}  MEM {p['mem']}"
            for p in procs[:15]
        )
    if action == "os_kill_process":
        return f"Terminated {result.get('killed', '?')} (PID {result.get('pid', '?')})"
    if action == "os_get_clipboard":
        ct = result.get("text", "").strip()
        return f"Clipboard: {ct}" if ct else "Clipboard is empty"
    if action == "os_get_volume":
        return f"Current volume: {result.get('volume', '?')}%"
    if action == "os_set_brightness":
        return (
            f"Brightness set to {result.get('brightness', kwargs.get('level', '?'))}%"
        )
    if action == "os_get_brightness":
        return f"Current brightness: {result.get('brightness', '?')}%"
    if action == "os_get_battery":
        r = result
        charging = "charging" if r.get("charging") else "on battery"
        return (
            f"Battery: {r.get('percent')} ({charging}), {r.get('time_remaining', '')}"
        )
    if action == "os_get_network_info":
        r = result
        ifaces = ", ".join(
            f"{i['interface']} ({i['ip']})" for i in r.get("interfaces", [])
        )
        ssid = r.get("wifi_ssid")
        return f"Network: {ifaces or 'none'}" + (f"\nWi-Fi: {ssid}" if ssid else "")
    if action == "os_toggle_wifi":
        return f"Wi-Fi turned {kwargs.get('state', '?')}"
    if action == "os_list_windows":
        wins = result.get("windows", [])
        return "\n".join(f"{w['app']}: {w['title']}" for w in wins[:12])
    if action == "os_sleep_computer":
        return "Putting computer to sleep…"
    if action == "os_lock_screen":
        return "Screen locked."
    if action == "os_shutdown_computer":
        return f"Shutting down in {result.get('delay_sec', 30)} seconds. Type 'cancel shutdown' to abort."
    if action == "os_restart_computer":
        return f"Restarting in {result.get('delay_sec', 30)} seconds. Type 'cancel shutdown' to abort."
    if action == "os_cancel_shutdown":
        return "Shutdown/restart cancelled."
    return str(result)


# ── Gated dispatch ───────────────────────────────────────────────────────────


def _outcome(result):
    """Reduce action results to categories; None means approval is still pending."""
    if result is None:
        return "pending"
    if isinstance(result, dict):
        if result.get("cancelled"):
            return "cancelled"
        if result.get("denied"):
            return "denied"
        if result.get("error") or result.get("returncode", 0) != 0:
            return "error"
    return "ok"


def _park_confirmation(ws, token, capability, resume_token=None):
    """Bind a gate token to its socket, input revision, and feedback records."""
    session = _resolutions.get(ws)
    note = _active_notes.get(ws, {})
    note["outcome"] = "pending"
    _confirmations[token] = {
        "ws": ws,
        "revision": session.revision if session else None,
        "client_revision": _connections.get(ws, {}).get("client_revision"),
        "capability": capability,
        "resume_token": resume_token,
        "feedback_id": session.feedback_id if session else None,
        "episode_id": note.get("episode_id"),
    }
    return _confirmations[token]


def _finish_feedback(meta, outcome):
    """Complete the correction and episode records for a parked action."""
    store = getattr(_state.get("resolver"), "feedback", None)
    if store and meta.get("feedback_id"):
        store.record_outcome(meta["feedback_id"], outcome)
    episodes = _state.get("episodes")
    if episodes and meta.get("episode_id"):
        episodes.set_outcome(meta["episode_id"], outcome)


async def _invalidate_input(ws, *, notify=False, resolution=True):
    """An edit/disconnect revokes approvals, including suspended model actions."""
    for token, meta in list(_confirmations.items()):
        if meta["ws"] is not ws:
            continue
        _confirmations.pop(token, None)
        actions.confirm(token, granted=False)
        resume = meta.get("resume_token")
        if resume:
            _state["brain"]._suspended.pop(resume, None)
        _finish_feedback(meta, "cancelled")
    if resolution and ws in _resolutions:
        _resolutions[ws].invalidate()
    if notify:
        await ws.send_json({"type": "input_invalidated"})


async def _show_resolution(ws, text, client_revision=None):
    """Publish a core selection snapshot tagged with the renderer's draft ID."""
    meta = _connections.setdefault(ws, {})
    if meta.get("text") != text or meta.get("client_revision") != client_revision:
        # With changed text, session.update retains the prior display long
        # enough to record a manual edit. Either change still revokes approvals.
        await _invalidate_input(ws, resolution=meta.get("text") == text)
    meta.update(text=text, client_revision=client_revision)
    session = _resolutions.setdefault(ws, CorrectionSession(_state["resolver"]))
    # No LLM or candidate subprocess is consulted by this hot path.
    session.update(text, context={"cwd": os.getcwd(), "ts": time.time()})
    await ws.send_json(
        {"type": "resolution", **session.snapshot(), "client_revision": client_revision}
    )


async def _reset_learning_views():
    """Rebuild learning views after persisted learning is erased or disabled.

    Invalidate every peer before replacing the resolver, then stop the old
    worker before starting a predictor that can no longer read forgotten state.
    """
    for peer in list(_resolutions):
        await _invalidate_input(peer, notify=True)
    mem = _state["mem"]
    episodes = _state["episodes"]
    _state["resolver"] = create_default_resolver(mem, enabled=lambda: episodes.enabled)
    for peer in list(_resolutions):
        _resolutions[peer] = CorrectionSession(_state["resolver"])
        _windows[peer] = PredictionWindow()
    cfg = _state["cfg"].get("prediction", {}) or {}
    calibrator = _state["calibration_store"].load()
    predictor = Predictor(
        store=PredictorStore(mem) if episodes.enabled else None,
        half_life_s=float(cfg.get("half_life_s", 7 * 24 * 3600)),
        min_episodes=int(cfg.get("min_episodes", 50)),
        calibrator=calibrator,
        rules=_state["rules"],
    )
    _state.update(predictor=predictor, calibrator=calibrator)
    old_ant = _state["ant"]
    old_ant.stop()
    old_ant.invalidate()
    _state["sensor"] = ContextSensor(journal=get_journal())
    cfg = _state["cfg"].get("anticipation", {}) or {}
    ant = Anticipator(
        prewarm_fn=old_ant.prewarm_fn,
        predictor=predictor,
        context_fn=lambda: _state["sensor"].snapshot(),
        enabled=bool(cfg.get("enabled", True)),
        debounce_ms=int(cfg.get("debounce_ms", 180)),
        match_threshold=float(cfg.get("match_threshold", 0.6)),
        thresholds=_state["thresholds"],
    )
    _state["ant"] = ant
    ant.start()


async def _dispatch(
    ws: WebSocket, name: str, kwargs: dict, actor: str = "user", confidence: float = 1.0
):
    """Run one action through the gate, surfacing a confirmation if it needs one.

    Returns the action's result dict, or None when the action was parked awaiting
    a human. The parked case is the important one: nothing has run yet, and
    nothing will until a `confirm` message comes back with the token.
    """
    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(
        _executor,
        lambda: actions.dispatch(name, kwargs, actor=actor, confidence=confidence),
    )
    note = _active_notes.get(ws)
    if note is not None:
        note.update(
            capability=name,
            outcome=_outcome(res),
            exit_code=res.get("returncode") if isinstance(res, dict) else None,
        )
    if isinstance(res, dict) and res.get("needs_confirmation"):
        binding = _park_confirmation(ws, res["token"], res["capability"])
        await ws.send_json(
            {
                "type": "confirm_request",
                "token": res["token"],
                "capability": res["capability"],
                "args": res.get("args", {}),
                "reason": res.get("reason", ""),
                "reversibility": res.get("reversibility", ""),
                "summary": res.get("summary", ""),
                "client_revision": binding["client_revision"],
            }
        )
        return None
    return res


async def _resolve_confirmation(ws: WebSocket, token: str, granted: bool):
    """Resolve an owned, current gate token and finish its action or model turn."""
    loop = asyncio.get_running_loop()
    meta = _confirmations.get(token)
    session = _resolutions.get(ws)
    if (
        not meta
        or meta["ws"] is not ws
        or (session and meta["revision"] != session.revision)
    ):
        await ws.send_json(
            {
                "type": "error",
                "text": "Confirmation expired, changed, or belongs to another connection.",
            }
        )
        return
    _confirmations.pop(token, None)

    # A confirmation raised from inside the tool loop resumes that loop rather
    # than just running the action, so the model gets to see how it was answered
    # and finish its turn either way.
    resume_token = meta.get("resume_token")
    if resume_token is not None:
        brain: Brain = _state["brain"]
        out = await loop.run_in_executor(
            _executor,
            lambda: brain.resume(resume_token, granted, on_token=_token_sink(ws, loop)),
        )
        _finish_feedback(
            meta, "ok" if granted and not out.get("error") else "cancelled"
        )
        await _send_brain_result(ws, out)
        return

    res = await loop.run_in_executor(
        _executor, lambda: actions.confirm(token, granted=granted)
    )
    _finish_feedback(meta, _outcome(res))
    mem: Memory = _state["mem"]
    if res.get("error"):
        await ws.send_json({"type": "error", "text": res["error"]})
        return
    if res.get("cancelled"):
        await ws.send_json({"type": "reply", "text": f"Cancelled {res['capability']}."})
        return
    pending = meta["capability"]
    text = (
        _format_os_result(pending, res, {})
        if pending.startswith("os_")
        else _summarise(res)
    )
    await ws.send_json({"type": "reply", "text": text})
    await ws.send_json(
        {
            "type": "status",
            "safe_mode": is_safe_mode(),
            "tasks_count": len(mem.list_open()),
        }
    )


def _token_sink(ws: WebSocket, loop):
    """Forward model tokens to the HUD as they arrive.

    With a tool loop a single turn can take several seconds per iteration, so
    without this the user watches a still panel and assumes it has hung.
    """

    def sink(piece: str):
        asyncio.run_coroutine_threadsafe(
            ws.send_json({"type": "token", "text": piece}), loop
        )

    return sink


async def _send_brain_result(ws: WebSocket, out: dict):
    """Deliver a Brain result: a reply, or a confirmation that suspended it."""
    if out.get("needs_confirmation"):
        binding = _park_confirmation(
            ws, out["confirm_token"], out["capability"], out["resume_token"]
        )
        await ws.send_json(
            {
                "type": "confirm_request",
                "token": out["confirm_token"],
                "capability": out["capability"],
                "args": out.get("args", {}),
                "reason": out.get("reason", ""),
                "reversibility": out.get("reversibility", ""),
                "summary": "",
                "client_revision": binding["client_revision"],
            }
        )
        return
    await ws.send_json(
        {"type": "reply", "text": out.get("reply", ""), "plan": out.get("plan", [])}
    )


def _summarise(res: dict) -> str:
    """Produce the generic response used when no action-specific renderer exists."""
    if not isinstance(res, dict):
        return str(res)
    if res.get("error"):
        return f"⚠  {res['error']}"
    return json.dumps(res, indent=2, default=str)


_KNOWN_CMDS = list(KNOWN_COMMANDS)


def _fuzzy_cmd(base: str) -> str:
    """Compatibility/evaluation only; submission never calls this matcher."""
    return legacy_fuzzy_slash(base)


async def _handle_command(ws: WebSocket, text: str):
    """Handle an already committed IntuitionOS slash command without correction."""
    mem: Memory = _state["mem"]
    brain: Brain = _state["brain"]
    loop = asyncio.get_running_loop()

    if text == "/help":
        await ws.send_json(
            {
                "type": "reply",
                "text": "Commands: " + ", ".join(_KNOWN_CMDS) + "\n"
                "Type gti status or git statsu to preview a correction. "
                "Up/Down or Ctrl+N/P selects alternatives; Escape keeps the original; "
                "Enter submits the displayed choice. Shell commands use the capability gate.\n"
                '/exec <command> (or /exec "command" [cwd]); /write <path> "text"; '
                "/read <path>; /hw schema <device>; /task_payload '{JSON}' <when>.",
            }
        )
        return

    if text == "/exit":
        await _invalidate_input(ws)
        await ws.send_json(
            {"type": "exit", "text": "HUD hidden. Backend remains available."}
        )
        return

    if text == "/config":
        await ws.send_json(
            {
                "type": "reply",
                "text": yaml.safe_dump(_state.get("cfg", {}), sort_keys=False),
            }
        )
        return

    if text == "/reload":
        cfg = _load_config()
        # Live settings and prompts are safe to replace. Database, voice and
        # hardware wiring require a restart rather than orphaning live workers.
        brain.system_prompt = _read_text(
            cfg.get("system_prompt_path", "config/system_prompt.txt")
        )
        brain.schema = _read_json(
            cfg.get("planner_schema_path", "config/planner_schema.json")
        )
        _state["cfg"] = cfg
        episodes = _state["episodes"]
        episodes.enabled = bool((cfg.get("episodes") or {}).get("enabled", True))
        thresholds = load_thresholds(cfg.get("thresholds"))
        set_thresholds(thresholds)
        _state["thresholds"] = thresholds
        _state["ant"].enabled = bool(
            (cfg.get("anticipation") or {}).get("enabled", True)
        )
        await _reset_learning_views()
        await ws.send_json(
            {
                "type": "reply",
                "text": "Reloaded prompts, logging, anticipation and execution thresholds. "
                "Restart the backend for model, database, voice or hardware changes.",
            }
        )
        return

    if text == "/memory":
        rows = mem.recent(limit=12)
        await ws.send_json(
            {
                "type": "memory",
                "rows": [
                    {"id": r[0], "ts": r[1], "role": r[2], "text": r[3], "tags": r[4]}
                    for r in reversed(rows)
                ],
            }
        )
        return

    if text == "/tasks":
        await ws.send_json({"type": "tasks", "rows": mem.list_open()})
        return

    if text == "/dream":
        episodes = _state.get("episodes")
        rules = _state.get("rules")
        if not (episodes and rules):
            await ws.send_json({"type": "error", "text": "consolidation unavailable"})
            return
        await ws.send_json({"type": "thinking"})
        ccfg = (_state.get("cfg") or {}).get("consolidation", {}) or {}
        report = await loop.run_in_executor(
            _executor,
            lambda: consolidate(
                episodes.recent(limit=int(ccfg.get("window", 2000))),
                rules,
                llm=_state["brain"].llm,
                min_support=int(ccfg.get("min_support", 4)),
                min_confidence=float(ccfg.get("min_confidence", 0.5)),
                calibrator=_state.get("calibrator"),
                calibration_store=_state.get("calibration_store"),
            ),
        )
        await ws.send_json({"type": "reply", "text": report.summary()})
        return

    if text.startswith("/rules"):
        rules = _state.get("rules")
        if not rules:
            await ws.send_json({"type": "error", "text": "rule store unavailable"})
            return
        parts = text.split()
        if len(parts) >= 3 and parts[1] == "delete":
            try:
                rule_id = int(parts[2])
            except ValueError:
                await ws.send_json(
                    {"type": "error", "text": "usage: /rules delete <id>"}
                )
                return
            ok = rules.delete(rule_id)
            await ws.send_json(
                {
                    "type": "reply",
                    "text": f"Deleted rule #{rule_id}."
                    if ok
                    else f"No rule #{rule_id}.",
                }
            )
            return
        show_all = len(parts) >= 2 and parts[1] in ("--all", "all")
        await ws.send_json(
            {"type": "reply", "text": render_rules(rules.all(active_only=not show_all))}
        )
        return

    if text.startswith("/save "):
        note = text[6:].strip().strip('"')
        mem.add("note", note, tags="note")
        await ws.send_json({"type": "reply", "text": "Saved."})
        return

    if text.startswith("/recall "):
        term = text[8:].strip().strip('"')
        retriever = _state.get("retriever")
        sensor = _state.get("sensor")
        snapshot = sensor.snapshot() if sensor else None

        # Notes first. Appendix A #16: Brain writes both sides of every
        # conversation into `mem`, so an unfiltered search buries the notes under
        # the transcript.
        hits = retriever.search(term, limit=12, roles=("note",)) if retriever else []
        if not hits and retriever:
            hits = retriever.search(term, limit=12)
        if not hits:
            hits = [
                type(
                    "R",
                    (),
                    {"id": r[0], "ts": r[1], "role": r[2], "text": r[3], "tags": r[4]},
                )()
                for r in mem.search(term, limit=12)
            ]

        await ws.send_json(
            {
                "type": "memory",
                "rows": [
                    {
                        "id": h.id,
                        "ts": h.ts,
                        "role": h.role,
                        "text": h.text,
                        "tags": h.tags,
                    }
                    for h in hits
                ],
            }
        )

        # And show what the situation alone would have surfaced, which is the
        # point of cue-driven retrieval: you should not have needed to ask.
        if retriever and snapshot is not None:
            cued = [
                n
                for n in retriever.retrieve("", snapshot, k=2)
                if all(n.id != h.id for h in hits)
            ]
            if cued:
                await ws.send_json(
                    {
                        "type": "reply",
                        "text": "Also relevant here right now:"
                        + chr(10)
                        + chr(10).join(f"- {n.text}" for n in cued),
                    }
                )
        return

    if text.startswith("/done "):
        try:
            tid = int(text.split(" ", 1)[1])
            actions.call("complete_task", task_id=tid)
            await ws.send_json({"type": "reply", "text": f"Task {tid} marked done."})
            await ws.send_json(
                {
                    "type": "status",
                    "tasks_count": len(mem.list_open()),
                    "safe_mode": is_safe_mode(),
                }
            )
        except Exception:
            await ws.send_json({"type": "error", "text": "usage: /done <id>"})
        return

    if text.startswith("/delete "):
        try:
            tid = int(text.split(" ", 1)[1])
        except Exception:
            await ws.send_json({"type": "error", "text": "usage: /delete <id>"})
            return
        res = await _dispatch(ws, "delete_task", {"task_id": tid})
        if res is None:
            return  # parked awaiting confirmation
        if res.get("error"):
            await ws.send_json({"type": "error", "text": res["error"]})
        else:
            await ws.send_json({"type": "reply", "text": f"Task {tid} deleted."})
        return

    if text.startswith("/snooze "):
        try:
            parts = text.split(" ")
            actions.call("snooze_task", task_id=int(parts[1]), delta=parts[2])
            await ws.send_json(
                {"type": "reply", "text": f"Task {parts[1]} snoozed {parts[2]}."}
            )
        except Exception:
            await ws.send_json(
                {"type": "error", "text": "usage: /snooze <id> <15m|2h|1d>"}
            )
        return

    if text.startswith("/safe"):
        parts = text.split()
        if len(parts) >= 2:
            res = set_safe_mode_action(state=parts[1])
            safe_on = is_safe_mode()
            await ws.send_json(
                {
                    "type": "status",
                    "safe_mode": safe_on,
                    "tasks_count": len(mem.list_open()),
                    "text": res.get("result", ""),
                }
            )
        else:
            await ws.send_json({"type": "error", "text": "usage: /safe on|off"})
        return

    if re.match(r"^/exec\s", text):
        rest = text[6:]
        cmd, cwd = None, "."
        action_name = "run_command"
        if rest.startswith('"'):
            idx = rest.find('"', 1)
            if idx != -1:
                cmd = rest[1:idx]
                cwd = rest[idx + 1 :].strip() or "."
                action_name = "run_local"
        if not cmd:
            cmd = rest
        res = await _dispatch(ws, action_name, {"cmd": cmd, "cwd": cwd})
        if res is None:
            return  # parked awaiting confirmation
        out = res.get("stdout", "") or res.get("error", "") or json.dumps(res)
        await ws.send_json({"type": "reply", "text": out})
        return

    if text.startswith("/write "):
        match = re.fullmatch(
            r'/write\s+(?:"([^"]+)"|(\S+))\s+"(.*)"\s*', text, re.DOTALL
        )
        if not match:
            await ws.send_json({"type": "error", "text": 'usage: /write <path> "text"'})
            return
        result = await _dispatch(
            ws, "write_file", {"path": match[1] or match[2], "text": match[3]}
        )
        if result is not None:
            await ws.send_json({"type": "reply", "text": _summarise(result)})
        return

    if text.startswith("/task_payload "):
        rest = text[len("/task_payload ") :].lstrip()
        try:
            quote = rest[0] if rest.startswith(("'", '"')) else ""
            payload, end = json.JSONDecoder().raw_decode(rest[1:] if quote else rest)
            tail = rest[1 + end :] if quote else rest[end:]
            if quote:
                if not tail.startswith(quote):
                    raise ValueError("missing closing quote")
                tail = tail[1:]
            when = tail.strip()
            if not when:
                raise ValueError("missing time")
            result = await _dispatch(
                ws,
                "create_task",
                {"text": None, "when": when, "repeat": "", "payload": payload},
            )
            if result is not None:
                await ws.send_json({"type": "reply", "text": _summarise(result)})
        except (ValueError, IndexError) as error:
            await ws.send_json(
                {
                    "type": "error",
                    "text": f"usage: /task_payload '{{JSON}}' <when>: {error}",
                }
            )
        return

    if text.startswith("/read "):
        path = text.split(" ", 1)[1].strip()
        res = await loop.run_in_executor(
            _executor, lambda: actions.call("read_file", path=path)
        )
        await ws.send_json(
            {"type": "reply", "text": res.get("text", res.get("error", ""))}
        )
        return

    if text == "/hw":
        await ws.send_json(
            {"type": "reply", "text": json.dumps(actions.call("hw_list"), indent=2)}
        )
        return

    if text.startswith("/hw schema "):
        result = await _dispatch(
            ws, "hw_schema", {"device": text[len("/hw schema ") :]}
        )
        if result is not None:
            await ws.send_json({"type": "reply", "text": _summarise(result)})
        return

    if text == "/actions":
        await ws.send_json({"type": "reply", "text": ", ".join(sorted(actions.names))})
        return

    if text == "/calibration":
        episodes = _state.get("episodes")
        if not episodes:
            await ws.send_json({"type": "error", "text": "episode log unavailable"})
            return
        report = await loop.run_in_executor(
            _executor, lambda: reliability(episodes.shown_predictions())
        )
        await ws.send_json({"type": "reply", "text": report.table()})
        return

    if text == "/thresholds":
        th = _state.get("thresholds") or {}
        lines = [
            f"  {k:<14} {'never' if v is None else format(float(v), '.2f')}"
            for k, v in th.items()
        ]
        await ws.send_json(
            {
                "type": "reply",
                "text": "Cost-gated thresholds" + chr(10) + "\n".join(lines),
            }
        )
        return

    if text == "/forget":
        episodes = _state.get("episodes")
        if not episodes:
            await ws.send_json({"type": "error", "text": "episode log unavailable"})
            return
        n = await loop.run_in_executor(_executor, episodes.forget)
        await _reset_learning_views()
        await ws.send_json({"type": "reply", "text": f"Forgot {n} episode(s)."})
        return

    if text == "/episodes":
        episodes = _state.get("episodes")
        if not episodes:
            await ws.send_json({"type": "error", "text": "episode log unavailable"})
            return
        rows = episodes.recent(limit=15)
        if not rows:
            await ws.send_json(
                {
                    "type": "reply",
                    "text": "No episodes recorded yet."
                    if episodes.enabled
                    else "Episode logging is disabled in config.yaml.",
                }
            )
            return
        lines = []
        for e in rows:
            when = datetime.datetime.fromtimestamp(e.ts).strftime("%H:%M:%S")
            hint = ""
            if e.accepted_prediction is not None:
                hint = (
                    "  hint taken"
                    if e.accepted_prediction
                    else f"  hint ignored ({e.predicted})"
                )
            lines.append(f"{when}  {e.action}{hint}")
        await ws.send_json({"type": "reply", "text": "\n".join(lines)})
        return

    if text == "/undo":
        res = await loop.run_in_executor(_executor, undo_last)
        if res.get("error"):
            await ws.send_json({"type": "error", "text": res["error"]})
        else:
            await ws.send_json(
                {
                    "type": "reply",
                    "text": f"Undid {res['capability']} (journal #{res['id']}).",
                }
            )
        return

    if text.startswith("/journal"):
        parts = text.split()
        limit = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 15
        rows = journal_recent(limit=limit).get("rows", [])
        if not rows:
            await ws.send_json({"type": "reply", "text": "The journal is empty."})
            return
        lines = []
        for r in rows:
            when = datetime.datetime.fromtimestamp(r["ts"]).strftime("%H:%M:%S")
            mark = " (undone)" if r["undone_at"] else (" ↩" if r["undo"] else "")
            lines.append(
                f"#{r['id']} {when} {r['actor']}/{r['capability']} "
                f"{r['decision']}{'/' + r['outcome'] if r['outcome'] else ''}{mark}"
            )
        await ws.send_json({"type": "reply", "text": "\n".join(lines)})
        return

    if text == "/capabilities":
        lines = [
            f"{c['name']:<24} {c['reversibility']:<13}"
            f"{'confirm' if c['requires_confirmation'] else '':<9}{c['summary']}"
            for c in capabilities.manifest()
        ]
        await ws.send_json({"type": "reply", "text": "\n".join(lines)})
        return

    if ws in _active_notes:
        _active_notes[ws]["outcome"] = "error"
    await ws.send_json({"type": "error", "text": f"Unknown command: {text}"})


async def _handle_input(ws: WebSocket, text: str):
    """Encode one episode, then handle the input.

    This is the involuntary part: the row is written because the user submitted
    something, not because they asked for it to be remembered. The capability and
    the outcome are filled in by the handler as they become known.
    """
    episodes = _state.get("episodes")
    sensor = _state.get("sensor")
    window = _windows.get(ws)

    signals = window.take(text) if window else {}
    ctx = sensor.snapshot() if sensor else None
    note: dict = {}

    # Forget must stay forgotten; raw shell arguments are deliberately absent
    # from correction feedback (handled by CorrectionSession's token store).
    learned = learning_text(text)
    for key in ("keystroke_prefix", "predicted"):
        if signals.get(key):
            signals[key] = learning_text(signals[key])
    episode_id = (
        episodes.record(learned, ctx, **signals)
        if episodes and text.strip() != "/forget"
        else None
    )
    note["episode_id"] = episode_id
    _active_notes[ws] = note

    try:
        await _handle_input_inner(ws, text, note)
    except Exception:
        if episodes and episode_id:
            episodes.set_outcome(episode_id, "error")
        raise
    finally:
        _active_notes.pop(ws, None)

    if text.strip() in ("/forget", "/reload"):
        return

    if sensor:
        sensor.note_submission(learned, exit_code=note.get("exit_code"))
    if episodes and episode_id:
        if note.get("capability"):
            episodes.set_capability(episode_id, note["capability"])
        episodes.set_outcome(episode_id, note.get("outcome", "ok"))

    # Online update: this submission is the ground truth for whatever was
    # predicted a moment ago, including the times the prediction was wrong.
    predictor = _state.get("predictor")
    if predictor is not None and episodes and episodes.enabled:
        from core.episodes import Episode as _Episode

        predictor.update(
            _Episode(
                ts=time.time(),
                action=learned,
                context=ctx,
                keystroke_prefix=signals.get("keystroke_prefix") or learned,
            )
        )

    # A write may have invalidated something prewarmed against the old state.
    session = _resolutions.get(ws)
    if session:
        session.outcome(note.get("outcome", "ok"))
    if note.get("capability") in (
        "write_file",
        "/write",
        "/exec",
        "run_local",
        "run_command",
    ):
        ant = _state.get("ant")
        if ant:
            ant.invalidate()


async def _handle_input_inner(ws: WebSocket, text: str, note: dict):
    """Route committed text through shell, built-in, OS-intent, or model paths."""
    brain: Brain = _state["brain"]
    mem: Memory = _state["mem"]
    ant: Anticipator = _state["ant"]
    loop = asyncio.get_running_loop()

    # Parsing/ranking never authorizes a replacement here: `text` is exactly
    # the raw input or a candidate already displayed and explicitly committed.
    resolution = _state["resolver"].resolve(text, context={"cwd": os.getcwd()})
    if not text.lstrip().startswith("/") and (
        (resolution.namespace == "git" and resolution.status != "unsupported")
        or (resolution.namespace == "shell" and resolution.status == "exact")
    ):
        result = await _dispatch(ws, "run_command", {"cmd": text, "cwd": "."})
        if result is not None:
            await ws.send_json(
                {
                    "type": "reply",
                    "text": result.get("stdout")
                    or result.get("stderr")
                    or result.get("error")
                    or _summarise(result),
                }
            )
        return

    if text.lstrip().startswith("/"):
        text = text.lstrip()
        if len(text.split()) == 1:
            text = text.rstrip()
        note["capability"] = text.split()[0]
        await _handle_command(ws, text)
        return

    if resolution.status in ("correction", "incomplete", "ambiguous"):
        note["outcome"] = "error"
        await ws.send_json(
            {
                "type": "error",
                "text": "Original input kept unchanged; command is not recognized. Choose a displayed correction or edit it.",
            }
        )
        return

    if resolution.status == "unsupported" and resolution.namespace in ("shell", "git"):
        note["outcome"] = "error"
        await ws.send_json({"type": "error", "text": resolution.reason})
        return

    m = re.match(r"^remind\s+me\s+(.+?)\s+((?:in|at)\s+\S.*)$", text, re.IGNORECASE)
    if m:
        note["capability"] = "create_task"
        res = actions.call(
            "create_task", text=m.group(1).strip(), when=m.group(2).strip()
        )
        if res.get("ok"):
            due_str = datetime.datetime.fromtimestamp(res["due_ts"]).strftime(
                "%I:%M %p, %b %d"
            )
            reply = f'Reminder set: "{m.group(1).strip()}" at {due_str}'
        else:
            reply = f"⚠  {res.get('error', 'Could not parse time')}"
        await ws.send_json({"type": "reply", "text": reply})
        return

    if text.strip() == "ls":
        note["capability"] = "list_dir"
        res = await loop.run_in_executor(
            _executor, lambda: actions.call("list_dir", path=".")
        )
        if isinstance(res, list):
            lines = "\n".join(
                f"{'📁' if r['type'] == 'dir' else '📄'} {r['name']}" for r in res
            )
        else:
            lines = str(res)
        await ws.send_json({"type": "reply", "text": lines})
        return

    if text.strip() == "tree":
        note["capability"] = "list_tree"
        res = await loop.run_in_executor(
            _executor, lambda: actions.call("list_tree", path=".")
        )

        def _fmt(items, indent=0):
            out = []
            for item in items:
                prefix = "  " * indent
                if item["type"] == "dir":
                    out.append(f"{prefix}📁 {item['name']}/")
                    out.extend(_fmt(item.get("children", []), indent + 1))
                else:
                    out.append(f"{prefix}📄 {item['name']}")
            return out

        lines = "\n".join(_fmt(res)) if isinstance(res, list) else str(res)
        await ws.send_json({"type": "reply", "text": lines})
        return

    # ── OS intent (runs before LLM so voice/text can control Windows) ────
    os_intent = _try_os_intent(text)
    if os_intent:
        action_name, kwargs = os_intent
        note["capability"] = action_name
        await ws.send_json({"type": "thinking"})
        # A regex match on speech is a guess, not an instruction. Anything the
        # manifest calls irreversible now comes back parked for confirmation
        # instead of firing, which is what stops "close this" from killing a
        # process and "shut down" from being heard across the room.
        try:
            result = await _dispatch(ws, action_name, kwargs, actor="user")
        except Exception as e:
            await ws.send_json({"type": "error", "text": str(e)})
            return
        if result is None:
            return  # parked awaiting confirmation
        reply = _format_os_result(action_name, result, kwargs)
        mem.add("user", text)
        mem.add("assistant", reply)
        await ws.send_json({"type": "reply", "text": reply})
        return

    pre = ant.try_serve(text)
    if pre and isinstance(pre, dict) and ("reply" in pre or "plan" in pre):
        mem.add("assistant", pre.get("reply", ""))
        await ws.send_json(
            {
                "type": "reply",
                "text": pre.get("reply", ""),
                "plan": pre.get("plan", []),
                "from_cache": True,
            }
        )
        return

    await ws.send_json({"type": "thinking"})
    try:
        sink = _token_sink(ws, loop)
        sensor = _state.get("sensor")
        snapshot = sensor.snapshot() if sensor else None
        out = await loop.run_in_executor(
            _executor, lambda: brain.step(text, context=snapshot, on_token=sink)
        )
        await _send_brain_result(ws, out)
    except Exception as e:
        await ws.send_json({"type": "error", "text": f"LLM error: {e}"})


async def _maybe_send_anticipation(ws: WebSocket, text: str):
    """Reveal a sufficiently confident warmed prediction and record its display."""
    await asyncio.sleep(0.35)
    ant = _state.get("ant")
    if not ant:
        return
    pre = ant.try_serve(text)
    if pre and isinstance(pre, dict):
        # Prewarming happens above the "free" threshold; revealing needs the
        # higher "reveal" one, because a wrong hint costs the user attention
        # rather than a few background milliseconds.
        if float(pre.get("confidence", 0.0)) < ant.reveal_threshold:
            return
        try:
            await ws.send_json(
                {
                    "type": "anticipation",
                    "data": pre,
                    "text": text,
                    "reveal_threshold": ant.reveal_threshold,
                }
            )
        except Exception:
            return
        # Recorded only once the send succeeded: a hint that never reached the
        # user must not be counted against them for ignoring it.
        window = _windows.get(ws)
        if window:
            window.note_shown(text, pre.get("confidence"))


async def _submit_input(ws, data):
    """Validate displayed text before dispatch, or preserve exact legacy input.

    A correction commitment accepts a selection only. Capability permission is
    checked later by the selected action's normal dispatch path.
    """
    original = data.get("text", "")
    if not isinstance(original, str) or not original.strip():
        return
    session = _resolutions[ws]
    if "token" in data:
        # Validate the untouched buffer AND selected rendered command. An old
        # candidate cannot borrow new argument text or another socket's token.
        try:
            snapshot = session.snapshot()
            index = data.get("candidate_index")
            expected = (
                snapshot["original"]
                if index is None
                else snapshot["candidates"][index]["text"]
            )
            if data.get("selected_text", expected) != expected:
                raise ValueError("Displayed command does not match this selection")
            text = session.commit(
                original,
                token=data["token"],
                revision=data.get("revision"),
                candidate_index=index,
            )
        except (ValueError, KeyError, TypeError, IndexError) as error:
            await ws.send_json(
                {
                    "type": "error",
                    "text": f"Stale or invalid correction: {error}. Review the current input again.",
                }
            )
            return
    else:
        # Older clients can still submit exact commands. A misspelling is only
        # offered for review, never silently fixed in the submission handler.
        resolution = _state["resolver"].resolve(original, context={"cwd": os.getcwd()})
        if resolution.candidates and resolution.status != "exact":
            await _show_resolution(ws, original, data.get("client_revision"))
            return
        await _invalidate_input(ws)
        _connections[ws].update(
            text=original, client_revision=data.get("client_revision")
        )
        session.update(original, context={"cwd": os.getcwd()})
        session.commit(original, token=session.token, revision=session.revision)
        text = original
    await _handle_input(ws, text)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    """Own one HUD connection and revoke its parked work when it disconnects."""
    await ws.accept()
    _clients.add(ws)
    _windows[ws] = PredictionWindow()
    _resolutions[ws] = CorrectionSession(_state["resolver"])
    _connections[ws] = {"text": "", "client_revision": None}

    mem: Memory = _state["mem"]
    await ws.send_json(
        {
            "type": "status",
            "safe_mode": is_safe_mode(),
            "tasks_count": len(mem.list_open()),
            "version": "1.0",
            "voice": _voice_info(),
        }
    )

    try:
        while True:
            data = await ws.receive_json()
            t = data.get("type")

            if t == "buffer":
                buf = data.get("text", "")
                if not isinstance(buf, str):
                    continue
                if "client_revision" in data:
                    await _show_resolution(ws, buf, data["client_revision"])
                else:
                    await _invalidate_input(ws)
                    _connections[ws]["text"] = buf
                _state["ant"].update_buffer(buf)
                window = _windows.get(ws)
                if window:
                    window.note_keystroke(buf)
                asyncio.create_task(_maybe_send_anticipation(ws, buf))

            elif t == "resolve":
                await _show_resolution(
                    ws, data.get("text", ""), data.get("client_revision")
                )

            elif t == "input":
                await _submit_input(ws, data)

            elif t == "confirm":
                await _resolve_confirmation(
                    ws, data.get("token", ""), bool(data.get("granted"))
                )

            elif t == "get_status":
                await ws.send_json(
                    {
                        "type": "status",
                        "safe_mode": is_safe_mode(),
                        "tasks_count": len(mem.list_open()),
                        "voice": _voice_info(),
                    }
                )

            elif t == "voice_start":
                voice = _state.get("voice")
                info = _voice_info()
                if not voice or not info["available"]:
                    await ws.send_json(
                        {
                            "type": "error",
                            "source": "voice",
                            "text": info["text"],
                        }
                    )
                    await ws.send_json({"type": "voice_status", **info})
                elif voice.is_busy():
                    await ws.send_json(
                        {
                            "type": "error",
                            "source": "voice",
                            "text": "Voice is already recording or transcribing.",
                        }
                    )
                else:
                    _loop = asyncio.get_running_loop()
                    _state["voice_owner"] = ws

                    def _on_silence():
                        async def report():
                            await _set_voice_status(
                                "transcribing", True, "Transcribing speech…"
                            )
                            await ws.send_json(
                                {"type": "voice_recording", "active": False}
                            )

                        _queue_voice_callback(_loop, report)

                    def _on_complete(recognized_text: str):
                        async def _finish():
                            _state["voice_owner"] = None
                            await _set_voice_status("ready", True, "Voice is ready.")
                            if recognized_text:
                                await ws.send_json(
                                    {"type": "voice_text", "text": recognized_text}
                                )
                                # Speech is editable draft input, subject to the
                                # same visible correction/Enter flow as typing.
                            else:
                                await ws.send_json(
                                    {
                                        "type": "error",
                                        "source": "voice",
                                        "text": "No speech detected. Check the selected Windows input device and microphone level.",
                                    }
                                )

                        _queue_voice_callback(_loop, _finish)

                    def _on_error(detail):
                        async def report():
                            _state["voice_owner"] = None
                            await _set_voice_status("error", True, detail)
                            await ws.send_json(
                                {"type": "error", "source": "voice", "text": detail}
                            )

                        _queue_voice_callback(_loop, report)

                    try:
                        # Announce capture before starting its worker: a fast
                        # device failure must never be followed by stale "on".
                        await _set_voice_status("recording", True, "Listening…")
                        await ws.send_json({"type": "voice_recording", "active": True})
                        voice.start_recording_vad(
                            on_silence=_on_silence,
                            on_complete=_on_complete,
                            on_error=_on_error,
                        )
                    except Exception as e:
                        _state["voice_owner"] = None
                        await _set_voice_status("error", True, f"Microphone error: {e}")
                        await ws.send_json(
                            {
                                "type": "error",
                                "source": "voice",
                                "text": f"Microphone error: {e}",
                            }
                        )

            elif t == "voice_stop":
                voice = _state.get("voice")
                if voice and voice.is_recording():
                    voice.stop_now()  # signals VAD to stop; callbacks still fire

    except WebSocketDisconnect:
        pass
    finally:
        if _state.get("voice_owner") is ws and _state.get("voice"):
            _state["voice"].stop_now()
        await _invalidate_input(ws)
        _clients.discard(ws)
        _windows.pop(ws, None)
        _resolutions.pop(ws, None)
        _connections.pop(ws, None)


@app.get("/health")
async def health():
    """Report that the local API is responsive without querying other services."""
    return {"ok": True, "version": "1.0"}
