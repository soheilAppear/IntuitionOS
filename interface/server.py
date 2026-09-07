import asyncio
import datetime
import difflib
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
from core.calibration import CalibrationStore, Calibrator, load_thresholds, reliability
from core.capabilities import capabilities, is_safe_mode
from core.context import ContextSensor
from core.episodes import EpisodeLog, PredictionWindow
from core.predictor import Predictor, PredictorStore
from core.logger import make_logger
from core.llm import LLMClient
from core.memory import Memory
from core.scheduler import Scheduler
from core.voice import VoiceRecognizer
import core.os_sandbox as _os

_state: dict = {}
_clients: set = set()
# One prediction window per connection: what the user is typing, and what we put
# in front of them. Keyed by socket so two HUDs do not credit each other's hints.
_windows: dict = {}
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
    bcfg = cfg.get("brain", {}) or {}
    brain = Brain(
        llm, mem, sys_prompt, schema, logger=logger,
        max_iters=int(bcfg.get("max_iters", 5)),
        budget_ms=int(bcfg.get("budget_ms", 20000)),
        history_turns=int(bcfg.get("history_turns", 6)),
    )

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
    sensor = ContextSensor(journal=get_journal())

    thresholds = load_thresholds(cfg.get("thresholds"))
    set_thresholds(thresholds)

    calibration_store = CalibrationStore(mem)
    calibrator = calibration_store.load()

    pcfg = cfg.get("prediction", {}) or {}
    predictor = Predictor(
        store=PredictorStore(mem),
        half_life_s=float(pcfg.get("half_life_s", 7 * 24 * 3600)),
        min_episodes=int(pcfg.get("min_episodes", 50)),
        calibrator=calibrator,
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
            return str(actions.dispatch(name, args, actor="anticipator", confidence=conf))[:4000]

        t = text.strip()
        if t == "tree" or t.startswith("tree "):
            return (text, {"reply": warm("list_tree", {"path": "."}), "confidence": conf,
                           "why": prediction.why, "action": text})
        if t == "ls":
            return (text, {"reply": warm("list_dir", {"path": "."}), "confidence": conf,
                           "why": prediction.why, "action": text})
        if t.startswith("read file "):
            path = t[len("read file "):].strip()
            return (text, {"reply": warm("read_file", {"path": path}), "confidence": conf,
                           "why": prediction.why, "action": text})
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
    if voice_cfg.get("enabled", True):
        try:
            voice = VoiceRecognizer(
                model_size=voice_cfg.get("model", "base"),
                language=voice_cfg.get("language", "en"),
            )
        except Exception:
            voice = None

    _state.update({"cfg": cfg, "brain": brain, "mem": mem, "sched": sched, "ant": ant,
                   "voice": voice, "episodes": episodes, "sensor": sensor,
                   "predictor": predictor, "calibrator": calibrator,
                   "calibration_store": calibration_store, "thresholds": thresholds})

    # Pre-load Whisper model in background so first voice use is instant
    if voice:
        def _preload():
            try:
                voice._load_model()
            except Exception:
                pass
        threading.Thread(target=_preload, daemon=True, name="whisper-preload").start()

    yield

    ant.stop()
    try:
        predictor.save()
    except Exception:
        logger("could not save predictor state")
    try:
        sched.stop()
    except Exception:
        pass


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_VOL_WORDS = {
    'zero': 0, 'muted': 0, 'ten': 10, 'twenty': 20, 'thirty': 30,
    'forty': 40, 'fifty': 50, 'half': 50, 'sixty': 60, 'seventy': 70,
    'eighty': 80, 'ninety': 90, 'hundred': 100, 'full': 100, 'max': 100,
    'maximum': 100, 'twenty five': 25, 'seventy five': 75,
}

# App names that should always route to os_open_app
_KNOWN_APP_NAMES = {
    'chrome', 'google chrome', 'google', 'firefox', 'edge', 'browser',
    'web browser', 'my browser', 'vscode', 'vs code', 'visual studio code',
    'code', 'notepad', 'calculator', 'calc', 'explorer', 'file explorer',
    'terminal', 'cmd', 'spotify', 'discord', 'slack', 'teams',
    'word', 'excel', 'powerpoint', 'paint', 'task manager', 'steam',
    'obs', 'vlc', 'settings', 'control panel',
}

_APP_NAME_ALIAS = {
    'google chrome': 'chrome', 'google': 'chrome',
    'browser': 'chrome', 'web browser': 'chrome', 'my browser': 'chrome',
    'vs code': 'vscode', 'visual studio code': 'vscode', 'code editor': 'vscode',
    'windows terminal': 'terminal', 'command prompt': 'terminal', 'cmd': 'terminal',
    'file manager': 'explorer', 'file explorer': 'explorer', 'files': 'explorer',
    'calc': 'calculator',
}


def _try_os_intent(text: str):
    """Match natural language OS commands. Returns (action, kwargs) or None."""
    t = text.lower().strip()

    # open / launch / start  (action-first: "open chrome") ────────────
    m = re.match(
        r'^(?:open|launch|start|run|execute)\s+(?:the\s+|my\s+|a\s+)?'
        r'(.+?)(?:\s+(?:app(?:lication)?|program|browser))?[.!?]?$', t
    )
    if m:
        name = m.group(1).strip().rstrip('.,!?')
        name = _APP_NAME_ALIAS.get(name, name)
        words = name.split()
        if name in _KNOWN_APP_NAMES or (1 <= len(words) <= 3 and not any(
            w in ('a', 'an', 'the', 'new', 'quick', 'file', 'search', 'question')
            for w in words
        )):
            return ('os_open_app', {'name': name})

    # open / launch  (name-first: "chrome open", "spotify launch") ─────
    m = re.match(r'^(.+?)\s+(?:open|launch|start|run)[.!?]?$', t)
    if m:
        name = m.group(1).strip().rstrip('.,!?')
        name = _APP_NAME_ALIAS.get(name, name)
        words = name.split()
        if name in _KNOWN_APP_NAMES or (1 <= len(words) <= 2):
            return ('os_open_app', {'name': name})

    # volume numeric — "set/put/change volume to/at/as X[%]" ──────────
    m = re.search(
        r'(?:set|put|change|turn|make|adjust)\s+(?:the\s+)?(?:volume|sound)\s+(?:to|at|as|=)\s*(\d+)',
        t
    ) or re.search(r'\bvolume\s+(?:to\s+|at\s+|=\s*)?(\d+)\b', t) \
      or re.search(r'\b(\d+)\s*%?\s*(?:volume|loudness)\b', t)
    if m:
        return ('os_set_volume', {'level': int(m.group(1))})

    # volume word numbers ("set volume to fifty") ───────────────────────
    for phrase, num in _VOL_WORDS.items():
        pesc = re.escape(phrase)
        if re.search(rf'(?:volume|sound)\s+(?:to\s+|at\s+|=\s*)?{pesc}\b', t) or \
           re.search(rf'(?:set|put|change|make)\s+(?:the\s+)?(?:volume|sound)\s+(?:to\s+|at\s+)?{pesc}\b', t) or \
           re.search(rf'\b{pesc}\s*(?:percent\s+)?(?:volume|loudness)\b', t):
            return ('os_set_volume', {'level': num})

    # volume relative ───────────────────────────────────────────────────
    if re.search(r'\bmute\b|\bno\s+sound\b|\bsilent(?:ce)?\b', t):
        return ('os_set_volume', {'level': 0})
    if re.search(r'(?:volume|sound)\s+(?:up|max|full|loud|higher|louder)', t) or \
       re.search(r'turn\s+(?:up|the\s+volume\s+up|volume\s+up)', t) or \
       re.search(r'turn\s+up\s+(?:the\s+)?(?:volume|sound)', t) or \
       re.search(r'(?:raise|increase)\s+(?:the\s+)?(?:volume|sound)', t):
        return ('os_set_volume', {'level': 90})
    if re.search(r'(?:volume|sound)\s+(?:down|low|quiet(?:er)?|half|lower)', t) or \
       re.search(r'turn\s+(?:down|the\s+volume\s+down|volume\s+down)', t) or \
       re.search(r'turn\s+(?:the\s+)?(?:volume|sound)\s+down', t) or \
       re.search(r'(?:lower|decrease|reduce)\s+(?:the\s+)?(?:volume|sound)', t):
        return ('os_set_volume', {'level': 20})

    # get/check current volume ──────────────────────────────────────────
    if re.search(r'(?:what|check|get|show)\s+(?:is\s+)?(?:the\s+)?(?:current\s+)?(?:volume|sound\s+level)', t):
        return ('os_get_volume', {})

    # brightness ────────────────────────────────────────────────────────
    m = re.search(
        r'(?:set|put|change|adjust)\s+(?:the\s+)?brightness\s+(?:to|at|=)\s*(\d+)', t
    ) or re.search(r'\bbrightness\s+(?:to\s+|at\s+)?(\d+)\b', t) \
      or re.search(r'\b(\d+)\s*%?\s*brightness\b', t)
    if m:
        return ('os_set_brightness', {'level': int(m.group(1))})
    if re.search(r'\bbrightness\s+(?:up|higher|brighter|max|full)\b', t) or \
       re.search(r'(?:increase|raise|turn\s+up)\s+(?:the\s+)?brightness', t):
        return ('os_set_brightness', {'level': 100})
    if re.search(r'\bbrightness\s+(?:down|lower|dim|half|low)\b', t) or \
       re.search(r'(?:decrease|lower|dim|reduce|turn\s+down)\s+(?:the\s+)?brightness', t):
        return ('os_set_brightness', {'level': 30})
    if re.search(r'(?:what|check|get|show)\s+(?:is\s+)?(?:the\s+)?(?:current\s+)?brightness', t):
        return ('os_get_brightness', {})

    # battery ───────────────────────────────────────────────────────────
    if re.search(
        r'\b(?:battery|charge|power\s+level|how\s+much\s+(?:battery|charge|power)|'
        r'battery\s+(?:life|level|status|percentage|percent)|'
        r'how\s+long\s+(?:until|till|before).*(?:battery|dies|dead))\b', t
    ):
        return ('os_get_battery', {})

    # network / wifi ────────────────────────────────────────────────────
    if re.search(
        r'\b(?:network\s+info(?:rmation)?|(?:what(?:\'s|\s+is)\s+(?:my|the)\s+)?(?:ip|wifi|wi-fi|ssid|'
        r'connection|internet)\s+(?:address|status|info|name)?|'
        r'am\s+i\s+connected|what\s+network|which\s+wifi|show\s+network)\b', t
    ):
        return ('os_get_network_info', {})
    if re.search(r'\b(?:enable|turn\s+on)\s+(?:the\s+)?(?:wifi|wi-fi|wireless)\b', t):
        return ('os_toggle_wifi', {'state': 'on'})
    if re.search(r'\b(?:disable|turn\s+off)\s+(?:the\s+)?(?:wifi|wi-fi|wireless)\b', t):
        return ('os_toggle_wifi', {'state': 'off'})

    # sleep / lock ──────────────────────────────────────────────────────
    if re.search(
        r'\b(?:sleep|hibernate|suspend)\s+(?:(?:the|my)\s+)?(?:computer|pc|machine|laptop)?\b', t
    ) and 'wake' not in t:
        return ('os_sleep_computer', {})
    if re.search(
        r'\b(?:lock\s+(?:(?:the|my)\s+)?(?:screen|computer|pc|machine|laptop)?|'
        r'(?:screen\s+)?lock)\b', t
    ):
        return ('os_lock_screen', {})

    # power / shutdown / restart ────────────────────────────────────────
    if re.search(
        r'\b(?:shut\s*down|shutdown|power\s*off|turn\s+off)\s+(?:(?:the|my|this|your)\s+)?(?:computer|pc|machine|system|laptop)\b',
        t
    ):
        return ('os_shutdown_computer', {'delay_sec': 30})
    if re.search(
        r'\b(?:restart|reboot)\s+(?:(?:the|my|this|your)\s+)?(?:computer|pc|machine|system|laptop)\b', t
    ):
        return ('os_restart_computer', {'delay_sec': 30})
    if re.search(r'\bcancel\s+(?:the\s+)?(?:shutdown|restart|reboot)\b', t):
        return ('os_cancel_shutdown', {})

    # screenshot ────────────────────────────────────────────────────────
    if re.search(r'screenshot|screen\s+cap(?:ture)?|snap\s+(?:the\s+)?screen', t):
        return ('os_take_screenshot', {})

    # system info / resource report ────────────────────────────────────
    if re.search(
        r'\b(?:'
        # direct terms
        r'system\s+info(?:rmation)?|sys(?:tem)?\s+status|pc\s+info(?:rmation)?|'
        r'hardware\s+info(?:rmation)?|computer\s+stats?|machine\s+info|'
        # resource variants
        r'(?:windows|system|pc|computer|my)\s+resources?|'
        r'resource\s+(?:report|usage|status|info|check)|'
        r'performance\s+(?:report|status|info|stats?)|'
        # RAM / memory
        r'ram(?:\s+(?:usage|status|info|report|check))?|'
        r'memory\s+(?:usage|status|info|report|check|left)|'
        r'how\s+much\s+(?:ram|memory|storage|disk\s+space)|'
        # CPU
        r'cpu(?:\s+(?:usage|load|status|info|report|check|temp(?:erature)?))?|'
        r'processor\s+(?:usage|load|status|info)|'
        # disk
        r'disk\s+(?:space|usage|status|info)|storage\s+(?:space|status|info)|'
        # catch-alls
        r'tell\s+me\s+about\s+(?:my\s+)?(?:system|resources?|computer|pc)|'
        r'(?:show|give\s+me|check)\s+(?:(?:my|the)\s+)?(?:system|resource|performance|pc|computer)\s+(?:report|status|info|stats?|usage)'
        r')\b', t
    ):
        return ('os_system_info', {})

    # processes ─────────────────────────────────────────────────────────
    if re.search(
        r'\b(?:(?:list|show|display|what(?:\'s|\s+is)?)\s+(?:running|'
        r'(?:all\s+)?processes?|(?:open\s+)?apps?|programs?)|'
        r'running\s+(?:processes?|apps?|programs?)|'
        r'what\s+(?:apps?|programs?)\s+(?:are\s+)?(?:open|running))\b', t
    ):
        return ('os_list_processes', {})

    # kill process ──────────────────────────────────────────────────────
    m = re.match(
        r'^(?:kill|close|stop|quit|end|terminate|force\s+close)\s+'
        r'(?:the\s+)?(?:process\s+)?(.+?)(?:\s+(?:process|app))?[.!?]?$', t
    )
    if m:
        name = m.group(1).strip().rstrip('.,!?')
        if name not in ('window', 'panel', 'hud', 'overlay', 'this', 'app', 'application', 'it'):
            return ('os_kill_process', {'name': name})

    # clipboard ─────────────────────────────────────────────────────────
    if re.search(
        r'(?:what(?:\'s|\s+is)\s+in\s+(?:my\s+)?clipboard|'
        r'read\s+clipboard|show\s+clipboard|get\s+clipboard|clipboard\s+content)', t
    ):
        return ('os_get_clipboard', {})

    return None


def _format_os_result(action: str, result: dict, kwargs: dict) -> str:
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
        return f"Brightness set to {result.get('brightness', kwargs.get('level', '?'))}%"
    if action == "os_get_brightness":
        return f"Current brightness: {result.get('brightness', '?')}%"
    if action == "os_get_battery":
        r = result
        charging = "charging" if r.get("charging") else "on battery"
        return f"Battery: {r.get('percent')} ({charging}), {r.get('time_remaining', '')}"
    if action == "os_get_network_info":
        r = result
        ifaces = ", ".join(f"{i['interface']} ({i['ip']})" for i in r.get("interfaces", []))
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


async def _dispatch(ws: WebSocket, name: str, kwargs: dict, actor: str = "user",
                    confidence: float = 1.0):
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
    if isinstance(res, dict) and res.get("needs_confirmation"):
        await ws.send_json({
            "type": "confirm_request",
            "token": res["token"],
            "capability": res["capability"],
            "args": res.get("args", {}),
            "reason": res.get("reason", ""),
            "reversibility": res.get("reversibility", ""),
            "summary": res.get("summary", ""),
        })
        return None
    return res


async def _resolve_confirmation(ws: WebSocket, token: str, granted: bool):
    loop = asyncio.get_running_loop()

    # A confirmation raised from inside the tool loop resumes that loop rather
    # than just running the action, so the model gets to see how it was answered
    # and finish its turn either way.
    resume_token = _state.get("brain_resume", {}).pop(token, None)
    if resume_token is not None:
        brain: Brain = _state["brain"]
        out = await loop.run_in_executor(
            _executor, lambda: brain.resume(resume_token, granted, on_token=_token_sink(ws, loop))
        )
        await _send_brain_result(ws, out)
        return

    res = await loop.run_in_executor(_executor, lambda: actions.confirm(token, granted=granted))
    mem: Memory = _state["mem"]
    if res.get("error"):
        await ws.send_json({"type": "error", "text": res["error"]})
        return
    if res.get("cancelled"):
        await ws.send_json({"type": "reply", "text": f"Cancelled {res['capability']}."})
        return
    pending = _state.pop("last_confirm_action", None)
    text = _format_os_result(pending, res, {}) if pending else _summarise(res)
    await ws.send_json({"type": "reply", "text": text})
    await ws.send_json({"type": "status", "safe_mode": is_safe_mode(),
                        "tasks_count": len(mem.list_tasks(status="pending"))})


def _token_sink(ws: WebSocket, loop):
    """Forward model tokens to the HUD as they arrive.

    With a tool loop a single turn can take several seconds per iteration, so
    without this the user watches a still panel and assumes it has hung.
    """
    def sink(piece: str):
        asyncio.run_coroutine_threadsafe(ws.send_json({"type": "token", "text": piece}), loop)
    return sink


async def _send_brain_result(ws: WebSocket, out: dict):
    """Deliver a Brain result: a reply, or a confirmation that suspended it."""
    if out.get("needs_confirmation"):
        _state.setdefault("brain_resume", {})[out["confirm_token"]] = out["resume_token"]
        await ws.send_json({
            "type": "confirm_request",
            "token": out["confirm_token"],
            "capability": out["capability"],
            "args": out.get("args", {}),
            "reason": out.get("reason", ""),
            "reversibility": out.get("reversibility", ""),
            "summary": "",
        })
        return
    await ws.send_json({"type": "reply", "text": out.get("reply", ""), "plan": out.get("plan", [])})


def _summarise(res: dict) -> str:
    if not isinstance(res, dict):
        return str(res)
    if res.get("error"):
        return f"⚠  {res['error']}"
    return json.dumps(res, indent=2, default=str)


_KNOWN_CMDS = [
    "/help", "/exit", "/memory", "/dream", "/save", "/recall", "/actions",
    "/config", "/reload", "/tasks", "/done", "/delete", "/snooze",
    "/hw", "/task_payload", "/safe", "/exec", "/write", "/read",
    "/undo", "/journal", "/capabilities", "/forget", "/episodes",
    "/calibration", "/thresholds",
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
        await ws.send_json({"type": "reply", "text":
                            "Consolidation is not implemented yet (Phase 6). "
                            "It will cluster the episode log and promote recurring "
                            "patterns into rules you can inspect with /rules."})
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
                                "safe_mode": is_safe_mode()})
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
            await ws.send_json({"type": "reply", "text": f"Task {parts[1]} snoozed {parts[2]}."})
        except Exception:
            await ws.send_json({"type": "error", "text": "usage: /snooze <id> <15m|2h|1d>"})
        return

    if text.startswith("/safe"):
        parts = text.split()
        if len(parts) >= 2:
            res = set_safe_mode_action(state=parts[1])
            safe_on = is_safe_mode()
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
        res = await _dispatch(ws, "run_local", {"cmd": cmd, "cwd": cwd})
        if res is None:
            return  # parked awaiting confirmation
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
        lines = [f"  {k:<14} {'never' if v is None else format(float(v), '.2f')}"
                 for k, v in th.items()]
        await ws.send_json({"type": "reply", "text":
                            "Cost-gated thresholds" + chr(10) + "\n".join(lines)})
        return

    if text == "/forget":
        episodes = _state.get("episodes")
        if not episodes:
            await ws.send_json({"type": "error", "text": "episode log unavailable"})
            return
        n = await loop.run_in_executor(_executor, episodes.forget)
        await ws.send_json({"type": "reply", "text": f"Forgot {n} episode(s)."})
        return

    if text == "/episodes":
        episodes = _state.get("episodes")
        if not episodes:
            await ws.send_json({"type": "error", "text": "episode log unavailable"})
            return
        rows = episodes.recent(limit=15)
        if not rows:
            await ws.send_json({"type": "reply", "text":
                                "No episodes recorded yet." if episodes.enabled
                                else "Episode logging is disabled in config.yaml."})
            return
        lines = []
        for e in rows:
            when = datetime.datetime.fromtimestamp(e.ts).strftime("%H:%M:%S")
            hint = ""
            if e.accepted_prediction is not None:
                hint = "  hint taken" if e.accepted_prediction else f"  hint ignored ({e.predicted})"
            lines.append(f"{when}  {e.action}{hint}")
        await ws.send_json({"type": "reply", "text": "\n".join(lines)})
        return

    if text == "/undo":
        res = await loop.run_in_executor(_executor, undo_last)
        if res.get("error"):
            await ws.send_json({"type": "error", "text": res["error"]})
        else:
            await ws.send_json({"type": "reply",
                                "text": f"Undid {res['capability']} (journal #{res['id']})."})
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
            lines.append(f"#{r['id']} {when} {r['actor']}/{r['capability']} "
                         f"{r['decision']}{'/' + r['outcome'] if r['outcome'] else ''}{mark}")
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

    episode_id = episodes.record(text, ctx, **signals) if episodes else None

    try:
        await _handle_input_inner(ws, text, note)
    except Exception:
        if episodes and episode_id:
            episodes.set_outcome(episode_id, "error")
        raise

    if sensor:
        sensor.note_submission(text, exit_code=note.get("exit_code"))
    if episodes and episode_id:
        if note.get("capability"):
            episodes.set_capability(episode_id, note["capability"])
        episodes.set_outcome(episode_id, note.get("outcome", "ok"))

    # Online update: this submission is the ground truth for whatever was
    # predicted a moment ago, including the times the prediction was wrong.
    predictor = _state.get("predictor")
    if predictor is not None:
        from core.episodes import Episode as _Episode
        predictor.update(_Episode(
            ts=time.time(), action=text, context=ctx,
            keystroke_prefix=signals.get("keystroke_prefix") or text,
        ))

    # A write may have invalidated something prewarmed against the old state.
    if note.get("capability") in ("write_file", "/write", "/exec", "run_local"):
        ant = _state.get("ant")
        if ant:
            ant.invalidate()


async def _handle_input_inner(ws: WebSocket, text: str, note: dict):
    brain: Brain = _state["brain"]
    mem: Memory = _state["mem"]
    ant: Anticipator = _state["ant"]
    loop = asyncio.get_running_loop()

    if text.startswith("/"):
        tokens = text.split()
        fixed = _fuzzy_cmd(tokens[0])
        if fixed != tokens[0]:
            text = " ".join([fixed] + tokens[1:]).strip()
        note["capability"] = text.split()[0]
        await _handle_command(ws, text)
        return

    m = re.match(r"^remind\s+me\s+(.+?)\s+((?:in|at)\s+\S.*)$", text, re.IGNORECASE)
    if m:
        note["capability"] = "create_task"
        res = actions.call("create_task", text=m.group(1).strip(), when=m.group(2).strip())
        if res.get("ok"):
            due_str = datetime.datetime.fromtimestamp(res["due_ts"]).strftime("%I:%M %p, %b %d")
            reply = f"Reminder set: \"{m.group(1).strip()}\" at {due_str}"
        else:
            reply = f"⚠  {res.get('error', 'Could not parse time')}"
        await ws.send_json({"type": "reply", "text": reply})
        return

    if text == "ls":
        note["capability"] = "list_dir"
        res = await loop.run_in_executor(_executor, lambda: actions.call("list_dir", path="."))
        if isinstance(res, list):
            lines = "\n".join(
                f"{'📁' if r['type']=='dir' else '📄'} {r['name']}" for r in res
            )
        else:
            lines = str(res)
        await ws.send_json({"type": "reply", "text": lines})
        return

    if text == "tree":
        note["capability"] = "list_tree"
        res = await loop.run_in_executor(_executor, lambda: actions.call("list_tree", path="."))
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
        _state["last_confirm_action"] = action_name
        try:
            result = await _dispatch(ws, action_name, kwargs, actor="user")
        except Exception as e:
            await ws.send_json({"type": "error", "text": str(e)})
            return
        if result is None:
            return  # parked awaiting confirmation
        _state.pop("last_confirm_action", None)
        reply = _format_os_result(action_name, result, kwargs)
        mem.add("user", text)
        mem.add("assistant", reply)
        await ws.send_json({"type": "reply", "text": reply})
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
        sink = _token_sink(ws, loop)
        out = await loop.run_in_executor(_executor, lambda: brain.step(text, on_token=sink))
        await _send_brain_result(ws, out)
    except Exception as e:
        await ws.send_json({"type": "error", "text": f"LLM error: {e}"})


async def _maybe_send_anticipation(ws: WebSocket, text: str):
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
            await ws.send_json({"type": "anticipation", "data": pre, "text": text,
                                "reveal_threshold": ant.reveal_threshold})
        except Exception:
            return
        # Recorded only once the send succeeded: a hint that never reached the
        # user must not be counted against them for ignoring it.
        window = _windows.get(ws)
        if window:
            window.note_shown(text, pre.get("confidence"))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.add(ws)
    _windows[ws] = PredictionWindow()

    mem: Memory = _state["mem"]
    await ws.send_json({
        "type": "status",
        "safe_mode": is_safe_mode(),
        "tasks_count": len(mem.list_tasks(status="pending")),
        "version": "1.0",
    })

    try:
        while True:
            data = await ws.receive_json()
            t = data.get("type")

            if t == "buffer":
                buf = data.get("text", "")
                _state["ant"].update_buffer(buf)
                window = _windows.get(ws)
                if window:
                    window.note_keystroke(buf)
                asyncio.create_task(_maybe_send_anticipation(ws, buf))

            elif t == "input":
                text = data.get("text", "").strip()
                if text:
                    await _handle_input(ws, text)

            elif t == "confirm":
                await _resolve_confirmation(ws, data.get("token", ""), bool(data.get("granted")))

            elif t == "get_status":
                await ws.send_json({
                    "type": "status",
                    "safe_mode": is_safe_mode(),
                    "tasks_count": len(mem.list_tasks(status="pending")),
                })

            elif t == "voice_start":
                voice = _state.get("voice")
                if not voice:
                    await ws.send_json({"type": "error", "text": "Voice unavailable — install faster-whisper and sounddevice"})
                elif voice.is_recording():
                    pass
                else:
                    _loop = asyncio.get_running_loop()

                    def _on_silence():
                        # mic closed, transcription starting
                        asyncio.run_coroutine_threadsafe(
                            ws.send_json({"type": "voice_recording", "active": False}),
                            _loop,
                        )

                    def _on_complete(recognized_text: str):
                        async def _finish():
                            if recognized_text:
                                await ws.send_json({"type": "voice_text", "text": recognized_text})
                                await _handle_input(ws, recognized_text)
                            else:
                                await ws.send_json({"type": "error", "text": "No speech detected"})
                        asyncio.run_coroutine_threadsafe(_finish(), _loop)

                    try:
                        voice.start_recording_vad(on_silence=_on_silence, on_complete=_on_complete)
                        await ws.send_json({"type": "voice_recording", "active": True})
                    except Exception as e:
                        await ws.send_json({"type": "error", "text": f"Microphone error: {e}"})

            elif t == "voice_stop":
                voice = _state.get("voice")
                if voice and voice.is_recording():
                    voice.stop_now()  # signals VAD to stop; callbacks still fire

    except WebSocketDisconnect:
        _clients.discard(ws)
        _windows.pop(ws, None)
    except Exception:
        _clients.discard(ws)
        _windows.pop(ws, None)


@app.get("/health")
async def health():
    return {"ok": True, "version": "1.0"}
