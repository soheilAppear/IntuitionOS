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
from contextlib import asynccontextmanager, suppress

import yaml
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
from core.capabilities import capabilities, is_safe_mode, set_safe_mode
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
from core.gestures import GestureRecognizer
from core.hand_control import HandControls
from core.hand_feedback import HandClickSound
from core.multimodal import MultimodalPreview
from core.voice import VoiceRecognizer
from plugins.brainbit import BrainBit

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


async def _broadcast_status():
    """Safe Mode is shared by every HUD connected to this process."""
    await _broadcast({
        "type": "status",
        "safe_mode": is_safe_mode(),
        "tasks_count": len(_state["mem"].list_open()),
        "gestures": _gesture_info(),
        "brainbit": _brainbit_info(),
    })


def _brainbit_info():
    driver = _state.get("brainbit")
    if driver is None:
        return {"state": "disabled", "available": False, "busy": False,
                "devices": [], "device": None, "text": "BrainBit is disabled."}
    return driver.status()


def _multimodal_info():
    preview = _state.get("multimodal")
    return preview.status() if preview else {
        "state": "unavailable", "running": False, "armed": False,
        "reason": "Restart the backend to enable the combined preview."}


def _multimodal_busy():
    info = _multimodal_info()
    camera = _state.get("preview_camera")
    return (info.get("state") in ("starting", "running", "contact", "stopping")
            or bool(info.get("cleanup_pending"))
            or (camera is not None and (camera.is_running()
                or camera.status().get("state") in ("starting", "stopping")
                or camera.status().get("tracker_cleanup_pending"))))


async def _poll_brainbit(driver):
    """Refresh connected-device metadata off the event loop; never reconnect."""
    previous = None
    next_refresh = 0.0
    refresh_task = None

    async def refresh_metadata():
        nonlocal next_refresh
        try:
            await asyncio.to_thread(driver.call, "status", refresh=True)
        finally:
            next_refresh = time.monotonic() + 3.0

    try:
        while True:
            if refresh_task is not None and refresh_task.done():
                # Retrieve failures without losing cached signal broadcasts;
                # the driver exposes expected SDK failures in its status.
                with suppress(Exception):
                    refresh_task.result()
                refresh_task = None
            info = driver.status()
            if (refresh_task is None and info.get("state") == "connected"
                    and not info.get("busy") and time.monotonic() >= next_refresh):
                refresh_task = asyncio.create_task(refresh_metadata())
            active_signal = (info.get("signal") or {}).get("state") in (
                "starting", "running", "stale", "contact", "stopping")
            # The collapsed HUD still displays signal freshness. Renew its
            # cached evidence before expiry, even if summary values repeat.
            if active_signal or info != previous:
                await _broadcast({"type": "brainbit_status", **info})
                previous = info
            await asyncio.sleep(0.25 if active_signal else 1)
    finally:
        if refresh_task is not None:
            refresh_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await refresh_task


def _safe_mode_sink(loop):
    """A worker acknowledges a granted mode change before the action runs."""
    def changed():
        asyncio.run_coroutine_threadsafe(_broadcast_status(), loop).result()
    return changed


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


def _queue_gesture(loop, event) -> None:
    """Act in capture order; queue only feedback, never a delayed window target."""
    if loop.is_closed():
        return
    message = _handle_gesture(event)
    if message:
        _queue_hand_feedback(loop, message)


def _queue_gesture_status(loop, status) -> None:
    if loop.is_closed():
        return
    future = asyncio.run_coroutine_threadsafe(
        _broadcast_gesture_status(), loop
    )
    future.add_done_callback(
        lambda done: done.exception() if not done.cancelled() else None
    )


def _queue_hand_feedback(loop, message) -> None:
    if loop.is_closed():
        return
    future = asyncio.run_coroutine_threadsafe(_deliver_hand_feedback(message), loop)
    future.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)


async def _deliver_hand_feedback(message):
    gestures = _state.get("gestures")
    if message.get("type") in ("gesture_progress", "gesture_click") and (
            not gestures or not gestures.is_running()):
        return
    if message.get("type") == "gesture_click" and (
            getattr(gestures, "input_mode", "desktop") != "mouse"
            or not 0 <= time.monotonic() - message.get("at", float("-inf")) <= 1.0):
        return
    await _broadcast(message)


def _queue_mouse_click(loop, event):
    """A completed manual click gets one local tick and transient HUD feedback."""
    gestures = _state.get("gestures")
    if (not gestures or not gestures.is_running()
            or getattr(gestures, "input_mode", "desktop") != "mouse"
            or event.get("source") not in ("bend", "pinch")
            or not isinstance(event.get("id"), str)
            or not 0 <= time.monotonic() - event.get("at", float("-inf")) <= 1.0):
        return
    sound = _state.get("hand_click_sound")
    if sound:
        sound.play(event["id"])
    _queue_hand_feedback(loop, {**event, "type": "gesture_click"})


def _gesture_info() -> dict:
    gestures = _state.get("gestures")
    if gestures is None:
        return {"state": "unavailable", "available": False, "running": False,
                "text": "Gesture recognition is unavailable.", "bindings": {}}
    info = gestures.status()
    controls = _state.get("hand_controls")
    if controls:
        info.update(controls.status())
    info["settings"] = dict(_state.get("gesture_settings", {}))
    if _state.get("hand_click_sound"):
        info["click_sound"] = _state["hand_click_sound"].status()
    return info


async def _broadcast_gesture_status() -> dict:
    # Read when the callback is delivered: a queued startup snapshot must not
    # overwrite a later stop/failure or make another HUD show the camera on.
    info = _gesture_info()
    await _broadcast({"type": "gesture_status", **info})
    return info


async def _change_gestures(enabled) -> dict:
    """Serialize camera changes across HTTP, sockets, and slash commands."""
    if type(enabled) is not bool:
        return {"error": "Gestures require enabled to be a boolean.",
                "status_code": 400}

    # Capture startup and shutdown can block on a camera driver. Serialize them
    # across HUDs while keeping the event loop free to send lifecycle events.
    async with _state["gesture_lock"]:
        if enabled and _multimodal_busy():
            return {"error": "Stop the combined EEG + camera preview before turning on Hand controls.",
                    "status_code": 409}
        gestures = _state.get("gestures")
        if gestures is None:
            outcome = {"error": "Gesture recognition is unavailable."}
        else:
            try:
                controls = _state.get("hand_controls")
                if enabled and controls and controls.status().get("desktop", {}).get("cleanup_pending"):
                    return {"error": "Navigation input is still releasing. Turn the camera off again to retry.",
                            "status_code": 503}
                # Camera control must stay available even if all model/action
                # workers are busy with long requests from connected HUDs.
                outcome = await asyncio.to_thread(
                    gestures.start if enabled else gestures.stop)
                if not enabled and controls:
                    cleanup = await asyncio.to_thread(controls.stop)
                    if isinstance(cleanup, dict) and cleanup.get("error"):
                        errors = [outcome.get("error"), cleanup["error"]]
                        outcome = {**outcome, "error": "; ".join(error for error in errors if error)}
                if not enabled and _state.get("hand_click_sound"):
                    _state["hand_click_sound"].cancel_pending()
            except Exception as error:
                outcome = {"error": f"Could not change gesture recognition: {error}"}
        return {**outcome, "status_code": 503 if outcome.get("error") else 200}


async def _set_gestures(ws: WebSocket, enabled, *, reply=False) -> None:
    outcome = await _change_gestures(enabled)
    if outcome.get("error"):
        await ws.send_json({"type": "error", "source": "gestures",
                            "text": outcome["error"]})
    info = await _broadcast_gesture_status()
    if reply and not outcome.get("error"):
        await ws.send_json({"type": "reply", "text": info["text"]})


def _handle_gesture(event):
    """Short gated actions run on the capture thread, independently of the model."""
    gestures = _state.get("gestures")
    if not gestures or not gestures.is_running():
        return
    if getattr(gestures, "input_mode", "desktop") != "desktop":
        return
    controls = _state.get("hand_controls")
    if controls is None:
        return
    try:
        outcome = controls.handle(event, gestures.action_for(event.name))
        if outcome is None:
            return
        capability, args, result = outcome
    except Exception as error:
        capability, args = "hand_control", {}
        result = {"error": str(error)}

    return {
        "type": "gesture",
        "gesture": event.name,
        "capability": capability,
        "text": _format_os_result(capability, result, args)
        if isinstance(result, dict) else str(result),
        "ok": bool(isinstance(result, dict) and not result.get("error")),
    }


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
        offer_safe_mode_confirmation=True,
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

    # Construction only probes package availability. Discovery and connection
    # remain explicit, and each native session belongs to an isolated worker.
    brainbit_cfg = next((d for d in cfg.get("hardware", {}).get("drivers", [])
                        if d.get("name") == "brainbit"), {})
    brainbit = BrainBit(enabled=brainbit_cfg.get("enabled", False))
    from core.actions import register_driver
    register_driver(brainbit)

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
                    "rule_id": prediction.rule_id,
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
                    "rule_id": prediction.rule_id,
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
                    "rule_id": prediction.rule_id,
                },
            )
        # Anything else is still worth predicting even though there is nothing
        # cheap to precompute: the hint alone has value.
        return (text, {"confidence": conf, "why": prediction.why, "action": text,
                       "rule_id": prediction.rule_id})

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

    # ── Gestures ─────────────────────────────────────────────────────────
    # The camera is opened only once gestures are switched on, so a machine with
    # no webcam — or a user who does not want one watching — costs nothing.
    gesture_cfg = cfg.get("gestures", {}) or {}
    gesture_settings = {
        "input_mode": gesture_cfg.get("input_mode", "desktop"),
        "bend_click": gesture_cfg.get("bend_click", False),
        "desktop_mode": gesture_cfg.get("desktop_mode", "auto"),
        "travel_palms": float(gesture_cfg.get("travel_palms", 1.2)),
        "model_complexity": gesture_cfg.get("model_complexity", 1),
        "tracker_backend": gesture_cfg.get("tracker_backend", "mediapipe"),
    }
    hand_controls = HandControls(
        active=lambda: gestures.is_running() and (
            gestures.input_mode == "desktop" or gestures.navigation_active),
        feedback=lambda message: _queue_hand_feedback(loop, message),
        mode=gesture_settings["desktop_mode"],
    )
    click_sound = HandClickSound(
        enabled=gesture_cfg.get("click_sound", True),
        on_status=lambda status: _queue_hand_feedback(loop, {"type": "gesture_sound", **status}),
    )
    gestures = GestureRecognizer(
        camera_index=int(gesture_cfg.get("camera_index", 0)),
        hold_frames=int(gesture_cfg.get("hold_frames", 4)),
        cooldown_s=float(gesture_cfg.get("cooldown_s", 0.8)),
        on_gesture=lambda event: _queue_gesture(loop, event),
        on_status=lambda status: _queue_gesture_status(loop, status),
        on_motion=hand_controls.motion,
        on_progress=lambda progress: _queue_hand_feedback(
            loop, {**progress, "type": "gesture_progress"}),
        travel_palms=gesture_settings["travel_palms"],
        model_complexity=gesture_settings["model_complexity"],
        tracker_backend=gesture_settings["tracker_backend"],
        input_mode=gesture_settings["input_mode"],
        bend_click=gesture_settings["bend_click"],
        on_click=lambda event: _queue_mouse_click(loop, event),
    )

    # Independent preview ownership: this camera has no input callbacks and
    # empty gesture bindings. Only the explicitly armed service can dispatch.
    preview_camera = GestureRecognizer(
        camera_index=int(gesture_cfg.get("camera_index", 0)), bindings={},
        input_mode="desktop", model_complexity=gesture_settings["model_complexity"],
        tracker_backend=gesture_settings["tracker_backend"],
    )
    multimodal = MultimodalPreview(
        preview_camera, brainbit,
        gesture_busy=lambda: (gestures.is_running()
            or gestures.status().get("state") in ("starting", "stopping")
            or gestures.status().get("tracker_cleanup_pending", False)
            or hand_controls.status().get("desktop", {}).get("cleanup_pending", False)),
        dispatch=actions.dispatch,
    )

    _state.update(
        {
            "cfg": cfg,
            "brain": brain,
            "mem": mem,
            "sched": sched,
            "ant": ant,
            "gestures": gestures,
            "hand_controls": hand_controls,
            "hand_click_sound": click_sound,
            "gesture_settings": gesture_settings,
            "gesture_lock": asyncio.Lock(),
            "voice": voice,
            "voice_status": voice_status,
            "voice_owner": None,
            "brainbit": brainbit,
            "multimodal": multimodal,
            "preview_camera": preview_camera,
            "episodes": episodes,
            "sensor": sensor,
            "predictor": predictor,
            "calibrator": calibrator,
            "calibration_store": calibration_store,
            "thresholds": thresholds,
            "rules": rule_store,
            "retriever": retriever,
            "resolver": resolver,
            "logger": logger,
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

    brainbit_task = asyncio.create_task(_poll_brainbit(brainbit))
    try:
        yield
    finally:
        brainbit_task.cancel()
        await asyncio.to_thread(multimodal.stop)
        # close() also cancels a blocked connect/refresh; do not await an
        # abandoned native operation before terminating its owned worker.
        await asyncio.to_thread(brainbit.close)
        with suppress(asyncio.CancelledError):
            await brainbit_task

    # Read replaceable services from _state: /forget may have replaced the
    # initial predictor/anticipator while this lifespan was active.
    _state["ant"].stop()
    if _state.get("voice"):
        _state["voice"].stop_now()
    if _state.get("gestures"):
        if _state.get("hand_click_sound"):
            _state["hand_click_sound"].close()
        # Release the camera. A held device stays unavailable to every other
        # application until the process actually exits.
        _state["gestures"].stop()
    if _state.get("hand_controls"):
        _state["hand_controls"].stop()
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
    if action == "create_empty_file":
        return f"Created empty file: {result['path']}"
    if action == "os_open_url":
        browser = result.get("browser", kwargs.get("browser", "default"))
        browser_name = {
            "default": "your default browser", "chrome": "Chrome",
            "edge": "Edge", "firefox": "Firefox",
        }.get(browser, browser)
        return f"Opening {result.get('url', kwargs.get('url', '?'))} in {browser_name}."
    if action == "os_open_app":
        # Show the friendly name the user requested, not the full exe path
        return f"Opened {kwargs.get('name', result.get('launched', '?')).title()}"
    if action == "os_close_window":
        return f'Close requested for {result.get("title", "the window")}. The application may ask you to save.'
    if action == "os_window_state":
        verb = {"maximize": "Maximized", "minimize": "Minimized", "restore": "Restored"}.get(result.get("state"), "Changed")
        return f'{verb} {result.get("title", "the active window")}. '
    if action == "os_cycle_window":
        return f'Switched to {result.get("title", "the next window")}. '
    if action == "os_switch_desktop":
        return f'Desktop switch requested: {result.get("direction", "next")}. '
    if action == "os_desktop_view":
        return ("Task View toggle requested." if result.get("view") == "overview"
                else "Show/hide desktop requested.")
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
        lambda: actions.dispatch(
            name, kwargs, actor=actor, confidence=confidence,
            offer_safe_mode_confirmation=True,
        ),
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
                "requires_safe_mode_off": res.get("requires_safe_mode_off", False),
                "client_revision": binding["client_revision"],
            }
        )
        return None
    return res


async def _resolve_confirmation(
    ws: WebSocket, token: str, granted: bool, allow_safe_mode_change: bool = True,
):
    """Resolve an owned, current gate token and finish its action or model turn."""
    if not isinstance(token, str) or type(granted) is not bool:
        await ws.send_json({
            "type": "error",
            "text": "Confirmation requires a token and a boolean decision.",
        })
        return
    if type(allow_safe_mode_change) is not bool:
        await ws.send_json({
            "type": "error",
            "text": "allow_safe_mode_change requires a boolean.",
        })
        return
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
            lambda: brain.resume(
                resume_token, granted, on_token=_token_sink(ws, loop),
                on_safe_mode_change=_safe_mode_sink(loop),
                allow_safe_mode_change=allow_safe_mode_change,
            ),
        )
        _finish_feedback(
            meta, "ok" if granted and not out.get("error") else "cancelled"
        )
        await _send_brain_result(ws, out)
        return

    res = await loop.run_in_executor(
        _executor, lambda: actions.confirm(
            token, granted=granted, on_safe_mode_change=_safe_mode_sink(loop),
            allow_safe_mode_change=allow_safe_mode_change,
        )
    )
    _finish_feedback(meta, _outcome(res))
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
    await _broadcast_status()


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
                "requires_safe_mode_off": out.get("requires_safe_mode_off", False),
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
                logger=_state.get("logger"),
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
        if len(parts) == 2:
            res = set_safe_mode_action(state=parts[1])
            if res.get("error"):
                await ws.send_json({"type": "error", "text": res["error"]})
            else:
                await _broadcast_status()
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

    if text.startswith("/gestures"):
        parts = text.split()
        argument = parts[1].lower() if len(parts) > 1 else "status"

        if argument in ("on", "off"):
            await _set_gestures(ws, argument == "on", reply=True)
        elif argument == "status":
            status = await _broadcast_gesture_status()
            lines = [
                f"available: {status['available']}",
                f"running:   {status['running']}",
                f"state:     {status['state']}",
                status["text"],
                "",
                "Bindings:",
            ]
            lines += [f"  {name:<14} {capability}"
                      for name, capability in sorted(status["bindings"].items())]
            await ws.send_json({"type": "reply", "text": "\n".join(lines)})
        else:
            await ws.send_json({"type": "error", "source": "gestures",
                                "text": "usage: /gestures on|off|status"})
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


def _record_rule_outcome(window, text: str) -> None:
    """Feed a shown rule's outcome back to the rule store.

    Kept tolerant: learning is never worth failing a submission over, and the
    episode log may be disabled entirely.
    """
    if not window:
        return
    outcome = window.take_rule_outcome()
    if not outcome:
        return
    rules = _state.get("rules")
    episodes = _state.get("episodes")
    if not rules or (episodes is not None and not episodes.enabled):
        return
    if text.strip() == "/forget":
        return
    try:
        rules.record_outcome(outcome[0], outcome[1])
    except Exception:
        pass


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

    # This submission is the verdict on any rule-sourced hint that was shown, and
    # the only thing that ever moves a rule's hit rate off its mined confidence.
    _record_rule_outcome(window, text)
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
        "create_empty_file",
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
        await ws.send_json({
            "type": "thinking",
            "text": {
                "os_open_url": "Opening browser…",
                "create_empty_file": "Creating file…",
            }.get(action_name, "Working…"),
        })
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
    pre = ant.try_hint(text)
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
            window.note_shown(text, pre.get("confidence"), rule_id=pre.get("rule_id"))


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
            "gestures": _gesture_info(),
            "brainbit": _brainbit_info(),
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
                    ws, data.get("token", ""), data.get("granted"),
                    data.get("allow_safe_mode_change", True),
                )

            elif t == "set_safe_mode":
                enabled = data.get("enabled")
                if type(enabled) is not bool:
                    await ws.send_json({
                        "type": "error", "source": "safe_mode",
                        "text": "Safe Mode requires enabled to be a boolean.",
                    })
                    continue
                # This changes only the mode. It neither consumes an action's
                # pending approval nor edits the socket's current draft.
                set_safe_mode(enabled)
                await _broadcast_status()

            elif t == "set_gestures":
                await _set_gestures(ws, data.get("enabled"))

            elif t == "get_status":
                await ws.send_json(
                    {
                        "type": "status",
                        "safe_mode": is_safe_mode(),
                        "tasks_count": len(mem.list_open()),
                        "voice": _voice_info(),
                        "gestures": _gesture_info(),
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


@app.post("/gestures")
async def set_gestures(request: Request):
    """Control the camera even while the HUD socket awaits an AI reply."""
    try:
        payload = await request.json()
    except (ValueError, UnicodeError):
        payload = None
    if not isinstance(payload, dict):
        return JSONResponse(
            {"gestures": _gesture_info(),
             "error": "Expected a JSON object with an enabled boolean."},
            status_code=400,
        )
    outcome = await _change_gestures(payload.get("enabled"))
    result = {"gestures": await _broadcast_gesture_status()}
    if outcome.get("error"):
        result["error"] = outcome["error"]
    return JSONResponse(result, status_code=outcome["status_code"])


@app.get("/gestures/preview")
async def gesture_preview(request: Request):
    """Read the existing capture thread's snapshot; never open another camera."""
    # The HUD uses native Node HTTP. Browser scripts cannot omit Origin on a
    # cross-origin request with this custom header; the Host check also rejects
    # same-origin DNS rebinding. Apply this before reading or leasing a frame.
    if ("origin" in request.headers
            or request.headers.get("x-intuition-preview") != "1"
            or request.headers.get("host") not in ("127.0.0.1:7432", "localhost:7432")):
        return JSONResponse({"error": "Camera preview is only available through the local HUD."},
                            status_code=403, headers={"Cache-Control": "no-store"})
    gestures = _state.get("gestures")
    if gestures is None:
        return JSONResponse({"running": False, "image": None, "tracked": False,
                             "landmarks": [], "error": "Gesture recognition is unavailable."},
                            status_code=503, headers={"Cache-Control": "no-store"})
    return JSONResponse(gestures.preview(), headers={"Cache-Control": "no-store"})


@app.post("/gestures/sound")
async def gesture_sound(request: Request):
    """Mute click feedback without interrupting the user's pointer session."""
    try:
        payload = await request.json()
    except (ValueError, UnicodeError):
        payload = None
    if not isinstance(payload, dict) or set(payload) != {"enabled"} or type(payload["enabled"]) is not bool:
        return JSONResponse({"error": "Click sound requires an enabled boolean."}, status_code=400)
    sound = _state.get("hand_click_sound")
    if sound is None:
        return JSONResponse({"error": "Restart the backend to enable click sound."}, status_code=503)
    status = sound.set_enabled(payload["enabled"])
    await _broadcast({"type": "gesture_sound", **status})
    return {"click_sound": status}


@app.post("/gestures/settings")
async def gesture_settings(request: Request):
    """Change hand controls and model only while the camera is off."""
    try:
        payload = await request.json()
    except (ValueError, UnicodeError):
        payload = None
    if (not isinstance(payload, dict)
            or not {"desktop_mode", "travel_palms"} <= set(payload)
            or not set(payload) <= {"desktop_mode", "travel_palms", "model_complexity", "tracker_backend", "input_mode", "bend_click"}
            or payload.get("desktop_mode") not in ("auto", "shortcut")
            or type(payload.get("travel_palms")) not in (int, float)
            or not 0.8 <= payload["travel_palms"] <= 3.0
            or ("model_complexity" in payload and (
                type(payload["model_complexity"]) is not int
                or payload["model_complexity"] not in (0, 1)))
            or ("tracker_backend" in payload and (
                type(payload["tracker_backend"]) is not str
                or payload["tracker_backend"] not in ("mediapipe", "rtmpose", "wilor")))
            or ("input_mode" in payload and payload["input_mode"] not in ("desktop", "mouse"))
            or ("bend_click" in payload and type(payload["bend_click"]) is not bool)):
        return JSONResponse({"error": "Choose desktop or mouse controls, auto or shortcut desktop movement, hand travel from 0.8 to 3.0 palms, tracker mediapipe, rtmpose or wilor, MediaPipe model 0 (light) or 1 (full), and bend_click true or false."}, status_code=400)
    async with _state["gesture_lock"]:
        if _multimodal_busy():
            return JSONResponse({"error": "Stop the combined preview before changing hand controls."}, status_code=409)
        gestures = _state["gestures"]
        if gestures.status()["state"] not in ("off", "error", "unavailable"):
            return JSONResponse({"error": "Stop the camera before changing hand controls."}, status_code=409)
        if _state["hand_controls"].status().get("desktop", {}).get("cleanup_pending"):
            return JSONResponse({"error": "Finish releasing navigation input before changing hand controls."}, status_code=409)
        recognizer_settings = {"travel_palms": float(payload["travel_palms"])}
        if "model_complexity" in payload:
            recognizer_settings["model_complexity"] = payload["model_complexity"]
        if "tracker_backend" in payload:
            recognizer_settings["tracker_backend"] = payload["tracker_backend"]
        if "bend_click" in payload:
            recognizer_settings["bend_click"] = payload["bend_click"]
        if "input_mode" in payload:
            recognizer_settings["input_mode"] = payload["input_mode"]
            cleanup = _state["hand_controls"].stop()
            if isinstance(cleanup, dict) and cleanup.get("error"):
                return JSONResponse(cleanup, status_code=409)
        result = gestures.configure(**recognizer_settings)
        if result.get("error"):
            return JSONResponse(result, status_code=409)
        result = _state["hand_controls"].desktop.configure(mode=payload["desktop_mode"])
        if result.get("error"):
            return JSONResponse(result, status_code=409)
        _state["gesture_settings"] = {**_state.get("gesture_settings", {}), **payload}
    return {"gestures": await _broadcast_gesture_status()}


def _brainbit_local_request(request):
    # Native HUD HTTP omits Origin. A web page cannot forge that omission plus
    # this header, including through the existing permissive CORS middleware.
    return ("origin" not in request.headers
            and request.headers.get("x-intuition-brainbit") == "1"
            and request.headers.get("host", "").split(":", 1)[0]
            in ("127.0.0.1", "localhost"))


@app.get("/brainbit/status")
async def brainbit_status(request: Request):
    if not _brainbit_local_request(request):
        return JSONResponse({"error": "BrainBit status is only available through the local HUD."},
                            status_code=403, headers={"Cache-Control": "no-store"})
    return JSONResponse(_brainbit_info(), headers={"Cache-Control": "no-store"})


@app.post("/brainbit/{operation}")
async def brainbit_operation(operation: str, request: Request):
    """Keep manual connection controls responsive during model or SDK work."""
    headers = {"Cache-Control": "no-store"}
    if not _brainbit_local_request(request):
        return JSONResponse({"error": "BrainBit controls are only available through the local HUD."},
                            status_code=403, headers=headers)
    if operation not in ("discover", "connect", "disconnect", "refresh"):
        return JSONResponse({"error": "Unknown BrainBit operation."}, status_code=404, headers=headers)
    try:
        payload = await request.json()
    except (ValueError, UnicodeError):
        payload = None
    expected = {"device_id"} if operation == "connect" else set()
    if (not isinstance(payload, dict) or set(payload) != expected
            or (operation == "connect" and (not isinstance(payload["device_id"], str)
                or not 1 <= len(payload["device_id"]) <= 128))):
        return JSONResponse({**_brainbit_info(), "error": "Select a discovered device." if
                             operation == "connect" else "Expected an empty JSON object."},
                            status_code=400, headers=headers)
    driver = _state.get("brainbit")
    if driver is None or _brainbit_info().get("state") == "disabled":
        return JSONResponse({**_brainbit_info(), "error": "BrainBit is disabled in config.yaml."},
                            status_code=503, headers=headers)
    action = "status" if operation == "refresh" else operation
    args = {"refresh": True} if operation == "refresh" else payload
    # All user operations retain the shared schema, gate and journal boundary.
    outcome = await asyncio.to_thread(actions.dispatch, "hw_call",
                                     {"device": "brainbit", "action": action, "args": args},
                                     actor="user")
    info = driver.status()
    error = outcome.get("error") or (outcome.get("result") or {}).get("error")
    if outcome.get("needs_confirmation"):
        error = "BrainBit operation requires confirmation through the command interface."
    await _broadcast({"type": "brainbit_status", **info})
    return JSONResponse({**info, **({"error": error} if error else {})},
                        status_code=503 if error else 200, headers=headers)


def _multimodal_local_request(request):
    return ("origin" not in request.headers
            and request.headers.get("x-intuition-multimodal") == "1"
            and request.headers.get("host", "").split(":", 1)[0]
            in ("127.0.0.1", "localhost"))


@app.get("/multimodal/status")
@app.get("/multimodal/preview")
async def multimodal_status(request: Request):
    headers = {"Cache-Control": "no-store"}
    if not _multimodal_local_request(request):
        return JSONResponse({"error": "Combined preview is only available through the local HUD."},
                            status_code=403, headers=headers)
    preview = _state.get("multimodal")
    # Raw frames and samples have exactly one private, non-caching path. They
    # never enter the action dispatcher, WebSocket broadcasts, or journal.
    info = (preview.snapshot() if preview and request.url.path.endswith("/preview")
            else _multimodal_info())
    return JSONResponse(info, headers=headers)


@app.post("/multimodal/{operation}")
async def multimodal_operation(operation: str, request: Request):
    headers = {"Cache-Control": "no-store"}
    if not _multimodal_local_request(request):
        return JSONResponse({"error": "Combined preview controls require the local HUD."},
                            status_code=403, headers=headers)
    operations = {"start": "start", "stop": "stop", "contact": "check_contact",
                  "arm": "arm", "calibrate": "mark_trial", "reset_calibration": "reset_calibration",
                  "control_mode": "set_control_mode", "eeg_trial": "eeg_trial",
                  "eeg_train": "eeg_train", "eeg_reset": "eeg_reset",
                  "eeg_arm": "eeg_arm", "eeg_guard": "eeg_guard"}
    if operation not in operations:
        return JSONResponse({"error": "Unknown combined preview operation."}, status_code=404, headers=headers)
    try:
        payload = await request.json()
    except (ValueError, UnicodeError):
        payload = None
    expected = {"arm": {"enabled"}, "calibrate": {"label"}, "control_mode": {"mode"},
                "eeg_trial": {"label", "phase"}, "eeg_arm": {"enabled"},
                "eeg_guard": {"token"}}.get(operation, set())
    if operation == "eeg_arm" and isinstance(payload, dict) and payload.get("enabled") is True:
        expected = {"enabled", "guard_token"}
    valid = isinstance(payload, dict) and set(payload) == expected
    if valid and operation in ("arm", "eeg_arm"):
        valid = type(payload["enabled"]) is bool
    if valid and operation == "calibrate":
        valid = payload["label"] in ("left", "right")
    if valid and operation == "control_mode":
        valid = payload["mode"] in ("webcam", "eeg")
    if valid and operation == "eeg_trial":
        valid = payload["label"] in ("left", "right", "rest") and payload["phase"] in ("train", "validate")
    token = (payload.get("token") if operation == "eeg_guard" else payload.get("guard_token")) if valid else None
    if valid and (operation == "eeg_guard" or operation == "eeg_arm" and payload["enabled"]):
        # The token binds explicit arming to the desktop main process's live
        # Escape shortcut lease. It is not authentication against local code.
        valid = isinstance(token, str) and re.fullmatch(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}", token) is not None
    if not valid:
        return JSONResponse({"error": "Invalid fields or values for this combined preview operation."},
                            status_code=400, headers=headers)
    preview = _state.get("multimodal")
    if preview is None:
        return JSONResponse({**_multimodal_info(), "error": "Restart the backend to enable the combined preview."},
                            status_code=503, headers=headers)
    method = getattr(preview, operations[operation])
    args = ((payload["enabled"], payload.get("guard_token")) if operation == "eeg_arm" else
            (payload["enabled"],) if operation == "arm" else
            (payload["label"],) if operation == "calibrate" else
            (payload["mode"],) if operation == "control_mode" else
            (payload["label"], payload["phase"]) if operation == "eeg_trial" else
            (payload["token"],) if operation == "eeg_guard" else ())
    # Serialize acquisition starts against normal camera starts. Stop/disarm
    # bypass that lock so they remain available during blocked SDK startup.
    if operation in ("start", "contact"):
        async with _state["gesture_lock"]:
            outcome = await asyncio.to_thread(method, *args)
    else:
        outcome = await asyncio.to_thread(method, *args)
    return JSONResponse(outcome, status_code=409 if outcome.get("error") else 200, headers=headers)


@app.get("/health")
async def health():
    """Report that the local API is responsive without querying other services."""
    return {"ok": True, "version": "1.0"}
