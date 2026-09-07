"""Interactive terminal adapter for shared resolution and gated execution.

``CorrectionPrompt`` owns the rendered selection, while the core resolver owns
ranking and correction feedback. Committing a displayed candidate chooses text;
``run_action`` separately obtains any capability approval required to run it.

``bootstrap`` keeps the existing nine-item collaborator contract. The REPL owns
those collaborators, replaces workers during reload/forget, and saves enabled
prediction state when the user exits normally.
"""

import json
import re
import time
from contextlib import ExitStack

import yaml
from rich import print as rprint
from rich.panel import Panel
from rich.markup import escape
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from core.llm import LLMClient
from core.memory import Memory
from core.retrieval import Retriever
from core.brain import Brain
from core.logger import make_logger
from core.actions import (
    actions,
    set_scheduler,
    set_memory,
    set_logger,
    set_safe_mode_action,
    set_thresholds,
    undo_last,
    journal_recent,
    get_journal,
)
from core.calibration import CalibrationStore, load_thresholds, reliability
from core.capabilities import capabilities
from core.consolidation import RuleStore, consolidate, render_rules
from core.context import ContextSensor
from core.episodes import Episode, EpisodeLog, PredictionWindow
from core.predictor import Predictor, PredictorStore
from core.scheduler import Scheduler
from core.anticipator import Anticipator
from core.command_resolver import (
    CorrectionSession,
    create_default_resolver,
    learning_text,
    legacy_fuzzy_slash,
)


def load_config():
    """Read configuration relative to the project's working directory."""
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_text(path: str) -> str:
    """Read a UTF-8 prompt or configuration resource."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_json(path: str) -> dict:
    """Read the planner's JSON schema without changing its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def show_help():
    """Show terminal commands and controls for the visible correction list."""
    rprint(
        Panel.fit(
            "\n".join(
                [
                    "/help - show this help",
                    "Corrections: Ctrl+N / Ctrl+P selects alternatives; Escape keeps original",
                    "Enter submits only the selected command displayed below the prompt",
                    "Available shell commands use Safe Mode, confirmation, and the journal",
                    "/exit - quit",
                    "/memory - show recent memory",
                    "/dream - consolidate the episode log into rules",
                    "/rules [--all] - what the system believes about your habits",
                    "/rules delete <id> - forget one belief",
                    '/save "text" - save a note',
                    '/recall "term" - search memory',
                    "/actions - list actions",
                    "/config - print current config",
                    "/reload - reload config and prompts",
                    "/tasks - list pending tasks",
                    "/done <id> - mark a task done",
                    "/delete <id> - delete a task",
                    "/snooze <id> <15m|2h|1d> - snooze a task",
                    "/hw - list hardware devices",
                    "/hw schema <name> - show a device schema",
                    "/safe on|off - toggle Safe Mode",
                    "/undo - reverse the last reversible action",
                    "/journal [n] - show recent gated actions",
                    "/calibration - reliability of stated confidence",
                    "/thresholds - show the cost-gated thresholds",
                    "/episodes - show what the episode log has recorded",
                    "/forget - erase the episode log",
                    "/capabilities - list actions with their declared cost",
                    '/exec "python your_script.py" [cwd] - run local venv python',
                    '/write <path> "text" - write a file',
                    "/read <path> - read a file",
                    "ls - list directory",
                    "tree - show a recursive directory view",
                    '/task_payload "{...json...}" <when> - schedule a payload action at a time',
                ]
            ),
            title="IntuitionOS Help",
        )
    )


def fuzzy_slash(base: str) -> str:
    """Compatibility for callers; submission never silently fuzzy-corrects."""
    return legacy_fuzzy_slash(base)


class CorrectionPrompt:
    """A real prompt_toolkit prompt bound to the command it last rendered.

    Resolution happens on render, not on Enter. Every edit invalidates that
    rendered snapshot. If typing and Enter arrive before a redraw, Enter asks
    for the redraw instead of accepting a correction nobody has seen.
    """

    def __init__(self, resolver, context_fn=None, **session_kwargs):
        self.resolver = resolver
        self.corrections = CorrectionSession(resolver)
        self.context_fn = context_fn
        self.context = None
        # Selection state tracks what the renderer actually showed. The core
        # snapshot's token/revision binds text, not permission to execute it.
        self._snapshot = None
        self._displayed_text = None
        self._selected_index = None
        self._displayed_index = None
        self._keep_original_text = None
        self._submitted = False
        # Keep the committed feedback row after the prompt buffer is reset so
        # dispatch can attach an outcome independently of correction acceptance.
        self.feedback_id = None
        self.submission_resolution = None
        self.submission_index = None
        bindings = KeyBindings()

        @bindings.add("c-n")
        def next_candidate(event):
            self._cycle(1)
            event.app.invalidate()

        @bindings.add("c-p")
        def previous_candidate(event):
            self._cycle(-1)
            event.app.invalidate()

        @bindings.add("escape", eager=True)
        def keep_original(event):
            self._keep_original_text = event.current_buffer.text
            self._selected_index = None
            event.app.invalidate()

        @bindings.add("enter")
        def submit(event):
            raw = event.current_buffer.text
            if not raw.strip():
                event.app.exit(result=raw)
                return
            if (
                self._snapshot is None
                or self._displayed_text != raw
                or self._selected_index != self._displayed_index
            ):
                event.app.invalidate()
                return
            try:
                selected = self.corrections.commit(
                    raw,
                    token=self._snapshot["token"],
                    revision=self._snapshot["revision"],
                    candidate_index=self._selected_index,
                )
            except ValueError:
                self._snapshot = None
                event.app.invalidate()
                return
            self.feedback_id = self.corrections.feedback_id
            self.submission_resolution = self.corrections.resolution
            self.submission_index = self._selected_index
            self._submitted = True
            self.session.history.append_string(selected)
            event.app.exit(result=selected)

        self.session = PromptSession(
            "io> ",
            key_bindings=bindings,
            bottom_toolbar=self._toolbar,
            style=Style.from_dict(
                {
                    "correction.changed": "bold ansiblack bg:ansiyellow",
                    "correction.selected": "bold ansiwhite",
                    "correction.help": "ansibrightblack",
                }
            ),
            **session_kwargs,
        )
        self.session.default_buffer.on_text_changed += self._on_text

    def _on_text(self, buffer):
        """Invalidate the displayed commitment whenever draft text changes."""
        if not self._submitted and buffer.text != self._displayed_text:
            self.corrections.invalidate()
            self._snapshot = None
            self._selected_index = None
            self._keep_original_text = None

    def _cycle(self, amount):
        """Cycle through ranked candidates and the final keep-original option."""
        if self._snapshot is None:
            return
        count = len(self._snapshot["candidates"])
        if not count:
            return
        current = count if self._selected_index is None else self._selected_index
        current = (current + amount) % (count + 1)
        self._selected_index = current if current < count else None

    def _toolbar(self):
        """Resolve the current draft and render the exact selected command."""
        raw = self.session.default_buffer.text
        if not raw.strip():
            return [
                (
                    "class:correction.help",
                    "Command corrections appear here while you type.",
                )
            ]
        if not self._submitted and (
            self._snapshot is None or self._displayed_text != raw
        ):
            self.corrections.update(raw, self.context)
            self._snapshot = self.corrections.snapshot()
            self._displayed_text = raw
            self._selected_index = (
                0
                if self._snapshot["candidates"] and self._keep_original_text != raw
                else None
            )
        if self._snapshot is None:
            return []
        candidates = self._snapshot["candidates"]
        chosen = (
            candidates[self._selected_index]
            if self._selected_index is not None
            else None
        )
        self._displayed_index = self._selected_index
        fragments = [("class:correction.help", "Enter submits: │")]
        if chosen:
            start, end = chosen["span"]
            # The replacement can be longer than the original span. Prefix and
            # suffix remain untouched, including all argument whitespace.
            replacement_end = len(chosen["text"]) - (len(raw) - end)
            fragments += [
                ("", chosen["text"][:start]),
                ("class:correction.changed", chosen["text"][start:replacement_end]),
                ("", chosen["text"][replacement_end:]),
            ]
        else:
            fragments.append(("", raw))
        fragments.append(("class:correction.help", "│"))
        if candidates:
            fragments.append(("class:correction.help", "\nCtrl+N/P: "))
            for index, candidate in enumerate(candidates):
                style = (
                    "class:correction.selected" if self._selected_index == index else ""
                )
                fragments.append((style, f"{index + 1}. {candidate['token']}  "))
            style = "class:correction.selected" if chosen is None else ""
            fragments.append((style, "Keep original"))
            fragments.append(("class:correction.help", "  | Escape: original"))
        elif self._snapshot.get("reason") and self._snapshot["status"] != "exact":
            fragments.append(("class:correction.help", "\n" + self._snapshot["reason"]))
        return fragments

    def prompt(self, message="io> "):
        """Start one input window and return only its committed display text."""
        self.context = self.context_fn() if self.context_fn else None
        self._submitted = False
        self._snapshot = None
        self._displayed_text = None
        self._selected_index = None
        self._keep_original_text = None
        self.feedback_id = None
        self.submission_resolution = None
        self.submission_index = None
        return self.session.prompt(message)

    def outcome(self, value):
        """Attach a categorical execution result to the committed feedback row."""
        if self.feedback_id is not None and self.resolver.feedback:
            self.resolver.feedback.record_outcome(self.feedback_id, value)

    def reset(self, resolver=None):
        """Drop displayed text and feedback after /forget or a reload."""
        if resolver is not None:
            self.resolver = resolver
        self.corrections.invalidate()
        self.corrections = CorrectionSession(self.resolver)
        self._snapshot = None
        self._displayed_text = None
        self._keep_original_text = None
        self.feedback_id = None
        self.submission_resolution = None


def _execution_outcome(result):
    """Map action results to feedback categories without retaining output text."""
    if isinstance(result, dict):
        if result.get("denied"):
            return "denied"
        if result.get("cancelled"):
            return "cancelled"
        if result.get("error") or result.get("returncode", result.get("code", 0)):
            return "error"
    return "ok"


def run_action(name, **kwargs):
    """Dispatch through the gate, asking at the prompt if the gate says to.

    The REPL and the HUD answer a CONFIRM the same way — nothing has run when
    the parked result comes back, and the token is what unparks it — they just
    ask the question through different surfaces.
    """
    res = actions.call(name, **kwargs)
    if not (isinstance(res, dict) and res.get("needs_confirmation")):
        return res

    detail = " ".join(f"{k}={v}" for k, v in (res.get("args") or {}).items())
    title = (
        "CANNOT BE UNDONE" if res.get("reversibility") == "irreversible" else "Confirm"
    )
    rprint(
        Panel.fit(
            escape(f"{res['capability']}  {detail}\n{res.get('reason', '')}"),
            title=title,
            border_style="red"
            if res.get("reversibility") == "irreversible"
            else "yellow",
        )
    )
    try:
        answer = input("proceed? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"
    return actions.confirm(res["token"], granted=answer in ("y", "yes"))


def bootstrap():
    """Initialize the existing terminal services and return their stable tuple."""
    cfg = load_config()
    sys_prompt = read_text(cfg.get("system_prompt_path", "config/system_prompt.txt"))
    schema = read_json(cfg.get("planner_schema_path", "config/planner_schema.json"))

    mem = Memory(cfg.get("memory_db_path", "data/intuition.db"))
    logger = make_logger(cfg.get("log_path", "data/log.txt"))
    set_logger(logger)
    set_memory(mem)

    thresholds = load_thresholds(cfg.get("thresholds"))
    set_thresholds(thresholds)

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
        rprint(Panel.fit(f"Remind #{task_id}: {title}", title="Reminder"))
        mem.add("reminder", f"#{task_id} {title}", tags="reminder")

    def execute(outcome: dict):
        # The scheduler dispatches through the gate itself now, as actor
        # 'scheduler', so this reports rather than being a second execution path
        # with its own private allowlist.
        mem.add("tool", json.dumps({"scheduled": True, **outcome}, default=str)[:2000])
        rprint(
            Panel.fit(
                f"Scheduled action {outcome.get('action')} -> {outcome.get('result')}",
                title="Run",
            )
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

    # Register hardware drivers
    drivers = cfg.get("hardware", {}).get("drivers", [])
    for d in drivers:
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

    # Every submitted input is recorded whether or not the user asks. That is
    # the involuntary encoding /save lacks; it is disclosed in the README and can
    # be switched off in config.yaml.
    ecfg = cfg.get("episodes", {}) or {}
    episodes = EpisodeLog(mem, enabled=bool(ecfg.get("enabled", True)))
    sensor = ContextSensor(journal=get_journal())

    calibration_store = CalibrationStore(mem)
    rule_store = RuleStore(mem)
    pcfg = cfg.get("prediction", {}) or {}
    predictor = Predictor(
        store=PredictorStore(mem),
        half_life_s=float(pcfg.get("half_life_s", 7 * 24 * 3600)),
        min_episodes=int(pcfg.get("min_episodes", 50)),
        calibrator=calibration_store.load(),
        rules=rule_store,
    )
    if predictor.seen == 0:
        # No saved state: relearn from the log rather than starting cold.
        predictor.fit(episodes.recent(limit=int(pcfg.get("replay_limit", 5000))))

    return (
        cfg,
        brain,
        mem,
        sched,
        episodes,
        sensor,
        predictor,
        rule_store,
        calibration_store,
    )


def make_anticipator(brain, cfg, predictor=None, sensor=None):
    """Start a predictor-driven worker whose speculative actions use the gate."""

    def prewarm(prediction):
        # Speculative work runs as the anticipator, never as the user — that
        # actor is what confines it to free capabilities.
        text = prediction.action
        conf = prediction.confidence

        def warm(name, args):
            return str(
                actions.dispatch(name, args, actor="anticipator", confidence=conf)
            )[:4000]

        t = text.strip()
        common = {"confidence": conf, "why": prediction.why, "action": text}
        if t == "tree" or t.startswith("tree "):
            return (text, dict(common, reply=warm("list_tree", {"path": "."})))
        if t == "ls":
            return (text, dict(common, reply=warm("list_dir", {"path": "."})))
        if t.startswith("read file "):
            path = t[len("read file ") :].strip()
            return (text, dict(common, reply=warm("read_file", {"path": path})))
        # Nothing cheap to precompute, but the prediction itself still has value.
        return (text, common)

    a = cfg.get("anticipation", {}) or {}
    ant = Anticipator(
        prewarm_fn=prewarm,
        predictor=predictor,
        context_fn=(lambda: sensor.snapshot()) if sensor else None,
        enabled=bool(a.get("enabled", True)),
        debounce_ms=int(a.get("debounce_ms", 180)),
        match_threshold=float(a.get("match_threshold", 0.6)),
        thresholds=cfg.get("thresholds"),
    )
    ant.start()
    return ant


def main():
    """Run the REPL and always stop its current workers on exit or failure."""
    with ExitStack() as cleanup:
        _run_terminal(cleanup)


def _run_terminal(cleanup):
    """Run the terminal, keeping raw execution text separate from learning data."""
    cfg, brain, mem, sched, episodes, sensor, predictor, rules, calib_store = (
        bootstrap()
    )
    # Closures read the current services after /reload and /forget replacements.
    cleanup.callback(lambda: sched.stop())
    window = PredictionWindow()
    # Print banner
    rprint(
        Panel.fit(
            "IntuitionOS - learned prediction + gated actions with /undo - type /help",
            title="IntuitionOS",
        )
    )

    # Create prompt session
    resolver = create_default_resolver(mem, enabled=lambda: episodes.enabled)
    session = CorrectionPrompt(resolver, context_fn=lambda: sensor.snapshot())
    # Make anticipator
    ant = make_anticipator(brain, cfg, predictor=predictor, sensor=sensor)
    # The buffer hook outlives worker replacements on both /reload and /forget.
    _ant_ref = [ant]
    cleanup.callback(lambda: _ant_ref[0].stop())

    def _on_text(buf):
        _ant_ref[0].update_buffer(buf.text)
        window.note_keystroke(buf.text)

    session.session.default_buffer.on_text_changed += _on_text

    # Main REPL loop
    while True:
        try:
            raw_user = session.prompt("io> ")
        except (EOFError, KeyboardInterrupt):
            try:
                predictor.save()
            except Exception:
                pass
            rprint("\nbye.")
            break

        if not raw_user.strip():
            continue

        # Normalization here is only for the existing IntuitionOS built-in
        # parsers. External execution below receives raw_user byte for byte.
        user = raw_user.strip()
        _ctx = session.context or sensor.snapshot()
        resolved = resolver.resolve(raw_user, _ctx)

        # Keep correction acceptance in its own feedback table; it is not a
        # calibrated next-action prediction or an execution-success signal.
        _signals = window.take(raw_user)
        learned_input = user
        if resolved.namespace in ("shell", "git") or re.match(r"^/exec\s", user):
            # Correction preferences and general prediction must not become a
            # second store of potentially secret shell argument values.
            learned_input = learning_text(raw_user)
            _signals.update(
                keystroke_prefix=learned_input,
                predicted=None,
                predicted_conf=None,
                accepted_prediction=None,
            )
        _episode_id = episodes.record(learned_input, _ctx, **_signals)
        sensor.note_submission(learned_input)
        # This submission is the ground truth for whatever was predicted a moment
        # ago, including the times the prediction was wrong.
        if episodes.enabled:
            predictor.update(
                Episode(
                    ts=time.time(),
                    action=learned_input,
                    context=_ctx,
                    keystroke_prefix=_signals.get("keystroke_prefix") or learned_input,
                )
            )

        _outcome = {"value": "ok"}

        def dispatch(name, **kwargs):
            """Use the common approval flow and collect this episode's result."""
            result = run_action(name, **kwargs)
            episodes.set_capability(_episode_id, name)
            outcome = _execution_outcome(result)
            _outcome["value"] = outcome
            if isinstance(result, dict):
                sensor.note_exit_code(result.get("returncode", result.get("code")))
            return result

        try:
            # Built-ins
            if user == "/exit":
                try:
                    predictor.save()
                except Exception:
                    pass
                rprint("bye.")
                break
            if user == "/help":
                show_help()
                continue
            if user == "/config":
                rprint(cfg)
                continue
            if user == "/reload":
                try:
                    sched.stop()
                    (
                        cfg,
                        brain,
                        mem,
                        sched,
                        episodes,
                        sensor,
                        predictor,
                        rules,
                        calib_store,
                    ) = bootstrap()
                    _ant_ref[0].stop()
                    ant = make_anticipator(
                        brain, cfg, predictor=predictor, sensor=sensor
                    )
                    _ant_ref[0] = ant
                    resolver = create_default_resolver(
                        mem, enabled=lambda: episodes.enabled
                    )
                    session.reset(resolver)
                    _episode_id = None
                    rprint({"result": "reloaded"})
                except Exception as e:
                    rprint(f"reload error: {e}")
                continue
            if user == "/actions":
                rprint(sorted(actions.names))
                continue
            if user == "/memory":
                rows = mem.recent(limit=12)
                for _id, ts, role, text, tags in rows[::-1]:
                    rprint(f"[{role}] {text}")
                continue
            if user == "/dream":
                ccfg = cfg.get("consolidation", {}) or {}
                report = consolidate(
                    episodes.recent(limit=int(ccfg.get("window", 2000))),
                    rules,
                    llm=brain.llm,
                    min_support=int(ccfg.get("min_support", 4)),
                    min_confidence=float(ccfg.get("min_confidence", 0.5)),
                    calibrator=predictor.calibrator,
                    calibration_store=calib_store,
                )
                rprint(Panel.fit(report.summary(), title="Consolidation"))
                continue
            if user.startswith("/rules"):
                parts = user.split()
                if len(parts) >= 3 and parts[1] == "delete":
                    try:
                        rid = int(parts[2])
                    except ValueError:
                        rprint("usage: /rules delete <id>")
                        continue
                    rprint(
                        f"Deleted rule #{rid}."
                        if rules.delete(rid)
                        else f"No rule #{rid}."
                    )
                    continue
                show_all = len(parts) >= 2 and parts[1] in ("--all", "all")
                rprint(render_rules(rules.all(active_only=not show_all)))
                continue
            if user.startswith("/save "):
                note = user[6:].strip().strip('"')
                mem.add("note", note, tags="note")
                rprint("saved.")
                continue
            if user.startswith("/recall "):
                term = user[8:].strip().strip('"')
                # Notes first: the transcript would otherwise bury them (Appendix A #16).
                hits = brain.retriever.search(term, limit=12, roles=("note",))
                if not hits:
                    hits = brain.retriever.search(term, limit=12)
                for h in hits:
                    rprint(f"[{h.role}] {h.text}")
                cued = [
                    n
                    for n in brain.retriever.retrieve("", _ctx, k=2)
                    if all(n.id != h.id for h in hits)
                ]
                if cued:
                    rprint("[dim]Also relevant here right now:[/dim]")
                    for n in cued:
                        rprint(f"  - {n.text}")
                continue
            if user == "/tasks":
                rows = mem.list_open()
                if not rows:
                    rprint("No open tasks.")
                for t in rows:
                    when = time.strftime("%b %d %H:%M", time.localtime(t["due"]))
                    mark = " [fired]" if t["status"] == "fired" else ""
                    rprint(f"#{t['id']} {t['title']}  ({when}){mark}")
                continue
            if user.startswith("/done "):
                try:
                    tid = int(user.split(" ", 1)[1])
                    rprint(dispatch("complete_task", task_id=tid))
                except Exception:
                    rprint("usage: /done <id>")
                continue
            if user.startswith("/delete "):
                try:
                    tid = int(user.split(" ", 1)[1])
                    rprint(dispatch("delete_task", task_id=tid))
                except Exception:
                    rprint("usage: /delete <id>")
                continue
            if user.startswith("/snooze "):
                try:
                    _, tid, d = user.split(" ")
                    rprint(dispatch("snooze_task", task_id=int(tid), delta=d))
                except Exception:
                    rprint("usage: /snooze <id> <15m|2h|1d>")
                continue
            if user == "/hw":
                rprint(dispatch("hw_list"))
                continue
            if user.startswith("/hw schema "):
                name = user.split(" ", 2)[2]
                rprint(dispatch("hw_schema", device=name))
                continue
            if user.startswith("/task_payload "):
                try:
                    _, payload_json, when = user.split(" ", 2)
                    payload = json.loads(payload_json.strip("\"'"))
                    rprint(
                        dispatch(
                            "create_task",
                            text=None,
                            when=when,
                            repeat="",
                            payload=payload,
                        )
                    )
                except Exception as e:
                    rprint(f"usage: /task_payload '{{...json...}}' <when>. error: {e}")
                continue
            if user == "/calibration":
                rprint(reliability(episodes.shown_predictions()).table())
                continue
            if user == "/thresholds":
                for k, v in (cfg.get("thresholds") or {}).items():
                    rprint(f"  {k:<14} {'never' if v is None else v}")
                continue
            if user == "/forget":
                # Delete persisted learning first, then replace every in-memory
                # reader/worker so an old model cannot repopulate forgotten data.
                forgotten = episodes.forget()
                _ant_ref[0].stop()
                rules = RuleStore(mem)
                calib_store = CalibrationStore(mem)
                pcfg = cfg.get("prediction", {}) or {}
                predictor = Predictor(
                    store=PredictorStore(mem),
                    half_life_s=float(pcfg.get("half_life_s", 7 * 24 * 3600)),
                    min_episodes=int(pcfg.get("min_episodes", 50)),
                    calibrator=calib_store.load(),
                    rules=rules,
                )
                sensor = ContextSensor(journal=get_journal())
                window.reset()
                resolver = create_default_resolver(
                    mem, enabled=lambda: episodes.enabled
                )
                session.reset(resolver)
                ant = make_anticipator(brain, cfg, predictor=predictor, sensor=sensor)
                _ant_ref[0] = ant
                rprint(
                    f"Forgot {forgotten} episode(s) and learned correction preferences."
                )
                continue
            if user == "/episodes":
                rows = episodes.recent(limit=15)
                if not rows:
                    rprint(
                        "No episodes recorded yet."
                        if episodes.enabled
                        else "Episode logging is disabled in config.yaml."
                    )
                for e in rows:
                    when = time.strftime("%H:%M:%S", time.localtime(e.ts))
                    hint = ""
                    if e.accepted_prediction is not None:
                        hint = (
                            "  hint taken"
                            if e.accepted_prediction
                            else f"  hint ignored ({e.predicted})"
                        )
                    rprint(f"{when}  {e.action}{hint}")
                continue
            if user == "/undo":
                rprint(undo_last())
                continue
            if user.startswith("/journal"):
                parts = user.split()
                limit = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 15
                rows = journal_recent(limit=limit).get("rows", [])
                if not rows:
                    rprint("The journal is empty.")
                for r in rows:
                    when = time.strftime("%H:%M:%S", time.localtime(r["ts"]))
                    mark = (
                        " (undone)"
                        if r["undone_at"]
                        else (" [undoable]" if r["undo"] else "")
                    )
                    outcome = f"/{r['outcome']}" if r["outcome"] else ""
                    rprint(
                        f"#{r['id']} {when} {r['actor']}/{r['capability']} {r['decision']}{outcome}{mark}"
                    )
                continue
            if user == "/capabilities":
                for c in capabilities.manifest():
                    flag = "confirm" if c["requires_confirmation"] else ""
                    rprint(
                        f"{c['name']:<24} {c['reversibility']:<13} {flag:<8} {c['summary']}"
                    )
                continue
            if user.startswith("/safe"):
                parts = user.split()
                if len(parts) >= 2:
                    state = parts[1]
                    rprint(set_safe_mode_action(state=state))
                else:
                    rprint("usage: /safe on|off")
                continue
            if re.match(r"^/exec\s", user):
                try:
                    rest = re.match(r"^\s*/exec\s(.*)$", raw_user, re.DOTALL).group(1)
                    cmd = None
                    cwd = "."
                    legacy = rest.lstrip()
                    if legacy.startswith('"'):
                        idx = legacy.find('"', 1)
                        if idx != -1:
                            cmd = legacy[1:idx]
                            tail = legacy[idx + 1 :].strip()
                            cwd = tail or "."
                    if cmd is not None:
                        rprint(dispatch("run_local", cmd=cmd, cwd=cwd))
                    elif resolved.status == "unsupported" and resolved.namespace in (
                        "shell",
                        "git",
                    ):
                        rprint({"error": resolved.reason})
                        _outcome["value"] = "unsupported"
                    else:
                        rprint(dispatch("run_command", cmd=rest, cwd="."))
                except Exception as e:
                    rprint(f'usage: /exec "python your_script.py" [cwd]. error: {e}')
                continue
            if user.startswith("/write "):
                try:
                    _, rest = user.split(" ", 1)
                    rest = rest.strip()
                    if '"' in rest:
                        path, txt = rest.split('"', 1)
                        path = path.strip()
                        if not txt.endswith('"'):
                            last = txt.rfind('"')
                            if last != -1:
                                txt = txt[: last + 1]
                        text = txt.strip('"')
                    else:
                        path, text = rest.split(" ", 1)
                    rprint(dispatch("write_file", path=path, text=text))
                except Exception as e:
                    rprint(f'usage: /write <path> "text". error: {e}')
                continue
            if user.startswith("/read "):
                try:
                    path = user.split(" ", 1)[1].strip()
                    rprint(dispatch("read_file", path=path))
                except Exception as e:
                    rprint(f"usage: /read <path>. error: {e}")
                continue

            # Shortcuts for common shell like inputs
            if user == "ls":
                rprint(dispatch("list_dir", path="."))
                continue
            if user == "tree":
                rprint(dispatch("list_tree", path="."))
                continue

            # Deterministic shell recognition precedes speculation or an LLM. The
            # selected text is never corrected a second time during dispatch.
            if resolved.namespace in ("shell", "git"):
                if resolved.status == "unsupported":
                    rprint({"error": resolved.reason})
                    _outcome["value"] = "unsupported"
                elif resolved.status == "exact" or resolved.namespace == "git":
                    rprint(dispatch("run_command", cmd=raw_user, cwd="."))
                else:
                    rprint(
                        {
                            "error": "Unknown command kept unchanged. Select a displayed correction or edit it."
                        }
                    )
                    _outcome["value"] = "unsupported"
                continue
            if user.startswith("/"):
                rprint(
                    {
                        "error": "Unknown IntuitionOS command kept unchanged. Type /help for commands."
                    }
                )
                _outcome["value"] = "unsupported"
                continue

            # Natural language remind: "remind me <title> in/at <when>"
            _m = re.match(
                r"^remind\s+me\s+(.+?)\s+((?:in|at)\s+\S.*)$", user, re.IGNORECASE
            )
            if _m:
                rprint(
                    dispatch(
                        "create_task",
                        text=_m.group(1).strip(),
                        when=_m.group(2).strip(),
                    )
                )
                continue

            # Try anticipator cache first
            pre = None
            try:
                pre = ant.try_serve(user)
            except Exception:
                pre = None
            if isinstance(pre, dict) and ("reply" in pre or "plan" in pre):
                plan = pre.get("plan") or []
                if isinstance(plan, str):
                    plan = [plan]
                if plan:
                    rprint(Panel.fit("\n".join(f"- {p}" for p in plan), title="Plan"))
                rprint(pre.get("reply", ""))
                mem.add("assistant", pre.get("reply", ""))
                continue

            # Default to LLM brain — a real propose/validate/execute/observe loop.
            out = brain.step(user, context=_ctx)
            while out.get("needs_confirmation"):
                detail = " ".join(
                    f"{k}={v}" for k, v in (out.get("args") or {}).items()
                )
                title = (
                    "CANNOT BE UNDONE"
                    if out.get("reversibility") == "irreversible"
                    else "Confirm"
                )
                rprint(
                    Panel.fit(
                        escape(
                            f"{out['capability']}  {detail}"
                            + chr(10)
                            + f"{out.get('reason', '')}"
                        ),
                        title=title,
                        border_style="red"
                        if out.get("reversibility") == "irreversible"
                        else "yellow",
                    )
                )
                try:
                    answer = input("proceed? [y/N] ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    answer = "n"
                out = brain.resume(out["resume_token"], granted=answer in ("y", "yes"))
            plan = out.get("plan") or []
            if isinstance(plan, str):
                plan_lines = [plan]
            elif isinstance(plan, list):
                plan_lines = plan
            else:
                plan_lines = [str(plan)]
            if plan_lines:
                rprint(Panel.fit("\n".join(f"- {p}" for p in plan_lines), title="Plan"))
            rprint(out.get("reply", ""))
            _outcome["value"] = _execution_outcome(out)
        except Exception:
            _outcome["value"] = "error"
            raise
        finally:
            episodes.set_outcome(_episode_id, _outcome["value"])
            session.outcome(_outcome["value"])
