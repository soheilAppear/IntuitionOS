"""Polling scheduler for reminders and payload execution.

Four things were wrong, and the first is the kind that makes a feature quietly
useless rather than visibly broken.

`parse_when` parsed with RETURN_AS_TIMEZONE_AWARE=False and then called
`time.mktime(dt.timetuple())`. dateparser interpreted the text in the *configured*
timezone; mktime interpreted the resulting naive tuple in the *system* timezone.
When those differ — which is exactly when someone bothers to set `timezone:` in
config.yaml — every reminder fires wrong by the offset, with nothing anywhere
reporting an error. Parsing timezone-aware and storing a UTC epoch removes the
ambiguity rather than papering over it.

The rest: `repeat` was accepted and silently ignored; `_run` swallowed every
exception, so a persistent failure was invisible; a task was marked done the
instant it fired, so a notification missed with the app closed was lost forever;
and `execute_cb` ran a stored payload with no gate at all.
"""

import json
import threading
import time
from datetime import datetime, timezone as dt_timezone

import dateparser

try:  # Python 3.9+
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None

# Repeat intervals we accept, in seconds.
_REPEATS = {
    "": None, "none": None, "once": None,
    "hourly": 3600, "daily": 86400, "weekly": 7 * 86400, "monthly": 30 * 86400,
}


class Scheduler:
    def __init__(self, db_path: str, tz: str, tick_seconds: int, notify_cb, execute_cb,
                 logger=None, dispatcher=None):
        # Save fields
        self.db_path = db_path
        self.tz = tz
        self.tick_seconds = tick_seconds
        self.notify_cb = notify_cb
        self.execute_cb = execute_cb
        self.log = logger or (lambda s: None)
        self.dispatcher = dispatcher
        # External memory will be assigned by set_scheduler
        self.memory = None
        # Start thread
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="scheduler", daemon=True)
        self._thread.start()

    def set_memory(self, mem):
        # Bind memory object
        self.memory = mem

    def set_dispatcher(self, dispatcher):
        # Bind the gated action registry used for scheduled payloads
        self.dispatcher = dispatcher

    # ── Time ─────────────────────────────────────────────────────────────

    def parse_when(self, when: str):
        """Parse a friendly time and return a UTC epoch.

        The old version mixed two timezones: dateparser read the text in
        `self.tz`, then time.mktime read the naive result in the system zone. The
        fix is to stay timezone-aware the whole way through and convert once, at
        the end, so there is never a naive datetime for anything to guess about.
        """
        if not when:
            return None
        dt = dateparser.parse(
            when,
            settings={
                "TIMEZONE": self.tz,
                "RETURN_AS_TIMEZONE_AWARE": True,
                "PREFER_DATES_FROM": "future",  # "at 9am" means the next 9am
                "RELATIVE_BASE": self._now_local(),
            },
        )
        if dt is None:
            return None
        if dt.tzinfo is None:
            # dateparser can still hand back something naive for odd inputs;
            # attach the configured zone rather than letting the platform guess.
            dt = dt.replace(tzinfo=self._tzinfo())
        return dt.astimezone(dt_timezone.utc).timestamp()

    def _tzinfo(self):
        if ZoneInfo is not None:
            try:
                return ZoneInfo(self.tz)
            except Exception:
                self.log(f"scheduler: unknown timezone {self.tz!r}, falling back to UTC")
        return dt_timezone.utc

    def _now_local(self):
        return datetime.now(self._tzinfo())

    # ── Creating ─────────────────────────────────────────────────────────

    def create(self, title: str, when: str, payload: dict = None, repeat: str = ""):
        # Parse when
        due_ts = self.parse_when(when)
        if due_ts is None:
            return {"error": f"could not parse when: {when}"}

        interval = self.repeat_seconds(repeat)
        if repeat and interval is None and repeat.lower() not in ("", "none", "once"):
            return {"error": f"unknown repeat {repeat!r}; use "
                             f"{', '.join(k for k in _REPEATS if k)}"}

        # `repeat` used to be accepted here and then never used anywhere. It now
        # travels with the task, so a repeating reminder is rescheduled when it
        # fires instead of quietly becoming a one-off.
        body = dict(payload or {})
        if interval:
            body["_repeat"] = repeat.lower()
        payload_json = json.dumps(body)

        task_id = self.memory.create_task(title, due_ts, payload_json)
        return {"ok": True, "id": task_id, "due_ts": due_ts, "repeat": repeat or None}

    @staticmethod
    def repeat_seconds(repeat: str):
        return _REPEATS.get((repeat or "").strip().lower())

    # ── The loop ─────────────────────────────────────────────────────────

    def _run(self):
        while not self._stop:
            self.tick_safely()
            time.sleep(self.tick_seconds)

    def tick_safely(self) -> bool:
        """One tick, with failures logged rather than swallowed.

        Appendix A #7: the loop used to be `except Exception: pass`, so a
        scheduler that had stopped working looked exactly like one with nothing
        to do. Returns False when the tick failed, which is also what makes the
        behaviour testable without running the thread.
        """
        try:
            self._tick()
            return True
        except Exception as e:
            self.log(f"scheduler tick failed: {type(e).__name__}: {e}")
            return False

    def _tick(self):
        if not self.memory:
            return
        for t in self.memory.due_tasks(time.time()):
            self._fire(t)

    def _fire(self, task):
        task_id = task["id"]
        try:
            self.notify_cb(task_id, task["title"])
        except Exception as e:
            self.log(f"scheduler: notify failed for task {task_id}: {e}")

        try:
            payload = json.loads(task["payload"] or "{}")
        except Exception:
            payload = {}

        repeat = payload.pop("_repeat", None)
        if payload:
            self._execute(task_id, payload)

        interval = self.repeat_seconds(repeat) if repeat else None
        if interval:
            # Reschedule rather than complete. Stepping from the due time rather
            # than from now keeps a daily reminder on its hour instead of
            # drifting later by however long the tick took to notice it.
            self.memory.snooze_task(task_id, interval)
            self.memory.set_task_status(task_id, "pending")
            return

        # Appendix A: a task used to be marked 'done' the moment it fired, so a
        # notification the user never saw — app closed, machine asleep — was
        # indistinguishable from one they acted on. 'fired' says what actually
        # happened; the user still completes it themselves.
        self.memory.set_task_status(task_id, "fired")

    def _execute(self, task_id: int, payload: dict):
        """Run a scheduled payload through the gate.

        A payload is stored data that fires unattended, with nobody present to
        answer a confirmation prompt, so it runs as actor='scheduler' — which the
        gate restricts to capabilities that can be taken back.
        """
        name = payload.get("action")
        kwargs = payload.get("kwargs", {}) or {}
        if not name:
            return

        if self.dispatcher is None:
            self.log(f"scheduler: no dispatcher bound, skipping payload {name!r}")
            return

        try:
            result = self.dispatcher.dispatch(name, kwargs, actor="scheduler", confidence=1.0)
        except Exception as e:
            self.log(f"scheduler: payload {name!r} raised: {e}")
            return

        if isinstance(result, dict) and result.get("denied"):
            self.log(f"scheduler: payload {name!r} denied: {result.get('error')}")
        elif isinstance(result, dict) and result.get("needs_confirmation"):
            # Nobody is here to answer. Refusing is the only honest option.
            self.log(f"scheduler: payload {name!r} needs confirmation and was not run")
            return

        try:
            self.execute_cb({"action": name, "kwargs": kwargs, "result": result})
        except Exception as e:
            self.log(f"scheduler: execute callback failed for {name!r}: {e}")

    def stop(self):
        self._stop = True
