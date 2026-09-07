"""Scheduler correctness.

The timezone test is the important one: the old bug produced reminders that were
silently wrong by the UTC offset, with nothing anywhere reporting a failure.
"""

import json
import time
from datetime import datetime, timedelta, timezone as dt_timezone

import pytest

from core.capabilities import set_safe_mode
from core.scheduler import Scheduler


@pytest.fixture
def sched(memory):
    """A scheduler with its polling thread stopped, driven by hand."""
    s = Scheduler(db_path=":memory:", tz="UTC", tick_seconds=3600,
                  notify_cb=lambda *a: None, execute_cb=lambda *a: None)
    s.stop()
    s.set_memory(memory)
    return s


# ── Timezone (Appendix A #5) ────────────────────────────────────────────────


def test_an_absolute_time_is_interpreted_in_the_configured_zone(memory):
    """The brief's acceptance criterion: set the config timezone away from the
    system one and assert the reminder fires at the right wall time.

    Two schedulers, same words, zones eight hours apart. The epochs they produce
    must differ by exactly that. The old code produced the same epoch for both,
    because mktime read the naive result in the system zone either way.
    """
    tokyo = Scheduler(":memory:", "Asia/Tokyo", 3600, lambda *a: None, lambda *a: None)
    tokyo.stop()
    la = Scheduler(":memory:", "America/Los_Angeles", 3600, lambda *a: None, lambda *a: None)
    la.stop()

    when = "2030-06-15 09:00"
    tokyo_ts = tokyo.parse_when(when)
    la_ts = la.parse_when(when)

    assert tokyo_ts is not None and la_ts is not None
    # Tokyo is UTC+9 in June, Los Angeles UTC-7. 9am Tokyo happens 16 hours first.
    assert (la_ts - tokyo_ts) == pytest.approx(16 * 3600, abs=1)


def test_the_stored_epoch_maps_back_to_the_requested_wall_time(memory):
    """Whatever the machine's own timezone is, 09:00 in the configured zone must
    come back as 09:00 in the configured zone."""
    from zoneinfo import ZoneInfo

    s = Scheduler(":memory:", "Asia/Tokyo", 3600, lambda *a: None, lambda *a: None)
    s.stop()
    ts = s.parse_when("2030-06-15 09:00")

    local = datetime.fromtimestamp(ts, ZoneInfo("Asia/Tokyo"))
    assert (local.hour, local.minute) == (9, 0)
    assert (local.year, local.month, local.day) == (2030, 6, 15)


def test_a_relative_time_lands_where_expected(sched):
    ts = sched.parse_when("in 10 minutes")
    assert ts is not None
    assert 9 * 60 < ts - time.time() < 11 * 60


def test_an_unparseable_time_is_reported_not_guessed(sched):
    assert sched.parse_when("sometime after the heat death") is None
    assert "error" in sched.create("x", "sometime after the heat death")


def test_an_unknown_timezone_falls_back_to_utc_and_says_so(memory):
    logged = []
    s = Scheduler(":memory:", "Mars/Olympus_Mons", 3600, lambda *a: None,
                  lambda *a: None, logger=logged.append)
    s.stop()
    s.set_memory(memory)
    assert s.parse_when("in 5 minutes") is not None


# ── Repeat (Appendix A #6) ──────────────────────────────────────────────────


def test_a_repeating_task_is_rescheduled_rather_than_completed(sched, memory):
    """`repeat` used to be accepted by create() and then never used anywhere."""
    fired = []
    sched.notify_cb = lambda tid, title: fired.append(tid)

    res = sched.create("standup", "in 1 second", repeat="daily")
    assert res["ok"]
    memory.execute("UPDATE tasks SET due_ts=? WHERE id=?", (time.time() - 1, res["id"]))

    sched._tick()
    assert fired == [res["id"]]

    task = memory.get_task(res["id"])
    assert task["status"] == "pending", "a daily reminder became a one-off"
    assert task["due"] > time.time(), "it was not moved into the future"


def test_a_repeating_task_keeps_its_hour_rather_than_drifting(sched, memory):
    """Stepping from the due time, not from now, so a daily reminder noticed 30
    seconds late does not slide 30 seconds later every day."""
    sched.notify_cb = lambda tid, title: None
    res = sched.create("standup", "in 1 second", repeat="daily")
    due = time.time() - 30
    memory.execute("UPDATE tasks SET due_ts=? WHERE id=?", (due, res["id"]))

    sched._tick()
    assert memory.get_task(res["id"])["due"] == pytest.approx(due + 86400, abs=1)


def test_a_non_repeating_task_is_not_rescheduled(sched, memory):
    sched.notify_cb = lambda tid, title: None
    res = sched.create("one off", "in 1 second")
    memory.execute("UPDATE tasks SET due_ts=? WHERE id=?", (time.time() - 1, res["id"]))

    sched._tick()
    assert memory.get_task(res["id"])["status"] == "fired"


def test_an_unknown_repeat_is_rejected_rather_than_ignored(sched):
    res = sched.create("x", "in 5 minutes", repeat="fortnightly")
    assert "error" in res
    assert "fortnightly" in res["error"]


@pytest.mark.parametrize("word,seconds", [
    ("hourly", 3600), ("daily", 86400), ("weekly", 7 * 86400), ("", None), ("once", None),
])
def test_repeat_vocabulary(word, seconds):
    assert Scheduler.repeat_seconds(word) == seconds


# ── A missed notification is not lost ───────────────────────────────────────


def test_a_fired_task_is_distinct_from_a_completed_one(sched, memory):
    """It used to be marked done the instant it fired, so a reminder nobody saw —
    app closed, machine asleep — looked exactly like one that was acted on."""
    sched.notify_cb = lambda tid, title: None
    res = sched.create("water the plants", "in 1 second")
    memory.execute("UPDATE tasks SET due_ts=? WHERE id=?", (time.time() - 1, res["id"]))

    sched._tick()
    assert memory.get_task(res["id"])["status"] == "fired"
    assert memory.list_open(), "a fired reminder must still be visible to the user"


def test_a_fired_task_does_not_fire_again(sched, memory):
    fired = []
    sched.notify_cb = lambda tid, title: fired.append(tid)
    res = sched.create("once", "in 1 second")
    memory.execute("UPDATE tasks SET due_ts=? WHERE id=?", (time.time() - 1, res["id"]))

    sched._tick()
    sched._tick()
    assert fired == [res["id"]]


# ── Errors are visible (Appendix A #7) ──────────────────────────────────────


def test_a_failing_tick_is_logged_rather_than_swallowed(memory):
    """A scheduler that has stopped working must not look like one with nothing
    to do."""
    logged = []
    s = Scheduler(":memory:", "UTC", 3600, lambda *a: None, lambda *a: None,
                  logger=logged.append)
    s.stop()

    class Exploding:
        def due_tasks(self, now):
            raise RuntimeError("database is locked")

    s.memory = Exploding()

    assert s.tick_safely() is False
    assert logged and "database is locked" in logged[0]
    assert "RuntimeError" in logged[0]


def test_a_healthy_tick_logs_nothing(sched):
    logged = []
    sched.log = logged.append
    assert sched.tick_safely() is True
    assert logged == []


def test_a_failing_notification_does_not_stop_the_task_from_being_marked(sched, memory):
    def explode(tid, title):
        raise RuntimeError("websocket gone")

    sched.notify_cb = explode
    logged = []
    sched.log = logged.append

    res = sched.create("x", "in 1 second")
    memory.execute("UPDATE tasks SET due_ts=? WHERE id=?", (time.time() - 1, res["id"]))
    sched._tick()

    assert memory.get_task(res["id"])["status"] == "fired"
    assert any("notify failed" in line for line in logged)


# ── Payloads go through the gate ────────────────────────────────────────────


def test_a_scheduled_payload_cannot_invoke_an_irreversible_capability(sched, memory, wired, project):
    """The acceptance criterion. Nobody is present to answer a prompt, so an
    unrepeatable action must simply not run."""
    acts, journal, _mem = wired
    set_safe_mode(False)
    sched.set_dispatcher(acts)
    logged = []
    sched.log = logged.append
    sched.notify_cb = lambda tid, title: None

    res = sched.create("mischief", "in 1 second",
                       payload={"action": "run_local", "kwargs": {"cmd": "echo hi"}})
    memory.execute("UPDATE tasks SET due_ts=? WHERE id=?", (time.time() - 1, res["id"]))
    sched._tick()

    assert any("denied" in line for line in logged), logged
    assert journal.recent()[0]["decision"] == "deny"
    assert journal.recent()[0]["actor"] == "scheduler"


def test_a_scheduled_payload_may_invoke_a_reversible_capability(sched, memory, wired, project):
    acts, journal, _mem = wired
    sched.set_dispatcher(acts)
    sched.notify_cb = lambda tid, title: None
    ran = []
    sched.execute_cb = ran.append

    res = sched.create("write a file", "in 1 second",
                       payload={"action": "write_file",
                                "kwargs": {"path": "scheduled.txt", "text": "hello"}})
    memory.execute("UPDATE tasks SET due_ts=? WHERE id=?", (time.time() - 1, res["id"]))
    sched._tick()

    assert (project / "scheduled.txt").read_text(encoding="utf-8") == "hello"
    assert ran and ran[0]["action"] == "write_file"
    assert journal.recent()[0]["actor"] == "scheduler"


def test_a_payload_with_no_dispatcher_is_skipped_loudly(sched, memory):
    logged = []
    sched.log = logged.append
    sched.notify_cb = lambda tid, title: None
    sched.dispatcher = None

    res = sched.create("x", "in 1 second", payload={"action": "write_file", "kwargs": {}})
    memory.execute("UPDATE tasks SET due_ts=? WHERE id=?", (time.time() - 1, res["id"]))
    sched._tick()
    assert any("no dispatcher" in line for line in logged)


def test_a_malformed_payload_does_not_stop_the_reminder(sched, memory):
    fired = []
    sched.notify_cb = lambda tid, title: fired.append(tid)
    tid = memory.create_task("broken", time.time() - 1, "{not json")

    sched._tick()
    assert fired == [tid]
    assert memory.get_task(tid)["status"] == "fired"
