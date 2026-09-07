"""Reminder delivery races and snooze behavior, without real OS actions."""

import json
import time

import pytest

from core import actions as actions_mod
from core.memory import Memory
from core.scheduler import Scheduler


def stopped_scheduler(memory, notify=lambda *args: None):
    scheduler = Scheduler(memory.db_path, "UTC", 3600, notify, lambda *args: None)
    scheduler.stop()
    scheduler.set_memory(memory)
    return scheduler


def test_snoozing_fired_reminder_reactivates_it_from_now(project, wired, monkeypatch):
    acts, _, memory = wired
    monkeypatch.setattr(time, "time", lambda: 10000.0)
    task_id = memory.create_task("past reminder", 1000.0)
    memory.set_task_status(task_id, "fired")
    assert acts.call("snooze_task", task_id=task_id, delta="15m")["ok"]
    task = memory.get_task(task_id)
    assert task["due"] == 10900.0
    assert task["status"] == "pending"
    assert actions_mod.undo_last()["ok"]
    task = memory.get_task(task_id)
    assert (task["due"], task["status"]) == (1000.0, "fired")


def test_missing_reminder_does_not_report_snooze_success(project, wired):
    acts, _, _ = wired
    assert "error" in acts.call("snooze_task", task_id=999, delta="15m")


def test_missed_repeats_fire_once_and_keep_original_cadence(memory):
    now = time.time()
    original_due = now - 5 * 86400 - 30
    task_id = memory.create_task("daily", original_due, json.dumps({"_repeat": "daily"}))
    fired = []
    scheduler = stopped_scheduler(memory, lambda tid, title: fired.append(tid))
    scheduler._tick()
    scheduler._tick()
    assert fired == [task_id]
    assert memory.get_task(task_id)["due"] == pytest.approx(original_due + 6 * 86400)


def test_two_database_connections_cannot_deliver_same_occurrence(memory):
    second = Memory(memory.db_path)
    task_id = memory.create_task("once", time.time() - 1)
    first_snapshot = memory.due_tasks(time.time())[0]
    second_snapshot = second.due_tasks(time.time())[0]
    fired = []
    one = stopped_scheduler(memory, lambda tid, title: fired.append(tid))
    two = stopped_scheduler(second, lambda tid, title: fired.append(tid))
    try:
        # Both instances have already fetched the same pending occurrence.
        one._fire(first_snapshot)
        two._fire(second_snapshot)
        assert fired == [task_id]
    finally:
        second.close()


def test_callback_can_complete_repeat_without_scheduler_overwriting_it(memory):
    task_id = memory.create_task("daily", time.time() - 1, '{"_repeat":"daily"}')
    scheduler = stopped_scheduler(memory, lambda tid, title: memory.complete_task(tid))
    scheduler._tick()
    assert memory.get_task(task_id)["status"] == "done"


@pytest.mark.parametrize("payload", ["[]", "null", '"text"', "42"])
def test_valid_nonobject_json_does_not_poison_reminder_queue(memory, payload):
    broken = memory.create_task("bad payload", time.time() - 1, payload)
    healthy = memory.create_task("healthy", time.time() - 1)
    fired = []
    scheduler = stopped_scheduler(memory, lambda tid, title: fired.append(tid))
    assert scheduler.tick_safely()
    assert fired == [broken, healthy]
    assert memory.get_task(broken)["status"] == "fired"


def test_stopping_scheduler_wakes_sleeping_worker(memory):
    scheduler = stopped_scheduler(memory)
    scheduler._thread.join(timeout=0.5)
    assert not scheduler._thread.is_alive()


def test_snoozed_occurrence_cannot_be_delivered_from_stale_snapshot(memory):
    task_id = memory.create_task("snoozed", time.time() - 1)
    snapshot = memory.due_tasks(time.time())[0]
    memory.snooze_task(task_id, 900)
    fired = []
    scheduler = stopped_scheduler(memory, lambda tid, title: fired.append(tid))
    scheduler._fire(snapshot)
    assert fired == []


def test_legacy_snooze_undo_keeps_original_delta_semantics(memory, monkeypatch):
    monkeypatch.setattr(actions_mod, "_memory", memory)
    task_id = memory.create_task("old journal entry", 1900.0)
    actions_mod._undo_snooze({"task_id": task_id, "delta": "15m"})
    assert memory.get_task(task_id)["due"] == 1000.0


def test_zero_snooze_is_rejected(project, wired):
    acts, _, memory = wired
    task_id = memory.create_task("keep", time.time() - 1)
    memory.set_task_status(task_id, "fired")
    assert "error" in acts.call("snooze_task", task_id=task_id, delta="0m")
    assert memory.get_task(task_id)["status"] == "fired"
