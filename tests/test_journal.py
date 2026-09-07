"""The journal is what makes `reversible` mean something in practice, and what
turns "sandboxed exec" into the honest claim: scoped exec with a record and an undo."""

import pytest

from core import actions as actions_mod
from core.capabilities import capabilities, set_safe_mode


# ── Undo ────────────────────────────────────────────────────────────────────


def test_undo_restores_a_file_overwritten_by_write_file(project, wired):
    acts, journal, _mem = wired
    target = project / "notes.txt"
    target.write_text("the original", encoding="utf-8")

    assert acts.call("write_file", path="notes.txt", text="clobbered").get("ok")
    assert target.read_text(encoding="utf-8") == "clobbered"

    result = actions_mod.undo_last()
    assert result.get("ok"), result
    assert target.read_text(encoding="utf-8") == "the original"


def test_undo_removes_a_file_that_did_not_exist_before(project, wired):
    acts, _journal, _mem = wired
    acts.call("write_file", path="fresh.txt", text="new")
    assert (project / "fresh.txt").exists()

    assert actions_mod.undo_last().get("ok")
    assert not (project / "fresh.txt").exists()


def test_undo_is_one_shot(project, wired):
    acts, _journal, _mem = wired
    (project / "a.txt").write_text("v1", encoding="utf-8")
    acts.call("write_file", path="a.txt", text="v2")

    assert actions_mod.undo_last().get("ok")
    # The entry is spent; a second /undo must not silently re-apply it.
    assert "error" in actions_mod.undo_last()


def test_undo_walks_back_through_history(project, wired):
    acts, _journal, _mem = wired
    (project / "a.txt").write_text("v1", encoding="utf-8")
    acts.call("write_file", path="a.txt", text="v2")
    acts.call("write_file", path="a.txt", text="v3")

    actions_mod.undo_last()
    assert (project / "a.txt").read_text(encoding="utf-8") == "v2"
    actions_mod.undo_last()
    assert (project / "a.txt").read_text(encoding="utf-8") == "v1"


def test_nothing_to_undo_is_an_error_not_a_crash(project, wired):
    assert "error" in actions_mod.undo_last()


def test_a_failed_action_leaves_no_undo_entry(project, wired):
    acts, journal, _mem = wired
    # A directory where the file should go makes the write fail.
    (project / "blocked").mkdir()
    res = acts.call("write_file", path="blocked", text="x")
    assert "error" in res
    assert journal.last_undoable() is None, "a failed write must not look undoable"


# ── What gets recorded ──────────────────────────────────────────────────────


def test_free_capabilities_are_not_journalled(project, wired):
    acts, journal, _mem = wired
    acts.call("list_dir", path=".")
    assert journal.recent() == [], "reads should not fill the journal with noise"


def test_a_write_is_journalled_with_its_actor(project, wired):
    acts, journal, _mem = wired
    acts.dispatch("write_file", {"path": "a.txt", "text": "x"}, actor="model", confidence=0.8)
    rows = journal.recent()
    assert len(rows) == 1
    assert rows[0]["capability"] == "write_file"
    assert rows[0]["actor"] == "model"
    assert rows[0]["decision"] == "allow"
    assert rows[0]["outcome"] == "ok"
    assert rows[0]["confidence"] == pytest.approx(0.8)


def test_denials_are_journalled_too(project, wired):
    """A model repeatedly reaching for something it is not allowed to have is
    exactly the signal Phase 6 wants to see, so denials are recorded."""
    acts, journal, _mem = wired
    res = acts.dispatch("write_file", {"path": str(project.parent / "out.txt"), "text": "x"},
                        actor="model", confidence=0.9)
    assert res.get("denied")
    rows = journal.recent()
    assert rows[0]["decision"] == "deny"
    assert rows[0]["outcome"] is None


def test_journal_records_the_resolved_path_not_the_relative_one(project, wired):
    acts, journal, _mem = wired
    acts.call("write_file", path="sub/a.txt", text="x")
    assert journal.recent()[0]["args"]["path"] == str(project / "sub" / "a.txt")


# ── The confirmation round trip ─────────────────────────────────────────────


def test_dispatch_parks_an_irreversible_action_instead_of_running_it(project, wired):
    acts, journal, mem = wired
    set_safe_mode(False)
    tid = mem.create_task("keep me", due_ts=0)

    res = acts.call("delete_task", task_id=tid)
    assert res.get("needs_confirmation")
    assert res["token"]
    assert mem.get_task(tid) is not None, "the task must still exist while awaiting confirmation"


def test_granting_a_confirmation_executes_it_once(project, wired):
    acts, journal, mem = wired
    set_safe_mode(False)
    tid = mem.create_task("delete me", due_ts=0)

    token = acts.call("delete_task", task_id=tid)["token"]
    assert acts.confirm(token, granted=True).get("ok")
    assert mem.get_task(tid) is None

    # The token is spent, so a replayed approval cannot fire a second time.
    assert "error" in acts.confirm(token, granted=True)


def test_declining_a_confirmation_records_it_and_does_nothing(project, wired):
    acts, journal, mem = wired
    set_safe_mode(False)
    tid = mem.create_task("keep me", due_ts=0)

    token = acts.call("delete_task", task_id=tid)["token"]
    res = acts.confirm(token, granted=False)
    assert res.get("cancelled")
    assert mem.get_task(tid) is not None
    assert journal.recent()[0]["decision"] == "confirm_denied"


def test_an_expired_confirmation_cannot_be_granted(project, wired, monkeypatch):
    acts, _journal, mem = wired
    set_safe_mode(False)
    from core.capabilities import pending_confirmations

    monkeypatch.setattr(pending_confirmations, "ttl_s", -1.0)
    tid = mem.create_task("keep me", due_ts=0)
    token = acts.call("delete_task", task_id=tid)["token"]
    assert "error" in acts.confirm(token, granted=True)
    assert mem.get_task(tid) is not None


# ── Task undo ───────────────────────────────────────────────────────────────


def test_completing_a_task_can_be_undone(project, wired):
    acts, _journal, mem = wired
    tid = mem.create_task("write the brief", due_ts=0)

    acts.call("complete_task", task_id=tid)
    assert mem.get_task(tid)["status"] == "done"

    assert actions_mod.undo_last().get("ok")
    assert mem.get_task(tid)["status"] == "pending"


def test_snoozing_a_task_can_be_undone(project, wired, monkeypatch):
    monkeypatch.setattr("core.memory.time.time", lambda: 500.0)
    acts, _journal, mem = wired
    tid = mem.create_task("standup", due_ts=1000.0)

    acts.call("snooze_task", task_id=tid, delta="15m")
    assert mem.get_task(tid)["due"] == pytest.approx(1900.0)

    assert actions_mod.undo_last().get("ok")
    assert mem.get_task(tid)["due"] == pytest.approx(1000.0)


# ── Files touched, for the Phase 3 context sensor ───────────────────────────


def test_touched_files_reports_recent_writes(project, wired):
    acts, journal, _mem = wired
    acts.call("write_file", path="a.txt", text="x")
    acts.call("read_file", path="b.txt")
    touched = journal.touched_files(since_ts=0)
    assert touched == [str(project / "a.txt")], "reads are free and stay out of the journal"
