"""The episode log is the training data for everything after Phase 3, so what it
records — and what it records when the system was *wrong* — is the whole value."""

import sqlite3
import time

import pytest

from core.context import ContextSensor
from core.episodes import Episode, EpisodeLog
from core.memory import Memory


@pytest.fixture
def log(memory):
    return EpisodeLog(memory, enabled=True)


# ── Involuntary encoding ────────────────────────────────────────────────────


def test_every_submission_produces_exactly_one_row(log):
    for text in ("git status", "pytest", "ls"):
        log.record(text)
    assert log.count() == 3
    assert [e.action for e in log.all()] == ["git status", "pytest", "ls"]


def test_recording_happens_without_being_asked(log, project):
    """The contrast with /save: nothing here required the user to opt in."""
    ctx = ContextSensor().snapshot()
    log.record("git commit -m wip", ctx)
    stored = log.all()[0]
    assert stored.action == "git commit -m wip"
    assert stored.context is not None
    assert stored.context.cwd == str(project)


def test_the_disabled_flag_suppresses_writes(memory):
    log = EpisodeLog(memory, enabled=False)
    assert log.record("secret command") is None
    assert log.count() == 0


def test_forget_clears_the_log(log):
    for i in range(5):
        log.record(f"cmd {i}")
    assert log.forget() == 5
    assert log.count() == 0


def test_forget_before_keeps_recent_episodes(log):
    log.record("old")
    cutoff = time.time() + 0.001
    time.sleep(0.01)
    log.record("new")

    assert log.forget_before(cutoff) == 1
    assert [e.action for e in log.all()] == ["new"]


# ── The prediction-error signal ─────────────────────────────────────────────


def test_a_prediction_shown_and_ignored_records_a_zero(log):
    """This is the negative training signal. Without it the predictor only ever
    learns from the times it happened to be right."""
    log.record("pytest", predicted="git push", predicted_conf=0.8, accepted_prediction=0)
    e = log.all()[0]
    assert e.accepted_prediction == 0
    assert e.was_shown_and_ignored
    assert e.predicted == "git push"
    assert e.predicted_conf == pytest.approx(0.8)


def test_a_prediction_that_was_taken_records_a_one(log):
    log.record("git push", predicted="git push", predicted_conf=0.9, accepted_prediction=1)
    e = log.all()[0]
    assert e.accepted_prediction == 1
    assert not e.was_shown_and_ignored


def test_a_prediction_never_shown_records_null(log):
    """A hint the user never saw carries no signal about whether they wanted it,
    and must not be counted as a rejection."""
    log.record("ls", predicted="ls -la", predicted_conf=0.4, accepted_prediction=None)
    assert log.all()[0].accepted_prediction is None
    assert log.shown_predictions() == []


def test_shown_predictions_separates_signal_from_silence(log):
    log.record("a", predicted="a", accepted_prediction=1)
    log.record("b", predicted="c", accepted_prediction=0)
    log.record("d", predicted="e", accepted_prediction=None)
    log.record("f")

    shown = log.shown_predictions()
    assert [e.action for e in shown] == ["a", "b"]


def test_hesitation_is_recorded(log):
    log.record("rm -rf build", hesitation_ms=4200)
    assert log.all()[0].hesitation_ms == 4200


def test_outcome_can_be_filled_in_after_the_fact(log):
    eid = log.record("pytest")
    log.set_outcome(eid, "error")
    assert log.all()[0].outcome == "error"


def test_capability_can_be_attached_after_resolution(log):
    eid = log.record("/read config.yaml")
    log.set_capability(eid, "read_file")
    assert log.all()[0].capability == "read_file"


def test_setting_a_field_on_a_none_id_is_a_no_op(log):
    """record() returns None when logging is off; callers should not have to
    branch on that before attaching an outcome."""
    log.set_outcome(None, "ok")
    log.set_capability(None, "read_file")


# ── Migration ───────────────────────────────────────────────────────────────


def test_migration_runs_against_a_database_that_predates_episodes(tmp_path):
    """The upgrade path for a user who has real notes in data/intuition.db."""
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE mem (id INTEGER PRIMARY KEY, ts REAL, role TEXT, text TEXT, tags TEXT)")
    conn.execute("CREATE TABLE tasks (id INTEGER PRIMARY KEY, ts REAL, title TEXT, due_ts REAL, status TEXT, payload TEXT)")
    conn.execute("INSERT INTO mem (ts, role, text, tags) VALUES (1.0, 'note', 'do not lose me', 'note')")
    conn.commit()
    conn.close()

    mem = Memory(str(db))
    log = EpisodeLog(mem)
    log.record("first episode")

    assert log.count() == 1
    assert mem.search("do not lose me"), "the pre-existing note must survive the migration"


def test_migration_adds_columns_to_a_partial_episodes_table(tmp_path):
    """An episodes table from an intermediate version, missing later columns."""
    db = tmp_path / "partial.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE episodes (id INTEGER PRIMARY KEY, ts REAL, context_json TEXT, action TEXT)")
    conn.execute("INSERT INTO episodes (ts, context_json, action) VALUES (1.0, NULL, 'old row')")
    conn.commit()
    conn.close()

    log = EpisodeLog(Memory(str(db)))
    log.record("new row", predicted="x", accepted_prediction=0, hesitation_ms=10)

    actions_seen = [e.action for e in log.all()]
    assert actions_seen == ["old row", "new row"]
    assert log.all()[1].accepted_prediction == 0


def test_migration_is_idempotent(memory):
    EpisodeLog(memory)
    EpisodeLog(memory)
    log = EpisodeLog(memory)
    log.record("still works")
    assert log.count() == 1


# ── Context binding ─────────────────────────────────────────────────────────


def test_context_survives_the_round_trip_through_sqlite(log, project):
    sensor = ContextSensor()
    sensor.note_submission("git commit", exit_code=0)
    log.record("pytest", sensor.snapshot())

    ctx = log.all()[0].context
    assert ctx.cwd == str(project)
    assert ctx.recent_commands[-1]["text"] == "git commit"
    assert ctx.last_exit_code == 0


def test_a_corrupt_context_blob_does_not_break_reading(log, memory):
    memory.execute("INSERT INTO episodes (ts, context_json, action) VALUES (?,?,?)",
                   (time.time(), "{not json", "still readable"))
    episodes = log.all()
    assert episodes[0].action == "still readable"
    assert episodes[0].context is None


# ── The prediction window ───────────────────────────────────────────────────


def test_window_records_acceptance_when_the_user_submits_what_was_predicted():
    from core.episodes import PredictionWindow

    w = PredictionWindow()
    w.note_keystroke("git pu")
    w.note_shown("git push", 0.82)
    assert w.take("git push")["accepted_prediction"] == 1


def test_window_records_rejection_when_a_hint_was_shown_and_ignored():
    from core.episodes import PredictionWindow

    w = PredictionWindow()
    w.note_keystroke("git pu")
    w.note_shown("git push", 0.82)
    out = w.take("git pull")
    assert out["accepted_prediction"] == 0
    assert out["predicted"] == "git push"


def test_window_records_null_when_nothing_was_shown():
    from core.episodes import PredictionWindow

    w = PredictionWindow()
    w.note_keystroke("ls")
    assert w.take("ls")["accepted_prediction"] is None


def test_window_measures_hesitation_from_the_last_keystroke():
    from core.episodes import PredictionWindow

    w = PredictionWindow()
    w.note_keystroke("rm -rf build")
    time.sleep(0.03)
    assert w.take("rm -rf build")["hesitation_ms"] >= 25


def test_window_is_consumed_by_take_so_one_submission_counts_once():
    from core.episodes import PredictionWindow

    w = PredictionWindow()
    w.note_keystroke("ls")
    w.note_shown("ls", 0.9)
    assert w.take("ls")["accepted_prediction"] == 1
    assert w.take("ls")["accepted_prediction"] is None
