"""The context sensor has one hard requirement beyond correctness: it must be
cheap enough to run on every submitted input without being felt."""

import subprocess
import time

import pytest

from core.context import Context, ContextSensor, clear_git_cache


@pytest.fixture(autouse=True)
def fresh_git_cache():
    clear_git_cache()
    yield
    clear_git_cache()


def test_snapshot_reports_the_working_directory(project):
    ctx = ContextSensor().snapshot()
    assert ctx.cwd == str(project)


def test_snapshot_is_under_five_milliseconds_on_a_warm_repo(project):
    """The budget from the brief. Measured after one warm-up call, because the
    first call is what populates the git cache."""
    sensor = ContextSensor()
    sensor.snapshot()  # warm the git cache

    runs = []
    for _ in range(20):
        start = time.perf_counter()
        sensor.snapshot()
        runs.append((time.perf_counter() - start) * 1000)

    runs.sort()
    median = runs[len(runs) // 2]
    assert median < 5.0, f"median snapshot took {median:.2f} ms"


def test_git_state_is_cached_so_it_is_not_shelled_out_per_keystroke(project, monkeypatch):
    calls = []
    real_run = subprocess.run

    def counting_run(*args, **kwargs):
        calls.append(args[0])
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", counting_run)
    sensor = ContextSensor()
    for _ in range(10):
        sensor.snapshot()

    git_calls = [c for c in calls if c and c[0] == "git"]
    assert len(git_calls) <= 2, f"git shelled out {len(git_calls)} times for 10 snapshots"


def test_a_directory_that_is_not_a_repo_reports_no_branch(project):
    ctx = ContextSensor().snapshot()
    assert ctx.git_branch is None
    assert ctx.git_dirty is False


def test_branch_and_dirtiness_are_detected_in_a_real_repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@e", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@e"}
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    def git(*args):
        return subprocess.run(["git"] + list(args), cwd=repo, capture_output=True, text=True)

    if git("init", "-b", "trunk").returncode != 0:
        pytest.skip("git unavailable")
    (repo / "a.txt").write_text("x", encoding="utf-8")
    git("add", "a.txt")
    git("commit", "-m", "first")

    ctx = ContextSensor().snapshot()
    assert ctx.git_branch == "trunk"
    assert ctx.git_dirty is False

    (repo / "b.txt").write_text("y", encoding="utf-8")
    clear_git_cache()
    assert ContextSensor().snapshot().git_dirty is True


def test_recent_commands_are_bounded(project):
    sensor = ContextSensor(keep_commands=3)
    for i in range(10):
        sensor.note_submission(f"cmd {i}")
    ctx = sensor.snapshot()
    assert len(ctx.recent_commands) == 3
    assert ctx.recent_commands[-1]["text"] == "cmd 9"


def test_the_last_exit_code_travels_with_the_context(project):
    sensor = ContextSensor()
    sensor.note_submission("pytest", exit_code=1)
    assert sensor.snapshot().last_exit_code == 1
    sensor.note_exit_code(0)
    assert sensor.snapshot().last_exit_code == 0


def test_idle_gap_is_zero_on_the_first_input_and_positive_afterwards(project):
    sensor = ContextSensor()
    assert sensor.snapshot().idle_gap_s == 0.0
    sensor.note_submission("git status")
    time.sleep(0.02)
    assert sensor.snapshot().idle_gap_s > 0


def test_recent_files_come_from_the_journal(project, wired):
    acts, journal, _mem = wired
    acts.call("write_file", path="notes.txt", text="x")

    ctx = ContextSensor(journal=journal).snapshot()
    assert str(project / "notes.txt") in ctx.recent_files


def test_recent_files_respects_the_time_window(project, wired):
    acts, journal, _mem = wired
    acts.call("write_file", path="old.txt", text="x")
    sensor = ContextSensor(journal=journal, recent_window_s=-1)
    assert sensor.snapshot().recent_files == []


def test_a_broken_journal_does_not_break_the_snapshot(project):
    class Exploding:
        def touched_files(self, **kw):
            raise RuntimeError("db is gone")

    ctx = ContextSensor(journal=Exploding()).snapshot()
    assert ctx.recent_files == []
    assert ctx.cwd  # the rest of the snapshot survived


def test_context_round_trips_through_a_dict(project):
    original = ContextSensor().snapshot()
    original.recent_commands = [{"text": "ls", "exit": 0}]
    restored = Context.from_dict(original.to_dict())
    assert restored.cwd == original.cwd
    assert restored.recent_commands == original.recent_commands
    assert restored.hour_of_day == original.hour_of_day


def test_from_dict_tolerates_a_row_written_by_an_older_version():
    """Episodes persist across upgrades; a snapshot missing newer fields must
    still load rather than taking the log down with it."""
    ctx = Context.from_dict({"ts": 1.0, "cwd": "/x"})
    assert ctx.cwd == "/x"
    assert ctx.recent_commands == []
    assert ctx.git_branch is None
