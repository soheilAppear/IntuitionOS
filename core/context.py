"""Context sensing: a cheap, portable snapshot of the situation an input arrived in.

The predictor in Phase 4 cannot learn "you run pytest after git commit" unless
something records *after what*. This module is that something. It answers, in
under five milliseconds, the question a person answers without noticing: where am
I, what did I just do, what time is it, and how long did I hesitate.

Three constraints shaped it:

  * Cheap. It runs on every submitted input, so a snapshot that shells out
    liberally would be felt. Git state is the only subprocess, it is cached, and
    it degrades to None rather than blocking.
  * Portable. Everything here works the same on Windows, macOS and Linux. Focused
    window and process-list sensing are deliberately absent — they are
    OS-specific and belong to a separate effort.
  * Local. Nothing leaves the machine.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

# Git state is asked for on every input but changes rarely, so it is cached per
# working directory for a short interval. Without this the snapshot would spend
# most of its budget waiting on two subprocesses.
_GIT_TTL_S = 3.0
_git_cache: dict[str, tuple[float, Optional[str], bool]] = {}


@dataclass
class Context:
    ts: float
    cwd: str
    recent_commands: list = field(default_factory=list)  # last N inputs with exit status
    recent_files: list = field(default_factory=list)     # files written recently, from the journal
    git_branch: Optional[str] = None
    git_dirty: bool = False
    last_exit_code: Optional[int] = None
    idle_gap_s: float = 0.0      # seconds since the previous input
    hour_of_day: int = 0
    day_of_week: int = 0
    session_age_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Context":
        known = {f: d.get(f) for f in cls.__dataclass_fields__}
        known["recent_commands"] = list(known.get("recent_commands") or [])
        known["recent_files"] = list(known.get("recent_files") or [])
        known["ts"] = float(known.get("ts") or 0.0)
        known["cwd"] = known.get("cwd") or ""
        return cls(**known)


class ContextSensor:
    """Produces a Context on demand, holding the small amount of session state
    that no single call site owns: when the session started, when the last input
    arrived, and how the last one turned out."""

    def __init__(self, journal=None, recent_window_s: float = 600.0, keep_commands: int = 5):
        self.journal = journal
        self.recent_window_s = recent_window_s
        self.keep_commands = keep_commands
        self.session_start = time.time()
        self._last_input_ts: Optional[float] = None
        self._recent: list = []  # [{"text":..., "exit":...}]
        self._last_exit_code: Optional[int] = None

    # ── Session bookkeeping ──────────────────────────────────────────────

    def note_submission(self, text: str, exit_code: Optional[int] = None) -> None:
        """Record that the user submitted something, and how it went."""
        self._recent.append({"text": text[:200], "exit": exit_code})
        del self._recent[:-self.keep_commands]
        self._last_exit_code = exit_code
        self._last_input_ts = time.time()

    def note_exit_code(self, exit_code: Optional[int]) -> None:
        """Attach a result to the submission already recorded."""
        self._last_exit_code = exit_code
        if self._recent:
            self._recent[-1]["exit"] = exit_code

    # ── The snapshot ─────────────────────────────────────────────────────

    def snapshot(self) -> Context:
        now = time.time()
        local = time.localtime(now)
        cwd = os.getcwd()
        branch, dirty = _git_state(cwd)
        return Context(
            ts=now,
            cwd=cwd,
            recent_commands=[dict(c) for c in self._recent],
            recent_files=self._recent_files(now),
            git_branch=branch,
            git_dirty=dirty,
            last_exit_code=self._last_exit_code,
            idle_gap_s=(now - self._last_input_ts) if self._last_input_ts else 0.0,
            hour_of_day=local.tm_hour,
            day_of_week=local.tm_wday,
            session_age_s=now - self.session_start,
        )

    def _recent_files(self, now: float) -> list:
        if not self.journal:
            return []
        try:
            return self.journal.touched_files(since_ts=now - self.recent_window_s, limit=10)
        except Exception:
            return []


def _git_state(cwd: str) -> tuple[Optional[str], bool]:
    """Branch name and dirtiness, cached, never blocking for long.

    A repository with a huge worktree can make `status` slow, so the timeout is
    short and a timeout reads as "unknown" rather than stalling every keystroke.
    """
    cached = _git_cache.get(cwd)
    now = time.monotonic()
    if cached and now - cached[0] < _GIT_TTL_S:
        return cached[1], cached[2]

    branch, dirty = None, False
    try:
        branch_out = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
        if branch_out is not None:
            branch = branch_out.strip() or None
            status = _git(["status", "--porcelain"], cwd)
            dirty = bool(status and status.strip())
    except Exception:
        branch, dirty = None, False

    _git_cache[cwd] = (now, branch, dirty)
    return branch, dirty


def _git(args: list, cwd: str) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=1.5,
            # Keep a console window from flashing on Windows every few seconds.
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
        )
    except Exception:
        return None
    return proc.stdout if proc.returncode == 0 else None


def clear_git_cache() -> None:
    """Used by tests, and by anything that knows the working directory changed."""
    _git_cache.clear()
