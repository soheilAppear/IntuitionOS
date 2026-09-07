"""Synthetic episode logs with patterns planted in them.

The harness has to run on a fresh clone, and a fresh clone has no history. It
also has to be able to say whether the predictor learned *the thing that was
there*, which means the ground truth has to be known — so the generator plants
specific habits and the metrics are read against them.

Three properties matter, and all three are things a naive generator gets wrong:

  * Noise. A log where every pattern holds every time makes any predictor look
    perfect. Real habits are interrupted.
  * Context dependence. If the same prefix always has the same answer regardless
    of situation, a plain frequency table scores as well as anything and the
    context sensor is unfalsifiable.
  * Drift. People change what they are working on. A log with no drift cannot
    distinguish a model that weights recency from one that does not.
"""

from __future__ import annotations

import random
import time

from core.context import Context
from core.episodes import Episode

# Habits, as (previous command, what usually follows).
SEQUENCES = [
    ("git add -A", "git commit -m wip"),
    ("git commit -m wip", "pytest"),
    ("pytest", "git push"),
    ("npm install", "npm run dev"),
    ("cd ui", "npm run dev"),
]

# Situational routines, as (directory, hour bucket, action).
ROUTINES = [
    ("/home/u/intuition", 1, "git pull"),      # morning
    ("/home/u/intuition", 3, "git status"),    # evening
    ("/home/u/notes", 1, "ls"),
]

# Filler that follows nothing in particular. It must include the commands that
# *start* the chains above, or the sequential habits can never begin and the log
# contains none of the patterns it claims to plant.
FILLER = ["ls", "tree", "clear", "git status", "git diff", "cat README.md",
          "/tasks", "/memory", "htop", "cd ..",
          "git add -A", "npm install", "cd ui"]

# What a user types before submitting, so prefix matching has something to chew on.
def _prefix(action: str, rng: random.Random) -> str:
    if not action:
        return ""
    cut = rng.randint(1, max(1, min(len(action), 6)))
    return action[:cut]


def generate(n: int = 600, seed: int = 7, noise: float = 0.25,
             drift: bool = True, start_ts: float = None) -> list:
    """Build a log of `n` episodes with known habits planted in it.

    `noise` is the probability that a habit is *not* followed on a given
    occasion, which is what keeps the ceiling below 100% and makes the metrics
    mean something.
    """
    rng = random.Random(seed)
    now = start_ts if start_ts is not None else time.time()
    # One episode every few minutes, spread over roughly the last three weeks.
    span_s = 21 * 24 * 3600
    step = span_s / max(1, n)

    episodes: list = []
    previous = ""
    branch = "main"

    for i in range(n):
        ts = now - span_s + i * step
        hour_bucket = (i // 7) % 4
        hour = hour_bucket * 6 + rng.randint(0, 5)

        # Drift: the second half of the log lives on a different branch and in a
        # different directory, so a recency-weighted model can beat one that
        # treats a month-old habit as current.
        if drift and i > n // 2:
            branch = "release"
            cwd = "/home/u/notes"
        else:
            cwd = "/home/u/intuition"

        action = None

        # A sequential habit fires if the previous command triggers one.
        for trigger, follows in SEQUENCES:
            if previous == trigger and rng.random() > noise:
                action = follows
                break

        # Otherwise a situational routine may.
        if action is None:
            for routine_cwd, routine_hour, routine_action in ROUTINES:
                if cwd == routine_cwd and hour_bucket == routine_hour and rng.random() > noise + 0.3:
                    action = routine_action
                    break

        if action is None:
            action = rng.choice(FILLER)

        ctx = Context(
            ts=ts,
            cwd=cwd,
            recent_commands=([{"text": previous, "exit": 0}] if previous else []),
            recent_files=[],
            git_branch=branch,
            git_dirty=bool(i % 3),
            last_exit_code=0 if rng.random() > 0.1 else 1,
            idle_gap_s=rng.choice([1.0, 5.0, 30.0, 300.0]),
            hour_of_day=hour,
            day_of_week=(i // 24) % 7,
            session_age_s=float(i % 120) * 60,
        )

        episodes.append(Episode(
            id=i + 1, ts=ts, context=ctx,
            keystroke_prefix=_prefix(action, rng),
            action=action,
            hesitation_ms=rng.randint(80, 3000),
        ))
        previous = action

    return episodes


def split(episodes: list, train_fraction: float = 0.7) -> tuple:
    """Chronological split.

    Deliberately not shuffled: this is a time series, and a random split lets the
    model learn from the user's future, which flatters every metric.
    """
    cut = int(len(episodes) * train_fraction)
    return episodes[:cut], episodes[cut:]
