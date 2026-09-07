"""The predictor: what the user is about to do, and how sure we are.

This replaces the function that was supposed to embody the project's thesis and
was in fact a four-branch if-chain returning the literals 0.9, 0.9, 0.85 and 0.65.
Those numbers had never been compared to anything. They could not be wrong,
because nothing checked them, which is another way of saying they meant nothing.

Two stages, so there is always a working system:

  Stage 1 — frequency and recency. For each (context bucket, prefix) pair, count
  what the user actually did next, weighted by an exponential recency decay. No
  machine learning, fully explainable, and it beats the if-chain within a few
  dozen episodes. Explainability matters here: a HUD that shows a hunch has to be
  able to say why, and "you have done this 7 times in this directory this week"
  is a reason a person can check.

  Stage 2 — feature scoring. A small multinomial logistic model over hand-built
  features from the Context plus the prefix. Pure NumPy, trained on one user's
  command history, which is thousands of rows at most. A deep learning framework
  for this dataset would be a costume, not an improvement.

Cold start is the failure mode that kills systems like this, so the prefix
heuristics survive as an explicit fallback below `min_episodes`. A system that
predicts nothing until it has data is worse than the if-chain it replaced.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Optional

# One week. Fast enough to follow a change in what you are working on, slow
# enough that Monday still remembers Friday.
DEFAULT_HALF_LIFE_S = 7 * 24 * 3600.0

# Below this many episodes the learned scores are noise, and the heuristics win.
DEFAULT_MIN_EPISODES = 50


@dataclass
class Prediction:
    action: str
    confidence: float
    source: str = "learned"   # learned | heuristic | rule
    why: str = ""             # human-readable, shown by /why and the HUD
    intent: Optional[str] = None
    capability: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "action": self.action, "confidence": self.confidence, "source": self.source,
            "why": self.why, "intent": self.intent, "capability": self.capability,
        }


# ── The heuristics that were the whole predictor ─────────────────────────────


def heuristic_predictions(prefix: str) -> list:
    """Exactly the behaviour of the old inline predict(), preserved.

    These stay reachable so that a fresh install is no worse than the version it
    replaced, and so a test can assert that with an empty log nothing changed.
    """
    t = (prefix or "").strip()
    if not t:
        return []
    if t == "tree" or t.startswith("tree "):
        return [Prediction(t, 0.9, "heuristic", "'tree' is a known command", intent="tree",
                           capability="list_tree")]
    if t == "ls":
        return [Prediction(t, 0.9, "heuristic", "'ls' is a known command", intent="ls",
                           capability="list_dir")]
    if t.startswith("read file "):
        return [Prediction(t, 0.85, "heuristic", "'read file <path>' is a known form",
                           intent="read_file", capability="read_file")]
    return [Prediction(t, 0.65, "heuristic", "no learned pattern yet", intent="plan")]


# ── Context bucketing ────────────────────────────────────────────────────────


def context_bucket(ctx) -> tuple:
    """Collapse a Context into a coarse key the counts can be grouped under.

    Coarse on purpose. Bucketing on the exact minute or the exact idle gap would
    give every episode its own bucket and nothing would ever accumulate support.
    """
    if ctx is None:
        return ("", "", 0, 0)
    branch = getattr(ctx, "git_branch", None) or ""
    cwd = getattr(ctx, "cwd", "") or ""
    hour = _hour_bucket(getattr(ctx, "hour_of_day", 0) or 0)
    dirty = 1 if getattr(ctx, "git_dirty", False) else 0
    return (cwd, branch, hour, dirty)


def _hour_bucket(hour: int) -> int:
    # night / morning / afternoon / evening
    return int(hour) // 6


def last_command(ctx) -> str:
    """What the user did immediately before. The single strongest cue there is:
    'pytest after git commit' is a sequence, not a time of day."""
    if ctx is None:
        return ""
    recent = getattr(ctx, "recent_commands", None) or []
    if not recent:
        return ""
    entry = recent[-1]
    return (entry.get("text") if isinstance(entry, dict) else str(entry)) or ""


# ── Stage 1: frequency and recency ───────────────────────────────────────────


@dataclass
class _Counter:
    """Recency-weighted counts, updated in place rather than recomputed.

    The weight of an observation decays continuously, so instead of storing
    timestamps and re-summing on every query, the accumulated weight is aged
    forward to the moment it is read.
    """
    weight: float = 0.0
    last_ts: float = 0.0
    hits: int = 0

    def add(self, ts: float, half_life: float, amount: float = 1.0):
        self.weight = self.decayed(ts, half_life) + amount
        self.last_ts = ts
        self.hits += 1

    def decayed(self, now: float, half_life: float) -> float:
        if self.weight == 0.0:
            return 0.0
        age = max(0.0, now - self.last_ts)
        return self.weight * math.pow(0.5, age / half_life)


class FrequencyModel:
    """Counts of what followed what, under three cues of decreasing specificity.

    Backing off across cues is what lets it say something useful early: a
    (bucket, prefix) pair needs several observations before it means anything, but
    "what usually follows git commit" accumulates much faster.
    """

    def __init__(self, half_life_s: float = DEFAULT_HALF_LIFE_S):
        self.half_life_s = half_life_s
        self.by_prefix: dict = {}    # (bucket, prefix) -> {action: _Counter}
        self.by_previous: dict = {}  # (bucket, previous_action) -> {action: _Counter}
        self.by_bucket: dict = {}    # bucket -> {action: _Counter}
        self.total = 0

    def observe(self, action: str, ctx, prefix: str = "", ts: Optional[float] = None):
        if not action:
            return
        ts = time.time() if ts is None else ts
        bucket = context_bucket(ctx)
        prev = last_command(ctx)

        self._bump(self.by_bucket, bucket, action, ts)
        if prev:
            self._bump(self.by_previous, (bucket, prev), action, ts)
        for p in _prefixes(prefix or action):
            self._bump(self.by_prefix, (bucket, p), action, ts)
        self.total += 1

    def _bump(self, table, key, action, ts):
        slot = table.setdefault(key, {})
        counter = slot.get(action)
        if counter is None:
            counter = slot[action] = _Counter()
        counter.add(ts, self.half_life_s)

    def score(self, prefix: str, ctx, now: Optional[float] = None) -> list:
        """Ranked (action, weight, why) under the most specific cue that fires."""
        now = time.time() if now is None else now
        bucket = context_bucket(ctx)
        prefix = (prefix or "").strip()

        if prefix:
            slot = self.by_prefix.get((bucket, prefix))
            if slot:
                return self._rank(slot, now, f"you have typed this here before")

        prev = last_command(ctx)
        if prev:
            slot = self.by_previous.get((bucket, prev))
            if slot:
                return self._rank(slot, now, f"usually follows {prev!r} here")

        slot = self.by_bucket.get(bucket)
        if slot:
            return self._rank(slot, now, "common in this directory at this hour")
        return []

    def _rank(self, slot: dict, now: float, why: str) -> list:
        scored = [(a, c.decayed(now, self.half_life_s), c.hits) for a, c in slot.items()]
        scored = [(a, w, h) for a, w, h in scored if w > 1e-6]
        scored.sort(key=lambda x: -x[1])
        return [(a, w, f"{why} ({h}x)") for a, w, h in scored]

    # ── Persistence ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "half_life_s": self.half_life_s,
            "total": self.total,
            "by_prefix": _dump(self.by_prefix),
            "by_previous": _dump(self.by_previous),
            "by_bucket": _dump(self.by_bucket),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FrequencyModel":
        m = cls(half_life_s=float(d.get("half_life_s", DEFAULT_HALF_LIFE_S)))
        m.total = int(d.get("total", 0))
        m.by_prefix = _load(d.get("by_prefix", {}))
        m.by_previous = _load(d.get("by_previous", {}))
        m.by_bucket = _load(d.get("by_bucket", {}))
        return m


def _prefixes(text: str) -> list:
    """Prefixes worth indexing: the whole thing, and each growing token prefix.

    Character-level prefixes would explode the table for no gain; a user typing
    `git com` is matched by the `git` entry.
    """
    text = (text or "").strip()
    if not text:
        return []
    out = [text]
    tokens = text.split()
    for i in range(1, len(tokens)):
        out.append(" ".join(tokens[:i]))
    return out


def _dump(table: dict) -> list:
    return [
        [list(key) if isinstance(key, tuple) else key,
         {a: [c.weight, c.last_ts, c.hits] for a, c in slot.items()}]
        for key, slot in table.items()
    ]


def _load(rows) -> dict:
    out = {}
    for key, slot in rows or []:
        k = tuple(_tuplify(x) for x in key) if isinstance(key, list) else key
        out[k] = {a: _Counter(weight=v[0], last_ts=v[1], hits=int(v[2])) for a, v in slot.items()}
    return out


def _tuplify(x):
    return tuple(x) if isinstance(x, list) else x


# ── Stage 2: feature-based scoring ───────────────────────────────────────────

_TOKEN = re.compile(r"[a-z0-9_./-]+")


def features(prefix: str, ctx) -> dict:
    """Hand-built features. Named, sparse, and inspectable — which is what lets a
    weight be read back as an explanation rather than a number."""
    prefix = (prefix or "").strip()
    lower = prefix.lower()
    f: dict = {"bias": 1.0}

    for n in (2, 3):
        for i in range(max(0, len(lower) - n + 1)):
            f[f"ng{n}:{lower[i:i + n]}"] = 1.0
    for tok in _TOKEN.findall(lower)[:4]:
        f[f"tok:{tok}"] = 1.0

    f[f"len:{min(len(prefix) // 4, 8)}"] = 1.0
    if prefix.startswith("/"):
        f["slash"] = 1.0

    if ctx is not None:
        f[f"hour:{_hour_bucket(getattr(ctx, 'hour_of_day', 0) or 0)}"] = 1.0
        f[f"dow:{getattr(ctx, 'day_of_week', 0) or 0}"] = 1.0
        exit_code = getattr(ctx, "last_exit_code", None)
        if exit_code is not None:
            f["exit:ok" if exit_code == 0 else "exit:fail"] = 1.0
        if getattr(ctx, "git_dirty", False):
            f["git:dirty"] = 1.0
        branch = getattr(ctx, "git_branch", None)
        if branch:
            f[f"branch:{branch}"] = 1.0
        prev = last_command(ctx)
        if prev:
            f[f"prev:{prev[:40]}"] = 1.0
        gap = getattr(ctx, "idle_gap_s", 0.0) or 0.0
        f[f"gap:{_gap_bucket(gap)}"] = 1.0
        cwd = getattr(ctx, "cwd", "") or ""
        if cwd:
            f[f"cwd:{cwd}"] = 1.0
    return f


def _gap_bucket(gap: float) -> int:
    for i, edge in enumerate((2, 10, 60, 300, 1800)):
        if gap < edge:
            return i
    return 5


class LogisticScorer:
    """A small multinomial logistic model, trained online with SGD.

    Sparse dict weights rather than NumPy matrices: the feature space is
    open-ended (every new branch name is a new feature) and the vocabulary of
    actions grows as the user works, so a dense matrix would have to be
    reallocated constantly for no speed gain at this scale.
    """

    def __init__(self, lr: float = 0.25, l2: float = 1e-6):
        self.lr = lr
        self.l2 = l2
        self.weights: dict = {}   # action -> {feature: weight}
        self.actions: list = []
        self.updates = 0

    def _scores(self, feats: dict) -> dict:
        out = {}
        for action in self.actions:
            w = self.weights.get(action, {})
            out[action] = sum(w.get(f, 0.0) * v for f, v in feats.items())
        return out

    def probabilities(self, feats: dict) -> dict:
        scores = self._scores(feats)
        if not scores:
            return {}
        top = max(scores.values())
        exps = {a: math.exp(s - top) for a, s in scores.items()}
        total = sum(exps.values()) or 1.0
        return {a: e / total for a, e in exps.items()}

    def observe(self, action: str, feats: dict):
        if action not in self.weights:
            self.weights[action] = {}
            self.actions.append(action)
        probs = self.probabilities(feats)
        for a in self.actions:
            target = 1.0 if a == action else 0.0
            error = target - probs.get(a, 0.0)
            w = self.weights[a]
            for f, v in feats.items():
                w[f] = w.get(f, 0.0) * (1.0 - self.l2) + self.lr * error * v
        self.updates += 1

    def to_dict(self) -> dict:
        return {"lr": self.lr, "l2": self.l2, "updates": self.updates,
                "weights": self.weights, "actions": self.actions}

    @classmethod
    def from_dict(cls, d: dict) -> "LogisticScorer":
        m = cls(lr=float(d.get("lr", 0.25)), l2=float(d.get("l2", 1e-6)))
        m.weights = {a: dict(w) for a, w in (d.get("weights") or {}).items()}
        m.actions = list(d.get("actions") or [])
        m.updates = int(d.get("updates", 0))
        return m


# ── The predictor ────────────────────────────────────────────────────────────


class Predictor:
    """Stable interface: predict(prefix, ctx) -> ranked candidates; update(episode)."""

    def __init__(self, store=None, half_life_s: float = DEFAULT_HALF_LIFE_S,
                 min_episodes: int = DEFAULT_MIN_EPISODES, calibrator=None,
                 rules=None, blend: float = 0.5):
        self.frequency = FrequencyModel(half_life_s=half_life_s)
        self.logistic = LogisticScorer()
        self.min_episodes = min_episodes
        self.store = store
        self.calibrator = calibrator
        self.rules = rules          # Phase 6 promotes patterns into these
        self.blend = blend          # how much of the score comes from the logistic stage
        self.seen = 0
        if store is not None:
            self.load()

    # ── Query ────────────────────────────────────────────────────────────

    def predict(self, prefix: str, ctx=None, k: int = 3) -> list:
        """Ranked candidates with calibrated probabilities."""
        prefix = prefix or ""
        if not prefix.strip():
            return []

        if self.seen < self.min_episodes:
            # Cold start: a system that predicts nothing is worse than the
            # if-chain it replaced, so the if-chain is still here.
            return self._calibrate(heuristic_predictions(prefix))[:k]

        ranked = self._rules_first(prefix, ctx)
        if ranked:
            return self._calibrate(ranked)[:k]

        freq = self.frequency.score(prefix, ctx)
        if not freq:
            return self._calibrate(heuristic_predictions(prefix))[:k]

        total = sum(w for _a, w, _why in freq) or 1.0
        probs = self.logistic.probabilities(features(prefix, ctx))

        merged = []
        for action, weight, why in freq:
            freq_p = weight / total
            model_p = probs.get(action, 0.0)
            score = (1.0 - self.blend) * freq_p + self.blend * model_p
            merged.append(Prediction(action=action, confidence=score, source="learned", why=why))

        merged.sort(key=lambda p: -p.confidence)
        return self._calibrate(merged)[:k]

    def _rules_first(self, prefix: str, ctx) -> list:
        """A rule promoted by consolidation short-circuits scoring: it was already
        judged to be a genuine habit, and it carries a human-readable reason."""
        if not self.rules:
            return []
        try:
            matches = self.rules.match(prefix, ctx)
        except Exception:
            return []
        return [
            Prediction(action=r["action"], confidence=float(r.get("hit_rate") or 0.5),
                       source="rule", why=r.get("description") or "a learned rule")
            for r in matches
        ]

    def _calibrate(self, predictions: list) -> list:
        if not self.calibrator:
            return predictions
        for p in predictions:
            try:
                p.confidence = self.calibrator.calibrate(p.confidence)
            except Exception:
                pass
        return predictions

    # ── Learning ─────────────────────────────────────────────────────────

    def update(self, episode) -> None:
        """Learn from one observed episode."""
        action = getattr(episode, "action", None) or ""
        if not action:
            return
        ctx = getattr(episode, "context", None)
        prefix = getattr(episode, "keystroke_prefix", "") or action
        ts = getattr(episode, "ts", None) or time.time()

        self.frequency.observe(action, ctx, prefix=prefix, ts=ts)
        self.logistic.observe(action, features(prefix, ctx))
        self.seen += 1

    def fit(self, episodes) -> "Predictor":
        """Replay a whole log. Used at startup and by the eval harness."""
        for e in episodes:
            self.update(e)
        return self

    # ── Persistence ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "seen": self.seen,
            "frequency": self.frequency.to_dict(),
            "logistic": self.logistic.to_dict(),
        }

    def load_dict(self, d: dict) -> None:
        if not d:
            return
        self.seen = int(d.get("seen", 0))
        if d.get("frequency"):
            self.frequency = FrequencyModel.from_dict(d["frequency"])
        if d.get("logistic"):
            self.logistic = LogisticScorer.from_dict(d["logistic"])

    def save(self) -> None:
        if self.store is not None:
            self.store.save(self.to_dict())

    def load(self) -> None:
        if self.store is not None:
            self.load_dict(self.store.load())


class PredictorStore:
    """Predictor state, kept in the same SQLite file as everything else.

    A separate pickle next to the database would be one more thing to keep in
    sync and one more thing to corrupt independently.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS predictor_state (
      id INTEGER PRIMARY KEY CHECK (id = 1),
      updated_ts REAL,
      state_json TEXT
    )
    """

    def __init__(self, memory):
        self.mem = memory
        self.mem.executescript(self.SCHEMA)

    def save(self, state: dict) -> None:
        import json
        self.mem.execute(
            "INSERT INTO predictor_state (id, updated_ts, state_json) VALUES (1,?,?)"
            " ON CONFLICT(id) DO UPDATE SET updated_ts=excluded.updated_ts,"
            " state_json=excluded.state_json",
            (time.time(), json.dumps(state, default=str)),
        )

    def load(self) -> dict:
        import json
        rows = self.mem.query("SELECT state_json FROM predictor_state WHERE id=1")
        if not rows or not rows[0][0]:
            return {}
        try:
            return json.loads(rows[0][0])
        except ValueError:
            # Corrupt state is not worth taking the process down for; relearning
            # from the episode log is cheap.
            return {}
