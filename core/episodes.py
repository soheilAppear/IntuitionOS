"""The episode log: involuntary encoding of what actually happened.

This is the primitive the project was missing. `/save` and `/recall` are voluntary
— nothing is stored unless the user types it, and nothing is retrieved unless the
user asks. Human episodic memory does not work that way: it encodes without being
asked, binds what happened to the situation it happened in, and is retrieved by
cues rather than by queries.

An episode row is one submitted input, bound to the Context it arrived in, and —
crucially — to what the system had predicted just beforehand. That last column is
the whole point. A prediction that was shown and ignored records
accepted_prediction = 0, and that negative signal is what lets Phase 4 learn from
being wrong rather than only from being right.

Recording everything a person types needs an off switch and an honest disclosure,
so there is `enabled`, `/forget`, and a paragraph in the README.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

from .context import Context

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
  id INTEGER PRIMARY KEY,
  ts REAL,
  context_json TEXT,        -- serialized Context
  keystroke_prefix TEXT,    -- what was in the buffer when prediction fired
  predicted TEXT,           -- what the anticipator guessed, NULL if it did not
  predicted_conf REAL,
  action TEXT,              -- what the user actually submitted
  capability TEXT,          -- resolved capability name, if any
  outcome TEXT,             -- ok | error | undone
  hesitation_ms INTEGER,    -- time between last keystroke and Enter
  accepted_prediction INTEGER  -- 1 taken, 0 shown and ignored, NULL not shown
);
CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(ts);
CREATE INDEX IF NOT EXISTS idx_episodes_action ON episodes(action);
"""


@dataclass
class Episode:
    id: Optional[int] = None
    ts: float = 0.0
    context: Optional[Context] = None
    keystroke_prefix: str = ""
    predicted: Optional[str] = None
    predicted_conf: Optional[float] = None
    action: str = ""
    capability: Optional[str] = None
    outcome: Optional[str] = None
    hesitation_ms: Optional[int] = None
    accepted_prediction: Optional[int] = None

    @property
    def was_shown_and_ignored(self) -> bool:
        return self.accepted_prediction == 0


class EpisodeLog:
    def __init__(self, memory, enabled: bool = True):
        self.mem = memory
        self.enabled = enabled
        self._migrate()

    def _migrate(self):
        """Create the table, and add columns to one that predates them.

        Runs against a database that has only `mem` and `tasks`, which is what a
        user upgrading from an earlier version will have.
        """
        self.mem.executescript(SCHEMA)
        existing = set(self.mem.columns("episodes"))
        for column, decl in (
            ("keystroke_prefix", "TEXT"),
            ("predicted", "TEXT"),
            ("predicted_conf", "REAL"),
            ("capability", "TEXT"),
            ("outcome", "TEXT"),
            ("hesitation_ms", "INTEGER"),
            ("accepted_prediction", "INTEGER"),
        ):
            if column not in existing:
                self.mem.execute(f"ALTER TABLE episodes ADD COLUMN {column} {decl}")

    # ── Writing ──────────────────────────────────────────────────────────

    def record(
        self,
        action: str,
        context: Optional[Context] = None,
        *,
        keystroke_prefix: str = "",
        predicted: Optional[str] = None,
        predicted_conf: Optional[float] = None,
        capability: Optional[str] = None,
        outcome: Optional[str] = None,
        hesitation_ms: Optional[int] = None,
        accepted_prediction: Optional[int] = None,
    ) -> Optional[int]:
        """Encode one submitted input. Returns the row id, or None if disabled."""
        if not self.enabled:
            return None
        return self.mem.insert(
            "INSERT INTO episodes (ts, context_json, keystroke_prefix, predicted,"
            " predicted_conf, action, capability, outcome, hesitation_ms, accepted_prediction)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                time.time(),
                json.dumps(context.to_dict(), default=str) if context else None,
                keystroke_prefix,
                predicted,
                predicted_conf,
                action,
                capability,
                outcome,
                hesitation_ms,
                accepted_prediction,
            ),
        )

    def set_outcome(self, episode_id: int, outcome: str) -> None:
        if episode_id is None:
            return
        self.mem.execute("UPDATE episodes SET outcome=? WHERE id=?", (outcome, episode_id))

    def set_capability(self, episode_id: int, capability: str) -> None:
        if episode_id is None:
            return
        self.mem.execute("UPDATE episodes SET capability=? WHERE id=?", (capability, episode_id))

    # ── Reading ──────────────────────────────────────────────────────────

    _COLUMNS = ("id, ts, context_json, keystroke_prefix, predicted, predicted_conf,"
                " action, capability, outcome, hesitation_ms, accepted_prediction")

    def recent(self, limit: int = 200) -> list:
        rows = self.mem.query(
            f"SELECT {self._COLUMNS} FROM episodes ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [_row_to_episode(r) for r in rows]

    def all(self) -> list:
        rows = self.mem.query(f"SELECT {self._COLUMNS} FROM episodes ORDER BY id ASC")
        return [_row_to_episode(r) for r in rows]

    def since(self, ts: float) -> list:
        rows = self.mem.query(
            f"SELECT {self._COLUMNS} FROM episodes WHERE ts>=? ORDER BY id ASC", (ts,)
        )
        return [_row_to_episode(r) for r in rows]

    def count(self) -> int:
        return self.mem.query("SELECT COUNT(*) FROM episodes")[0][0]

    def shown_predictions(self) -> list:
        """Episodes where a hint was actually put in front of the user.

        These are the only ones that carry a calibration signal: a prediction the
        user never saw tells you nothing about whether they would have taken it.
        """
        rows = self.mem.query(
            f"SELECT {self._COLUMNS} FROM episodes WHERE accepted_prediction IS NOT NULL"
            " ORDER BY id ASC"
        )
        return [_row_to_episode(r) for r in rows]

    # ── Forgetting ───────────────────────────────────────────────────────

    def forget(self) -> int:
        """Erase episodes and their derived learning, including correction data.

        Notes, tasks and the action journal have their own lifetimes. Interfaces
        also reset live predictors; the correction store's generation invalidates
        other sessions' displayed candidates and cached preference reads.
        """
        n = self.count()
        self.mem.execute("DELETE FROM episodes")
        for table in ("predictor_state", "calibration_state", "rules"):
            if self.mem.has_table(table):
                self.mem.execute(f"DELETE FROM {table}")
        from .command_resolver import CorrectionFeedbackStore
        CorrectionFeedbackStore(self.mem, enabled=self.enabled).forget()
        return n

    def forget_before(self, ts: float) -> int:
        n = self.mem.query("SELECT COUNT(*) FROM episodes WHERE ts<?", (ts,))[0][0]
        self.mem.execute("DELETE FROM episodes WHERE ts<?", (ts,))
        # Aggregates cannot subtract old observations reliably; rebuild them
        # from remaining episodes instead of retaining forgotten evidence.
        for table in ("predictor_state", "calibration_state", "rules"):
            if self.mem.has_table(table):
                self.mem.execute(f"DELETE FROM {table}")
        from .command_resolver import CorrectionFeedbackStore
        CorrectionFeedbackStore(self.mem, enabled=self.enabled).forget(before=ts)
        return n


def _row_to_episode(r) -> Episode:
    ctx = None
    if r[2]:
        try:
            ctx = Context.from_dict(json.loads(r[2]))
        except Exception:
            ctx = None
    return Episode(
        id=r[0], ts=r[1], context=ctx, keystroke_prefix=r[3] or "",
        predicted=r[4], predicted_conf=r[5], action=r[6] or "",
        capability=r[7], outcome=r[8], hesitation_ms=r[9], accepted_prediction=r[10],
    )


class PredictionWindow:
    """Tracks, per input session, what the user typed and what we showed them.

    Both interfaces need the same three facts at submit time — the prefix in the
    buffer, how long the user hesitated before Enter, and whether a hint was in
    front of them — and both would get them subtly differently if each worked it
    out on its own. `take()` consumes the window, so one submission cannot be
    counted twice.
    """

    def __init__(self):
        self.buffer = ""
        self.last_keystroke_ts: Optional[float] = None
        self.shown: Optional[str] = None       # the buffer a hint was shown for
        self.shown_conf: Optional[float] = None

    def note_keystroke(self, text: str) -> None:
        self.buffer = text
        self.last_keystroke_ts = time.time()

    def note_shown(self, predicted: str, confidence: Optional[float] = None) -> None:
        """A hint reached the user's eyes. Only these carry a calibration signal."""
        self.shown = predicted
        self.shown_conf = confidence

    def take(self, submitted: str) -> dict:
        """Resolve the window against what was actually submitted, and reset.

        accepted_prediction is 1 when the user submitted what we had predicted
        they would, 0 when a hint was shown and they submitted something else,
        and None when nothing was shown — because a hint the user never saw says
        nothing about whether they wanted it.
        """
        accepted = None
        if self.shown is not None:
            accepted = 1 if submitted.strip() == self.shown.strip() else 0

        hesitation = None
        if self.last_keystroke_ts is not None:
            hesitation = max(0, int((time.time() - self.last_keystroke_ts) * 1000))

        out = {
            "keystroke_prefix": self.buffer,
            "predicted": self.shown,
            "predicted_conf": self.shown_conf,
            "accepted_prediction": accepted,
            "hesitation_ms": hesitation,
        }
        self.reset()
        return out

    def reset(self) -> None:
        self.buffer = ""
        self.last_keystroke_ts = None
        self.shown = None
        self.shown_conf = None
