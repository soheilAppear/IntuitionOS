"""The action journal: what was attempted, by whom, with what result, and how to take it back.

The gate decides. The journal remembers. Every non-free dispatch lands here
before it runs, which buys two things the project did not have:

  * an honest audit trail, so "sandboxed exec" can be restated as the thing it
    actually is — scoped exec with a record and an undo;
  * a reversal path, so a `reversible` capability is reversible in practice and
    not just in the manifest.

Undo payloads are captured *before* the action runs, because after a file has
been overwritten its prior contents are no longer available to capture.
"""

from __future__ import annotations

import json
import time
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS journal (
  id INTEGER PRIMARY KEY,
  ts REAL,
  actor TEXT,          -- user | anticipator | model | scheduler
  capability TEXT,
  args_json TEXT,
  confidence REAL,
  decision TEXT,       -- allow | confirm_granted | confirm_denied | deny
  outcome TEXT,        -- ok | error
  undo_json TEXT,      -- payload needed to reverse, NULL if not reversible
  undone_at REAL       -- NULL until reversed
)
"""

# A write that finished and can still be taken back.
_UNDOABLE = "decision IN ('allow','confirm_granted') AND outcome='ok' AND undo_json IS NOT NULL AND undone_at IS NULL"


class Journal:
    """Append-only record of dispatched actions, with replayable undo."""

    def __init__(self, memory):
        self.mem = memory
        self.mem.executescript(SCHEMA)

    def record(
        self,
        *,
        actor: str,
        capability: str,
        args: dict,
        confidence: float,
        decision: str,
        outcome: Optional[str] = None,
        undo: Optional[dict] = None,
    ) -> int:
        """Write one entry and return its id."""
        return self.mem.insert(
            "INSERT INTO journal (ts, actor, capability, args_json, confidence, decision, outcome, undo_json, undone_at)"
            " VALUES (?,?,?,?,?,?,?,?,NULL)",
            (
                time.time(),
                actor,
                capability,
                _dumps(args),
                float(confidence),
                decision,
                outcome,
                _dumps(undo) if undo is not None else None,
            ),
        )

    def finish(self, entry_id: int, outcome: str, undo: Optional[dict] = None) -> None:
        """Fill in the result once the action has actually run."""
        if undo is None:
            self.mem.execute("UPDATE journal SET outcome=? WHERE id=?", (outcome, entry_id))
        else:
            self.mem.execute(
                "UPDATE journal SET outcome=?, undo_json=? WHERE id=?",
                (outcome, _dumps(undo), entry_id),
            )

    def recent(self, limit: int = 20) -> list[dict]:
        rows = self.mem.query(
            "SELECT id, ts, actor, capability, args_json, confidence, decision, outcome, undo_json, undone_at"
            " FROM journal ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [_row_to_dict(r) for r in rows]

    def last_undoable(self) -> Optional[dict]:
        rows = self.mem.query(
            "SELECT id, ts, actor, capability, args_json, confidence, decision, outcome, undo_json, undone_at"
            f" FROM journal WHERE {_UNDOABLE} ORDER BY id DESC LIMIT 1"
        )
        return _row_to_dict(rows[0]) if rows else None

    def get(self, entry_id: int) -> Optional[dict]:
        rows = self.mem.query(
            "SELECT id, ts, actor, capability, args_json, confidence, decision, outcome, undo_json, undone_at"
            " FROM journal WHERE id=?",
            (entry_id,),
        )
        return _row_to_dict(rows[0]) if rows else None

    def mark_undone(self, entry_id: int) -> None:
        self.mem.execute("UPDATE journal SET undone_at=? WHERE id=?", (time.time(), entry_id))

    def touched_files(self, since_ts: float, limit: int = 20) -> list[str]:
        """Paths *written* recently, for Phase 3's context sensor.

        Reads are free capabilities and therefore deliberately absent from the
        journal — recording every read would bury the audit trail in noise. The
        context sensor reads this rather than walking the filesystem, which is
        what keeps a snapshot inside its 5 ms budget.
        """
        rows = self.mem.query(
            "SELECT args_json FROM journal WHERE ts>=? AND capability='write_file'"
            " ORDER BY id DESC LIMIT ?",
            (since_ts, limit),
        )
        out: list[str] = []
        for (args_json,) in rows:
            path = (_loads(args_json) or {}).get("path")
            if path and path not in out:
                out.append(path)
        return out

    def undo_entry(self, entry_id: int, registry) -> dict:
        """Reverse one entry using its capability's declared undo."""
        entry = self.get(entry_id)
        if not entry:
            return {"error": f"no journal entry {entry_id}"}
        if entry["undone_at"] is not None:
            return {"error": f"journal entry {entry_id} was already undone"}
        if entry["undo"] is None:
            return {"error": f"journal entry {entry_id} carries no undo payload"}
        cap = registry.get(entry["capability"])
        if not cap or not cap.undo:
            return {"error": f"capability {entry['capability']} declares no undo"}
        try:
            result = cap.undo(entry["undo"])
        except Exception as e:  # a failed undo must not look like a successful one
            return {"error": f"undo failed: {e}"}
        self.mark_undone(entry_id)
        return {"ok": True, "id": entry_id, "capability": entry["capability"], "result": result}

    def undo_last(self, registry) -> dict:
        entry = self.last_undoable()
        if not entry:
            return {"error": "nothing to undo"}
        return self.undo_entry(entry["id"], registry)


def _dumps(value) -> str:
    try:
        return json.dumps(value, default=str)
    except Exception:
        return json.dumps({"unserializable": str(value)})


def _loads(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _row_to_dict(r) -> dict:
    return {
        "id": r[0],
        "ts": r[1],
        "actor": r[2],
        "capability": r[3],
        "args": _loads(r[4]) or {},
        "confidence": r[5],
        "decision": r[6],
        "outcome": r[7],
        "undo": _loads(r[8]),
        "undone_at": r[9],
    }
