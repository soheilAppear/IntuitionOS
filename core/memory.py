"""Shared SQLite access for notes, conversations and reminders.

Feature stores (episodes, corrections, prediction, retrieval and the journal)
add their own schemas through the connection helpers. Public methods serialize
access to the connection because the scheduler, anticipator and request workers
can use it from different threads. Application code should not use ``conn``
directly or hold a cursor while another thread is operating on the store.
"""

import os
import sqlite3
import threading
import time


class Memory:
    """Own one SQLite connection and the base ``mem`` and ``tasks`` tables.

    Each write helper commits before returning. ``query`` materializes rows
    under the lock; ``insert`` is the helper for obtaining a new row ID. Use
    ``insert_many`` for a batch that should commit together, and ``close`` when
    the store is no longer in use (especially before deleting temporary files).
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        # Disabling SQLite's thread check is safe only while every public
        # connection operation is protected by the lock below.
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        # Reentrant so a helper that already holds the lock can call another.
        self._lock = threading.RLock()
        self._init()

    # ── Connection helpers ───────────────────────────────────────────────
    # Keep direct connection access inside these helpers.

    def execute(self, sql: str, params: tuple = ()):
        """Execute one statement and commit; use insert/query for returned data."""
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def executescript(self, script: str):
        """Apply trusted schema/migration SQL containing multiple statements."""
        with self._lock:
            self.conn.executescript(script)
            self.conn.commit()

    def query(self, sql: str, params: tuple = ()):
        """Return materialized rows before releasing the connection lock."""
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def insert(self, sql: str, params: tuple = ()) -> int:
        """Commit one INSERT and return its SQLite row ID."""
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur.lastrowid

    def insert_many(self, sql: str, rows) -> None:
        """Commit a parameterized batch together instead of once per row."""
        # One transaction for many rows. A per-row commit is fine for the handful
        # of writes a session makes, and unusable for a bulk import or a test
        # fixture that needs thousands.
        with self._lock:
            self.conn.executemany(sql, rows)
            self.conn.commit()

    def add_many(self, entries) -> None:
        """Insert an iterable of ``(role, text, tags)`` messages with one timestamp."""
        now = time.time()
        self.insert_many(
            "INSERT INTO mem (ts, role, text, tags) VALUES (?,?,?,?)",
            [(now, role, text, tags) for role, text, tags in entries],
        )

    def close(self):
        """Release the connection; callers must stop workers using it first."""
        with self._lock:
            try:
                self.conn.close()
            except Exception:
                pass

    def has_table(self, name: str) -> bool:
        """Check whether a feature's table already exists before migration."""
        return bool(
            self.query(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
            )
        )

    def columns(self, table: str) -> list:
        """List columns for an internal, trusted table name during migration."""
        return [r[1] for r in self.query(f"PRAGMA table_info({table})")]

    def _init(self):
        # Create a table for messages and notes
        self.execute(
            "CREATE TABLE IF NOT EXISTS mem (id INTEGER PRIMARY KEY, ts REAL, role TEXT, text TEXT, tags TEXT)"
        )
        # Create a table for tasks
        self.execute(
            "CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY, ts REAL, title TEXT, due_ts REAL, status TEXT, payload TEXT)"
        )

    def add(self, role: str, text: str, tags: str = ""):
        """Store an explicit note or conversation entry without redacting text."""
        self.execute(
            "INSERT INTO mem (ts, role, text, tags) VALUES (?,?,?,?)",
            (time.time(), role, text, tags),
        )

    def recent(self, limit: int = 10):
        """Return newest-first ``(id, ts, role, text, tags)`` rows."""
        return self.query(
            "SELECT id, ts, role, text, tags FROM mem ORDER BY id DESC LIMIT ?",
            (limit,),
        )

    def search(self, term: str, limit: int = 10):
        """Run the legacy substring search; Retriever provides ranked FTS search."""
        q = f"%{term}%"
        return self.query(
            "SELECT id, ts, role, text, tags FROM mem WHERE text LIKE ? ORDER BY id DESC LIMIT ?",
            (q, limit),
        )

    # Task APIs
    def create_task(self, title: str, due_ts: float, payload: str = ""):
        """Create a pending task with a UTC epoch due time and return its ID."""
        return self.insert(
            "INSERT INTO tasks (ts, title, due_ts, status, payload) VALUES (?,?,?,?,?)",
            (time.time(), title, due_ts, "pending", payload),
        )

    def list_tasks(self, status: str = "pending"):
        """Return task dictionaries for one status, ordered by due time."""
        rows = self.query(
            "SELECT id, title, due_ts, status FROM tasks WHERE status=? ORDER BY due_ts ASC",
            (status,),
        )
        return [
            {"id": tid, "title": title, "due": due_ts, "status": st}
            for tid, title, due_ts, st in rows
        ]

    def list_open(self):
        """Return pending and fired-but-uncompleted tasks in due-time order."""
        # Delivery and completion are separate: a reminder must not disappear
        # from the user's open tasks merely because its notification fired.
        rows = self.query(
            "SELECT id, title, due_ts, status FROM tasks WHERE status IN ('pending','fired')"
            " ORDER BY due_ts ASC"
        )
        return [
            {"id": tid, "title": title, "due": due_ts, "status": st}
            for tid, title, due_ts, st in rows
        ]

    def get_task(self, task_id: int):
        """Return one task dictionary, including its payload, or None."""
        rows = self.query(
            "SELECT id, title, due_ts, status, payload FROM tasks WHERE id=?",
            (task_id,),
        )
        if not rows:
            return None
        tid, title, due_ts, st, payload = rows[0]
        return {
            "id": tid,
            "title": title,
            "due": due_ts,
            "status": st,
            "payload": payload,
        }

    def complete_task(self, task_id: int):
        """Mark a task done; user-facing calls use the complete_task capability."""
        self.execute("UPDATE tasks SET status='done' WHERE id=?", (task_id,))

    def set_task_status(self, task_id: int, status: str):
        """Set status for scheduler transitions and restoring task undo state."""
        self.execute("UPDATE tasks SET status=? WHERE id=?", (status, task_id))

    def delete_task(self, task_id: int):
        """Remove a task row; user-facing deletion must pass the action gate."""
        self.execute("DELETE FROM tasks WHERE id=?", (task_id,))

    def snooze_task(self, task_id: int, delta_seconds: int):
        """Reactivate a reminder, delaying from now or its future due time.

        Update in one statement so a scheduler cannot claim a partially changed
        task. Recurrence advancement uses claim_due_task instead of this API.
        """
        return self.execute(
            "UPDATE tasks SET due_ts=MAX(due_ts, ?)+?, status='pending' WHERE id=?",
            (time.time(), delta_seconds, task_id),
        ).rowcount == 1

    def restore_task_schedule(self, task_id: int, due_ts: float, status: str):
        """Restore exact snooze undo state without calculating from the clock."""
        return self.execute(
            "UPDATE tasks SET due_ts=?, status=? WHERE id=?",
            (due_ts, status, task_id),
        ).rowcount == 1

    def claim_due_task(self, task_id: int, expected_due: float, now_ts: float,
                       next_due: float = None):
        """Atomically consume one pending occurrence across SQLite connections.

        Advance repeats or mark one-offs fired before notifying. A crash after
        this update may miss a notification, but cannot replay a payload; fired
        one-offs remain visible until the user completes them.
        """
        return self.execute(
            "UPDATE tasks SET status=?, due_ts=?"
            " WHERE id=? AND status='pending' AND due_ts=? AND due_ts<=?",
            ("pending" if next_due is not None else "fired",
             next_due if next_due is not None else expected_due,
             task_id, expected_due, now_ts),
        ).rowcount == 1

    def due_tasks(self, now_ts: float):
        """Return pending occurrences, including the due time used to claim them."""
        rows = self.query(
            "SELECT id, title, payload, due_ts FROM tasks WHERE status='pending' AND due_ts<=?"
            " ORDER BY due_ts, id",
            (now_ts,),
        )
        return [{"id": r[0], "title": r[1], "payload": r[2], "due": r[3]} for r in rows]
