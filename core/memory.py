# SQLite based memory store with add, recent, search

import os, sqlite3, threading, time


class Memory:
    def __init__(self, db_path: str):
        # Save db path
        self.db_path = db_path
        # Ensure folder exists
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        # Create connection. check_same_thread=False lets the scheduler thread,
        # the anticipator thread and the FastAPI executor share one connection —
        # which is only safe because every statement below goes through _lock
        # (Appendix A #11: previously they did not, and the races were latent).
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        # Reentrant so a helper that already holds the lock can call another.
        self._lock = threading.RLock()
        # Create schema
        self._init()

    # ── Connection helpers ───────────────────────────────────────────────
    # Everything that touches self.conn goes through one of these four.

    def execute(self, sql: str, params: tuple = ()):
        # Run a write statement and commit
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def executescript(self, script: str):
        # Run DDL (may contain several statements)
        with self._lock:
            self.conn.executescript(script)
            self.conn.commit()

    def query(self, sql: str, params: tuple = ()):
        # Run a read statement and materialise the rows before releasing
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def insert(self, sql: str, params: tuple = ()) -> int:
        # Run an INSERT and return the new row id
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur.lastrowid

    def insert_many(self, sql: str, rows) -> None:
        # One transaction for many rows. A per-row commit is fine for the handful
        # of writes a session makes, and unusable for a bulk import or a test
        # fixture that needs thousands.
        with self._lock:
            self.conn.executemany(sql, rows)
            self.conn.commit()

    def add_many(self, entries) -> None:
        # entries: iterable of (role, text, tags)
        now = time.time()
        self.insert_many(
            "INSERT INTO mem (ts, role, text, tags) VALUES (?,?,?,?)",
            [(now, role, text, tags) for role, text, tags in entries],
        )

    def close(self):
        # Release the file handle. Mostly this process lives as long as the
        # database does, but a temporary database — a test, an eval run — has to
        # be closable or Windows will refuse to delete the directory holding it.
        with self._lock:
            try:
                self.conn.close()
            except Exception:
                pass

    def has_table(self, name: str) -> bool:
        # Used by migrations to tell a fresh database from an existing one
        return bool(self.query("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)))

    def columns(self, table: str) -> list:
        # Column names of an existing table, for additive migrations
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
        # Insert a new memory row
        self.execute(
            "INSERT INTO mem (ts, role, text, tags) VALUES (?,?,?,?)",
            (time.time(), role, text, tags),
        )

    def recent(self, limit: int = 10):
        # Query the most recent memory rows
        return self.query("SELECT id, ts, role, text, tags FROM mem ORDER BY id DESC LIMIT ?", (limit,))

    def search(self, term: str, limit: int = 10):
        # Simple LIKE search
        q = f"%{term}%"
        return self.query(
            "SELECT id, ts, role, text, tags FROM mem WHERE text LIKE ? ORDER BY id DESC LIMIT ?",
            (q, limit),
        )

    # Task APIs
    def create_task(self, title: str, due_ts: float, payload: str = ""):
        # Insert new task as pending and return its id
        return self.insert(
            "INSERT INTO tasks (ts, title, due_ts, status, payload) VALUES (?,?,?,?,?)",
            (time.time(), title, due_ts, "pending", payload),
        )

    def list_tasks(self, status: str = "pending"):
        # Fetch tasks by status
        rows = self.query(
            "SELECT id, title, due_ts, status FROM tasks WHERE status=? ORDER BY due_ts ASC", (status,)
        )
        # Map to dicts
        return [{"id": tid, "title": title, "due": due_ts, "status": st} for tid, title, due_ts, st in rows]

    def list_open(self):
        # Pending plus already-fired-but-not-completed. A reminder that fired
        # while the app was closed is still the user's to deal with, so it must
        # not vanish from the list the moment it rang.
        rows = self.query(
            "SELECT id, title, due_ts, status FROM tasks WHERE status IN ('pending','fired')"
            " ORDER BY due_ts ASC"
        )
        return [{"id": tid, "title": title, "due": due_ts, "status": st} for tid, title, due_ts, st in rows]

    def get_task(self, task_id: int):
        # Fetch one task or None
        rows = self.query("SELECT id, title, due_ts, status, payload FROM tasks WHERE id=?", (task_id,))
        if not rows:
            return None
        tid, title, due_ts, st, payload = rows[0]
        return {"id": tid, "title": title, "due": due_ts, "status": st, "payload": payload}

    def complete_task(self, task_id: int):
        # Mark a task done
        self.execute("UPDATE tasks SET status='done' WHERE id=?", (task_id,))

    def set_task_status(self, task_id: int, status: str):
        # Move a task to an explicit status (used by undo and by the scheduler)
        self.execute("UPDATE tasks SET status=? WHERE id=?", (status, task_id))

    def delete_task(self, task_id: int):
        # Delete a task
        self.execute("DELETE FROM tasks WHERE id=?", (task_id,))

    def snooze_task(self, task_id: int, delta_seconds: int):
        # Move due time
        rows = self.query("SELECT due_ts FROM tasks WHERE id=?", (task_id,))
        if not rows:
            return
        self.execute("UPDATE tasks SET due_ts=? WHERE id=?", (rows[0][0] + delta_seconds, task_id))

    def due_tasks(self, now_ts: float):
        # Fetch pending tasks that are due
        rows = self.query(
            "SELECT id, title, payload FROM tasks WHERE status='pending' AND due_ts<=?", (now_ts,)
        )
        return [{"id": r[0], "title": r[1], "payload": r[2]} for r in rows]
