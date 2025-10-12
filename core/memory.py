# SQLite based memory store with add, recent, search

import os, sqlite3, time

class Memory:
    def __init__(self, db_path:str):
        # Save db path
        self.db_path = db_path
        # Ensure folder exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        # Create connection
        self.conn = sqlite3.connect(db_path)
        # Create schema
        self._init()

    def _init(self):
        # Create a table for messages and notes
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS mem (id INTEGER PRIMARY KEY, ts REAL, role TEXT, text TEXT, tags TEXT)"
        )
        # Create a table for tasks
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY, ts REAL, title TEXT, due_ts REAL, status TEXT, payload TEXT)"
        )
        # Commit the schema
        self.conn.commit()

    def add(self, role:str, text:str, tags:str=""):
        # Insert a new memory row
        self.conn.execute("INSERT INTO mem (ts, role, text, tags) VALUES (?,?,?,?)",
                          (time.time(), role, text, tags))
        # Persist the insert
        self.conn.commit()

    def recent(self, limit:int=10):
        # Query the most recent memory rows
        cur = self.conn.execute("SELECT id, ts, role, text, tags FROM mem ORDER BY id DESC LIMIT ?", (limit,))
        # Return as list
        return cur.fetchall()

    def search(self, term:str, limit:int=10):
        # Simple LIKE search
        q = f"%{term}%"
        cur = self.conn.execute(
            "SELECT id, ts, role, text, tags FROM mem WHERE text LIKE ? ORDER BY id DESC LIMIT ?", (q, limit)
        )
        return cur.fetchall()

    # Task APIs
    def create_task(self, title:str, due_ts:float, payload:str=""):
        # Insert new task as pending
        self.conn.execute("INSERT INTO tasks (ts, title, due_ts, status, payload) VALUES (?,?,?,?,?)",
                          (time.time(), title, due_ts, "pending", payload))
        self.conn.commit()
        # Return new id
        cur = self.conn.execute("SELECT last_insert_rowid()")
        return cur.fetchone()[0]

    def list_tasks(self, status:str="pending"):
        # Fetch tasks by status
        cur = self.conn.execute("SELECT id, title, due_ts, status FROM tasks WHERE status=? ORDER BY due_ts ASC", (status,))
        # Map to dicts
        out = []
        for tid, title, due_ts, st in cur.fetchall():
            out.append({"id": tid, "title": title, "due": due_ts, "status": st})
        return out

    def complete_task(self, task_id:int):
        # Mark a task done
        self.conn.execute("UPDATE tasks SET status='done' WHERE id=?", (task_id,))
        self.conn.commit()

    def delete_task(self, task_id:int):
        # Delete a task
        self.conn.execute("DELETE FROM tasks WHERE id=?", (task_id,))
        self.conn.commit()

    def snooze_task(self, task_id:int, delta_seconds:int):
        # Move due time
        # Get current due time
        cur = self.conn.execute("SELECT due_ts FROM tasks WHERE id=?", (task_id,))
        row = cur.fetchone()
        if not row:
            return
        new_due = row[0] + delta_seconds
        self.conn.execute("UPDATE tasks SET due_ts=? WHERE id=?", (new_due, task_id))
        self.conn.commit()

    def due_tasks(self, now_ts:float):
        # Fetch pending tasks that are due
        cur = self.conn.execute("SELECT id, title, payload FROM tasks WHERE status='pending' AND due_ts<=?", (now_ts,))
        return [{"id": r[0], "title": r[1], "payload": r[2]} for r in cur.fetchall()]
