# Polling scheduler for reminders and payload execution

import time, threading, dateparser, json

class Scheduler:
    def __init__(self, db_path:str, tz:str, tick_seconds:int, notify_cb, execute_cb):
        # Save fields
        self.db_path = db_path
        self.tz = tz
        self.tick_seconds = tick_seconds
        self.notify_cb = notify_cb
        self.execute_cb = execute_cb
        # External memory will be assigned by set_scheduler
        self.memory = None
        # Start thread
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="scheduler", daemon=True)
        self._thread.start()

    def set_memory(self, mem):
        # Bind memory object
        self.memory = mem

    def parse_when(self, when:str):
        # Use dateparser to parse friendly times like "in 10m"
        dt = dateparser.parse(when, settings={"TIMEZONE": self.tz, "RETURN_AS_TIMEZONE_AWARE": False})
        # Return unix timestamp
        return time.mktime(dt.timetuple()) if dt else None

    def create(self, title:str, when:str, payload:dict=None, repeat:str=""):
        # Parse when
        due_ts = self.parse_when(when)
        if due_ts is None:
            return {"error": f"could not parse when: {when}"}
        # Serialize payload
        payload_json = json.dumps(payload or {})
        # Store as task using bound memory
        task_id = self.memory.create_task(title, due_ts, payload_json)
        # Return result
        return {"ok": True, "id": task_id, "due_ts": due_ts}

    def _run(self):
        # Poll loop
        while not self._stop:
            try:
                if self.memory:
                    now = time.time()
                    # Fetch due tasks
                    due = self.memory.due_tasks(now)
                    for t in due:
                        # Notify user
                        self.notify_cb(t["id"], t["title"])
                        # Execute payload if any
                        try:
                            payload = json.loads(t["payload"] or "{}")
                        except Exception:
                            payload = {}
                        if payload:
                            self.execute_cb(payload)
                        # Mark done
                        self.memory.complete_task(t["id"])
            except Exception:
                # Ignore scheduler exceptions
                pass
            time.sleep(self.tick_seconds)
