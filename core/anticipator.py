# Anticipator that observes the input buffer and prewarms results in background

import time, threading

class Anticipator:
    def __init__(self, predict_fn, prewarm_fn, enabled=True, debounce_ms=180, match_threshold=0.6):
        # Save callbacks and knobs
        self.predict_fn = predict_fn
        self.prewarm_fn = prewarm_fn
        self.enabled = enabled
        self.debounce_ms = debounce_ms
        self.match_threshold = match_threshold
        # Current buffer
        self._buffer = ""
        # Cache of prepared results keyed by text
        self._cache = {}
        # Thread state
        self._stop = False
        self._thread = None
        # Last time we saw a change
        self._last_change = 0

    def start(self):
        # Start background worker
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._run, name="anticipator", daemon=True)
        self._thread.start()

    def stop(self):
        # Stop the thread
        self._stop = True

    def update_buffer(self, text:str):
        # Update watched buffer and time
        self._buffer = text
        self._last_change = time.time()

    def try_serve(self, text:str):
        # Try to return a prepared result for the exact text
        return self._cache.get(text)

    def _run(self):
        # Simple loop with debounce
        while not self._stop:
            time.sleep(0.05)
            buf = self._buffer
            if not buf:
                continue
            # Debounce logic
            if (time.time() - self._last_change) * 1000 < self.debounce_ms:
                continue
            # Predict intent
            intent = self.predict_fn(buf)
            if not intent or intent.get("confidence", 0) < self.match_threshold:
                continue
            # Prewarm and store in cache
            try:
                result = self.prewarm_fn(intent)
                if result:
                    key, value = result
                    self._cache[key] = value
            except Exception:
                # Ignore prewarm errors
                pass
